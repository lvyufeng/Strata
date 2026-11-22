# Copyright (c) 2026 Niko1221 and the Strata contributors
# SPDX-License-Identifier: MIT
"""Opt-in, bounded local GGUF-header inventory. No payload, network or engine.

The wire parser is the existing gguf_reader.GGUFFile._parse/_value. This adapter
adds inspection ceilings, strict text, local-file checks and receipt semantics;
it is not a second parser, an artifact authenticator or an OS/resource sandbox.
"""
from __future__ import annotations

import fnmatch
import glob
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import unicodedata

import gguf_reader as G

MAX_INTEGER = (1 << 63) - 1
MAX_RECEIPT_BYTES = 64 << 20
SCHEMA_VERSION = 2
# v2 raises only the metadata-array-item ceilings, after measuring 744,317
# declared items in one real shard header, and confirms the six additional GGML
# types present in that artifact's declared type census. Every other ceiling is
# unchanged from the reviewed v1 adapter at base revision 3142b66b.
LIMITS = {
    "max_shards": 64,
    "max_header_bytes_per_file": 16777216,
    "max_header_bytes_total": 33554432,
    "max_string_bytes": 1048576,
    "max_string_bytes_total": 16777216,
    "max_metadata_entries_per_file": 4096,
    "max_metadata_entries_total": 8192,
    "max_tensors_total": 100000,
    "max_array_items_per_file": 2000000,
    "max_array_items_total": 4000000,
    "max_rank": 8,
    "max_glob_entries": 4096,
    "max_path_arguments": 64,
}
PRODUCER = {
    "tool": "Strata/terminal/tools/quant_formats.py",
    "mode": "bounded-local-v2",
    "base_revision": "3142b66b96e9c0e775e30f620b11b191a1427c2a",
    "parser_sha256": "c84a0f88c6c75a588cb7de74abd23b2e6387c92c58e088012aedef5bd3c6c431",
}
CLAIMS = {
    "header_only": True,
    "input_provenance_verified": False,
    "artifact_integrity_verified": False,
    "model_completeness_verified": False,
    "backend_compatibility_verified": False,
    "device_ready": False,
    "inference_ready": False,
}
# An inventory allowlist, not a replacement ggml type/geometry table. The v2 set
# adds the six ids that the target artifact declares itself: Q6_K 14, IQ2_XXS 16,
# IQ4_NL 20, IQ2_S 22, IQ4_XS 23 and IQ1_M 29. In particular, IQ1_S 19, MXFP4 39,
# NVFP4 40, the ambiguous id 41 and other layouts still stay unknown.
CONFIRMED_TYPE_IDS = frozenset((0, 1, 8, 14, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 42))
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


class InventoryError(ValueError):
    """A stable, sanitized diagnostic; neither code nor message includes input."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _reject(code: str, message: str) -> None:
    raise InventoryError(code, message)


def _safe_text(value: str) -> bool:
    return bool(value) and not any(unicodedata.category(c) in ("Cc", "Cf", "Cs") for c in value)


def _add(a: int, b: int) -> int:
    if a < 0 or b < 0 or a > MAX_INTEGER - b:
        _reject("integer-overflow", "Inventory byte arithmetic exceeds its integer ceiling.")
    return a + b


def _multiply(a: int, b: int) -> int:
    if a < 0 or b < 0 or (b and a > MAX_INTEGER // b):
        _reject("integer-overflow", "Inventory shape or storage exceeds its integer ceiling.")
    return a * b


class _Budget:
    def __init__(self):
        self.header_bytes = 0
        self.string_bytes = 0
        self.metadata_entries = 0
        self.tensors = 0
        self.array_items = 0


class _UniqueMetadata(dict):
    def __setitem__(self, key, value):
        if key in self:
            _reject("duplicate-metadata-key", "Metadata keys must be unique within a shard.")
        super().__setitem__(key, value)


class BoundedHeader(G.GGUFFile):
    """Thin guards around the reviewed parser, on an already-open raw handle.

    The inherited _parse/_value still interpret the GGUF wire layout. Overrides
    cap counts before their loops/unpack allocations and preserve exact consumed
    bytes for a header-only SHA256. Metadata is not included in the receipt.
    """

    def __init__(self, fh, size: int, budget: _Budget):
        self.path = Path("header")  # never a user path in inherited diagnostics
        self.metadata = _UniqueMetadata()
        self.tensors = []
        self.version, self.alignment = 0, 32
        self._size, self._pos = size, 0
        self._budget = budget
        self._header_hash = hashlib.sha256()
        self._array_items = 0
        self._value_depth = 0
        self._text_kind = "metadata-key"
        self._current_key = None
        try:
            self._parse(fh)
        except InventoryError:
            raise
        except (ValueError, OverflowError, UnicodeError):
            _reject("invalid-header", "The GGUF v3 logical header is malformed or unsupported.")
        self.data_start = self._data_start
        self.header_sha256 = self._header_hash.hexdigest()

    def _read(self, fh, n: int) -> bytes:
        if (n < 0 or n > LIMITS["max_header_bytes_per_file"] - self._pos
                or n > LIMITS["max_header_bytes_total"] - self._budget.header_bytes):
            _reject("header-byte-limit", "Logical header reads exceed an inspection ceiling.")
        raw = super()._read(fh, n)
        self._budget.header_bytes += len(raw)
        self._header_hash.update(raw)
        return raw

    def _unpack(self, fh, fmt: str) -> tuple:
        position = self._pos
        values = super()._unpack(fh, fmt)
        if position == 4 and fmt == "<IQQ":
            # Guard both top-level declarations before the first metadata loop,
            # even if the tensor count would otherwise be checked much later.
            _, tensors, metadata = values
            if tensors > LIMITS["max_tensors_total"] - self._budget.tensors:
                _reject("tensor-count-limit", "Declared tensors exceed the inspection ceiling.")
            if (metadata > LIMITS["max_metadata_entries_per_file"]
                    or metadata > LIMITS["max_metadata_entries_total"] - self._budget.metadata_entries):
                _reject("metadata-count-limit", "Declared metadata exceeds an inspection ceiling.")
        return values

    def _count(self, fh, count: int, min_bytes: int, what: str) -> int:
        if what == "metadata entries":
            if (count > LIMITS["max_metadata_entries_per_file"]
                    or count > LIMITS["max_metadata_entries_total"] - self._budget.metadata_entries):
                _reject("metadata-count-limit", "Declared metadata exceeds an inspection ceiling.")
            self._text_kind = "metadata-key"
        elif what == "tensors":
            if count > LIMITS["max_tensors_total"] - self._budget.tensors:
                _reject("tensor-count-limit", "Declared tensors exceed the inspection ceiling.")
            self._text_kind = "tensor-name"
        elif what == "dimensions":
            if not 1 <= count <= LIMITS["max_rank"]:
                _reject("tensor-rank", "Tensor rank must be between one and the inspection ceiling.")
        elif what in ("array strings", "array elements"):
            if (count > LIMITS["max_array_items_per_file"] - self._array_items
                    or count > LIMITS["max_array_items_total"] - self._budget.array_items):
                _reject("array-count-limit", "Metadata arrays exceed an inspection ceiling.")
        # Reject an impossible bounded minimum before base-parser loops or format
        # strings. File-size bounds alone would permit very large allocations.
        needed = count * min_bytes
        if (needed > LIMITS["max_header_bytes_per_file"] - self._pos
                or needed > LIMITS["max_header_bytes_total"] - self._budget.header_bytes):
            _reject("header-byte-limit", "Declared entries cannot fit in the bounded logical header.")
        result = super()._count(fh, count, min_bytes, what)
        if what == "metadata entries":
            self._budget.metadata_entries += count
        elif what == "tensors":
            self._budget.tensors += count
        elif what in ("array strings", "array elements"):
            self._array_items += count
            self._budget.array_items += count
        return result

    def _str(self, fh) -> str:
        (size,) = self._unpack(fh, "<Q")
        if (size > LIMITS["max_string_bytes"]
                or size > LIMITS["max_string_bytes_total"] - self._budget.string_bytes):
            _reject("string-byte-limit", "A string exceeds an inspection ceiling.")
        raw = self._read(fh, size)
        self._budget.string_bytes += size
        try:
            value = raw.decode("utf-8", "strict")
        except UnicodeError:
            _reject("invalid-utf8", "Logical header strings must be valid UTF-8.")
        if not self._value_depth:
            if not _safe_text(value):
                _reject("unsafe-name", "Tensor names and metadata keys must be nonempty safe text.")
            if self._text_kind == "metadata-key":
                if value in self.metadata:
                    _reject("duplicate-metadata-key", "Metadata keys must be unique within a shard.")
                self._current_key = value
        return value

    def _value(self, fh):
        self._value_depth += 1
        try:
            value = super()._value(fh)
        finally:
            self._value_depth -= 1
        values = value if isinstance(value, list) else (value,)
        if any(isinstance(v, float) and not math.isfinite(v) for v in values):
            _reject("nonfinite-metadata", "Floating-point metadata must be finite.")
        if self._current_key == "general.alignment":
            if (type(value) is not int or not 8 <= value <= (1 << 20)
                    or value & (value - 1)):
                _reject("alignment", "Inventory alignment must be a power of two from 8 to 1048576.")
        if self._current_key in SPLIT_KEYS and type(value) is not int:
            _reject("split-scalar", "Split declarations must be scalar integers, not booleans or arrays.")
        return value


def _expand_paths(arguments) -> list[Path]:
    if not isinstance(arguments, (list, tuple)) or not arguments:
        _reject("no-inputs", "Provide explicit local files or nonrecursive basename globs.")
    if len(arguments) > LIMITS["max_path_arguments"]:
        _reject("path-argument-limit", "Path arguments exceed the inspection ceiling.")
    paths, seen_paths, seen_names = [], set(), set()
    scanned = 0

    def collect(path: Path):
        identity = os.path.abspath(path)
        if identity in seen_paths:
            _reject("duplicate-input", "Each local input may be provided only once.")
        if path.name in seen_names:
            _reject("duplicate-basename", "Input basenames must be unique for stable shard labels.")
        if len(paths) >= LIMITS["max_shards"]:
            _reject("shard-count-limit", "Matched shards exceed the inspection ceiling.")
        seen_paths.add(identity)
        seen_names.add(path.name)
        paths.append(path)

    for argument in arguments:
        if type(argument) is not str:
            _reject("invalid-path", "Local input arguments must be path strings.")
        try:
            encoded = argument.encode("utf-8", "strict")
        except UnicodeError:
            _reject("invalid-path", "Local input arguments must be UTF-8 path strings.")
        if not encoded or len(encoded) > 4096 or "\0" in argument:
            _reject("invalid-path", "Local input arguments must be nonempty and at most 4096 bytes.")
        path = Path(argument)
        if "**" in argument or glob.has_magic(str(path.parent)):
            _reject("unsafe-glob", "Only nonrecursive basename globs are accepted.")
        if not glob.has_magic(path.name):
            collect(path)
            continue
        matches = 0
        try:
            with os.scandir(path.parent) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned > LIMITS["max_glob_entries"]:
                        _reject("glob-entry-limit", "Directory scans exceed the aggregate inspection ceiling.")
                    if fnmatch.fnmatchcase(entry.name, path.name):
                        matches += 1
                        collect(path.parent / entry.name)
        except OSError:
            _reject("input-io", "A local input directory could not be inspected.")
        if not matches:
            _reject("no-matches", "A local basename glob matched no files.")
    return sorted(paths, key=lambda p: p.name)


def _snapshot(info) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _split_values(metadata: dict) -> dict:
    result = dict(zip(("index", "count", "total_tensors"), (metadata.get(k) for k in SPLIT_KEYS)))
    index, count, total = result.values()
    if index is not None and (type(index) is not int or not 0 <= index < LIMITS["max_shards"]):
        _reject("split-index", "Split indices must fit the inspection ceiling.")
    if count is not None and (type(count) is not int or not 1 <= count <= LIMITS["max_shards"]):
        _reject("split-count", "Split shard counts must be positive and bounded.")
    if total is not None and (type(total) is not int or not 0 <= total <= LIMITS["max_tensors_total"]):
        _reject("split-total", "Split tensor totals must be nonnegative and bounded.")
    if index is not None and count is not None and index >= count:
        _reject("split-index", "A split index must be less than its declared shard count.")
    return result


def _tensor_receipts(header: BoundedHeader, label: str, size: int) -> list[dict]:
    if header.tensors and header.data_start > size:
        _reject("data-region", "A nonempty tensor directory needs a data region within the file.")
    if header.data_start > MAX_INTEGER:
        _reject("integer-overflow", "The aligned data start exceeds the inventory integer ceiling.")
    tensors, known, unknown = [], [], []
    names = set()
    for tensor in header.tensors:
        if tensor.name in names:
            _reject("duplicate-tensor-name", "Tensor names must be globally unique.")
        names.add(tensor.name)
        elements = 1
        for dimension in tensor.shape:
            if not 1 <= dimension <= MAX_INTEGER:
                _reject("tensor-shape", "Tensor dimensions must be positive bounded integers.")
            elements = _multiply(elements, dimension)
        if tensor.offset % header.alignment:
            _reject("tensor-alignment", "Tensor relative offsets must be alignment multiples.")
        absolute = _add(header.data_start, tensor.offset)
        if absolute >= size:
            _reject("tensor-offset", "Tensor offsets must point inside the local file data region.")
        storage = {"layout": "unknown", "block_elements": None, "block_bytes": None,
                   "row_stride_bytes": None, "data_bytes": None}
        if tensor.type_id in CONFIRMED_TYPE_IDS:
            block_elements, block_bytes = G.BLOCK_GEOMETRY[tensor.type_name]
            if tensor.shape[0] % block_elements:
                _reject("row-divisibility", "The fastest tensor dimension must contain whole confirmed blocks.")
            row_stride = _multiply(tensor.shape[0] // block_elements, block_bytes)
            data_bytes = _multiply(elements // tensor.shape[0], row_stride)
            end = _add(absolute, data_bytes)
            if end > size:
                _reject("tensor-extent", "A confirmed tensor extent exceeds the local file size.")
            storage = {"layout": "gguf-row-blocks", "block_elements": block_elements,
                       "block_bytes": block_bytes, "row_stride_bytes": row_stride, "data_bytes": data_bytes}
            known.append((absolute, end))
        else:
            unknown.append(absolute)
        tensors.append({
            "name": tensor.name, "shard": label, "shape": tensor.shape,
            "ggml_type_id": tensor.type_id, "ggml_type_name": tensor.type_name,
            "relative_offset_bytes": tensor.offset, "absolute_offset_bytes": absolute,
            "storage": storage, "tensor_sha256": None,
        })
    known.sort()
    for previous, current in zip(known, known[1:]):
        if current[0] < previous[1]:
            _reject("tensor-overlap", "Confirmed tensor intervals must not overlap.")
    unknown.sort()
    position = 0
    for start, end in known:
        while position < len(unknown) and unknown[position] <= start:
            position += 1
        if position < len(unknown) and unknown[position] < end:
            _reject("tensor-overlap", "An unknown tensor offset lies strictly inside a confirmed interval.")
    return tensors


def _read_shard(path: Path, label: str, budget: _Budget, identities: set) -> tuple[dict, list[dict]]:
    flags = (getattr(os, "O_NOFOLLOW", None), getattr(os, "O_NONBLOCK", None))
    if os.name != "posix" or any(flag is None for flag in flags):
        _reject("file-safety-unavailable", "This mode requires POSIX no-follow and nonblocking file opens.")
    try:
        before_path = os.lstat(path)
        if not stat.S_ISREG(before_path.st_mode):
            _reject("nonregular-input", "Inputs must be ordinary nonsymlink local files.")
        if not 0 <= before_path.st_size <= MAX_INTEGER:
            _reject("file-size-limit", "The local file exceeds the inventory integer ceiling.")
        descriptor = os.open(path, os.O_RDONLY | flags[0] | flags[1])
        try:
            handle = os.fdopen(descriptor, "rb", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or _snapshot(before_path) != _snapshot(before):
                _reject("input-changed", "A local input changed during read-only inspection.")
            identity = (before.st_dev, before.st_ino)
            if identity in identities:
                _reject("duplicate-input", "Local input aliases must not refer to the same file.")
            identities.add(identity)
            header = BoundedHeader(handle, before.st_size, budget)
            tensors = _tensor_receipts(header, label, before.st_size)
            split = _split_values(header.metadata)
            after = os.fstat(handle.fileno())
            if _snapshot(before) != _snapshot(after):
                _reject("input-changed", "A local input changed during read-only inspection.")
            shard = {
                "label": label, "file_size_bytes": before.st_size, "gguf_version": header.version,
                "alignment_bytes": header.alignment, "header_end_bytes": header._pos,
                "data_start_bytes": header.data_start, "header_sha256": header.header_sha256,
                "artifact_sha256": None, "payload_read": False, "tensor_count": len(tensors), "split": split,
            }
            return shard, tensors
    except OSError:
        _reject("input-io", "A local input could not be inspected as a read-only ordinary file.")


def _split_receipt(shards: list[dict], tensor_count: int) -> dict:
    declarations = [s["split"] for s in shards]
    counts = {d["count"] for d in declarations if d["count"] is not None}
    totals = {d["total_tensors"] for d in declarations if d["total_tensors"] is not None}
    indices = [d["index"] for d in declarations if d["index"] is not None]
    if len(counts) > 1 or len(totals) > 1:
        _reject("split-conflict", "Available split counts and totals must agree across shards.")
    if len(set(indices)) != len(indices):
        _reject("split-duplicate-index", "Available split indices must be unique.")
    count = next(iter(counts), None)
    total = next(iter(totals), None)
    if count is not None and (len(shards) > count or any(i >= count for i in indices)):
        _reject("split-count", "Provided shards and indices exceed the available declared shard count.")
    if total is not None and tensor_count > total:
        _reject("split-total", "Provided tensors exceed the available declared total.")
    indices.sort()
    missing = sorted(set(range(count)) - set(indices)) if count is not None else []
    receipt = {"status": "unknown", "declared_shards": count, "declared_tensors": total,
               "present_indices": indices, "missing_indices": missing, "diagnostics": []}
    if not any(value is not None for d in declarations for value in d.values()):
        receipt["diagnostics"] = ["split-not-declared"]
        return receipt
    partial = any(any(value is None for value in d.values()) for d in declarations)
    if partial or missing or (count is not None and len(shards) < count):
        receipt["status"] = "incomplete"
        if partial:
            receipt["diagnostics"].append("split-declarations-partial")
        if missing:
            receipt["diagnostics"].append("split-shards-missing")
        return receipt
    if tensor_count != total:
        _reject("split-total", "A complete declared shard set must match its exact tensor total.")
    receipt["status"] = "complete"
    return receipt


def inventory(arguments, input_kind: str = "local-unverified") -> dict:
    """Inspect only explicitly supplied local shards; no filename split inference."""
    if input_kind not in ("synthetic", "local-unverified"):
        _reject("input-kind", "Input kind must be synthetic or local-unverified.")
    paths = _expand_paths(arguments)
    shards, tensors, identities, names = [], [], set(), set()
    budget = _Budget()
    known_bytes, unknown_count = 0, 0
    for number, path in enumerate(paths):
        label = f"shard-{number:04d}"
        shard, shard_tensors = _read_shard(path, label, budget, identities)
        shards.append(shard)
        for tensor in shard_tensors:
            if tensor["name"] in names:
                _reject("duplicate-tensor-name", "Tensor names must be globally unique.")
            names.add(tensor["name"])
            data_bytes = tensor["storage"]["data_bytes"]
            if data_bytes is None:
                unknown_count += 1
            else:
                known_bytes = _add(known_bytes, data_bytes)
            tensors.append(tensor)
    tensors.sort(key=lambda t: (t["shard"], t["name"]))
    return {
        "schema_version": SCHEMA_VERSION, "kind": "gguf-header-inventory", "input_kind": input_kind,
        "producer": dict(PRODUCER), "limits": dict(LIMITS), "shards": shards, "tensors": tensors,
        "split": _split_receipt(shards, len(tensors)),
        "totals": {"tensor_count": len(tensors), "known_storage_bytes": known_bytes,
                   "unknown_storage_tensors": unknown_count},
        "claims": dict(CLAIMS),
    }


def dumps(document: dict) -> str:
    """Deterministic JSON to stdout only; bound assembly before emitting anything."""
    encoder = json.JSONEncoder(sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    output = io.StringIO()
    size = 1  # final newline; JSON strings below are ASCII, so bytes == chars
    for piece in encoder.iterencode(document):
        size += len(piece)
        if size > MAX_RECEIPT_BYTES:
            _reject("receipt-byte-limit", "The generated receipt exceeds the JSON inspection ceiling.")
        output.write(piece)
    output.write("\n")
    return output.getvalue()


def diagnostic(error: InventoryError) -> str:
    return json.dumps({"code": error.code, "message": error.message}, sort_keys=True) + "\n"
