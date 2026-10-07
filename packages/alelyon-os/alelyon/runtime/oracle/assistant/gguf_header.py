"""Read what a GGUF file says about itself, without loading it (ADR-0041).

Ollama answered "what is this model" through `/api/show`. A model here is a GGUF
file, so the answer comes from the file's own header: its metadata key-value
pairs and its tensor table. Nothing past the tensor table is read, so describing
a 40 GB model costs a few megabytes of reads, most of them the tokenizer's
string arrays.

No third-party package: the research scripts use `gguf`'s `GGUFReader`, but
the product does not take on a dependency to read a documented header.
Reference: the GGUF specification in ggml's `docs/gguf.md`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

MAGIC = b"GGUF"

#: Metadata value types (`gguf_metadata_value_type`).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, \
    _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)
_SCALAR = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h", _UINT32: "<I",
    _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?", _UINT64: "<Q", _INT64: "<q",
    _FLOAT64: "<d",
}

#: Arrays longer than this are counted, not kept: the tokenizer's vocabulary is
#: an array of 150,000 strings that nobody describing a model needs to hold.
_KEEP_ARRAY_MAX = 64

#: Bounds that make a corrupt or hostile header fail instead of exhausting memory.
_MAX_STRING = 1 << 20
_MAX_COUNT = 1 << 24
_MAX_DIMS = 8

#: `general.file_type` (llama.cpp's `llama_ftype`), as llama.cpp names them.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE",
}


#: ggml tensor types (`ggml_type`), as ggml names them.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}


class GGUFError(ValueError):
    """The file is not a GGUF file this reader understands."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, f"type {self.ggml_type}")

    @property
    def elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    metadata: Dict[str, Any]
    tensors: Tuple[TensorInfo, ...] = field(repr=False)
    #: Array-valued keys that were longer than `_KEEP_ARRAY_MAX`: key -> length.
    array_lengths: Dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name", ""))

    @property
    def quantization(self) -> str:
        ftype = self.metadata.get("general.file_type")
        if ftype is None:
            return ""
        return FILE_TYPES.get(int(ftype), f"file_type {int(ftype)}")

    @property
    def context_length(self) -> Optional[int]:
        value = self.metadata.get(f"{self.architecture}.context_length")
        return int(value) if value is not None else None

    @property
    def parameters(self) -> int:
        """Weights, counted from the tensor table (no tensor data is read)."""
        return sum(t.elements for t in self.tensors)

    def show_payload(self, model: str) -> Dict[str, Any]:
        """The shape Ollama's `/api/show` had, which morphometry, the footprint
        desk and the Foundry panel read. Only what the file DECLARES goes into
        `model_info`: a parameter count is not added when the file has none, so
        morphometry's declared-versus-measured comparison stays honest."""
        info = {k: v for k, v in self.metadata.items()
                if isinstance(v, (str, int, float, bool))}
        return {
            "model": model,
            "details": {"family": self.architecture,
                        "quantization_level": self.quantization},
            "model_info": info,
            "tensors": [{"name": t.name, "shape": list(t.shape), "type": t.type_name}
                        for t in self.tensors],
        }

    def describe(self) -> Dict[str, Any]:
        """The fields a model picker or a self-description shows."""
        return {
            "architecture": self.architecture,
            "name": self.name,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "parameters": self.parameters,
            "tensors": len(self.tensors),
            "block_count": self.metadata.get(f"{self.architecture}.block_count"),
            "embedding_length": self.metadata.get(
                f"{self.architecture}.embedding_length"),
            "expert_count": self.metadata.get(f"{self.architecture}.expert_count"),
            "gguf_version": self.version,
        }


def _read(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise GGUFError("the header ends early")
    return data


def _unpack(handle: BinaryIO, fmt: str):
    return struct.unpack(fmt, _read(handle, struct.calcsize(fmt)))[0]


def _string(handle: BinaryIO) -> str:
    length = _unpack(handle, "<Q")
    if length > _MAX_STRING:
        raise GGUFError(f"a string of {length} bytes is not a header field")
    return _read(handle, length).decode("utf-8", errors="replace")


def _value(handle: BinaryIO, kind: int, key: str, lengths: Dict[str, int]):
    if kind in _SCALAR:
        return _unpack(handle, _SCALAR[kind])
    if kind == _STRING:
        return _string(handle)
    if kind == _ARRAY:
        item_kind = _unpack(handle, "<I")
        count = _unpack(handle, "<Q")
        if count > _MAX_COUNT:
            raise GGUFError(f"an array of {count} items is not a header field")
        keep = count <= _KEEP_ARRAY_MAX
        items: List[Any] = []
        if item_kind in _SCALAR and not keep:
            # Fixed-width items are skipped in one seek rather than read.
            handle.seek(count * struct.calcsize(_SCALAR[item_kind]), 1)
        else:
            for _ in range(count):
                item = _value(handle, item_kind, key, lengths)
                if keep:
                    items.append(item)
        if not keep:
            lengths[key] = count
            return None
        return items
    raise GGUFError(f"unknown metadata type {kind} at {key!r}")


def read_header(path: str | Path) -> GGUFHeader:
    """Read a GGUF file's metadata and tensor table. Raises `GGUFError`."""
    with open(path, "rb") as handle:
        if _read(handle, 4) != MAGIC:
            raise GGUFError("not a GGUF file (no GGUF magic)")
        version = _unpack(handle, "<I")
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} is not supported")
        tensor_count = _unpack(handle, "<Q")
        kv_count = _unpack(handle, "<Q")
        if tensor_count > _MAX_COUNT or kv_count > _MAX_COUNT:
            raise GGUFError("implausible tensor or key count")
        metadata: Dict[str, Any] = {}
        lengths: Dict[str, int] = {}
        for _ in range(kv_count):
            key = _string(handle)
            kind = _unpack(handle, "<I")
            value = _value(handle, kind, key, lengths)
            if value is not None:
                metadata[key] = value
        tensors: List[TensorInfo] = []
        for _ in range(tensor_count):
            name = _string(handle)
            n_dims = _unpack(handle, "<I")
            if n_dims > _MAX_DIMS:
                raise GGUFError(f"tensor {name!r} has {n_dims} dimensions")
            shape = tuple(_unpack(handle, "<Q") for _ in range(n_dims))
            ggml_type = _unpack(handle, "<I")
            _unpack(handle, "<Q")          # data offset: not needed to describe
            tensors.append(TensorInfo(name, shape, ggml_type))
    return GGUFHeader(version=version, metadata=metadata,
                      tensors=tuple(tensors), array_lengths=lengths)


def is_gguf(path: str | Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == MAGIC
    except OSError:
        return False


__all__ = ["FILE_TYPES", "GGML_TYPES", "GGUFError", "GGUFHeader", "TensorInfo", "is_gguf",
           "read_header"]
