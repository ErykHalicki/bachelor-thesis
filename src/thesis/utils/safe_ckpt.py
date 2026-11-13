"""Read a whole-object torch checkpoint (`torch.save(module)`) without running its code.

A pickled `nn.Module` names its classes, and `torch.load(weights_only=False)` imports and
calls whatever those names resolve to, so loading a downloaded one executes code from the
file. This unpickler resolves only the handful of torch globals a tensor needs; every other
global becomes an inert `_Stub` that records its name and state. The stub tree carries each
module's `_parameters` / `_buffers` / `_modules`, which is all a state dict is, plus its
plain attributes (strides, levels, sizes) for callers that need a model's geometry.
"""

import collections
import io
import pickle
import zipfile

import torch

_DTYPES = {
    "FloatStorage": torch.float32, "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64, "IntStorage": torch.int32, "ShortStorage": torch.int16,
    "CharStorage": torch.int8, "ByteStorage": torch.uint8, "BoolStorage": torch.bool,
}


class _Stub:
    _qualname = "?"

    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = {**(state[0] or {}), **state[1]}
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def __repr__(self):
        return f"<{self._qualname}>"


def _stub_class(module, name):
    return type(name, (_Stub,), {"_qualname": f"{module}.{name}"})


def _storage_type(module, name):
    return ("storage", _DTYPES.get(name, torch.float32))


def _rebuild_tensor(storage, offset, size, stride, *rest):
    return torch.as_strided(storage, tuple(size), tuple(stride), int(offset))


def _rebuild_parameter(data, *rest):
    return data


def _reconstructor(cls, base, state):
    return cls.__new__(cls) if isinstance(cls, type) else _Stub()


class _Unpickler(pickle.Unpickler):
    def __init__(self, data, archive, prefix):
        super().__init__(data)
        self.archive = archive
        self.prefix = prefix
        self.storages = {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module in ("builtins", "__builtin__") and name in ("set", "frozenset", "dict", "list", "tuple"):
            return {"set": set, "frozenset": frozenset, "dict": dict, "list": list,
                    "tuple": tuple}[name]
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name in ("_rebuild_parameter",
                                                 "_rebuild_parameter_with_state"):
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _storage_type(module, name)
        if module == "torch" and name == "Size":
            return tuple
        if module == "torch" and name == "device":
            return str
        if module == "copyreg" and name == "_reconstructor":
            return _reconstructor
        return _stub_class(module, name)

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid
        assert kind == "storage", f"unexpected persistent id {pid!r}"
        dtype = storage_type[1] if isinstance(storage_type, tuple) else torch.float32
        if key not in self.storages:
            raw = bytearray(self.archive.read(f"{self.prefix}/data/{key}"))
            flat = torch.frombuffer(raw, dtype=torch.uint8) if raw else torch.empty(0, dtype=torch.uint8)
            self.storages[key] = flat.view(dtype) if dtype != torch.bool else flat.bool()
        return self.storages[key]


def load_object(path):
    """The pickled object as a tree of `_Stub`s with real tensors at the leaves."""
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        return _Unpickler(io.BytesIO(archive.read(pkl)), archive, prefix).load()


def _walk(stub, prefix, sd, meta):
    state = stub.__dict__
    meta[prefix.rstrip(".")] = {
        "class": stub._qualname,
        **{k: v for k, v in state.items()
           if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))},
    }
    skip = set(state.get("_non_persistent_buffers_set") or ())
    for name, value in (state.get("_parameters") or {}).items():
        if value is not None:
            sd[prefix + name] = value.clone()
    for name, value in (state.get("_buffers") or {}).items():
        if value is not None and name not in skip:
            sd[prefix + name] = value.clone()
    for name, child in (state.get("_modules") or {}).items():
        if isinstance(child, _Stub):
            _walk(child, f"{prefix}{name}.", sd, meta)


def load_object_state_dict(path):
    """`torch.save(module)` file -> (state_dict, {module path: {class, plain attrs}})."""
    root = load_object(path)
    sd, meta = collections.OrderedDict(), {}
    _walk(root, "", sd, meta)
    return sd, meta
