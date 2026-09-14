"""Caller-owned native BF16 shadow for a complete declared TL master set.

No checkpoint/model is loaded and torch is imported only at construction.
The differentiable path is a search surrogate; finite calls use ordinary native
BF16 parameters after synchronization. See SHADOW.md for deployment limitations.
"""

import importlib
import re
from collections.abc import Mapping, MutableMapping


class ShadowScopeError(RuntimeError):
    pass


_MASTER_NAME = re.compile(r"^blocks\.(0|[1-9][0-9]*)\.mlp\.(W_in|W_gate|W_out)$")
_HF_NAMES = {"W_in": "up_proj", "W_gate": "gate_proj", "W_out": "down_proj"}


def hf_name(master_name):
    match = _MASTER_NAME.fullmatch(master_name) if isinstance(master_name, str) else None
    if match is None:
        raise ShadowScopeError("Unsupported TL master name: %r" % (master_name,))
    layer, matrix = match.groups()
    return "model.layers.%s.mlp.%s.weight" % (layer, _HF_NAMES[matrix])


class NativeBF16Shadow:
    """Exclusive native HF object plus all edited/inherited FP32 TL masters.

    `declared_master_names` is a caller-bound complete roster, not inferred from
    the current active window. Constructor freezes native parameters, switches
    native inference to eval mode, syncs the whole roster and records the first
    accepted native snapshot.
    """

    def __init__(self, native_model, masters, *, declared_master_names,
                 torch_module=None, external_cache_refs=None):
        self.torch = torch_module or importlib.import_module("torch")
        self._native = native_model
        if isinstance(masters, Mapping):
            self._masters = masters
        else:
            pairs = tuple(masters)
            if len({name for name, _ in pairs}) != len(pairs):
                raise ShadowScopeError("Duplicate master names in the supplied sequence")
            self._masters = dict(pairs)
        self._declared = tuple(declared_master_names)
        if not self._declared or len(set(self._declared)) != len(self._declared):
            raise ShadowScopeError("A nonempty unique complete master roster is required")
        self._mapping = {name: hf_name(name) for name in self._declared}
        if external_cache_refs is None:
            external_cache_refs = {}
        if not isinstance(external_cache_refs, MutableMapping):
            raise ShadowScopeError("external_cache_refs must be a mutable mapping")
        self.external_cache_refs = external_cache_refs
        self.cache_epoch = 0
        self._accepted = None
        self._native_parameter_ids = None
        self._master_ids = None
        self._validate(require_frozen=False)
        for parameter in self._native.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._native.eval()
        self._native_parameter_ids = {name: id(p) for name, p in self._native.named_parameters()}
        self._master_ids = {name: id(self._masters[name]) for name in self._declared}
        self.sync_from_masters()
        self.mark_accepted()

    def __getattr__(self, name):
        # Generation guards inspect config/generation_config/dtype and HF's
        # configuration helper. Native forward/generate are explicitly wrapped.
        if name.startswith("__"):
            raise AttributeError(name)
        native = self.__dict__.get("_native")
        if native is None:
            raise AttributeError(name)
        return getattr(native, name)

    @property
    def dtype(self):
        return self.torch.bfloat16

    @property
    def device(self):
        return next(self._native.parameters()).device

    @property
    def native_model(self):
        """Caller-owned object. Direct use bypasses this bridge's synchronization."""
        return self._native

    def _validate(self, require_frozen=True):
        torch = self.torch
        if set(self._masters) != set(self._declared):
            missing = sorted(set(self._declared) - set(self._masters))
            extra = sorted(set(self._masters) - set(self._declared))
            raise ShadowScopeError("Master roster mismatch; missing=%r extra=%r" % (missing, extra))
        native = dict(self._native.named_parameters())
        if not native:
            raise ShadowScopeError("Native model has no parameters")
        if self._native_parameter_ids is not None and {name: id(p) for name, p in native.items()} != self._native_parameter_ids:
            raise ShadowScopeError("Native parameter bindings changed")
        for name, parameter in native.items():
            if not parameter.is_floating_point() or parameter.dtype != torch.bfloat16:
                raise ShadowScopeError("Native parameters must all be BF16: %s" % name)
            if require_frozen and parameter.requires_grad:
                raise ShadowScopeError("Native parameter became trainable: %s" % name)
        if require_frozen and self._native.training:
            raise ShadowScopeError("Native shadow left evaluation mode")
        for name, target_name in self._mapping.items():
            if target_name not in native:
                raise ShadowScopeError("Declared native matrix is missing: %s" % target_name)
            master = self._masters[name]
            if (not isinstance(master, torch.Tensor) or master.ndim != 2
                    or master.dtype != torch.float32 or master.layout != torch.strided):
                raise ShadowScopeError("TL masters must be dense FP32 matrices: %s" % name)
            if self._master_ids is not None and id(master) != self._master_ids[name]:
                raise ShadowScopeError("TL master object binding changed: %s" % name)
            target = native[target_name]
            if target.ndim != 2 or tuple(master.shape[::-1]) != tuple(target.shape):
                raise ShadowScopeError("TL/HF transposed matrix shape mismatch: %s" % name)
        return native

    def discard_external_cache_references(self):
        """Drop only references in the caller-supplied mapping, not native private caches."""
        self.external_cache_refs.clear()
        self.cache_epoch += 1

    def _converted(self, master, target):
        # Contiguous HF orientation matches the writer's export mapping. The
        # cast/device copy preserves autograd connectivity on the search path.
        return master.transpose(0, 1).to(device=target.device, dtype=self.torch.bfloat16).contiguous()

    def sync_from_masters(self):
        """Validate the entire roster before any copy, then materialize every matrix."""
        native = self._validate()
        self.discard_external_cache_references()
        with self.torch.no_grad():
            for name, target_name in self._mapping.items():
                target = native[target_name]
                materialized = self._converted(self._masters[name], target)
                # Checking all masters need not rewrite an unchanged tensor.
                # Avoid invalidating an existing search graph merely because
                # the controller performs a finite baseline forward before its
                # gradient call. A changed master still requires a fresh graph.
                if not self.torch.equal(target, materialized):
                    target.copy_(materialized)
        return self._changed_without_sync(native)

    def _changed_without_sync(self, native):
        if self._accepted is None:
            return self._declared
        return tuple(name for name, target_name in self._mapping.items()
                     if not self.torch.equal(native[target_name], self._accepted[name]))

    def changed_from_accepted(self):
        """All declared native matrices whose BF16 values differ from the last accepted snapshot."""
        return self.sync_from_masters()

    def mark_accepted(self):
        """Call only for a fully accepted controller result; this does not approve a candidate."""
        self.sync_from_masters()
        native = self._validate()
        self._accepted = {name: native[target_name].detach().clone() for name, target_name in self._mapping.items()}

    @staticmethod
    def _reject_external_cache(kwargs):
        if kwargs.get("past_key_values") is not None or kwargs.get("past") is not None:
            raise ShadowScopeError("Reusing external caches across synchronized candidates is unsupported")

    def functional_forward(self, active_names, *args, **kwargs):
        """Teacher-forced search through active transpose/cast replacements.

        All inactive edited/inherited masters are synchronized as frozen native
        BF16 weights. The full parameter/buffer mapping is supplied with strict
        functional_call validation. This is not the finite deployment oracle.
        """
        active = tuple(active_names)
        if not active or len(set(active)) != len(active) or not set(active).issubset(self._mapping):
            raise ShadowScopeError("Active names must be a nonempty unique subset of the complete roster")
        self._reject_external_cache(kwargs)
        if kwargs.get("use_cache", False) is not False:
            raise ShadowScopeError("Search forwards must disable cache reuse")
        kwargs["use_cache"] = False
        self.sync_from_masters()
        native = self._validate()
        parameters = dict(native)
        buffers = {name: buffer.detach().clone() for name, buffer in self._native.named_buffers()}
        for name in active:
            master = self._masters[name]
            if not master.requires_grad or not master.is_leaf:
                raise ShadowScopeError("Active masters must be trainable leaves: %s" % name)
            target_name = self._mapping[name]
            parameters[target_name] = self._converted(master, native[target_name])
        try:
            return self.torch.func.functional_call(
                self._native, (parameters, buffers), args, kwargs,
                tie_weights=True, strict=True,
            )
        finally:
            self.discard_external_cache_references()

    def forward(self, *args, **kwargs):
        """Finite ordinary native BF16 forward, resynchronized before every call."""
        self._reject_external_cache(kwargs)
        self.sync_from_masters()
        try:
            with self.torch.no_grad():
                return self._native(*args, **kwargs)
        finally:
            self.discard_external_cache_references()

    __call__ = forward

    def generate(self, *args, **kwargs):
        """Finite ordinary native generation, with fresh caller-level cache references."""
        self._reject_external_cache(kwargs)
        self.sync_from_masters()
        try:
            with self.torch.no_grad():
                return self._native.generate(*args, **kwargs)
        finally:
            self.discard_external_cache_references()

    def run_step(self, controller_call, *args, **kwargs):
        """Resync in finally AFTER the controller's own transaction has unwound.

        The callable must own master/optimizer rollback and return an object with
        a Boolean `accepted`. This wrapper owns only the native shadow/cache refs.
        """
        try:
            result = controller_call(*args, **kwargs)
            if type(getattr(result, "accepted", None)) is not bool:
                raise ShadowScopeError("Controller result must have a Boolean accepted field")
            changed = self.sync_from_masters()
            if result.accepted:
                self.mark_accepted()
            elif changed:
                raise ShadowScopeError("Rejected controller left native weights different from the last accepted state")
            return result
        finally:
            self.sync_from_masters()
            self.discard_external_cache_references()
