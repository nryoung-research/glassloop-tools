"""Bounded transaction for the pilot's editable stock AdamW/SGD parameters.

Import-safe: torch is imported only when a transaction is entered. No model is
loaded. The CPU path never calls torch.cuda. See TRANSACTION.md for its scope.
"""

from collections import defaultdict
import importlib
import random
from types import MappingProxyType


class ScopeViolation(RuntimeError):
    """The candidate changed ownership, tensor metadata or unsupported state."""


class RollbackError(RuntimeError):
    """Restoration failed; the caller must discard the live candidate."""


def _tensor_metadata(tensor):
    return (
        tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
        tensor.device, tensor.layout, tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(), bool(tensor.requires_grad),
    )


class TensorTransaction:
    """Capture on context entry; commit explicitly, otherwise roll back.

    `named_parameters` must contain exactly the optimizer's unique parameter
    objects. Only stock torch.optim.AdamW and torch.optim.SGD are accepted.
    CUDA devices, if needed, must be explicitly declared and already initialized.
    """

    def __init__(self, named_parameters, optimizer, torch_module=None, capture_cuda_devices=()):
        self._named = tuple(named_parameters)
        self.optimizer = optimizer
        self._torch_arg = torch_module
        self._cuda_devices = tuple(capture_cuda_devices)
        self._entered = False
        self._closed = False
        self._committed = False

    def _clone(self, value):
        torch = self.torch
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ScopeViolation("Only dense strided tensor state is supported")
            if value.device.type == "cuda" and value.device.index not in self._cuda_devices:
                raise ScopeViolation("Optimizer tensor uses an undeclared CUDA RNG device")
            if value.device.type not in ("cpu", "cuda"):
                raise ScopeViolation("Only CPU/CUDA tensor state is supported")
            return value.detach().clone()
        if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
            return value
        if isinstance(value, list):
            return [self._clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone(item) for item in value)
        if isinstance(value, dict):
            if isinstance(value, defaultdict):
                if value.default_factory not in (None, dict):
                    raise ScopeViolation("Unsupported defaultdict factory")
                result = defaultdict(value.default_factory)
            else:
                result = type(value)()
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, tuple)):
                    raise ScopeViolation("Unsupported optimizer-state key")
                result[key] = self._clone(item)
            return result
        raise ScopeViolation("Unsupported optimizer state type: %s" % type(value).__name__)

    def _validate_tree(self, value):
        """The same scope validation without copying large moment tensors."""
        torch = self.torch
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.device.type not in ("cpu", "cuda"):
                raise ScopeViolation("Unsupported optimizer tensor layout/device")
            if value.device.type == "cuda" and value.device.index not in self._cuda_devices:
                raise ScopeViolation("Optimizer tensor uses an undeclared CUDA RNG device")
        elif value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
            return
        elif isinstance(value, (tuple, list)):
            for item in value:
                self._validate_tree(item)
        elif isinstance(value, dict):
            if isinstance(value, defaultdict) and value.default_factory not in (None, dict):
                raise ScopeViolation("Unsupported optimizer state factory")
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, tuple)):
                    raise ScopeViolation("Unsupported optimizer-state key")
                self._validate_tree(item)
        else:
            raise ScopeViolation("Unsupported optimizer state type: %s" % type(value).__name__)

    def _check_hooks(self):
        for name, value in vars(self.optimizer).items():
            if name.startswith("_optimizer_") and name.endswith("_hooks") and value:
                raise ScopeViolation("Optimizer hooks are outside this transaction's scope")
        optimizer_module = importlib.import_module("torch.optim.optimizer")
        for name in ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks"):
            if getattr(optimizer_module, name, None):
                raise ScopeViolation("Global optimizer hooks are outside this transaction's scope")

    def _check_optimizer_tensor_scope(self):
        torch = self.torch
        for group in self.optimizer.param_groups:
            if group.get("differentiable", False):
                raise ScopeViolation("Differentiable optimizer graphs are outside this transaction's scope")
            for parameter in group["params"]:
                state = self.optimizer.state.get(parameter, {})
                if not isinstance(state, dict):
                    raise ScopeViolation("Per-parameter optimizer state must be a dictionary")
                if not state:
                    continue
                if type(self.optimizer) is torch.optim.AdamW:
                    moments = {"exp_avg", "exp_avg_sq"}
                    if group.get("amsgrad", False):
                        moments.add("max_exp_avg_sq")
                    if set(state) != moments | {"step"}:
                        raise ScopeViolation("AdamW state schema changed")
                    step = state["step"]
                    if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                            or step.dtype not in (torch.float32, torch.float64)
                            or step.device.type not in ("cpu", parameter.device.type)
                            or step.requires_grad):
                        raise ScopeViolation("AdamW step-counter metadata changed")
                else:
                    moments = {"momentum_buffer"}
                    if set(state) != moments:
                        raise ScopeViolation("SGD state schema changed")
                for key in moments:
                    value = state[key]
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or value.dtype != parameter.dtype or value.device != parameter.device
                            or value.layout != torch.strided or value.requires_grad):
                        raise ScopeViolation("Optimizer moment metadata changed: %s" % key)
                self._validate_tree(state)

    def _check_group_ownership(self):
        current = self.optimizer.param_groups
        if len(current) != len(self._groups):
            raise ScopeViolation("Optimizer group count changed")
        for group, saved in zip(current, self._groups):
            if tuple(id(p) for p in group["params"]) != saved["ids"]:
                raise ScopeViolation("Optimizer parameter membership or order changed")
        if any(id(key) not in self._parameter_ids for key in self.optimizer.state):
            raise ScopeViolation("Optimizer owns state outside the declared parameters")

    def _check_live_scope(self):
        self._check_group_ownership()
        self._check_hooks()
        self._check_optimizer_tensor_scope()
        for saved in self._parameters:
            parameter = saved["parameter"]
            if _tensor_metadata(parameter) != saved["metadata"]:
                raise ScopeViolation("Parameter metadata/storage changed: %s" % saved["name"])
            gradient = parameter.grad
            if gradient is not None:
                if (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                        or gradient.device != parameter.device or gradient.layout != self.torch.strided):
                    raise ScopeViolation("Gradient metadata changed: %s" % saved["name"])

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("Transactions are single-use")
        self.torch = self._torch_arg or importlib.import_module("torch")
        torch = self.torch
        if type(self.optimizer) not in (torch.optim.AdamW, torch.optim.SGD):
            raise ScopeViolation("Only exact stock AdamW/SGD classes are supported")
        if not self._named:
            raise ScopeViolation("Editable scope is empty")
        if len(set(self._cuda_devices)) != len(self._cuda_devices) or any(
            type(device) is not int or device < 0 for device in self._cuda_devices
        ):
            raise ScopeViolation("CUDA devices must be unique nonnegative integer indices")
        names = [name for name, _ in self._named]
        parameters = [parameter for _, parameter in self._named]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ScopeViolation("Parameter names must be unique nonempty strings")
        self._parameter_ids = {id(parameter) for parameter in parameters}
        if len(self._parameter_ids) != len(parameters):
            raise ScopeViolation("Duplicate parameter objects")
        owned = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        if len(owned) != len(parameters) or {id(parameter) for parameter in owned} != self._parameter_ids:
            raise ScopeViolation("Editable scope must exactly equal optimizer ownership")
        self._check_hooks()
        self._check_optimizer_tensor_scope()
        storage_keys = set()
        for name, parameter in self._named:
            if not isinstance(parameter, torch.Tensor) or not parameter.is_leaf:
                raise ScopeViolation("Only leaf tensors can be transacted: %s" % name)
            if parameter.layout != torch.strided or parameter.numel() == 0:
                raise ScopeViolation("Only nonempty dense strided parameters are supported")
            if parameter.device.type not in ("cpu", "cuda"):
                raise ScopeViolation("Only CPU/CUDA parameters are supported")
            if parameter.device.type == "cuda" and parameter.device.index not in self._cuda_devices:
                raise ScopeViolation("CUDA parameter device was not declared for RNG capture")
            storage_key = (parameter.device, parameter.untyped_storage().data_ptr())
            if storage_key in storage_keys:
                raise ScopeViolation("Aliased editable parameter storage is unsupported")
            storage_keys.add(storage_key)

        # No CUDA API is touched in the all-CPU, empty-device-list path.
        if self._cuda_devices and not torch.cuda.is_initialized():
            raise ScopeViolation("CUDA must already be initialized; transaction will not initialize it")
        self._python_rng = random.getstate()
        self._cpu_rng = torch.get_rng_state().clone()
        self._cuda_rng = {device: torch.cuda.get_rng_state(device).clone() for device in self._cuda_devices}

        self._parameters = []
        for name, parameter in self._named:
            gradient = parameter.grad
            self._parameters.append({
                "name": name, "parameter": parameter, "metadata": _tensor_metadata(parameter),
                "storage": parameter.detach(), "value": parameter.detach().clone(),
                "gradient_object": gradient,
                "gradient_requires_grad": None if gradient is None else bool(gradient.requires_grad),
                "gradient_storage": None if gradient is None else gradient.detach(),
                "gradient_value": None if gradient is None else gradient.detach().clone(),
            })
        self._groups_ref = self.optimizer.param_groups
        self._groups = []
        for group in self.optimizer.param_groups:
            self._groups.append({
                "object": group, "params_object": group["params"],
                "parameters": tuple(group["params"]), "ids": tuple(id(p) for p in group["params"]),
                "values": self._clone({key: value for key, value in group.items() if key != "params"}),
            })
        self._state_ref = self.optimizer.state
        self._state_factory = getattr(self._state_ref, "default_factory", None)
        if type(self._state_ref) not in (dict, defaultdict):
            raise ScopeViolation("Unsupported optimizer state mapping")
        if isinstance(self._state_ref, defaultdict) and self._state_ref.default_factory not in (None, dict):
            raise ScopeViolation("Unsupported optimizer state factory")
        if any(id(parameter) not in self._parameter_ids for parameter in self.optimizer.state):
            raise ScopeViolation("Optimizer contains state for an undeclared parameter")
        # Clone the actual parameter-keyed state directly. Unlike load_state_dict,
        # restoring this tree does not recast moment/counter dtypes or devices.
        self._optimizer_state = [(parameter, self._clone(value)) for parameter, value in self.optimizer.state.items()]
        self._optimizer_aux = {
            key: self._clone(value) for key, value in vars(self.optimizer).items()
            if key not in ("state", "param_groups")
        }
        self._entered = True
        return self

    @property
    def original_weights(self):
        """Read-only mapping; its tensor values are INTERNAL snapshots: never mutate them."""
        self._require_open()
        return MappingProxyType({saved["name"]: saved["value"] for saved in self._parameters})

    def _require_open(self):
        if not self._entered or self._closed:
            raise RuntimeError("Transaction is not open")

    def restore_weights_only(self):
        """Restore pre-step values, keeping proposed moments, gradients and RNG state."""
        self._require_open()
        self._check_live_scope()
        with self.torch.no_grad():
            for saved in self._parameters:
                saved["parameter"].copy_(saved["value"])

    def commit(self):
        """Call only after the caller's actual candidate postcheck succeeds."""
        self._require_open()
        self._check_live_scope()
        # Reject newly introduced unsupported auxiliary/state values as well.
        for key, value in vars(self.optimizer).items():
            if key not in ("state", "param_groups"):
                self._validate_tree(value)
        for value in self.optimizer.state.values():
            self._validate_tree(value)
        self._committed = True

    def rollback(self):
        self._require_open()
        try:
            with self.torch.no_grad():
                for saved in self._parameters:
                    parameter = saved["parameter"]
                    parameter.data = saved["storage"]
                    parameter.copy_(saved["value"])
                    parameter.requires_grad_(saved["metadata"][-1])
                    original_gradient = saved["gradient_object"]
                    if original_gradient is None:
                        parameter.grad = None
                    else:
                        original_gradient.data = saved["gradient_storage"]
                        original_gradient.copy_(saved["gradient_value"])
                        original_gradient.requires_grad_(saved["gradient_requires_grad"])
                        parameter.grad = original_gradient

            # Restore group objects/references as well as their hyperparameters.
            self._groups_ref[:] = [saved["object"] for saved in self._groups]
            self.optimizer.param_groups = self._groups_ref
            for saved in self._groups:
                saved["params_object"][:] = saved["parameters"]
                group = saved["object"]
                group.clear()
                group.update(self._clone(saved["values"]))
                group["params"] = saved["params_object"]
            self._state_ref.clear()
            if isinstance(self._state_ref, defaultdict):
                self._state_ref.default_factory = self._state_factory
            for parameter, value in self._optimizer_state:
                self._state_ref[parameter] = self._clone(value)
            self.optimizer.state = self._state_ref
            for key in tuple(vars(self.optimizer)):
                if key not in ("state", "param_groups") and key not in self._optimizer_aux:
                    delattr(self.optimizer, key)
            for key, value in self._optimizer_aux.items():
                setattr(self.optimizer, key, self._clone(value))
            random.setstate(self._python_rng)
            self.torch.set_rng_state(self._cpu_rng)
            for device, state in self._cuda_rng.items():
                self.torch.cuda.set_rng_state(state, device)
            self._check_live_scope()
        except BaseException as exc:
            self._closed = True
            raise RollbackError("Full rollback failed; discard the live candidate") from exc
        self._committed = False
        self._closed = True

    def __exit__(self, exc_type, exc_value, traceback):
        if self._closed:
            return False
        if exc_type is not None or not self._committed:
            self.rollback()
            return False
        try:
            self._check_live_scope()
        except BaseException:
            self.rollback()
            raise
        self._closed = True
        return False
