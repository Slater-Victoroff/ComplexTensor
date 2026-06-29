# Stub framework — replace with nightly source:
#
#   python -c "
#   import inspect, torch._subclasses.complex_tensor._ops.common as m
#   print(inspect.getsource(m))
#   "
#
# The interface below matches what the nightly version exports so that _core.py
# and the op registration decorators work the same way.

from __future__ import annotations

from typing import Any, Callable

import torch

# Maps ATen/prims op → complex implementation function.
# Populated via @register_complex / register_simple / register_binary_nonlinear.
COMPLEX_OPS_TABLE: dict[Any, Callable] = {}


def lookup_complex(func) -> Callable:
    impl = COMPLEX_OPS_TABLE.get(func)
    if impl is None:
        raise NotImplementedError(
            f"ComplexTensor: no implementation registered for {func}. "
            "Fill in _ops/aten.py from the nightly source, or add a new registration."
        )
    return impl


def register_complex(func):
    """Decorator: @register_complex(aten.some_op.default)"""
    def decorator(impl: Callable) -> Callable:
        COMPLEX_OPS_TABLE[func] = impl
        return impl
    return decorator


def register_simple(aten_op):
    """Register a linear op whose real/imag parts are independent.
    E.g. add: (a+bi) + (c+di) = (a+c) + (b+d)i
    """
    from complex_tensor._core import ComplexTensor

    def impl(*args, **kwargs):
        re_args = [a.re if isinstance(a, ComplexTensor) else a for a in args]
        im_args = [a.im if isinstance(a, ComplexTensor) else a for a in args]
        return ComplexTensor(aten_op(*re_args, **kwargs), aten_op(*im_args, **kwargs))

    COMPLEX_OPS_TABLE[aten_op] = impl
    return impl


def register_binary_nonlinear(aten_op):
    """Register a nonlinear binary op via real/imag decomposition.
    E.g. mul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    """
    from complex_tensor._core import ComplexTensor

    def impl(x, y, **kwargs):
        a, b = (x.re, x.im) if isinstance(x, ComplexTensor) else (x, torch.zeros_like(x))
        c, d = (y.re, y.im) if isinstance(y, ComplexTensor) else (y, torch.zeros_like(y))
        return ComplexTensor(a * c - b * d, a * d + b * c)

    COMPLEX_OPS_TABLE[aten_op] = impl
    return impl


def split_complex_arg(arg):
    """Return (re, im) for a ComplexTensor, or (arg, zeros) for a real tensor."""
    from complex_tensor._core import ComplexTensor
    if isinstance(arg, ComplexTensor):
        return arg.re, arg.im
    return arg, torch.zeros_like(arg)


def is_complex_tensor(x) -> bool:
    from complex_tensor._core import ComplexTensor
    return isinstance(x, ComplexTensor)


class ComplexTensorMode(torch.overrides.TorchFunctionMode):
    """Context manager: wraps outputs of ops into ComplexTensor where appropriate."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        return result


class WrapComplexMode(torch.overrides.TorchFunctionMode):
    """Context manager: automatically promotes native complex tensors to ComplexTensor."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        from complex_tensor._core import ComplexTensor

        def maybe_wrap(x):
            if isinstance(x, torch.Tensor) and x.is_complex() and not isinstance(x, ComplexTensor):
                return ComplexTensor.from_interleaved(x)
            return x

        args = torch.utils._pytree.tree_map(maybe_wrap, args)
        kwargs = torch.utils._pytree.tree_map(maybe_wrap, kwargs or {})
        return func(*args, **kwargs)
