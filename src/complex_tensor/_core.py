# Stub based on the public API described in torch._subclasses.complex_tensor._core.
# Replace this file with the nightly source once you have it:
#
#   python -c "
#   import inspect, torch._subclasses.complex_tensor._core as m
#   print(inspect.getsource(m))
#   "
#
# What this stub provides: enough to import the package, construct ComplexTensors,
# and round-trip via from_interleaved/as_interleaved. Op dispatch raises
# NotImplementedError until _ops/aten.py is filled in.

from __future__ import annotations

import torch
from complex_tensor._ops.common import lookup_complex


class Complex(torch.autograd.Function):
    """Autograd bridge: lifts a (re, im) → ComplexTensor forward pass so that
    gradients flow back to the real/imaginary parameter tensors."""

    @staticmethod
    def forward(ctx, re: torch.Tensor, im: torch.Tensor) -> "ComplexTensor":
        return ComplexTensor(re, im)

    @staticmethod
    def backward(ctx, grad_output: "ComplexTensor"):
        return grad_output.re, grad_output.im


class ComplexTensor(torch.Tensor):
    """Uninterleaved complex tensor subclass compatible with torch.compile.

    Stores real and imaginary parts as separate float tensors rather than
    PyTorch's native interleaved complex layout, so inductor can lower every
    op to real-arithmetic kernels without hitting the complex fallback path.
    """

    @staticmethod
    def __new__(
        cls,
        re: torch.Tensor,
        im: torch.Tensor,
    ) -> "ComplexTensor":
        assert re.shape == im.shape, f"re/im shape mismatch: {re.shape} vs {im.shape}"
        assert re.dtype == im.dtype, f"re/im dtype mismatch: {re.dtype} vs {im.dtype}"
        assert re.device == im.device, f"re/im device mismatch"

        _complex_dtype = {
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }.get(re.dtype, torch.complex64)

        res = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size=re.size(),
            strides=re.stride(),
            storage_offset=0,
            dtype=_complex_dtype,
            layout=re.layout,
            device=re.device,
            requires_grad=re.requires_grad or im.requires_grad,
        )
        res.re = re
        res.im = im
        return res

    @classmethod
    def from_interleaved(cls, tensor: torch.Tensor) -> "ComplexTensor":
        """Construct from a native PyTorch complex tensor."""
        assert tensor.is_complex(), f"expected complex tensor, got {tensor.dtype}"
        return cls(tensor.real.clone(), tensor.imag.clone())

    def as_interleaved(self) -> torch.Tensor:
        """Convert back to a native PyTorch complex tensor."""
        return torch.view_as_complex(torch.stack([self.re, self.im], dim=-1).contiguous())

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from complex_tensor._ops.aten import (
            _mul, _mul_scalar, _add, _add_scalar, _sub, _neg,
            _div_tensor, _div_scalar, _einsum, _irfft2,
        )

        def _scalar(x):
            return isinstance(x, (int, float, bool))

        if func is torch.einsum:
            equation = args[0]
            operands = list(args[1]) if (len(args) == 2 and isinstance(args[1], (list, tuple))) else list(args[1:])
            return _einsum(equation, operands, path=kwargs.get("path"))

        if func is torch.fft.irfft2:
            return _irfft2(
                args[0],
                s=args[1] if len(args) > 1 else kwargs.get("s"),
                dim=args[2] if len(args) > 2 else kwargs.get("dim", (-2, -1)),
                norm=args[3] if len(args) > 3 else kwargs.get("norm"),
            )

        # Match arithmetic operators by name — during JIT tracing, func arrives as a
        # C-level descriptor (torch._C._TensorBase.__mul__) not torch.Tensor.__mul__,
        # so identity checks are version-fragile.
        fn = getattr(func, '__name__', '')

        if fn in ('__mul__', '__rmul__', 'mul'):
            x, y = args[0], args[1]
            if _scalar(y): return _mul_scalar(x, y)
            if _scalar(x): return _mul_scalar(y, x)
            return _mul(x, y)

        if fn in ('__add__', '__radd__', 'add'):
            x, y = args[0], args[1]
            if _scalar(y): return _add_scalar(x, y)
            if _scalar(x): return _add_scalar(y, x)
            return _add(x, y, **kwargs)

        if fn in ('__sub__', '__rsub__', 'sub'):
            x, y = args[0], args[1]
            if _scalar(y): return _add_scalar(x, -y)
            return _sub(x, y, **kwargs)

        if fn in ('__neg__', 'neg'):
            return _neg(args[0])

        if fn in ('__truediv__', '__div__', 'div', 'true_divide'):
            x, y = args[0], args[1]
            if _scalar(y): return _div_scalar(x, y)
            return _div_tensor(x, y)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        impl = lookup_complex(func)
        return impl(*args, **(kwargs or {}))

    def __tensor_flatten__(self):
        return ["re", "im"], {"dtype": str(self.re.dtype)}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        re = inner_tensors["re"]
        im = inner_tensors["im"]
        return ComplexTensor(re, im)

    def __repr__(self):
        return f"ComplexTensor(re={self.re}, im={self.im})"
