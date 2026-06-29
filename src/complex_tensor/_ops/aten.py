from itertools import product as _iter_product

import torch

# ComplexTensor is imported lazily inside each function to break the circular dep:
#   _core.py → _ops.common → _ops/__init__.py → aten.py → _core.py
from complex_tensor._ops.common import register_complex

aten = torch.ops.aten


def _split(x):
    """(re, im) for ComplexTensor; (x, zeros) for a real tensor."""
    from complex_tensor._core import ComplexTensor
    if isinstance(x, ComplexTensor):
        return x.re, x.im
    return x, torch.zeros_like(x)


# Infrastructure ops — called by PyTorch internally before user ops reach dispatch.

_COMPLEX_TO_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}


@register_complex(aten._to_copy.default)
def _to_copy(x, *, dtype=None, layout=None, device=None, pin_memory=None,
             non_blocking=False, memory_format=None):
    from complex_tensor._core import ComplexTensor
    # If caller requested a complex dtype, translate it to the real counterpart.
    real_dtype = _COMPLEX_TO_REAL.get(dtype, dtype) if dtype is not None else None
    kw = {k: v for k, v in dict(
        dtype=real_dtype, layout=layout, device=device,
        pin_memory=pin_memory, memory_format=memory_format,
    ).items() if v is not None}
    new_re = aten._to_copy.default(x.re, non_blocking=non_blocking, **kw)
    new_im = aten._to_copy.default(x.im, non_blocking=non_blocking, **kw)
    return ComplexTensor(new_re, new_im)


@register_complex(aten.detach.default)
def _detach(x):
    from complex_tensor._core import ComplexTensor
    return ComplexTensor(x.re.detach(), x.im.detach())


@register_complex(aten.clone.default)
def _clone(x, *, memory_format=None):
    from complex_tensor._core import ComplexTensor
    return ComplexTensor(
        x.re.clone(memory_format=memory_format),
        x.im.clone(memory_format=memory_format),
    )


# Indexing / shape ops.

def _apply_to_both(fn):
    """Helper: apply an ATen op to re and im identically, return ComplexTensor."""
    def impl(x, *args, **kwargs):
        from complex_tensor._core import ComplexTensor
        return ComplexTensor(fn(x.re, *args, **kwargs), fn(x.im, *args, **kwargs))
    return impl


@register_complex(aten.select.int)
def _select(x, dim, index):
    return _apply_to_both(aten.select.int)(x, dim, index)


@register_complex(aten.slice.Tensor)
def _slice(x, dim=0, start=None, end=None, step=1):
    return _apply_to_both(aten.slice.Tensor)(x, dim, start, end, step)


@register_complex(aten.index.Tensor)
def _index(x, indices):
    return _apply_to_both(aten.index.Tensor)(x, indices)


@register_complex(aten.view.default)
def _view(x, size):
    return _apply_to_both(aten.view.default)(x, size)


@register_complex(aten.permute.default)
def _permute(x, dims):
    return _apply_to_both(aten.permute.default)(x, dims)


@register_complex(aten.unsqueeze.default)
def _unsqueeze(x, dim):
    return _apply_to_both(aten.unsqueeze.default)(x, dim)


@register_complex(aten.squeeze.dim)
def _squeeze_dim(x, dim):
    return _apply_to_both(aten.squeeze.dim)(x, dim)


@register_complex(aten.squeeze.default)
def _squeeze(x):
    return _apply_to_both(aten.squeeze.default)(x)


@register_complex(aten.t.default)
def _t(x):
    return _apply_to_both(aten.t.default)(x)


@register_complex(aten.transpose.int)
def _transpose(x, dim0, dim1):
    return _apply_to_both(aten.transpose.int)(x, dim0, dim1)


# Arithmetic.

@register_complex(aten.add.Tensor)
def _add(x, y, *, alpha=1):
    from complex_tensor._core import ComplexTensor
    a, b = _split(x)
    c, d = _split(y)
    return ComplexTensor(a + alpha * c, b + alpha * d)


@register_complex(aten.add.Scalar)
def _add_scalar(x, other, alpha=1):
    from complex_tensor._core import ComplexTensor
    return ComplexTensor(x.re + alpha * other, x.im)


@register_complex(aten.sub.Tensor)
def _sub(x, y, *, alpha=1):
    from complex_tensor._core import ComplexTensor
    a, b = _split(x)
    c, d = _split(y)
    return ComplexTensor(a - alpha * c, b - alpha * d)


@register_complex(aten.neg.default)
def _neg(x):
    from complex_tensor._core import ComplexTensor
    return ComplexTensor(-x.re, -x.im)


@register_complex(aten.mul.Scalar)
def _mul_scalar(x, scalar):
    from complex_tensor._core import ComplexTensor
    return ComplexTensor(x.re * scalar, x.im * scalar)


@register_complex(aten.div.Tensor)
def _div_tensor(x, y):
    from complex_tensor._core import ComplexTensor
    a, b = _split(x)
    c, d = _split(y)
    # (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
    denom = c * c + d * d
    return ComplexTensor((a * c + b * d) / denom, (b * c - a * d) / denom)


@register_complex(aten.div.Scalar)
def _div_scalar(x, scalar):
    from complex_tensor._core import ComplexTensor
    return ComplexTensor(x.re / scalar, x.im / scalar)


@register_complex(aten.mul.Tensor)
def _mul(x, y):
    from complex_tensor._core import ComplexTensor
    a, b = _split(x)
    c, d = _split(y)
    return ComplexTensor(a * c - b * d, a * d + b * c)


# einsum — 2^n real-einsum decomposition.
# For n complex factors Z_k = a_k + i*b_k:
#   einsum(eq, Z_0..Z_{n-1}) = Σ_{s∈{re,im}^n}  i^(#im) * einsum(eq, F_0[s_0]..F_{n-1}[s_{n-1}])
# i^0=+re, i^1=+im, i^2=-re, i^3=-im.  n=5 → 32 real einsums on float tensors.

@register_complex(aten.einsum.default)
def _einsum(equation, operands, path=None):
    from complex_tensor._core import ComplexTensor
    pairs = [_split(op) for op in operands]
    n = len(pairs)
    result_re = result_im = None

    for selections in _iter_product([0, 1], repeat=n):
        n_imag = sum(selections)
        power = n_imag % 4
        sign = 1 if power < 2 else -1
        real_ops = [pairs[i][s] for i, s in enumerate(selections)]
        term = torch.einsum(equation, *real_ops) * sign

        if power % 2 == 1:  # imaginary contribution
            result_im = term if result_im is None else result_im + term
        else:               # real contribution
            result_re = term if result_re is None else result_re + term

    re = result_re if result_re is not None else torch.zeros_like(result_im)
    im = result_im if result_im is not None else torch.zeros_like(result_re)
    return ComplexTensor(re, im)


@register_complex(aten.fft_irfft2.default)
def _irfft2(input, s=None, dim=(-2, -1), norm=None):
    """2D inverse real FFT via DFT matrix multiplication.

    Two real matmul passes so the ONNX exporter never sees aten::view_as_complex
    or aten::complex (both unsupported in opset 17). DFT matrices are computed
    from numpy at trace time and become ONNX Constant nodes.

    Pass 1: full IFFT along H    → complex intermediate (re1, im1)
    Pass 2: one-sided IRFFT along W  → real output
    """
    import math
    import numpy as np

    re, im = input.re, input.im      # [..., H_in, W_rfft]
    H_in   = re.shape[-2]
    W_rfft = re.shape[-1]

    if s is not None:
        H_out, W_out = int(s[-2]), int(s[-1])
    else:
        H_out = H_in
        W_out = 2 * (W_rfft - 1)    # irfft default: n = 2*(m-1)

    device, dtype = re.device, re.dtype
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    # Normalisation
    if norm == 'ortho':
        sh, sw = 1.0 / math.sqrt(H_out), 1.0 / math.sqrt(W_out)
    elif norm == 'forward':
        sh, sw = 1.0, 1.0
    else:   # 'backward' (PyTorch default)
        sh, sw = 1.0 / H_out, 1.0 / W_out

    # ── Pass 1: full IFFT along H (all H_in spectral lines → H_out spatial) ──
    k1  = np.arange(H_in,  dtype=np_dtype)
    n1  = np.arange(H_out, dtype=np_dtype)
    th  = (2 * math.pi / H_out) * np.outer(n1, k1)    # [H_out, H_in]
    Cr_h = torch.tensor(np.cos(th) * sh, dtype=dtype, device=device)
    Sr_h = torch.tensor(np.sin(th) * sh, dtype=dtype, device=device)

    # re[...,H_in,W] → re.T[...,W,H_in] @ Cr_h.T[H_in,H_out] → re1.T[...,W,H_out]
    reT, imT = re.transpose(-2, -1), im.transpose(-2, -1)
    re1 = (reT @ Cr_h.t() - imT @ Sr_h.t()).transpose(-2, -1)   # [..., H_out, W_rfft]
    im1 = (reT @ Sr_h.t() + imT @ Cr_h.t()).transpose(-2, -1)

    # ── Pass 2: one-sided IRFFT along W (W_rfft → W_out, real output) ────────
    k2 = np.arange(W_rfft, dtype=np_dtype)
    n2 = np.arange(W_out,  dtype=np_dtype)
    tw = (2 * math.pi / W_out) * np.outer(n2, k2)     # [W_out, W_rfft]

    # Hermitian scale: DC and Nyquist appear once; all interior bins appear twice
    sk = np.full(W_rfft, 2.0, dtype=np_dtype)
    sk[0] = 1.0                                         # DC
    if W_out % 2 == 0 and W_rfft > 1:                  # even length → Nyquist bin
        sk[-1] = 1.0

    C_w = torch.tensor(np.cos(tw) * (sk * sw), dtype=dtype, device=device)
    S_w = torch.tensor(np.sin(tw) * (sk * sw), dtype=dtype, device=device)

    # result[...,h,n] = Σ_k C_w[n,k]*re1[...,h,k] - S_w[n,k]*im1[...,h,k]
    return re1 @ C_w.t() - im1 @ S_w.t()               # [..., H_out, W_out]
