"""Tests for ComplexTensor op dispatch.

Organized into three sections:
  - Existing ops  (marked @pytest.mark.nightly)  — should pass once nightly aten.py is pasted in
  - einsum        (marked @pytest.mark.new_op)    — new, will fail until we implement it
  - fft           (marked @pytest.mark.new_op)    — new, will fail until we implement it

Every test follows the same pattern: compute via ComplexTensor subclass, compute via
native torch.complex as reference, assert they match via torch.testing.assert_close.
"""

import pytest
import torch
import torch.testing

from complex_tensor import ComplexTensor
from conftest import rand_complex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ct(*shape, dtype=torch.float32, device="cpu"):
    re, im, _ = rand_complex(shape, dtype=dtype, device=device)
    return ComplexTensor(re, im)


def native(*shape, dtype=torch.float32, device="cpu"):
    re, im, n = rand_complex(shape, dtype=dtype, device=device)
    return n


def assert_ct_close(result, reference, **kwargs):
    """Compare a ComplexTensor result against a native complex reference."""
    if isinstance(result, ComplexTensor):
        result = result.as_interleaved()
    torch.testing.assert_close(result, reference, **kwargs)


# ---------------------------------------------------------------------------
# Existing ops (nightly)
# ---------------------------------------------------------------------------

@pytest.mark.nightly
class TestLinearOps:
    """Ops whose real/imag parts transform independently: (a+bi) op (c+di) = (a op c) + (b op d)i"""

    def test_add(self, dtype, device):
        re1, im1, n1 = rand_complex((4, 4), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((4, 4), dtype=dtype, device=device)
        result = ComplexTensor(re1, im1) + ComplexTensor(re2, im2)
        assert_ct_close(result, n1 + n2)

    def test_sub(self, dtype, device):
        re1, im1, n1 = rand_complex((4, 4), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((4, 4), dtype=dtype, device=device)
        result = ComplexTensor(re1, im1) - ComplexTensor(re2, im2)
        assert_ct_close(result, n1 - n2)

    def test_neg(self, dtype, device):
        re, im, n = rand_complex((4,), dtype=dtype, device=device)
        result = -ComplexTensor(re, im)
        assert_ct_close(result, -n)

    def test_add_scalar(self, dtype, device):
        re, im, n = rand_complex((4,), dtype=dtype, device=device)
        result = ComplexTensor(re, im) + 2.0
        assert_ct_close(result, n + 2.0)


@pytest.mark.nightly
class TestNonlinearOps:
    def test_mul(self, dtype, device):
        re1, im1, n1 = rand_complex((4, 4), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((4, 4), dtype=dtype, device=device)
        result = ComplexTensor(re1, im1) * ComplexTensor(re2, im2)
        assert_ct_close(result, n1 * n2)

    def test_mm(self, dtype, device):
        re1, im1, n1 = rand_complex((3, 4), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((4, 5), dtype=dtype, device=device)
        result = ComplexTensor(re1, im1) @ ComplexTensor(re2, im2)
        assert_ct_close(result, n1 @ n2)

    def test_abs(self, dtype, device):
        re, im, n = rand_complex((4,), dtype=dtype, device=device)
        result = ComplexTensor(re, im).abs()
        torch.testing.assert_close(result, n.abs())

    def test_angle(self, dtype, device):
        re, im, n = rand_complex((4,), dtype=dtype, device=device)
        result = ComplexTensor(re, im).angle()
        torch.testing.assert_close(result, n.angle())

    def test_conj(self, dtype, device):
        re, im, n = rand_complex((4,), dtype=dtype, device=device)
        result = ComplexTensor(re, im).conj()
        assert_ct_close(result, n.conj())


@pytest.mark.nightly
class TestTensorManipulation:
    def test_reshape(self, dtype, device):
        re, im, n = rand_complex((4, 4), dtype=dtype, device=device)
        result = ComplexTensor(re, im).reshape(2, 8)
        assert_ct_close(result, n.reshape(2, 8))

    def test_cat(self, dtype, device):
        re1, im1, n1 = rand_complex((2, 4), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((3, 4), dtype=dtype, device=device)
        result = torch.cat([ComplexTensor(re1, im1), ComplexTensor(re2, im2)], dim=0)
        assert_ct_close(result, torch.cat([n1, n2], dim=0))

    def test_getitem(self, dtype, device):
        re, im, n = rand_complex((4, 4), dtype=dtype, device=device)
        result = ComplexTensor(re, im)[1:3]
        assert_ct_close(result, n[1:3])


@pytest.mark.nightly
class TestCompileWithOps:
    """torch.compile(fullgraph=True) smoke tests for nightly ops."""

    def test_compile_add(self, dtype):
        @torch.compile(fullgraph=True)
        def f(a, b):
            return a + b

        re1, im1, n1 = rand_complex((4,), dtype=dtype)
        re2, im2, n2 = rand_complex((4,), dtype=dtype)
        result = f(ComplexTensor(re1, im1), ComplexTensor(re2, im2))
        assert_ct_close(result, n1 + n2)

    def test_compile_mul(self, dtype):
        @torch.compile(fullgraph=True)
        def f(a, b):
            return a * b

        re1, im1, n1 = rand_complex((4,), dtype=dtype)
        re2, im2, n2 = rand_complex((4,), dtype=dtype)
        result = f(ComplexTensor(re1, im1), ComplexTensor(re2, im2))
        assert_ct_close(result, n1 * n2)


# ---------------------------------------------------------------------------
# New op: einsum
# ---------------------------------------------------------------------------

@pytest.mark.new_op
class TestEinsum:
    """Complex einsum via ComplexTensor.

    The general case decomposes the n-factor complex einsum into real-arithmetic
    einsums by accumulating real/imag parts pairwise across factors.
    """

    def test_einsum_two_factors(self, dtype, device):
        """(a+bi)(c+di) via einsum should match native complex."""
        re1, im1, n1 = rand_complex((3, 4), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((4, 5), dtype=dtype, device=device)
        result = torch.einsum("ij,jk->ik", ComplexTensor(re1, im1), ComplexTensor(re2, im2))
        assert_ct_close(result, torch.einsum("ij,jk->ik", n1, n2))

    def test_einsum_three_factors(self, dtype, device):
        re1, im1, n1 = rand_complex((2, 3), dtype=dtype, device=device)
        re2, im2, n2 = rand_complex((3, 4), dtype=dtype, device=device)
        re3, im3, n3 = rand_complex((4, 5), dtype=dtype, device=device)
        result = torch.einsum("ij,jk,kl->il",
                              ComplexTensor(re1, im1), ComplexTensor(re2, im2), ComplexTensor(re3, im3))
        assert_ct_close(result, torch.einsum("ij,jk,kl->il", n1, n2, n3))

    def test_einsum_tucker_5factor(self, dtype, device):
        """The exact einsum ComplexTucker uses: 'ijkl,ti,cj,hk,wl->tchw'."""
        rT, rC, rH, rW = 3, 2, 4, 4
        C, H, W_half, T = 2, 4, 5, 3  # W_half = W//2+1 = 5 for W=8

        def mk(*shape):
            re, im, n = rand_complex(shape, dtype=dtype, device=device)
            return ComplexTensor(re, im), n

        G_ct, G_n = mk(rT, rC, rH, rW)
        UT_ct, UT_n = mk(T, rT)
        UC_ct, UC_n = mk(C, rC)
        UH_ct, UH_n = mk(H, rH)
        UW_ct, UW_n = mk(W_half, rW)

        pattern = "ijkl,ti,cj,hk,wl->tchw"
        result = torch.einsum(pattern, G_ct, UT_ct, UC_ct, UH_ct, UW_ct)
        reference = torch.einsum(pattern, G_n, UT_n, UC_n, UH_n, UW_n)
        assert_ct_close(result, reference)

    def test_einsum_compile(self, dtype):
        """Verify torch.compile(fullgraph=True) works with complex einsum."""
        @torch.compile(fullgraph=True)
        def f(a, b):
            return torch.einsum("ij,jk->ik", a, b)

        re1, im1, n1 = rand_complex((3, 4), dtype=dtype)
        re2, im2, n2 = rand_complex((4, 5), dtype=dtype)
        result = f(ComplexTensor(re1, im1), ComplexTensor(re2, im2))
        assert_ct_close(result, torch.einsum("ij,jk->ik", n1, n2))


# ---------------------------------------------------------------------------
# New op: fft
# ---------------------------------------------------------------------------

@pytest.mark.new_op
class TestFFT:
    """irfft2 on ComplexTensor.

    irfft2 takes a complex half-spectrum [B, C, H, W//2+1] and returns a real
    spatial tensor [B, C, H, W]. This is the op at the heart of ComplexTucker.
    """

    def test_irfft2_matches_reference(self, dtype, device):
        B, C, H, W_half = 2, 4, 8, 5  # W=8, W_half=5
        re, im, native = rand_complex((B, C, H, W_half), dtype=dtype, device=device)
        ref = torch.fft.irfft2(native, norm="ortho")
        result = torch.fft.irfft2(ComplexTensor(re, im), norm="ortho")
        torch.testing.assert_close(result, ref)

    def test_irfft2_output_is_real(self, dtype, device):
        re, im, _ = rand_complex((2, 3, 8, 5), dtype=dtype, device=device)
        result = torch.fft.irfft2(ComplexTensor(re, im), norm="ortho")
        assert not result.is_complex()

    def test_irfft2_output_shape(self, dtype, device):
        B, C, H, W = 2, 4, 8, 8
        re, im, _ = rand_complex((B, C, H, W // 2 + 1), dtype=dtype, device=device)
        result = torch.fft.irfft2(ComplexTensor(re, im), norm="ortho")
        assert result.shape == (B, C, H, W)

    def test_irfft2_explicit_s(self, dtype, device):
        """irfft2 with explicit output size s=(H, W)."""
        B, C, H, W = 2, 3, 6, 10
        re, im, native = rand_complex((B, C, H, W // 2 + 1), dtype=dtype, device=device)
        ref = torch.fft.irfft2(native, s=(H, W), norm="ortho")
        result = torch.fft.irfft2(ComplexTensor(re, im), s=(H, W), norm="ortho")
        torch.testing.assert_close(result, ref)

    def test_irfft2_compile(self, dtype):
        """Verify torch.compile(fullgraph=True) works with irfft2."""
        @torch.compile(fullgraph=True)
        def f(ct):
            return torch.fft.irfft2(ct, norm="ortho")

        re, im, native = rand_complex((2, 4, 8, 5), dtype=dtype)
        ref = torch.fft.irfft2(native, norm="ortho")
        result = f(ComplexTensor(re, im))
        torch.testing.assert_close(result, ref)

    def test_rfft2_then_irfft2_roundtrip(self, dtype, device):
        """rfft2 followed by irfft2 should recover the original signal."""
        x = torch.randn(2, 4, 8, 8, dtype=dtype, device=device)
        spectrum = torch.fft.rfft2(x, norm="ortho")
        ct = ComplexTensor.from_interleaved(spectrum)
        recovered = torch.fft.irfft2(ct, norm="ortho")
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)
