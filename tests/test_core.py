"""Tests for ComplexTensor subclass mechanics (construction, layout, autograd, compile).

These tests cover the core subclass behavior independent of any specific op.
They should pass as soon as _core.py is filled in from nightly.
"""

import pytest
import torch
import torch.testing

from complex_tensor import ComplexTensor
from conftest import rand_complex


class TestConstruction:
    def test_from_re_im(self, dtype, device):
        re = torch.randn(3, 4, dtype=dtype, device=device)
        im = torch.randn(3, 4, dtype=dtype, device=device)
        ct = ComplexTensor(re, im)
        assert ct.shape == re.shape
        assert ct.device.type == device
        assert ct.re is re
        assert ct.im is im

    def test_dtype_promoted_to_complex(self, dtype, device):
        re = torch.randn(3, dtype=dtype, device=device)
        im = torch.randn(3, dtype=dtype, device=device)
        ct = ComplexTensor(re, im)
        expected = torch.complex64 if dtype == torch.float32 else torch.complex128
        assert ct.dtype == expected

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            ComplexTensor(torch.randn(3), torch.randn(4))

    def test_dtype_mismatch_raises(self):
        with pytest.raises(AssertionError):
            ComplexTensor(torch.randn(3, dtype=torch.float32), torch.randn(3, dtype=torch.float64))


class TestInterleaved:
    def test_from_interleaved_round_trip(self, dtype, device):
        native = torch.randn(3, 4, dtype=dtype, device=device) + \
                 1j * torch.randn(3, 4, dtype=dtype, device=device)
        ct = ComplexTensor.from_interleaved(native)
        torch.testing.assert_close(ct.re, native.real)
        torch.testing.assert_close(ct.im, native.imag)

    def test_as_interleaved_round_trip(self, dtype, device):
        re, im, native = rand_complex((3, 4), dtype=dtype, device=device)
        ct = ComplexTensor(re, im)
        recovered = ct.as_interleaved()
        torch.testing.assert_close(recovered.real, re)
        torch.testing.assert_close(recovered.imag, im)

    def test_from_interleaved_rejects_real(self):
        with pytest.raises(AssertionError):
            ComplexTensor.from_interleaved(torch.randn(3))


class TestTensorFlatten:
    """__tensor_flatten__ / __tensor_unflatten__ are required for torch.compile."""

    def test_flatten_keys(self, dtype):
        re = torch.randn(3, dtype=dtype)
        im = torch.randn(3, dtype=dtype)
        ct = ComplexTensor(re, im)
        keys, meta = ct.__tensor_flatten__()
        assert "re" in keys
        assert "im" in keys

    def test_unflatten_round_trip(self, dtype):
        re = torch.randn(3, dtype=dtype)
        im = torch.randn(3, dtype=dtype)
        ct = ComplexTensor(re, im)
        keys, meta = ct.__tensor_flatten__()
        inner = {k: getattr(ct, k) for k in keys}
        restored = ComplexTensor.__tensor_unflatten__(inner, meta, ct.shape, ct.stride())
        torch.testing.assert_close(restored.re, re)
        torch.testing.assert_close(restored.im, im)


class TestAutograd:
    def test_grad_flows_to_re_and_im(self):
        re = torch.randn(4, requires_grad=True)
        im = torch.randn(4, requires_grad=True)
        ct = ComplexTensor(re, im)
        # Use as_interleaved so we can do a simple op with a real output
        loss = ct.as_interleaved().abs().sum()
        loss.backward()
        assert re.grad is not None
        assert im.grad is not None


class TestCompile:
    def test_compile_construction(self, dtype):
        """torch.compile should be able to trace through ComplexTensor construction."""
        re = torch.randn(4, dtype=dtype)
        im = torch.randn(4, dtype=dtype)

        @torch.compile(fullgraph=True)
        def f(re, im):
            ct = ComplexTensor(re, im)
            return ct.re + ct.im  # simple op that doesn't need dispatch

        result = f(re, im)
        torch.testing.assert_close(result, re + im)

    def test_compile_round_trip(self, dtype):
        """as_interleaved should be traceable."""
        re = torch.randn(4, dtype=dtype)
        im = torch.randn(4, dtype=dtype)
        native = torch.complex(re, im)

        @torch.compile(fullgraph=True)
        def f(re, im):
            ct = ComplexTensor(re, im)
            return ct.as_interleaved()

        result = f(re, im)
        torch.testing.assert_close(result, native)
