import pytest
import torch

FLOAT_DTYPES = [torch.float32, torch.float64]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=FLOAT_DTYPES, ids=["f32", "f64"])
def dtype(request):
    return request.param


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def rand_complex(shape, dtype=torch.float32, device="cpu", requires_grad=False):
    """Return (re, im, native_complex) matching the same random values."""
    re = torch.randn(shape, dtype=dtype, device=device)
    im = torch.randn(shape, dtype=dtype, device=device)
    if requires_grad:
        re = re.requires_grad_(True)
        im = im.requires_grad_(True)
    native = torch.complex(re.detach(), im.detach())
    return re, im, native
