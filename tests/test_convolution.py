import numpy as np
import pytest

from udft import dftn, fr2ir, hnorm, idftn, ir2fr, irdftn, rdftn

rng = np.random.default_rng(0)


def circ_conv(x, h, origin):
    """Direct circular convolution y[n] = sum_i h[i] x[(n - (i - origin)) mod N]."""
    y = np.zeros_like(x)
    for idx in np.ndindex(h.shape):
        shift = tuple(i - o for i, o in zip(idx, origin))
        y += h[idx] * np.roll(x, shift, axis=tuple(range(x.ndim)))
    return y


@pytest.mark.parametrize("grid", [(32, 24), (32, 25), (31, 25)])
@pytest.mark.parametrize(
    "hshape, origin", [((3, 3), None), ((5, 3), None), ((4, 4), None), ((4, 4), (1, 2))]
)
class TestConvolution:
    def test_real(self, grid, hshape, origin):
        x = rng.standard_normal(grid)
        h = rng.standard_normal(hshape)
        expected = circ_conv(x, h, origin or [s // 2 for s in hshape])
        result = irdftn(ir2fr(h, grid, origin=origin) * rdftn(x), grid)
        assert np.allclose(result, expected)

    def test_complex(self, grid, hshape, origin):
        x = rng.standard_normal(grid)
        h = rng.standard_normal(hshape)
        expected = circ_conv(x, h, origin or [s // 2 for s in hshape])
        result = idftn(ir2fr(h, grid, origin=origin, onesided=False) * dftn(x))
        assert np.allclose(result.real, expected)


class TestIR2FR:
    def test_dirac(self):
        h = np.zeros((3, 3))
        h[1, 1] = 1
        assert np.allclose(ir2fr(h, (8, 8)), 1)

    def test_null_frequency(self):
        # Without normalization, Λ₀ is the sum of the impulse response
        h = rng.standard_normal((3, 3))
        assert np.isclose(ir2fr(h, (8, 8))[0, 0], np.sum(h))

    def test_shape(self):
        h = rng.standard_normal((3, 3))
        assert ir2fr(h, (8, 9)).shape == (8, 5)
        assert ir2fr(h, (8, 9), onesided=False).shape == (8, 9)

    def test_scipy(self):
        ndimage = pytest.importorskip("scipy.ndimage")
        x = rng.standard_normal((32, 24))
        h = rng.standard_normal((3, 3))
        result = irdftn(ir2fr(h, x.shape) * rdftn(x), x.shape)
        assert np.allclose(result, ndimage.convolve(x, h, mode="wrap"))

    def test_batch(self):
        x = rng.standard_normal((32, 24))
        hb = rng.standard_normal((4, 3, 3))
        frb = ir2fr(hb, x.shape)
        assert frb.shape == (4, 32, 13)
        result = irdftn(frb * rdftn(x), x.shape)
        for k in range(4):
            assert np.allclose(result[k], circ_conv(x, hb[k], (1, 1)))

    def test_complex_ir(self):
        h = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        with pytest.raises(ValueError):
            ir2fr(h, (8, 8))
        result = fr2ir(ir2fr(h, (8, 8), onesided=False), (3, 3), onesided=False)
        assert np.allclose(result, h)


@pytest.mark.parametrize("grid", [(32, 24), (32, 25), (31, 25)])
@pytest.mark.parametrize("hshape", [(3, 3), (5, 4)])
class TestFR2IR:
    def test_real(self, grid, hshape):
        h = rng.standard_normal(hshape)
        result = fr2ir(ir2fr(h, grid), hshape, inshape=grid)
        assert np.allclose(result, h)

    def test_complex(self, grid, hshape):
        h = rng.standard_normal(hshape)
        result = fr2ir(ir2fr(h, grid, onesided=False), hshape, onesided=False)
        assert np.allclose(result.real, h)

    def test_origin(self, grid, hshape):
        h = rng.standard_normal(hshape)
        origin = (1, 1)
        result = fr2ir(
            ir2fr(h, grid, origin=origin), hshape, origin=origin, inshape=grid
        )
        assert np.allclose(result, h)

    def test_batch(self, grid, hshape):
        hb = rng.standard_normal((4,) + hshape)
        result = fr2ir(ir2fr(hb, grid), hshape, inshape=grid)
        assert result.shape == hb.shape
        assert np.allclose(result, hb)


def test_fr2ir_even_grid_without_inshape():
    h = rng.standard_normal((3, 3))
    assert np.allclose(fr2ir(ir2fr(h, (32, 24)), (3, 3)), h)


def test_fr2ir_inshape_length():
    fr = ir2fr(rng.standard_normal((3, 3)), (8, 8))
    with pytest.raises(ValueError):
        fr2ir(fr, (3, 3), inshape=(8,))


def test_fr2ir_shape_exceeds_grid():
    fr = ir2fr(rng.standard_normal((3, 3)), (8, 6))
    with pytest.raises(ValueError):
        fr2ir(fr, (10, 10), inshape=(8, 6))


@pytest.mark.parametrize("shape", [(32, 24), (32, 25)])
def test_hnorm_batch(shape):
    x = rng.standard_normal((4,) + shape)
    expected = np.linalg.norm(x.reshape(4, -1), axis=-1)
    assert np.allclose(hnorm(rdftn(x, 2), shape), expected)


@pytest.mark.parametrize("shape", [(32, 24), (32, 25)])
def test_hnorm(shape):
    x = rng.standard_normal(shape)
    assert np.isclose(hnorm(rdftn(x), shape), np.linalg.norm(x))


@pytest.mark.parametrize(
    "func",
    [
        lambda: ir2fr(np.ones((3, 3)), ()),
        lambda: ir2fr(np.ones((3, 3)), (), origin=()),
        lambda: fr2ir(ir2fr(np.ones((3, 3)), (8, 6)), ()),
        lambda: hnorm(rdftn(np.ones((8, 6))), ()),
        lambda: ir2fr(np.ones((3, 3)), (8, 8, 8)),
        lambda: hnorm(rdftn(np.ones((8, 6))), (2, 8, 6)),
    ],
)
def test_shape_length(func):
    with pytest.raises(ValueError):
        func()
