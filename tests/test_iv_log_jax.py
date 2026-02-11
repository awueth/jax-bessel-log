from __future__ import annotations
import math
from typing import Final
import jax.numpy as jnp
import pytest
from flint import arb, ctx
from jax_bessel_log.iv_log import iv_log

RTOL: Final[float] = 1e-6
ATOL: Final[float] = 1e-4
ARB_PREC: Final[int] = 256

V_VALUES: Final[tuple[float, ...]] = (0.0, 0.1, 0.7, 1.0, 3.0, 10.0, 20.0, 80.0, 100.0, 200.0, 500.0, 1000.0)
X_VALUES: Final[tuple[float, ...]] = (1e-6, 1e-3, 0.1, 1.0, 5.0, 20.0, 60.0, 200.0)


def arb_log_iv(v: float, x: float, prec: int = ARB_PREC) -> arb:
    """Return Arb interval log(I_v(x))."""
    with ctx.workprec(prec):
        ref = arb(x).bessel_i(arb(v)).log()
        return ref


def assert_close_to_arb(
    got: float,
    ref: arb,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    mid = float(ref.mid())
    assert math.isclose(got, mid, rel_tol=rtol, abs_tol=atol), (
        f"got={got}, ref={ref}, mid={mid}, error={abs(got - mid)}"
    )


GRID_POINTS: Final[tuple[tuple[float, float], ...]] = tuple(
    (v, x) for v in V_VALUES for x in X_VALUES if v >= 0.0 and x > 0.0
)


@pytest.mark.parametrize(("v", "x"), GRID_POINTS)
def test_iv_log_matches_arb_on_thorough_grid(v: float, x: float) -> None:
    got = float(iv_log(v, x))
    ref = arb_log_iv(v, x)
    assert_close_to_arb(got, ref)


def test_iv_log_broadcast_matches_elementwise_arb() -> None:
    v = jnp.asarray([[0.0], [0.7], [3.0], [20.0], [80.0]], dtype=jnp.float32)
    x = jnp.asarray([[1e-6, 1.0, 20.0, 60.0, 200.0]], dtype=jnp.float32)
    out = iv_log(v, x)
    assert out.shape == (5, 5)

    out_np = out.tolist()
    for i, vv in enumerate(v[:, 0].tolist()):
        for j, xx in enumerate(x[0, :].tolist()):
            got = float(out_np[i][j])
            ref = arb_log_iv(float(vv), float(xx))
            assert_close_to_arb(got, ref)


def test_iv_log_zero_x_behavior() -> None:
    assert float(iv_log(0.0, 0.0)) == 0.0
    assert math.isinf(float(iv_log(0.1, 0.0)))
    assert float(iv_log(0.1, 0.0)) < 0.0
    assert math.isinf(float(iv_log(10.0, 0.0)))
    assert float(iv_log(10.0, 0.0)) < 0.0


@pytest.mark.parametrize(
    ("v", "x"),
    (
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (-0.1, 0.0),
    ),
)
def test_iv_log_invalid_negative_inputs_return_nan(v: float, x: float) -> None:
    assert math.isnan(float(iv_log(v, x)))


@pytest.mark.parametrize(
    ("v", "x"),
    (
        (0.0, 1e-6),
        (0.1, 0.1),
        (0.7, 1.0),
        (3.0, 5.0),
        (10.0, 20.0),
        (20.0, 60.0),
        (80.0, 200.0),
    ),
)
def test_iv_log_finite_for_positive_inputs(v: float, x: float) -> None:
    got = float(iv_log(v, x))
    assert math.isfinite(got)
