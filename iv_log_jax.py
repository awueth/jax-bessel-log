"""JAX-native port of CUSF `iv_log` from `iv_log.cu.tpp`.

This module implements a piecewise approximation for `log(I_v(x))` using:
- a mu-order-20 asymptotic branch,
- a uniform asymptotic u_k-order-13 branch,
- a stabilized power-series branch.

The public entrypoint is `iv_log(v, x)`, which is JIT compiled.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

N_TERMS = 43
LOG_INV_SQRT_2PI = -0.91893853320467267
LOG_2 = 0.69314718055994529
LOG_4 = 1.3862943611198906


def _as_float_dtype(v: jax.Array, x: jax.Array) -> jnp.dtype:
    dtype = jnp.result_type(v, x)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(dtype, jnp.float32)
    return dtype


def _c(value: float, dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(value, dtype=dtype)


def _u1(t: jax.Array, t2: jax.Array) -> jax.Array:
    return t * (_c(0.125, t.dtype) - _c(0.2083333333333333, t.dtype) * t2)


def _u2(t2: jax.Array, t4: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t2 * (
        _c(0.0703125, dtype)
        + _c(-0.40104166666666663, dtype) * t2
        + _c(0.3342013888888889, dtype) * t4
    )


def _u3(t: jax.Array, t2: jax.Array) -> jax.Array:
    dtype = t.dtype
    return t * t2 * (
        _c(0.0732421875, dtype)
        + t2
        * (
            _c(-0.8912109375, dtype)
            + t2
            * (
                _c(1.8464626736111112, dtype)
                + t2 * _c(-1.0258125964506173, dtype)
            )
        )
    )


def _u4(t2: jax.Array, t4: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t4 * (
        _c(0.112152099609375, dtype)
        + t2
        * (
            _c(-2.3640869140625, dtype)
            + t2
            * (
                _c(8.78912353515625, dtype)
                + t2
                * (
                    _c(-11.207002616222994, dtype)
                    + t2 * _c(4.669584423426247, dtype)
                )
            )
        )
    )


def _u5(t: jax.Array, t2: jax.Array, t4: jax.Array) -> jax.Array:
    dtype = t.dtype
    return t4 * t * (
        _c(0.2271080017089844, dtype)
        + t2
        * (
            _c(-7.368794359479632, dtype)
            + t2
            * (
                _c(42.53499874538845, dtype)
                + t2
                * (
                    _c(-91.81824154324002, dtype)
                    + t2
                    * (
                        _c(84.63621767460073, dtype)
                        + t2 * _c(-28.21207255820024, dtype)
                    )
                )
            )
        )
    )


def _u6(t2: jax.Array, t4: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t2 * t4 * (
        _c(0.5725014209747314, dtype)
        + t2
        * (
            _c(-26.49143048695156, dtype)
            + t2
            * (
                _c(218.1905117442116, dtype)
                + t2
                * (
                    _c(-699.5796273761325, dtype)
                    + t2
                    * (
                        _c(1059.990452528000, dtype)
                        + t2
                        * (
                            _c(-765.2524681411816, dtype)
                            + t2 * _c(212.5701300392171, dtype)
                        )
                    )
                )
            )
        )
    )


def _u7(t: jax.Array, t2: jax.Array, t4: jax.Array) -> jax.Array:
    dtype = t.dtype
    return t * t2 * t4 * (
        _c(1.727727502584457, dtype)
        + t2
        * (
            _c(-108.0909197883947, dtype)
            + t2
            * (
                _c(1200.902913216352, dtype)
                + t2
                * (
                    _c(-5305.646978613403, dtype)
                    + t2
                    * (
                        _c(11655.39333686453, dtype)
                        + t2
                        * (
                            _c(-13586.55000643414, dtype)
                            + t2
                            * (
                                _c(8061.722181737309, dtype)
                                + t2 * _c(-1919.457662318407, dtype)
                            )
                        )
                    )
                )
            )
        )
    )


def _u8(t2: jax.Array, t8: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t8 * (
        _c(6.074042001273483, dtype)
        + t2
        * (
            _c(-493.9153047730880, dtype)
            + t2
            * (
                _c(7109.514302489364, dtype)
                + t2
                * (
                    _c(-41192.65496889755, dtype)
                    + t2
                    * (
                        _c(122200.4649830175, dtype)
                        + t2
                        * (
                            _c(-203400.1772804155, dtype)
                            + t2
                            * (
                                _c(192547.0012325315, dtype)
                                + t2
                                * (
                                    _c(-96980.59838863751, dtype)
                                    + t2 * _c(20204.29133096615, dtype)
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _u9(t: jax.Array, t2: jax.Array, t8: jax.Array) -> jax.Array:
    dtype = t.dtype
    return t * t8 * (
        _c(24.38052969955606, dtype)
        + t2
        * (
            _c(-2499.830481811210, dtype)
            + t2
            * (
                _c(45218.76898136273, dtype)
                + t2
                * (
                    _c(-331645.1724845636, dtype)
                    + t2
                    * (
                        _c(1.268365273321625e6, dtype)
                        + t2
                        * (
                            _c(-2.813563226586534e6, dtype)
                            + t2
                            * (
                                _c(3.763271297656404e6, dtype)
                                + t2
                                * (
                                    _c(-2.998015918538107e6, dtype)
                                    + t2
                                    * (
                                        _c(1.311763614662977e6, dtype)
                                        + t2 * _c(-242919.1879005513, dtype)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _u10(t2: jax.Array, t8: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t2 * t8 * (
        _c(110.0171402692467, dtype)
        + t2
        * (
            _c(-13886.08975371704, dtype)
            + t2
            * (
                _c(308186.4046126624, dtype)
                + t2
                * (
                    _c(-2.785618128086455e6, dtype)
                    + t2
                    * (
                        _c(1.328876716642182e7, dtype)
                        + t2
                        * (
                            _c(-3.756717666076335e7, dtype)
                            + t2
                            * (
                                _c(6.634451227472903e7, dtype)
                                + t2
                                * (
                                    _c(-7.410514821153266e7, dtype)
                                    + t2
                                    * (
                                        _c(5.095260249266464e7, dtype)
                                        + t2
                                        * (
                                            _c(-1.970681911843223e7, dtype)
                                            + t2 * _c(3.284469853072038e6, dtype)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _u11(t: jax.Array, t2: jax.Array, t8: jax.Array) -> jax.Array:
    dtype = t.dtype
    return t * t2 * t8 * (
        _c(551.3358961220206, dtype)
        + t2
        * (
            _c(-84005.43360302409, dtype)
            + t2
            * (
                _c(2.243768177922449e6, dtype)
                + t2
                * (
                    _c(-2.447406272573873e7, dtype)
                    + t2
                    * (
                        _c(1.420629077975331e8, dtype)
                        + t2
                        * (
                            _c(-4.958897842750303e8, dtype)
                            + t2
                            * (
                                _c(1.106842816823014e9, dtype)
                                + t2
                                * (
                                    _c(-1.621080552108337e9, dtype)
                                    + t2
                                    * (
                                        _c(1.553596899570580e9, dtype)
                                        + t2
                                        * (
                                            _c(-9.394623596815784e8, dtype)
                                            + t2
                                            * (
                                                _c(3.255730741857657e8, dtype)
                                                + t2
                                                * _c(-4.932925366450996e7, dtype)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _u12(t2: jax.Array, t4: jax.Array, t8: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t4 * t8 * (
        _c(3038.090510922384, dtype)
        + t2
        * (
            _c(-549842.3275722887, dtype)
            + t2
            * (
                _c(1.739510755397816e7, dtype)
                + t2
                * (
                    _c(-2.251056618894153e8, dtype)
                    + t2
                    * (
                        _c(1.559279864879258e9, dtype)
                        + t2
                        * (
                            _c(-6.563293792619284e9, dtype)
                            + t2
                            * (
                                _c(1.795421373115560e10, dtype)
                                + t2
                                * (
                                    _c(-3.302659974980072e10, dtype)
                                    + t2
                                    * (
                                        _c(4.128018557975397e10, dtype)
                                        + t2
                                        * (
                                            _c(-3.463204338815878e10, dtype)
                                            + t2
                                            * (
                                                _c(1.868820750929582e10, dtype)
                                                + t2
                                                * (
                                                    _c(-5.866481492051847e9, dtype)
                                                    + t2
                                                    * _c(
                                                        8.147890961183121e8,
                                                        dtype,
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _u13(t: jax.Array, t2: jax.Array, t4: jax.Array, t8: jax.Array) -> jax.Array:
    dtype = t.dtype
    return t * t4 * t8 * (
        _c(18257.75547429317, dtype)
        + t2
        * (
            _c(-3.871833442572613e6, dtype)
            + t2
            * (
                _c(1.431578767188890e8, dtype)
                + t2
                * (
                    _c(-2.167164983223795e9, dtype)
                    + t2
                    * (
                        _c(1.763473060683497e10, dtype)
                        + t2
                        * (
                            _c(-8.786707217802327e10, dtype)
                            + t2
                            * (
                                _c(2.879006499061506e11, dtype)
                                + t2
                                * (
                                    _c(-6.453648692453765e11, dtype)
                                    + t2
                                    * (
                                        _c(1.008158106865382e12, dtype)
                                        + t2
                                        * (
                                            _c(-1.098375156081223e12, dtype)
                                            + t2
                                            * (
                                                _c(8.192186695485773e11, dtype)
                                                + t2
                                                * (
                                                    _c(-3.990961752244665e11, dtype)
                                                    + t2
                                                    * (
                                                        _c(
                                                            1.144982377320258e11,
                                                            dtype,
                                                        )
                                                        + t2
                                                        * _c(
                                                            -1.467926124769562e10,
                                                            dtype,
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _branch_mu20(v: jax.Array, x: jax.Array, log_x: jax.Array) -> jax.Array:
    dtype = v.dtype
    mu = _c(4.0, dtype) * v * v
    one = _c(1.0, dtype)
    eight = _c(8.0, dtype)

    def body(i: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        curr_term, sum_terms = state
        k = _c(2 * i + 1, dtype)
        c = _c(i + 1, dtype)
        curr_term = curr_term * -(mu - k * k) / (c * eight * x)
        sum_terms = sum_terms + curr_term
        return curr_term, sum_terms

    _, sum_terms = lax.fori_loop(0, 20, body, (one, one))
    return (
        x
        + _c(LOG_INV_SQRT_2PI, dtype)
        - _c(0.5, dtype) * log_x
        + jnp.log(jnp.abs(sum_terms))
    )


def _branch_uk13(
    v: jax.Array,
    x: jax.Array,
    log_x: jax.Array,
    log_v: jax.Array,
) -> jax.Array:
    dtype = v.dtype
    one = _c(1.0, dtype)
    quarter = _c(0.25, dtype)
    half = _c(0.5, dtype)

    v2 = v * v
    v4 = v2 * v2
    v8 = v4 * v4

    x_prime_2 = (x * x) / v2
    sqrt_1_plus_x2 = jnp.sqrt(one + x_prime_2)
    t = one / sqrt_1_plus_x2
    t2 = one / (one + x_prime_2)
    t4 = t2 * t2
    t8 = t4 * t4

    poly = (
        one
        + _u1(t, t2) / v
        + _u2(t2, t4) / v2
        + _u3(t, t2) / (v * v2)
        + _u4(t2, t4) / v4
        + _u5(t, t2, t4) / (v * v4)
        + _u6(t2, t4) / (v2 * v4)
        + _u7(t, t2, t4) / (v * v2 * v4)
        + _u8(t2, t8) / v8
        + _u9(t, t2, t8) / (v * v8)
        + _u10(t2, t8) / (v2 * v8)
        + _u11(t, t2, t8) / (v * v2 * v8)
        + _u12(t2, t4, t8) / (v4 * v8)
        + _u13(t, t2, t4, t8) / (v * v4 * v8)
    )

    return (
        _c(LOG_INV_SQRT_2PI, dtype)
        - half * log_v
        + v * (sqrt_1_plus_x2 + log_x - log_v - jnp.log1p(sqrt_1_plus_x2))
        - quarter * jnp.log1p(x_prime_2)
        + jnp.log(jnp.abs(poly))
    )


def _branch_series(v: jax.Array, x: jax.Array, log_x: jax.Array) -> jax.Array:
    dtype = v.dtype
    one = _c(1.0, dtype)
    two = _c(2.0, dtype)
    four = _c(4.0, dtype)

    # Keep this for structural parity with the CUDA implementation.
    _ = two * log_x - _c(LOG_4, dtype)

    x2_over_4 = x * x / four
    inv_x2_over_4 = one / x2_over_4
    v_inv_x2_over_4 = v * inv_x2_over_4

    first_term = -gammaln(v + one)
    peak_k = jnp.floor((-v + jnp.sqrt(v * v + x * x)) / two).astype(jnp.int32)

    def max_cond(state: tuple[jnp.int32, jax.Array]) -> jax.Array:
        k, _ = state
        return k <= peak_k

    def max_body(state: tuple[jnp.int32, jax.Array]) -> tuple[jnp.int32, jax.Array]:
        k, max_term = state
        kf = k.astype(dtype)
        max_term = max_term - jnp.log(kf * (kf * inv_x2_over_4 + v_inv_x2_over_4))
        return k + jnp.int32(1), max_term

    _, max_term = lax.while_loop(max_cond, max_body, (jnp.int32(1), first_term))

    sum_terms0 = jnp.exp(first_term - max_term)

    def sum_body(k: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        term, sum_terms = state
        kf = _c(k, dtype)
        term = term - jnp.log(kf * (kf * inv_x2_over_4 + v_inv_x2_over_4))
        sum_terms = sum_terms + jnp.exp(term - max_term)
        return term, sum_terms

    _, sum_terms = lax.fori_loop(1, N_TERMS, sum_body, (first_term, sum_terms0))
    return v * (log_x - _c(LOG_2, dtype)) + max_term + jnp.log(sum_terms)


def _iv_log_scalar_positive(v: jax.Array, x: jax.Array) -> jax.Array:
    log_x = jnp.log(x)
    log_v = jnp.log(v)

    cond_mu = ((x > _c(30.0, v.dtype)) & (v < _c(15.3919, v.dtype))) | (
        ((_c(0.5113, v.dtype) * log_x + _c(0.7939, v.dtype)) > log_v)
        & (x > _c(59.6925, v.dtype))
    )
    cond_uk = ((x > _c(19.6931, v.dtype)) & (v > _c(0.7, v.dtype))) | (
        v > _c(12.6964, v.dtype)
    )

    return lax.cond(
        cond_mu,
        lambda _: _branch_mu20(v, x, log_x),
        lambda _: lax.cond(
            cond_uk,
            lambda __: _branch_uk13(v, x, log_x, log_v),
            lambda __: _branch_series(v, x, log_x),
            operand=None,
        ),
        operand=None,
    )


def _iv_log_scalar(v: jax.Array, x: jax.Array) -> jax.Array:
    dtype = v.dtype
    zero = _c(0.0, dtype)
    nan = _c(float("nan"), dtype)
    neg_inf = _c(float("-inf"), dtype)

    invalid = (x < zero) | (v < zero)

    def invalid_case(_: None) -> jax.Array:
        return nan

    def valid_case(_: None) -> jax.Array:
        return lax.cond(
            x == zero,
            lambda __: lax.cond(v == zero, lambda ___: zero, lambda ___: neg_inf, None),
            lambda __: _iv_log_scalar_positive(v, x),
            operand=None,
        )

    return lax.cond(invalid, invalid_case, valid_case, operand=None)


def _iv_log_eager(v: jax.Array, x: jax.Array) -> jax.Array:
    """Eager (non-jitted) implementation of log(I_v(x))."""
    dtype = _as_float_dtype(v, x)
    v_arr = jnp.asarray(v, dtype=dtype)
    x_arr = jnp.asarray(x, dtype=dtype)

    v_b, x_b = jnp.broadcast_arrays(v_arr, x_arr)
    v_flat = jnp.reshape(v_b, (-1,))
    x_flat = jnp.reshape(x_b, (-1,))
    out_flat = jax.vmap(_iv_log_scalar)(v_flat, x_flat)
    return jnp.reshape(out_flat, v_b.shape)


_iv_log_jit = jax.jit(_iv_log_eager)


def iv_log(v: jax.Array, x: jax.Array) -> jax.Array:
    """JIT-compiled log(I_v(x)) with broadcasting over `v` and `x`."""
    return _iv_log_jit(v, x)


__all__ = ["iv_log", "_iv_log_eager"]
