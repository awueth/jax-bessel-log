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

from jax_bessel_log.u_polynomials import u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13

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
        + u1(t, t2) / v
        + u2(t2, t4) / v2
        + u3(t, t2) / (v * v2)
        + u4(t2, t4) / v4
        + u5(t, t2, t4) / (v * v4)
        + u6(t2, t4) / (v2 * v4)
        + u7(t, t2, t4) / (v * v2 * v4)
        + u8(t2, t8) / v8
        + u9(t, t2, t8) / (v * v8)
        + u10(t2, t8) / (v2 * v8)
        + u11(t, t2, t8) / (v * v2 * v8)
        + u12(t2, t4, t8) / (v4 * v8)
        + u13(t, t2, t4, t8) / (v * v4 * v8)
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
