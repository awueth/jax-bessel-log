from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from flint import arb, ctx

from jax_bessel_log.iv_log import iv_log


def log_iv_callback(v: jax.Array, z: jax.Array) -> jax.Array:
    v, z = jnp.asarray(v), jnp.asarray(z)
    z = z.astype(jnp.result_type(float, z.dtype))

    def _scipy_iv(v_host, z_host):
        with np.errstate(divide="ignore"):
            return np.log(scipy.special.ive(v_host, z_host).astype(z_host.dtype)) + z_host

    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(jnp.shape(v), z.shape),
        dtype=z.dtype)
    return jax.pure_callback(_scipy_iv, result_shape_dtype, v, z, vmap_method="broadcast_all")


log_iv_callback_jit = jax.jit(log_iv_callback)


ARB_PREC = 256


def arb_log_iv(v: float, x: float, prec: int = ARB_PREC) -> arb:
    """Return Arb interval log(I_v(x))."""
    with ctx.workprec(prec):
        ref = arb(x).bessel_i(arb(v)).log()
        return ref


def _time_ms(fn, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args).block_until_ready()
    t1 = time.perf_counter()
    return ((t1 - t0) / iters) * 1e3


def main() -> None:
    v = 100
    n_points = 1000
    x_min, x_max = 1.0, 200.0
    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64
    warmup, iters = 3, 20

    # Generate x values
    x = jnp.linspace(x_min, x_max, n_points, dtype=dtype)

    # Compute jax result
    jax_out = iv_log(v, x).block_until_ready()

    # Compute scipy callback result
    scipy_out = log_iv_callback_jit(v, x).block_until_ready()

    # Compute arb reference (vectorized)
    arb_refs = [arb_log_iv(v, float(xi)) for xi in x]
    arb_out = np.array([float(ref.mid()) for ref in arb_refs], dtype=dtype)

    # Time implementations
    jax_ms = _time_ms(iv_log, v, x, warmup=warmup, iters=iters)
    scipy_ms = _time_ms(log_iv_callback_jit, v, x, warmup=warmup, iters=iters)

    # Compute accuracy metrics for jax
    jax_np = np.asarray(jax_out)
    jax_abs_diff = np.abs(jax_np - arb_out)
    jax_max_abs_diff = float(np.max(jax_abs_diff))
    jax_max_rel_diff = float(np.max(jax_abs_diff / np.abs(arb_out)))

    # Compute accuracy metrics for scipy
    scipy_np = np.asarray(scipy_out)
    scipy_abs_diff = np.abs(scipy_np - arb_out)
    scipy_max_abs_diff = float(np.max(scipy_abs_diff))
    scipy_max_rel_diff = float(np.max(scipy_abs_diff / np.abs(arb_out)))

    # Print results
    print(f"Benchmark: iv_log single order")
    print(f"  order v:    {v}")
    print(f"  n_points:   {n_points}")
    print(f"  x range:    [{x_min}, {x_max}]")
    print(f"  dtype:      {dtype}")
    print(f"  arb prec:   {ARB_PREC} bits")
    print()
    speedup = scipy_ms / jax_ms if jax_ms > 0 else float("inf")
    print(f"Timing (warmup={warmup}, iters={iters}):")
    print(f"  jax:        {jax_ms:.4f} ms")
    print(f"  scipy:      {scipy_ms:.4f} ms")
    print(f"  speedup:    {speedup:.2f}x")
    print()
    print(f"Accuracy vs arb:")
    print(f"  jax   max |diff|: {jax_max_abs_diff:.4e}  max |rel|: {jax_max_rel_diff:.4e}")
    print(f"  scipy max |diff|: {scipy_max_abs_diff:.4e}  max |rel|: {scipy_max_rel_diff:.4e}")


if __name__ == "__main__":
    main()
