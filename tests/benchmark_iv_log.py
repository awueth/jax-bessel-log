from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jax_bessel_log.iv_log import iv_log


def log_iv_callback(v: jax.Array, z: jax.Array) -> jax.Array:
    v, z = jnp.asarray(v), jnp.asarray(z)
    assert jnp.issubdtype(v.dtype, jnp.integer)
    z = z.astype(jnp.result_type(float, z.dtype))

    def _scipy_iv(v_host, z_host):
        with np.errstate(divide="ignore"):
            return np.log(scipy.special.ive(v_host, z_host).astype(z_host.dtype)) + z_host

    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(v.shape, z.shape),
        dtype=z.dtype)
    return jax.pure_callback(_scipy_iv, result_shape_dtype, v, z, vmap_method="broadcast_all")

log_iv_callback_jit = jax.jit(log_iv_callback)


def _time_ms(fn, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args).block_until_ready()
    t1 = time.perf_counter()
    return ((t1 - t0) / iters) * 1e3


def main() -> None:
    sizes = [1, 128, 4096, 65536]
    dtype = jnp.float32
    warmup, iters = 3, 20
    key = jax.random.PRNGKey(0)

    header = f"{'size':>10} | {'jax (ms)':>12} | {'callback (ms)':>14} | {'callback/jax':>12} | {'max |diff|':>12} | {'max |rel|':>12}"
    print(header)
    print("-" * len(header))

    for size in sizes:
        key, key_v, key_x = jax.random.split(key, 3)
        v_int = jax.random.randint(key_v, (size,), minval=1, maxval=40, dtype=jnp.int32)
        x = jax.random.uniform(key_x, (size,), minval=0.1, maxval=80.0, dtype=dtype)

        jax_out = iv_log(v_int.astype(dtype), x).block_until_ready()
        callback_out = log_iv_callback_jit(v_int, x).block_until_ready()

        finite_mask = jnp.isfinite(jax_out) & jnp.isfinite(callback_out)
        if not finite_mask.any():
            raise RuntimeError(f"No finite samples for size={size}")

        v_int_f, x_f = v_int[finite_mask], x[finite_mask]

        jax_ms = _time_ms(iv_log, v_int_f.astype(dtype), x_f, warmup=warmup, iters=iters)
        callback_ms = _time_ms(log_iv_callback_jit, v_int_f, x_f, warmup=warmup, iters=iters)

        jax_out_f = iv_log(v_int_f.astype(dtype), x_f)
        callback_out_f = log_iv_callback_jit(v_int_f, x_f)
        abs_diff = jnp.abs(jax_out_f - callback_out_f)
        max_diff = float(jnp.max(abs_diff))
        max_rel = float(jnp.max(abs_diff / jnp.abs(callback_out_f)))
        speedup = callback_ms / jax_ms if jax_ms > 0 else float("inf")

        print(f"{int(v_int_f.shape[0]):10d} | {jax_ms:12.4f} | {callback_ms:14.4f} | {speedup:12.2f} | {max_diff:12.4e} | {max_rel:12.4e}")


if __name__ == "__main__":
    main()
