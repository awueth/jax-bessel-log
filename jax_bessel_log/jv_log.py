import jax
import jax.numpy as jnp
import scipy.special


def jv_log(v, z):
  v, z = jnp.asarray(v), jnp.asarray(z)

  # Require the order v to be integer type: this simplifies
  # the JVP rule below.
  assert jnp.issubdtype(v.dtype, jnp.integer)

  # Promote the input to inexact (float/complex).
  # Note that jnp.result_type() accounts for the enable_x64 flag.
  z = z.astype(jnp.result_type(float, z.dtype))

  # Wrap scipy function to return the expected dtype.
  _scipy_jv = lambda v, z: jnp.log(scipy.special.jv(v, z).astype(z.dtype))

  # Define the expected shape & dtype of output.
  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=jnp.broadcast_shapes(v.shape, z.shape),
      dtype=z.dtype)

  # Use vmap_method="broadcast_all" because scipy.special.jv handles broadcasted inputs.
  return jax.pure_callback(_scipy_jv, result_shape_dtype, v, z, vmap_method="broadcast_all")