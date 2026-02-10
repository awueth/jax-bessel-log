"""u_k polynomial helpers for the uniform asymptotic iv_log branch.

Each helper keeps the original hard-coded prefactor and evaluates the inner
polynomial in ``t2`` via direct ``jnp.polyval(...)`` calls.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def u1(t: jax.Array, t2: jax.Array) -> jax.Array:
    return t * jnp.polyval(jnp.asarray((-0.2083333333333333, 0.125), dtype=t2.dtype), t2)


def u2(t2: jax.Array, t4: jax.Array) -> jax.Array:
    return t2*(0.0703125 - 0.40104166666666663*t2 + 0.3342013888888889*t4)


def u3(t: jax.Array, t2: jax.Array) -> jax.Array:
    return t * t2 * jnp.polyval(
        jnp.asarray(
            (-1.0258125964506173, 1.8464626736111112, -0.8912109375, 0.0732421875),
            dtype=t2.dtype,
        ),
        t2,
    )


def u4(t2: jax.Array, t4: jax.Array) -> jax.Array:
    return t4 * jnp.polyval(
        jnp.asarray(
            (
                4.669584423426247,
                -11.207002616222994,
                8.78912353515625,
                -2.3640869140625,
                0.112152099609375,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u5(t: jax.Array, t2: jax.Array, t4: jax.Array) -> jax.Array:
    return t * t4 * jnp.polyval(
        jnp.asarray(
            (
                -28.21207255820024,
                84.63621767460073,
                -91.81824154324002,
                42.53499874538845,
                -7.368794359479632,
                0.2271080017089844,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u6(t2: jax.Array, t4: jax.Array) -> jax.Array:
    return t2 * t4 * jnp.polyval(
        jnp.asarray(
            (
                212.5701300392171,
                -765.2524681411816,
                1059.990452528000,
                -699.5796273761325,
                218.1905117442116,
                -26.49143048695156,
                0.5725014209747314,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u7(t: jax.Array, t2: jax.Array, t4: jax.Array) -> jax.Array:
    return t * t2 * t4 * jnp.polyval(
        jnp.asarray(
            (
                -1919.457662318407,
                8061.722181737309,
                -13586.55000643414,
                11655.39333686453,
                -5305.646978613403,
                1200.902913216352,
                -108.0909197883947,
                1.727727502584457,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u8(t2: jax.Array, t8: jax.Array) -> jax.Array:
    return t8 * jnp.polyval(
        jnp.asarray(
            (
                20204.29133096615,
                -96980.59838863751,
                192547.0012325315,
                -203400.1772804155,
                122200.4649830175,
                -41192.65496889755,
                7109.514302489364,
                -493.9153047730880,
                6.074042001273483,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u9(t: jax.Array, t2: jax.Array, t8: jax.Array) -> jax.Array:
    return t * t8 * jnp.polyval(
        jnp.asarray(
            (
                -242919.1879005513,
                1.311763614662977e6,
                -2.998015918538107e6,
                3.763271297656404e6,
                -2.813563226586534e6,
                1.268365273321625e6,
                -331645.1724845636,
                45218.76898136273,
                -2499.830481811210,
                24.38052969955606,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u10(t2: jax.Array, t8: jax.Array) -> jax.Array:
    return t2 * t8 * jnp.polyval(
        jnp.asarray(
            (
                3.284469853072038e6,
                -1.970681911843223e7,
                5.095260249266464e7,
                -7.410514821153266e7,
                6.634451227472903e7,
                -3.756717666076335e7,
                1.328876716642182e7,
                -2.785618128086455e6,
                308186.4046126624,
                -13886.08975371704,
                110.0171402692467,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u11(t: jax.Array, t2: jax.Array, t8: jax.Array) -> jax.Array:
    return t * t2 * t8 * jnp.polyval(
        jnp.asarray(
            (
                -4.932925366450996e7,
                3.255730741857657e8,
                -9.394623596815784e8,
                1.553596899570580e9,
                -1.621080552108337e9,
                1.106842816823014e9,
                -4.958897842750303e8,
                1.420629077975331e8,
                -2.447406272573873e7,
                2.243768177922449e6,
                -84005.43360302409,
                551.3358961220206,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u12(t2: jax.Array, t4: jax.Array, t8: jax.Array) -> jax.Array:
    return t4 * t8 * jnp.polyval(
        jnp.asarray(
            (
                8.147890961183121e8,
                -5.866481492051847e9,
                1.868820750929582e10,
                -3.463204338815878e10,
                4.128018557975397e10,
                -3.302659974980072e10,
                1.795421373115560e10,
                -6.563293792619284e9,
                1.559279864879258e9,
                -2.251056618894153e8,
                1.739510755397816e7,
                -549842.3275722887,
                3038.090510922384,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


def u13(t: jax.Array, t2: jax.Array, t4: jax.Array, t8: jax.Array) -> jax.Array:
    return t * t4 * t8 * jnp.polyval(
        jnp.asarray(
            (
                -1.467926124769562e10,
                1.144982377320258e11,
                -3.990961752244665e11,
                8.192186695485773e11,
                -1.098375156081223e12,
                1.008158106865382e12,
                -6.453648692453765e11,
                2.879006499061506e11,
                -8.786707217802327e10,
                1.763473060683497e10,
                -2.167164983223795e9,
                1.431578767188890e8,
                -3.871833442572613e6,
                18257.75547429317,
            ),
            dtype=t2.dtype,
        ),
        t2,
    )


__all__ = [
    "u1",
    "u2",
    "u3",
    "u4",
    "u5",
    "u6",
    "u7",
    "u8",
    "u9",
    "u10",
    "u11",
    "u12",
    "u13",
]
