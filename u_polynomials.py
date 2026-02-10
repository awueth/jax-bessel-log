"""u_k polynomial helpers for the uniform asymptotic iv_log branch.

These are a direct port of the GPU u-polynomial formulas used in CUSF.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _c(value: float, dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(value, dtype=dtype)


def u1(t: jax.Array, t2: jax.Array) -> jax.Array:
    return t * (_c(0.125, t.dtype) - _c(0.2083333333333333, t.dtype) * t2)


def u2(t2: jax.Array, t4: jax.Array) -> jax.Array:
    dtype = t2.dtype
    return t2 * (
        _c(0.0703125, dtype)
        + _c(-0.40104166666666663, dtype) * t2
        + _c(0.3342013888888889, dtype) * t4
    )


def u3(t: jax.Array, t2: jax.Array) -> jax.Array:
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


def u4(t2: jax.Array, t4: jax.Array) -> jax.Array:
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


def u5(t: jax.Array, t2: jax.Array, t4: jax.Array) -> jax.Array:
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


def u6(t2: jax.Array, t4: jax.Array) -> jax.Array:
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


def u7(t: jax.Array, t2: jax.Array, t4: jax.Array) -> jax.Array:
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


def u8(t2: jax.Array, t8: jax.Array) -> jax.Array:
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


def u9(t: jax.Array, t2: jax.Array, t8: jax.Array) -> jax.Array:
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


def u10(t2: jax.Array, t8: jax.Array) -> jax.Array:
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


def u11(t: jax.Array, t2: jax.Array, t8: jax.Array) -> jax.Array:
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


def u12(t2: jax.Array, t4: jax.Array, t8: jax.Array) -> jax.Array:
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


def u13(t: jax.Array, t2: jax.Array, t4: jax.Array, t8: jax.Array) -> jax.Array:
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
