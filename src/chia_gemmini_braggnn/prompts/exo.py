# ruff: noqa: F821

from __future__ import annotations

from exo.libs.externs import expf, fmaxf, select

from exo import DRAM, proc


@proc
def conv2d(
    mbsz: size,
    ic: size,
    oc: size,
    ih: size,
    iw: size,
    oh: size,
    ow: size,
    kh: size,
    kw: size,
    x: f32[mbsz, ic, ih, iw] @ DRAM,
    weight: f32[oc, ic, kh, kw] @ DRAM,
    bias: f32[oc] @ DRAM,
    out: f32[mbsz, oc, oh, ow] @ DRAM,
):
    assert ih >= oh + kh - 1
    assert iw >= ow + kw - 1
    for n in seq(0, mbsz):
        for o in seq(0, oc):
            for r in seq(0, oh):
                for c in seq(0, ow):
                    out[n, o, r, c] = bias[o]
                    for i in seq(0, ic):
                        for y in seq(0, kh):
                            for k in seq(0, kw):
                                out[n, o, r, c] += (
                                    x[n, i, r + y, c + k] * weight[o, i, y, k]
                                )


@proc
def attention_scores(
    mbsz: size,
    ch: size,
    h: size,
    w: size,
    theta: f32[mbsz, ch, h, w] @ DRAM,
    phi: f32[mbsz, ch, h, w] @ DRAM,
    out: f32[mbsz, h, w, h, w] @ DRAM,
):
    for n in seq(0, mbsz):
        for r1 in seq(0, h):
            for c1 in seq(0, w):
                for r2 in seq(0, h):
                    for c2 in seq(0, w):
                        out[n, r1, c1, r2, c2] = 0.0
                        for i in seq(0, ch):
                            out[n, r1, c1, r2, c2] += (
                                theta[n, i, r1, c1] * phi[n, i, r2, c2]
                            )


@proc
def attention_softmax(
    mbsz: size, h: size, w: size, values: f32[mbsz, h, w, h, w] @ DRAM
):
    assert h >= 1
    assert w >= 1
    for n in seq(0, mbsz):
        for r1 in seq(0, h):
            for c1 in seq(0, w):
                maximum: f32 @ DRAM
                denominator: f32 @ DRAM
                maximum = values[n, r1, c1, 0, 0]
                for r2 in seq(0, h):
                    for c2 in seq(0, w):
                        maximum = fmaxf(maximum, values[n, r1, c1, r2, c2])
                denominator = 0.0
                for r2 in seq(0, h):
                    for c2 in seq(0, w):
                        values[n, r1, c1, r2, c2] = expf(
                            values[n, r1, c1, r2, c2] - maximum
                        )
                        denominator += values[n, r1, c1, r2, c2]
                for r2 in seq(0, h):
                    for c2 in seq(0, w):
                        values[n, r1, c1, r2, c2] = (
                            values[n, r1, c1, r2, c2] / denominator
                        )


@proc
def attention_apply(
    mbsz: size,
    ch: size,
    h: size,
    w: size,
    attention: f32[mbsz, h, w, h, w] @ DRAM,
    g: f32[mbsz, ch, h, w] @ DRAM,
    out: f32[mbsz, ch, h, w] @ DRAM,
):
    for n in seq(0, mbsz):
        for i in seq(0, ch):
            for r1 in seq(0, h):
                for c1 in seq(0, w):
                    out[n, i, r1, c1] = 0.0
                    for r2 in seq(0, h):
                        for c2 in seq(0, w):
                            out[n, i, r1, c1] += (
                                attention[n, r1, c1, r2, c2] * g[n, i, r2, c2]
                            )


@proc
def residual_add(
    mbsz: size,
    ch: size,
    h: size,
    w: size,
    values: f32[mbsz, ch, h, w] @ DRAM,
    addend: f32[mbsz, ch, h, w] @ DRAM,
):
    for n in seq(0, mbsz):
        for c in seq(0, ch):
            for r in seq(0, h):
                for k in seq(0, w):
                    values[n, c, r, k] += addend[n, c, r, k]


@proc
def leaky_relu_map(
    mbsz: size, ch: size, h: size, w: size, values: f32[mbsz, ch, h, w] @ DRAM
):
    for n in seq(0, mbsz):
        for c in seq(0, ch):
            for r in seq(0, h):
                for k in seq(0, w):
                    values[n, c, r, k] = select(
                        values[n, c, r, k],
                        0.0,
                        values[n, c, r, k] * 0.01,
                        values[n, c, r, k],
                    )


@proc
def leaky_relu_vec(mbsz: size, features: size, values: f32[mbsz, features] @ DRAM):
    for n in seq(0, mbsz):
        for i in seq(0, features):
            values[n, i] = select(values[n, i], 0.0, values[n, i] * 0.01, values[n, i])


@proc
def linear_from_map(
    mbsz: size,
    ch: size,
    h: size,
    w: size,
    out_features: size,
    x: f32[mbsz, ch, h, w] @ DRAM,
    weight: f32[out_features, ch, h, w] @ DRAM,
    bias: f32[out_features] @ DRAM,
    out: f32[mbsz, out_features] @ DRAM,
):
    for n in seq(0, mbsz):
        for o in seq(0, out_features):
            out[n, o] = bias[o]
            for c in seq(0, ch):
                for y in seq(0, h):
                    for k in seq(0, w):
                        out[n, o] += weight[o, c, y, k] * x[n, c, y, k]


@proc
def linear(
    mbsz: size,
    in_features: size,
    out_features: size,
    x: f32[mbsz, in_features] @ DRAM,
    weight: f32[out_features, in_features] @ DRAM,
    bias: f32[out_features] @ DRAM,
    out: f32[mbsz, out_features] @ DRAM,
):
    for n in seq(0, mbsz):
        for o in seq(0, out_features):
            out[n, o] = bias[o]
            for i in seq(0, in_features):
                out[n, o] += weight[o, i] * x[n, i]


@proc
def braggnn_forward(
    mbsz: size,
    sz: size,
    c0: size,
    ci: size,
    c1: size,
    c2: size,
    f0: size,
    f1: size,
    f2: size,
    f3: size,
    patches: f32[mbsz, 1, sz, sz] @ DRAM,
    cnn0_weight: f32[c0, 1, 3, 3] @ DRAM,
    cnn0_bias: f32[c0] @ DRAM,
    theta_weight: f32[ci, c0, 1, 1] @ DRAM,
    theta_bias: f32[ci] @ DRAM,
    phi_weight: f32[ci, c0, 1, 1] @ DRAM,
    phi_bias: f32[ci] @ DRAM,
    g_weight: f32[ci, c0, 1, 1] @ DRAM,
    g_bias: f32[ci] @ DRAM,
    out_cnn_weight: f32[c0, ci, 1, 1] @ DRAM,
    out_cnn_bias: f32[c0] @ DRAM,
    cnn2_weight: f32[c1, c0, 3, 3] @ DRAM,
    cnn2_bias: f32[c1] @ DRAM,
    cnn4_weight: f32[c2, c1, 3, 3] @ DRAM,
    cnn4_bias: f32[c2] @ DRAM,
    fc0_weight: f32[f0, c2, sz - 6, sz - 6] @ DRAM,
    fc0_bias: f32[f0] @ DRAM,
    fc2_weight: f32[f1, f0] @ DRAM,
    fc2_bias: f32[f1] @ DRAM,
    fc4_weight: f32[f2, f1] @ DRAM,
    fc4_bias: f32[f2] @ DRAM,
    fc6_weight: f32[f3, f2] @ DRAM,
    fc6_bias: f32[f3] @ DRAM,
    fc8_weight: f32[2, f3] @ DRAM,
    fc8_bias: f32[2] @ DRAM,
    out: f32[mbsz, 2] @ DRAM,
):
    assert sz >= 7

    block: f32[mbsz, c0, sz - 2, sz - 2] @ DRAM
    conv2d(
        mbsz,
        1,
        c0,
        sz,
        sz,
        sz - 2,
        sz - 2,
        3,
        3,
        patches,
        cnn0_weight,
        cnn0_bias,
        block,
    )

    theta: f32[mbsz, ci, sz - 2, sz - 2] @ DRAM
    phi: f32[mbsz, ci, sz - 2, sz - 2] @ DRAM
    g: f32[mbsz, ci, sz - 2, sz - 2] @ DRAM
    conv2d(
        mbsz,
        c0,
        ci,
        sz - 2,
        sz - 2,
        sz - 2,
        sz - 2,
        1,
        1,
        block,
        theta_weight,
        theta_bias,
        theta,
    )
    conv2d(
        mbsz,
        c0,
        ci,
        sz - 2,
        sz - 2,
        sz - 2,
        sz - 2,
        1,
        1,
        block,
        phi_weight,
        phi_bias,
        phi,
    )
    conv2d(
        mbsz, c0, ci, sz - 2, sz - 2, sz - 2, sz - 2, 1, 1, block, g_weight, g_bias, g
    )

    attention: f32[mbsz, sz - 2, sz - 2, sz - 2, sz - 2] @ DRAM
    attention_scores(mbsz, ci, sz - 2, sz - 2, theta, phi, attention)
    attention_softmax(mbsz, sz - 2, sz - 2, attention)

    theta_phi_g: f32[mbsz, ci, sz - 2, sz - 2] @ DRAM
    attention_apply(mbsz, ci, sz - 2, sz - 2, attention, g, theta_phi_g)

    nlb_out: f32[mbsz, c0, sz - 2, sz - 2] @ DRAM
    conv2d(
        mbsz,
        ci,
        c0,
        sz - 2,
        sz - 2,
        sz - 2,
        sz - 2,
        1,
        1,
        theta_phi_g,
        out_cnn_weight,
        out_cnn_bias,
        nlb_out,
    )
    residual_add(mbsz, c0, sz - 2, sz - 2, nlb_out, block)

    leaky_relu_map(mbsz, c0, sz - 2, sz - 2, nlb_out)

    conv2_out: f32[mbsz, c1, sz - 4, sz - 4] @ DRAM
    conv2d(
        mbsz,
        c0,
        c1,
        sz - 2,
        sz - 2,
        sz - 4,
        sz - 4,
        3,
        3,
        nlb_out,
        cnn2_weight,
        cnn2_bias,
        conv2_out,
    )
    leaky_relu_map(mbsz, c1, sz - 4, sz - 4, conv2_out)

    conv4_out: f32[mbsz, c2, sz - 6, sz - 6] @ DRAM
    conv2d(
        mbsz,
        c1,
        c2,
        sz - 4,
        sz - 4,
        sz - 6,
        sz - 6,
        3,
        3,
        conv2_out,
        cnn4_weight,
        cnn4_bias,
        conv4_out,
    )
    leaky_relu_map(mbsz, c2, sz - 6, sz - 6, conv4_out)

    dense0: f32[mbsz, f0] @ DRAM
    linear_from_map(
        mbsz, c2, sz - 6, sz - 6, f0, conv4_out, fc0_weight, fc0_bias, dense0
    )
    leaky_relu_vec(mbsz, f0, dense0)

    dense2: f32[mbsz, f1] @ DRAM
    linear(mbsz, f0, f1, dense0, fc2_weight, fc2_bias, dense2)
    leaky_relu_vec(mbsz, f1, dense2)

    dense4: f32[mbsz, f2] @ DRAM
    linear(mbsz, f1, f2, dense2, fc4_weight, fc4_bias, dense4)
    leaky_relu_vec(mbsz, f2, dense4)

    dense6: f32[mbsz, f3] @ DRAM
    linear(mbsz, f2, f3, dense4, fc6_weight, fc6_bias, dense6)
    leaky_relu_vec(mbsz, f3, dense6)

    linear(mbsz, f3, 2, dense6, fc8_weight, fc8_bias, out)
