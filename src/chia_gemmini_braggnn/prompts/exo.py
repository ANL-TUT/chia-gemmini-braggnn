# ruff: noqa: F821, F823


from __future__ import annotations

from exo.frontend.syntax import f32, size
from exo.libs.externs import fmaxf, select

from exo import DRAM, proc

__all__ = ["braggnn_inference_exo_impl"]


def _fail_dummy() -> None:
    raise SyntaxError("Exo Python dummy functions should never be called directly.")


def seq(lo, hi):
    """Sequential range helper for Exo loop bounds."""
    _fail_dummy()


class i8(int):
    """Dummy type symbol used by Exo kernels for 8-bit integer tensors."""

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


class i32(int):
    """Dummy type symbol used by Exo kernels for 32-bit integer tensors."""

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


@proc
def leaky_relu_requant(
    d0: size,
    d1: size,
    d2: size,
    values: i8[d0, d1, d2] @ DRAM,
    scale: f32 @ DRAM,
):
    for i in seq(0, d0):
        for j in seq(0, d1):
            for k in seq(0, d2):
                x: f32
                val: f32
                x = values[i, j, k]
                val = select(0.0, x, x * scale, x * 0.01 * scale)
                val = val + select(val, 0.0, -0.5, 0.5)
                val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                values[i, j, k] = val


@proc
def cpu_conv2d(
    in_h: size,
    in_w: size,
    in_ch: size,
    out_ch: size,
    kernel_size: size,
    out_h: size,
    out_w: size,
    input: i8[in_h, in_w, in_ch] @ DRAM,
    weights: i8[out_ch, kernel_size, kernel_size, in_ch] @ DRAM,
    bias: i32[out_ch] @ DRAM,
    output: i8[out_h, out_w, out_ch] @ DRAM,
    acc_scale: f32 @ DRAM,
):
    assert in_h >= out_h + kernel_size - 1
    assert in_w >= out_w + kernel_size - 1

    for oh in seq(0, out_h):
        for ow in seq(0, out_w):
            for oc in seq(0, out_ch):
                sum: i32
                sum = bias[oc]

                for kh in seq(0, kernel_size):
                    for kw in seq(0, kernel_size):
                        for ic in seq(0, in_ch):
                            x: i32
                            w: i32
                            x = input[oh + kh, ow + kw, ic]
                            w = weights[oc, kh, kw, ic]
                            sum += x * w

                val: f32
                val = sum
                val = val * acc_scale
                val = val + select(val, 0.0, -0.5, 0.5)
                val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                output[oh, ow, oc] = val


@proc
def cpu_fc(
    d0: size,
    d1: size,
    d2: size,
    out_features: size,
    input: i8[d0, d1, d2] @ DRAM,
    weights: i8[out_features, d0, d1, d2] @ DRAM,
    bias: i32[out_features] @ DRAM,
    output: i8[out_features, 1, 1] @ DRAM,
    acc_scale: f32 @ DRAM,
):
    # in_features = d0 * d1 * d2
    for j in seq(0, out_features):
        sum: i32
        sum = bias[j]
        for k0 in seq(0, d0):
            for k1 in seq(0, d1):
                for k2 in seq(0, d2):
                    x: i32
                    w: i32
                    x = input[k0, k1, k2]
                    w = weights[j, k0, k1, k2]
                    sum += x * w
        val: f32
        val = sum
        val = val * acc_scale
        val = val + select(val, 0.0, -0.5, 0.5)
        val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
        output[j, 0, 0] = val


@proc
def cpu_matmul_transA(
    dim: size,
    K: size,
    A: i8[K, dim, dim] @ DRAM,
    B: i8[K, dim, dim] @ DRAM,
    C: i8[dim, dim, dim, dim] @ DRAM,
    acc_scale: f32 @ DRAM,
):
    # M = N = dim * dim
    for i1 in seq(0, dim):
        for i2 in seq(0, dim):
            for j1 in seq(0, dim):
                for j2 in seq(0, dim):
                    sum: i32
                    sum = 0.0
                    for k in seq(0, K):
                        a: i32
                        b: i32
                        a = A[k, i1, i2]
                        b = B[k, j1, j2]
                        sum += a * b
                    val: f32
                    val = sum
                    val = val * acc_scale
                    val = val + select(val, 0.0, -0.5, 0.5)
                    val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                    C[i1, i2, j1, j2] = val


@proc
def cpu_matmul_transB(
    dim: size,
    N: size,
    A: i8[dim, dim, dim, dim] @ DRAM,
    B: i8[N, dim, dim] @ DRAM,
    C: i8[dim, dim, N] @ DRAM,
    acc_scale: f32 @ DRAM,
):
    # M = K = dim * dim
    for i1 in seq(0, dim):
        for i2 in seq(0, dim):
            for j in seq(0, N):
                sum: i32
                sum = 0.0
                for k1 in seq(0, dim):
                    for k2 in seq(0, dim):
                        a: i32
                        b: i32
                        a = A[i1, i2, k1, k2]
                        b = B[j, k1, k2]
                        sum += a * b
                val: f32
                val = sum
                val = val * acc_scale
                val = val + select(val, 0.0, -0.5, 0.5)
                val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                C[i1, i2, j] = val


@proc
def cpu_resadd(
    dim: size,
    cols: size,
    A_scale: f32 @ DRAM,
    B_scale: f32 @ DRAM,
    A: i8[dim, dim, cols] @ DRAM,
    B: i8[dim, dim, cols] @ DRAM,
    C: i8[dim, dim, cols] @ DRAM,
):
    # rows = dim * dim
    for i1 in seq(0, dim):
        for i2 in seq(0, dim):
            for j in seq(0, cols):
                a: f32
                b: f32
                val: f32
                a = A[i1, i2, j]
                b = B[i1, i2, j]
                val = a * A_scale + b * B_scale
                val = val + select(val, 0.0, -0.5, 0.5)
                val = fmaxf(-128.0, select(val, 127.0, val, 127.0))
                C[i1, i2, j] = val


@proc
def cpu_nlb(
    dim: size,
    c0: size,
    ci: size,
    input: i8[dim, dim, c0] @ DRAM,
    nlb_theta_weights: i8[ci, 1, 1, c0] @ DRAM,
    nlb_theta_bias: i32[ci] @ DRAM,
    nlb_theta_scale: f32 @ DRAM,
    nlb_phi_weights: i8[ci, 1, 1, c0] @ DRAM,
    nlb_phi_bias: i32[ci] @ DRAM,
    nlb_phi_scale: f32 @ DRAM,
    nlb_g_weights: i8[ci, 1, 1, c0] @ DRAM,
    nlb_g_bias: i32[ci] @ DRAM,
    nlb_g_scale: f32 @ DRAM,
    nlb_matmul_scale: f32 @ DRAM,
    softmax_input_scale: f32 @ DRAM,
    softmax_output_scale: f32 @ DRAM,
    nlb_matmul_1_scale: f32 @ DRAM,
    nlb_out_weights: i8[c0, 1, 1, ci] @ DRAM,
    nlb_out_bias: i32[c0] @ DRAM,
    nlb_out_scale: f32 @ DRAM,
    nlb_add_a_scale: f32 @ DRAM,
    nlb_add_b_scale: f32 @ DRAM,
    output: i8[dim, dim, c0] @ DRAM,
):
    assert dim >= 1

    # --- Theta 1x1 conv: 9x9x64 -> 9x9x32 ---
    theta_out: i8[dim, dim, ci] @ DRAM
    cpu_conv2d(
        dim,
        dim,
        c0,
        ci,
        1,
        dim,
        dim,
        input,
        nlb_theta_weights,
        nlb_theta_bias,
        theta_out,
        nlb_theta_scale,
    )

    # Reshape NHWC [9x9x32] -> [32][81]
    theta_reshaped: i8[ci, dim, dim] @ DRAM
    for c in seq(0, ci):
        for h in seq(0, dim):
            for w in seq(0, dim):
                theta_reshaped[c, h, w] = theta_out[h, w, c]

    # --- Phi 1x1 conv: 9x9x64 -> 9x9x32 ---
    phi_out: i8[dim, dim, ci] @ DRAM
    cpu_conv2d(
        dim,
        dim,
        c0,
        ci,
        1,
        dim,
        dim,
        input,
        nlb_phi_weights,
        nlb_phi_bias,
        phi_out,
        nlb_phi_scale,
    )

    phi_reshaped: i8[ci, dim, dim] @ DRAM
    for c in seq(0, ci):
        for h in seq(0, dim):
            for w in seq(0, dim):
                phi_reshaped[c, h, w] = phi_out[h, w, c]

    # --- G 1x1 conv: 9x9x64 -> 9x9x32 ---
    g_out: i8[dim, dim, ci] @ DRAM
    cpu_conv2d(
        dim,
        dim,
        c0,
        ci,
        1,
        dim,
        dim,
        input,
        nlb_g_weights,
        nlb_g_bias,
        g_out,
        nlb_g_scale,
    )

    g_reshaped: i8[ci, dim, dim] @ DRAM
    for c in seq(0, ci):
        for h in seq(0, dim):
            for w in seq(0, dim):
                g_reshaped[c, h, w] = g_out[h, w, c]

    # --- Attention: theta^T @ phi -> [81][81] ---
    # theta_reshaped[32][81], phi_reshaped[32][81]
    # C[81][81] = theta^T[81][32] @ phi[32][81]
    attention: i8[dim, dim, dim, dim] @ DRAM
    cpu_matmul_transA(
        dim, ci, theta_reshaped, phi_reshaped, attention, nlb_matmul_scale
    )

    # --- Softmax (per row, matching Gemmini Taylor approximation) ---
    for i1 in seq(0, dim):
        for i2 in seq(0, dim):
            row_float: f32[dim, dim] @ DRAM
            max_val: f32 @ DRAM
            max_val = -1000000000.0
            for j1 in seq(0, dim):
                for j2 in seq(0, dim):
                    row_float[j1, j2] = attention[i1, i2, j1, j2]
                    row_float[j1, j2] = row_float[j1, j2] * softmax_input_scale
                    max_val = fmaxf(max_val, row_float[j1, j2])

            sum_exp: f32 @ DRAM
            sum_exp = 0.0
            for j1 in seq(0, dim):
                for j2 in seq(0, dim):
                    x: f32
                    x2: f32
                    x3: f32
                    x4: f32
                    exp_val: f32
                    x = row_float[j1, j2] - max_val
                    x2 = x * x
                    x3 = x2 * x
                    x4 = x2 * x2
                    exp_val = 1.0 + x
                    exp_val += x2 * 0.5
                    exp_val += x3 * 0.166667
                    exp_val += x4 * 0.041667
                    exp_val = select(exp_val, 0.0, 0.0001, exp_val)
                    exp_val = select(-8.0, x, exp_val, 0.0001)
                    row_float[j1, j2] = exp_val
                    sum_exp += exp_val

            for j1 in seq(0, dim):
                for j2 in seq(0, dim):
                    softmax_val: f32
                    quantized: f32
                    softmax_val = row_float[j1, j2] / sum_exp
                    quantized = softmax_val / softmax_output_scale + 0.5
                    quantized = fmaxf(
                        -128.0, select(quantized, 127.0, quantized, 127.0)
                    )
                    attention[i1, i2, j1, j2] = quantized

    # --- Attended output: attention @ g^T -> [81][32] ---
    # attention[81][81], g_reshaped[32][81]
    # C[81][32] = attention[81][81] @ g^T[81][32]
    attended: i8[dim, dim, ci] @ DRAM
    cpu_matmul_transB(dim, ci, attention, g_reshaped, attended, nlb_matmul_1_scale)

    # Reshape [81][32] -> NHWC [9][9][32]
    tpg_output: i8[dim, dim, ci] @ DRAM
    for c in seq(0, ci):
        for h in seq(0, dim):
            for w in seq(0, dim):
                tpg_output[h, w, c] = attended[h, w, c]

    # --- out_cnn 1x1 conv: 9x9x32 -> 9x9x64 ---
    nlb_conv_out: i8[dim, dim, c0] @ DRAM
    cpu_conv2d(
        dim,
        dim,
        ci,
        c0,
        1,
        dim,
        dim,
        tpg_output,
        nlb_out_weights,
        nlb_out_bias,
        nlb_conv_out,
        nlb_out_scale,
    )

    # --- Residual add: output = input * B_scale + nlb_conv_out * A_scale ---
    cpu_resadd(
        dim, c0, nlb_add_b_scale, nlb_add_a_scale, input, nlb_conv_out, output
    )


@proc
def braggnn_inference_exo_impl(
    input_dim: size,
    conv1_dim: size,
    conv2_dim: size,
    conv3_dim: size,
    conv1_filters: size,
    conv2_filters: size,
    conv3_filters: size,
    fc1_units: size,
    fc2_units: size,
    fc3_units: size,
    fc4_units: size,
    fp32_input: f32[input_dim, input_dim, 1] @ DRAM,
    conv1_weights: i8[conv1_filters, 3, 3, 1] @ DRAM,
    conv1_bias: i32[conv1_filters] @ DRAM,
    conv1_scale: f32 @ DRAM,
    nlb_theta_weights: i8[conv2_filters, 1, 1, conv1_filters] @ DRAM,
    nlb_theta_bias: i32[conv2_filters] @ DRAM,
    nlb_theta_scale: f32 @ DRAM,
    nlb_phi_weights: i8[conv2_filters, 1, 1, conv1_filters] @ DRAM,
    nlb_phi_bias: i32[conv2_filters] @ DRAM,
    nlb_phi_scale: f32 @ DRAM,
    nlb_g_weights: i8[conv2_filters, 1, 1, conv1_filters] @ DRAM,
    nlb_g_bias: i32[conv2_filters] @ DRAM,
    nlb_g_scale: f32 @ DRAM,
    nlb_matmul_scale: f32 @ DRAM,
    softmax_input_scale: f32 @ DRAM,
    softmax_output_scale: f32 @ DRAM,
    nlb_matmul_1_scale: f32 @ DRAM,
    nlb_out_weights: i8[conv1_filters, 1, 1, conv2_filters] @ DRAM,
    nlb_out_bias: i32[conv1_filters] @ DRAM,
    nlb_out_scale: f32 @ DRAM,
    nlb_add_a_scale: f32 @ DRAM,
    nlb_add_b_scale: f32 @ DRAM,
    leaky1_scale: f32 @ DRAM,
    conv2_weights: i8[conv2_filters, 3, 3, conv1_filters] @ DRAM,
    conv2_bias: i32[conv2_filters] @ DRAM,
    conv2_scale: f32 @ DRAM,
    leaky3_scale: f32 @ DRAM,
    conv3_weights: i8[conv3_filters, 3, 3, conv2_filters] @ DRAM,
    conv3_bias: i32[conv3_filters] @ DRAM,
    conv3_scale: f32 @ DRAM,
    leaky5_scale: f32 @ DRAM,
    fc1_weights: i8[fc1_units, conv3_filters, conv3_dim, conv3_dim] @ DRAM,
    fc1_bias: i32[fc1_units] @ DRAM,
    fc1_scale: f32 @ DRAM,
    dense1_leaky_scale: f32 @ DRAM,
    fc2_weights: i8[fc2_units, fc1_units, 1, 1] @ DRAM,
    fc2_bias: i32[fc2_units] @ DRAM,
    fc2_scale: f32 @ DRAM,
    dense3_leaky_scale: f32 @ DRAM,
    fc3_weights: i8[fc3_units, fc2_units, 1, 1] @ DRAM,
    fc3_bias: i32[fc3_units] @ DRAM,
    fc3_scale: f32 @ DRAM,
    dense5_leaky_scale: f32 @ DRAM,
    fc4_weights: i8[fc4_units, fc3_units, 1, 1] @ DRAM,
    fc4_bias: i32[fc4_units] @ DRAM,
    fc4_scale: f32 @ DRAM,
    dense7_leaky_scale: f32 @ DRAM,
    output_weights: i8[2, fc4_units, 1, 1] @ DRAM,
    output_bias: i32[2] @ DRAM,
    output_scale: f32 @ DRAM,
    output: i8[2, 1, 1] @ DRAM,
):
    assert input_dim >= conv1_dim + 2
    assert conv1_dim >= conv2_dim + 2
    assert conv2_dim >= conv3_dim + 2
    assert conv1_dim >= 1

    # QuantizeLinear: y_scale = 0.007874015718698502 -> 1/0.007874 = 127
    input: i8[input_dim, input_dim, 1] @ DRAM
    for h in seq(0, input_dim):
        for w in seq(0, input_dim):
            q: f32
            q = fp32_input[h, w, 0] * 127.0
            input[h, w, 0] = q

    # Conv1: 11x11x1 -> 9x9x64
    conv1_out: i8[conv1_dim, conv1_dim, conv1_filters] @ DRAM
    cpu_conv2d(
        input_dim,
        input_dim,
        1,
        conv1_filters,
        3,
        conv1_dim,
        conv1_dim,
        input,
        conv1_weights,
        conv1_bias,
        conv1_out,
        conv1_scale,
    )

    # Non-Local Block: 9x9x64 -> 9x9x64
    nlb_out: i8[conv1_dim, conv1_dim, conv1_filters] @ DRAM
    cpu_nlb(
        conv1_dim,
        conv1_filters,
        conv2_filters,
        conv1_out,
        nlb_theta_weights,
        nlb_theta_bias,
        nlb_theta_scale,
        nlb_phi_weights,
        nlb_phi_bias,
        nlb_phi_scale,
        nlb_g_weights,
        nlb_g_bias,
        nlb_g_scale,
        nlb_matmul_scale,
        softmax_input_scale,
        softmax_output_scale,
        nlb_matmul_1_scale,
        nlb_out_weights,
        nlb_out_bias,
        nlb_out_scale,
        nlb_add_a_scale,
        nlb_add_b_scale,
        nlb_out,
    )

    # LeakyReLU + requant (NLB add output scale -> Conv2 input scale)
    leaky_relu_requant(conv1_dim, conv1_dim, conv1_filters, nlb_out, leaky1_scale)

    # Conv2: 9x9x64 -> 7x7x32
    conv2_out: i8[conv2_dim, conv2_dim, conv2_filters] @ DRAM
    cpu_conv2d(
        conv1_dim,
        conv1_dim,
        conv1_filters,
        conv2_filters,
        3,
        conv2_dim,
        conv2_dim,
        nlb_out,
        conv2_weights,
        conv2_bias,
        conv2_out,
        conv2_scale,
    )

    # LeakyReLU + requant (Conv2 output scale -> Conv3 input scale)
    leaky_relu_requant(conv2_dim, conv2_dim, conv2_filters, conv2_out, leaky3_scale)

    # Conv3: 7x7x32 -> 5x5x8
    conv3_out: i8[conv3_dim, conv3_dim, conv3_filters] @ DRAM
    cpu_conv2d(
        conv2_dim,
        conv2_dim,
        conv2_filters,
        conv3_filters,
        3,
        conv3_dim,
        conv3_dim,
        conv2_out,
        conv3_weights,
        conv3_bias,
        conv3_out,
        conv3_scale,
    )

    # Flatten: NHWC [5][5][8] -> NCHW order [8][5][5] = 200 elements
    flattened: i8[conv3_filters, conv3_dim, conv3_dim] @ DRAM
    for ch in seq(0, conv3_filters):
        for r in seq(0, conv3_dim):
            for c in seq(0, conv3_dim):
                flattened[ch, r, c] = conv3_out[r, c, ch]

    # LeakyReLU + requant (Conv3 output scale -> FC1 input scale)
    leaky_relu_requant(
        conv3_filters, conv3_dim, conv3_dim, flattened, leaky5_scale
    )

    # FC1: 200 -> 16
    fc1_out: i8[fc1_units, 1, 1] @ DRAM
    cpu_fc(
        conv3_filters,
        conv3_dim,
        conv3_dim,
        fc1_units,
        flattened,
        fc1_weights,
        fc1_bias,
        fc1_out,
        fc1_scale,
    )

    leaky_relu_requant(fc1_units, 1, 1, fc1_out, dense1_leaky_scale)

    # FC2: 16 -> 8
    fc2_out: i8[fc2_units, 1, 1] @ DRAM
    cpu_fc(
        fc1_units, 1, 1, fc2_units, fc1_out, fc2_weights, fc2_bias, fc2_out, fc2_scale
    )

    leaky_relu_requant(fc2_units, 1, 1, fc2_out, dense3_leaky_scale)

    # FC3: 8 -> 4
    fc3_out: i8[fc3_units, 1, 1] @ DRAM
    cpu_fc(
        fc2_units, 1, 1, fc3_units, fc2_out, fc3_weights, fc3_bias, fc3_out, fc3_scale
    )

    leaky_relu_requant(fc3_units, 1, 1, fc3_out, dense5_leaky_scale)

    # FC4: 4 -> 2
    fc4_out: i8[fc4_units, 1, 1] @ DRAM
    cpu_fc(
        fc3_units, 1, 1, fc4_units, fc3_out, fc4_weights, fc4_bias, fc4_out, fc4_scale
    )

    leaky_relu_requant(fc4_units, 1, 1, fc4_out, dense7_leaky_scale)

    # Output: 2 -> 2
    cpu_fc(
        fc4_units, 1, 1, 2, fc4_out, output_weights, output_bias, output, output_scale
    )
