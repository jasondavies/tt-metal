// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"

namespace ttnn::operations::unary_backward {

std::vector<ttnn::Tensor> _mul_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, scalar, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _clamp_min_bw(
    const Tensor& grad, const Tensor& input, float min, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor minT = ttnn::ge(input, min, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _clamp_max_bw(
    const Tensor& grad, const Tensor& input, float max, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor maxT = ttnn::le(input, max, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, maxT, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _clamp_bw(
    const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor minT = ttnn::ge(input, min, std::nullopt, output_mem_config);
    Tensor maxT = ttnn::le(input, max, std::nullopt, output_mem_config);
    Tensor result = ttnn::logical_and(minT, maxT, std::nullopt, output_mem_config);
    result = ttnn::multiply(grad, result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _multigammaln_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor digamma_result = ttnn::multiply(grad, tt::tt_metal::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor digamma_result_2 = ttnn::multiply(
        grad, tt::tt_metal::digamma(ttnn::add(input, -0.5, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::add(digamma_result, digamma_result_2, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad, tt::tt_metal::digamma(ttnn::add(input, -1.0, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad, tt::tt_metal::digamma(ttnn::add(input, -1.5, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _add_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _unary_comp_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::operations::creation::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

std::vector<Tensor> _eq_bw(
    const Tensor& grad, const Tensor& input, float other, const MemoryConfig& output_mem_config) {
    return _unary_comp_bw(grad, output_mem_config);
}

std::vector<Tensor> _lgamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(grad, tt::tt_metal::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _sub_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _frac_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _trunc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::operations::creation::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// return: grad_output * (max_deriv - sign * (z / (1 + z)))
// z = exp(-abs(input))
std::vector<Tensor> _log_sigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor max_deriv = where(ttnn::ltz(input, output_mem_config), 1, 0, output_mem_config);
    Tensor in_sign = where(ttnn::ltz(input, output_mem_config), 1, -1, output_mem_config);
    Tensor in_abs = ttnn::abs(input, output_mem_config);
    Tensor z = ttnn::exp(ttnn::neg(in_abs, output_mem_config), false, output_mem_config);

    Tensor mul_z = ttnn::multiply(z, ttnn::reciprocal((ttnn::add(z, 1.0f, std::nullopt, output_mem_config)), output_mem_config), std::nullopt, output_mem_config);

    Tensor mul_sign = ttnn::multiply(in_sign, mul_z, std::nullopt, output_mem_config);
    Tensor sub_max = ttnn::subtract(max_deriv, mul_sign, std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::multiply(grad, sub_max, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _fill_zero_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::operations::creation::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _i0_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor value = ttnn::multiply(
        ttnn::multiply(ttnn::i0(input, output_mem_config), ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config),
        0.5,
        std::nullopt,
        output_mem_config);
    Tensor result = where(
        ttnn::ltz(input, output_mem_config),
        ttnn::multiply(grad,
            ttnn::subtract(ttnn::neg(ttnn::i0(input, output_mem_config), output_mem_config), value, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::multiply(grad,
            ttnn::subtract(ttnn::i0(input, output_mem_config), value, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    result = where(
        ttnn::ge(ttnn::abs(ttnn::i0(input, output_mem_config), output_mem_config), 3.4e+38, std::nullopt, output_mem_config),
        t_inf,
        result,
        output_mem_config);
    result =
        where(ttnn::ge(ttnn::abs(result, output_mem_config), 3.4e+38, std::nullopt, output_mem_config), t_inf, result, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _tan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tan_result = ttnn::tan(input, output_mem_config);
    Tensor result =
        ttnn::multiply(grad, ttnn::add(ttnn::square(tan_result, output_mem_config), 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// grad(sigmoid) = grad*(1 - sigmoid(x))*sigmoid(x)
std::vector<Tensor> _sigmoid_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    std::vector<Tensor> grad_tensor;
    Tensor sig_result = ttnn::sigmoid(input, output_mem_config);
    Tensor rsub_term = ttnn::rsub(sig_result, 1.0f, output_mem_config);
    Tensor prod_term_1 = ttnn::multiply(sig_result, rsub_term, std::nullopt, output_mem_config);
    Tensor prod_term_2 = ttnn::multiply(prod_term_1, grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(prod_term_2);
    return grad_tensor;
}

std::vector<Tensor> _rsqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor rsqrt_result = ttnn::power(ttnn::rsqrt(input, true, output_mem_config), 3, output_mem_config);
    Tensor result = ttnn::multiply(ttnn::multiply(grad, rsqrt_result, std::nullopt, output_mem_config), -0.5, std::nullopt, output_mem_config);
    float t_inf = std::numeric_limits<float>::infinity();
    result = where(ttnn::eqz(input, output_mem_config), t_inf, result, output_mem_config);
    float t_nan = std::nanf("");
    result = where(ttnn::ltz(input, output_mem_config), t_nan, result, output_mem_config);
    result = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        result,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _neg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::neg(grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _relu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(ttnn::gtz(input, output_mem_config), grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _fill_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor val = grad;
    val = global_sum(val, output_mem_config);
    Tensor result = tt::tt_metal::zeros_like(grad, output_mem_config);
    result = bcast(result, val, BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _hardsigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = where(
        ttnn::logical_or(
            ttnn::le(input, -3, std::nullopt, output_mem_config),
            ttnn::ge(input, 3, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::operations::creation::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config),
        ttnn::multiply(grad, 1.0 / 6),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// name: cos(Tensor self) -> Tensor
// self: grad * -self.sin()
std::vector<Tensor> _cos_bw(const Tensor& grad, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result =
        ttnn::multiply(grad, (ttnn::neg(ttnn::sin(input_tensor, output_mem_config), output_mem_config)), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _acosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor in_rsqrt = ttnn::square(input, output_mem_config);
    in_rsqrt = ttnn::rsqrt(ttnn::subtract(in_rsqrt, 1.0, std::nullopt, output_mem_config), true, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor cond_result = ttnn::logical_or(
        ttnn::lt(input, ttnn::operations::creation::full_like(input, -1.0), std::nullopt, output_mem_config),
        ttnn::gt(input, ttnn::operations::creation::full_like(input, 1.0), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(ttnn::eqz(cond_result, output_mem_config), t_nan, grad_a, output_mem_config);
    cond_result = ttnn::logical_or(
        ttnn::eq(input, ttnn::operations::creation::full_like(input, -1.0), std::nullopt, output_mem_config),
        ttnn::eq(input, ttnn::operations::creation::full_like(input, 1.0), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(
        ttnn::eq(cond_result, ttnn::operations::creation::full_like(input, 1.0), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// # - name: acos(Tensor self) -> Tensor
// #   self: grad * -((-self * self + 1).rsqrt())
std::vector<Tensor> _acos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor neg_in = ttnn::neg(input, output_mem_config);
    Tensor in_rsqrt =
        ttnn::rsqrt(ttnn::add(ttnn::multiply(neg_in, input, std::nullopt, output_mem_config), 1.0f, std::nullopt, output_mem_config), true, output_mem_config);
    in_rsqrt = ttnn::neg(in_rsqrt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    Tensor neg_one = ttnn::operations::creation::full_like(input, -1.0);
    Tensor pos_one = ttnn::operations::creation::full_like(input, 1.0);
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_or(
            ttnn::lt(input, neg_one, std::nullopt, output_mem_config),
            ttnn::gt(input, pos_one, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::eq(input, neg_one, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::eq(input, pos_one, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _atan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::SQUARE},
    UnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam{UnaryOpType::RECIP}};
    Tensor grad_a = ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _rad2deg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_180_PI = 180 / M_PI;
    Tensor grad_result = ttnn::multiply(grad, M_180_PI, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _logit_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result =
        ttnn::multiply(grad,
            ttnn::reciprocal(ttnn::multiply(input, ttnn::rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)),
            std::nullopt,
            output_mem_config);
    Tensor status = ttnn::logical_and(
        ttnn::ge(input, 0.0f, std::nullopt, output_mem_config),
        ttnn::le(input, 1.0f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(status, tt::tt_metal::ones_like(input, output_mem_config), std::nullopt, output_mem_config), grad_result, std::nanf(""));
    grad_result = where(
        ttnn::logical_or(
            ttnn::eq(input, 0.0, std::nullopt, output_mem_config),
            ttnn::eq(input, 1.0, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
        grad_result,
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
// square
// result:  2 * input * grad_data
std::vector<Tensor> _square_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(ttnn::multiply(grad, 2.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _hardshrink_bw(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor hardshrink_result = hardshrink(input_tensor, lambd, output_mem_config);
    Tensor result = where(ttnn::eqz(hardshrink_result, output_mem_config), 0.0f, grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}


// softshrink
//  result: torch.where(self < -lambd, grad, torch.where(self > lambd, grad, torch.tensor(0.0)))
std::vector<Tensor> _softshrink_bw(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = where(
        ttnn::logical_or(
            ttnn::lt(input_tensor, ttnn::operations::creation::full_like(input_tensor, -lambd, input_tensor.get_dtype(), input_tensor.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            ttnn::gt(input_tensor, ttnn::operations::creation::full_like(input_tensor, lambd, input_tensor.get_dtype(), input_tensor.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        ttnn::operations::creation::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}


// Leaky_Relu
// result: torch.where(self > 0, grad_output, grad_output * negative_slope)
std::vector<Tensor> _leaky_relu_bw(
    const Tensor& grad, const Tensor& input, float negative_slope, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config), grad,  ttnn::multiply(grad, negative_slope, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// ELU
// result : grad * (torch.where(input >= 0, 1, alpha * torch.exp(input)))
std::vector<Tensor> _elu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gez(input, output_mem_config),
        grad,
        ttnn::multiply(grad, ttnn::multiply(ttnn::exp(input, false, output_mem_config), alpha, std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// Celu
// result: torch.where((input > 0), grad, grad * torch.exp(input / alpha))
std::vector<Tensor> _celu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor div_result = ttnn::multiply(
        input, ttnn::reciprocal(ttnn::operations::creation::full_like(input, alpha, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    Tensor exp_result = ttnn::exp(div_result, false, output_mem_config);
    Tensor grad_result = where(
        ttnn::gt(input, ttnn::operations::creation::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        grad,
        ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


std::vector<Tensor> _rpow_bw(
    const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    Tensor grad_result = ttnn::operations::creation::zeros_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    if (exponent != 0.0) {
        grad_result =
            ttnn::multiply(grad,
                ttnn::multiply(pow(input, exponent - 1, output_mem_config), exponent, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config);
        grad_result = where(ltz(input, output_mem_config), t_nan, grad_result, output_mem_config);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


std::vector<Tensor> _floor_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::operations::creation::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> _round_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::operations::creation::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> _log_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::operations::creation::full_like(input, std::numeric_limits<float>::infinity());
    Tensor t_nan = ttnn::operations::creation::full_like(input, std::nanf(""));
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(grad, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        grad_a,
        output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> _relu6_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_tensor = ttnn::operations::creation::zeros_like(input);
    Tensor one_tensor = ttnn::operations::creation::ones_like(input);
    Tensor six_tensor = ttnn::operations::creation::full_like(input, 6);
    Tensor grad_result =
        where(ttnn::le(input, zero_tensor, std::nullopt, output_mem_config), zero_tensor, six_tensor, output_mem_config);
    grad_result = where(
        ttnn::logical_and(
            ttnn::gtz(input, output_mem_config),
            ttnn::lt(input, six_tensor, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        grad_result,
        output_mem_config);
    grad_result =
        where(ttnn::ge(input, six_tensor, std::nullopt, output_mem_config), zero_tensor, grad_result, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Silu
// result:  grad * sigmoid_result * (1 + input * (1 - sigmoid_result))
std::vector<Tensor> _silu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_sigmoid = ttnn::multiply(grad, ttnn::sigmoid(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor add_sub = ttnn::add(
        ttnn::multiply(ttnn::subtract(ttnn::operations::creation::full_like(input, 1.0f) , ttnn::sigmoid(input, output_mem_config), std::nullopt, output_mem_config),
            input,
            std::nullopt,
            output_mem_config),
        1.0f,
        std::nullopt,
        output_mem_config);
    Tensor grad_result = ttnn::multiply(grad_sigmoid, add_sub, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Selu
// result:  torch.where(input > 0, grad * lambd, grad * lambd * alpha * torch.exp(input))
std::vector<Tensor> _selu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_lambd = ttnn::multiply(grad, 1.0507f, std::nullopt, output_mem_config);
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad_lambd,
        ttnn::multiply(ttnn::multiply(grad_lambd, 1.673260f, std::nullopt, output_mem_config),
            ttnn::exp(input, false, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Hardswish
// result: torch.where(input < -3,0.0,torch.where(input <= 3, grad * ((input / 3) + 0.5), grad),)
std::vector<Tensor> _hardswish_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::lt(input, ttnn::operations::creation::full_like(input, -3.0f), std::nullopt, output_mem_config),
        0.0,
        where(
            ttnn::le(input, ttnn::operations::creation::full_like(input, 3.0f), std::nullopt, output_mem_config),
            ttnn::multiply(grad,
                ttnn::add(ttnn::multiply(input, 0.3333f, std::nullopt, output_mem_config), 0.5f, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            grad),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> _tanhshrink_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = ttnn::square(ttnn::tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> _atanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::SUB_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::NEG},
    UnaryWithParam {UnaryOpType::RECIP}};

    Tensor grad_a =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eqz(grad, output_mem_config), t_nan, grad_a, output_mem_config);
    grad_a = where(ttnn::logical_and(ttnn::eqz(grad, output_mem_config), ttnn::eqz(input, output_mem_config)), 0, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::logical_or(
                ttnn::eq(input, 1, std::nullopt, output_mem_config),
                ttnn::eq(input, -1, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::nez(grad, output_mem_config)),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eq(grad_a, t_inf, std::nullopt, output_mem_config), ttnn::ltz(grad, output_mem_config)),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// Asin
// result: grad * (-self * self + 1).rsqrt()
std::vector<Tensor> _asin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::NEG},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::RSQRT, true}};

    Tensor grad_result =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::operations::creation::full_like(input, std::numeric_limits<float>::infinity());
    Tensor t_nan = ttnn::operations::creation::full_like(input, std::nanf(""));
    Tensor sub_one = ttnn::add(input, -1, std::nullopt, output_mem_config);
    Tensor sub_minus_one = ttnn::add(input, 1, std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::ltz(sub_minus_one, output_mem_config),
        t_nan,
        where(
            ttnn::gtz(sub_one, output_mem_config),
            t_nan,
            where(
                ttnn::eqz(sub_minus_one, output_mem_config),
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                where(
                    ttnn::eqz(sub_one, output_mem_config),
                    ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    grad_result,
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Asinh
// result: grad * (self * self + 1).rsqrt()
std::vector<Tensor> _asinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::RSQRT, true}};
    Tensor grad_result =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// name: sin(Tensor self) -> Tensor
// self: grad * self.cos()
std::vector<Tensor> _sin_bw(const Tensor& grad, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_input = ttnn::multiply(grad, ttnn::cos(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_input);
    return grad_tensor;
}

// name: sinh(Tensor self) -> Tensor
// self: grad * self.cosh()
std::vector<Tensor> _sinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, ttnn::operations::creation::full_like(input, 88.5), std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, ttnn::operations::creation::full_like(input, -88.5), std::nullopt, output_mem_config),
            t_inf,
            ttnn::multiply(grad, cosh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log10(in)) = grad/(in * 2.30258509299404568402)
std::vector<Tensor> _log10_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad, ttnn::reciprocal(ttnn::multiply(input, M_LN10, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log1p(in)) = grad/(in + 1)
// for -1 = inf
std::vector<Tensor> _log1p_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor t_inp1 = ttnn::add(input, 1.0f, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(t_inp1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::eq(input, ttnn::operations::creation::full_like(input, -1.0), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(t_inp1, output_mem_config), eqz(grad, output_mem_config)),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> _erfc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        -M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _ceil_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::operations::creation::zeros_like(grad);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

// softsign
// result = grad_data / torch.square(1 + torch.abs(input))
std::vector<Tensor> _softsign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::ABS},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::RECIP}};
    grad_tensor.emplace_back(
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config));
    return grad_tensor;
}


// name: cosh(Tensor self) -> Tensor
// self: grad * self.sinh()
std::vector<Tensor> _cosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor t_neg_inf =
        ttnn::multiply(ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input,
        ttnn::operations::creation::full_like(input, 88.50, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config), t_inf,
        where(
            ttnn::lt(input,
            ttnn::operations::creation::full_like(input, -88.50, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            t_neg_inf,
            ttnn::multiply(grad, sinh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_neg_inf.deallocate();
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}


// Torch reference
// # if eps is not None:
// #         lo = eps
// #         hi = 1.0 - lo
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= lo, self <= hi),
// #             grad_output / (self * (1.0 - self)),
// #             0.0,
// #         )
// #     else:
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= 0.0, self <= 1.0),
// #             grad_output / (self * (1.0 - self)),
// #             self.new_full((), float("nan")),
// #         )
std::vector<Tensor> _logiteps_bw(
    const Tensor& grad, const Tensor& input, float eps, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float low, high;
    low = eps;
    high = 1.0 - low;
    Tensor grad_result =
        ttnn::multiply(grad,
            ttnn::reciprocal(ttnn::multiply(input, ttnn::rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)),
            std::nullopt,
            output_mem_config);
    Tensor t_eps = ttnn::operations::creation::full_like(input, eps, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor t_low = ttnn::operations::creation::full_like(input, low, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor t_high = ttnn::operations::creation::full_like(input, high, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor ltl_gth = ttnn::logical_or(
        ttnn::lt(input, t_low, std::nullopt, output_mem_config),
        ttnn::gt(input, t_high, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(ltl_gth, ttnn::operations::creation::ones_like(input, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        where(ttnn::ltz(t_eps, output_mem_config), std::nanf(" "), 0.0, output_mem_config),
        where(
            ttnn::logical_or(
                ttnn::eq(input, 0.0, std::nullopt, output_mem_config),
                ttnn::eq(input, 1.0, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
            grad_result,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// bw(log2(in)) = grad/(in * 0.69314718055994530942)
std::vector<Tensor> _log2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad, ttnn::reciprocal(ttnn::multiply(input, M_LN2, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}


std::vector<Tensor> _sign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::operations::creation::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}


std::vector<Tensor> _fmod_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}


std::vector<Tensor> _remainder_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}


std::vector<Tensor> _div_no_nan_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zeros = ttnn::operations::creation::zeros_like(grad, grad.get_dtype(), grad.get_layout(), std::nullopt, output_mem_config);
    Tensor val = ttnn::operations::creation::full_like(input, scalar, input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::eq(val, 0, std::nullopt, output_mem_config), zeros, ttnn::multiply(grad, 1 / scalar, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> UnaryBackwardFunction::get_function_type1(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::ASSIGN_BW:
            return _assign_bw;
        case UnaryBackwardOpType::MULTIGAMMALN_BW:
            return _multigammaln_bw;
        case UnaryBackwardOpType::LGAMMA_BW:
            return _lgamma_bw;
        case UnaryBackwardOpType::FILL_BW:
            return _fill_bw;
        case UnaryBackwardOpType::HARDSIGMOID_BW:
            return _hardsigmoid_bw;
        case UnaryBackwardOpType::COS_BW:
            return _cos_bw;
        case UnaryBackwardOpType::ACOSH_BW:
            return _acosh_bw;
        case UnaryBackwardOpType::ACOS_BW:
            return _acos_bw;
        case UnaryBackwardOpType::ATAN_BW:
            return _atan_bw;
        case UnaryBackwardOpType::RAD2DEG_BW:
            return _rad2deg_bw;
        case UnaryBackwardOpType::FRAC_BW:
            return _frac_bw;
        case UnaryBackwardOpType::TRUNC_BW:
            return _trunc_bw;
        case UnaryBackwardOpType::LOG_SIGMOID_BW:
            return _log_sigmoid_bw;
        case UnaryBackwardOpType::FILL_ZERO_BW:
            return _fill_zero_bw;
        case UnaryBackwardOpType::I0_BW:
            return _i0_bw;
        case UnaryBackwardOpType::TAN_BW:
            return _tan_bw;
        case UnaryBackwardOpType::SIGMOID_BW:
            return _sigmoid_bw;
        case UnaryBackwardOpType::RSQRT_BW:
            return _rsqrt_bw;
        case UnaryBackwardOpType::NEG_BW:
            return _neg_bw;
        case UnaryBackwardOpType::RELU_BW:
            return _relu_bw;
        case UnaryBackwardOpType::LOGIT_BW:
            return _logit_bw;
        case UnaryBackwardOpType::FLOOR_BW:
            return _floor_bw;
        case UnaryBackwardOpType::ROUND_BW:
            return _round_bw;
        case UnaryBackwardOpType::LOG_BW:
            return _log_bw;
        case UnaryBackwardOpType::RELU6_BW:
            return _relu6_bw;
        case UnaryBackwardOpType::ABS_BW:
            return _abs_bw;
        case UnaryBackwardOpType::SILU_BW:
            return _silu_bw;
        case UnaryBackwardOpType::SELU_BW:
            return _selu_bw;
        case UnaryBackwardOpType::SQUARE_BW:
            return _square_bw;
        case UnaryBackwardOpType::HARDSWISH_BW:
            return _hardswish_bw;
        case UnaryBackwardOpType::TANHSHRINK_BW:
            return _tanhshrink_bw;
        case UnaryBackwardOpType::ATANH_BW:
            return _atanh_bw;
        case UnaryBackwardOpType::ASIN_BW:
            return _asin_bw;
        case UnaryBackwardOpType::ASINH_BW:
            return _asinh_bw;
        case UnaryBackwardOpType::SIN_BW:
            return _sin_bw;
        case UnaryBackwardOpType::SINH_BW:
            return _sinh_bw;
        case UnaryBackwardOpType::LOG10_BW:
            return _log10_bw;
        case UnaryBackwardOpType::LOG1P_BW:
            return _log1p_bw;
        case UnaryBackwardOpType::ERFC_BW:
            return _erfc_bw;
        case UnaryBackwardOpType::CEIL_BW:
            return _ceil_bw;
        case UnaryBackwardOpType::SOFTSIGN_BW:
            return _softsign_bw;
        case UnaryBackwardOpType::COSH_BW:
            return _cosh_bw;
        case UnaryBackwardOpType::LOG2_BW:
            return _log2_bw;
        case UnaryBackwardOpType::SIGN_BW:
            return _sign_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> UnaryBackwardFunction::get_function_type1_w_float(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::MUL_BW:
            return _mul_bw;
        case UnaryBackwardOpType::CLAMP_MIN_BW:
            return _clamp_min_bw;
        case UnaryBackwardOpType::CLAMP_MAX_BW:
            return _clamp_max_bw;
        case UnaryBackwardOpType::ADD_BW:
            return _add_bw;
        case UnaryBackwardOpType::EQ_BW:
            return _eq_bw;
        case UnaryBackwardOpType::SUB_BW:
            return _sub_bw;
        case UnaryBackwardOpType::HARDSHRINK_BW:
            return _hardshrink_bw;
        case UnaryBackwardOpType::SOFTSHRINK_BW:
            return _softshrink_bw;
        case UnaryBackwardOpType::LEAKY_RELU_BW:
            return _leaky_relu_bw;
        case UnaryBackwardOpType::ELU_BW:
            return _elu_bw;
        case UnaryBackwardOpType::CELU_BW:
            return _celu_bw;
        case UnaryBackwardOpType::RPOW_BW:
            return _rpow_bw;
        case UnaryBackwardOpType::LOGITEPS_BW:
            return _logiteps_bw;
        case UnaryBackwardOpType::FMOD_BW:
            return _fmod_bw;
        case UnaryBackwardOpType::REMAINDER_BW:
            return _remainder_bw;
        case UnaryBackwardOpType::DIV_NO_NAN_BW:
            return _div_no_nan_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, float, const MemoryConfig&)> UnaryBackwardFunction::get_function_type1_w_two_float(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::CLAMP_BW:
            return _clamp_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

}  // namespace ttnn::operations::unary