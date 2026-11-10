// Convolutions on CNNL: the direct form, its three gradients, and the
// transposed form, which is the direct form's backward-data with the operands
// exchanged.  The card runs 4-D and 5-D convolutions, so a 1-D convolution is
// lifted to the 2-D one with a unit third axis.  A 4-D forward pass and every
// 5-D pass run in the channel-last order the card requires, while a 4-D
// backward-data pass (which is also the transposed forward pass) runs
// channel-first and moves nothing.  Every kernel returns nothing when the
// shape, the type or the device leaves it without a CNNL kernel, and the
// operator then runs on the host.
//
// The backward routes follow from the adjoint of the forward convolution:
// with ``y = conv(x, w)`` the gradient of the input is the backward-data pass
// ``BwdData(filter=w, diff_y=dy)``, the gradient of the filter is
// ``BwdFilter(x=x, diff_y=dy)``, and the gradient of the bias is ``dy`` summed
// over every axis but the channel one.  The transposed convolution is
// ``BwdData(filter=w, diff_y=x)``; its input gradient is the forward pass on
// the output gradient with the same weight, and its filter gradient is the
// backward-filter pass with the two data operands exchanged.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

// ================================================================ geometry

// A convolution's geometry once its spatial parameters are expanded to one
// entry per spatial dimension.
struct Geometry {
    int dim = 4;  // 4 for a 2-D convolution, 5 for a 3-D one
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    std::vector<int> output_padding;  // transposed form only
    int64_t groups = 1;

    int spatial() const { return dim - 2; }
};

// Types the convolution kernels compute in on device ``index``.  bfloat16
// needs a 500-series card and the 4-D form (the 5-D form requires the result
// type itself as the compute type, which bfloat16 does not allow).
bool conv_dtype(DType dtype, int64_t ndim, int index) {
    if (dtype == DType::Float32 || dtype == DType::Float16) return true;
    return dtype == DType::BFloat16 && ndim <= 4 && capability(index).first >= 5;
}

// The type a convolution computes in.  4-D results always compute in float; a
// 5-D result must name its own type as the compute type.
cnnlDataType_t compute_type(DType dtype, int dim) {
    if (dim == 5 && dtype == DType::Float16) return CNNL_DTYPE_HALF;
    return CNNL_DTYPE_FLOAT;
}

// Validates an operator call's tensors and parameters and returns the
// normalized geometry, or nothing when the card cannot run the call.
std::optional<Geometry> prepare(const Tensor& input, const Tensor& weight, int64_t groups,
                                const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation,
                                const std::vector<int64_t>& output_padding, bool transposed) {
    if (!input.device().is_privateuse1() || !weight.device().is_privateuse1()) {
        return std::nullopt;
    }
    if (input.dim() != weight.dim() || input.dim() < 4 || input.dim() > 5) return std::nullopt;
    const int index = index_of(input.device());
    if (index_of(weight.device()) != index || weight.dtype() != input.dtype()) {
        return std::nullopt;
    }
    if (!conv_dtype(input.dtype(), input.dim(), index)) return std::nullopt;

    const int n = static_cast<int>(input.dim()) - 2;
    Geometry g;
    g.dim = static_cast<int>(input.dim());
    g.groups = groups;
    auto s = spread(stride, n, 1);
    auto p = spread(padding, n, 0);
    auto d = spread(dilation, n, 1);
    auto op = spread(output_padding, n, 0);
    if (!s || !p || !d || !op || groups < 1) return std::nullopt;
    g.stride = *s;
    g.padding = *p;
    g.dilation = *d;
    g.output_padding = *op;

    for (int i = 0; i < n; ++i) {
        const size_t u = static_cast<size_t>(i);
        if (g.stride[u] < 1 || g.padding[u] < 0 || g.dilation[u] < 1 || g.output_padding[u] < 0) {
            return std::nullopt;
        }
        // A transposed convolution widens its result by less than one stride;
        // the direct form has no output padding at all.
        const int limit = transposed ? g.stride[u] : 1;
        if (g.output_padding[u] >= limit) return std::nullopt;
    }

    const int64_t in_channels = input.size(1);
    if (input.size(0) <= 0 || in_channels <= 0 || weight.size(0) <= 0 || weight.size(1) <= 0) {
        return std::nullopt;
    }
    if (transposed) {
        // weight = (input channels, output channels per group, kernel...)
        if (weight.size(0) != in_channels || in_channels % groups != 0 ||
            weight.size(1) % groups != 0) {
            return std::nullopt;
        }
    } else {
        // weight = (output channels, input channels per group, kernel...)
        if (weight.size(1) * groups != in_channels || weight.size(0) % groups != 0) {
            return std::nullopt;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (input.size(2 + i) <= 0 || weight.size(2 + i) <= 0) return std::nullopt;
    }
    return g;
}

// ================================================================ CNNL glue

// ``t`` reduced to its leading region of shape ``sizes`` (output padding can
// make a derived shape one step too wide for the operator's own shape).
Tensor leading_region(const Tensor& t, const std::vector<int64_t>& sizes) {
    if (t.sizes().vec() == sizes) return t;
    Tensor out = t;
    for (size_t d = 0; d < sizes.size(); ++d) {
        if (out.size(static_cast<int64_t>(d)) != sizes[d]) {
            out = out.slice(static_cast<int64_t>(d), 0, sizes[d]);
        }
    }
    return out.contiguous();
}

// Owning convolution descriptor.
class ConvOpDesc {
public:
    ConvOpDesc() = default;
    ConvOpDesc(const ConvOpDesc&) = delete;
    ConvOpDesc& operator=(const ConvOpDesc&) = delete;
    ~ConvOpDesc() {
        if (desc_ != nullptr) cnnlDestroyConvolutionDescriptor(desc_);
    }

    // False when the card declines the geometry.
    bool build(const Geometry& g, cnnlDataType_t compute) {
        TP_CNNL_CHECK(cnnlCreateConvolutionDescriptor(&desc_));
        const int n = g.spatial();
        // Two entries per spatial dimension: the amount concatenated at the
        // start and at the end of the axis, equal for a symmetric padding.
        std::vector<int> pad(static_cast<size_t>(2 * n));
        for (int i = 0; i < n; ++i) {
            pad[static_cast<size_t>(2 * i)] = g.padding[static_cast<size_t>(i)];
            pad[static_cast<size_t>(2 * i + 1)] = g.padding[static_cast<size_t>(i)];
        }
        const cnnlStatus_t status =
            cnnlSetConvolutionDescriptor(desc_, g.dim, pad.data(), g.stride.data(),
                                         g.dilation.data(), static_cast<int>(g.groups), compute);
        if (cnnl_declined(status)) return false;
        TP_CHECK(status == CNNL_STATUS_SUCCESS, "CNNL error: ", cnnlGetErrorString(status),
                 " (cnnlSetConvolutionDescriptor)");
        // Full float precision: the reduced-precision fast path would drift
        // the result away from the exact convolution.
        cnnlSetConvolutionDescriptorAllowTF32(desc_, 0);
        return true;
    }

    cnnlConvolutionDescriptor_t get() const { return desc_; }

private:
    cnnlConvolutionDescriptor_t desc_ = nullptr;
};

// The bias of a convolution is a channel vector; CNNL reads it as a
// degenerate channel-last tensor whose last axis is the channel.
TensorDesc bias_desc(int64_t channels, int dim, DType dtype) {
    std::vector<int64_t> sizes(static_cast<size_t>(dim), 1);
    sizes.back() = channels;
    return TensorDesc(sizes, channel_layout(dim), dtype);
}

// =============================================================== launchers

// y = convolution of ``x`` with ``w`` (plus ``bias``); every operand is
// channel-last already.
bool launch_forward(const Tensor& x, const Tensor& w, const Tensor* bias, const Geometry& g,
                    const Tensor& y) {
    const int index = index_of(y.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x.sizes().vec(), channel_layout(g.dim), x.dtype());
    TensorDesc w_desc(w.sizes().vec(), channel_layout(g.dim), w.dtype());
    TensorDesc y_desc(y.sizes().vec(), channel_layout(g.dim), y.dtype());
    TensorDesc b_desc;
    const bool has_bias = bias != nullptr;
    if (has_bias) b_desc = bias_desc(w.size(0), g.dim, w.dtype());

    ConvOpDesc conv;
    if (!conv.build(g, compute_type(x.dtype(), g.dim))) return false;

    const cnnlConvolutionForwardAlgo_t algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    const cnnlTensorDescriptor_t b = has_bias ? b_desc.get() : nullptr;
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetConvolutionForwardWorkspaceSize(h, x_desc.get(), w_desc.get(), y_desc.get(),
                                                       b, conv.get(), algo, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlConvolutionForward(h, conv.get(), algo, nullptr, x_desc.get(), x.data_ptr(),
                                       w_desc.get(), w.data_ptr(), b,
                                       has_bias ? bias->data_ptr() : nullptr, ws.get(), ws.size(),
                                       nullptr, y_desc.get(), y.data_ptr()));
    return true;
}

// diff_x = backward-data(filter, diff_y): the gradient of a convolution's
// input, and equally the transposed convolution's forward result.  A 4-D pass
// runs channel-first, a 5-D pass channel-last; the operands arrive in
// whichever of the two the pass runs in.
bool launch_bwd_data(const Tensor& filter, const Tensor& diff_y, const Geometry& g,
                     const Tensor& diff_x) {
    const int index = index_of(diff_x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    const cnnlTensorLayout_t layout = g.dim == 4 ? CNNL_LAYOUT_NCHW : CNNL_LAYOUT_NDHWC;
    TensorDesc f_desc(filter.sizes().vec(), layout, filter.dtype());
    TensorDesc dy_desc(diff_y.sizes().vec(), layout, diff_y.dtype());
    TensorDesc dx_desc(diff_x.sizes().vec(), layout, diff_x.dtype());

    ConvOpDesc conv;
    if (!conv.build(g, compute_type(filter.dtype(), g.dim))) return false;

    const cnnlConvolutionBwdDataAlgo_t algo = CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT;
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetConvolutionBackwardDataWorkspaceSize(h, f_desc.get(), dy_desc.get(),
                                                            conv.get(), dx_desc.get(), algo,
                                                            &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlConvolutionBackwardData(h, nullptr, f_desc.get(), filter.data_ptr(),
                                            dy_desc.get(), diff_y.data_ptr(), conv.get(), algo,
                                            ws.get(), ws.size(), nullptr, dx_desc.get(),
                                            diff_x.data_ptr()));
    return true;
}

// diff_w = backward-filter(x, diff_y): the gradient of a convolution's
// filter.  The data operands arrive channel-last and the filter gradient is
// written channel-first.
bool launch_bwd_filter(const Tensor& x, const Tensor& diff_y, const Geometry& g,
                       const Tensor& diff_w) {
    const int index = index_of(diff_w.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x.sizes().vec(), channel_layout(g.dim), x.dtype());
    TensorDesc dy_desc(diff_y.sizes().vec(), channel_layout(g.dim), diff_y.dtype());
    TensorDesc dw_desc(diff_w.sizes().vec(), g.dim == 4 ? CNNL_LAYOUT_NCHW : CNNL_LAYOUT_NCDHW,
                       diff_w.dtype());

    ConvOpDesc conv;
    if (!conv.build(g, compute_type(diff_w.dtype(), g.dim))) return false;

    const cnnlConvolutionBwdFilterAlgo_t algo = CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT;
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetConvolutionBackwardFilterWorkspaceSize(h, x_desc.get(), dy_desc.get(),
                                                              dw_desc.get(), conv.get(), algo,
                                                              &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlConvolutionBackwardFilter(h, nullptr, x_desc.get(), x.data_ptr(),
                                              dy_desc.get(), diff_y.data_ptr(), conv.get(), algo,
                                              ws.get(), ws.size(), nullptr, dw_desc.get(),
                                              diff_w.data_ptr()));
    return true;
}

// out += bias, broadcast over every axis but the channel one.  ``out`` is a
// contiguous channel-first tensor.
bool add_channel_bias(const Tensor& out, const Tensor& bias) {
    const int index = index_of(out.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    const size_t nd = static_cast<size_t>(std::max<int64_t>(out.dim(), 1));
    std::vector<int64_t> b_sizes(nd, 1);
    std::vector<int64_t> b_strides(nd, 1);
    b_sizes[1] = bias.numel();
    TensorDesc b_desc(b_sizes, b_strides, bias.dtype());
    TensorDesc o_desc(out);

    cnnlOpTensorDescriptor_t op_desc = nullptr;
    TP_CNNL_CHECK(cnnlCreateOpTensorDescriptor(&op_desc));
    struct Destroy {
        cnnlOpTensorDescriptor_t d;
        ~Destroy() { cnnlDestroyOpTensorDescriptor(d); }
    } destroy{op_desc};
    TP_CNNL_TRY(cnnlSetOpTensorDescriptor(op_desc, CNNL_OP_TENSOR_ADD, CNNL_DTYPE_FLOAT,
                                          CNNL_NOT_PROPAGATE_NAN));
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetOpTensorWorkspaceSize(h, o_desc.get(), b_desc.get(), o_desc.get(),
                                             &ws_size));
    Workspace ws(ws_size, index);
    const float alpha1 = 1.0f;
    const float alpha2 = 1.0f;
    const float beta = 0.0f;
    TP_CNNL_TRY(cnnlOpTensor(h, op_desc, &alpha1, o_desc.get(), out.data_ptr(), &alpha2,
                             b_desc.get(), bias.data_ptr(), ws.get(), ws.size(), &beta,
                             o_desc.get(), out.data_ptr()));
    return true;
}

// =================================================================== cores

// Whether the card can add ``bias`` to a result with ``channels`` channels.
bool bias_matches(const Tensor* bias, int64_t channels, const Tensor& like) {
    if (bias == nullptr) return true;
    return bias->defined() && bias->numel() == channels && bias->dtype() == like.dtype() &&
           index_of(bias->device()) == index_of(like.device());
}

std::optional<Tensor> conv_forward_core(const Tensor& input, const Tensor& weight,
                                        const Tensor* bias, const Geometry& g) {
    const int index = index_of(input.device());
    const int n = g.spatial();
    const int64_t nd = g.dim;

    std::vector<int64_t> y_sizes(static_cast<size_t>(nd));
    y_sizes[0] = input.size(0);
    y_sizes[1] = weight.size(0);
    for (int i = 0; i < n; ++i) {
        const size_t u = static_cast<size_t>(i);
        const int64_t in = input.size(2 + i);
        const int64_t kernel = weight.size(2 + i);
        const int64_t numerator = in + 2 * g.padding[u] -
                                  static_cast<int64_t>(g.dilation[u]) * (kernel - 1) - 1;
        if (numerator < 0) return std::nullopt;
        const int64_t out = numerator / g.stride[u] + 1;
        if (out <= 0) return std::nullopt;
        y_sizes[u + 2] = out;
    }
    if (!bias_matches(bias, y_sizes[1], input)) return std::nullopt;

    Tensor y_cl = empty_on(to_channel_last_sizes(y_sizes), input.dtype(), index);
    if (y_cl.numel() > 0) {
        if (!launch_forward(channel_last(input), channel_last(weight), bias, g, y_cl)) {
            return std::nullopt;
        }
    }
    return channel_first(y_cl);
}

std::optional<Tensor> grad_input_core(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& weight, const Geometry& g) {
    const int index = index_of(input.device());
    if (g.dim == 4) {
        // Channel-first throughout: the 4-D backward-data pass takes every
        // operand in the order it already is.
        Tensor dx = empty_on(input.sizes().vec(), input.dtype(), index);
        if (dx.numel() == 0) return dx;
        dx.zero_();
        if (!launch_bwd_data(contiguous_on_device(weight), contiguous_on_device(grad_output), g,
                             dx)) {
            return std::nullopt;
        }
        return dx;
    }
    Tensor dx = empty_on(channel_last_sizes(input), input.dtype(), index);
    if (dx.numel() == 0) return channel_first(dx);
    dx.zero_();
    if (!launch_bwd_data(channel_last(weight), channel_last(grad_output), g, dx)) {
        return std::nullopt;
    }
    return channel_first(dx);
}

std::optional<Tensor> grad_weight_core(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& weight, const Geometry& g) {
    const int index = index_of(input.device());
    Tensor dw = empty_on(weight.sizes().vec(), input.dtype(), index);
    if (dw.numel() == 0) return dw;
    dw.zero_();
    if (!launch_bwd_filter(channel_last(input), channel_last(grad_output), g, dw)) {
        return std::nullopt;
    }
    return dw;
}

std::optional<Tensor> transpose_core(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                     const Geometry& g) {
    const int index = index_of(input.device());
    const int n = g.spatial();
    const int64_t nd = g.dim;
    const int64_t channels = weight.size(1) * g.groups;

    // The transposed form's result is the backward-data pass over an input of
    // the whole output size; ``output_padding`` widens that size, and the
    // positions it adds are the taps a narrower size would have clipped, so
    // the widening happens in the descriptor rather than by a copy.
    std::vector<int64_t> out_sizes(static_cast<size_t>(nd));
    out_sizes[0] = input.size(0);
    out_sizes[1] = channels;
    for (int i = 0; i < n; ++i) {
        const size_t u = static_cast<size_t>(i);
        const int64_t in = input.size(2 + i);
        const int64_t kernel = weight.size(2 + i);
        const int64_t out = (in - 1) * g.stride[u] - 2 * g.padding[u] +
                            static_cast<int64_t>(g.dilation[u]) * (kernel - 1) +
                            g.output_padding[u] + 1;
        if (out <= 0) return std::nullopt;
        out_sizes[u + 2] = out;
    }
    if (!bias_matches(bias, channels, input)) return std::nullopt;

    // The transposed forward pass is the backward-data pass; the 4-D form
    // runs channel-first and moves nothing, the 5-D form channel-last.
    const bool last = g.dim == 5;
    Tensor buffer = last ? empty_on(to_channel_last_sizes(out_sizes), input.dtype(), index)
                         : empty_on(out_sizes, input.dtype(), index);
    if (buffer.numel() > 0) {
        buffer.zero_();
        if (!launch_bwd_data(last ? channel_last(weight) : contiguous_on_device(weight),
                             last ? channel_last(input) : contiguous_on_device(input), g, buffer)) {
            return std::nullopt;
        }
    }
    Tensor out = last ? channel_first(buffer) : buffer;
    if (bias != nullptr && out.numel() > 0 && !add_channel_bias(out, contiguous_on_device(*bias))) {
        return std::nullopt;
    }
    return out;
}

std::optional<Tensor> transpose_grad_weight_core(const Tensor& grad_output, const Tensor& input,
                                                 const Tensor& weight, const Geometry& g) {
    const int index = index_of(input.device());
    Tensor dw = empty_on(weight.sizes().vec(), input.dtype(), index);
    if (dw.numel() == 0) return dw;
    dw.zero_();
    if (!launch_bwd_filter(channel_last(grad_output), channel_last(input), g, dw)) {
        return std::nullopt;
    }
    return dw;
}

// The bias gradient is the output gradient summed over every axis but the
// channel one; ``channels`` is the length the operator's bias has (an output
// channel count, which for the transposed form is the weight's second axis
// scaled by the group count).
std::optional<Tensor> grad_bias_core(const Tensor& grad_output, int64_t channels) {
    const int index = index_of(grad_output.device());
    if (!grad_output.device().is_privateuse1() || grad_output.dim() < 3 ||
        grad_output.dim() > 5 || grad_output.size(1) != channels) {
        return std::nullopt;
    }
    if (!is_native_float(grad_output.dtype(), index)) return std::nullopt;
    std::vector<int64_t> dims;
    for (int64_t i = 0; i < grad_output.dim(); ++i) {
        if (i != 1) dims.push_back(i);
    }
    Tensor out = grad_output.sum(dims, false);
    if (out.dim() != 1 || out.size(0) != channels) return std::nullopt;
    return out;
}

// ================================================================== entries

std::optional<Tensor> forward_entry(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& padding,
                                    const std::vector<int64_t>& dilation, int64_t groups) {
    const UnitAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor w = axis.lift(weight);
    const auto g = prepare(x, w, groups, axis.lift(stride, 1), axis.lift(padding, 0),
                           axis.lift(dilation, 1),
                           {}, false);
    if (!g) return std::nullopt;
    auto r = conv_forward_core(x, w, bias, *g);
    if (!r) return std::nullopt;
    return axis.lower(*r);
}

std::optional<Tensor> grad_input_entry(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& weight, const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& padding,
                                       const std::vector<int64_t>& dilation, int64_t groups) {
    const UnitAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor w = axis.lift(weight);
    const auto g = prepare(x, w, groups, axis.lift(stride, 1), axis.lift(padding, 0),
                           axis.lift(dilation, 1),
                           {}, false);
    if (!g) return std::nullopt;
    auto r = grad_input_core(axis.lift(grad_output), x, w, *g);
    if (!r) return std::nullopt;
    return axis.lower(*r);
}

std::optional<Tensor> grad_weight_entry(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& weight, const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& padding,
                                        const std::vector<int64_t>& dilation, int64_t groups) {
    const UnitAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor w = axis.lift(weight);
    const auto g = prepare(x, w, groups, axis.lift(stride, 1), axis.lift(padding, 0),
                           axis.lift(dilation, 1),
                           {}, false);
    if (!g) return std::nullopt;
    auto r = grad_weight_core(axis.lift(grad_output), x, w, *g);
    if (!r) return std::nullopt;
    return axis.lower(*r);
}

std::optional<Tensor> transpose_entry(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& output_padding, int64_t groups,
                                      const std::vector<int64_t>& dilation) {
    const UnitAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor w = axis.lift(weight);
    const auto g = prepare(x, w, groups, axis.lift(stride, 1), axis.lift(padding, 0),
                           axis.lift(dilation, 1), axis.lift(output_padding, 0), true);
    if (!g) return std::nullopt;
    auto r = transpose_core(x, w, bias, *g);
    if (!r) return std::nullopt;
    return axis.lower(*r);
}

std::optional<Tensor> transpose_grad_input_entry(const Tensor& grad_output, const Tensor& input,
                                                 const Tensor& weight,
                                                 const std::vector<int64_t>& stride,
                                                 const std::vector<int64_t>& padding,
                                                 const std::vector<int64_t>& output_padding,
                                                 int64_t groups,
                                                 const std::vector<int64_t>& dilation) {
    const UnitAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor w = axis.lift(weight);
    const auto g = prepare(x, w, groups, axis.lift(stride, 1), axis.lift(padding, 0),
                           axis.lift(dilation, 1), axis.lift(output_padding, 0), true);
    if (!g) return std::nullopt;
    // The gradient of the input of a transposed convolution is the forward
    // pass on the output gradient, with the same weight.
    auto r = conv_forward_core(axis.lift(grad_output), w, nullptr, *g);
    if (!r) return std::nullopt;
    const std::vector<int64_t> want = x.sizes().vec();
    const std::vector<int64_t> got = r->sizes().vec();
    for (size_t d = 0; d < want.size(); ++d) {
        if (got[d] < want[d]) return std::nullopt;
    }
    return axis.lower(leading_region(*r, want));
}

std::optional<Tensor> transpose_grad_weight_entry(const Tensor& grad_output, const Tensor& input,
                                                  const Tensor& weight,
                                                  const std::vector<int64_t>& stride,
                                                  const std::vector<int64_t>& padding,
                                                  const std::vector<int64_t>& output_padding,
                                                  int64_t groups,
                                                  const std::vector<int64_t>& dilation) {
    const UnitAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor w = axis.lift(weight);
    const auto g = prepare(x, w, groups, axis.lift(stride, 1), axis.lift(padding, 0),
                           axis.lift(dilation, 1), axis.lift(output_padding, 0), true);
    if (!g) return std::nullopt;
    auto r = transpose_grad_weight_core(axis.lift(grad_output), x, w, *g);
    if (!r) return std::nullopt;
    return axis.lower(*r);
}

const Tensor* bias_or_null(const Tensor& bias) { return bias.defined() ? &bias : nullptr; }

const Tensor* bias_or_null(const std::optional<Tensor>& bias) {
    return bias.has_value() && bias->defined() ? &*bias : nullptr;
}

// ================================================================= kernels

Tensor conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = forward_entry(input, weight, bias_or_null(bias), stride, padding, dilation,
                               groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv1d", sig::conv1d>::call(input, weight, bias, stride, padding,
                                                             dilation, groups);
}

Tensor conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = forward_entry(input, weight, bias_or_null(bias), stride, padding, dilation,
                               groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv2d", sig::conv2d>::call(input, weight, bias, stride, padding,
                                                             dilation, groups);
}

Tensor conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = forward_entry(input, weight, bias_or_null(bias), stride, padding, dilation,
                               groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv3d", sig::conv3d>::call(input, weight, bias, stride, padding,
                                                             dilation, groups);
}

Tensor conv1d_grad_input_kernel(const Tensor& grad_output, const Tensor& input,
                                const Tensor& weight, const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_input_entry(grad_output, input, weight, stride, padding, dilation, groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv1d_grad_input", sig::conv1d_grad_input>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv2d_grad_input_kernel(const Tensor& grad_output, const Tensor& input,
                                const Tensor& weight, const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_input_entry(grad_output, input, weight, stride, padding, dilation, groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv2d_grad_input", sig::conv2d_grad_input>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv3d_grad_input_kernel(const Tensor& grad_output, const Tensor& input,
                                const Tensor& weight, const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_input_entry(grad_output, input, weight, stride, padding, dilation, groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv3d_grad_input", sig::conv3d_grad_input>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv1d_grad_weight_kernel(const Tensor& grad_output, const Tensor& input,
                                 const Tensor& weight, const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_weight_entry(grad_output, input, weight, stride, padding, dilation, groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv1d_grad_weight", sig::conv1d_grad_weight>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv2d_grad_weight_kernel(const Tensor& grad_output, const Tensor& input,
                                 const Tensor& weight, const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_weight_entry(grad_output, input, weight, stride, padding, dilation, groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv2d_grad_weight", sig::conv2d_grad_weight>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv3d_grad_weight_kernel(const Tensor& grad_output, const Tensor& input,
                                 const Tensor& weight, const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_weight_entry(grad_output, input, weight, stride, padding, dilation, groups)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv3d_grad_weight", sig::conv3d_grad_weight>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv1d_grad_bias_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& weight, const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_bias_core(grad_output, weight.size(0))) return *r;
    return cpu_fallback::Kernel<"conv1d_grad_bias", sig::conv1d_grad_bias>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv2d_grad_bias_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& weight, const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_bias_core(grad_output, weight.size(0))) return *r;
    return cpu_fallback::Kernel<"conv2d_grad_bias", sig::conv2d_grad_bias>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv3d_grad_bias_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& weight, const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, int64_t groups) {
    if (auto r = grad_bias_core(grad_output, weight.size(0))) return *r;
    return cpu_fallback::Kernel<"conv3d_grad_bias", sig::conv3d_grad_bias>::call(
        grad_output, input, weight, stride, padding, dilation, groups);
}

Tensor conv_transpose1d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding, int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (auto r = transpose_entry(input, weight, bias_or_null(bias), stride, padding, output_padding,
                                 groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose1d", sig::conv_transpose1d>::call(
        input, weight, bias, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose2d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding, int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (auto r = transpose_entry(input, weight, bias_or_null(bias), stride, padding, output_padding,
                                 groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose2d", sig::conv_transpose2d>::call(
        input, weight, bias, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose3d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding, int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (auto r = transpose_entry(input, weight, bias_or_null(bias), stride, padding, output_padding,
                                 groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose3d", sig::conv_transpose3d>::call(
        input, weight, bias, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose1d_grad_input_kernel(const Tensor& grad_output, const Tensor& input,
                                          const Tensor& weight, const std::vector<int64_t>& stride,
                                          const std::vector<int64_t>& padding,
                                          const std::vector<int64_t>& output_padding,
                                          int64_t groups, const std::vector<int64_t>& dilation) {
    if (auto r = transpose_grad_input_entry(grad_output, input, weight, stride, padding,
                                            output_padding, groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose1d_grad_input",
                                sig::conv_transpose1d_grad_input>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose2d_grad_input_kernel(const Tensor& grad_output, const Tensor& input,
                                          const Tensor& weight, const std::vector<int64_t>& stride,
                                          const std::vector<int64_t>& padding,
                                          const std::vector<int64_t>& output_padding,
                                          int64_t groups, const std::vector<int64_t>& dilation) {
    if (auto r = transpose_grad_input_entry(grad_output, input, weight, stride, padding,
                                            output_padding, groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose2d_grad_input",
                                sig::conv_transpose2d_grad_input>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose3d_grad_input_kernel(const Tensor& grad_output, const Tensor& input,
                                          const Tensor& weight, const std::vector<int64_t>& stride,
                                          const std::vector<int64_t>& padding,
                                          const std::vector<int64_t>& output_padding,
                                          int64_t groups, const std::vector<int64_t>& dilation) {
    if (auto r = transpose_grad_input_entry(grad_output, input, weight, stride, padding,
                                            output_padding, groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose3d_grad_input",
                                sig::conv_transpose3d_grad_input>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose1d_grad_weight_kernel(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& weight,
                                           const std::vector<int64_t>& stride,
                                           const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& output_padding,
                                           int64_t groups, const std::vector<int64_t>& dilation) {
    if (auto r = transpose_grad_weight_entry(grad_output, input, weight, stride, padding,
                                             output_padding, groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose1d_grad_weight",
                                sig::conv_transpose1d_grad_weight>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose2d_grad_weight_kernel(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& weight,
                                           const std::vector<int64_t>& stride,
                                           const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& output_padding,
                                           int64_t groups, const std::vector<int64_t>& dilation) {
    if (auto r = transpose_grad_weight_entry(grad_output, input, weight, stride, padding,
                                             output_padding, groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose2d_grad_weight",
                                sig::conv_transpose2d_grad_weight>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose3d_grad_weight_kernel(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& weight,
                                           const std::vector<int64_t>& stride,
                                           const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& output_padding,
                                           int64_t groups, const std::vector<int64_t>& dilation) {
    if (auto r = transpose_grad_weight_entry(grad_output, input, weight, stride, padding,
                                             output_padding, groups, dilation)) {
        return *r;
    }
    return cpu_fallback::Kernel<"conv_transpose3d_grad_weight",
                                sig::conv_transpose3d_grad_weight>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose1d_grad_bias_kernel(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& weight, const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& output_padding, int64_t groups,
                                         const std::vector<int64_t>& dilation) {
    if (auto r = grad_bias_core(grad_output, weight.size(1) * groups)) return *r;
    return cpu_fallback::Kernel<"conv_transpose1d_grad_bias",
                                sig::conv_transpose1d_grad_bias>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose2d_grad_bias_kernel(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& weight, const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& output_padding, int64_t groups,
                                         const std::vector<int64_t>& dilation) {
    if (auto r = grad_bias_core(grad_output, weight.size(1) * groups)) return *r;
    return cpu_fallback::Kernel<"conv_transpose2d_grad_bias",
                                sig::conv_transpose2d_grad_bias>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor conv_transpose3d_grad_bias_kernel(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& weight, const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& output_padding, int64_t groups,
                                         const std::vector<int64_t>& dilation) {
    if (auto r = grad_bias_core(grad_output, weight.size(1) * groups)) return *r;
    return cpu_fallback::Kernel<"conv_transpose3d_grad_bias",
                                sig::conv_transpose3d_grad_bias>::call(
        grad_output, input, weight, stride, padding, output_padding, groups, dilation);
}

Tensor convolution_kernel(const Tensor& input, const Tensor& weight,
                          const std::optional<Tensor>& bias,
                          const std::vector<int64_t>& stride,
                          const std::vector<int64_t>& padding,
                          const std::vector<int64_t>& dilation, bool transposed,
                          const std::vector<int64_t>& output_padding, int64_t groups) {
    std::optional<Tensor> r;
    if (transposed) {
        r = transpose_entry(input, weight, bias_or_null(bias), stride, padding, output_padding,
                            groups, dilation);
    } else {
        r = forward_entry(input, weight, bias_or_null(bias), stride, padding, dilation, groups);
    }
    if (r) return *r;
    return cpu_fallback::Kernel<"convolution", sig::convolution>::call(
        input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

// Which gradient slots a backward call asked for.
struct GradRequest {
    bool input = false;
    bool weight = false;
    bool bias = false;
};

GradRequest decode_mask(const std::vector<bool>& mask) {
    return {mask.size() > 0 && mask[0], mask.size() > 1 && mask[1], mask.size() > 2 && mask[2]};
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::optional<std::vector<int64_t>>& bias_sizes,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool transposed,
    const std::vector<int64_t>& output_padding, int64_t groups,
    const std::vector<bool>& output_mask) {
    const GradRequest want = decode_mask(output_mask);
    const int64_t rank = input.dim();
    if (rank >= 3 && rank <= 5) {
        // Every requested slot has to run on the card; one that cannot falls
        // back with the whole call so the three gradients stay consistent.
        bool ok = true;
        Tensor grad_input;
        Tensor grad_weight;
        Tensor grad_bias;
        if (want.input) {
            auto r = transposed
                         ? transpose_grad_input_entry(grad_output, input, weight, stride, padding,
                                                      output_padding, groups, dilation)
                         : grad_input_entry(grad_output, input, weight, stride, padding, dilation,
                                            groups);
            if (r) {
                grad_input = *r;
            } else {
                ok = false;
            }
        }
        if (ok && want.weight) {
            auto r = transposed
                         ? transpose_grad_weight_entry(grad_output, input, weight, stride, padding,
                                                       output_padding, groups, dilation)
                         : grad_weight_entry(grad_output, input, weight, stride, padding, dilation,
                                             groups);
            if (r) {
                grad_weight = *r;
            } else {
                ok = false;
            }
        }
        if (ok && want.bias) {
            const int64_t channels =
                transposed ? weight.size(1) * groups : weight.size(0);
            if (auto r = grad_bias_core(grad_output, channels)) {
                grad_bias = *r;
            } else {
                ok = false;
            }
        }
        if (ok) return {grad_input, grad_weight, grad_bias};
    }
    return cpu_fallback::Kernel<"convolution_backward", sig::convolution_backward>::call(
        grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed,
        output_padding, groups, output_mask);
}

Tensor convolution_overrideable_kernel(const Tensor& input, const Tensor& weight,
                                       const std::optional<Tensor>& bias,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& padding,
                                       const std::vector<int64_t>& dilation, bool transposed,
                                       const std::vector<int64_t>& output_padding,
                                       int64_t groups) {
    return convolution_kernel(input, weight, bias, stride, padding, dilation, transposed,
                             output_padding, groups);
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool transposed,
    const std::vector<int64_t>& output_padding, int64_t groups,
    const std::vector<bool>& output_mask) {
    return convolution_backward_kernel(grad_output, input, weight, std::nullopt, stride, padding,
                                      dilation, transposed, output_padding, groups, output_mask);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUConvKernels) {
    m.impl("conv1d", static_cast<sig::conv1d*>(&conv1d_kernel));
    m.impl("conv1d_grad_input", static_cast<sig::conv1d_grad_input*>(&conv1d_grad_input_kernel));
    m.impl("conv1d_grad_weight", static_cast<sig::conv1d_grad_weight*>(&conv1d_grad_weight_kernel));
    m.impl("conv1d_grad_bias", static_cast<sig::conv1d_grad_bias*>(&conv1d_grad_bias_kernel));
    m.impl("conv2d", static_cast<sig::conv2d*>(&conv2d_kernel));
    m.impl("conv2d_grad_input", static_cast<sig::conv2d_grad_input*>(&conv2d_grad_input_kernel));
    m.impl("conv2d_grad_weight", static_cast<sig::conv2d_grad_weight*>(&conv2d_grad_weight_kernel));
    m.impl("conv2d_grad_bias", static_cast<sig::conv2d_grad_bias*>(&conv2d_grad_bias_kernel));
    m.impl("conv3d", static_cast<sig::conv3d*>(&conv3d_kernel));
    m.impl("conv3d_grad_input", static_cast<sig::conv3d_grad_input*>(&conv3d_grad_input_kernel));
    m.impl("conv3d_grad_weight", static_cast<sig::conv3d_grad_weight*>(&conv3d_grad_weight_kernel));
    m.impl("conv3d_grad_bias", static_cast<sig::conv3d_grad_bias*>(&conv3d_grad_bias_kernel));
    m.impl("conv_transpose1d", static_cast<sig::conv_transpose1d*>(&conv_transpose1d_kernel));
    m.impl("conv_transpose1d_grad_input",
           static_cast<sig::conv_transpose1d_grad_input*>(&conv_transpose1d_grad_input_kernel));
    m.impl("conv_transpose1d_grad_weight",
           static_cast<sig::conv_transpose1d_grad_weight*>(&conv_transpose1d_grad_weight_kernel));
    m.impl("conv_transpose1d_grad_bias",
           static_cast<sig::conv_transpose1d_grad_bias*>(&conv_transpose1d_grad_bias_kernel));
    m.impl("conv_transpose2d", static_cast<sig::conv_transpose2d*>(&conv_transpose2d_kernel));
    m.impl("conv_transpose2d_grad_input",
           static_cast<sig::conv_transpose2d_grad_input*>(&conv_transpose2d_grad_input_kernel));
    m.impl("conv_transpose2d_grad_weight",
           static_cast<sig::conv_transpose2d_grad_weight*>(&conv_transpose2d_grad_weight_kernel));
    m.impl("conv_transpose2d_grad_bias",
           static_cast<sig::conv_transpose2d_grad_bias*>(&conv_transpose2d_grad_bias_kernel));
    m.impl("conv_transpose3d", static_cast<sig::conv_transpose3d*>(&conv_transpose3d_kernel));
    m.impl("conv_transpose3d_grad_input",
           static_cast<sig::conv_transpose3d_grad_input*>(&conv_transpose3d_grad_input_kernel));
    m.impl("conv_transpose3d_grad_weight",
           static_cast<sig::conv_transpose3d_grad_weight*>(&conv_transpose3d_grad_weight_kernel));
    m.impl("conv_transpose3d_grad_bias",
           static_cast<sig::conv_transpose3d_grad_bias*>(&conv_transpose3d_grad_bias_kernel));
    m.impl("convolution", static_cast<sig::convolution*>(&convolution_kernel));
    m.impl("convolution_backward",
           static_cast<sig::convolution_backward*>(&convolution_backward_kernel));
    m.impl("convolution_overrideable",
           static_cast<sig::convolution_overrideable*>(&convolution_overrideable_kernel));
    m.impl("convolution_backward_overrideable",
           static_cast<sig::convolution_backward_overrideable*>(
               &convolution_backward_overrideable_kernel));
}

} // namespace mlu
} // namespace tensorplay
