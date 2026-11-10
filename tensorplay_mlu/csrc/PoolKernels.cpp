// Pooling on CNNL: the sliding-window forms (maximum with the winning
// positions, average) and the adaptive forms, each with the backward pass
// that scatters or spreads an output gradient.
//
// The card pools 4-D and 5-D channel-last operands, so an operand is carried
// across in the channel-last order and the result comes back channel-first.
// A maximum reports its winning position as the row-major offset inside the
// pooling window, while the operator carries the offset inside the input
// plane; the positions are rewritten between the two conventions.
// An unbatched operand (a 2-D pooling over a 3-D tensor, or a 3-D pooling
// over a 4-D one) runs as the batched call with a unit batch axis, which is
// the same operator on the same data.  Every kernel returns nothing when the
// shape, the type or the device leaves it without a CNNL kernel, and the
// operator then runs on the host.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
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

// A pooling call's geometry once its spatial parameters are expanded to one
// entry per spatial dimension.
struct PoolGeometry {
    int dim = 4;  // 4 for a 2-D pooling, 5 for a 3-D one
    std::vector<int> kernel;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    bool ceil_mode = false;

    int spatial() const { return dim - 2; }
};

// The spatial extents of a channel-first shape (its entries after batch and
// channel).
std::vector<int64_t> spatial_of(const std::vector<int64_t>& sizes) {
    return std::vector<int64_t>(sizes.begin() + 2, sizes.end());
}

// Division truncated toward negative infinity.
int64_t div_floor(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// The extent one spatial axis pools to.  A ceil-mode window that would start
// inside the operand's padding may fold back.
int64_t pooled_extent(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil,
                      bool ceil_mode) {
    int64_t out = div_floor(in + 2 * pad - dil * (k - 1) - 1 + (ceil_mode ? stride - 1 : 0),
                            stride) + 1;
    if (ceil_mode && (out - 1) * stride >= in + pad) --out;
    return out;
}

// Validates a call's tensor and parameters and returns the normalized
// geometry, or nothing when the card cannot run the call.
std::optional<PoolGeometry> prepare_pool(const Tensor& input,
                                         const std::vector<int64_t>& kernel_size,
                                         const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (!input.device().is_privateuse1()) return std::nullopt;
    if (input.dim() != 4 && input.dim() != 5) return std::nullopt;
    if (!is_native_float(input.dtype(), index_of(input.device()))) return std::nullopt;
    if (kernel_size.empty()) return std::nullopt;

    const int n = static_cast<int>(input.dim()) - 2;
    auto k = spread(kernel_size, n, 1);
    auto s = spread(stride.empty() ? kernel_size : stride, n, 1);
    auto p = spread(padding, n, 0);
    auto d = spread(dilation, n, 1);
    if (!k || !s || !p || !d) return std::nullopt;

    PoolGeometry g;
    g.dim = static_cast<int>(input.dim());
    g.kernel = *k;
    g.stride = *s;
    g.padding = *p;
    g.dilation = *d;
    g.ceil_mode = ceil_mode;
    for (int i = 0; i < n; ++i) {
        const size_t u = static_cast<size_t>(i);
        if (g.kernel[u] <= 0 || g.stride[u] <= 0 || g.padding[u] < 0 || g.dilation[u] < 1) {
            return std::nullopt;
        }
    }
    if (input.size(0) <= 0 || input.size(1) <= 0) return std::nullopt;
    for (int i = 0; i < n; ++i) {
        if (input.size(2 + i) <= 0) return std::nullopt;
    }
    return g;
}

// Output shape of a sliding-window pooling; nothing when an extent is
// non-positive.
std::optional<std::vector<int64_t>> pooled_sizes(const Tensor& input, const PoolGeometry& g) {
    std::vector<int64_t> sizes = {input.size(0), input.size(1)};
    for (int i = 0; i < g.spatial(); ++i) {
        const size_t u = static_cast<size_t>(i);
        const int64_t out = pooled_extent(input.size(2 + i), g.kernel[u], g.padding[u],
                                          g.stride[u], g.dilation[u], g.ceil_mode);
        if (out <= 0) return std::nullopt;
        sizes.push_back(out);
    }
    return sizes;
}

// ============================================================== descriptors

// Owning pooling descriptor, built for a 4-D or 5-D channel-last geometry.
class PoolOpDesc {
public:
    PoolOpDesc() = default;
    PoolOpDesc(const PoolOpDesc&) = delete;
    PoolOpDesc& operator=(const PoolOpDesc&) = delete;
    ~PoolOpDesc() {
        if (desc_ != nullptr) cnnlDestroyPoolingDescriptor(desc_);
    }

    // False when the card declines the geometry.
    bool build(const PoolGeometry& g, cnnlPoolingMode_t mode) {
        TP_CNNL_CHECK(cnnlCreatePoolingDescriptor(&desc_));
        if (g.dim == 4) {
            TP_CNNL_TRY(cnnlSetPooling2dDescriptor_v2(
                desc_, mode, CNNL_PROPAGATE_NAN, g.kernel[0], g.kernel[1], g.padding[0],
                g.padding[0], g.padding[1], g.padding[1], g.stride[0], g.stride[1],
                g.dilation[0], g.dilation[1], g.ceil_mode));
        } else {
            // Two padding entries per spatial axis: the amount concatenated at
            // the start and at the end, equal for a symmetric padding.
            std::vector<int> pad(static_cast<size_t>(2 * g.spatial()));
            for (int i = 0; i < g.spatial(); ++i) {
                pad[static_cast<size_t>(2 * i)] = g.padding[static_cast<size_t>(i)];
                pad[static_cast<size_t>(2 * i + 1)] = g.padding[static_cast<size_t>(i)];
            }
            TP_CNNL_TRY(cnnlSetPoolingNdDescriptor_v2(
                desc_, mode, CNNL_NOT_PROPAGATE_NAN, g.dim, g.kernel.data(), pad.data(),
                g.stride.data(), g.dilation.data(), g.ceil_mode));
        }
        return true;
    }

    cnnlPoolingDescriptor_t get() const { return desc_; }

private:
    cnnlPoolingDescriptor_t desc_ = nullptr;
};

// =========================================================== index windows

// For each spatial axis, where a pooling window of each output coordinate
// starts (its first covered input coordinate), how many entries it covers,
// and the input distance between two adjacent entries.  A sliding-window
// operator has a fixed extent with a dilated step; an adaptive one reads the
// window off the two extents.
struct IndexWindows {
    int sp = 2;
    std::vector<std::vector<int64_t>> origin;
    std::vector<std::vector<int64_t>> extent;
    std::vector<int64_t> step;
};

IndexWindows sliding_windows(const PoolGeometry& g, const std::vector<int64_t>& out_dim) {
    IndexWindows w;
    w.sp = g.spatial();
    for (int i = 0; i < w.sp; ++i) {
        const size_t n = static_cast<size_t>(out_dim[static_cast<size_t>(i)]);
        std::vector<int64_t> origin(n), extent(n);
        for (size_t c = 0; c < n; ++c) {
            origin[c] = static_cast<int64_t>(c) * g.stride[static_cast<size_t>(i)] -
                        g.padding[static_cast<size_t>(i)];
            extent[c] = g.kernel[static_cast<size_t>(i)];
        }
        w.origin.push_back(std::move(origin));
        w.extent.push_back(std::move(extent));
        w.step.push_back(g.dilation[static_cast<size_t>(i)]);
    }
    return w;
}

IndexWindows adaptive_windows(const std::vector<int64_t>& in_dim,
                              const std::vector<int64_t>& out_dim) {
    IndexWindows w;
    w.sp = static_cast<int>(in_dim.size());
    for (int i = 0; i < w.sp; ++i) {
        const int64_t in = in_dim[static_cast<size_t>(i)];
        const int64_t out = out_dim[static_cast<size_t>(i)];
        std::vector<int64_t> origin(static_cast<size_t>(out)), extent(static_cast<size_t>(out));
        for (int64_t c = 0; c < out; ++c) {
            const int64_t start = (c * in) / out;
            const int64_t end = 1 + (((c + 1) * in) - 1) / out;
            origin[static_cast<size_t>(c)] = start;
            extent[static_cast<size_t>(c)] = end - start;
        }
        w.origin.push_back(std::move(origin));
        w.extent.push_back(std::move(extent));
        w.step.push_back(1);
    }
    return w;
}

// Rewrite a batch of maximum-pooling positions.  The card reports a position
// as the row-major offset inside its pooling window, and the operator carries
// the offset inside the input plane.  ``src`` and ``dst`` are dense int64
// buffers of the pooled shape ``out_sizes``; with ``to_plane`` set, ``src``
// is channel last (the card's layout) and ``dst`` channel first, and the
// other way round otherwise.
void translate_indices(const Tensor& src, const Tensor& dst, const IndexWindows& w,
                       const std::vector<int64_t>& in_dim, const std::vector<int64_t>& out_sizes,
                       bool to_plane) {
    const int64_t total = src.numel();
    if (total == 0) return;
    const int index = index_of(src.device());
    DeviceGuard guard(index);
    std::vector<int64_t> raw(static_cast<size_t>(total));
    copy_to_host(raw.data(), src.data_ptr(), static_cast<size_t>(total) * sizeof(int64_t), index);

    const int sp = w.sp;
    const std::vector<int64_t> out_dim = spatial_of(out_sizes);
    const int64_t n_batch = out_sizes[0];
    const int64_t channels = out_sizes[1];
    int64_t count = 1;
    for (int i = 0; i < sp; ++i) count *= out_dim[static_cast<size_t>(i)];

    std::vector<int64_t> coord(static_cast<size_t>(sp));
    std::vector<int64_t> part(static_cast<size_t>(sp));
    std::vector<int64_t> out(raw.size());
    for (int64_t n = 0; n < n_batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const int64_t plane_base = (n * channels + c) * count;
            for (int64_t linear = 0; linear < count; ++linear) {
                int64_t rem = linear;
                for (int i = sp - 1; i >= 0; --i) {
                    const size_t u = static_cast<size_t>(i);
                    coord[u] = rem % out_dim[u];
                    rem /= out_dim[u];
                }
                int64_t window_off = 0;
                for (int i = 0; i < sp; ++i) {
                    window_off = window_off * out_dim[static_cast<size_t>(i)] +
                                 coord[static_cast<size_t>(i)];
                }
                // Channel-last row-major offset over the whole batch: the
                // batch's own block of windows precedes the spatial part.
                window_off = (n * count + window_off) * channels + c;
                const size_t src_at =
                    static_cast<size_t>(to_plane ? window_off : plane_base + linear);

                int64_t value = 0;
                if (to_plane) {
                    int64_t krem = raw[src_at];
                    for (int i = sp - 1; i >= 0; --i) {
                        const size_t u = static_cast<size_t>(i);
                        const int64_t ext = w.extent[u][static_cast<size_t>(coord[u])];
                        part[u] = krem % ext;
                        krem /= ext;
                    }
                    for (int i = 0; i < sp; ++i) {
                        const size_t u = static_cast<size_t>(i);
                        value = value * in_dim[u] + w.origin[u][static_cast<size_t>(coord[u])] +
                                part[u] * w.step[u];
                    }
                } else {
                    int64_t prem = raw[src_at];
                    for (int i = sp - 1; i >= 0; --i) {
                        const size_t u = static_cast<size_t>(i);
                        part[u] = prem % in_dim[u];
                        prem /= in_dim[u];
                    }
                    for (int i = 0; i < sp; ++i) {
                        const size_t u = static_cast<size_t>(i);
                        const int64_t rel =
                            (part[u] - w.origin[u][static_cast<size_t>(coord[u])]) / w.step[u];
                        value = value * w.extent[u][static_cast<size_t>(coord[u])] + rel;
                    }
                }
                out[static_cast<size_t>(to_plane ? plane_base + linear : window_off)] = value;
            }
        }
    }
    copy_to_device(dst.data_ptr(), out.data(), static_cast<size_t>(total) * sizeof(int64_t), index);
}

// ================================================================= forward

// y = maximum pooling of ``x``; ``idx`` receives the winning positions in the
// channel-last shape, as the card reports them.
bool launch_max_forward(const Tensor& x, const PoolGeometry& g, const Tensor& y,
                        const Tensor& idx) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x.sizes().vec(), channel_layout(g.dim), x.dtype());
    TensorDesc y_desc(y.sizes().vec(), channel_layout(g.dim), y.dtype());
    TensorDesc i_desc(idx.sizes().vec(), channel_layout(g.dim), DType::Int64);

    PoolOpDesc desc;
    if (!desc.build(g, CNNL_POOLING_MAX)) return false;

    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetPoolingWithIndexWorkspaceSize_v2(h, x_desc, y_desc, i_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlPoolingForwardWithIndex(h, desc.get(), nullptr, x_desc, x.data_ptr(),
                                            nullptr, y_desc, y.data_ptr(), i_desc,
                                            idx.data_ptr(), ws_size == 0 ? nullptr : ws.get(),
                                            ws_size));
    return true;
}

// y = average pooling of ``x``.
bool launch_avg_forward(const Tensor& x, const PoolGeometry& g, bool count_include_pad,
                        const Tensor& y) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x.sizes().vec(), channel_layout(g.dim), x.dtype());
    TensorDesc y_desc(y.sizes().vec(), channel_layout(g.dim), y.dtype());
    const cnnlPoolingMode_t mode = count_include_pad
        ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    PoolOpDesc desc;
    if (!desc.build(g, mode)) return false;

    // Excluding the padding divides by the covered window rather than by the
    // window's full area.  The card derives those per-position weights once on
    // the host into a small table, empty for the counting mode and the 3-D
    // form.
    size_t extra_size = 0;
    if (g.dim == 4) {
        // ``y`` is carried in the channel-last order, so its height and width
        // sit on axes 1 and 2; the query takes the width first.
        TP_CNNL_TRY(cnnlGetPoolingExtraInputSize(h, mode, static_cast<int>(y.size(2)),
                                                static_cast<int>(y.size(1)), &extra_size));
    }
    Workspace extra(extra_size, index);
    void* extra_ptr = nullptr;
    if (extra_size > 0) {
        std::vector<float> host_table(extra_size / sizeof(float));
        TP_CNNL_TRY(cnnlInitPoolingExtraInput(h, desc.get(), x_desc, y_desc, host_table.data()));
        copy_to_device(extra.get(), host_table.data(), extra_size, index);
        extra_ptr = extra.get();
    }

    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetPoolingWorkspaceSize_v2(h, desc.get(), x_desc, y_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlPoolingForward_v2(h, desc.get(), nullptr, x_desc, x.data_ptr(), nullptr,
                                      extra_ptr, y_desc, y.data_ptr(),
                                      ws_size == 0 ? nullptr : ws.get(), ws_size));
    return true;
}

std::optional<std::tuple<Tensor, Tensor>> max_pool_core(const Tensor& input,
                                                        const PoolGeometry& g) {
    auto out_sizes = pooled_sizes(input, g);
    if (!out_sizes) return std::nullopt;
    const int index = index_of(input.device());
    Tensor y_cl = empty_on(to_channel_last_sizes(*out_sizes), input.dtype(), index);
    Tensor raw_cl = empty_on(to_channel_last_sizes(*out_sizes), DType::Int64, index);
    if (y_cl.numel() > 0) {
        if (!launch_max_forward(channel_last(input), g, y_cl, raw_cl)) return std::nullopt;
    }
    Tensor idx = empty_on(*out_sizes, DType::Int64, index);
    if (idx.numel() > 0) {
        const IndexWindows w = sliding_windows(g, spatial_of(*out_sizes));
        translate_indices(raw_cl, idx, w, spatial_of(input.sizes().vec()), *out_sizes, true);
    }
    return std::make_tuple(channel_first(y_cl), idx);
}

std::optional<Tensor> avg_pool_core(const Tensor& input, const PoolGeometry& g,
                                    bool count_include_pad) {
    auto sizes = pooled_sizes(input, g);
    if (!sizes) return std::nullopt;
    const int index = index_of(input.device());
    Tensor y = empty_on(to_channel_last_sizes(*sizes), input.dtype(), index);
    if (y.numel() > 0) {
        if (!launch_avg_forward(channel_last(input), g, count_include_pad, y)) {
            return std::nullopt;
        }
    }
    return channel_first(y);
}

// ================================================================ backward

// diff_x = gradient of a sliding-window pooling of ``x``.  ``idx`` is the
// winning positions for a maximum and absent for an average.
bool launch_pool_backward(const Tensor& x, const Tensor& diff_y, const Tensor* idx,
                          const PoolGeometry& g, cnnlPoolingMode_t mode,
                          const Tensor& diff_x) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x.sizes().vec(), channel_layout(g.dim), x.dtype());
    TensorDesc dy_desc(diff_y.sizes().vec(), channel_layout(g.dim), diff_y.dtype());
    TensorDesc dx_desc(diff_x.sizes().vec(), channel_layout(g.dim), diff_x.dtype());
    // An index descriptor is part of the workspace query even when the
    // operator scatters no positions; its shape is the pooled output's.
    TensorDesc i_desc(diff_y.sizes().vec(), channel_layout(g.dim), DType::Int64);

    PoolOpDesc desc;
    if (!desc.build(g, mode)) return false;

    const void* i_ptr = idx != nullptr ? idx->data_ptr() : nullptr;
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetPoolingBackwardWorkspaceSize(h, desc.get(), i_desc, dy_desc, x_desc,
                                                    dx_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlPoolingBackward_v2(h, desc.get(), nullptr, i_desc, i_ptr, dy_desc,
                                       diff_y.data_ptr(), x_desc, x.data_ptr(), nullptr, dx_desc,
                                       diff_x.data_ptr(), ws_size == 0 ? nullptr : ws.get(),
                                       ws_size));
    return true;
}

std::optional<Tensor> pool_backward_core(const Tensor& grad_output, const Tensor& input,
                                         const PoolGeometry& g, bool count_include_pad,
                                         const Tensor* indices) {
    const int index = index_of(input.device());
    if (!grad_output.device().is_privateuse1() || index_of(grad_output.device()) != index) {
        return std::nullopt;
    }
    if (grad_output.dtype() != input.dtype()) return std::nullopt;
    if (indices != nullptr && !indices->device().is_privateuse1()) return std::nullopt;

    Tensor dx = empty_on(channel_last_sizes(input), input.dtype(), index);
    if (dx.numel() > 0) {
        Tensor idx_cl;
        if (indices != nullptr) {
            const std::vector<int64_t> out_sizes = grad_output.sizes().vec();
            if (indices->sizes().vec() != out_sizes) return std::nullopt;
            idx_cl = empty_on(to_channel_last_sizes(out_sizes), DType::Int64, index);
            if (idx_cl.numel() > 0) {
                const IndexWindows w = sliding_windows(g, spatial_of(out_sizes));
                translate_indices(contiguous_on_device(*indices), idx_cl, w,
                                  spatial_of(input.sizes().vec()), out_sizes, false);
            }
        }
        const cnnlPoolingMode_t mode = indices != nullptr
            ? CNNL_POOLING_MAX
            : (count_include_pad ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                 : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
        if (!launch_pool_backward(channel_last(input), channel_last(grad_output),
                                  indices != nullptr ? &idx_cl : nullptr, g, mode, dx)) {
            return std::nullopt;
        }
    }
    return channel_first(dx);
}

// ======================================================== adaptive pooling

// Output shape of an adaptive pooling; nothing when the requested extents do
// not match the operand's spatial rank.
std::optional<std::vector<int64_t>> adaptive_sizes(const Tensor& input,
                                                   const std::vector<int64_t>& output_size) {
    if (input.dim() != 4 && input.dim() != 5) return std::nullopt;
    const int n = static_cast<int>(input.dim()) - 2;
    if (static_cast<int>(output_size.size()) != n) return std::nullopt;
    std::vector<int64_t> sizes = {input.size(0), input.size(1)};
    for (int64_t out : output_size) {
        if (out <= 0) return std::nullopt;
        sizes.push_back(out);
    }
    return sizes;
}

// The position type the card writes and reads for an adaptive maximum: 16-bit
// positions for a half or bfloat16 operand, 32-bit ones otherwise.
DType position_dtype(DType operand) {
    return (operand == DType::Float16 || operand == DType::BFloat16) ? DType::Int16
                                                                     : DType::Int32;
}

// y = adaptive pooling of ``x``; ``idx`` (if any) receives the winning
// positions in ``idx_dtype``.
bool launch_adaptive_forward(const Tensor& x, const Tensor& y, cnnlPoolingMode_t mode,
                             const Tensor* idx, DType idx_dtype) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x.sizes().vec(), channel_layout(x.dim()), x.dtype());
    TensorDesc y_desc(y.sizes().vec(), channel_layout(y.dim()), y.dtype());
    TensorDesc i_desc;
    void* i_ptr = nullptr;
    if (idx != nullptr) {
        i_desc = TensorDesc(idx->sizes().vec(), channel_layout(idx->dim()), idx_dtype);
        i_ptr = idx->data_ptr();
    }
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetAdaptivePoolingForwardWorkspaceSize(h, x_desc, mode, y_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlAdaptivePoolingForward_v2(h, x_desc, x.data_ptr(), mode,
                                              ws_size == 0 ? nullptr : ws.get(), ws_size,
                                              y_desc, y.data_ptr(),
                                              idx != nullptr ? i_desc.get() : nullptr, i_ptr));
    return true;
}

// diff_x = gradient of an adaptive pooling of ``x``; ``idx`` is the winning
// positions for a maximum and absent for an average.
bool launch_adaptive_backward(const Tensor& diff_y, const Tensor* idx, DType idx_dtype,
                              cnnlPoolingMode_t mode, const Tensor& diff_x) {
    const int index = index_of(diff_x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc dy_desc(diff_y.sizes().vec(), channel_layout(diff_y.dim()), diff_y.dtype());
    TensorDesc dx_desc(diff_x.sizes().vec(), channel_layout(diff_x.dim()), diff_x.dtype());
    TensorDesc i_desc;
    const void* i_ptr = nullptr;
    if (idx != nullptr) {
        i_desc = TensorDesc(idx->sizes().vec(), channel_layout(idx->dim()), idx_dtype);
        i_ptr = idx->data_ptr();
    }
    TP_CNNL_TRY(cnnlAdaptivePoolingBackward(h, dy_desc, diff_y.data_ptr(),
                                            idx != nullptr ? i_desc.get() : nullptr, i_ptr,
                                            mode, dx_desc, diff_x.data_ptr()));
    return true;
}

std::optional<Tensor> adaptive_avg_core(const Tensor& input,
                                        const std::vector<int64_t>& output_size) {
    if (!input.device().is_privateuse1()) return std::nullopt;
    const int index = index_of(input.device());
    if (!is_native_float(input.dtype(), index)) return std::nullopt;
    auto sizes = adaptive_sizes(input, output_size);
    if (!sizes) return std::nullopt;
    Tensor y = empty_on(to_channel_last_sizes(*sizes), input.dtype(), index);
    if (y.numel() > 0) {
        if (!launch_adaptive_forward(channel_last(input), y,
                                     CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, nullptr,
                                     DType::Int32)) {
            return std::nullopt;
        }
    }
    return channel_first(y);
}

std::optional<std::tuple<Tensor, Tensor>> adaptive_max_core(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    if (!input.device().is_privateuse1()) return std::nullopt;
    const int index = index_of(input.device());
    if (!is_native_float(input.dtype(), index)) return std::nullopt;
    // The card's adaptive maximum, which is the only form that reports
    // positions, is 2-D.
    if (input.dim() != 4) return std::nullopt;
    auto out_sizes = adaptive_sizes(input, output_size);
    if (!out_sizes) return std::nullopt;
    const DType narrow = position_dtype(input.dtype());
    Tensor y_cl = empty_on(to_channel_last_sizes(*out_sizes), input.dtype(), index);
    Tensor positions = empty_on(to_channel_last_sizes(*out_sizes), narrow, index);
    if (y_cl.numel() > 0) {
        if (!launch_adaptive_forward(channel_last(input), y_cl, CNNL_POOLING_MAX, &positions,
                                     narrow)) {
            return std::nullopt;
        }
    }
    Tensor idx = empty_on(*out_sizes, DType::Int64, index);
    if (idx.numel() > 0) {
        const IndexWindows w =
            adaptive_windows(spatial_of(input.sizes().vec()), spatial_of(*out_sizes));
        translate_indices(to_dtype(positions, DType::Int64), idx, w,
                          spatial_of(input.sizes().vec()), *out_sizes, true);
    }
    return std::make_tuple(channel_first(y_cl), idx);
}

// dx = gradient of an adaptive average pooling.
std::optional<Tensor> adaptive_avg_backward_core(const Tensor& grad_output,
                                                 const Tensor& input) {
    if (!input.device().is_privateuse1() || !grad_output.device().is_privateuse1()) {
        return std::nullopt;
    }
    const int index = index_of(input.device());
    if (index_of(grad_output.device()) != index) return std::nullopt;
    if (input.dim() != 4 && input.dim() != 5) return std::nullopt;
    if (grad_output.dtype() != input.dtype() || !is_native_float(input.dtype(), index)) {
        return std::nullopt;
    }
    Tensor dx = empty_on(channel_last_sizes(input), input.dtype(), index);
    if (dx.numel() > 0) {
        if (!launch_adaptive_backward(channel_last(grad_output), nullptr, DType::Int32,
                                      CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, dx)) {
            return std::nullopt;
        }
    }
    return channel_first(dx);
}

// dx = gradient of an adaptive maximum pooling, scattered at ``indices``.
std::optional<Tensor> adaptive_max_backward_core(const Tensor& grad_output, const Tensor& input,
                                                 const Tensor& indices) {
    if (!input.device().is_privateuse1() || !grad_output.device().is_privateuse1() ||
        !indices.device().is_privateuse1()) {
        return std::nullopt;
    }
    const int index = index_of(input.device());
    if (index_of(grad_output.device()) != index || index_of(indices.device()) != index) {
        return std::nullopt;
    }
    if (input.dim() != 4) return std::nullopt;
    if (grad_output.dtype() != input.dtype() || !is_native_float(input.dtype(), index)) {
        return std::nullopt;
    }
    const std::vector<int64_t> out_sizes = grad_output.sizes().vec();
    if (indices.sizes().vec() != out_sizes) return std::nullopt;

    Tensor dx = empty_on(channel_last_sizes(input), input.dtype(), index);
    if (dx.numel() > 0) {
        Tensor wide_cl = empty_on(to_channel_last_sizes(out_sizes), DType::Int64, index);
        if (wide_cl.numel() > 0) {
            const IndexWindows w =
                adaptive_windows(spatial_of(input.sizes().vec()), spatial_of(out_sizes));
            translate_indices(contiguous_on_device(indices), wide_cl, w,
                              spatial_of(input.sizes().vec()), out_sizes, false);
        }
        const DType narrow = position_dtype(input.dtype());
        Tensor positions = to_dtype(wide_cl, narrow);
        if (!launch_adaptive_backward(channel_last(grad_output), &positions, narrow,
                                      CNNL_POOLING_MAX, dx)) {
            return std::nullopt;
        }
    }
    return channel_first(dx);
}

// ============================================================ batch axis

// A 2-D pooling over a 3-D operand, or a 3-D one over a 4-D operand, is the
// batched call with a unit batch axis: this adds that axis to each operand and
// drops it from the result.
class BatchAxis {
public:
    explicit BatchAxis(bool active) : active_(active) {}

    Tensor lift(const Tensor& t) const { return active_ ? t.unsqueeze(0) : t; }
    Tensor lower(const Tensor& t) const { return active_ ? t.squeeze(0) : t; }

private:
    bool active_;
};

// ================================================================ kernels

std::tuple<Tensor, Tensor> max_pool2d_with_indices_kernel(const Tensor& input,
                                                          const std::vector<int64_t>& kernel_size,
                                                          const std::vector<int64_t>& stride,
                                                          const std::vector<int64_t>& padding,
                                                          const std::vector<int64_t>& dilation,
                                                          bool ceil_mode) {
    const BatchAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    if (auto g = prepare_pool(x, kernel_size, stride, padding, dilation, ceil_mode)) {
        if (auto r = max_pool_core(x, *g)) {
            return {axis.lower(std::get<0>(*r)), axis.lower(std::get<1>(*r))};
        }
    }
    return cpu_fallback::Kernel<"max_pool2d_with_indices",
                                sig::max_pool2d_with_indices>::call(input, kernel_size, stride,
                                                                    padding, dilation, ceil_mode);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_kernel(const Tensor& input,
                                                          const std::vector<int64_t>& kernel_size,
                                                          const std::vector<int64_t>& stride,
                                                          const std::vector<int64_t>& padding,
                                                          const std::vector<int64_t>& dilation,
                                                          bool ceil_mode) {
    const BatchAxis axis(input.dim() == 4);
    const Tensor x = axis.lift(input);
    if (auto g = prepare_pool(x, kernel_size, stride, padding, dilation, ceil_mode)) {
        if (auto r = max_pool_core(x, *g)) {
            return {axis.lower(std::get<0>(*r)), axis.lower(std::get<1>(*r))};
        }
    }
    return cpu_fallback::Kernel<"max_pool3d_with_indices",
                                sig::max_pool3d_with_indices>::call(input, kernel_size, stride,
                                                                    padding, dilation, ceil_mode);
}

Tensor max_pool2d_with_indices_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                               const std::vector<int64_t>& kernel_size,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& padding,
                                               const std::vector<int64_t>& dilation, bool ceil_mode,
                                               const std::optional<Tensor>& indices) {
    const BatchAxis axis(input.dim() == 3);
    if (indices.has_value() && indices->defined()) {
        const Tensor x = axis.lift(input);
        if (auto g = prepare_pool(x, kernel_size, stride, padding, dilation, ceil_mode)) {
            const Tensor idx = axis.lift(*indices);
            if (auto r = pool_backward_core(axis.lift(grad_output), x, *g, true, &idx)) {
                return axis.lower(*r);
            }
        }
    }
    return cpu_fallback::Kernel<"max_pool2d_with_indices_backward",
                                sig::max_pool2d_with_indices_backward>::call(
        grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

Tensor max_pool3d_with_indices_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                               const std::vector<int64_t>& kernel_size,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& padding,
                                               const std::vector<int64_t>& dilation, bool ceil_mode,
                                               const std::optional<Tensor>& indices) {
    const BatchAxis axis(input.dim() == 4);
    if (indices.has_value() && indices->defined()) {
        const Tensor x = axis.lift(input);
        if (auto g = prepare_pool(x, kernel_size, stride, padding, dilation, ceil_mode)) {
            const Tensor idx = axis.lift(*indices);
            if (auto r = pool_backward_core(axis.lift(grad_output), x, *g, true, &idx)) {
                return axis.lower(*r);
            }
        }
    }
    return cpu_fallback::Kernel<"max_pool3d_with_indices_backward",
                                sig::max_pool3d_with_indices_backward>::call(
        grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

// The window form of a maximum's gradient: the winning positions are found
// again on this input and the output gradient is scattered at them.
Tensor max_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, bool ceil_mode) {
    const BatchAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    if (auto g = prepare_pool(x, kernel_size, stride, padding, dilation, ceil_mode)) {
        if (auto fwd = max_pool_core(x, *g)) {
            const Tensor idx = std::get<1>(*fwd);
            if (auto r = pool_backward_core(axis.lift(grad_output), x, *g, true, &idx)) {
                return axis.lower(*r);
            }
        }
    }
    return cpu_fallback::Kernel<"max_pool2d_backward", sig::max_pool2d_backward>::call(
        grad_output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor max_pool3d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, bool ceil_mode) {
    const BatchAxis axis(input.dim() == 4);
    const Tensor x = axis.lift(input);
    if (auto g = prepare_pool(x, kernel_size, stride, padding, dilation, ceil_mode)) {
        if (auto fwd = max_pool_core(x, *g)) {
            const Tensor idx = std::get<1>(*fwd);
            if (auto r = pool_backward_core(axis.lift(grad_output), x, *g, true, &idx)) {
                return axis.lower(*r);
            }
        }
    }
    return cpu_fallback::Kernel<"max_pool3d_backward", sig::max_pool3d_backward>::call(
        grad_output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor avg_pool2d_kernel(const Tensor& input, const std::vector<int64_t>& kernel_size,
                         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                         bool ceil_mode, bool count_include_pad,
                         std::optional<int64_t> divisor_override) {
    const BatchAxis axis(input.dim() == 3);
    if (!divisor_override.has_value()) {
        const Tensor x = axis.lift(input);
        if (auto g = prepare_pool(x, kernel_size, stride, padding, {}, ceil_mode)) {
            if (auto r = avg_pool_core(x, *g, count_include_pad)) return axis.lower(*r);
        }
    }
    return cpu_fallback::Kernel<"avg_pool2d", sig::avg_pool2d>::call(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor avg_pool3d_kernel(const Tensor& input, const std::vector<int64_t>& kernel_size,
                         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                         bool ceil_mode, bool count_include_pad,
                         std::optional<int64_t> divisor_override) {
    const BatchAxis axis(input.dim() == 4);
    if (!divisor_override.has_value()) {
        const Tensor x = axis.lift(input);
        if (auto g = prepare_pool(x, kernel_size, stride, padding, {}, ceil_mode)) {
            if (auto r = avg_pool_core(x, *g, count_include_pad)) return axis.lower(*r);
        }
    }
    return cpu_fallback::Kernel<"avg_pool3d", sig::avg_pool3d>::call(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

Tensor avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding, bool ceil_mode,
                                  bool count_include_pad, std::optional<int64_t> divisor_override) {
    const BatchAxis axis(input.dim() == 3);
    if (!divisor_override.has_value()) {
        const Tensor x = axis.lift(input);
        if (auto g = prepare_pool(x, kernel_size, stride, padding, {}, ceil_mode)) {
            if (auto r = pool_backward_core(axis.lift(grad_output), x, *g, count_include_pad,
                                            nullptr)) {
                return axis.lower(*r);
            }
        }
    }
    return cpu_fallback::Kernel<"avg_pool2d_backward", sig::avg_pool2d_backward>::call(
        grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad,
        divisor_override);
}

Tensor avg_pool3d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding, bool ceil_mode,
                                  bool count_include_pad, std::optional<int64_t> divisor_override) {
    const BatchAxis axis(input.dim() == 4);
    if (!divisor_override.has_value()) {
        const Tensor x = axis.lift(input);
        if (auto g = prepare_pool(x, kernel_size, stride, padding, {}, ceil_mode)) {
            if (auto r = pool_backward_core(axis.lift(grad_output), x, *g, count_include_pad,
                                            nullptr)) {
                return axis.lower(*r);
            }
        }
    }
    return cpu_fallback::Kernel<"avg_pool3d_backward", sig::avg_pool3d_backward>::call(
        grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad,
        divisor_override);
}

Tensor adaptive_avg_pool2d_kernel(const Tensor& input,
                                  const std::vector<int64_t>& output_size) {
    const BatchAxis axis(input.dim() == 3);
    if (auto r = adaptive_avg_core(axis.lift(input), output_size)) return axis.lower(*r);
    return cpu_fallback::Kernel<"adaptive_avg_pool2d", sig::adaptive_avg_pool2d>::call(
        input, output_size);
}

Tensor adaptive_avg_pool3d_kernel(const Tensor& input,
                                  const std::vector<int64_t>& output_size) {
    const BatchAxis axis(input.dim() == 4);
    if (auto r = adaptive_avg_core(axis.lift(input), output_size)) return axis.lower(*r);
    return cpu_fallback::Kernel<"adaptive_avg_pool3d", sig::adaptive_avg_pool3d>::call(
        input, output_size);
}

Tensor adaptive_avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input) {
    const BatchAxis axis(input.dim() == 3);
    if (auto r = adaptive_avg_backward_core(axis.lift(grad_output), axis.lift(input))) {
        return axis.lower(*r);
    }
    return cpu_fallback::Kernel<"adaptive_avg_pool2d_backward",
                                sig::adaptive_avg_pool2d_backward>::call(grad_output, input);
}

Tensor adaptive_avg_pool3d_backward_kernel(const Tensor& grad_output, const Tensor& input) {
    const BatchAxis axis(input.dim() == 4);
    if (auto r = adaptive_avg_backward_core(axis.lift(grad_output), axis.lift(input))) {
        return axis.lower(*r);
    }
    return cpu_fallback::Kernel<"adaptive_avg_pool3d_backward",
                                sig::adaptive_avg_pool3d_backward>::call(grad_output, input);
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices_kernel(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    const BatchAxis axis(input.dim() == 3);
    if (auto r = adaptive_max_core(axis.lift(input), output_size)) {
        return {axis.lower(std::get<0>(*r)), axis.lower(std::get<1>(*r))};
    }
    return cpu_fallback::Kernel<"adaptive_max_pool2d_with_indices",
                                sig::adaptive_max_pool2d_with_indices>::call(input, output_size);
}

Tensor adaptive_max_pool2d_with_indices_backward_kernel(const Tensor& grad_output,
                                                        const Tensor& input,
                                                        const std::vector<int64_t>& output_size,
                                                        const Tensor& indices) {
    const BatchAxis axis(input.dim() == 3);
    if (auto r = adaptive_max_backward_core(axis.lift(grad_output), axis.lift(input),
                                            axis.lift(indices))) {
        return axis.lower(*r);
    }
    return cpu_fallback::Kernel<"adaptive_max_pool2d_with_indices_backward",
                                sig::adaptive_max_pool2d_with_indices_backward>::call(
        grad_output, input, output_size, indices);
}

// The window form of an adaptive maximum's gradient: the winning positions
// are found again on this input and the output gradient is scattered at them.
Tensor adaptive_max_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input) {
    const BatchAxis axis(input.dim() == 3);
    const Tensor x = axis.lift(input);
    const Tensor gy = axis.lift(grad_output);
    if (x.dim() == 4) {
        const std::vector<int64_t> output_size = {gy.size(2), gy.size(3)};
        if (auto fwd = adaptive_max_core(x, output_size)) {
            const Tensor idx = std::get<1>(*fwd);
            if (auto r = adaptive_max_backward_core(gy, x, idx)) return axis.lower(*r);
        }
    }
    return cpu_fallback::Kernel<"adaptive_max_pool2d_backward",
                                sig::adaptive_max_pool2d_backward>::call(grad_output, input);
}

std::tuple<Tensor, Tensor> adaptive_max_pool3d_kernel(const Tensor& input,
                                                      const std::vector<int64_t>& output_size) {
    const BatchAxis axis(input.dim() == 4);
    if (auto r = adaptive_max_core(axis.lift(input), output_size)) {
        return {axis.lower(std::get<0>(*r)), axis.lower(std::get<1>(*r))};
    }
    return cpu_fallback::Kernel<"adaptive_max_pool3d", sig::adaptive_max_pool3d>::call(
        input, output_size);
}

Tensor adaptive_max_pool3d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& indices) {
    const BatchAxis axis(input.dim() == 4);
    if (auto r = adaptive_max_backward_core(axis.lift(grad_output), axis.lift(input),
                                            axis.lift(indices))) {
        return axis.lower(*r);
    }
    return cpu_fallback::Kernel<"adaptive_max_pool3d_backward",
                                sig::adaptive_max_pool3d_backward>::call(grad_output, input,
                                                                         indices);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUPoolKernels) {
    m.impl("max_pool2d_with_indices",
           static_cast<sig::max_pool2d_with_indices*>(&max_pool2d_with_indices_kernel));
    m.impl("max_pool3d_with_indices",
           static_cast<sig::max_pool3d_with_indices*>(&max_pool3d_with_indices_kernel));
    m.impl("max_pool2d_with_indices_backward",
           static_cast<sig::max_pool2d_with_indices_backward*>(
               &max_pool2d_with_indices_backward_kernel));
    m.impl("max_pool3d_with_indices_backward",
           static_cast<sig::max_pool3d_with_indices_backward*>(
               &max_pool3d_with_indices_backward_kernel));
    m.impl("max_pool2d_backward", static_cast<sig::max_pool2d_backward*>(&max_pool2d_backward_kernel));
    m.impl("max_pool3d_backward", static_cast<sig::max_pool3d_backward*>(&max_pool3d_backward_kernel));
    m.impl("avg_pool2d", static_cast<sig::avg_pool2d*>(&avg_pool2d_kernel));
    m.impl("avg_pool3d", static_cast<sig::avg_pool3d*>(&avg_pool3d_kernel));
    m.impl("avg_pool2d_backward", static_cast<sig::avg_pool2d_backward*>(&avg_pool2d_backward_kernel));
    m.impl("avg_pool3d_backward", static_cast<sig::avg_pool3d_backward*>(&avg_pool3d_backward_kernel));
    m.impl("adaptive_avg_pool2d",
           static_cast<sig::adaptive_avg_pool2d*>(&adaptive_avg_pool2d_kernel));
    m.impl("_adaptive_avg_pool2d",
           static_cast<sig::_adaptive_avg_pool2d*>(&adaptive_avg_pool2d_kernel));
    m.impl("adaptive_avg_pool3d",
           static_cast<sig::adaptive_avg_pool3d*>(&adaptive_avg_pool3d_kernel));
    m.impl("_adaptive_avg_pool3d",
           static_cast<sig::_adaptive_avg_pool3d*>(&adaptive_avg_pool3d_kernel));
    m.impl("adaptive_avg_pool2d_backward",
           static_cast<sig::adaptive_avg_pool2d_backward*>(&adaptive_avg_pool2d_backward_kernel));
    m.impl("_adaptive_avg_pool2d_backward",
           static_cast<sig::_adaptive_avg_pool2d_backward*>(
               &adaptive_avg_pool2d_backward_kernel));
    m.impl("adaptive_avg_pool3d_backward",
           static_cast<sig::adaptive_avg_pool3d_backward*>(&adaptive_avg_pool3d_backward_kernel));
    m.impl("_adaptive_avg_pool3d_backward",
           static_cast<sig::_adaptive_avg_pool3d_backward*>(
               &adaptive_avg_pool3d_backward_kernel));
    m.impl("adaptive_max_pool2d_with_indices",
           static_cast<sig::adaptive_max_pool2d_with_indices*>(
               &adaptive_max_pool2d_with_indices_kernel));
    m.impl("adaptive_max_pool2d_with_indices_backward",
           static_cast<sig::adaptive_max_pool2d_with_indices_backward*>(
               &adaptive_max_pool2d_with_indices_backward_kernel));
    m.impl("adaptive_max_pool2d_backward",
           static_cast<sig::adaptive_max_pool2d_backward*>(
               &adaptive_max_pool2d_backward_kernel));
    m.impl("adaptive_max_pool3d",
           static_cast<sig::adaptive_max_pool3d*>(&adaptive_max_pool3d_kernel));
    m.impl("adaptive_max_pool3d_backward",
           static_cast<sig::adaptive_max_pool3d_backward*>(&adaptive_max_pool3d_backward_kernel));
}

} // namespace mlu
} // namespace tensorplay
