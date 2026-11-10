// Indexing, scattering, prefix scans and where on CNNL.
//
// gather/scatter run the value math in float32, float16, bfloat16 (500-series
// cards) and int32; the prefix scans add the integer types.  Every kernel
// returns an empty result when a shape or type leaves it without a CNNL
// kernel, and the operator then runs on the host.

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

// Index tensors feed CNNL as int32 or int64.
bool native_index(const Tensor& t) {
    return t.device().is_privateuse1() &&
           (t.dtype() == DType::Int64 || t.dtype() == DType::Int32);
}

// The value types scatter and gather compute in on device ``index``.
bool value_dtype(DType dtype, int index) {
    if (dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::Int32) {
        return true;
    }
    return dtype == DType::BFloat16 && capability(index).first >= 5;
}

// The value types the prefix scans compute in; CNNL has no bfloat16 scan.
bool scan_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::Int32 ||
           dtype == DType::Int64;
}

int64_t resolve_dim(int64_t dim, int64_t nd) { return dim < 0 ? dim + nd : dim; }

std::optional<Tensor> gather_impl(const Tensor& self, int64_t dim, const Tensor& index) {
    const int dev = index_of(self.device());
    if (self.dim() == 0 || self.dim() > kMaxCnnlDims || !value_dtype(self.dtype(), dev) ||
        !native_index(index) || index_of(index.device()) != dev) {
        return std::nullopt;
    }
    const int64_t d = resolve_dim(dim, self.dim());
    if (d < 0 || d >= self.dim()) return std::nullopt;
    const Tensor x = contiguous_on_device(self);
    const Tensor idx = contiguous_on_device(index);
    Tensor out = empty_on(idx.sizes().vec(), self.dtype(), dev);
    if (x.numel() > 0 && idx.numel() > 0) {
        DeviceGuard guard(dev);
        TensorDesc x_desc(x), idx_desc(idx), out_desc(out);
        size_t ws_size = 0;
        TP_CNNL_TRY(cnnlGetGatherWorkspaceSize(handle(dev), x_desc, idx_desc, out_desc,
                                               static_cast<int>(d), &ws_size));
        Workspace ws(ws_size, dev);
        TP_CNNL_TRY(cnnlGather(handle(dev), static_cast<int>(d), x_desc, x.data_ptr(), idx_desc,
                               idx.data_ptr(), out_desc, out.data_ptr()));
    }
    return out;
}

// ``mode`` reduces ``src`` into ``base`` at ``index`` along ``dim``; for the
// replace mode ``with_self`` is irrelevant.  ``with_self`` false starts the
// reduction from zeros, matching scatter_reduce with include_self=False.
std::optional<Tensor> scatter_impl(const Tensor& self, int64_t dim, const Tensor& index,
                                   const Tensor& src, cnnlScatterMode_t mode, bool with_self) {
    const int dev = index_of(self.device());
    if (!self.device().is_privateuse1() || self.dim() == 0 || self.dim() > kMaxCnnlDims ||
        !native_index(index) || index_of(index.device()) != dev ||
        index_of(src.device()) != dev || src.dim() == 0) {
        return std::nullopt;
    }
    const int64_t d = resolve_dim(dim, self.dim());
    if (d < 0 || d >= self.dim()) return std::nullopt;
    const DType common = binary_result_dtype(self, src);
    if (!value_dtype(common, dev)) return std::nullopt;
    Tensor base = with_self || mode == CNNL_SCATTER
                      ? to_dtype(contiguous_on_device(self), common)
                      : empty_on(self.sizes().vec(), common, dev);
    const Tensor idx = contiguous_on_device(index);
    const Tensor val = to_dtype(contiguous_on_device(src), common);
    Tensor out = empty_on(self.sizes().vec(), common, dev);
    if (idx.numel() == 0) {
        strided_copy(out, base);
        return out;
    }
    DeviceGuard guard(dev);
    TensorDesc base_desc(base), idx_desc(idx), val_desc(val), out_desc(out);
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetScatterWorkspaceSize(handle(dev), idx_desc, &ws_size));
    Workspace ws(ws_size, dev);
    TP_CNNL_TRY(cnnlScatter(handle(dev), static_cast<int>(d), base_desc, base.data_ptr(),
                            idx_desc, idx.data_ptr(), val_desc, val.data_ptr(), out_desc,
                            out.data_ptr(), mode));
    return out;
}

cnnlScatterMode_t reduce_mode(const std::string& reduce, bool* supported) {
    *supported = true;
    if (reduce.empty()) return CNNL_SCATTER;
    if (reduce == "add" || reduce == "sum") return CNNL_SCATTER_ADD;
    if (reduce == "amax") return CNNL_SCATTER_MAX;
    if (reduce == "amin") return CNNL_SCATTER_MIN;
    *supported = false;
    return CNNL_SCATTER;
}

bool native_scatter(const Tensor& self, const Tensor& index, const Tensor& src) {
    const int dev = index_of(self.device());
    return self.device().is_privateuse1() && native_index(index) &&
           value_dtype(binary_result_dtype(self, src), dev);
}

// Prefix scan along ``dim``: cumulative sums or products, inclusive, in the
// forward direction.
std::optional<Tensor> scan_impl(const Tensor& self, int64_t dim, DType out_dtype, bool product) {
    const int dev = index_of(self.device());
    if (!self.device().is_privateuse1() || self.dim() == 0 || self.dim() > kMaxCnnlDims) {
        return std::nullopt;
    }
    const int64_t d = resolve_dim(dim, self.dim());
    if (d < 0 || d >= self.dim()) return std::nullopt;
    const Tensor x = contiguous_on_device(to_dtype(self, out_dtype));
    Tensor out = empty_on(x.sizes().vec(), out_dtype, dev);
    DeviceGuard guard(dev);
    TensorDesc x_desc(x), out_desc(out);
    if (product) {
        TP_CNNL_TRY(cnnlCumprod(handle(dev), x_desc, x.data_ptr(), static_cast<int>(d), false,
                                false, out_desc, out.data_ptr()));
    } else {
        size_t ws_size = 0;
        TP_CNNL_TRY(cnnlGetCumsumWorkspaceSize(handle(dev), x_desc, static_cast<int>(d),
                                               &ws_size));
        Workspace ws(ws_size, dev);
        TP_CNNL_TRY(cnnlCumsum_v2(handle(dev), x_desc, x.data_ptr(), static_cast<int>(d), false,
                                  false, CNNL_NOT_PROPAGATE_NAN, out_desc, out.data_ptr(),
                                  ws.get(), ws.size()));
    }
    return out;
}

// Zeros of ``shape`` accumulating ``grad`` at ``index`` along ``dim``: the
// backward kernel of the gathers.
std::optional<Tensor> gather_backward_impl(const Tensor& grad, const std::vector<int64_t>& shape,
                                           int64_t dim, const Tensor& index) {
    const int dev = index_of(grad.device());
    if (grad.dim() == 0 || grad.dim() > kMaxCnnlDims || !native_index(index) ||
        index_of(index.device()) != dev || !value_dtype(grad.dtype(), dev)) {
        return std::nullopt;
    }
    const int64_t d = resolve_dim(dim, static_cast<int64_t>(shape.size()));
    if (d < 0 || d >= static_cast<int64_t>(shape.size())) return std::nullopt;
    Tensor base = empty_on(shape, grad.dtype(), dev);
    base.zero_();
    return scatter_impl(base, dim, index, grad, CNNL_SCATTER_ADD, true);
}

Tensor gather_kernel(const Tensor& self, int64_t dim, const Tensor& index) {
    if (native_scatter(self, index, self)) {
        if (auto r = gather_impl(self, dim, index)) return *r;
    }
    return cpu_fallback::Kernel<"gather", sig::gather>::call(self, dim, index);
}

Tensor gather_backward_kernel(const Tensor& grad, const Tensor& self, int64_t dim,
                              const Tensor& index, bool sparse_grad) {
    if (!sparse_grad && native_scatter(self, index, grad)) {
        if (auto r = gather_backward_impl(grad, self.sizes().vec(), dim, index)) return *r;
    }
    return cpu_fallback::Kernel<"gather_backward", sig::gather_backward>::call(grad, self, dim,
                                                                               index,
                                                                               sparse_grad);
}

Tensor index_select_kernel(const Tensor& self, int64_t dim, const Tensor& index) {
    const int dev = index_of(self.device());
    // CNNL describes the 1-D index at the selected dimension of the input,
    // and the output keeps the input's rank.
    if (self.device().is_privateuse1() && index_of(index.device()) == dev && index.dim() == 1 &&
        self.dim() > 0 && self.dim() <= kMaxCnnlDims && native_index(index) &&
        value_dtype(self.dtype(), dev)) {
        const int64_t d = resolve_dim(dim, self.dim());
        if (d >= 0 && d < self.dim()) {
            const Tensor x = contiguous_on_device(self);
            const Tensor idx = contiguous_on_device(index);
            std::vector<int64_t> out_sizes = self.sizes().vec();
            out_sizes[d] = idx.numel();
            Tensor out = empty_on(out_sizes, self.dtype(), dev);
            if (x.numel() > 0 && idx.numel() > 0) {
                DeviceGuard guard(dev);
                TensorDesc x_desc(x), idx_desc(idx), out_desc(out);
                TP_CNNL_TRY(cnnlIndexSelect(handle(dev), static_cast<int>(d), x_desc,
                                            x.data_ptr(), idx_desc, idx.data_ptr(), out_desc,
                                            out.data_ptr()));
            }
            return out;
        }
    }
    return cpu_fallback::Kernel<"index_select", sig::index_select>::call(self, dim, index);
}

Tensor index_select_backward_kernel(const Tensor& grad, const std::vector<int64_t>& self_sizes,
                                    int64_t dim, const Tensor& index) {
    if (auto r = grad.dim() > 0 ? gather_backward_impl(grad, self_sizes, dim, index)
                                : std::nullopt) {
        return *r;
    }
    return cpu_fallback::Kernel<"index_select_backward", sig::index_select_backward>::call(
        grad, self_sizes, dim, index);
}

Tensor scatter_add_kernel(const Tensor& self, int64_t dim, const Tensor& index,
                          const Tensor& src) {
    if (native_scatter(self, index, src)) {
        if (auto r = scatter_impl(self, dim, index, src, CNNL_SCATTER_ADD, true)) return *r;
    }
    return cpu_fallback::Kernel<"scatter_add", sig::scatter_add>::call(self, dim, index, src);
}

Tensor& scatter_add__kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    if (native_scatter(self, index, src) && binary_result_dtype(self, src) == self.dtype()) {
        if (auto r = scatter_impl(self, dim, index, src, CNNL_SCATTER_ADD, true)) {
            strided_copy(self, *r);
            return self;
        }
    }
    return cpu_fallback::Kernel<"scatter_add_", sig::scatter_add_>::call(self, dim, index, src);
}

Tensor scatter_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                      const std::string& reduce) {
    bool supported = false;
    const cnnlScatterMode_t mode = reduce_mode(reduce, &supported);
    if (supported && native_scatter(self, index, src)) {
        if (auto r = scatter_impl(self, dim, index, src, mode,
                                  mode != CNNL_SCATTER)) return *r;
    }
    return cpu_fallback::Kernel<"scatter.reduce", sig::scatter__reduce>::call(self, dim, index,
                                                                              src, reduce);
}

Tensor& scatter__kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                        const std::string& reduce) {
    bool supported = false;
    const cnnlScatterMode_t mode = reduce_mode(reduce, &supported);
    if (supported && native_scatter(self, index, src) &&
        binary_result_dtype(self, src) == self.dtype()) {
        if (auto r = scatter_impl(self, dim, index, src, mode, true)) {
            strided_copy(self, *r);
            return self;
        }
    }
    return cpu_fallback::Kernel<"scatter_.reduce", sig::scatter___reduce>::call(self, dim, index,
                                                                                src, reduce);
}

Tensor scatter_reduce_kernel(const Tensor& self, int64_t dim, const Tensor& index,
                             const Tensor& src, const std::string& reduce, bool include_self) {
    bool supported = false;
    const cnnlScatterMode_t mode = reduce_mode(reduce, &supported);
    if (supported && reduce != "add" && native_scatter(self, index, src)) {
        if (auto r = scatter_impl(self, dim, index, src, mode, include_self)) return *r;
    }
    return cpu_fallback::Kernel<"scatter_reduce.two", sig::scatter_reduce__two>::call(
        self, dim, index, src, reduce, include_self);
}

Tensor cumsum_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    const DType out = dtype.value_or(self.dtype());
    if (scan_dtype(out)) {
        if (auto r = scan_impl(self, dim, out, false)) return *r;
    }
    return cpu_fallback::Kernel<"cumsum", sig::cumsum>::call(self, dim, dtype);
}

Tensor& cumsum__kernel(Tensor& self, int64_t dim, std::optional<DType> dtype) {
    if ((!dtype || *dtype == self.dtype()) && scan_dtype(self.dtype())) {
        if (auto r = scan_impl(self, dim, self.dtype(), false)) {
            strided_copy(self, *r);
            return self;
        }
    }
    return cpu_fallback::Kernel<"cumsum_", sig::cumsum_>::call(self, dim, dtype);
}

Tensor cumprod_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    const DType out = dtype.value_or(self.dtype());
    if (scan_dtype(out)) {
        if (auto r = scan_impl(self, dim, out, true)) return *r;
    }
    return cpu_fallback::Kernel<"cumprod", sig::cumprod>::call(self, dim, dtype);
}

Tensor& cumprod__kernel(Tensor& self, int64_t dim, std::optional<DType> dtype) {
    if ((!dtype || *dtype == self.dtype()) && scan_dtype(self.dtype())) {
        if (auto r = scan_impl(self, dim, self.dtype(), true)) {
            strided_copy(self, *r);
            return self;
        }
    }
    return cpu_fallback::Kernel<"cumprod_", sig::cumprod_>::call(self, dim, dtype);
}

Tensor where_self_kernel(const Tensor& condition, const Tensor& self, const Tensor& other) {
    const int dev = index_of(condition.device());
    const bool same_dev = index_of(self.device()) == dev && index_of(other.device()) == dev;
    const DType common = binary_result_dtype(self, other);
    if (condition.device().is_privateuse1() && same_dev && condition.dtype() == DType::Bool &&
        condition.dim() <= kMaxCnnlDims && common != DType::Float64 &&
        cnnl_dtype(common)) {
        const int64_t nd = std::max({condition.dim(), self.dim(), other.dim()});
        std::vector<int64_t> shape = broadcast_shape(broadcast_shape(condition.sizes().vec(),
                                                                     self.sizes().vec()),
                                                     other.sizes().vec());
        if (static_cast<int64_t>(shape.size()) >= nd &&
            static_cast<int64_t>(shape.size()) <= kMaxCnnlDims) {
            const Tensor cond = contiguous_on_device(condition);
            const Tensor a = to_dtype(contiguous_on_device(self), common);
            const Tensor b = to_dtype(contiguous_on_device(other), common);
            Tensor out = empty_on(shape, common, dev);
            if (out.numel() > 0) {
                DeviceGuard guard(dev);
                TensorDesc cond_desc = padded_desc(cond, shape.size());
                TensorDesc a_desc = padded_desc(a, shape.size());
                TensorDesc b_desc = padded_desc(b, shape.size());
                TensorDesc out_desc(out);
                size_t ws_size = 0;
                TP_CNNL_TRY(cnnlGetSelectV2WorkspaceSize(handle(dev), cond_desc, a_desc, b_desc,
                                                         &ws_size));
                Workspace ws(ws_size, dev);
                TP_CNNL_TRY(cnnlSelectV2(handle(dev), cond_desc, cond.data_ptr(), a_desc,
                                         a.data_ptr(), b_desc, b.data_ptr(), ws.get(), ws.size(),
                                         out_desc, out.data_ptr()));
            }
            return out;
        }
    }
    return cpu_fallback::Kernel<"where.self", sig::where__self>::call(condition, self, other);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUIndexKernels) {
    m.impl("gather", static_cast<sig::gather*>(&gather_kernel));
    m.impl("gather_backward", static_cast<sig::gather_backward*>(&gather_backward_kernel));
    m.impl("index_select", static_cast<sig::index_select*>(&index_select_kernel));
    m.impl("index_select_backward",
           static_cast<sig::index_select_backward*>(&index_select_backward_kernel));
    m.impl("scatter_add", static_cast<sig::scatter_add*>(&scatter_add_kernel));
    m.impl("scatter_add_", static_cast<sig::scatter_add_*>(&scatter_add__kernel));
    m.impl("scatter.reduce", static_cast<sig::scatter__reduce*>(&scatter_kernel));
    m.impl("scatter_.reduce", static_cast<sig::scatter___reduce*>(&scatter__kernel));
    m.impl("scatter_reduce.two", static_cast<sig::scatter_reduce__two*>(&scatter_reduce_kernel));
    m.impl("cumsum", static_cast<sig::cumsum*>(&cumsum_kernel));
    m.impl("cumsum_", static_cast<sig::cumsum_*>(&cumsum__kernel));
    m.impl("cumprod", static_cast<sig::cumprod*>(&cumprod_kernel));
    m.impl("cumprod_", static_cast<sig::cumprod_*>(&cumprod__kernel));
    m.impl("where.self", static_cast<sig::where__self*>(&where_self_kernel));
}

} // namespace mlu
} // namespace tensorplay
