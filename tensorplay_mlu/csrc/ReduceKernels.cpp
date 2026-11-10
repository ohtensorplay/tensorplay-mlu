// Reductions (sum, mean) and softmax / log-softmax with their backward
// kernels, on CNNL.

#include <algorithm>
#include <optional>
#include <vector>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

// One reduce launch: ``out`` is ``x`` with the ``axes`` reduced to size 1.
bool launch_reduce(const Tensor& x, const Tensor& out, const std::vector<int>& axes,
                   cnnlReduceOp_t op) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x), out_desc(out);
    cnnlReduceDescriptor_t desc = nullptr;
    TP_CNNL_CHECK(cnnlCreateReduceDescriptor(&desc));
    struct Destroy {
        cnnlReduceDescriptor_t d;
        ~Destroy() { cnnlDestroyReduceDescriptor(d); }
    } destroy{desc};
    TP_CNNL_TRY(cnnlSetReduceDescriptor_v2(desc, const_cast<int*>(axes.data()),
                                           static_cast<int>(axes.size()), op, CNNL_DTYPE_FLOAT,
                                           CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES,
                                           CNNL_32BIT_INDICES, 0.0f));
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetReduceOpWorkspaceSize_v2(h, x_desc, out_desc, nullptr, desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlReduce_v2(h, desc, x_desc, x.data_ptr(), nullptr, nullptr, ws.get(),
                              ws.size(), out_desc, out.data_ptr(), nullptr, nullptr));
    return true;
}

// Reduce ``self`` over ``dims`` (all when empty) into a keepdim-shaped
// tensor, then view it without the reduced dimensions unless keepdim.
std::optional<Tensor> reduce(const Tensor& self, std::vector<int64_t> dims, bool keepdim,
                             cnnlReduceOp_t op, DType out_dtype) {
    const int index = index_of(self.device());
    const int64_t nd = self.dim();
    Tensor x = contiguous_on_device(to_dtype(self, out_dtype));
    if (nd == 0) return x.clone();
    if (dims.empty()) {
        dims.resize(nd);
        for (int64_t i = 0; i < nd; ++i) dims[i] = i;
    }
    std::vector<bool> reduced(nd, false);
    for (int64_t d : dims) reduced[d < 0 ? d + nd : d] = true;
    std::vector<int64_t> keep_shape = x.sizes().vec();
    std::vector<int64_t> final_shape;
    std::vector<int> axes;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[i]) {
            keep_shape[i] = 1;
            axes.push_back(static_cast<int>(i));
            if (keepdim) final_shape.push_back(1);
        } else {
            final_shape.push_back(x.sizes()[i]);
        }
    }
    Tensor out = empty_on(keep_shape, out_dtype, index);
    if (x.numel() == 0) {
        out.zero_();
    } else if (!launch_reduce(x, out, axes, op)) {
        return std::nullopt;
    }
    if (keepdim) return out;
    const auto& impl = out.unsafeGetTensorImpl();
    std::vector<int64_t> strides(final_shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(final_shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * final_shape[i + 1];
    }
    return Tensor(impl->storage(), final_shape, strides, out_dtype, impl->storage_offset());
}

DType reduce_dtype(const Tensor& self, DType requested) {
    return requested == DType::Undefined ? self.dtype() : requested;
}

bool native_reduce(const Tensor& self, DType out_dtype) {
    if (!self.device().is_privateuse1() || self.dim() > kMaxCnnlDims) return false;
    const int index = index_of(self.device());
    return is_native_float(out_dtype, index) &&
           (is_native_float(self.dtype(), index) || self.dtype() == out_dtype);
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    const DType out = reduce_dtype(self, dtype);
    if (native_reduce(self, out)) {
        if (auto r = reduce(self, {}, false, CNNL_REDUCE_ADD, out)) return *r;
    }
    return cpu_fallback::Kernel<"sum", sig::sum>::call(self, dtype);
}

Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim,
                      DType dtype) {
    const DType out = reduce_dtype(self, dtype);
    if (native_reduce(self, out)) {
        if (auto r = reduce(self, dim, keepdim, CNNL_REDUCE_ADD, out)) return *r;
    }
    return cpu_fallback::Kernel<"sum.dim_IntList", sig::sum__dim_IntList>::call(self, dim, keepdim,
                                                                               dtype);
}

Tensor amax_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (native_reduce(self, self.dtype())) {
        if (auto r = reduce(self, dim, keepdim, CNNL_REDUCE_MAX, self.dtype())) return *r;
    }
    return cpu_fallback::Kernel<"amax", sig::amax>::call(self, dim, keepdim);
}

Tensor amin_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (native_reduce(self, self.dtype())) {
        if (auto r = reduce(self, dim, keepdim, CNNL_REDUCE_MIN, self.dtype())) return *r;
    }
    return cpu_fallback::Kernel<"amin", sig::amin>::call(self, dim, keepdim);
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    const DType out = reduce_dtype(self, dtype);
    if (native_reduce(self, out)) {
        if (auto r = reduce(self, {}, false, CNNL_REDUCE_AVG, out)) return *r;
    }
    return cpu_fallback::Kernel<"mean", sig::mean>::call(self, dtype);
}

Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim,
                       DType dtype) {
    const DType out = reduce_dtype(self, dtype);
    if (native_reduce(self, out)) {
        if (auto r = reduce(self, dim, keepdim, CNNL_REDUCE_AVG, out)) return *r;
    }
    return cpu_fallback::Kernel<"mean.dim", sig::mean__dim>::call(self, dim, keepdim, dtype);
}

// Softmax over ``dim`` of a dense tensor seen as [outer, dim, inner].
struct SoftmaxShape {
    std::vector<int64_t> sizes;
    cnnlSoftmaxMode_t mode;
};

SoftmaxShape softmax_shape(const Tensor& t, int64_t dim) {
    const int64_t nd = std::max<int64_t>(t.dim(), 1);
    if (dim < 0) dim += nd;
    const auto s = t.dim() == 0 ? std::vector<int64_t>{1} : t.sizes().vec();
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= s[i];
    for (int64_t i = dim + 1; i < nd; ++i) inner *= s[i];
    if (inner == 1) return {{1, outer, s[dim]}, CNNL_SOFTMAX_MODE_LOW_DIMENSION};
    return {{outer, s[dim], inner}, CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION};
}

TensorDesc dense_desc(const std::vector<int64_t>& sizes, DType dtype) {
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return TensorDesc(sizes, strides, dtype);
}

std::optional<Tensor> softmax_forward(const Tensor& self, int64_t dim,
                                      cnnlSoftmaxAlgorithm_t algorithm) {
    const int index = index_of(self.device());
    const Tensor x = contiguous_on_device(self);
    Tensor y = empty_on(x.sizes().vec(), x.dtype(), index);
    if (y.numel() == 0) return y;
    const SoftmaxShape shape = softmax_shape(x, dim);
    DeviceGuard guard(index);
    TensorDesc desc = dense_desc(shape.sizes, x.dtype());
    TP_CNNL_TRY(cnnlSoftmaxForward(handle(index), algorithm, shape.mode, nullptr, desc,
                                   x.data_ptr(), nullptr, desc, y.data_ptr()));
    return y;
}

std::optional<Tensor> softmax_backward(const Tensor& grad, const Tensor& output, int64_t dim,
                                       cnnlSoftmaxAlgorithm_t algorithm) {
    const int index = index_of(grad.device());
    const Tensor dy = contiguous_on_device(grad);
    const Tensor y = contiguous_on_device(output);
    Tensor dx = empty_on(dy.sizes().vec(), dy.dtype(), index);
    if (dx.numel() == 0) return dx;
    const SoftmaxShape shape = softmax_shape(dy, dim);
    DeviceGuard guard(index);
    TensorDesc desc = dense_desc(shape.sizes, dy.dtype());
    TP_CNNL_TRY(cnnlSoftmaxBackward(handle(index), algorithm, shape.mode, nullptr, desc,
                                    y.data_ptr(), desc, dy.data_ptr(), nullptr, desc,
                                    dx.data_ptr()));
    return dx;
}

bool native_softmax(const Tensor& self, bool half_to_float) {
    return self.device().is_privateuse1() && !half_to_float &&
           is_native_float(self.dtype(), index_of(self.device()));
}

// ``softmax`` / ``log_softmax``: the result takes the requested dtype (the
// input's when none is asked for) and the same CNNL call computes it.
Tensor softmax_dtype_kernel(const Tensor& self, int64_t dim, DType dtype,
                            cnnlSoftmaxAlgorithm_t algorithm) {
    const DType out = dtype == DType::Undefined ? self.dtype() : dtype;
    if (self.device().is_privateuse1() && is_native_float(out, index_of(self.device())) &&
        is_native_float(self.dtype(), index_of(self.device()))) {
        const Tensor x = out == self.dtype() ? self : to_dtype(self, out);
        if (auto r = softmax_forward(x, dim, algorithm)) return *r;
    }
    return algorithm == CNNL_SOFTMAX_LOG
        ? cpu_fallback::Kernel<"log_softmax", sig::log_softmax>::call(self, dim, dtype)
        : cpu_fallback::Kernel<"softmax", sig::softmax>::call(self, dim, dtype);
}

Tensor softmax_dtype(const Tensor& self, int64_t dim, DType dtype) {
    return softmax_dtype_kernel(self, dim, dtype, CNNL_SOFTMAX_ACCURATE);
}

Tensor log_softmax_dtype(const Tensor& self, int64_t dim, DType dtype) {
    return softmax_dtype_kernel(self, dim, dtype, CNNL_SOFTMAX_LOG);
}

Tensor softmax_kernel(const Tensor& self, int64_t dim, bool half_to_float) {
    if (native_softmax(self, half_to_float)) {
        if (auto r = softmax_forward(self, dim, CNNL_SOFTMAX_ACCURATE)) return *r;
    }
    return cpu_fallback::Kernel<"_softmax", sig::_softmax>::call(self, dim, half_to_float);
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, bool half_to_float) {
    if (native_softmax(self, half_to_float)) {
        if (auto r = softmax_forward(self, dim, CNNL_SOFTMAX_LOG)) return *r;
    }
    return cpu_fallback::Kernel<"_log_softmax", sig::_log_softmax>::call(self, dim,
                                                                         half_to_float);
}

Tensor softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim,
                               DType input_dtype) {
    if (native_softmax(grad_output, false) && output.dtype() == grad_output.dtype() &&
        input_dtype == grad_output.dtype()) {
        if (auto r = softmax_backward(grad_output, output, dim, CNNL_SOFTMAX_ACCURATE)) return *r;
    }
    return cpu_fallback::Kernel<"_softmax_backward_data", sig::_softmax_backward_data>::call(
        grad_output, output, dim, input_dtype);
}

Tensor log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim,
                                   DType input_dtype) {
    if (native_softmax(grad_output, false) && output.dtype() == grad_output.dtype() &&
        input_dtype == grad_output.dtype()) {
        if (auto r = softmax_backward(grad_output, output, dim, CNNL_SOFTMAX_LOG)) return *r;
    }
    return cpu_fallback::Kernel<"_log_softmax_backward_data",
                                sig::_log_softmax_backward_data>::call(grad_output, output, dim,
                                                                       input_dtype);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUReduceKernels) {
    m.impl("sum", static_cast<sig::sum*>(&sum_kernel));
    m.impl("sum.dim_IntList", static_cast<sig::sum__dim_IntList*>(&sum_dim_kernel));
    m.impl("amax", static_cast<sig::amax*>(&amax_kernel));
    m.impl("amin", static_cast<sig::amin*>(&amin_kernel));
    m.impl("mean", static_cast<sig::mean*>(&mean_kernel));
    m.impl("mean.dim", static_cast<sig::mean__dim*>(&mean_dim_kernel));
    m.impl("softmax", static_cast<sig::softmax*>(&softmax_dtype));
    m.impl("log_softmax", static_cast<sig::log_softmax*>(&log_softmax_dtype));
    m.impl("_softmax", static_cast<sig::_softmax*>(&softmax_kernel));
    m.impl("_log_softmax", static_cast<sig::_log_softmax*>(&log_softmax_kernel));
    m.impl("_softmax_backward_data",
           static_cast<sig::_softmax_backward_data*>(&softmax_backward_kernel));
    m.impl("_log_softmax_backward_data",
           static_cast<sig::_log_softmax_backward_data*>(&log_softmax_backward_kernel));
}

} // namespace mlu
} // namespace tensorplay
