// Neural-network layers on CNNL: layer norm, dropout, embedding and the
// negative log-likelihood loss, forward and backward.
//
// Every kernel follows the host kernels' conventions (statistics shapes and
// dtypes, mask dtype, which totals a reduction reports), so a forward on the
// device pairs with a backward on either side.  Inputs outside what the
// device kernel covers run on the host.

#include <optional>
#include <tuple>
#include <vector>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

bool on_mlu(const Tensor& t) { return t.defined() && t.device().is_privateuse1(); }

bool native_float(const Tensor& t) {
    return on_mlu(t) && is_native_float(t.dtype(), index_of(t.device()));
}

// An optional affine parameter usable next to ``x`` (same device and dtype).
bool usable_param(const std::optional<Tensor>& p, const Tensor& x) {
    return !p.has_value() || !p->defined() ||
           (on_mlu(*p) && p->dtype() == x.dtype() && index_of(p->device()) == index_of(x.device()));
}

Tensor filled(const std::vector<int64_t>& sizes, DType dtype, int index, double value) {
    Tensor t = empty_on(sizes, dtype, index);
    t.fill_(Scalar(value));
    return t;
}

// Dense descriptor of ``sizes`` (row-major strides).
TensorDesc dense_desc(const std::vector<int64_t>& sizes, DType dtype) {
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return TensorDesc(sizes, strides, dtype);
}

DType stats_dtype(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16 ? DType::Float32 : dtype;
}

// -- layer norm ---------------------------------------------------------------

std::optional<std::tuple<Tensor, Tensor, Tensor>> layer_norm_native(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps) {
    const int64_t nd = input.dim();
    const int64_t axis = nd - static_cast<int64_t>(normalized_shape.size());
    if (!native_float(input) || normalized_shape.empty() || axis <= 0 ||
        !usable_param(weight, input) || !usable_param(bias, input)) {
        return std::nullopt;
    }
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        if (input.sizes()[axis + static_cast<int64_t>(i)] != normalized_shape[i]) return std::nullopt;
    }
    const int index = index_of(input.device());
    const Tensor x = contiguous_on_device(input);
    if (x.numel() == 0) return std::nullopt;
    const DType dtype = x.dtype(), sdtype = stats_dtype(dtype);
    int64_t outer = 1;
    std::vector<int64_t> outer_shape;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= x.sizes()[i];
        outer_shape.push_back(x.sizes()[i]);
    }
    const Tensor w = weight && weight->defined() ? contiguous_on_device(*weight)
                                                 : filled(normalized_shape, dtype, index, 1.0);
    const Tensor b = bias && bias->defined() ? contiguous_on_device(*bias)
                                             : filled(normalized_shape, dtype, index, 0.0);
    Tensor y = empty_on(x.sizes().vec(), dtype, index);
    Tensor mean = empty_on({outer}, sdtype, index), rstd = empty_on({outer}, sdtype, index);
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x), y_desc(y), wb_desc = dense_desc(normalized_shape, dtype);
    TensorDesc stats_desc = dense_desc(outer_shape, sdtype);
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetLayerNormOpWorkspaceSize(h, static_cast<int>(axis), x_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlLayerNormForward(h, x_desc, x.data_ptr(), static_cast<int>(axis), wb_desc,
                                     w.data_ptr(), b.data_ptr(), static_cast<float>(eps),
                                     ws.get(), ws.size(), y_desc, y.data_ptr(), stats_desc,
                                     mean.data_ptr(), rstd.data_ptr()));
    return std::make_tuple(y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_kernel(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps) {
    if (auto r = layer_norm_native(input, normalized_shape, weight, bias, eps)) return *r;
    return cpu_fallback::Kernel<"native_layer_norm", sig::native_layer_norm>::call(
        input, normalized_shape, weight, bias, eps);
}

std::optional<std::tuple<Tensor, Tensor, Tensor>> layer_norm_backward_native(
    const Tensor& grad_out, const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& weight,
    const std::vector<bool>& output_mask) {
    const int64_t nd = input.dim();
    const int64_t axis = nd - static_cast<int64_t>(normalized_shape.size());
    if (!native_float(input) || !on_mlu(grad_out) || grad_out.dtype() != input.dtype() ||
        !on_mlu(mean) || !on_mlu(rstd) || normalized_shape.empty() || axis <= 0 ||
        !usable_param(weight, input) || grad_out.sizes().vec() != input.sizes().vec()) {
        return std::nullopt;
    }
    const int index = index_of(input.device());
    const Tensor x = contiguous_on_device(input);
    if (x.numel() == 0) return std::nullopt;
    const Tensor dz = contiguous_on_device(grad_out);
    const DType dtype = x.dtype(), sdtype = stats_dtype(dtype);
    std::vector<int64_t> outer_shape(x.sizes().vec().begin(), x.sizes().vec().begin() + axis);
    const Tensor m = contiguous_on_device(to_dtype(mean, sdtype));
    const Tensor r = contiguous_on_device(to_dtype(rstd, sdtype));
    const Tensor w = weight && weight->defined() ? contiguous_on_device(*weight)
                                                 : filled(normalized_shape, dtype, index, 1.0);
    Tensor dx = empty_on(x.sizes().vec(), dtype, index);
    Tensor dw = empty_on(normalized_shape, dtype, index);
    Tensor db = empty_on(normalized_shape, dtype, index);
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc(x), dz_desc(dz), dx_desc(dx), w_desc = dense_desc(normalized_shape, dtype);
    TensorDesc stats_desc = dense_desc(outer_shape, sdtype);
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetLayerNormBackwardWorkspaceSize(h, x_desc, static_cast<int>(axis), &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlLayerNormBackward_v2(h, x_desc, x.data_ptr(), static_cast<int>(axis), dz_desc,
                                         dz.data_ptr(), w_desc, w.data_ptr(), stats_desc,
                                         m.data_ptr(), r.data_ptr(), ws.get(), ws.size(), dx_desc,
                                         dx.data_ptr(), dw.data_ptr(), db.data_ptr()));
    auto pick = [&](size_t i, const Tensor& t) {
        return i < output_mask.size() && !output_mask[i] ? Tensor() : t;
    };
    return std::make_tuple(pick(0, dx), pick(1, dw), pick(2, db));
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_kernel(
    const Tensor& grad_out, const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, const std::vector<bool>& output_mask) {
    if (auto r = layer_norm_backward_native(grad_out, input, normalized_shape, mean, rstd, weight,
                                            output_mask)) {
        return *r;
    }
    return cpu_fallback::Kernel<"native_layer_norm_backward", sig::native_layer_norm_backward>::
        call(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

// -- dropout ------------------------------------------------------------------

std::tuple<Tensor, Tensor> native_dropout_kernel(const Tensor& input, double p,
                                                 std::optional<bool> train) {
    if (!native_float(input) || p < 0.0 || p > 1.0) {
        return cpu_fallback::Kernel<"native_dropout", sig::native_dropout>::call(input, p, train);
    }
    const int index = index_of(input.device());
    // Not training drops nothing; p == 1 drops everything (the scale that
    // would undo it is undefined).
    if ((train.has_value() && !*train) || p == 1.0) {
        const bool keep = train.has_value() && !*train;
        Tensor out = keep ? contiguous_on_device(input).clone()
                          : filled(input.sizes().vec(), input.dtype(), index, 0.0);
        return {out, filled(input.sizes().vec(), DType::Bool, index, keep ? 1.0 : 0.0)};
    }
    const Tensor x = contiguous_on_device(input);
    Tensor out = empty_on(x.sizes().vec(), x.dtype(), index);
    Tensor mask = empty_on(x.sizes().vec(), DType::Bool, index);
    if (x.numel() == 0) return {out, mask};
    // The device stream of random numbers is keyed from the host generator,
    // so tensorplay.manual_seed makes it reproducible.
    const uint64_t seed = default_generator().random64();
    DeviceGuard guard(index);
    TensorDesc x_desc(x), out_desc(out), mask_desc(mask);
    const cnnlStatus_t status = cnnlFusedDropout_v3(
        handle(index), CNNL_RAND_RNG_PHILOX, x_desc, x.data_ptr(), false, seed, 0, nullptr,
        nullptr, 0, static_cast<float>(p), mask_desc, mask.data_ptr(), out_desc, out.data_ptr());
    if (cnnl_declined(status)) {
        return cpu_fallback::Kernel<"native_dropout", sig::native_dropout>::call(input, p, train);
    }
    TP_CNNL_CHECK(status);
    return {out, mask};
}

// out = scale * a * b (b converted to a's dtype).
bool scaled_product(const Tensor& a, const Tensor& b, float scale, const Tensor& out) {
    const int index = index_of(out.device());
    const Tensor ac = contiguous_on_device(a);
    const Tensor bc = contiguous_on_device(to_dtype(b, a.dtype()));
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc a_desc(ac), b_desc(bc), c_desc(out);
    cnnlOpTensorDescriptor_t op = nullptr;
    TP_CNNL_CHECK(cnnlCreateOpTensorDescriptor(&op));
    struct Destroy {
        cnnlOpTensorDescriptor_t d;
        ~Destroy() { cnnlDestroyOpTensorDescriptor(d); }
    } destroy{op};
    TP_CNNL_TRY(cnnlSetOpTensorDescriptor(op, CNNL_OP_TENSOR_MUL, CNNL_DTYPE_FLOAT,
                                          CNNL_NOT_PROPAGATE_NAN));
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetOpTensorWorkspaceSize(h, a_desc, b_desc, c_desc, &ws_size));
    Workspace ws(ws_size, index);
    const float one = 1.0f, zero = 0.0f;
    TP_CNNL_TRY(cnnlOpTensor(h, op, &scale, a_desc, ac.data_ptr(), &one, b_desc, bc.data_ptr(),
                             ws.get(), ws.size(), &zero, c_desc, out.data_ptr()));
    return true;
}

Tensor native_dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask,
                                      double scale) {
    if (native_float(grad_output) && on_mlu(mask) &&
        grad_output.sizes().vec() == mask.sizes().vec()) {
        Tensor out = empty_on(grad_output.sizes().vec(), grad_output.dtype(),
                              index_of(grad_output.device()));
        if (out.numel() == 0 || scaled_product(grad_output, mask, static_cast<float>(scale), out)) {
            return out;
        }
    }
    return cpu_fallback::Kernel<"native_dropout_backward", sig::native_dropout_backward>::call(
        grad_output, mask, scale);
}

// -- embedding ----------------------------------------------------------------

bool index_tensor(const Tensor& t) {
    return on_mlu(t) && (t.dtype() == DType::Int64 || t.dtype() == DType::Int32);
}

Tensor embedding_kernel(const Tensor& weight, const Tensor& indices, int64_t padding_idx,
                        bool scale_grad_by_freq, bool sparse) {
    if (native_float(weight) && weight.dim() == 2 && index_tensor(indices) && !sparse &&
        index_of(weight.device()) == index_of(indices.device())) {
        const int index = index_of(weight.device());
        const Tensor w = contiguous_on_device(weight);
        const Tensor idx = contiguous_on_device(indices);
        std::vector<int64_t> shape = idx.sizes().vec();
        shape.push_back(w.sizes()[1]);
        Tensor out = empty_on(shape, w.dtype(), index);
        if (out.numel() == 0) return out;
        DeviceGuard guard(index);
        TensorDesc w_desc(w), idx_desc(idx), out_desc(out);
        // The padding row only matters to the gradient; the lookup reads it
        // like any other row.
        const cnnlStatus_t status =
            cnnlEmbeddingForward_v2(handle(index), w_desc, w.data_ptr(), idx_desc, idx.data_ptr(),
                                    -1, nullptr, nullptr, out_desc, out.data_ptr());
        if (!cnnl_declined(status)) {
            TP_CNNL_CHECK(status);
            return out;
        }
    }
    return cpu_fallback::Kernel<"embedding", sig::embedding>::call(weight, indices, padding_idx,
                                                                   scale_grad_by_freq, sparse);
}

Tensor embedding_dense_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                       int64_t num_weights, int64_t padding_idx,
                                       bool scale_grad_by_freq) {
    if (native_float(grad_output) && index_tensor(indices) && grad_output.dim() >= 1 &&
        index_of(grad_output.device()) == index_of(indices.device())) {
        const int index = index_of(grad_output.device());
        const Tensor idx = contiguous_on_device(to_dtype(indices, DType::Int32));
        const Tensor dy = contiguous_on_device(grad_output);
        const int64_t dim = dy.sizes()[dy.dim() - 1];
        Tensor out = empty_on({num_weights, dim}, dy.dtype(), index);
        if (out.numel() == 0) return out;
        DeviceGuard guard(index);
        cnnlHandle_t h = handle(index);
        TensorDesc idx_desc(idx), dy_desc(dy), out_desc(out);
        size_t ws_size = 0;
        cnnlStatus_t status = cnnlGetEmbeddingBackwardWorkspaceSize(h, dy_desc, out_desc,
                                                                    scale_grad_by_freq, &ws_size);
        if (!cnnl_declined(status)) {
            TP_CNNL_CHECK(status);
            Workspace ws(ws_size, index);
            status = cnnlEmbeddingBackward(h, static_cast<int>(padding_idx), scale_grad_by_freq,
                                           idx_desc, idx.data_ptr(), dy_desc, dy.data_ptr(),
                                           ws.get(), ws.size(), out_desc, out.data_ptr());
            if (!cnnl_declined(status)) {
                TP_CNNL_CHECK(status);
                return out;
            }
        }
    }
    return cpu_fallback::Kernel<"embedding_dense_backward", sig::embedding_dense_backward>::call(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

// -- negative log-likelihood --------------------------------------------------

std::optional<cnnlNlllossAlgorithm_t> nll_algorithm(int64_t reduction) {
    switch (reduction) {
        case 0: return CNNL_REDUCTION_NONE;
        case 1: return CNNL_REDUCTION_MEAN;
        case 2: return CNNL_REDUCTION_SUM;
        default: return std::nullopt;
    }
}

// (N, C) input with (N) targets on one device; the batch-less form and
// other layouts run on the host.
bool nll_native(const Tensor& self, const Tensor& target, const std::optional<Tensor>& weight) {
    return native_float(self) && self.dim() == 2 && index_tensor(target) && target.dim() == 1 &&
           target.sizes()[0] == self.sizes()[0] && self.numel() > 0 &&
           index_of(self.device()) == index_of(target.device()) && usable_param(weight, self) &&
           (!weight || !weight->defined() || weight->numel() == self.sizes()[1]);
}

std::tuple<Tensor, Tensor> nll_loss_kernel(const Tensor& self, const Tensor& target,
                                           const std::optional<Tensor>& weight, int64_t reduction,
                                           int64_t ignore_index) {
    const auto algorithm = nll_algorithm(reduction);
    if (algorithm && nll_native(self, target, weight)) {
        const int index = index_of(self.device());
        const DType dtype = self.dtype();
        const int64_t n = self.sizes()[0], c = self.sizes()[1];
        const Tensor x = contiguous_on_device(self);
        const Tensor t = contiguous_on_device(to_dtype(target, DType::Int32));
        const Tensor w = weight && weight->defined() ? contiguous_on_device(*weight)
                                                     : filled({c}, dtype, index, 1.0);
        Tensor y = empty_on(reduction == 0 ? std::vector<int64_t>{n} : std::vector<int64_t>{},
                            dtype, index);
        Tensor total = empty_on({}, dtype, index);
        DeviceGuard guard(index);
        cnnlHandle_t h = handle(index);
        TensorDesc x_desc(x), t_desc(t), w_desc(w);
        TensorDesc y_desc = dense_desc(reduction == 0 ? std::vector<int64_t>{n}
                                                      : std::vector<int64_t>{1},
                                       dtype);
        TensorDesc total_desc = dense_desc({1}, dtype);
        size_t ws_size = 0;
        TP_CNNL_CHECK(cnnlGetNlllossWorkspaceSize(h, x_desc, &ws_size));
        Workspace ws(ws_size, index);
        const cnnlStatus_t status = cnnlNlllossForward(
            h, *algorithm, ws.get(), ws.size(), x_desc, x.data_ptr(), t_desc, t.data_ptr(),
            static_cast<int>(ignore_index), w_desc, w.data_ptr(), total_desc, total.data_ptr(),
            y_desc, y.data_ptr());
        if (!cnnl_declined(status)) {
            TP_CNNL_CHECK(status);
            // A per-sample result of a batch reports no total.
            if (reduction == 0) total.zero_();
            return {y, total};
        }
    }
    return cpu_fallback::Kernel<"nll_loss", sig::nll_loss>::call(self, target, weight, reduction,
                                                                 ignore_index);
}

Tensor nll_loss_backward_kernel(const Tensor& grad_output, const Tensor& self,
                                const Tensor& target, const std::optional<Tensor>& weight,
                                int64_t reduction, int64_t ignore_index,
                                const Tensor& total_weight) {
    const auto algorithm = nll_algorithm(reduction);
    if (algorithm && nll_native(self, target, weight) && on_mlu(grad_output) &&
        grad_output.dtype() == self.dtype() && on_mlu(total_weight) &&
        grad_output.numel() == (reduction == 0 ? self.sizes()[0] : 1)) {
        const int index = index_of(self.device());
        const DType dtype = self.dtype();
        const int64_t n = self.sizes()[0], c = self.sizes()[1];
        const Tensor dy = contiguous_on_device(grad_output);
        const Tensor t = contiguous_on_device(to_dtype(target, DType::Int32));
        const Tensor w = weight && weight->defined() ? contiguous_on_device(*weight)
                                                     : filled({c}, dtype, index, 1.0);
        const Tensor total = contiguous_on_device(to_dtype(total_weight, dtype));
        Tensor dx = empty_on({n, c}, dtype, index);
        DeviceGuard guard(index);
        TensorDesc dy_desc = dense_desc(reduction == 0 ? std::vector<int64_t>{n}
                                                       : std::vector<int64_t>{1},
                                        dtype);
        TensorDesc t_desc(t), w_desc(w), total_desc = dense_desc({1}, dtype), dx_desc(dx);
        const cnnlStatus_t status = cnnlNlllossBackward(
            handle(index), *algorithm, dy_desc, dy.data_ptr(), t_desc, t.data_ptr(),
            static_cast<int>(ignore_index), w_desc, w.data_ptr(), total_desc, total.data_ptr(),
            dx_desc, dx.data_ptr());
        if (!cnnl_declined(status)) {
            TP_CNNL_CHECK(status);
            return dx;
        }
    }
    return cpu_fallback::Kernel<"nll_loss_backward", sig::nll_loss_backward>::call(
        grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUNNKernels) {
    m.impl("native_layer_norm", static_cast<sig::native_layer_norm*>(&native_layer_norm_kernel));
    m.impl("native_layer_norm_backward",
           static_cast<sig::native_layer_norm_backward*>(&native_layer_norm_backward_kernel));
    m.impl("native_dropout", static_cast<sig::native_dropout*>(&native_dropout_kernel));
    m.impl("native_dropout_backward",
           static_cast<sig::native_dropout_backward*>(&native_dropout_backward_kernel));
    m.impl("embedding", static_cast<sig::embedding*>(&embedding_kernel));
    m.impl("embedding_dense_backward",
           static_cast<sig::embedding_dense_backward*>(&embedding_dense_backward_kernel));
    m.impl("nll_loss", static_cast<sig::nll_loss*>(&nll_loss_kernel));
    m.impl("nll_loss_backward", static_cast<sig::nll_loss_backward*>(&nll_loss_backward_kernel));
}

} // namespace mlu
} // namespace tensorplay
