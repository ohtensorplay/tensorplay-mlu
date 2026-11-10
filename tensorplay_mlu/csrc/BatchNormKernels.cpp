// Batch normalization on CNNL: the training and inference forward passes and
// their backward passes.  A batch is normalized per channel using statistics
// taken over the batch and spatial axes; in training the running statistics
// are updated as well.
//
// The card normalizes in a small set of fixed combinations of operand types,
// and a caller may hand any floating width (a reduced input may even carry
// float32 affine parameters and statistics).  Every operand is therefore
// converted to float32 -- the type the batch statistics are reported in -- for
// the card to run, and the results are converted back.
//
// The card reads and writes channel-last operands, so they are carried across
// in that order; a rank-2 or rank-3 operand is lifted to rank 4 with unit
// spatial axes, which is the same normalization on the same data, and the
// result is lowered back to the operator's rank.
//
// Every kernel returns nothing when the shape, the type or the device leaves
// it without a CNNL kernel, and the operator then runs on the host.

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

// ------------------------------- operands -----------------------------------

// The floating widths this operator normalizes on the card.  A wider input has
// no CNNL kernel; a non-floating one is not normalized.
bool bn_operand(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}

// Rank 2 and rank 3 operands as the rank-4 form one pass covers: the unit
// axes are added after the spatial extent, keeping the channel axis where the
// channel-last ordering expects it.  A rank-4 or rank-5 operand already has
// that form.
Tensor lift_rank(const Tensor& t) {
    if (t.dim() == 2) return t.reshape({t.size(0), t.size(1), 1, 1});
    if (t.dim() == 3) return t.reshape({t.size(0), t.size(1), t.size(2), 1});
    return t;
}

// Undo ``lift_rank``: the rank-4 result back to the operator's shape.
Tensor lower_rank(const Tensor& t, const std::vector<int64_t>& sizes) {
    if (static_cast<int64_t>(sizes.size()) >= 4) return t;
    return t.reshape(sizes);
}

// A per-channel parameter on the device: the caller's when it is given (read
// as float32), a constant vector otherwise.
Tensor channel_param(const std::optional<Tensor>& p, int64_t channels, int index, double fill) {
    if (p.has_value() && p->defined()) {
        return to_dtype(contiguous_on_device(*p), DType::Float32);
    }
    Tensor out = empty_on({channels}, DType::Float32, index);
    std::vector<float> host(static_cast<size_t>(channels), static_cast<float>(fill));
    copy_to_device(out.data_ptr(), host.data(), static_cast<size_t>(channels) * sizeof(float),
                   index);
    return out;
}

// A fresh float32 copy of a per-channel statistic, zeros when absent.  The
// card reads and may write this buffer, so it is never the caller's tensor.
Tensor channel_scratch(const std::optional<Tensor>& src, int64_t channels, int index) {
    Tensor out = empty_on({channels}, DType::Float32, index);
    if (src.has_value() && src->defined()) {
        Tensor f = to_dtype(contiguous_on_device(*src), DType::Float32);
        copy_on_device(out.data_ptr(), f.data_ptr(), static_cast<size_t>(channels) * sizeof(float),
                       index);
        return out;
    }
    std::vector<float> zeros(static_cast<size_t>(channels), 0.0f);
    copy_to_device(out.data_ptr(), zeros.data(), static_cast<size_t>(channels) * sizeof(float),
                   index);
    return out;
}

// Copy a float32 per-channel buffer into the caller's statistic tensor,
// converting to its width when that differs.  Both live on the device, so the
// transfer stays on the queue.
void store_stats(const Tensor& dst, const Tensor& src_f32) {
    if (dst.dtype() == DType::Float32 && dst.is_contiguous()) {
        copy_on_device(dst.data_ptr(), src_f32.data_ptr(),
                       static_cast<size_t>(dst.numel()) * sizeof(float), index_of(dst.device()));
        return;
    }
    strided_copy(dst, to_dtype(src_f32, dst.dtype()));
}

// Whether a caller parameter is a per-channel vector this operator can use.
bool param_ok(const std::optional<Tensor>& p, int64_t channels, int index) {
    if (!p.has_value() || !p->defined()) return true;
    return isFloatingType(p->dtype()) && p->numel() == channels && p->device().is_privateuse1() &&
           index_of(p->device()) == index;
}

// ---------------------------------- CNNL glue --------------------------------

// The identity activation, which is what a plain normalization uses.
class BnActivation {
public:
    BnActivation() {
        TP_CNNL_CHECK(cnnlCreateActivationDescriptor(&desc_));
        TP_CNNL_CHECK(cnnlSetActivationDescriptor(desc_, CNNL_ACTIVATION_IDENTITY,
                                                  CNNL_NOT_PROPAGATE_NAN, 0.0));
    }
    ~BnActivation() {
        if (desc_ != nullptr) cnnlDestroyActivationDescriptor(desc_);
    }
    BnActivation(const BnActivation&) = delete;
    BnActivation& operator=(const BnActivation&) = delete;

    cnnlActivationDescriptor_t get() const { return desc_; }

private:
    cnnlActivationDescriptor_t desc_ = nullptr;
};

// The channel-last order in which the card describes a lifted operand.
cnnlTensorLayout_t operand_layout(int64_t dim) {
    return dim == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
}

// ---------------------------------- forward ----------------------------------

// y (and, in training, the batch statistics) from the card.  ``x`` is float32,
// channel-last and lifted; every parameter is float32 of length ``channels``.
std::optional<std::tuple<Tensor, Tensor, Tensor>> forward_card(
        const Tensor& x, const Tensor& weight, const Tensor& bias,
        const Tensor& running_mean, const Tensor& running_var, bool training, double momentum,
        double eps, int64_t channels) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    const cnnlTensorLayout_t layout = operand_layout(x.dim());

    Tensor y = empty_on(x.sizes().vec(), DType::Float32, index);
    Tensor save_mean = training ? empty_on({channels}, DType::Float32, index)
                                : empty_on({0}, DType::Float32, index);
    Tensor save_invstd = training ? empty_on({channels}, DType::Float32, index)
                                  : empty_on({0}, DType::Float32, index);

    TensorDesc x_desc(x.sizes().vec(), layout, DType::Float32);
    TensorDesc y_desc(y.sizes().vec(), layout, DType::Float32);
    TensorDesc p_desc({channels}, CNNL_LAYOUT_ARRAY, DType::Float32);

    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetBatchNormForwardWorkspaceSize(h, x_desc.get(), &ws_size));
    Workspace ws(ws_size, index);

    BnActivation act;
    const float eps_f = static_cast<float>(eps);
    if (training) {
        TP_CNNL_TRY(cnnlBatchNormForwardTraining_v2(
            h, act.get(), CNNL_BATCHNORM_SPATIAL, CNNL_BATCHNORM_OPS_BN, nullptr, nullptr,
            x_desc.get(), x.data_ptr(), nullptr, nullptr, p_desc.get(), weight.data_ptr(),
            bias.data_ptr(), running_mean.data_ptr(), running_var.data_ptr(), eps_f,
            static_cast<float>(momentum), y_desc.get(), y.data_ptr(), save_mean.data_ptr(),
            save_invstd.data_ptr(), ws.size() == 0 ? nullptr : ws.get(), ws.size(), nullptr, 0));
    } else {
        TP_CNNL_TRY(cnnlBatchNormForwardInferenceV2(
            h, act.get(), CNNL_BATCHNORM_SPATIAL, CNNL_BATCHNORM_OPS_BN, nullptr, nullptr,
            x_desc.get(), x.data_ptr(), p_desc.get(), weight.data_ptr(), bias.data_ptr(),
            nullptr, nullptr, running_mean.data_ptr(), running_var.data_ptr(), eps_f,
            y_desc.get(), y.data_ptr()));
    }
    return std::make_tuple(y, save_mean, save_invstd);
}

// The three results of a normalization's forward pass.
struct Forward {
    Tensor out;
    Tensor save_mean;
    Tensor save_invstd;
};

// The operator's forward pass, or nothing when it cannot run on the card.
std::optional<Forward> batch_norm_forward(const Tensor& input, const std::optional<Tensor>& weight,
                                          const std::optional<Tensor>& bias,
                                          const std::optional<Tensor>& running_mean,
                                          const std::optional<Tensor>& running_var, bool training,
                                          double momentum, double eps) {
    if (!input.device().is_privateuse1()) return std::nullopt;
    const int64_t rank = input.dim();
    if (rank < 2 || rank > 5) return std::nullopt;
    if (!bn_operand(input.dtype()) || input.numel() == 0) return std::nullopt;
    const int64_t channels = input.size(1);
    if (channels <= 0) return std::nullopt;
    const int index = index_of(input.device());

    const bool has_mean = running_mean.has_value() && running_mean->defined();
    const bool has_var = running_var.has_value() && running_var->defined();
    if (has_mean != has_var) return std::nullopt;  // the host reports the mismatch
    if (!training && !has_mean) return std::nullopt;  // and the missing statistics
    if (!param_ok(weight, channels, index) || !param_ok(bias, channels, index)) {
        return std::nullopt;
    }
    if (!param_ok(running_mean, channels, index) || !param_ok(running_var, channels, index)) {
        return std::nullopt;
    }

    Tensor x = channel_last(to_dtype(lift_rank(input), DType::Float32));
    Tensor w = channel_param(weight, channels, index, 1.0);
    Tensor b = channel_param(bias, channels, index, 0.0);
    Tensor rm = channel_scratch(running_mean, channels, index);
    Tensor rv = channel_scratch(running_var, channels, index);

    auto card = forward_card(x, w, b, rm, rv, training, momentum, eps, channels);
    if (!card) return std::nullopt;
    Tensor y = std::get<0>(*card);
    Tensor save_mean = std::get<1>(*card);
    Tensor save_invstd = std::get<2>(*card);

    // The card updated the scratch running statistics as it normalizes; carry
    // them back into the caller's buffers.
    if (training && has_mean) {
        store_stats(*running_mean, rm);
        store_stats(*running_var, rv);
    }

    Forward out;
    out.out = lower_rank(channel_first(to_dtype(y, input.dtype())), input.sizes().vec());
    out.save_mean = save_mean;
    out.save_invstd = save_invstd;
    return out;
}

// ---------------------------------- backward ---------------------------------

// The three gradients from the card.  ``x`` and ``grad_out`` are float32,
// channel-last and lifted; ``weight`` and the statistics are float32 of length
// ``channels``.
std::optional<std::tuple<Tensor, Tensor, Tensor>> backward_card(
        const Tensor& x, const Tensor& grad_out, const Tensor& weight, const Tensor& running_mean,
        const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
        bool training, double eps, int64_t channels) {
    const int index = index_of(x.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    const cnnlTensorLayout_t layout = operand_layout(x.dim());

    Tensor dx = empty_on(x.sizes().vec(), DType::Float32, index);
    Tensor dw = empty_on({channels}, DType::Float32, index);
    Tensor db = empty_on({channels}, DType::Float32, index);

    TensorDesc x_desc(x.sizes().vec(), layout, DType::Float32);
    TensorDesc dy_desc(grad_out.sizes().vec(), layout, DType::Float32);
    TensorDesc dx_desc(dx.sizes().vec(), layout, DType::Float32);
    TensorDesc p_desc({channels}, CNNL_LAYOUT_ARRAY, DType::Float32);

    BnActivation act;
    const float eps_f = static_cast<float>(eps);
    if (training) {
        size_t ws_size = 0;
        TP_CNNL_TRY(cnnlGetBatchNormBackwardWorkspaceSize(h, x_desc.get(), &ws_size));
        Workspace ws(ws_size, index);
        TP_CNNL_TRY(cnnlBatchNormBackward_v2(
            h, act.get(), CNNL_BATCHNORM_SPATIAL, CNNL_BATCHNORM_OPS_BN, nullptr, nullptr, nullptr,
            nullptr, x_desc.get(), x.data_ptr(), nullptr, nullptr, dy_desc.get(),
            grad_out.data_ptr(), p_desc.get(), weight.data_ptr(), nullptr, save_mean.data_ptr(),
            save_invstd.data_ptr(), eps_f, nullptr, nullptr, dx_desc.get(), dx.data_ptr(),
            dw.data_ptr(), db.data_ptr(), ws.size() == 0 ? nullptr : ws.get(), ws.size(), nullptr,
            0));
    } else {
        size_t ws_size = 0;
        TP_CNNL_TRY(cnnlGetFrozenBatchNormBackwardWorkspaceSize(h, x_desc.get(), &ws_size));
        Workspace ws(ws_size, index);
        TP_CNNL_TRY(cnnlFrozenBatchNormBackward_v2(
            h, act.get(), CNNL_BATCHNORM_SPATIAL, CNNL_BATCHNORM_OPS_BN, x_desc.get(),
            x.data_ptr(), nullptr, nullptr, dy_desc.get(), grad_out.data_ptr(), p_desc.get(),
            weight.data_ptr(), nullptr, running_mean.data_ptr(), running_var.data_ptr(), eps_f,
            ws.size() == 0 ? nullptr : ws.get(), ws.size(), nullptr, nullptr, dx_desc.get(),
            dx.data_ptr(), dw.data_ptr(), db.data_ptr()));
    }
    return std::make_tuple(dx, dw, db);
}

// Which gradient slots a backward call asked for.
struct GradRequest {
    bool input = true;
    bool weight = false;
    bool bias = false;
};

GradRequest decode_mask(const std::vector<bool>* mask, bool has_weight) {
    if (mask == nullptr) {
        // The plain backward always wants the input gradient and the parameter
        // gradients its weight has.
        return {true, has_weight, has_weight};
    }
    return {mask->empty() || (mask->size() > 0 && (*mask)[0]),
            has_weight && mask->size() > 1 && (*mask)[1], mask->size() > 2 && (*mask)[2]};
}

// The operator's backward pass, or nothing when it cannot run on the card.
std::optional<std::tuple<Tensor, Tensor, Tensor>> batch_norm_backward(
        const Tensor& grad_out, const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var,
        const std::optional<Tensor>& save_mean, const std::optional<Tensor>& save_invstd,
        bool training, double eps, const std::vector<bool>* mask) {
    if (!input.device().is_privateuse1()) return std::nullopt;
    const int64_t rank = input.dim();
    if (rank < 2 || rank > 5) return std::nullopt;
    if (!bn_operand(input.dtype()) || input.numel() == 0) return std::nullopt;
    if (grad_out.dim() != rank || !bn_operand(grad_out.dtype())) return std::nullopt;
    const int64_t channels = input.size(1);
    if (channels <= 0) return std::nullopt;
    const int index = index_of(input.device());
    if (!param_ok(weight, channels, index)) return std::nullopt;

    const bool has_mean = running_mean.has_value() && running_mean->defined();
    const bool has_var = running_var.has_value() && running_var->defined();
    if (has_mean != has_var) return std::nullopt;
    if (!training && !has_mean) return std::nullopt;
    if (!param_ok(running_mean, channels, index) || !param_ok(running_var, channels, index)) {
        return std::nullopt;
    }

    const bool has_weight = weight.has_value() && weight->defined();
    const GradRequest want = decode_mask(mask, has_weight);

    Tensor x = channel_last(to_dtype(lift_rank(input), DType::Float32));
    Tensor dy = channel_last(to_dtype(lift_rank(grad_out), DType::Float32));
    Tensor w = channel_param(weight, channels, index, 1.0);
    Tensor rm = channel_scratch(running_mean, channels, index);
    Tensor rv = channel_scratch(running_var, channels, index);

    Tensor sm;
    Tensor si;
    if (training) {
        const bool saved = save_mean.has_value() && save_mean->defined() &&
                           save_invstd.has_value() && save_invstd->defined() &&
                           save_mean->numel() == channels && save_invstd->numel() == channels;
        if (saved) {
            sm = to_dtype(contiguous_on_device(*save_mean), DType::Float32);
            si = to_dtype(contiguous_on_device(*save_invstd), DType::Float32);
        } else if (save_mean.has_value() || save_invstd.has_value()) {
            return std::nullopt;  // the host reports the invalid statistics
        } else {
            // The plain backward carries no batch statistics; they are taken
            // again from a training forward pass, exactly as the operator
            // defines them.
            Tensor ones = channel_param(std::nullopt, channels, index, 1.0);
            Tensor zeros = channel_param(std::nullopt, channels, index, 0.0);
            auto stats = forward_card(x, ones, zeros, rm, rv, true, 0.0, eps, channels);
            if (!stats) return std::nullopt;
            sm = std::get<1>(*stats);
            si = std::get<2>(*stats);
        }
    }

    auto card = backward_card(x, dy, w, rm, rv, sm, si, training, eps, channels);
    if (!card) return std::nullopt;
    Tensor dx = std::get<0>(*card);
    Tensor dw = std::get<1>(*card);
    Tensor db = std::get<2>(*card);

    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (want.input) {
        grad_input = lower_rank(channel_first(to_dtype(dx, grad_out.dtype())), input.sizes().vec());
    }
    if (want.weight) grad_weight = to_dtype(dw, weight->dtype());
    if (want.bias) grad_bias = to_dtype(db, has_weight ? weight->dtype() : DType::Float32);
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ---------------------------------- kernels ----------------------------------

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_kernel(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, bool training, double momentum, double eps) {
    if (auto r = batch_norm_forward(input, weight, bias, running_mean, running_var, training,
                                    momentum, eps)) {
        return std::make_tuple(r->out, r->save_mean, r->save_invstd);
    }
    return cpu_fallback::Kernel<"native_batch_norm", sig::native_batch_norm>::call(
        input, weight, bias, running_mean, running_var, training, momentum, eps);
}

Tensor batch_norm_kernel(const Tensor& input, const std::optional<Tensor>& weight,
                         const std::optional<Tensor>& bias,
                         const std::optional<Tensor>& running_mean,
                         const std::optional<Tensor>& running_var, bool training, double momentum,
                         double eps) {
    if (auto r = batch_norm_forward(input, weight, bias, running_mean, running_var, training,
                                    momentum, eps)) {
        return r->out;
    }
    return cpu_fallback::Kernel<"batch_norm", sig::batch_norm>::call(
        input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward_kernel(
        const Tensor& grad_out, const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var,
        const std::optional<Tensor>& save_mean, const std::optional<Tensor>& save_invstd,
        bool train, double eps, const std::vector<bool>& output_mask) {
    if (auto r = batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean,
                                     save_invstd, train, eps, &output_mask)) {
        return *r;
    }
    return cpu_fallback::Kernel<"native_batch_norm_backward",
                                sig::native_batch_norm_backward>::call(
        grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps,
        output_mask);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_kernel(
        const Tensor& grad_out, const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var,
        bool training, double eps) {
    if (auto r = batch_norm_backward(grad_out, input, weight, running_mean, running_var,
                                     std::nullopt, std::nullopt, training, eps, nullptr)) {
        return *r;
    }
    return cpu_fallback::Kernel<"batch_norm_backward", sig::batch_norm_backward>::call(
        grad_out, input, weight, running_mean, running_var, training, eps);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUBatchNormKernels) {
    m.impl("batch_norm", static_cast<sig::batch_norm*>(&batch_norm_kernel));
    m.impl("batch_norm_backward",
           static_cast<sig::batch_norm_backward*>(&batch_norm_backward_kernel));
    m.impl("native_batch_norm", static_cast<sig::native_batch_norm*>(&native_batch_norm_kernel));
    m.impl("native_batch_norm_backward",
           static_cast<sig::native_batch_norm_backward*>(&native_batch_norm_backward_kernel));
}

}  // namespace mlu
}  // namespace tensorplay
