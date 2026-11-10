// Elementwise unary math and activations on CNNL, with the backward
// kernels autograd calls for the activations.
//
// Each kernel returns an empty result when the operand's dtype or the card
// leaves it without a CNNL kernel, and the operator then runs on the host.

#include <algorithm>
#include <optional>
#include <string>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

bool native_operand(const Tensor& t) {
    return t.device().is_privateuse1() && is_native_float(t.dtype(), index_of(t.device()));
}

// Run ``launch(handle, x_desc, x, y_desc, y)`` from a dense copy of
// ``self`` into ``out`` (allocated when not given).
template <typename Launch>
std::optional<Tensor> unary(const Tensor& self, const std::optional<Tensor>& out, Launch launch) {
    const int index = index_of(self.device());
    const Tensor x = contiguous_on_device(self);
    Tensor y = out && out->is_contiguous() ? *out : empty_on(x.sizes().vec(), x.dtype(), index);
    if (y.numel() > 0) {
        DeviceGuard guard(index);
        TensorDesc x_desc(x), y_desc(y);
        TP_CNNL_TRY(launch(handle(index), x_desc.get(), x.data_ptr(), y_desc.get(),
                           y.data_ptr()));
    }
    if (out && !out->is_contiguous()) {
        strided_copy(*out, y);
        return *out;
    }
    return y;
}

// Activations run the accurate kernel variant.
cnnlActivationPreference_t preference(cnnlActivationMode_t, DType, int) {
    return CNNL_ACTIVATION_HIGH_PRECISION;
}

class ActivationDesc {
public:
    ActivationDesc(cnnlActivationMode_t mode, DType dtype, int index, float coef = 1.0f,
                   bool approximate = false) {
        TP_CNNL_CHECK(cnnlCreateActivationDescriptor(&desc_));
        status_ = cnnlSetActivationDescriptor_v6(desc_, mode, preference(mode, dtype, index),
                                                 CNNL_NOT_PROPAGATE_NAN, coef, 0, 0.0f, 0.0f,
                                                 false, approximate);
    }
    ~ActivationDesc() { cnnlDestroyActivationDescriptor(desc_); }
    ActivationDesc(const ActivationDesc&) = delete;
    ActivationDesc& operator=(const ActivationDesc&) = delete;
    operator cnnlActivationDescriptor_t() const { return desc_; }
    cnnlStatus_t status() const { return status_; }

private:
    cnnlActivationDescriptor_t desc_ = nullptr;
    cnnlStatus_t status_ = CNNL_STATUS_SUCCESS;
};

std::optional<Tensor> activation(const Tensor& self, cnnlActivationMode_t mode,
                                 const std::optional<Tensor>& out, bool approximate = false) {
    ActivationDesc desc(mode, self.dtype(), index_of(self.device()), 1.0f, approximate);
    TP_CNNL_TRY(desc.status());
    return unary(self, out, [&](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,
                                cnnlTensorDescriptor_t yd, void* y) {
        return cnnlActivationForward(h, desc, nullptr, xd, x, nullptr, yd, y);
    });
}

// grad_input from an activation's output (``y``) or input (``x``).
std::optional<Tensor> activation_backward(const Tensor& grad, const Tensor* y, const Tensor* x,
                                          cnnlActivationMode_t mode, bool approximate = false) {
    const int index = index_of(grad.device());
    ActivationDesc desc(mode, grad.dtype(), index, 1.0f, approximate);
    TP_CNNL_TRY(desc.status());
    const Tensor dy = contiguous_on_device(grad);
    const Tensor yc = y ? contiguous_on_device(*y) : Tensor();
    const Tensor xc = x ? contiguous_on_device(*x) : Tensor();
    Tensor dx = empty_on(dy.sizes().vec(), dy.dtype(), index);
    if (dx.numel() == 0) return dx;
    DeviceGuard guard(index);
    TensorDesc dy_desc(dy), dx_desc(dx);
    TensorDesc y_desc = y ? TensorDesc(yc) : TensorDesc();
    TensorDesc x_desc = x ? TensorDesc(xc) : TensorDesc();
    TP_CNNL_TRY(cnnlActivationBackward(handle(index), desc, nullptr,
                                       y ? y_desc.get() : nullptr,
                                       y ? yc.data_ptr() : nullptr, dy_desc, dy.data_ptr(),
                                       x ? x_desc.get() : nullptr,
                                       x ? xc.data_ptr() : nullptr, nullptr, dx_desc,
                                       dx.data_ptr()));
    return dx;
}

#define TP_MLU_UNARY(NAME, CALL, OP, SYM)                                                  \
    Tensor NAME##_kernel(const Tensor& self) {                                             \
        if (native_operand(self)) {                                                        \
            auto result = unary(self, std::nullopt,                                        \
                                [](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,     \
                                   cnnlTensorDescriptor_t yd, void* y) {                   \
                                    return CALL(h, xd, x, yd, y);                          \
                                });                                                        \
            if (result) return *result;                                                    \
        }                                                                                  \
        return cpu_fallback::Kernel<OP, sig::SYM>::call(self);                             \
    }

TP_MLU_UNARY(exp, cnnlExp, "exp", exp)
TP_MLU_UNARY(sqrt, cnnlSqrt, "sqrt", sqrt)
TP_MLU_UNARY(rsqrt, cnnlRsqrt, "rsqrt", rsqrt)
TP_MLU_UNARY(neg, cnnlNegTensor, "neg", neg)
TP_MLU_UNARY(abs, cnnlAbs, "abs", abs)
#undef TP_MLU_UNARY

Tensor log_kernel(const Tensor& self) {
    if (native_operand(self)) {
        auto result = unary(self, std::nullopt, [](cnnlHandle_t h, cnnlTensorDescriptor_t xd,
                                                   void* x, cnnlTensorDescriptor_t yd, void* y) {
            return cnnlLog(h, CNNL_LOG_E, xd, x, yd, y);
        });
        if (result) return *result;
    }
    return cpu_fallback::Kernel<"log", sig::log>::call(self);
}

// self ** exponent.  The common exponents take their own kernels (a square
// is what a squared-error loss and its backward ask for); any other runs
// the general power with a broadcast exponent.
std::optional<Tensor> pow_scalar_native(const Tensor& self, double exponent) {
    if (exponent == 1.0) return contiguous_on_device(self).clone();
    if (exponent == 0.0) {
        Tensor ones = empty_on(self.sizes().vec(), self.dtype(), index_of(self.device()));
        ones.fill_(Scalar(1.0));
        return ones;
    }
    using Desc = cnnlTensorDescriptor_t;
    if (exponent == 2.0) {
        return unary(self, std::nullopt, [](cnnlHandle_t h, Desc xd, void* x, Desc yd, void* y) {
            return cnnlSquare(h, xd, x, yd, y);
        });
    }
    if (exponent == 0.5) {
        return unary(self, std::nullopt, [](cnnlHandle_t h, Desc xd, void* x, Desc yd, void* y) {
            return cnnlSqrt(h, xd, x, yd, y);
        });
    }
    if (exponent == -0.5) {
        return unary(self, std::nullopt, [](cnnlHandle_t h, Desc xd, void* x, Desc yd, void* y) {
            return cnnlRsqrt(h, xd, x, yd, y);
        });
    }
    const int index = index_of(self.device());
    const Tensor x = contiguous_on_device(self);
    Tensor z = empty_on(x.sizes().vec(), x.dtype(), index);
    if (z.numel() == 0) return z;
    const size_t ndim = std::max<size_t>(x.dim(), 1);
    Tensor e = empty_on(std::vector<int64_t>(ndim, 1), x.dtype(), index);
    e.fill_(Scalar(exponent));
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    TensorDesc x_desc = padded_desc(x, ndim), e_desc(e), z_desc = padded_desc(z, ndim);
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetPowWorkspaceSize(h, x_desc, e_desc, z_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_TRY(cnnlPow(h, CNNL_COMPUTATION_HIGH_PRECISION, x_desc, x.data_ptr(), e_desc,
                        e.data_ptr(), ws.get(), ws.size(), z_desc, z.data_ptr()));
    return z;
}

Tensor pow_scalar_kernel(const Tensor& self, const Scalar& exponent) {
    if (native_operand(self) && (exponent.isFloatingPoint() || exponent.isIntegral(false))) {
        if (auto result = pow_scalar_native(self, exponent.toDouble())) return *result;
    }
    return cpu_fallback::Kernel<"pow.Tensor_Scalar", sig::pow__Tensor_Scalar>::call(self, exponent);
}

#define TP_MLU_ACTIVATION(NAME, MODE, OP, SYM)                                              \
    Tensor NAME##_kernel(const Tensor& self) {                                              \
        if (native_operand(self)) {                                                         \
            if (auto result = activation(self, MODE, std::nullopt)) return *result;         \
        }                                                                                   \
        return cpu_fallback::Kernel<OP, sig::SYM>::call(self);                              \
    }

TP_MLU_ACTIVATION(relu, CNNL_ACTIVATION_RELU, "relu", relu)
TP_MLU_ACTIVATION(sigmoid, CNNL_ACTIVATION_SIGMOID, "sigmoid", sigmoid)
TP_MLU_ACTIVATION(tanh, CNNL_ACTIVATION_TANH, "tanh", tanh)
TP_MLU_ACTIVATION(silu, CNNL_ACTIVATION_SILU, "silu", silu)
#undef TP_MLU_ACTIVATION

Tensor& relu__kernel(Tensor& self) {
    if (native_operand(self) && activation(self, CNNL_ACTIVATION_RELU, self)) return self;
    return cpu_fallback::Kernel<"relu_", sig::relu_>::call(self);
}

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
    if (native_operand(self)) {
        auto result = activation(self, CNNL_ACTIVATION_GELU, std::nullopt,
                                 approximate == "tanh");
        if (result) return *result;
    }
    return cpu_fallback::Kernel<"gelu", sig::gelu>::call(self, approximate);
}

// Both operands of a backward kernel must be native and agree in dtype.
bool native_backward(const Tensor& grad, const Tensor& other) {
    return native_operand(grad) && native_operand(other) && grad.dtype() == other.dtype() &&
           grad.sizes().vec() == other.sizes().vec();
}

std::optional<Tensor> threshold_backward_native(const Tensor& grad_output, const Tensor& self,
                                                const Scalar& threshold) {
    const int index = index_of(self.device());
    const Tensor x = contiguous_on_device(self);
    const Tensor dy = contiguous_on_device(grad_output);
    Tensor dx = empty_on(x.sizes().vec(), x.dtype(), index);
    if (dx.numel() == 0) return dx;
    alignas(8) unsigned char value[8] = {};
    switch (x.dtype()) {
        case DType::Float16: *reinterpret_cast<Half*>(value) = Half(threshold.to<float>()); break;
        case DType::BFloat16:
            *reinterpret_cast<BFloat16*>(value) = BFloat16(threshold.to<float>());
            break;
        default: *reinterpret_cast<float*>(value) = threshold.to<float>(); break;
    }
    DeviceGuard guard(index);
    TensorDesc x_desc(x), dy_desc(dy), dx_desc(dx);
    TP_CNNL_TRY(cnnlThresholdBackward(handle(index), x_desc, x.data_ptr(), dy_desc, dy.data_ptr(),
                                      value, dx_desc, dx.data_ptr()));
    return dx;
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& self,
                                 const Scalar& threshold) {
    if (native_backward(grad_output, self)) {
        if (auto r = threshold_backward_native(grad_output, self, threshold)) return *r;
    }
    return cpu_fallback::Kernel<"threshold_backward", sig::threshold_backward>::call(
        grad_output, self, threshold);
}

Tensor sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output) {
    if (native_backward(grad_output, output)) {
        auto r = activation_backward(grad_output, &output, nullptr, CNNL_ACTIVATION_SIGMOID);
        if (r) return *r;
    }
    return cpu_fallback::Kernel<"sigmoid_backward", sig::sigmoid_backward>::call(grad_output,
                                                                                  output);
}

Tensor tanh_backward_kernel(const Tensor& grad_output, const Tensor& output) {
    if (native_backward(grad_output, output)) {
        auto r = activation_backward(grad_output, &output, nullptr, CNNL_ACTIVATION_TANH);
        if (r) return *r;
    }
    return cpu_fallback::Kernel<"tanh_backward", sig::tanh_backward>::call(grad_output, output);
}

Tensor gelu_backward_kernel(const Tensor& grad_output, const Tensor& self,
                            const std::string& approximate) {
    if (native_backward(grad_output, self)) {
        auto r = activation_backward(grad_output, nullptr, &self, CNNL_ACTIVATION_GELU,
                                     approximate == "tanh");
        if (r) return *r;
    }
    return cpu_fallback::Kernel<"gelu_backward", sig::gelu_backward>::call(grad_output, self,
                                                                           approximate);
}

Tensor silu_backward_kernel(const Tensor& grad_output, const Tensor& self) {
    if (native_backward(grad_output, self)) {
        auto r = activation_backward(grad_output, nullptr, &self, CNNL_ACTIVATION_SILU);
        if (r) return *r;
    }
    return cpu_fallback::Kernel<"silu_backward", sig::silu_backward>::call(grad_output, self);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUUnaryKernels) {
    m.impl("exp", static_cast<sig::exp*>(&exp_kernel));
    m.impl("log", static_cast<sig::log*>(&log_kernel));
    m.impl("sqrt", static_cast<sig::sqrt*>(&sqrt_kernel));
    m.impl("rsqrt", static_cast<sig::rsqrt*>(&rsqrt_kernel));
    m.impl("neg", static_cast<sig::neg*>(&neg_kernel));
    m.impl("abs", static_cast<sig::abs*>(&abs_kernel));
    m.impl("pow.Tensor_Scalar", static_cast<sig::pow__Tensor_Scalar*>(&pow_scalar_kernel));
    m.impl("relu", static_cast<sig::relu*>(&relu_kernel));
    m.impl("relu_", static_cast<sig::relu_*>(&relu__kernel));
    m.impl("sigmoid", static_cast<sig::sigmoid*>(&sigmoid_kernel));
    m.impl("tanh", static_cast<sig::tanh*>(&tanh_kernel));
    m.impl("silu", static_cast<sig::silu*>(&silu_kernel));
    m.impl("gelu", static_cast<sig::gelu*>(&gelu_kernel));
    m.impl("threshold_backward", static_cast<sig::threshold_backward*>(&threshold_backward_kernel));
    m.impl("sigmoid_backward", static_cast<sig::sigmoid_backward*>(&sigmoid_backward_kernel));
    m.impl("tanh_backward", static_cast<sig::tanh_backward*>(&tanh_backward_kernel));
    m.impl("gelu_backward", static_cast<sig::gelu_backward*>(&gelu_backward_kernel));
    m.impl("silu_backward", static_cast<sig::silu_backward*>(&silu_backward_kernel));
}

} // namespace mlu
} // namespace tensorplay
