// Elementwise binary arithmetic on CNNL: add, sub, mul, div with
// broadcasting, their in-place forms, and the scalar forms (one affine
// transform).  Operands CNNL cannot compute in (integers, float64, mixed
// devices other than a host scalar) run through the host fallback.

#include <optional>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

enum class BinOp { Add, Sub, Mul, Div };

bool is_host_scalar(const Tensor& t) {
    return t.device().is_cpu() && t.numel() == 1 && t.dim() == 0;
}

// y = alpha * x + beta, written into ``out`` (same shape as x).  False when
// the card declines the kernel.  The transform reads and writes dense
// memory only: a strided operand (an expanded gradient, say) is made dense
// first and a strided output written through a dense result.
bool affine(const Tensor& x, float alpha, float beta, const Tensor& out) {
    const int index = index_of(out.device());
    DeviceGuard guard(index);
    const Tensor xc = x.is_contiguous() ? x : contiguous_on_device(x);
    const Tensor y = out.is_contiguous() ? out : empty_on(out.sizes().vec(), out.dtype(), index);
    TensorDesc x_desc(xc), y_desc(y);
    TP_CNNL_TRY(cnnlTransform_v2(handle(index), CNNL_POINTER_MODE_HOST, &alpha, x_desc,
                                 xc.data_ptr(), &beta, y_desc, y.data_ptr()));
    if (!out.is_contiguous()) strided_copy(out, y);
    return true;
}

// out = a (op) scale * b on the device (``scale`` applies to add and sub);
// both MLU, both of out's dtype.  False when the card declines the kernel.
bool launch_binary(const Tensor& a, const Tensor& b, BinOp op, const Tensor& out,
                   float scale = 1.0f) {
    const int index = index_of(out.device());
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    const size_t ndim = std::max<size_t>(out.dim(), 1);
    TensorDesc a_desc = padded_desc(a, ndim);
    TensorDesc b_desc = padded_desc(b, ndim);
    TensorDesc c_desc = padded_desc(out, ndim);
    if (op == BinOp::Div) {
        cnnlDivDescriptor_t div_desc = nullptr;
        TP_CNNL_CHECK(cnnlCreateDivDescriptor(&div_desc));
        struct DestroyDiv {
            cnnlDivDescriptor_t d;
            ~DestroyDiv() { cnnlDestroyDivDescriptor(d); }
        } destroy_div{div_desc};
        const int32_t mode = CNNL_DIV_TRUE;
        TP_CNNL_TRY(cnnlSetDivDescAttr(div_desc, CNNL_DIV_MODE, &mode, sizeof(mode)));
        size_t ws_size = 0;
        TP_CNNL_TRY(cnnlGetDivWorkspaceSize_v2(h, div_desc, a_desc, b_desc, c_desc, &ws_size));
        Workspace ws(ws_size, index);
        TP_CNNL_TRY(cnnlDiv_v3(h, div_desc, a_desc, a.data_ptr(), b_desc, b.data_ptr(), ws.get(),
                               ws.size(), c_desc, out.data_ptr()));
        return true;
    }
    cnnlOpTensorDescriptor_t op_desc = nullptr;
    TP_CNNL_CHECK(cnnlCreateOpTensorDescriptor(&op_desc));
    struct Destroy {
        cnnlOpTensorDescriptor_t d;
        ~Destroy() { cnnlDestroyOpTensorDescriptor(d); }
    } destroy{op_desc};
    const cnnlOpTensorDesc_t kind = op == BinOp::Mul ? CNNL_OP_TENSOR_MUL : CNNL_OP_TENSOR_ADD;
    TP_CNNL_TRY(cnnlSetOpTensorDescriptor(op_desc, kind, CNNL_DTYPE_FLOAT,
                                          CNNL_NOT_PROPAGATE_NAN));
    size_t ws_size = 0;
    TP_CNNL_TRY(cnnlGetOpTensorWorkspaceSize(h, a_desc, b_desc, c_desc, &ws_size));
    Workspace ws(ws_size, index);
    const float alpha1 = 1.0f;
    const float alpha2 = op == BinOp::Mul ? 1.0f : op == BinOp::Sub ? -scale : scale;
    const float beta = 0.0f;
    TP_CNNL_TRY(cnnlOpTensor(h, op_desc, &alpha1, a_desc, a.data_ptr(), &alpha2, b_desc,
                             b.data_ptr(), ws.get(), ws.size(), &beta, c_desc, out.data_ptr()));
    return true;
}

// The native path; nullopt when the operands need the host.  ``out``, when
// given, receives the result in place (it must already have the result's
// shape and dtype).
std::optional<Tensor> binary(const Tensor& a, const Tensor& b, double alpha, BinOp op,
                             const std::optional<Tensor>& out) {
    const bool a_mlu = a.device().is_privateuse1();
    const bool b_mlu = b.device().is_privateuse1();
    if (!(a_mlu || is_host_scalar(a)) || !(b_mlu || is_host_scalar(b))) return std::nullopt;
    if (!a_mlu && !b_mlu) return std::nullopt;
    const Tensor& anchor = a_mlu ? a : b;
    const int index = index_of(anchor.device());
    const DType dtype = binary_result_dtype(a, b);
    if (!is_native_float(dtype, index)) return std::nullopt;

    std::vector<int64_t> shape = broadcast_shape(a.sizes().vec(), b.sizes().vec());
    if (out) {
        if (out->dtype() != dtype || out->sizes().vec() != shape ||
            !out->device().is_privateuse1()) {
            return std::nullopt;
        }
    }
    Tensor result = out ? *out : empty_on(shape, dtype, index);
    if (result.numel() == 0) return result;

    // A host scalar folds into an affine transform of the other operand.
    if (!a_mlu || !b_mlu) {
        const double s = (a_mlu ? b : a).item().toDouble();
        const Tensor x = to_dtype(a_mlu ? a : b, dtype);
        if (x.sizes().vec() != shape) return std::nullopt;
        bool ok = false;
        if (a_mlu) {
            switch (op) {
                case BinOp::Add: ok = affine(x, 1.0f, static_cast<float>(alpha * s), result); break;
                case BinOp::Sub: ok = affine(x, 1.0f, static_cast<float>(-alpha * s), result); break;
                case BinOp::Mul: ok = affine(x, static_cast<float>(alpha * s), 0.0f, result); break;
                case BinOp::Div: ok = affine(x, static_cast<float>(1.0 / s), 0.0f, result); break;
            }
        } else {
            switch (op) {
                case BinOp::Add:
                    ok = affine(x, static_cast<float>(alpha), static_cast<float>(s), result);
                    break;
                case BinOp::Sub:
                    ok = affine(x, static_cast<float>(-alpha), static_cast<float>(s), result);
                    break;
                case BinOp::Mul: ok = affine(x, static_cast<float>(alpha * s), 0.0f, result); break;
                case BinOp::Div: ok = false; break;
            }
        }
        if (!ok) return std::nullopt;
        return result;
    }
    if (!launch_binary(to_dtype(a, dtype), to_dtype(b, dtype), op, result,
                       static_cast<float>(alpha))) {
        return std::nullopt;
    }
    return result;
}

Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (auto r = binary(self, other, alpha.toDouble(), BinOp::Add, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"add.Tensor", sig::add__Tensor>::call(self, other, alpha);
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (auto r = binary(self, other, alpha.toDouble(), BinOp::Sub, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"sub.Tensor", sig::sub__Tensor>::call(self, other, alpha);
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    if (auto r = binary(self, other, 1.0, BinOp::Mul, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"mul.Tensor", sig::mul__Tensor>::call(self, other);
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    if (auto r = binary(self, other, 1.0, BinOp::Div, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"div.Tensor", sig::div__Tensor>::call(self, other);
}

Tensor& add__kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (binary(self, other, alpha.toDouble(), BinOp::Add, self)) return self;
    return cpu_fallback::Kernel<"add_.Tensor", sig::add___Tensor>::call(self, other, alpha);
}

Tensor& sub__kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (binary(self, other, alpha.toDouble(), BinOp::Sub, self)) return self;
    return cpu_fallback::Kernel<"sub_.Tensor", sig::sub___Tensor>::call(self, other, alpha);
}

Tensor& mul__kernel(Tensor& self, const Tensor& other) {
    if (binary(self, other, 1.0, BinOp::Mul, self)) return self;
    return cpu_fallback::Kernel<"mul_.Tensor", sig::mul___Tensor>::call(self, other);
}

Tensor& div__kernel(Tensor& self, const Tensor& other) {
    if (binary(self, other, 1.0, BinOp::Div, self)) return self;
    return cpu_fallback::Kernel<"div_.Tensor", sig::div___Tensor>::call(self, other);
}

// Scalar forms: one affine transform when the tensor is a native float.
std::optional<Tensor> scalar_affine(const Tensor& self, double alpha, double beta,
                                    const std::optional<Tensor>& out) {
    if (!self.device().is_privateuse1() ||
        !is_native_float(self.dtype(), index_of(self.device()))) {
        return std::nullopt;
    }
    Tensor result = out ? *out
                        : empty_on(self.sizes().vec(), self.dtype(), index_of(self.device()));
    if (result.numel() > 0 &&
        !affine(self, static_cast<float>(alpha), static_cast<float>(beta), result)) {
        return std::nullopt;
    }
    return result;
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (auto r = scalar_affine(self, 1.0, alpha.toDouble() * other.toDouble(), std::nullopt)) {
        return *r;
    }
    return cpu_fallback::Kernel<"add.Scalar", sig::add__Scalar>::call(self, other, alpha);
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (auto r = scalar_affine(self, 1.0, -alpha.toDouble() * other.toDouble(), std::nullopt)) {
        return *r;
    }
    return cpu_fallback::Kernel<"sub.Scalar", sig::sub__Scalar>::call(self, other, alpha);
}

Tensor rsub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (auto r = scalar_affine(self, -alpha.toDouble(), other.toDouble(), std::nullopt)) {
        return *r;
    }
    return cpu_fallback::Kernel<"rsub.Scalar", sig::rsub__Scalar>::call(self, other, alpha);
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
    if (auto r = scalar_affine(self, other.toDouble(), 0.0, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"mul.Scalar", sig::mul__Scalar>::call(self, other);
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
    if (auto r = scalar_affine(self, 1.0 / other.toDouble(), 0.0, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"div.Scalar", sig::div__Scalar>::call(self, other);
}

Tensor& add__scalar_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (scalar_affine(self, 1.0, alpha.toDouble() * other.toDouble(), self)) return self;
    return cpu_fallback::Kernel<"add_.Scalar", sig::add___Scalar>::call(self, other, alpha);
}

Tensor& mul__scalar_kernel(Tensor& self, const Scalar& other) {
    if (scalar_affine(self, other.toDouble(), 0.0, self)) return self;
    return cpu_fallback::Kernel<"mul_.Scalar", sig::mul___Scalar>::call(self, other);
}

Tensor& div__scalar_kernel(Tensor& self, const Scalar& other) {
    if (scalar_affine(self, 1.0 / other.toDouble(), 0.0, self)) return self;
    return cpu_fallback::Kernel<"div_.Scalar", sig::div___Scalar>::call(self, other);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUBinaryKernels) {
    m.impl("add.Tensor", static_cast<sig::add__Tensor*>(&add_kernel));
    m.impl("sub.Tensor", static_cast<sig::sub__Tensor*>(&sub_kernel));
    m.impl("mul.Tensor", static_cast<sig::mul__Tensor*>(&mul_kernel));
    m.impl("div.Tensor", static_cast<sig::div__Tensor*>(&div_kernel));
    m.impl("add_.Tensor", static_cast<sig::add___Tensor*>(&add__kernel));
    m.impl("sub_.Tensor", static_cast<sig::sub___Tensor*>(&sub__kernel));
    m.impl("mul_.Tensor", static_cast<sig::mul___Tensor*>(&mul__kernel));
    m.impl("div_.Tensor", static_cast<sig::div___Tensor*>(&div__kernel));
    m.impl("add.Scalar", static_cast<sig::add__Scalar*>(&add_scalar_kernel));
    m.impl("sub.Scalar", static_cast<sig::sub__Scalar*>(&sub_scalar_kernel));
    m.impl("rsub.Scalar", static_cast<sig::rsub__Scalar*>(&rsub_scalar_kernel));
    m.impl("mul.Scalar", static_cast<sig::mul__Scalar*>(&mul_scalar_kernel));
    m.impl("div.Scalar", static_cast<sig::div__Scalar*>(&div_scalar_kernel));
    m.impl("add_.Scalar", static_cast<sig::add___Scalar*>(&add__scalar_kernel));
    m.impl("mul_.Scalar", static_cast<sig::mul___Scalar*>(&mul__scalar_kernel));
    m.impl("div_.Scalar", static_cast<sig::div___Scalar*>(&div__scalar_kernel));
}

} // namespace mlu
} // namespace tensorplay
