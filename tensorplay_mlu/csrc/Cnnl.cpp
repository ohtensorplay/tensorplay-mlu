#include "Cnnl.h"

#include <algorithm>
#include <numeric>

#include "TypePromotion.h"

namespace tensorplay {
namespace mlu {

std::optional<cnnlDataType_t> cnnl_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32: return CNNL_DTYPE_FLOAT;
        case DType::Float16: return CNNL_DTYPE_HALF;
        case DType::BFloat16: return CNNL_DTYPE_BFLOAT16;
        case DType::Float64: return CNNL_DTYPE_DOUBLE;
        case DType::Int8: return CNNL_DTYPE_INT8;
        case DType::Int16: return CNNL_DTYPE_INT16;
        case DType::Int32: return CNNL_DTYPE_INT32;
        case DType::Int64: return CNNL_DTYPE_INT64;
        case DType::UInt8: return CNNL_DTYPE_UINT8;
        case DType::UInt16: return CNNL_DTYPE_UINT16;
        case DType::UInt32: return CNNL_DTYPE_UINT32;
        case DType::UInt64: return CNNL_DTYPE_UINT64;
        case DType::Bool: return CNNL_DTYPE_BOOL;
        case DType::ComplexFloat: return CNNL_DTYPE_COMPLEX_FLOAT;
        case DType::ComplexHalf: return CNNL_DTYPE_COMPLEX_HALF;
        default: return std::nullopt;
    }
}

TensorDesc::TensorDesc(const Tensor& t) {
    std::vector<int64_t> sizes = t.sizes().vec();
    std::vector<int64_t> strides = t.strides();
    set(sizes, strides, t.dtype());
}

TensorDesc::TensorDesc(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                       DType dtype) {
    set(sizes, strides, dtype);
}

void TensorDesc::set(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                     DType dtype) {
    const auto type = cnnl_dtype(dtype);
    TP_CHECK(type.has_value(), "MLU: dtype ", toString(dtype), " has no CNNL equivalent");
    TP_CHECK(sizes.size() <= static_cast<size_t>(kMaxCnnlDims),
             "MLU: CNNL describes at most ", kMaxCnnlDims, " dimensions, got ", sizes.size());
    TP_CNNL_CHECK(cnnlCreateTensorDescriptor(&desc_));
    if (sizes.empty()) {
        const int64_t one = 1;
        TP_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(desc_, CNNL_LAYOUT_ARRAY, *type, 1, &one, &one));
        return;
    }
    TP_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(desc_, CNNL_LAYOUT_ARRAY, *type,
                                               static_cast<int>(sizes.size()), sizes.data(),
                                               strides.data()));
}

TensorDesc::~TensorDesc() {
    if (desc_ != nullptr) cnnlDestroyTensorDescriptor(desc_);
}

TensorDesc& TensorDesc::operator=(TensorDesc&& other) noexcept {
    if (this != &other) {
        if (desc_ != nullptr) cnnlDestroyTensorDescriptor(desc_);
        desc_ = other.desc_;
        other.desc_ = nullptr;
    }
    return *this;
}

Workspace::Workspace(size_t nbytes, int index)
    : data_(nbytes == 0 ? DataPtr()
                        : allocator()->allocate(nbytes, Device(DeviceType::PrivateUse1, index))),
      size_(nbytes) {}

Tensor empty_on(const std::vector<int64_t>& sizes, DType dtype, int index) {
    int64_t numel = 1;
    for (int64_t d : sizes) numel *= d;
    const Device placed(DeviceType::PrivateUse1, index);
    Storage storage(static_cast<size_t>(numel) * elementSize(dtype), allocator(), placed);
    return Tensor(std::move(storage), sizes, dtype);
}

bool is_dense(const Tensor& t) {
    const auto sizes = t.sizes().vec();
    const auto strides = t.strides();
    std::vector<size_t> order(sizes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return strides[a] < strides[b]; });
    int64_t expected = 1;
    for (size_t i : order) {
        if (sizes[i] == 1) continue;
        if (strides[i] != expected) return false;
        expected *= sizes[i];
    }
    return true;
}

namespace {

// Merge dimensions that are laid out back to back in both tensors and drop
// size-1 dimensions, so deep but simple layouts fit CNNL's dimension limit.
void coalesce(std::vector<int64_t>& sizes, std::vector<int64_t>& a, std::vector<int64_t>& b) {
    std::vector<int64_t> s2, a2, b2;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 1) continue;
        if (!s2.empty() && a2.back() == a[i] * sizes[i] && b2.back() == b[i] * sizes[i]) {
            s2.back() *= sizes[i];
            a2.back() = a[i];
            b2.back() = b[i];
            continue;
        }
        s2.push_back(sizes[i]);
        a2.push_back(a[i]);
        b2.push_back(b[i]);
    }
    if (s2.empty()) {
        s2 = {1};
        a2 = {1};
        b2 = {1};
    }
    sizes = std::move(s2);
    a = std::move(a2);
    b = std::move(b2);
}

} // namespace

void strided_copy(const Tensor& dst, const Tensor& src) {
    TP_CHECK(dst.dtype() == src.dtype(), "MLU strided_copy: dtype mismatch");
    if (dst.numel() == 0) return;
    const int index = index_of(dst.device());
    std::vector<int64_t> sizes = dst.sizes().vec();
    TP_CHECK(sizes == src.sizes().vec(), "MLU strided_copy: shape mismatch");
    std::vector<int64_t> dst_strides = dst.strides();
    std::vector<int64_t> src_strides = src.strides();
    coalesce(sizes, dst_strides, src_strides);
    DeviceGuard guard(index);
    cnnlHandle_t h = handle(index);
    if (sizes.size() == 1 && dst_strides[0] == 1 && src_strides[0] == 1) {
        copy_on_device(dst.data_ptr(), src.data_ptr(),
                       static_cast<size_t>(sizes[0]) * elementSize(dst.dtype()), index);
        return;
    }
    TensorDesc src_desc(sizes, src_strides, src.dtype());
    TensorDesc dst_desc(sizes, dst_strides, dst.dtype());
    size_t ws_size = 0;
    TP_CNNL_CHECK(cnnlGetCopyWorkspaceSize(h, src_desc, dst_desc, &ws_size));
    Workspace ws(ws_size, index);
    TP_CNNL_CHECK(cnnlCopy_v2(h, src_desc, src.data_ptr(), dst_desc, dst.data_ptr(), ws.get(),
                              ws.size()));
}

DType binary_result_dtype(const Tensor& a, const Tensor& b) {
    if (a.dtype() == b.dtype()) return a.dtype();
    const bool a_scalar = a.dim() == 0;
    const bool b_scalar = b.dim() == 0;
    if (a_scalar != b_scalar) {
        const Tensor& full = a_scalar ? b : a;
        const Tensor& scalar = a_scalar ? a : b;
        const bool same_kind = isFloatingType(full.dtype()) == isFloatingType(scalar.dtype()) &&
                               (full.dtype() == DType::Bool) == (scalar.dtype() == DType::Bool);
        if (same_kind || (isFloatingType(full.dtype()) && !isFloatingType(scalar.dtype()))) {
            return full.dtype();
        }
    }
    return promoteTypes(a.dtype(), b.dtype());
}

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
    const size_t n = std::max(a.size(), b.size());
    std::vector<int64_t> out(n, 1);
    for (size_t i = 0; i < n; ++i) {
        const int64_t da = i < n - a.size() ? 1 : a[i - (n - a.size())];
        const int64_t db = i < n - b.size() ? 1 : b[i - (n - b.size())];
        TP_CHECK(da == db || da == 1 || db == 1, "shapes cannot be broadcast together");
        out[i] = da == 1 ? db : da;
    }
    return out;
}

TensorDesc padded_desc(const Tensor& t, size_t ndim) {
    std::vector<int64_t> sizes = t.sizes().vec();
    std::vector<int64_t> strides = t.strides();
    while (sizes.size() < ndim) {
        sizes.insert(sizes.begin(), 1);
        strides.insert(strides.begin(), strides.empty() ? 1 : strides.front() * sizes[1]);
    }
    return TensorDesc(sizes, strides, t.dtype());
}

Tensor contiguous_on_device(const Tensor& t) {
    if (t.is_contiguous()) return t;
    Tensor out = empty_on(t.sizes().vec(), t.dtype(), index_of(t.device()));
    strided_copy(out, t);
    return out;
}

} // namespace mlu
} // namespace tensorplay
