#pragma once

// Tensor descriptors and helpers shared by the CNNL kernels.

#include <cstdint>
#include <optional>
#include <vector>

#include "MLUAllocator.h"
#include "MLURuntime.h"
#include "Tensor.h"

namespace tensorplay {
namespace mlu {

constexpr int kMaxCnnlDims = 8;

// CNNL element type for ``dtype``; nullopt when CNNL has none.
std::optional<cnnlDataType_t> cnnl_dtype(DType dtype);

// The floating types the native math kernels compute in on device
// ``index``.  bfloat16 arithmetic needs a 500-series card or newer; an
// older one runs those operators on the host.
inline bool is_native_float(DType dtype, int index) {
    if (dtype == DType::Float32 || dtype == DType::Float16) return true;
    return dtype == DType::BFloat16 && capability(index).first >= 5;
}

// Owning descriptor.  Sizes and strides come from the tensor (a 0-d tensor
// is described as one element); ``layout`` defaults to an array.
class TensorDesc {
public:
    TensorDesc() = default;
    explicit TensorDesc(const Tensor& t);
    TensorDesc(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
               DType dtype);
    // Descriptor for a tensor described in one of CNNL's named orders (the
    // sizes are given in that order and the strides are inferred as
    // contiguous).  The convolution operators only accept these orders.
    TensorDesc(const std::vector<int64_t>& sizes, cnnlTensorLayout_t layout, DType dtype);
    ~TensorDesc();
    TensorDesc(TensorDesc&& other) noexcept : desc_(other.desc_) { other.desc_ = nullptr; }
    TensorDesc& operator=(TensorDesc&& other) noexcept;
    TensorDesc(const TensorDesc&) = delete;
    TensorDesc& operator=(const TensorDesc&) = delete;

    cnnlTensorDescriptor_t get() const { return desc_; }
    operator cnnlTensorDescriptor_t() const { return desc_; }

private:
    void set(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
             DType dtype);
    void set_layout(const std::vector<int64_t>& sizes, cnnlTensorLayout_t layout, DType dtype);
    cnnlTensorDescriptor_t desc_ = nullptr;
};

// Device scratch memory from the caching allocator, released on scope exit.
class Workspace {
public:
    Workspace(size_t nbytes, int index);
    void* get() const { return const_cast<void*>(static_cast<const void*>(data_.get())); }
    size_t size() const { return size_; }

private:
    DataPtr data_;
    size_t size_;
};

// Allocate an uninitialised contiguous tensor on MLU device ``index``.
Tensor empty_on(const std::vector<int64_t>& sizes, DType dtype, int index);

// ``t`` itself when contiguous, otherwise a contiguous copy on its device.
Tensor contiguous_on_device(const Tensor& t);

// Copy ``src`` into ``dst`` on one device (same dtype, same shape; either
// may be strided).
void strided_copy(const Tensor& dst, const Tensor& src);

// ``t`` (an MLU tensor) converted to ``dtype`` on its device; ``t`` itself
// when it already has that dtype.
Tensor to_dtype(const Tensor& t, DType dtype);

// Element type two operands compute in: the dimensioned operand's type wins
// over a 0-d one of the same kind, otherwise the usual promotion.
DType binary_result_dtype(const Tensor& a, const Tensor& b);

// Broadcast shape of two operands.
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

// Descriptor of ``t`` with leading size-1 dimensions added up to ``ndim``.
TensorDesc padded_desc(const Tensor& t, size_t ndim);

// True when the tensor's elements are packed without gaps in some order
// of its dimensions (contiguous, or a permutation of contiguous).
bool is_dense(const Tensor& t);

} // namespace mlu
} // namespace tensorplay
