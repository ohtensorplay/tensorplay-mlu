// Allocation, copies, casts and fills: the kernels every other kernel of the
// backend (and the host fallback) is built on.

#include <cstring>
#include <mutex>
#include <set>
#include <optional>
#include <utility>

#include "Cnnl.h"
#include "Dispatcher.h"
#include "LocalDispatchKeySet.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

int64_t numel_of(const std::vector<int64_t>& sizes) {
    int64_t n = 1;
    for (int64_t d : sizes) {
        TP_CHECK(d >= 0, "MLU: negative dimension ", d, " in a requested size");
        n *= d;
    }
    return n;
}

// Run the host copy kernel directly: the call that reached this kernel has
// already passed autograd.
void host_copy(Tensor& dst, const Tensor& src) {
    impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys());
    DispatchStub<Tensor&, Tensor&, const Tensor&, bool>::call("copy_", DispatchKey::CPU, dst,
                                                              src, false);
}

Tensor host_tensor(const std::vector<int64_t>& sizes, DType dtype) {
    return Tensor(Storage(static_cast<size_t>(numel_of(sizes)) * elementSize(dtype)), sizes,
                  dtype);
}

std::optional<cnnlCastDataType_t> cast_kind(DType from, DType to) {
#define TP_MLU_CAST(F, T, KIND) \
    if (from == DType::F && to == DType::T) return KIND;
    TP_MLU_CAST(Float32, Float16, CNNL_CAST_FLOAT_TO_HALF)
    TP_MLU_CAST(Float32, BFloat16, CNNL_CAST_FLOAT_TO_BFLOAT16)
    TP_MLU_CAST(Float32, Int32, CNNL_CAST_FLOAT_TO_INT32)
    TP_MLU_CAST(Float32, Int64, CNNL_CAST_FLOAT_TO_INT64)
    TP_MLU_CAST(Float32, Int16, CNNL_CAST_FLOAT_TO_INT16)
    TP_MLU_CAST(Float32, Int8, CNNL_CAST_FLOAT_TO_INT8)
    TP_MLU_CAST(Float32, UInt8, CNNL_CAST_FLOAT_TO_UINT8)
    TP_MLU_CAST(Float32, Bool, CNNL_CAST_FLOAT_TO_BOOL)
    TP_MLU_CAST(Float32, Float64, CNNL_CAST_FLOAT_TO_DOUBLE)
    TP_MLU_CAST(Float16, Float32, CNNL_CAST_HALF_TO_FLOAT)
    TP_MLU_CAST(Float16, Int32, CNNL_CAST_HALF_TO_INT32)
    TP_MLU_CAST(Float16, Int64, CNNL_CAST_HALF_TO_INT64)
    TP_MLU_CAST(Float16, Int16, CNNL_CAST_HALF_TO_INT16)
    TP_MLU_CAST(Float16, Int8, CNNL_CAST_HALF_TO_INT8)
    TP_MLU_CAST(Float16, UInt8, CNNL_CAST_HALF_TO_UINT8)
    TP_MLU_CAST(Float16, Bool, CNNL_CAST_HALF_TO_BOOL)
    TP_MLU_CAST(BFloat16, Float32, CNNL_CAST_BFLOAT16_TO_FLOAT)
    TP_MLU_CAST(BFloat16, Int32, CNNL_CAST_BFLOAT16_TO_INT32)
    TP_MLU_CAST(BFloat16, Int64, CNNL_CAST_BFLOAT16_TO_INT64)
    TP_MLU_CAST(BFloat16, Int16, CNNL_CAST_BFLOAT16_TO_INT16)
    TP_MLU_CAST(BFloat16, Int8, CNNL_CAST_BFLOAT16_TO_INT8)
    TP_MLU_CAST(BFloat16, UInt8, CNNL_CAST_BFLOAT16_TO_UINT8)
    TP_MLU_CAST(BFloat16, Bool, CNNL_CAST_BFLOAT16_TO_BOOL)
    TP_MLU_CAST(Int32, Float32, CNNL_CAST_INT32_TO_FLOAT)
    TP_MLU_CAST(Int32, Float16, CNNL_CAST_INT32_TO_HALF)
    TP_MLU_CAST(Int32, BFloat16, CNNL_CAST_INT32_TO_BFLOAT16)
    TP_MLU_CAST(Int32, Int64, CNNL_CAST_INT32_TO_INT64)
    TP_MLU_CAST(Int32, Int16, CNNL_CAST_INT32_TO_INT16)
    TP_MLU_CAST(Int32, Int8, CNNL_CAST_INT32_TO_INT8)
    TP_MLU_CAST(Int32, UInt8, CNNL_CAST_INT32_TO_UINT8)
    TP_MLU_CAST(Int32, Bool, CNNL_CAST_INT32_TO_BOOL)
    TP_MLU_CAST(Int64, Float32, CNNL_CAST_INT64_TO_FLOAT)
    TP_MLU_CAST(Int64, Float16, CNNL_CAST_INT64_TO_HALF)
    TP_MLU_CAST(Int64, BFloat16, CNNL_CAST_INT64_TO_BFLOAT16)
    TP_MLU_CAST(Int64, Int32, CNNL_CAST_INT64_TO_INT32)
    TP_MLU_CAST(Int64, Int16, CNNL_CAST_INT64_TO_INT16)
    TP_MLU_CAST(Int64, Bool, CNNL_CAST_INT64_TO_BOOL)
    TP_MLU_CAST(Int64, Float64, CNNL_CAST_INT64_TO_DOUBLE)
    TP_MLU_CAST(Int16, Float32, CNNL_CAST_INT16_TO_FLOAT)
    TP_MLU_CAST(Int16, Float16, CNNL_CAST_INT16_TO_HALF)
    TP_MLU_CAST(Int16, BFloat16, CNNL_CAST_INT16_TO_BFLOAT16)
    TP_MLU_CAST(Int16, Int32, CNNL_CAST_INT16_TO_INT32)
    TP_MLU_CAST(Int16, Int64, CNNL_CAST_INT16_TO_INT64)
    TP_MLU_CAST(Int16, Int8, CNNL_CAST_INT16_TO_INT8)
    TP_MLU_CAST(Int8, Float32, CNNL_CAST_INT8_TO_FLOAT)
    TP_MLU_CAST(Int8, Float16, CNNL_CAST_INT8_TO_HALF)
    TP_MLU_CAST(Int8, BFloat16, CNNL_CAST_INT8_TO_BFLOAT16)
    TP_MLU_CAST(Int8, Int32, CNNL_CAST_INT8_TO_INT32)
    TP_MLU_CAST(Int8, Int16, CNNL_CAST_INT8_TO_INT16)
    TP_MLU_CAST(UInt8, Float32, CNNL_CAST_UINT8_TO_FLOAT)
    TP_MLU_CAST(UInt8, Float16, CNNL_CAST_UINT8_TO_HALF)
    TP_MLU_CAST(UInt8, BFloat16, CNNL_CAST_UINT8_TO_BFLOAT16)
    TP_MLU_CAST(UInt8, Int32, CNNL_CAST_UINT8_TO_INT32)
    TP_MLU_CAST(UInt8, Int64, CNNL_CAST_UINT8_TO_INT64)
    TP_MLU_CAST(Bool, Float32, CNNL_CAST_BOOL_TO_FLOAT)
    TP_MLU_CAST(Bool, Float16, CNNL_CAST_BOOL_TO_HALF)
    TP_MLU_CAST(Bool, BFloat16, CNNL_CAST_BOOL_TO_BFLOAT16)
    TP_MLU_CAST(Bool, Int32, CNNL_CAST_BOOL_TO_INT32)
    TP_MLU_CAST(Bool, Int64, CNNL_CAST_BOOL_TO_INT64)
    TP_MLU_CAST(Float64, Float32, CNNL_CAST_DOUBLE_TO_FLOAT)
    TP_MLU_CAST(Float64, Int64, CNNL_CAST_DOUBLE_TO_INT64)
#undef TP_MLU_CAST
    return std::nullopt;
}

std::mutex g_cast_lock;
std::set<int> g_unsupported_casts;

bool cast_unsupported(cnnlCastDataType_t kind) {
    std::lock_guard<std::mutex> lock(g_cast_lock);
    return g_unsupported_casts.count(static_cast<int>(kind)) != 0;
}

void mark_cast_unsupported(cnnlCastDataType_t kind) {
    std::lock_guard<std::mutex> lock(g_cast_lock);
    g_unsupported_casts.insert(static_cast<int>(kind));
}

// Dense ``src`` converted into the dense ``dst`` of the same shape; false
// when the device has no conversion for the pair.
bool cast_into(const Tensor& dst, const Tensor& src) {
    const auto kind = cast_kind(src.dtype(), dst.dtype());
    if (!kind || cast_unsupported(*kind)) return false;
    const int index = index_of(dst.device());
    DeviceGuard guard(index);
    TensorDesc in_desc(src), out_desc(dst);
    const cnnlStatus_t status = cnnlCastDataType(handle(index), in_desc, src.data_ptr(), *kind,
                                                 out_desc, dst.data_ptr());
    if (status == CNNL_STATUS_NOT_SUPPORTED) {
        // Which pairs a card converts depends on its generation.
        mark_cast_unsupported(*kind);
        return false;
    }
    TP_CNNL_CHECK(status);
    return true;
}

// Contiguous ``src`` converted to ``to`` on its device; nullopt when the
// device has no conversion for the pair.
std::optional<Tensor> cast_on_device(const Tensor& src, DType to) {
    const Tensor input = contiguous_on_device(src);
    const int index = index_of(input.device());
    auto convert = [&](const Tensor& in, DType target) -> std::optional<Tensor> {
        Tensor out = empty_on(in.sizes().vec(), target, index);
        if (!cast_into(out, in)) return std::nullopt;
        return out;
    };
    if (auto direct = convert(input, to)) return direct;
    // Through float32 when both legs exist (e.g. half -> bfloat16).
    if (input.dtype() != DType::Float32 && to != DType::Float32 &&
        cast_kind(input.dtype(), DType::Float32) && cast_kind(DType::Float32, to)) {
        return convert(*convert(input, DType::Float32), to);
    }
    return std::nullopt;
}

Tensor to_host_dense(const Tensor& src) {
    const Tensor dense = contiguous_on_device(src);
    Tensor host = host_tensor(dense.sizes().vec(), dense.dtype());
    copy_to_host(host.data_ptr(), dense.data_ptr(), static_cast<size_t>(dense.numel()) *
                 elementSize(dense.dtype()), index_of(dense.device()));
    return host;
}

void host_to_device(Tensor& self, const Tensor& src) {
    // Shape and dtype are settled on the host, so only dense bytes move.
    Tensor host = src;
    if (host.sizes().vec() != self.sizes().vec()) host = host.expand(self.sizes().vec());
    if (host.dtype() != self.dtype()) host = host.to(self.dtype());
    host = host.contiguous();
    const size_t nbytes = static_cast<size_t>(self.numel()) * elementSize(self.dtype());
    const int index = index_of(self.device());
    if (self.is_contiguous()) {
        copy_to_device(self.data_ptr(), host.data_ptr(), nbytes, index);
        return;
    }
    Tensor staging = empty_on(self.sizes().vec(), self.dtype(), index);
    copy_to_device(staging.data_ptr(), host.data_ptr(), nbytes, index);
    strided_copy(self, staging);
}

void device_to_host(Tensor& self, const Tensor& src) {
    if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype() &&
        self.sizes().vec() == src.sizes().vec()) {
        copy_to_host(self.data_ptr(), src.data_ptr(),
                     static_cast<size_t>(self.numel()) * elementSize(self.dtype()),
                     index_of(src.device()));
        return;
    }
    host_copy(self, to_host_dense(src));
}

void device_to_device(Tensor& self, const Tensor& src) {
    const int index = index_of(self.device());
    if (index_of(src.device()) != index) {
        host_to_device(self, to_host_dense(src));
        return;
    }
    Tensor source = src;
    if (source.sizes().vec() != self.sizes().vec()) source = source.expand(self.sizes().vec());
    if (source.dtype() == self.dtype()) {
        strided_copy(self, source);
        return;
    }
    // A dense conversion writes the result once, straight into the
    // destination; anything else converts into a dense copy first.
    if (source.is_contiguous() && self.is_contiguous() &&
        source.sizes().vec() == self.sizes().vec() && cast_into(self, source)) {
        return;
    }
    if (auto converted = cast_on_device(source, self.dtype())) {
        strided_copy(self, *converted);
        return;
    }
    host_to_device(self, to_host_dense(source));
}

} // namespace

Tensor to_dtype(const Tensor& t, DType dtype) {
    if (t.dtype() == dtype) return t;
    Tensor out = empty_on(t.sizes().vec(), dtype, index_of(t.device()));
    if (out.numel() > 0) device_to_device(out, t);
    return out;
}

Tensor empty_kernel(const std::vector<int64_t>& size, std::optional<DType> dtype,
                    std::optional<Device> device, bool pin_memory) {
    TP_CHECK(!pin_memory, "MLU tensors cannot be pinned host memory");
    numel_of(size);
    const int index = device ? index_of(*device) : current_device();
    return empty_on(size, dtype.value_or(DType::Float32), index);
}

Tensor empty_memory_format_kernel(const std::vector<int64_t>& size, std::optional<DType> dtype,
                                  std::optional<int64_t> layout, std::optional<Device> device,
                                  std::optional<bool> pin_memory,
                                  std::optional<int64_t> memory_format) {
    (void)layout;
    (void)memory_format;
    return empty_kernel(size, dtype, device, pin_memory.value_or(false));
}

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    (void)non_blocking;
    if (self.numel() == 0) return self;
    const bool self_mlu = self.device().is_privateuse1();
    const bool src_mlu = src.device().is_privateuse1();
    TP_CHECK((self_mlu || self.device().is_cpu()) && (src_mlu || src.device().is_cpu()),
             "MLU copies run between MLU devices and the host, not ", src.device().toString(),
             " -> ", self.device().toString());
    if (self_mlu && src_mlu) {
        device_to_device(self, src);
    } else if (self_mlu) {
        host_to_device(self, src);
    } else {
        device_to_host(self, src);
    }
    return self;
}

Tensor& fill_kernel(Tensor& self, const Scalar& value) {
    if (self.numel() == 0) return self;
    const DType dtype = self.dtype();
    alignas(16) unsigned char bytes[16] = {};
    bool supported = true;
    switch (dtype) {
        case DType::Float32: *reinterpret_cast<float*>(bytes) = value.to<float>(); break;
        case DType::Float16: *reinterpret_cast<Half*>(bytes) = Half(value.to<float>()); break;
        case DType::BFloat16:
            *reinterpret_cast<BFloat16*>(bytes) = BFloat16(value.to<float>());
            break;
        case DType::Int8: *reinterpret_cast<int8_t*>(bytes) = value.to<int8_t>(); break;
        case DType::Int16: *reinterpret_cast<int16_t*>(bytes) = value.to<int16_t>(); break;
        case DType::Int32: *reinterpret_cast<int32_t*>(bytes) = value.to<int32_t>(); break;
        case DType::Int64: *reinterpret_cast<int64_t*>(bytes) = value.to<int64_t>(); break;
        case DType::UInt8: *reinterpret_cast<uint8_t*>(bytes) = value.to<uint8_t>(); break;
        case DType::Bool: *reinterpret_cast<bool*>(bytes) = value.to<bool>(); break;
        default: supported = false;
    }
    if (!supported) {
        Tensor host = host_tensor(self.sizes().vec(), dtype);
        impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys());
        DispatchStub<Tensor&, Tensor&, const Scalar&>::call("fill_.Scalar", DispatchKey::CPU,
                                                            host, value);
        host_to_device(self, host);
        return self;
    }
    const int index = index_of(self.device());
    DeviceGuard guard(index);
    TensorDesc desc(self);
    TP_CNNL_CHECK(cnnlFill_v3(handle(index), CNNL_POINTER_MODE_HOST, bytes, desc,
                              self.data_ptr()));
    return self;
}

Tensor& zero_kernel(Tensor& self) { return fill_kernel(self, Scalar(int64_t(0))); }

Tensor clone_kernel(const Tensor& self, std::optional<int64_t> memory_format) {
    (void)memory_format;
    Tensor out = empty_on(self.sizes().vec(), self.dtype(), index_of(self.device()));
    if (out.numel() > 0) strided_copy(out, self);
    return out;
}

Tensor contiguous_kernel(const Tensor& self, int64_t memory_format) {
    (void)memory_format;
    if (self.is_contiguous()) return self;
    return clone_kernel(self, std::nullopt);
}

Tensor empty_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    const DType type = dtype == DType::Undefined ? self.dtype() : dtype;
    if (device && !device->is_privateuse1()) {
        return Tensor(self.sizes().vec(), type, *device);
    }
    const int index = device ? index_of(*device) : index_of(self.device());
    return empty_on(self.sizes().vec(), type, index);
}

// ``*_like`` factories: allocate as empty_like does, then fill on the
// result's own device.
Tensor full_like_kernel(const Tensor& self, const Scalar& value, DType dtype,
                        std::optional<Device> device) {
    Tensor result = empty_like_kernel(self, dtype, device);
    result.fill_(value);
    return result;
}

Tensor zeros_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return full_like_kernel(self, Scalar(int64_t(0)), dtype, device);
}

Tensor ones_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    return full_like_kernel(self, Scalar(int64_t(1)), dtype, device);
}

Scalar item_kernel(const Tensor& self) {
    TP_CHECK(self.numel() == 1, "item() only supported for 1-element tensors");
    return to_host_dense(self).item();
}

namespace sig = kernel_signatures;

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUCopyKernels) {
    m.impl("empty_like", static_cast<sig::empty_like*>(&empty_like_kernel));
    m.impl("zeros_like", static_cast<sig::zeros_like*>(&zeros_like_kernel));
    m.impl("ones_like", static_cast<sig::ones_like*>(&ones_like_kernel));
    m.impl("full_like", static_cast<sig::full_like*>(&full_like_kernel));
    m.impl("item", static_cast<sig::item*>(&item_kernel));
    m.impl("_local_scalar_dense", static_cast<sig::_local_scalar_dense*>(&item_kernel));
    m.impl("empty", static_cast<sig::empty*>(&empty_kernel));
    m.impl("empty.memory_format",
           static_cast<sig::empty__memory_format*>(&empty_memory_format_kernel));
    m.impl("copy_", static_cast<sig::copy_*>(&copy_kernel));
    m.impl("fill_.Scalar", static_cast<sig::fill___Scalar*>(&fill_kernel));
    m.impl("zero_", static_cast<sig::zero_*>(&zero_kernel));
    m.impl("clone", static_cast<sig::clone*>(&clone_kernel));
    m.impl("contiguous", static_cast<sig::contiguous*>(&contiguous_kernel));
}

} // namespace mlu
} // namespace tensorplay
