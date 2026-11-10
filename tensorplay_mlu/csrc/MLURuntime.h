#pragma once

// MLU runtime glue: error checking, the per-device queue every kernel of
// this backend is ordered on, the per-thread CNNL handle bound to it, and a
// guard that makes a device current for the length of a call.

#include <cstddef>
#include <cstdint>
#include <utility>

#include <cnnl.h>
#include <cnrt.h>

#include "Device.h"
#include "Exception.h"

namespace tensorplay {
namespace mlu {

#define TP_CNRT_CHECK(expr)                                                        \
    do {                                                                           \
        const cnrtRet_t tp_cnrt_status = (expr);                                   \
        TP_CHECK(tp_cnrt_status == cnrtSuccess, "MLU runtime error: ",             \
                 cnrtGetErrorStr(tp_cnrt_status), " (", #expr, ")");                \
    } while (0)

#define TP_CNNL_CHECK(expr)                                                        \
    do {                                                                           \
        const cnnlStatus_t tp_cnnl_status = (expr);                                \
        TP_CHECK(tp_cnnl_status == CNNL_STATUS_SUCCESS, "CNNL error: ",            \
                 cnnlGetErrorString(tp_cnnl_status), " (", #expr, ")");             \
    } while (0)

// Whether a status means "this card cannot run this kernel for these
// arguments": which operators, data types and shapes CNNL implements
// depends on the hardware generation, and a kernel it declines has not
// written anything.  Every kernel of this backend treats such a status as a
// reason to run the operator on the host instead of failing the call.
inline bool cnnl_declined(cnnlStatus_t status) {
    return status == CNNL_STATUS_ARCH_MISMATCH || status == CNNL_STATUS_NOT_SUPPORTED;
}

// Run a CNNL call inside a function that returns an empty result when the
// card declines the kernel.  Any other failure is an error.
#define TP_CNNL_TRY(expr)                                                          \
    do {                                                                           \
        const cnnlStatus_t tp_cnnl_status = (expr);                                \
        if (cnnl_declined(tp_cnnl_status)) return {};                              \
        TP_CHECK(tp_cnnl_status == CNNL_STATUS_SUCCESS, "CNNL error: ",            \
                 cnnlGetErrorString(tp_cnnl_status), " (", #expr, ")");             \
    } while (0)

// Number of visible MLU devices (0 when the driver is absent).
int device_count() noexcept;

int current_device();
void set_device(int index);

// Compute capability of a device, as (major, minor).  Which data types the
// kernel library implements depends on it.
std::pair<int, int> capability(int index);

// Device index a tensor on this backend lives on (a negative index means
// the current device).
int index_of(const Device& device);

// Make ``index`` the current device for the guard's lifetime.
class DeviceGuard {
public:
    explicit DeviceGuard(int index);
    explicit DeviceGuard(const Device& device) : DeviceGuard(index_of(device)) {}
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_;
    int current_;
};

// The queue all work on ``index`` is issued to.  One queue per device keeps
// kernels, copies and allocator reuse in program order.
cnrtQueue_t queue(int index);

// This thread's CNNL handle for ``index``, bound to ``queue(index)``.  The
// caller must have made ``index`` current.
cnnlHandle_t handle(int index);

// Block until all work issued to the device's queue has finished.
void synchronize(int index);

// Byte copies between the host and a device, ordered on the device's queue;
// they return once the bytes have landed.
void copy_to_device(void* device_dst, const void* host_src, size_t nbytes, int index);
void copy_to_host(void* host_dst, const void* device_src, size_t nbytes, int index);
void copy_on_device(void* dst, const void* src, size_t nbytes, int index);

} // namespace mlu
} // namespace tensorplay
