#include "MLURuntime.h"

#include <array>
#include <atomic>
#include <mutex>

namespace tensorplay {
namespace mlu {

namespace {

constexpr int kMaxDevices = 64;

std::array<std::atomic<cnrtQueue_t>, kMaxDevices> g_queues{};
std::mutex g_queue_lock;

struct ThreadHandles {
    std::array<cnnlHandle_t, kMaxDevices> handles{};
    ~ThreadHandles() {
        for (cnnlHandle_t h : handles) {
            if (h != nullptr) cnnlDestroy(h);
        }
    }
};

void check_index(int index) {
    const int count = device_count();
    TP_CHECK(index >= 0 && index < count, "invalid MLU device index ", index,
             " (", count, " device(s) visible)");
}

} // namespace

int device_count() noexcept {
    static const int count = [] {
        unsigned int n = 0;
        if (cnrtGetDeviceCount(&n) != cnrtSuccess) return 0;
        return static_cast<int>(n < kMaxDevices ? n : kMaxDevices);
    }();
    return count;
}

int current_device() {
    int index = 0;
    TP_CNRT_CHECK(cnrtGetDevice(&index));
    return index;
}

void set_device(int index) {
    check_index(index);
    TP_CNRT_CHECK(cnrtSetDevice(index));
}

std::pair<int, int> capability(int index) {
    static std::array<std::atomic<int>, kMaxDevices> cached{};
    check_index(index);
    int packed = cached[index].load(std::memory_order_acquire);
    if (packed == 0) {
        int major = 0, minor = 0;
        DeviceGuard guard(index);
        TP_CNRT_CHECK(cnrtDeviceGetAttribute(&major, cnrtAttrComputeCapabilityMajor, index));
        TP_CNRT_CHECK(cnrtDeviceGetAttribute(&minor, cnrtAttrComputeCapabilityMinor, index));
        packed = (major << 16) | (minor & 0xffff);
        cached[index].store(packed, std::memory_order_release);
    }
    return {packed >> 16, packed & 0xffff};
}

int index_of(const Device& device) {
    return device.index() < 0 ? current_device() : static_cast<int>(device.index());
}

DeviceGuard::DeviceGuard(int index) : previous_(current_device()), current_(index) {
    if (current_ != previous_) set_device(current_);
}

DeviceGuard::~DeviceGuard() {
    if (current_ != previous_) cnrtSetDevice(previous_);
}

cnrtQueue_t queue(int index) {
    check_index(index);
    cnrtQueue_t q = g_queues[index].load(std::memory_order_acquire);
    if (q != nullptr) return q;
    std::lock_guard<std::mutex> lock(g_queue_lock);
    q = g_queues[index].load(std::memory_order_relaxed);
    if (q == nullptr) {
        DeviceGuard guard(index);
        TP_CNRT_CHECK(cnrtQueueCreate(&q));
        g_queues[index].store(q, std::memory_order_release);
    }
    return q;
}

cnnlHandle_t handle(int index) {
    thread_local ThreadHandles local;
    check_index(index);
    cnnlHandle_t& h = local.handles[index];
    if (h == nullptr) {
        DeviceGuard guard(index);
        TP_CNNL_CHECK(cnnlCreate(&h));
        TP_CNNL_CHECK(cnnlSetQueue(h, queue(index)));
    }
    return h;
}

void synchronize(int index) {
    DeviceGuard guard(index);
    TP_CNRT_CHECK(cnrtQueueSync(queue(index)));
}

void copy_to_device(void* device_dst, const void* host_src, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    cnrtQueue_t q = queue(index);
    TP_CNRT_CHECK(cnrtMemcpyAsync_V2(device_dst, const_cast<void*>(host_src), nbytes, q,
                                     cnrtMemcpyHostToDev));
    // The source is pageable host memory the caller may free on return.
    TP_CNRT_CHECK(cnrtQueueSync(q));
}

void copy_to_host(void* host_dst, const void* device_src, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    cnrtQueue_t q = queue(index);
    TP_CNRT_CHECK(cnrtMemcpyAsync_V2(host_dst, const_cast<void*>(device_src), nbytes, q,
                                     cnrtMemcpyDevToHost));
    TP_CNRT_CHECK(cnrtQueueSync(q));
}

void copy_on_device(void* dst, const void* src, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    TP_CNRT_CHECK(cnrtMemcpyAsync_V2(dst, const_cast<void*>(src), nbytes, queue(index),
                                     cnrtMemcpyDevToDev));
}

} // namespace mlu
} // namespace tensorplay
