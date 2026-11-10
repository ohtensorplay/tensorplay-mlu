// Device-level hooks for the MLU slot: device count, current device,
// synchronization, byte copies and memory information.

#include <cstring>

#include "MLURuntime.h"
#include "PrivateUse1Hooks.h"

namespace tensorplay {
namespace mlu {

namespace {

struct MLUHooks final : PrivateUse1HooksInterface {
    bool is_available() const override { return device_count() > 0; }
    int64_t device_count() const override { return mlu::device_count(); }
    int64_t current_device() const override { return mlu::current_device(); }
    void set_device(int64_t index) const override { mlu::set_device(static_cast<int>(index)); }
    void synchronize(int64_t index) const override {
        mlu::synchronize(index < 0 ? mlu::current_device() : static_cast<int>(index));
    }

    void copy_bytes(void* dst, const Device& dst_device, const void* src,
                    const Device& src_device, size_t nbytes) const override {
        const bool dst_mlu = dst_device.is_privateuse1();
        const bool src_mlu = src_device.is_privateuse1();
        if (dst_mlu && src_mlu) {
            const int dst_index = index_of(dst_device);
            const int src_index = index_of(src_device);
            if (dst_index == src_index) {
                copy_on_device(dst, src, nbytes, dst_index);
                return;
            }
            // Between two cards: through the host.
            std::unique_ptr<char[]> bounce(new char[nbytes]);
            copy_to_host(bounce.get(), src, nbytes, src_index);
            copy_to_device(dst, bounce.get(), nbytes, dst_index);
        } else if (dst_mlu) {
            copy_to_device(dst, src, nbytes, index_of(dst_device));
        } else if (src_mlu) {
            copy_to_host(dst, src, nbytes, index_of(src_device));
        } else {
            std::memcpy(dst, src, nbytes);
        }
    }

    std::pair<size_t, size_t> memory_info(int64_t index) const override {
        DeviceGuard guard(index < 0 ? mlu::current_device() : static_cast<int>(index));
        size_t free_bytes = 0, total_bytes = 0;
        TP_CNRT_CHECK(cnrtMemGetInfo(&free_bytes, &total_bytes));
        return {free_bytes, total_bytes};
    }
};

MLUHooks g_hooks;
REGISTER_PRIVATEUSE1_HOOKS(&g_hooks);

} // namespace

} // namespace mlu
} // namespace tensorplay
