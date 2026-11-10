// Device time for the profiler: a notifier placed on the device's queue
// before and after each operation (see DeviceTimerHooks in Profiler.h).
// tensorplay.profiler.profile(activities=[ProfilerActivity.PrivateUse1]) or
// profile(use_device="mlu") then reports per-operation MLU time.
//
// Notifiers are pooled: a session records two per operation, and creating
// one is a driver call the dispatch path should not pay for each time.

#include <mutex>
#include <vector>

#include "MLURuntime.h"
#include "Profiler.h"

namespace tensorplay {
namespace mlu {

namespace {

class NotifierTimer final : public prof::DeviceTimerHooks {
public:
    int current_device() const override {
        int index = 0;
        return cnrtGetDevice(&index) == cnrtSuccess ? index : 0;
    }

    void* record(int device) const override {
        if (device < 0 || device >= device_count()) return nullptr;
        cnrtNotifier_t notifier = take();
        if (notifier == nullptr) return nullptr;
        try {
            DeviceGuard guard(device);
            if (cnrtPlaceNotifier(notifier, queue(device)) == cnrtSuccess) return notifier;
        } catch (...) {
        }
        give_back(notifier);
        return nullptr;
    }

    bool elapsed_ms(void* start, void* end, float* ms) const override {
        auto* s = static_cast<cnrtNotifier_t>(start);
        auto* e = static_cast<cnrtNotifier_t>(end);
        if (cnrtWaitNotifier(e) != cnrtSuccess) return false;
        return cnrtNotifierElapsedTime(s, e, ms) == cnrtSuccess;
    }

    void release(void* event) const override {
        if (event != nullptr) give_back(static_cast<cnrtNotifier_t>(event));
    }

private:
    cnrtNotifier_t take() const {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!pool_.empty()) {
                cnrtNotifier_t n = pool_.back();
                pool_.pop_back();
                return n;
            }
        }
        cnrtNotifier_t n = nullptr;
        return cnrtNotifierCreate(&n) == cnrtSuccess ? n : nullptr;
    }

    void give_back(cnrtNotifier_t n) const {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(n);
    }

    mutable std::mutex mutex_;
    mutable std::vector<cnrtNotifier_t> pool_;
};

// Leaked with the pool: a notifier must not be destroyed while the driver
// is tearing down at exit.
[[maybe_unused]] const NotifierTimer* const timer = [] {
    auto* t = new NotifierTimer;
    prof::register_privateuse1_timer_hooks(t);
    return t;
}();

} // namespace

} // namespace mlu
} // namespace tensorplay
