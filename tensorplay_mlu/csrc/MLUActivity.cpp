// Kernel-level device tracing for the profiler through CNPAPI (see
// DeviceActivityHooks in Profiler.h): the kernels, copies and fills the MLU
// ran during a session, each joined to the operation that launched it.
//
// libcnpapi is opened at the first traced session, so the backend loads and
// runs without it; a session then reports per-operation device time only.
//
// Join: every operation dispatched to the device pushes its record slot as
// a CUSTOM0 external correlation id.  CNPAPI emits an external-correlation
// record pairing that id with the driver correlation id of each API call
// made meanwhile, and the kernel/copy/fill records carry the same driver
// correlation id.

#include <dlfcn.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cnpapi.h>

#include "Profiler.h"

namespace tensorplay {
namespace mlu {

namespace {

constexpr size_t kBufferBytes = size_t(8) << 20;
constexpr size_t kBufferAlign = 8;

struct Api {
    decltype(&cnpapiInit) init = nullptr;
    decltype(&cnpapiActivityEnable) enable = nullptr;
    decltype(&cnpapiActivityDisable) disable = nullptr;
    decltype(&cnpapiActivityRegisterCallbacks) register_callbacks = nullptr;
    decltype(&cnpapiActivityGetNextRecord) next_record = nullptr;
    decltype(&cnpapiActivityFlushAll) flush_all = nullptr;
    decltype(&cnpapiActivityPushExternalCorrelationId) push = nullptr;
    decltype(&cnpapiActivityPopExternalCorrelationId) pop = nullptr;
    decltype(&cnpapiGetTimestamp) timestamp = nullptr;
    decltype(&cnpapiGetResultString) result_string = nullptr;
};

// Buffers CNPAPI filled, with their valid sizes.
std::mutex buffers_mutex;
std::vector<std::pair<uint64_t*, size_t>> completed;

void buffer_requested(uint64_t** buffer, size_t* size, size_t* max_records) {
    *buffer = static_cast<uint64_t*>(std::aligned_alloc(kBufferAlign, kBufferBytes));
    *size = *buffer ? kBufferBytes : 0;
    *max_records = 0;
}

void buffer_completed(uint64_t* buffer, size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(buffers_mutex);
    completed.emplace_back(buffer, valid);
}

uint64_t steady_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

const cnpapiActivityType kKinds[] = {
    CNPAPI_ACTIVITY_TYPE_KERNEL, CNPAPI_ACTIVITY_TYPE_MEMCPY, CNPAPI_ACTIVITY_TYPE_MEMSET,
    // Driver calls are what external-correlation records are made for.
    CNPAPI_ACTIVITY_TYPE_CNDRV_API, CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION};

const char* copy_name(cnpapiActivityMemcpyType kind) {
    switch (kind) {
        case CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOD: return "Memcpy HtoD";
        case CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOH: return "Memcpy DtoH";
        case CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOD: return "Memcpy DtoD";
        default: return "Memcpy";
    }
}

class CnpapiCollector final : public prof::DeviceActivityHooks {
public:
    bool start() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!load()) return false;
        names_.clear();
        {
            std::lock_guard<std::mutex> b(buffers_mutex);
            for (auto& entry : completed) std::free(entry.first);
            completed.clear();
        }
        // Map the collector's clock onto the steady clock op events use.
        const uint64_t before = steady_ns();
        const uint64_t device_now = api_.timestamp();
        const uint64_t after = steady_ns();
        offset_ = static_cast<int64_t>(before + (after - before) / 2) -
                  static_cast<int64_t>(device_now);
        for (cnpapiActivityType kind : kKinds) {
            const cnpapiResult r = api_.enable(kind);
            if (r != CNPAPI_SUCCESS) {
                error_ = std::string("enabling activity kind ") + std::to_string(kind) + ": " +
                         describe(r);
                for (cnpapiActivityType k : kKinds) api_.disable(k);
                return false;
            }
        }
        return true;
    }

    void stop_and_collect(std::vector<prof::GpuActivity>& out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_) return;
        api_.flush_all();
        for (cnpapiActivityType kind : kKinds) api_.disable(kind);
        std::vector<std::pair<uint64_t*, size_t>> buffers;
        {
            std::lock_guard<std::mutex> b(buffers_mutex);
            buffers.swap(completed);
        }
        std::unordered_map<uint64_t, uint64_t> external;  // correlation -> slot
        const size_t first = out.size();
        std::vector<uint64_t> correlations;
        for (const auto& [buffer, valid] : buffers) {
            cnpapiActivity* record = nullptr;
            while (api_.next_record(buffer, valid, &record) == CNPAPI_SUCCESS) {
                switch (record->type) {
                    case CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION: {
                        auto* r = reinterpret_cast<cnpapiActivityExternalCorrelation*>(record);
                        if (r->external_type == CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0) {
                            external.emplace(r->correlation_id, r->external_id);
                        }
                        break;
                    }
                    case CNPAPI_ACTIVITY_TYPE_KERNEL: {
                        auto* r = reinterpret_cast<cnpapiActivityKernel*>(record);
                        names_.emplace_back(r->name ? r->name : "kernel");
                        out.push_back(make(names_.back().c_str(), 'k', r->start, r->end,
                                           r->device_id, r->queue_id, 0));
                        correlations.push_back(r->correlation_id);
                        break;
                    }
                    case CNPAPI_ACTIVITY_TYPE_MEMCPY: {
                        auto* r = reinterpret_cast<cnpapiActivityMemcpy*>(record);
                        out.push_back(make(copy_name(r->copy_type), 'm', r->start, r->end,
                                           r->device_id, r->queue_id, r->bytes));
                        out.back().copy_kind = static_cast<uint8_t>(r->copy_type);
                        correlations.push_back(r->correlation_id);
                        break;
                    }
                    case CNPAPI_ACTIVITY_TYPE_MEMSET: {
                        auto* r = reinterpret_cast<cnpapiActivityMemset*>(record);
                        out.push_back(make("Memset", 's', r->start, r->end, r->device_id,
                                           r->queue_id, r->bytes));
                        correlations.push_back(r->correlation_id);
                        break;
                    }
                    default:
                        break;
                }
            }
            std::free(buffer);
        }
        for (size_t i = first; i < out.size(); ++i) {
            const uint64_t correlation = correlations[i - first];
            out[i].correlation = static_cast<uint32_t>(correlation);
            auto it = external.find(correlation);
            if (it != external.end()) out[i].external_id = it->second;
        }
    }

    void push_external_id(uint64_t id) override {
        if (loaded_) api_.push(CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, id);
    }

    void pop_external_id() override {
        uint64_t last = 0;
        if (loaded_) api_.pop(CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, &last);
    }

    std::string last_error() const override { return error_; }

private:
    bool load() {
        if (loaded_) return true;
        if (!error_.empty()) return false;
        void* lib = dlopen("libcnpapi.so", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            const char* e = dlerror();
            error_ = std::string("libcnpapi.so not loadable: ") + (e ? e : "unknown");
            return false;
        }
        bool ok = true;
        auto bind = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
            if (fn == nullptr) {
                ok = false;
                error_ = std::string("libcnpapi.so lacks ") + name;
            }
        };
        bind(api_.init, "cnpapiInit");
        bind(api_.enable, "cnpapiActivityEnable");
        bind(api_.disable, "cnpapiActivityDisable");
        bind(api_.register_callbacks, "cnpapiActivityRegisterCallbacks");
        bind(api_.next_record, "cnpapiActivityGetNextRecord");
        bind(api_.flush_all, "cnpapiActivityFlushAll");
        bind(api_.push, "cnpapiActivityPushExternalCorrelationId");
        bind(api_.pop, "cnpapiActivityPopExternalCorrelationId");
        bind(api_.timestamp, "cnpapiGetTimestamp");
        bind(api_.result_string, "cnpapiGetResultString");
        if (!ok) return false;
        cnpapiResult r = api_.init();
        if (r == CNPAPI_SUCCESS) r = api_.register_callbacks(buffer_requested, buffer_completed);
        if (r != CNPAPI_SUCCESS) {
            error_ = "initializing CNPAPI: " + describe(r);
            return false;
        }
        loaded_ = true;
        return true;
    }

    std::string describe(cnpapiResult r) const {
        const char* text = nullptr;
        if (api_.result_string) api_.result_string(r, &text);
        return text ? text : std::to_string(static_cast<int>(r));
    }

    prof::GpuActivity make(const char* name, char kind, uint64_t start, uint64_t end,
                           uint64_t device, uint64_t queue, uint64_t bytes) const {
        prof::GpuActivity a{};
        a.name = name;
        a.kind = kind;
        a.start_ns = static_cast<uint64_t>(static_cast<int64_t>(start) + offset_);
        a.end_ns = static_cast<uint64_t>(static_cast<int64_t>(end) + offset_);
        a.external_id = prof::GpuActivity::kNoExt;
        a.device = static_cast<int32_t>(device);
        a.stream = static_cast<int32_t>(queue);
        a.bytes = bytes;
        return a;
    }

    std::mutex mutex_;
    Api api_;
    bool loaded_ = false;
    std::string error_;
    int64_t offset_ = 0;
    std::deque<std::string> names_;  // kernel names, valid until the next start
};

[[maybe_unused]] const bool registered = [] {
    prof::register_privateuse1_activity_hooks(new CnpapiCollector);
    return true;
}();

} // namespace

} // namespace mlu
} // namespace tensorplay
