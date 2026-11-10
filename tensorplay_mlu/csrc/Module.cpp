// C entry points the Python package calls through ctypes.

#include <cstdint>

#include "CpuFallback.h"
#include "MLUAllocator.h"
#include "MLURuntime.h"

using namespace tensorplay;

extern "C" {

// Called once after the library is loaded: every operator that still has
// no kernel for the device runs through the host.
int tp_mlu_init() {
    try {
        cpu_fallback::register_host_fallbacks(DispatchKey::PrivateUse1);
        return 0;
    } catch (...) {
        return 1;
    }
}

int tp_mlu_device_count() { return mlu::device_count(); }

// allocated, reserved, peak allocated, peak reserved (bytes).
void tp_mlu_memory_stats(int index, int64_t* out) {
    const mlu::MemoryStats stats = mlu::memory_stats(index);
    out[0] = stats.allocated;
    out[1] = stats.reserved;
    out[2] = stats.peak_allocated;
    out[3] = stats.peak_reserved;
}

void tp_mlu_reset_peak_memory_stats(int index) { mlu::reset_peak_memory_stats(index); }

void tp_mlu_empty_cache() { mlu::empty_cache(); }

} // extern "C"
