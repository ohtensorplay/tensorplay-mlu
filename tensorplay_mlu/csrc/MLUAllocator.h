#pragma once

#include <cstdint>

#include "Allocator.h"

namespace tensorplay {
namespace mlu {

struct MemoryStats {
    int64_t allocated = 0;       // bytes held by live tensors
    int64_t reserved = 0;        // bytes obtained from the driver (live + cached)
    int64_t peak_allocated = 0;
    int64_t peak_reserved = 0;
};

// The backend's allocator (also registered for DeviceType::PrivateUse1).
Allocator* allocator();

MemoryStats memory_stats(int index);
void reset_peak_memory_stats(int index);
// Return every cached, unused block to the driver.
void empty_cache();

} // namespace mlu
} // namespace tensorplay
