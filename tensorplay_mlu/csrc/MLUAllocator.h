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

// Recording a sequence of device work bakes the addresses it touches into the
// recording, so those blocks may not be handed out again.  While a recording
// is running every block handed out is marked; ``freeze_capture_blocks`` moves
// the marked blocks that are currently unused out of the pool, and the ones
// still held by a tensor are set aside when they come back.
void set_capture_holding(bool on);
void freeze_capture_blocks();
void drop_capture_blocks();
// Blocks handed out while recording, and blocks set aside for a recording.
int64_t capture_allocations();
int64_t capture_blocks();

} // namespace mlu
} // namespace tensorplay
