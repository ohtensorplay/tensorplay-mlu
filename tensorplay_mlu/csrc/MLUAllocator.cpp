// Caching allocator for MLU memory.
//
// cnrtMalloc and cnrtFree are slow and cnrtFree may synchronize the device,
// so freed blocks are kept per device and handed out again.  All work on a
// device is issued to one queue, so a block freed by the host can be reused
// by the next allocation without waiting: anything that later writes to it
// is queued behind every kernel that read it.
//
// Sizes are rounded up (512 bytes below 1 MiB, 2 MiB above) and a cached
// block is reused for a request within a quarter of its size.  When the
// device runs out of memory the cache is returned to the driver and the
// allocation retried once.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <mutex>

#include "Allocator.h"
#include "MLUAllocator.h"
#include "MLURuntime.h"

namespace tensorplay {
namespace mlu {

namespace {

constexpr int kMaxDevices = 64;
constexpr size_t kSmallLimit = size_t(1) << 20;
constexpr size_t kSmallRound = 512;
constexpr size_t kLargeRound = size_t(2) << 20;

size_t round_size(size_t nbytes) {
    if (nbytes <= kSmallLimit) return (nbytes + kSmallRound - 1) / kSmallRound * kSmallRound;
    return (nbytes + kLargeRound - 1) / kLargeRound * kLargeRound;
}

struct Block {
    void* ptr;
    size_t size;
    int device;
};

struct DevicePool {
    std::multimap<size_t, Block*> free_blocks;
    MemoryStats stats;
};

class CachingAllocator final : public Allocator {
public:
    DataPtr allocate(size_t nbytes) const override {
        return allocate(nbytes, Device(DeviceType::PrivateUse1, current_device()));
    }

    DataPtr allocate(size_t nbytes, const Device& device) const override {
        const int index = index_of(device);
        const Device placed(DeviceType::PrivateUse1, index);
        if (nbytes == 0) return DataPtr(nullptr, nullptr, &release, placed);
        Block* block = take(index, round_size(nbytes));
        return DataPtr(block->ptr, block, &release, placed);
    }

    MemoryStats stats(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        return pools_.at(index).stats;
    }

    void reset_peak(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& stats = pools_.at(index).stats;
        stats.peak_allocated = stats.allocated;
        stats.peak_reserved = stats.reserved;
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int index = 0; index < device_count(); ++index) release_cached(index);
    }

private:
    static void release(void* ctx) {
        if (ctx == nullptr) return;
        instance().give_back(static_cast<Block*>(ctx));
    }

    Block* take(int index, size_t size) const {
        std::lock_guard<std::mutex> lock(mutex_);
        DevicePool& pool = pools_.at(index);
        auto it = pool.free_blocks.lower_bound(size);
        if (it != pool.free_blocks.end() && it->first <= size + size / 4) {
            Block* block = it->second;
            pool.free_blocks.erase(it);
            note_allocated(pool, block->size);
            return block;
        }
        void* ptr = nullptr;
        {
            DeviceGuard guard(index);
            if (cnrtMalloc(&ptr, size) != cnrtSuccess) {
                release_cached(index);
                ptr = nullptr;
                const cnrtRet_t retry = cnrtMalloc(&ptr, size);
                if (retry != cnrtSuccess) {
                    TP_THROW(RuntimeError, "MLU out of memory allocating ", size,
                             " bytes on device ", index, " (", pool.stats.allocated,
                             " bytes allocated, ", pool.stats.reserved,
                             " reserved by the cache): ", cnrtGetErrorStr(retry));
                }
            }
        }
        pool.stats.reserved += static_cast<int64_t>(size);
        pool.stats.peak_reserved = std::max(pool.stats.peak_reserved, pool.stats.reserved);
        Block* block = new Block{ptr, size, index};
        note_allocated(pool, size);
        return block;
    }

    void give_back(Block* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        DevicePool& pool = pools_.at(block->device);
        pool.stats.allocated -= static_cast<int64_t>(block->size);
        pool.free_blocks.emplace(block->size, block);
    }

    static void note_allocated(DevicePool& pool, size_t size) {
        pool.stats.allocated += static_cast<int64_t>(size);
        pool.stats.peak_allocated = std::max(pool.stats.peak_allocated, pool.stats.allocated);
    }

    // Caller holds mutex_.  Waits for the device so no queued kernel still
    // touches a block that goes back to the driver.
    void release_cached(int index) const {
        DevicePool& pool = pools_.at(index);
        if (pool.free_blocks.empty()) return;
        DeviceGuard guard(index);
        cnrtQueueSync(queue(index));
        for (auto& [size, block] : pool.free_blocks) {
            cnrtFree(block->ptr);
            pool.stats.reserved -= static_cast<int64_t>(size);
            delete block;
        }
        pool.free_blocks.clear();
    }

public:
    static CachingAllocator& instance() {
        // Never destroyed: tensors released during interpreter teardown
        // still return their blocks here.
        static auto* allocator = new CachingAllocator();
        return *allocator;
    }

private:
    mutable std::mutex mutex_;
    mutable std::array<DevicePool, kMaxDevices> pools_;
};

static ::tensorplay::detail::AllocatorRegisterer g_register_allocator(
    DeviceType::PrivateUse1, &CachingAllocator::instance());

} // namespace

Allocator* allocator() { return &CachingAllocator::instance(); }

MemoryStats memory_stats(int index) { return CachingAllocator::instance().stats(index); }

void reset_peak_memory_stats(int index) { CachingAllocator::instance().reset_peak(index); }

void empty_cache() { CachingAllocator::instance().empty_cache(); }

} // namespace mlu
} // namespace tensorplay
