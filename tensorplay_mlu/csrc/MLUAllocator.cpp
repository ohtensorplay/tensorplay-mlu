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
//
// A block handed out while a sequence of work is being recorded is marked:
// the recording holds its address, so once the block comes back it is set
// aside instead of being handed to anyone else.  Until the recording ends
// such a block may still be reused, because the recorded work keeps its
// order; see ``freeze_capture_blocks``.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

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
    // Handed out while a sequence was being recorded: it may be read or
    // written on every replay, so it is kept for as long as that recording.
    bool recorded = false;
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
            return note_recorded(block);
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
        Block* block = new Block{ptr, size, index, false};
        note_allocated(pool, size);
        return note_recorded(block);
    }

    void give_back(Block* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        DevicePool& pool = pools_.at(block->device);
        pool.stats.allocated -= static_cast<int64_t>(block->size);
        if (block->recorded) {
            // Recorded work still touches this block on every replay, so it
            // waits here instead of going back into the pool.
            held_.push_back(block);
            return;
        }
        pool.free_blocks.emplace(block->size, block);
    }

public:
    void set_capture_holding(bool on) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (on) capture_allocations_.store(0, std::memory_order_relaxed);
        holding_.store(on, std::memory_order_release);
    }

    // Blocks handed out during a recording may be reused while the recording
    // still runs -- the recorded work keeps its order -- but not after it
    // ends, so the ones that are unused leave the pool here.
    void freeze_capture_blocks() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (DevicePool& pool : pools_) {
            for (auto it = pool.free_blocks.begin(); it != pool.free_blocks.end();) {
                if (it->second->recorded) {
                    held_.push_back(it->second);
                    it = pool.free_blocks.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    void drop_capture_blocks() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (held_.empty()) return;
        for (int index = 0; index < device_count(); ++index) {
            DeviceGuard guard(index);
            cnrtQueueSync(queue(index));
        }
        for (Block* block : held_) {
            DeviceGuard guard(block->device);
            cnrtFree(block->ptr);
            pools_.at(block->device).stats.reserved -= static_cast<int64_t>(block->size);
            delete block;
        }
        held_.clear();
    }

    int64_t capture_allocations() const {
        return capture_allocations_.load(std::memory_order_relaxed);
    }

    int64_t capture_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int64_t>(held_.size());
    }

private:
    // Caller holds mutex_.
    Block* note_recorded(Block* block) const {
        if (!holding_.load(std::memory_order_acquire)) return block;
        block->recorded = true;
        capture_allocations_.fetch_add(1, std::memory_order_relaxed);
        return block;
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
    // Recording support: blocks handed out while a sequence is recorded, and
    // the ones set aside because a recording still touches them.
    mutable std::atomic<bool> holding_{false};
    mutable std::vector<Block*> held_;
    mutable std::atomic<int64_t> capture_allocations_{0};
};

static ::tensorplay::detail::AllocatorRegisterer g_register_allocator(
    DeviceType::PrivateUse1, &CachingAllocator::instance());

} // namespace

Allocator* allocator() { return &CachingAllocator::instance(); }

MemoryStats memory_stats(int index) { return CachingAllocator::instance().stats(index); }

void reset_peak_memory_stats(int index) { CachingAllocator::instance().reset_peak(index); }

void empty_cache() { CachingAllocator::instance().empty_cache(); }

void set_capture_holding(bool on) { CachingAllocator::instance().set_capture_holding(on); }

void freeze_capture_blocks() { CachingAllocator::instance().freeze_capture_blocks(); }

void drop_capture_blocks() { CachingAllocator::instance().drop_capture_blocks(); }

int64_t capture_allocations() { return CachingAllocator::instance().capture_allocations(); }

int64_t capture_blocks() { return CachingAllocator::instance().capture_blocks(); }

} // namespace mlu
} // namespace tensorplay
