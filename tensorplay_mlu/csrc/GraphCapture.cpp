// Recording a fixed sequence of device work so it replays in one call.
//
// A decode step issues the same operators on the same shapes every token, so
// the cost of issuing them is pure overhead.  The device can record the work
// queued on a queue into a task graph, instantiate that graph once, and then
// run the whole sequence with a single invocation.
//
// Three rules shape the implementation.
//
//   * A recording holds device work only.  Anything the host does while a
//     sequence is recorded -- running an operator on the processor, reading a
//     value back, synchronizing -- is not part of the recording, so a step
//     that falls back to the host cannot be recorded.
//   * Every buffer the recorded work touches has to stay where it is.  The
//     caller therefore keeps the values the recorded call returns alive, and
//     the allocator sets aside the blocks handed out during the recording
//     once it ends.
//   * The queue has to be idle when the recording starts, because the
//     recording does not run the work it takes in.

#include <exception>
#include <mutex>
#include <string>

#include "MLUAllocator.h"
#include "MLURuntime.h"

namespace tensorplay {
namespace mlu {
namespace {

std::mutex g_lock;
cnrtTaskTopo_t g_topo = nullptr;
cnrtTaskTopoEntity_t g_entity = nullptr;
thread_local std::string g_message;

template <typename Body>
int guarded(const Body& body) {
    try {
        body();
        return 0;
    } catch (const std::exception& error) {
        g_message = error.what();
        return -1;
    }
}

} // namespace
} // namespace mlu
} // namespace tensorplay

using namespace tensorplay::mlu;

extern "C" {

// Text of the failure the last call reported, empty when it succeeded.
const char* tp_mlu_last_error() { return g_message.c_str(); }

// Whether a recording is held and can be invoked.
int tp_mlu_capture_ready() { return g_entity != nullptr ? 1 : 0; }

// Blocks handed out while the recording ran, and blocks it holds.
void tp_mlu_capture_stats(int64_t* out) {
    out[0] = capture_allocations();
    out[1] = capture_blocks();
}

// Start recording on the current device's queue.
int tp_mlu_capture_begin(int index) {
    return guarded([&] {
        std::lock_guard<std::mutex> lock(g_lock);
        TP_CHECK(g_entity == nullptr && g_topo == nullptr,
                 "a recorded sequence is held already; drop it before recording another");
        DeviceGuard guard(index);
        TP_CNRT_CHECK(cnrtQueueBeginCapture(queue(index), cnrtQueueCaptureModeRelaxed));
        set_capture_holding(true);
    });
}

// Stop recording and make the sequence invokable.
int tp_mlu_capture_end(int index) {
    return guarded([&] {
        std::lock_guard<std::mutex> lock(g_lock);
        TP_CHECK(g_entity == nullptr, "a recorded sequence is held already");
        cnrtTaskTopo_t topo = nullptr;
        {
            DeviceGuard guard(index);
            const cnrtRet_t status = cnrtQueueEndCapture(queue(index), &topo);
            set_capture_holding(false);
            TP_CNRT_CHECK(status);
        }
        TP_CHECK(topo != nullptr,
                 "the device discarded the recording; a recorded sequence has to hold "
                 "device work only, and the step must not allocate");
        freeze_capture_blocks();
        cnrtTaskTopoEntity_t entity = nullptr;
        {
            DeviceGuard guard(index);
            TP_CNRT_CHECK(cnrtTaskTopoInstantiate(&entity, topo, nullptr, nullptr, 0));
        }
        g_topo = topo;
        g_entity = entity;
    });
}

// Queue one run of the recorded sequence.
int tp_mlu_capture_invoke(int index) {
    return guarded([&] {
        std::lock_guard<std::mutex> lock(g_lock);
        TP_CHECK(g_entity != nullptr, "nothing has been recorded yet");
        DeviceGuard guard(index);
        TP_CNRT_CHECK(cnrtTaskTopoEntityInvoke(g_entity, queue(index)));
    });
}

// Drop the recording and return the blocks it held.
int tp_mlu_capture_reset() {
    return guarded([&] {
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_entity != nullptr) {
            cnrtTaskTopoEntityDestroy(g_entity);
            g_entity = nullptr;
        }
        if (g_topo != nullptr) {
            cnrtTaskTopoDestroy(g_topo);
            g_topo = nullptr;
        }
        drop_capture_blocks();
    });
}

} // extern "C"
