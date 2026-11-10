// Collective communication for the MLU device backend, on top of the CNCL
// library.
//
// Everything here is exported through a flat C ABI: the Python process-group
// object marshals tensors as device pointers, so a generated binding layer
// would only add a second place to describe the same arguments.  Each entry
// point returns 0 on success and a negative code on failure, with the message
// readable from ``tp_mlu_cncl_last_error``; no exception is allowed to cross
// the ``ctypes`` boundary.

#include <cstddef>
#include <cstdint>
#include <string>

#include <cncl.h>

#include "MLURuntime.h"

namespace tensorplay {
namespace mlu {
namespace {

thread_local std::string g_last_error;

void set_last_error(const char* message) {
    g_last_error = message == nullptr ? std::string() : std::string(message);
}

// Reduction codes and element-type codes are a private convention shared with
// ``tensorplay_mlu.distributed``; the values are chosen so the Python side
// never has to mirror the CNCL enumerations.
bool cncl_data_type(int code, cnclDataType_t* out) {
    switch (code) {
        case 0: *out = cnclUint8; return true;
        case 1: *out = cnclInt8; return true;
        case 2: *out = cnclInt16; return true;
        case 3: *out = cnclInt32; return true;
        case 4: *out = cnclInt64; return true;
        case 5: *out = cnclHalf; return true;
        case 6: *out = cnclFloat; return true;
        case 7: *out = cnclBfloat16; return true;
        default: return false;
    }
}

bool cncl_reduce_op(int code, cnclReduceOp_t* out) {
    switch (code) {
        case 0: *out = cnclSum; return true;
        case 1: *out = cnclProd; return true;
        case 2: *out = cnclMax; return true;
        case 3: *out = cnclMin; return true;
        default: return false;
    }
}

void check(cnclResult_t status, const char* what) {
    TP_CHECK(status == CNCL_RET_SUCCESS, "CNCL error: ", cnclGetErrorStr(status), " (", what,
             ")");
}

cnclComm_t as_comm(void* handle) {
    TP_CHECK(handle != nullptr, "CNCL: null communicator handle");
    return static_cast<cnclComm_t>(handle);
}

// Resolve the queue every collective of device ``index`` is ordered on.
cnrtQueue_t device_queue(int index) { return queue(index); }

} // namespace

extern "C" {

// Human-readable message for the most recent failure on this thread.
const char* tp_mlu_cncl_last_error() { return g_last_error.c_str(); }

// Bytes of an opaque clique identifier; the caller shares that many bytes
// verbatim to let every rank join the same communicator.
int tp_mlu_cncl_clique_id_bytes() { return static_cast<int>(sizeof(cnclCliqueId)); }

// Prepare ``out`` (``tp_mlu_cncl_clique_id_bytes()`` bytes) as a clique
// identifier.  Only the rank that hosts the rendezvous needs to call it.
int tp_mlu_cncl_get_clique_id(void* out) {
    try {
        TP_CHECK(out != nullptr, "CNCL: null clique identifier buffer");
        DeviceGuard guard(current_device());
        check(cnclGetCliqueId(static_cast<cnclCliqueId_t>(out)), "getCliqueId");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Create the communicator of ``rank`` (of ``nranks``) on device ``device``,
// joining the clique described by ``clique``.  ``*comm_out`` receives an
// opaque handle.  This call is collective and must be issued by every rank of
// the clique.
int tp_mlu_cncl_comm_init(int rank, int nranks, int device, const void* clique, void** comm_out) {
    try {
        TP_CHECK(comm_out != nullptr, "CNCL: null output communicator");
        TP_CHECK(rank >= 0 && rank < nranks, "CNCL: rank ", rank, " out of [0, ", nranks, ")");
        TP_CHECK(clique != nullptr, "CNCL: null clique identifier");
        DeviceGuard guard(device);
        cnclComm_t comm = nullptr;
        check(cnclInitComms(&comm, 1, &device, &rank, nranks,
                            static_cast<cnclCliqueId_t>(const_cast<void*>(clique))),
              "initComms");
        *comm_out = comm;
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Tear down a communicator once all of its work has completed.
int tp_mlu_cncl_comm_destroy(void* handle) {
    try {
        cnclComm_t comm = as_comm(handle);
        check(cnclDestroyComms(&comm, 1), "destroyComms");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Abort a communicator; in-flight work is not waited on.
int tp_mlu_cncl_comm_abort(void* handle) {
    try {
        check(cnclAbortComm(as_comm(handle)), "abortComm");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Accessors used when a caller has to decide what to exchange.
int tp_mlu_cncl_comm_rank(void* handle, int* rank_out) {
    try {
        check(cnclGetCommRank(rank_out, as_comm(handle)), "getCommRank");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_comm_count(void* handle, int* count_out) {
    try {
        check(cnclGetCommCount(count_out, as_comm(handle)), "getCommCount");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Queue the collectives on ``device``.  ``send``/``recv`` are device
// pointers; ``count`` is an element count in ``dtype``.
int tp_mlu_cncl_all_reduce(const void* send, void* recv, size_t count, int dtype, int op,
                           int device, void* handle) {
    try {
        cnclDataType_t type;
        cnclReduceOp_t reduce;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        TP_CHECK(cncl_reduce_op(op, &reduce), "CNCL: unsupported reduction code ", op);
        check(cnclAllReduce(send, recv, count, type, reduce, as_comm(handle), device_queue(device)),
              "allReduce");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_broadcast(const void* send, void* recv, size_t count, int dtype, int root,
                          int device, void* handle) {
    try {
        cnclDataType_t type;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        check(cnclBroadcast(send, recv, count, type, root, as_comm(handle), device_queue(device)),
              "broadcast");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_reduce(const void* send, void* recv, size_t count, int dtype, int op, int root,
                       int device, void* handle) {
    try {
        cnclDataType_t type;
        cnclReduceOp_t reduce;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        TP_CHECK(cncl_reduce_op(op, &reduce), "CNCL: unsupported reduction code ", op);
        check(cnclReduce(send, recv, count, type, reduce, root, as_comm(handle),
                         device_queue(device)),
              "reduce");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_all_gather(const void* send, void* recv, size_t count, int dtype, int device,
                           void* handle) {
    try {
        cnclDataType_t type;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        check(cnclAllGather(send, recv, count, type, as_comm(handle), device_queue(device)),
              "allGather");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_reduce_scatter(const void* send, void* recv, size_t count, int dtype, int op,
                               int device, void* handle) {
    try {
        cnclDataType_t type;
        cnclReduceOp_t reduce;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        TP_CHECK(cncl_reduce_op(op, &reduce), "CNCL: unsupported reduction code ", op);
        check(cnclReduceScatter(send, recv, count, type, reduce, as_comm(handle),
                                device_queue(device)),
              "reduceScatter");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Even split of ``count`` elements per rank.
int tp_mlu_cncl_all_to_all(const void* send, void* recv, size_t count, int dtype, int device,
                           void* handle) {
    try {
        cnclDataType_t type;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        check(cnclAlltoAll(send, recv, count, type, as_comm(handle), device_queue(device)),
              "alltoAll");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Point-to-point primitives.  A matched send/recv pair must be issued inside
// one group so the library can order them without deadlocking.
int tp_mlu_cncl_send(const void* buffer, size_t count, int dtype, int peer, int device,
                     void* handle) {
    try {
        cnclDataType_t type;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        check(cnclSend(const_cast<void*>(buffer), count, type, peer, as_comm(handle),
                       device_queue(device)),
              "send");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_recv(void* buffer, size_t count, int dtype, int peer, int device, void* handle) {
    try {
        cnclDataType_t type;
        TP_CHECK(cncl_data_type(dtype, &type), "CNCL: unsupported data type code ", dtype);
        check(cnclRecv(buffer, count, type, peer, as_comm(handle), device_queue(device)), "recv");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Group delimiters, so a set of point-to-point primitives is submitted as one
// batch.
int tp_mlu_cncl_group_start() {
    try {
        check(cnclGroupStart(), "groupStart");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

int tp_mlu_cncl_group_end() {
    try {
        check(cnclGroupEnd(), "groupEnd");
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// Block until every collective queued on ``device`` has finished.
int tp_mlu_cncl_synchronize(int device) {
    try {
        synchronize(device);
        return 0;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

// 1 when the device queue is idle, 0 while work is outstanding, negative on
// failure.  Lets a work handle answer ``is_completed`` without blocking.
int tp_mlu_cncl_queue_idle(int device) {
    try {
        DeviceGuard guard(device);
        const cnrtRet_t status = cnrtQueueQuery(device_queue(device));
        if (status == cnrtSuccess) return 1;
        if (status == cnrtErrorBusy || status == cnrtErrorNotReady) return 0;
        TP_CHECK(false, "CNCL: queue query failed: ", cnrtGetErrorStr(status));
        return -1;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return -1;
    } catch (...) {
        set_last_error("unknown CNCL failure");
        return -1;
    }
}

} // extern "C"

} // namespace mlu
} // namespace tensorplay
