// The product name CNNL sees.
//
// cnnlCreate reads the device's product name to pick the kernels and tiling
// tables of the exact chip.  Virtualized and rebranded cards can report a
// generic name instead ("mlu"); CNNL then falls back to an unknown-device
// configuration whose kernels fault the queue or return wrong data on many
// ordinary shapes: matrix products with 2 to 256 rows, square products above
// 2048, long reductions along the last axis, whole-tensor reductions over
// 65536 elements, large half-precision elementwise ops.
//
// This library exports cnDeviceGetName ahead of the driver (it is loaded
// with RTLD_GLOBAL before CNNL resolves the symbol), passes every
// recognizable name through, and replaces an unrecognizable one with the
// series name for the chip's compute capability.  TP_MLU_DEVICE_NAME
// overrides the result.

#include <dlfcn.h>

#include <cctype>
#include <cstdlib>
#include <cstring>

#include <cn_api.h>

namespace {

using GetName = CNresult (*)(char*, unsigned int, CNdev);
using GetAttribute = CNresult (*)(int*, CNdevice_attribute, CNdev);

// "MLU" followed by a digit: the form CNNL parses.
bool recognizable(const char* name) {
    return std::strncmp(name, "MLU", 3) == 0 && std::isdigit(static_cast<unsigned char>(name[3]));
}

const char* series_name(CNdev dev) {
    static const auto get_attribute =
        reinterpret_cast<GetAttribute>(dlsym(RTLD_NEXT, "cnDeviceGetAttribute"));
    int major = 0;
    if (get_attribute == nullptr ||
        get_attribute(&major, CN_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CN_SUCCESS) {
        return nullptr;
    }
    switch (major) {
        case 3: return "MLU370";
        case 5: return "MLU590";
        default: return nullptr;
    }
}

CNresult copy_name(char* name, unsigned int len, const char* value) {
    if (std::strlen(value) >= len) return CN_ERROR_INVALID_VALUE;
    std::strcpy(name, value);
    return CN_SUCCESS;
}

} // namespace

extern "C" __attribute__((visibility("default"))) CNresult cnDeviceGetName(char* name,
                                                                          unsigned int len,
                                                                          CNdev dev) {
    if (const char* forced = std::getenv("TP_MLU_DEVICE_NAME"); forced && *forced) {
        return copy_name(name, len, forced);
    }
    static const auto real = reinterpret_cast<GetName>(dlsym(RTLD_NEXT, "cnDeviceGetName"));
    if (real == nullptr) return CN_ERROR_NOT_FOUND;
    const CNresult status = real(name, len, dev);
    if (status != CN_SUCCESS || name == nullptr || recognizable(name)) return status;
    const char* series = series_name(dev);
    return series ? copy_name(name, len, series) : status;
}
