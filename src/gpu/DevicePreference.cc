// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Implementation file for CPU/GPU device selection.
 *
 * Introduces a global variable which value indicates whether GPU acceleration is enabled.
 * The value of this global variable can be forced to GPU-disable by setting the value of
 * environment variable DISABLE_GPU_ACCELERATION to 1.
 * The value of environment variable DISABLE_GPU_ACCELERATION is red by a global constructor
 * (at the moment when AFW library is loaded).
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#include <stdlib.h>

namespace lsst {
namespace afw {
namespace gpu {
namespace {

// globaly enables or disables GPU acceleration
// don't allow GPU acceleration until the constructor of EnvVarDisableGpuAcceleration is called
bool globalIsGpuEnabled = false;

class EnvVarDisableGpuAcceleration
{
public:
    bool val;

    EnvVarDisableGpuAcceleration()
    {
        const char *envVal = getenv("DISABLE_GPU_ACCELERATION");
        if (envVal != NULL && atoi(envVal) == 1) {
            val = true;
        } else {
            val = false;
            globalIsGpuEnabled = true;
        }
    }
} const envVarDisableGpuAccelerationSingleton;
}

void setGpuEnable(bool enable)
{
    if (enable) {
        if (!envVarDisableGpuAccelerationSingleton.val) {
            globalIsGpuEnabled = true;
        }
    } else {
        globalIsGpuEnabled = false;
    }
}

bool isGpuEnabled()
{
    return globalIsGpuEnabled;
}

}
}
} //namespace lsst::afw::gpu ends

