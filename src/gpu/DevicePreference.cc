// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008 - 2012 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
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

