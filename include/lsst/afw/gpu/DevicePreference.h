// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Interface for CPU/GPU device selection.
 *
 * Introduces required types and functions.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifndef LSST_AFW_MATH_DETAIL_GPU_DEVICEPREFERENCE_H
#define LSST_AFW_MATH_DETAIL_GPU_DEVICEPREFERENCE_H

namespace lsst {
namespace afw {
namespace gpu {

/**
 * @brief A type used to select whether to use CPU or GPU device
 *
 * AUTO - If all conditions for GPU execution and performance conditions are satisfied
 *   (AFW built with GPU support, a suitable GPU is present, GPU execution limitations and performance conditions)
 *   the code will attempt to use a GPU. Otherwise, CPU code path
 *   will be used. If GPU execution results in a failure, an exception will be thrown.
 * AUTO_WITH_CPU_FALLBACK - Same as AUTO, except that
 *    if the GPU code path throws an exception, it will fallback to CPU code path.
 * USE_CPU - GPU will not be used.
 * USE_GPU - code will always attempt to use a GPU (except when overriden by global settings).
 *    If a GPU can not be used, an exception will be thrown.
 *
 * @note To test whether a certain function call is GPU-accelerated, a dummy function call should be made,
 *       with the DevicePreference parameter set to USE_GPU. If it does not throw an exception, then 
 *       the given function call is GPU-accelerated.
 */
enum DevicePreference { AUTO, AUTO_WITH_CPU_FALLBACK, USE_CPU, USE_GPU };

/**
 * @brief Default DevicePreference value
 *
 * Used in all GPU-accelerated functions when DevicePreference is ommited by the caller.
 */
const DevicePreference DEFAULT_DEVICE_PREFERENCE = AUTO;

/**
 * @brief Enables or disables GPU acceleration globally
 *
 * Changes a global variable. When it is set to false, the GPU acceleration will be disabled.
 * Initial value of the global variable is true (enabled).
 *
 * Calls to GPU-accelerated functions with DevicePreference set to USE_GPU will be redirected to use CPU code path.
 *
 * Note: When the value of environment variable DISABLE_GPU_ACCELERATION is set to 1,
 * setting GPU acceleration to enabled by this function may not have any effect . Also, it will set the
 * initial value of the global variable to false (GPU acceleration disabled).
 */
void setGpuEnable(bool enable);

/**
 * @brief returns true if GPU acceleration is enabled
 */
bool isGpuEnabled();

}}} //namespace lsst::afw::gpu ends

#endif //not defined LSST_AFW_MATH_DETAIL_GPU_DEVICEPREFERENCE_H
