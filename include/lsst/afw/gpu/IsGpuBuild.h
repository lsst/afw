// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief A function to determine whether compiling for GPU is enabled
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */


namespace lsst {
namespace afw {
namespace gpu {

/**
 * \brief Inline function which returns true only when GPU_BUILD macro is defined
 *
 * Used to determine whether compiling for GPU is enabled
 */
inline bool isGpuBuild()
{
    #ifdef GPU_BUILD
        return true;
    #else
        return false;
    #endif
}
}}} //namespace lsst::afw::gpu ends
