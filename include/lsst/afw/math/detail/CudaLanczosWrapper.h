

namespace lsst {
namespace afw {
namespace math {
namespace detail {

/**
 * \brief GPU accelerated image warping for Lanczos resampling
 *
 * \return a std::pair<int,bool> containing:
 *                1) the number of valid pixels in destImage (those that are not edge pixels).
 *                2) whether the warping was performed (if false, then the first value is not defined)
 *
 * This function requires a warping kernel to perform the interpolation.
 * Only Lanczos kernel is a valid input, any other kernel will raise in exception
 *
 * This function will not perform the warping if kernel size is too large.
 * (currently, when the order of the Lanczos kernel is >50)
 * If warping is not performed, the return value will be (X,false).
 * If forceProcessing is false, the warping might not be performed if interpLength is too small
 *
 * Also see lsst::afw::math::warpImage()
 *
 * \b Implementation:
 * Calculates values of the coordinate transform function at some points, which are spaced by interpLength intervals
 * Calls CalculateInterpolationData().
 * Calls WarpImageGpuWrapper() to perform the wapring.
 *
 * \throw lsst::pex::exceptions::InvalidParameterException if the warping kernel is not a Lanczos kernel
 * \throw lsst::pex::exceptions::InvalidParameterException if interpLength < 1
 * \throw lsst::pex::exceptions::MemoryException when allocation of CPU memory fails
 * \throw lsst::afw::gpu::GpuMemoryException when allocation or transfer to/from GPU memory fails
 * \throw lsst::afw::gpu::GpuRuntimeErrorException when GPU code run fails
 *
 */
template<typename DestImageT, typename SrcImageT>
std::pair<int,bool> warpImageGPU(
    DestImageT &destImage,                  ///< remapped %image
    lsst::afw::image::Wcs const &destWcs,   ///< WCS of remapped %image
    SrcImageT const &srcImage,              ///< source %image
    lsst::afw::image::Wcs const &srcWcs,               ///< WCS of source %image
    lsst::afw::math::SeparableKernel &warpingKernel,   ///< warping kernel; determines warping algorithm
    int const interpLength,                  ///< Distance over which WCS can be linearily interpolated
                                             ///< must be >0
    lsst::afw::gpu::DevicePreference devPref
    );

}}}} //namespace lsst::afw::math::detail ends




