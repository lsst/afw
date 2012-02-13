

namespace lsst {
namespace afw {
namespace math {
namespace detail {

template<typename DestImageT, typename SrcImageT>
std::pair<int,bool> warpImageGPU(
    DestImageT &destImage,                  ///< remapped %image
    lsst::afw::image::Wcs const &destWcs,   ///< WCS of remapped %image
    SrcImageT const &srcImage,              ///< source %image
    lsst::afw::image::Wcs const &srcWcs,               ///< WCS of source %image
    lsst::afw::math::SeparableKernel &warpingKernel,   ///< warping kernel; determines warping algorithm
    int const interpLength,                  ///< Distance over which WCS can be linearily interpolated
                                             ///< must be >0
    lsst::afw::math::ConvolutionControl::DeviceSelection_t devSel
    );

}}}} //namespace lsst::afw::math::detail ends




