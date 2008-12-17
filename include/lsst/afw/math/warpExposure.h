// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file 
  *
  * @ingroup afw
  *
  * @brief Implementation of the templated utility function, warpExposure, for
  * Astrometric Image Remapping for the LSST.
  *
  * @author Nicole M. Silvestri, University of Washington
  *
  * Contact: nms@astro.washington.edu 
  *
  * @version
  *
  * LSST Legalese here...
  */

#ifndef LSST_AFW_MATH_WARPEXPOSURE_H
#define LSST_AFW_MATH_WARPEXPOSURE_H

#include <string>

#include <boost/shared_ptr.hpp>

#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
namespace math {
       
    typedef boost::uint16_t maskPixelType;
    
//     class WarpParameters {
//         const int MinType = 1;
//         const int Lancsoz = 1;
//         const int NearestNeighbor = 2;
//         const int LinearInterpolate = 3;
//         const int MaxType = 3;
//         WarpParameters(int type) : _type(type) {
//             if ((type < 1) || (type > MaxType)) {
//                 raise std::runtime_error("Invalid type");
//             }
//         };
// 
//         static WarpParameters lanczos(int order) {
//             retVal = WarpParameters(WarpParameters::Lanczos);
//             retVal._setLanczosOrder(order);
//             return retVal;
//         static WarpParameters nearestNeighbor() {
//             return WarpParameters(WarpParameters::NearestNeighbor);
//     };
//     class LanczosWarpParameters: WarpParameters {
//         LancsozParameters(int order) : _order(order) {};
//         int getOrder() return _order;
//     private:
//         int _type;
//         int _lanczosOrder;
//     }
//     class NearestNeighborWarpParameters {
//     }
//     class LinearInterpolateWarpParameters {
//     }

    template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
    int warpExposure(
        lsst::afw::image::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> &remapExposure,
        lsst::afw::image::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const &origExposure,
        std::string const &kernelType, 
        int kernelWidth, 
        int kernelHeight
        );

//     template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
//     lsst::afw::image::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> warpExposure(
//         int &numEdgePixels,
//         lsst::afw::image::Wcs const &remapWcs,
//         int remapWidth, 
//         int remapHeight, 
//         lsst::afw::image::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const &origExposure,
//         std::string const kernelType, 
//         int kernelWidth, 
//         int kernelHeight
//         );
       
}}} // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_WARPEXPOSURE_H)
