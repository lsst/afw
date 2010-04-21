// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definition of functions declared in ConvolveImage.h
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <limits>
#include <vector>
#include <string>

#include "boost/format.hpp"
#include "boost/assign/list_of.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

using boost::assign::map_list_of;
using boost::assign::list_of;

# define ISINSTANCE(A, B) (dynamic_cast<B const*>(&(A)) != NULL)

namespace {

    /**
     * @brief Compute the dot product of a kernel row or column and the overlapping portion of an %image
     *
     * @return computed dot product
     *
     * The pixel computed belongs at position imageIter + kernel center.
     *
     * @todo get rid of KernelPixelT parameter if possible by not computing local variable kVal,
     * or by using iterator traits:
     *     typedef typename std::iterator_traits<KernelIterT>::value_type KernelPixel;
     * Unfortunately, in either case compilation fails with this sort of message:
\verbatim
include/lsst/afw/image/Pixel.h: In instantiation of Ôlsst::afw::image::pixel::exprTraits<boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > > >Õ:
include/lsst/afw/image/Pixel.h:385:   instantiated from Ôlsst::afw::image::pixel::BinaryExpr<lsst::afw::image::pixel::Pixel<int, short unsigned int, float>, boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >, std::multiplies<int>, lsst::afw::image::pixel::bitwise_or<short unsigned int>, lsst::afw::image::pixel::variance_multiplies<float> >Õ
src/math/ConvolveImage.cc:59:   instantiated from ÔOutPixelT<unnamed>::kernelDotProduct(ImageIterT, KernelIterT, int) [with OutPixelT = lsst::afw::image::pixel::SinglePixel<int, short unsigned int, float>, ImageIterT = lsst::afw::image::MaskedImage<int, short unsigned int, float>::const_MaskedImageIterator<boost::gil::gray32s_pixel_t*, boost::gil::gray16_pixel_t*, boost::gil::gray32f_noscale_pixel_t*>, KernelIterT = const boost::gil::gray64f_noscalec_pixel_t*]Õ
src/math/ConvolveImage.cc:265:   instantiated from Ôvoid lsst::afw::math::basicConvolve(OutImageT&, const InImageT&, const lsst::afw::math::Kernel&, bool) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>]Õ
src/math/ConvolveImage.cc:451:   instantiated from Ôvoid lsst::afw::math::convolve(OutImageT&, const InImageT&, const KernelT&, bool, int) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, KernelT = lsst::afw::math::AnalyticKernel]Õ
src/math/ConvolveImage.cc:587:   instantiated from here
include/lsst/afw/image/Pixel.h:210: error: no type named ÔImagePixelTÕ in Ôstruct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >Õ
include/lsst/afw/image/Pixel.h:211: error: no type named ÔMaskPixelTÕ in Ôstruct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >Õ
include/lsst/afw/image/Pixel.h:212: error: no type named ÔVariancePixelTÕ in Ôstruct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >Õ
\endverbatim
     */
    template <typename OutPixelT, typename ImageIterT, typename KernelIterT, typename KernelPixelT>
    inline OutPixelT kernelDotProduct(
        ImageIterT imageIter,       ///< start of input %image that overlaps kernel vector
        KernelIterT kernelIter,     ///< start of kernel vector
        int kWidth      ///< width of kernel
    ) {
        OutPixelT outPixel(0);
        for (int x = 0; x < kWidth; ++x, ++imageIter, ++kernelIter) {
            KernelPixelT kVal = *kernelIter;
            if (kVal != 0) {
                outPixel += static_cast<OutPixelT>((*imageIter) * kVal);
            }
        }
        return outPixel;
    }
    
    /**
    * @brief Set the edge pixels of a convolved Image based on size of the convolution kernel used
    *
    * Separate specializations for Image and MaskedImage are required to set the EDGE bit of the Mask plane
    * (if there is one) when copyEdge is true.
    */
    template <typename OutImageT, typename InImageT>
    inline void setEdgePixels(
        OutImageT& outImage,        ///< %image whose edge pixels are to be set
        afwMath::Kernel const &kernel,  ///< convolution kernel; kernel size is used to determine the edge
        InImageT const &inImage,    ///< %image whose edge pixels are to be copied;
                                    ///< ignored if copyEdge is false
        bool copyEdge,              ///< if false (default), set edge pixels to the standard edge pixel;
                                    ///< if true, copy edge pixels from input and set EDGE bit of mask
        lsst::afw::image::detail::Image_tag
            ///< lsst::afw::image::detail::image_traits<ImageT>::image_category()
                                
    ) {
        const unsigned int imWidth = outImage.getWidth();
        const unsigned int imHeight = outImage.getHeight();
        const unsigned int kWidth = kernel.getWidth();
        const unsigned int kHeight = kernel.getHeight();
        const unsigned int kCtrX = kernel.getCtrX();
        const unsigned int kCtrY = kernel.getCtrY();

        const typename OutImageT::SinglePixel edgePixel = afwMath::edgePixel<OutImageT>(
            typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
        );
        std::vector<afwImage::BBox> bboxList;
    
        // create a list of bounding boxes describing edge regions, in this order:
        // bottom edge, top edge (both edge to edge),
        // left edge, right edge (both omitting pixels already in the bottom and top edge regions)
        int const numHeight = kHeight - (1 + kCtrY);
        int const numWidth = kWidth - (1 + kCtrX);
        bboxList.push_back(afwImage::BBox(afwImage::PointI(0, 0), imWidth, kCtrY));
        bboxList.push_back(afwImage::BBox(afwImage::PointI(0, imHeight - numHeight), imWidth, numHeight));
        bboxList.push_back(afwImage::BBox(afwImage::PointI(0, kCtrY), kCtrX, imHeight + 1 - kHeight));
        bboxList.push_back(afwImage::BBox(afwImage::PointI(imWidth - numWidth, kCtrY),
            numWidth, imHeight + 1 - kHeight));

        for (std::vector<afwImage::BBox>::const_iterator bboxIter = bboxList.begin();
            bboxIter != bboxList.end(); ++bboxIter) {
            OutImageT outView(outImage, *bboxIter);
            if (copyEdge) {
                // note: <<= only works with data of the same type
                // so convert the input image to output format
                outView <<= OutImageT(inImage, *bboxIter);
            } else {
                outView = edgePixel;
            }
        }
    }

    /**
    * @brief Set the edge pixels of a convolved MaskedImage based on size of the convolution kernel used
    *
    * Separate specializations for Image and MaskedImage are required to set the EDGE bit of the Mask plane
    * (if there is one) when copyEdge is true.
    */
    template <typename OutImageT, typename InImageT>
    inline void setEdgePixels(
        OutImageT& outImage,        ///< %image whose edge pixels are to be set
        afwMath::Kernel const &kernel,  ///< convolution kernel; kernel size is used to determine the edge
        InImageT const &inImage,    ///< %image whose edge pixels are to be copied; ignored if copyEdge false
        bool copyEdge,              ///< if false (default), set edge pixels to the standard edge pixel;
                                    ///< if true, copy edge pixels from input and set EDGE bit of mask
        lsst::afw::image::detail::MaskedImage_tag
            ///< lsst::afw::image::detail::image_traits<MaskedImageT>::image_category()
                                
    ) {
        const unsigned int imWidth = outImage.getWidth();
        const unsigned int imHeight = outImage.getHeight();
        const unsigned int kWidth = kernel.getWidth();
        const unsigned int kHeight = kernel.getHeight();
        const unsigned int kCtrX = kernel.getCtrX();
        const unsigned int kCtrY = kernel.getCtrY();

        const typename OutImageT::SinglePixel edgePixel = afwMath::edgePixel<OutImageT>(
            typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
        );
        std::vector<afwImage::BBox> bboxList;
    
        // create a list of bounding boxes describing edge regions, in this order:
        // bottom edge, top edge (both edge to edge),
        // left edge, right edge (both omitting pixels already in the bottom and top edge regions)
        int const numHeight = kHeight - (1 + kCtrY);
        int const numWidth = kWidth - (1 + kCtrX);
        bboxList.push_back(afwImage::BBox(afwImage::PointI(0, 0), imWidth, kCtrY));
        bboxList.push_back(afwImage::BBox(afwImage::PointI(0, imHeight - numHeight), imWidth, numHeight));
        bboxList.push_back(afwImage::BBox(afwImage::PointI(0, kCtrY), kCtrX, imHeight + 1 - kHeight));
        bboxList.push_back(afwImage::BBox(afwImage::PointI(imWidth - numWidth, kCtrY),
            numWidth, imHeight + 1 - kHeight));

        afwImage::MaskPixel const edgeMask = afwImage::Mask<afwImage::MaskPixel>::getPlaneBitMask("EDGE");
        for (std::vector<afwImage::BBox>::const_iterator bboxIter = bboxList.begin();
            bboxIter != bboxList.end(); ++bboxIter) {
            OutImageT outView(outImage, *bboxIter);
            if (copyEdge) {
                // note: <<= only works with data of the same type
                // so convert the input image to output format
                outView <<= OutImageT(inImage, *bboxIter);
                *(outView.getMask()) |= edgeMask;
            } else {
                outView = edgePixel;
            }
        }
    }

}   // anonymous namespace

/**
 * Compute the scaled sum of two images
 *
 * outImage = c1 inImage1 + c2 inImage2
 *
 * For example to linearly interpolate between two images set:
 *   c1 = 1.0 - fracDist
 *   c2 = fracDist
 * where fracDist is the fractional distance of outImage from inImage1:
 *              location of outImage - location of inImage1
 *   fracDist = -------------------------------------------
 *              location of inImage2 - location of inImage1
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if outImage is not same dimensions
 * as inImage1 and inImage2.
 */
template <typename OutImageT, typename InImageT>
void afwMath::scaledPlus(
        OutImageT &outImage,        ///< output image
        double c1,                  ///< coefficient for image 1
        InImageT const &inImage1,   ///< input image 1
        double c2,                  ///< coefficient for image 2
        InImageT const &inImage2)   ///< input image 2
{
    if (outImage.getDimensions() != inImage1.getDimensions()) {
        std::ostringstream os;
        os << "outImage dimensions = ( " << outImage.getWidth() << ", " << outImage.getHeight()
            << ") != (" << inImage1.getWidth() << ", " << inImage1.getHeight()
            << ") = inImage1 dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    } else if (inImage1.getDimensions() != inImage2.getDimensions()) {
        std::ostringstream os;
        os << "inImage1 dimensions = ( " << inImage1.getWidth() << ", " << inImage1.getHeight()
            << ") != (" << inImage2.getWidth() << ", " << inImage2.getHeight()
            << ") = inImage2 dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    
    typedef typename InImageT::const_x_iterator InConstXIter;
    typedef typename OutImageT::x_iterator OutXIter;
    for (int y = 0; y != inImage1.getHeight(); ++y) {
        InConstXIter const end1 = inImage1.row_end(y);
        InConstXIter inIter1 = inImage1.row_begin(y);
        InConstXIter inIter2 = inImage2.row_begin(y);
        OutXIter outIter = outImage.row_begin(y);
        for (; inIter1 != end1; ++inIter1, ++inIter2, ++outIter) {
            *outIter = (*inIter1 * c1) + (*inIter2 * c2);
        }
    }
}


/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage is not the same size as inImage
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage is smaller than kernel
 *  in columns or rows.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void afwMath::basicConvolve(
    OutImageT &convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::Kernel const& kernel,  ///< convolution kernel
    bool doNormalize                ///< if true, normalize the kernel, else use "as is"
) {
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;

    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    // Because convolve isn't a method of Kernel we can't always use Kernel's vtbl to dynamically
    // dispatch the correct version of basicConvolve. The case that fails is convolving with a kernel
    // obtained from a pointer or reference to a Kernel (base class), e.g. as used in LinearCombinationKernel.
    if (ISINSTANCE(kernel, afwMath::DeltaFunctionKernel)) {
        pexLog::TTrace<4>("lsst.afw.kernel.convolve",
            "generic basicConvolve: dispatch to DeltaFunctionKernel basicConvolve");
        afwMath::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::DeltaFunctionKernel const*>(&kernel),
            doNormalize);
        return;
    } else if (ISINSTANCE(kernel, afwMath::SeparableKernel)) {
        pexLog::TTrace<4>("lsst.afw.kernel.convolve",
            "generic basicConvolve: dispatch to SeparableKernel basicConvolve");
        afwMath::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::SeparableKernel const*>(&kernel),
            doNormalize);
        return;
    } else if (ISINSTANCE(kernel, afwMath::LinearCombinationKernel) && kernel.isSpatiallyVarying()) {
        pexLog::TTrace<4>("lsst.afw.kernel.convolve",
            "generic basicConvolve: dispatch to spatially varying LinearCombinationKernel basicConvolve");
        afwMath::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::LinearCombinationKernel const*>(&kernel),
            doNormalize);
        return;
    }
    // OK, use general (and slower) form

    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( "
            << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
            << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (inImage.getDimensions() < kernel.getDimensions()) {
        std::ostringstream os;
        os << "inImage dimensions = ( "
            << inImage.getWidth() << ", " << inImage.getHeight()
            << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
            << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    
    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    KernelImage kernelImage(kernel.getDimensions()); // the kernel at a point

    if (kernel.isSpatiallyVarying()) {
        pexLog::TTrace<3>("lsst.afw.kernel.convolve", "generic basicConvolve: kernel is spatially varying");

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
            
            InXYLocator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                KernelPixel kSum = kernel.computeImage(kernelImage, false, colPos, rowPos);
                KernelXYLocator kernelLoc = kernelImage.xy_at(0,0);
                *cnvXIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(
                    inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvXIter = *cnvXIter/kSum;
                }
            }
        }
    } else {
        pexLog::TTrace<3>("lsst.afw.kernel.convolve", "generic basicConvolve: kernel is spatially invariant");
        (void)kernel.computeImage(kernelImage, doNormalize);
        
        for (int inStartY = 0, cnvY = cnvStartY; inStartY < cnvHeight; ++inStartY, ++cnvY) {
            for (OutXIterator cnvXIter=convolvedImage.x_at(cnvStartX, cnvY),
                cnvXEnd = convolvedImage.row_end(cnvY); cnvXIter != cnvXEnd; ++cnvXIter) {
                *cnvXIter = 0;
            }
            for (int kernelY = 0, inY = inStartY; kernelY < kHeight; ++inY, ++kernelY) {
                KernelXIterator kernelXIter = kernelImage.x_at(0, kernelY);
                InXIterator inXIter = inImage.x_at(0, inY);
                OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
                for (int x = 0; x < cnvWidth; ++x, ++cnvXIter, ++inXIter) {
                    *cnvXIter += kernelDotProduct<OutPixel, InXIterator, KernelXIterator, KernelPixel>(
                        inXIter, kernelXIter, kWidth);
                }
            }
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void afwMath::basicConvolve(
    OutImageT& convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::DeltaFunctionKernel const &kernel,    ///< convolution kernel
    bool                                           ///< unused
) {
    assert (!kernel.isSpatiallyVarying());

    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( "
            << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
            << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (inImage.getDimensions() < kernel.getDimensions()) {
        std::ostringstream os;
        os << "inImage dimensions = ( "
            << inImage.getWidth() << ", " << inImage.getHeight()
            << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
            << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    
    int const mImageWidth = inImage.getWidth(); // size of input region
    int const mImageHeight = inImage.getHeight();
    int const cnvWidth = mImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = mImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const inStartX = kernel.getPixel().getX();
    int const inStartY = kernel.getPixel().getY();

    pexLog::TTrace<3>("lsst.afw.kernel.convolve", "DeltaFunctionKernel basicConvolve");

    for (int i = 0; i < cnvHeight; ++i) {
        typename InImageT::x_iterator inPtr = inImage.x_at(inStartX, i +  inStartY);
        for (typename OutImageT::x_iterator cnvPtr = convolvedImage.x_at(cnvStartX, i + cnvStartY),
                 cnvEnd = cnvPtr + cnvWidth; cnvPtr != cnvEnd; ++cnvPtr, ++inPtr){
            *cnvPtr = *inPtr;
        }
    }
}
\
/**
 * @brief A version of basicConvolve that should be used when convolving a LinearCombinationKernel
 *
 * The Algorithm:
 * - If the kernel is spatially varying, then convolves the input Image by each basis kernel in turn,
 *   solves the spatial model for that component and adds in the appropriate amount of the convolved %image.
 * - If the kernel is not spatially varying, then computes a fixed kernel and calls the
 *   the general version of basicConvolve.
 *
 * @warning The variance will be mis-computed if your basis kernels contain home-brew delta function kernels
 * (instead of instances of afwMath::DeltaFunctionKernel). It may also be mis-computed if your basis kernels
 * contain many pixels with value zero, or if your basis kernels contain a mix of
 * afwMath::DeltaFunctionKernel with other kernels.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage is not the same size as inImage
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage is smaller than kernel
 *  in columns or rows.
 * @throw lsst::pex::exception::InvalidParameterException if doNormalize true and kernel is spatially varying
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void afwMath::basicConvolve(
    OutImageT& convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel, ///< convolution kernel
    bool doNormalize                ///< if true, normalize the kernel, else use "as is"
) {
    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        typedef typename afwMath::Kernel::Pixel KernelPixel;

        pexLog::TTrace<4>("lsst.afw.kernel.convolve",
            "basicConvolve LinearCombinationKernel; kernel is not spatially varying");
        afwImage::Image<KernelPixel> kernelImage(kernel.getWidth(), kernel.getHeight());
        kernel.computeImage(kernelImage, doNormalize);
        afwMath::FixedKernel fixedKernel(kernelImage);
        return basicConvolve(convolvedImage, inImage, fixedKernel, doNormalize);
    }
    

    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( "
            << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
            << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (inImage.getDimensions() < kernel.getDimensions()) {
        std::ostringstream os;
        os << "inImage dimensions = ( "
            << inImage.getWidth() << ", " << inImage.getHeight()
            << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
            << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    
    pexLog::TTrace<3>("lsst.afw.kernel.convolve",
        "basicConvolve for LinearCombinationKernel: kernel is spatially varying");
    
    typedef typename InImageT::template ImageTypeFactory<double>::type BasisImage;
    typedef typename BasisImage::x_iterator BasisXIterator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef afwMath::KernelList KernelList;

    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const cnvWidth = imWidth + 1 - kernel.getWidth();
    int const cnvHeight = imHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1
    // create a BasisImage to hold the source convolved with a basis kernel
    BasisImage basisImage(inImage.getDimensions());

    // initialize good area of output image to zero so we can add the convolved basis images into it
    // surely there is a single call that will do this? but in lieu of that...
    typename OutImageT::SinglePixel const nullPixel(0);
    for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
        OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
        for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
            *cnvXIter = nullPixel;
        }
    }
    
    // iterate over basis kernels
    KernelList const basisKernelList = kernel.getKernelList();
    int i = 0;
    for (typename KernelList::const_iterator basisKernelIter = basisKernelList.begin();
        basisKernelIter != basisKernelList.end(); ++basisKernelIter, ++i) {
        afwMath::basicConvolve(basisImage, inImage, **basisKernelIter, false);
        double noiseCorrelationCoeff = 1.0;
        if (ISINSTANCE(**basisKernelIter, afwMath::DeltaFunctionKernel)) {
            noiseCorrelationCoeff = 0.0;
        }

        // iterate over matching pixels of all images to compute output image
        afwMath::Kernel::SpatialFunctionPtr spatialFunctionPtr = kernel.getSpatialFunction(i);
        std::vector<double> kernelCoeffList(kernel.getNKernelParameters());
            // weights of basis images at this point
        for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
        
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            BasisXIterator basisXIter = basisImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter, ++basisXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);
                double basisCoeff = (*spatialFunctionPtr)(colPos, rowPos);
                
                typename OutImageT::SinglePixel cnvPixel(*cnvXIter);
                cnvPixel = afwImage::pixel::plus(cnvPixel, (*basisXIter) * basisCoeff, noiseCorrelationCoeff);
                *cnvXIter = cnvPixel;
                // note: cnvPixel avoids compiler complaints; the following does not build:
                // *cnvXIter = afwImage::pixel::plus(
                //      *cnvXIter, (*basisXIter) * basisCoeff, noiseCorrelationCoeff);
            }
        }
    }

    if (doNormalize) {
        /*
        For each pixel of the output image: compute the kernel sum for that pixel and scale
        the output image. One obvious alternative is to create a temporary kernel sum image
        and accumulate into that while iterating over the basis kernels above. This saves
        computing the spatial functions again here, but requires a temporary image
        the same size as the output image, so it is likely to suffer from cache issues.
        */
        std::vector<double> const kernelSumList = kernel.getKernelSumList();
        std::vector<Kernel::SpatialFunctionPtr> spatialFunctionList = kernel.getSpatialFunctionList();
        for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
        
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                std::vector<double>::const_iterator kSumIter = kernelSumList.begin();
                std::vector<Kernel::SpatialFunctionPtr>::const_iterator spFuncIter =
                    spatialFunctionList.begin();
                double kSum = 0.0;
                for ( ; kSumIter != kernelSumList.end(); ++kSumIter, ++spFuncIter) {
                    kSum += (**spFuncIter)(colPos, rowPos) * (*kSumIter);
                }
                *cnvXIter /= kSum;
            }
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void afwMath::basicConvolve(
    OutImageT& convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::SeparableKernel const &kernel, ///< convolution kernel
    bool doNormalize                ///< if true, normalize the kernel, else use "as is"
) {
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef typename std::vector<KernelPixel> KernelVector;
    typedef KernelVector::const_iterator KernelIterator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::y_iterator OutYIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( "
            << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
            << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (inImage.getDimensions() < kernel.getDimensions()) {
        std::ostringstream os;
        os << "inImage dimensions = ( "
            << inImage.getWidth() << ", " << inImage.getHeight()
            << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
            << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    
    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = static_cast<int>(imWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(imHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartX = static_cast<int>(kernel.getCtrX());
    int const cnvStartY = static_cast<int>(kernel.getCtrY());
    int const cnvEndX = cnvStartX + cnvWidth; // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    KernelVector kXVec(kWidth);
    KernelVector kYVec(kHeight);
    
    if (kernel.isSpatiallyVarying()) {
        pexLog::TTrace<3>("lsst.afw.kernel.convolve",
            "SeparableKernel basicConvolve: kernel is spatially varying");

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
            
            InXYLocator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                KernelPixel kSum = kernel.computeVectors(kXVec, kYVec, doNormalize, colPos, rowPos);

                // why does this trigger warnings? It did not in the past.
                *cnvXIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(inImLoc, kXVec, kYVec);
                if (doNormalize) {
                    *cnvXIter = *cnvXIter/kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        pexLog::TTrace<3>("lsst.afw.kernel.convolve",
            "SeparableKernel basicConvolve: kernel is spatially invariant");

        kernel.computeVectors(kXVec, kYVec, doNormalize);
        KernelIterator const kXVecBegin = kXVec.begin();
        KernelIterator const kYVecBegin = kYVec.begin();

        // Handle the x kernel vector first, putting results into convolved image as a temporary buffer
        // (all remaining processing must read from and write to the convolved image,
        // thus being careful not to modify pixels that still need to be read)
        for (int imageY = 0; imageY < imHeight; ++imageY) {
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, imageY);
            InXIterator inXIter = inImage.x_at(0, imageY);
            InXIterator const inXIterEnd = inImage.x_at(cnvWidth, imageY);
            for ( ; inXIter != inXIterEnd; ++cnvXIter, ++inXIter) {
                *cnvXIter = kernelDotProduct<OutPixel, InXIterator, KernelIterator, KernelPixel>(
                    inXIter, kXVecBegin, kWidth);
            }
        }
        
        // Handle the y kernel vector. It turns out to be faster for the innermost loop to be along y,
        // probably because one can accumulate into a temporary variable.
        // For each row of output, compute the output pixel, putting it at the bottom
        // (a pixel that will not be read again).
        // The resulting image is correct, but shifted down by kernel ctr y pixels.
        for (int cnvY = 0; cnvY < cnvHeight; ++cnvY) {
            for (int x = cnvStartX; x < cnvEndX; ++x) {
                OutYIterator cnvYIter = convolvedImage.y_at(x, cnvY);
                *cnvYIter = kernelDotProduct<OutPixel, OutYIterator, KernelIterator, KernelPixel>(
                    cnvYIter, kYVecBegin, kHeight);
            }
        }

        // Move the good pixels up by kernel ctr Y (working down to avoid overwriting data)
        for (int destY = cnvEndY - 1, srcY = cnvHeight - 1; srcY >= 0; --destY, --srcY) {
            OutXIterator destIter = convolvedImage.x_at(cnvStartX, destY);
            OutXIterator const destIterEnd = convolvedImage.x_at(cnvEndX, destY);
            OutXIterator srcIter = convolvedImage.x_at(cnvStartX, srcY);
            for ( ; destIter != destIterEnd; ++destIter, ++srcIter) {
                *destIter = *srcIter;
            }
        }
    }
}

/**
 * @brief Convolve an Image or MaskedImage with a Kernel, setting pixels of an existing output %image.
 * 
 * Various convolution kernels are available, including:
 * - FixedKernel: a kernel based on an %image
 * - AnalyticKernel: a kernel based on a Function
 * - SeparableKernel: a kernel described by the product of two one-dimensional Functions: f0(x) * f1(y)
 * - LinearCombinationKernel: a linear combination of a set of spatially invariant basis kernels.
 * - DeltaFunctionKernel: a kernel that is all zeros except one pixel whose value is 1.
 *   Typically used as a basis kernel for LinearCombinationKernel.
 *
 * If a kernel is spatially varying, its spatial model is computed at each pixel position on the image
 * (pixel position, not pixel index). At present (2009-09-24) this position is computed relative
 * to the lower left corner of the sub-image, but it will almost certainly change to be
 * the lower left corner of the parent image.
 * 
 * All convolution is performed in real space. This allows convolution to handle masked pixels
 * and spatially varying kernels. Although convolution of an Image with a spatially invariant kernel could,
 * in fact, be performed in Fourier space, the code does not do this.
 * 
 * Note that mask bits are smeared by convolution; all nonzero pixels in the kernel smear the mask, even
 * pixels that have very small values. Larger kernels smear the mask more and are also slower to convolve.
 * Use the smallest kernel that will do the job.
 *
 * convolvedImage has a border of edge pixels which cannot be computed normally. Normally these pixels
 * are set to the standard edge pixel, as returned by edgePixel(). However, if your code cannot handle
 * nans in the %image or infs in the variance, you may set copyEdge true, in which case the edge pixels
 * are set to the corresponding pixels of the input %image and (if there is a mask) the mask EDGE bit is set.
 *
 * The border of edge pixels has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 * 
 * Convolution has been optimized for the various kinds of kernels, as follows (listed approximately
 * in order of decreasing speed):
 * - DeltaFunctionKernel convolution is a simple %image shift.
 * - SeparableKernel convolution is performed by convolving the input by one of the two functions,
 *   then the result by the other function. Thus convolution with a kernel of size nCols x nRows becomes
 *   convolution with a kernel of size nCols x 1, followed by convolution with a kernel of size 1 x nRows.
 * - Convolution with spatially invariant versions of the other kernels is performed by computing
 *   the kernel %image once and convolving with that. The code has been optimized for cache performance
 *   and so should be fairly efficient.
 * - Convolution with a spatially varying LinearCombinationKernel is performed by convolving the %image
 *   by each basis kernel and combining the result by solving the spatial model. This will be efficient
 *   provided the kernel does not contain too many or very large basis kernels.
 * - Convolution with spatially varying AnalyticKernel is likely to be slow. The code simply computes
 *   the output one pixel at a time by computing the AnalyticKernel at that point and applying it to
 *   the input %image. This is not favorable for cache performance (especially for large kernels)
 *   but avoids recomputing the AnalyticKernel. It is probably possible to do better.
 *
 * Additional convolution functions include:
 *  - convolveAtAPoint(): convolve a Kernel to an Image or MaskedImage at a point.
 *  - basicConvolve(): convolve a Kernel with an Image or MaskedImage, but do not set the edge pixels
 *    of the output. Optimization of convolution for different types of Kernel are handled by different
 *    specializations of basicConvolve().
 * 
 * afw/examples offers programs that time convolution including timeConvolve and timeSpatiallyVaryingConvolve.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage is not the same size as inImage
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage is smaller than kernel
 *  in columns and/or rows.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT, typename KernelT>
void afwMath::convolve(
    OutImageT& convolvedImage,  ///< convolved %image; must be the same size as inImage
    InImageT const& inImage,    ///< %image to convolve
    KernelT const& kernel,      ///< convolution kernel
    bool doNormalize,           ///< if true, normalize the kernel, else use "as is"
    bool copyEdge               ///< if false (default), set edge pixels to the standard edge pixel;
                                ///< if true, copy edge pixels from input and set EDGE bit of mask
                                
) {
    afwMath::basicConvolve(convolvedImage, inImage, kernel, doNormalize);
    setEdgePixels(convolvedImage, kernel, inImage, copyEdge,
        typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
    );
}


/*
 * Explicit instantiation of all convolve functions.
 *
 * This code needs to be compiled with full optimization, and there's no reason why
 * it should be instantiated in the swig wrappers.
 */
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
//
// Next a macro to generate needed instantiations for IMGMACRO (either IMAGE or MASKEDIMAGE)
// and the specified pixel types
//
/* NL's a newline for debugging -- don't define it and say
 g++ -C -E -I$(eups list -s -d boost)/include Convolve.cc | perl -pe 's| *NL *|\n|g'
*/
#define NL /* */
#define CONVOLUTIONFUNCSBYTYPE(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template void afwMath::scaledPlus( \
        IMGMACRO(OUTPIXTYPE)&, double, IMGMACRO(INPIXTYPE) const&, double, IMGMACRO(INPIXTYPE) const&); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, AnalyticKernel const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, DeltaFunctionKernel const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, FixedKernel const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, LinearCombinationKernel const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, SeparableKernel const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, Kernel const&, bool, bool);

//
// Now a macro to specify Image and MaskedImage
//
#define CONVOLUTIONFUNCS(OUTPIXTYPE, INPIXTYPE) \
    CONVOLUTIONFUNCSBYTYPE(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    CONVOLUTIONFUNCSBYTYPE(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)

CONVOLUTIONFUNCS(double, double)
CONVOLUTIONFUNCS(double, float)
CONVOLUTIONFUNCS(float, float)
CONVOLUTIONFUNCS(int, int)
CONVOLUTIONFUNCS(boost::uint16_t, boost::uint16_t)
