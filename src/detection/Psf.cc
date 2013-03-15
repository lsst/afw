// -*- LSST-C++ -*-
/*!
 * \brief Implementation of Psf code
 *
 * \file
 *
 * \ingroup algorithms
 */
#include <limits>
#include <typeinfo>
#include <cmath>
#include "boost/pointer_cast.hpp"
#include "lsst/pex/logging.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/KernelPsfFactory.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/Distortion.h"
#include "lsst/afw/math/offsetImage.h"

/************************************************************************************************************/

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;
namespace cameraGeom = lsst::afw::cameraGeom;

namespace lsst {
namespace afw {
namespace detection {


/************************************************************************************************************/
//
// Static helper functions for Psf::computeImage()
//

//
// Helper function for resizeKernelImage(); this is called twice for x,y directions
//
// Setup: we have a 1D array of data of legnth @nsrc, with special point ("center") at @srcCtr.
//
// We want to copy this into an array of length @ndst; the lengths need not be the same so we may
// need to zero-pad or truncate.
//
// Outputs:
//   @nout = length of buffer to be copied
//   @dstBase = index of copied output in dst array
//   @srcBase = index of copied output in src array
//   @dstCtr = location of special point in dst array after copy
//
namespace {
    void setup1dResize(int &nout, int &dstBase, int &srcBase, int &dstCtr, int ndst, int nsrc, int srcCtr)
    {
        if (nsrc <= 0 || ndst <= 0 || srcCtr < 0 || srcCtr >= nsrc) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "invalid parameters to setup1dResize()");
        }

        if (nsrc < ndst) {
            // extend by zero padding equally on both sides
            nout = nsrc;
            dstBase = (ndst-nsrc)/2;
            srcBase = 0;
            dstCtr = srcCtr + dstBase;
            return;
        }

        nout = ndst;
        dstBase = 0;
        
        int proposedSrcBase = srcCtr - ndst/2;

        if (proposedSrcBase < 0) {
            // truncate on right only
            srcBase = 0;
        }
        else if (proposedSrcBase + ndst > nsrc) {
            // truncate on left only
            srcBase = nsrc - ndst;
        }
        else {
            // truncate symmetrically around srcCtr
            srcBase = proposedSrcBase;
        }

        dstCtr = srcCtr - srcBase;

        //
        // The following sequence of asserts might be a little paranoid, but
        // this routine is only called on "heavyweight" code paths, so there's
        // no cost to being exhaustive...
        //

        assert(dstCtr >= 0 && dstCtr < ndst);
        assert(srcBase >= 0 && srcBase+ndst <= nsrc);
    }
}


afwGeom::Point2I Psf::resizeKernelImage(Image &dst, const Image &src, const afwGeom::Point2I &ctr)
{
    int nx, dstX0, srcX0, ctrX0;
    int ny, dstY0, srcY0, ctrY0;

    setup1dResize(nx, dstX0, srcX0, ctrX0, dst.getWidth(), src.getWidth(), ctr.getX());
    setup1dResize(ny, dstY0, srcY0, ctrY0, dst.getHeight(), src.getHeight(), ctr.getY());

    afwGeom::Extent2I subimage_size(nx,ny);

    Image sub_dst(dst, afwGeom::Box2I(afwGeom::Point2I(dstX0, dstY0),
				      afwGeom::Extent2I(nx,ny)));

    Image sub_src(src, afwGeom::Box2I(afwGeom::Point2I(srcX0, srcY0),
				      afwGeom::Extent2I(nx,ny)));

    dst = 0.;
    sub_dst <<= sub_src;
    return afwGeom::Point2I(ctrX0, ctrY0);
}


PTR(afwImage::Image<double>) 
Psf::recenterKernelImage(PTR(Image) im, const afwGeom::Point2I &ctr,  const afwGeom::Point2D &xy, 
                         std::string const &warpAlgorithm, unsigned int warpBuffer)
{
    // "ir" : (integer, residual)
    std::pair<int,double> const irX = afwImage::positionToIndex(xy.getX(), true);
    std::pair<int,double> const irY = afwImage::positionToIndex(xy.getY(), true);
    
    if (irX.second != 0.0 || irY.second != 0.0)
        im = afwMath::offsetImage(*im, irX.second, irY.second, warpAlgorithm, warpBuffer);

    im->setXY0(irX.first - ctr.getX(), irY.first - ctr.getY());
    return im;
}



/************************************************************************************************************/
/** Return an Image of the PSF
 *
 * Evaluates the PSF at the specified point, and for neutrally coloured source
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        afwGeom::Extent2I const& size, ///< Desired size of Image (overriding natural size of Kernel)
        bool normalizePeak,            ///< normalize the image to have a maximum value of 1.0
        bool distort                   ///< generate an image that includes the known camera distortion
                                    ) const
{
    lsst::afw::image::Color color;
    afwGeom::Point2D const ccdXY = lsst::afw::geom::Point2D(0, 0);

    return doComputeImage(color, ccdXY, size, normalizePeak, distort);
}

/** Return an Image of the PSF
 *
 * Evaluates the PSF at the specified point, and for neutrally coloured source
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        afwGeom::Point2D const& ccdXY,  ///< Position in image where PSF should be created
        bool normalizePeak,             ///< normalize the image to have a maximum value of 1.0
        bool distort                    ///< generate an image that includes the known camera distortion
                                    ) const
{
    lsst::afw::image::Color color;
    afwGeom::Extent2I const& size=lsst::afw::geom::Extent2I(0, 0);

    return doComputeImage(color, ccdXY, size, normalizePeak, distort);
}

/** Return an Image of the PSF
 *
 * Unless otherwise specified, the image is of the "natural" size, and correct for the point (0,0);
 * a neutrally coloured source is assumed
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        afwGeom::Point2D const& ccdXY,  ///< Position in image where PSF should be created
        afwGeom::Extent2I const& size,  ///< Desired size of Image (overriding natural size of Kernel)
        bool normalizePeak,             ///< normalize the image to have a maximum value of 1.0
        bool distort                    ///< generate an image that includes the known camera distortion
                                    ) const
{
    lsst::afw::image::Color color;
    return doComputeImage(color, ccdXY, size, normalizePeak, distort);

}

/** Return an Image of the PSF
 *
 * Unless otherwise specified, the image is of the "natural" size, and correct for the point (0,0)
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        lsst::afw::image::Color const& color, ///< Colour of source whose PSF is desired
        afwGeom::Point2D const& ccdXY,        ///< Position in image where PSF should be created
        afwGeom::Extent2I const& size,        ///< Desired size of Image (overriding natural size of Kernel)
        bool normalizePeak,                   ///< normalize the image to have a maximum value of 1.0
        bool distort                          ///< generate an image that includes the known camera distortion
                            ) const
{
    return doComputeImage(color, ccdXY, size, normalizePeak, distort);
}

/************************************************************************************************************/
/**
 * Return an Image of the the Psf at the point (x, y), setting the peak pixel (if centered) to 1.0
 *
 * The specified position is a floating point number, and the resulting image will
 * have a Psf with the correct fractional position, with the centre within pixel (width/2, height/2)
 * Specifically, fractional positions in [0, 0.5] will appear above/to the right of the center,
 * and fractional positions in (0.5, 1] will appear below/to the left (0.9999 is almost back at middle)
 *
 * The image's (X0, Y0) will be set correctly to reflect this 
 *
 * @note If a fractional position is specified, the calculated central pixel value may be less than 1.0
 */
Psf::Image::Ptr Psf::doComputeImage(
        lsst::afw::image::Color const& color,  ///< Colour of source
        lsst::afw::geom::Point2D const& ccdXY, ///< Position in parent (CCD) image
        lsst::afw::geom::Extent2I const& size, ///< Size of PSF image
        bool normalizePeak,                    ///< normalize the image to have a maximum value of 1.0
        bool distort                           ///< generate an image that includes the known camera distortion
                                           ) const
{
    if (distort) {
        if (!_detector) {
            distort = false;
        }
    }
    if (distort and !_detector->getDistortion()) {
        pexLog::Debug("afw.detection.Psf").debug<5>(
                          "Requested a distorted image but Detector.getDistortion() is NULL");

        distort = false;
    }
    
    // if they want it distorted, assume they want the PSF as it would appear
    // at ccdXY.  We'll undistort ccdXY to figure out where that point started
    // ... that's where it's really being distorted from!
    afwGeom::Point2D ccdXYundist = ccdXY;
#if 0
    if (distort) {
        ccdXYundist = _detector->getDistortion()->undistort(ccdXY, *_detector);
    } else {
        ccdXYundist = ccdXY;
    }
#endif

    afwMath::Kernel::ConstPtr kernel = getLocalKernel(ccdXYundist, color);
    if (!kernel) {
        throw LSST_EXCEPT(pexExcept::NotFoundException, "Psf is unable to return a kernel");
    }

    int width =  (size.getX() > 0) ? size.getX() : kernel->getWidth();
    int height = (size.getY() > 0) ? size.getY() : kernel->getHeight();
    afwGeom::Point2I ctr = kernel->getCtr();
    
    Psf::Image::Ptr im = boost::make_shared<Psf::Image>(
        geom::Extent2I(width, height)
    );
    try {
        kernel->computeImage(*im, !normalizePeak, ccdXYundist.getX(), ccdXYundist.getY());
    } catch(lsst::pex::exceptions::InvalidParameterException &e) {

        // OK, they didn't like the size of *im.  Compute a "native" image (i.e. the size of the Kernel)
        afwGeom::Extent2I kwid = kernel->getDimensions();
        Psf::Image::Ptr native_im = boost::make_shared<Psf::Image>(kwid);
        kernel->computeImage(*native_im, !normalizePeak, ccdXYundist.getX(), ccdXYundist.getY());

        // copy the native image into the requested one
	ctr = resizeKernelImage(*im, *native_im, ctr);
    }
    
    //
    // Do we want to normalize to the center being 1.0 (when centered in a pixel)?
    //
    if (normalizePeak) {
	double const centralPixelValue = (*im)(ctr.getX(),ctr.getY());
        *im /= centralPixelValue;
    }
    
    im = recenterKernelImage(im, ctr, ccdXYundist);
            
    // distort the image according to the camera distortion
    if (distort) {        
        cameraGeom::Distortion::ConstPtr distortion = _detector->getDistortion();

#if 1
        int lanc = distortion->getLanczosOrder();
        int edge = abs(0.5*((height > width) ? height : width) *
                       (1.0 - distortion->computeMaxShear(*_detector)));
        edge += lanc;
        Psf::Image::SinglePixel padValue(0.0);
        Psf::Image::Ptr overSizeImg = distortion->distort(ccdXYundist, *im, *_detector, padValue);
        afwGeom::Box2I bbox(afwGeom::Point2I(edge, edge), afwGeom::Extent2I(width-2*edge, height-2*edge));
        
        return Psf::Image::Ptr(new Psf::Image(*overSizeImg, bbox));
#else
        Psf::Image::SinglePixel padValue(0.0);
        // distort as though we're where ccdXY was before it got distorted
        Psf::Image::Ptr imDist = distortion->distort(ccdXYundist, *im, *_detector, padValue);
        // distort() keeps *im centered at ccdXYundist, so now shift to ccdXY
        afwGeom::Point2D shift = ccdXY - afwGeom::Extent2D(ccdXYundist);
        std::string const warpAlgorithm = "lanczos5"; // Algorithm to use in warping
        unsigned int const warpBuffer = 0; // Buffer to use in warping
        Psf::Image::Ptr psfIm = afwMath::offsetImage(*imDist, shift.getX(), shift.getY(),
                                                     warpAlgorithm, warpBuffer);
        return psfIm;
#endif
    } else {
        return im;
    }
}

std::string Psf::getPythonModule() const { return "lsst.afw.detection"; }

//
// We need to make an instance here so as to register it
//
// \cond
namespace {

KernelPsfFactory<> registration("KernelPsf");

} // anonymous

KernelPsfPersistenceHelper const & KernelPsfPersistenceHelper::get() {
    static KernelPsfPersistenceHelper instance;
    return instance;
}

KernelPsfPersistenceHelper::KernelPsfPersistenceHelper() :
    schema(),
    kernel(schema.addField<int>("kernel", "archive ID of nested kernel object"))
{
    schema.getCitizen().markPersistent();
}

std::string KernelPsf::getPersistenceName() const { return "KernelPsf"; }

void KernelPsf::write(OutputArchiveHandle & handle) const {
    static KernelPsfPersistenceHelper const & keys = KernelPsfPersistenceHelper::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    catalog.addNew()->set(keys.kernel, handle.put(_kernel));
    handle.saveCatalog(catalog);
}

// \endcond
}}}

