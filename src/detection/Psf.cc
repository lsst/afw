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

namespace lsst { namespace afw { namespace detection {

//-------- Static helper functions for Psf::computeImage() --------------------------------------------------

namespace {

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
void setup1dResize(int &nout, int &dstBase, int &srcBase, int &dstCtr, int ndst, int nsrc, int srcCtr) {
    if (nsrc <= 0 || ndst <= 0 || srcCtr < 0 || srcCtr >= nsrc) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException,
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

} // anonymous

//-------- Psf member function implementations --------------------------------------------------------------

PTR(image::Image<double>) 
Psf::recenterKernelImage(PTR(Image) im, const geom::Point2I &ctr,  const geom::Point2D &xy, 
                         std::string const &warpAlgorithm, unsigned int warpBuffer)
{
    // "ir" : (integer, residual)
    std::pair<int,double> const irX = image::positionToIndex(xy.getX(), true);
    std::pair<int,double> const irY = image::positionToIndex(xy.getY(), true);
    
    if (irX.second != 0.0 || irY.second != 0.0)
        im = math::offsetImage(*im, irX.second, irY.second, warpAlgorithm, warpBuffer);

    im->setXY0(irX.first - ctr.getX(), irY.first - ctr.getY());
    return im;
}

PTR(Psf::Image) Psf::computeImage(geom::Point2D const& ccdXY, bool normalizePeak, bool distort) const {
    image::Color color;
    return doComputeImage(color, ccdXY, normalizePeak, distort);
}

PTR(Psf::Image) Psf::computeImage(
    image::Color const & color, geom::Point2D const& ccdXY, bool normalizePeak, bool distort
) const {
    return doComputeImage(color, ccdXY, normalizePeak, distort);
}

PTR(Psf::Image) Psf::doComputeImage(
    image::Color const& color, geom::Point2D const& ccdXY, bool normalizePeak, bool distort
) const {
    if (distort) {
        if (!_detector) {
            distort = false;
        }
    }
    if (distort and !_detector->getDistortion()) {
        pex::logging::Debug("afw.detection.Psf").debug<5>(
            "Requested a distorted image but Detector.getDistortion() is NULL"
        );
        distort = false;
    }
    
    // if they want it distorted, assume they want the PSF as it would appear
    // at ccdXY.  We'll undistort ccdXY to figure out where that point started
    // ... that's where it's really being distorted from!
    geom::Point2D ccdXYundist = ccdXY;
#if 0
    if (distort) {
        ccdXYundist = _detector->getDistortion()->undistort(ccdXY, *_detector);
    } else {
        ccdXYundist = ccdXY;
    }
#endif

    PTR(math::Kernel const) kernel = getLocalKernel(ccdXYundist, color);
    if (!kernel) {
        throw LSST_EXCEPT(pex::exceptions::NotFoundException, "Psf is unable to return a kernel");
    }

    int width =  kernel->getWidth();
    int height = kernel->getHeight();
    geom::Point2I ctr = kernel->getCtr();
    
    PTR(Psf::Image) im = boost::make_shared<Psf::Image>(geom::Extent2I(width, height));
    kernel->computeImage(*im, !normalizePeak, ccdXYundist.getX(), ccdXYundist.getY());
    
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
        geom::Box2I bbox(geom::Point2I(edge, edge), geom::Extent2I(width-2*edge, height-2*edge));
        
        return Psf::Image::Ptr(new Psf::Image(*overSizeImg, bbox));
#else
        Psf::Image::SinglePixel padValue(0.0);
        // distort as though we're where ccdXY was before it got distorted
        Psf::Image::Ptr imDist = distortion->distort(ccdXYundist, *im, *_detector, padValue);
        // distort() keeps *im centered at ccdXYundist, so now shift to ccdXY
        geom::Point2D shift = ccdXY - geom::Extent2D(ccdXYundist);
        std::string const warpAlgorithm = "lanczos5"; // Algorithm to use in warping
        unsigned int const warpBuffer = 0; // Buffer to use in warping
        Psf::Image::Ptr psfIm = math::offsetImage(*imDist, shift.getX(), shift.getY(),
                                                     warpAlgorithm, warpBuffer);
        return psfIm;
#endif
    } else {
        return im;
    }
}

//-------- Psf and KernelPsf Persistence --------------------------------------------------------------------

std::string Psf::getPythonModule() const { return "lsst.afw.detection"; }

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

}}} // namespace lsst::afw::detection
