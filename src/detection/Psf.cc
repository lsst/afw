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
/** Return an Image of the PSF
 *
 * Evaluates the PSF at the specified point, and for neutrally coloured source
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        afwGeom::Extent2I const& size, ///< Desired size of Image (overriding natural size of Kernel)
        bool normalizePeak,              ///< normalize the image to have a maximum value of 1.0
        bool distort
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

    afwMath::Kernel::ConstPtr kernel = getKernel(color);
    if (!kernel) {
        throw LSST_EXCEPT(pexExcept::NotFoundException, "Psf is unable to return a kernel");
    }
    int width =  (size.getX() > 0) ? size.getX() : kernel->getWidth();
    int height = (size.getY() > 0) ? size.getY() : kernel->getHeight();
    
    
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
        *im = 0.0;

        std::pair<int, int> x0, y0;
        int w, h;
        if (native_im->getWidth() > im->getWidth()) {
            x0.first = 0;
            x0.second = (native_im->getWidth() - im->getWidth())/2;
            w = im->getWidth();
        } else {
            x0.first = (im->getWidth() - native_im->getWidth())/2;
            x0.second = 0;
            w = native_im->getWidth();
        }
        
        if (native_im->getHeight() > im->getHeight()) {
            y0.first = 0;
            y0.second = (native_im->getHeight() - im->getHeight())/2;
            h = im->getHeight();
        } else {
            y0.first = (im->getHeight() - native_im->getHeight())/2;
            y0.second = 0;
            h = native_im->getHeight();
        }

        Psf::Image sim(*im, afwGeom::Box2I(afwGeom::Point2I(x0.first, y0.first),
                                           afwGeom::Extent2I(w, h)));
        Psf::Image snative_im(*native_im, afwGeom::Box2I(afwGeom::Point2I(x0.second, y0.second),
                                                         afwGeom::Extent2I(w, h)));
        sim <<= snative_im;
        im->setXY0(snative_im.getX0() + (x0.second - x0.first),
                   snative_im.getY0() + (y0.second - y0.first));
    }
    
    //
    // Do we want to normalize to the center being 1.0 (when centered in a pixel)?
    //
    if (normalizePeak) {
        double const centralPixelValue = (*im)(kernel->getCtrX(), kernel->getCtrY());
        *im /= centralPixelValue;
    }
    // "ir" : (integer, residual)
    std::pair<int, double> const ir_dx = lsst::afw::image::positionToIndex(ccdXYundist.getX(), true);
    std::pair<int, double> const ir_dy = lsst::afw::image::positionToIndex(ccdXYundist.getY(), true);
    
    if (ir_dx.second != 0.0 || ir_dy.second != 0.0) {
        std::string const warpAlgorithm = "lanczos5"; // Algorithm to use in warping
        unsigned int const warpBuffer = 5; // Buffer to use in warping        
        im = lsst::afw::math::offsetImage(*im, ir_dx.second, ir_dy.second, warpAlgorithm, warpBuffer);
    }
    im->setXY0(ir_dx.first - kernel->getCtrX() + (ir_dx.second <= 0.5 ? 0 : 1),
               ir_dy.first - kernel->getCtrY() + (ir_dy.second <= 0.5 ? 0 : 1));

            
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

/************************************************************************************************************/
/*
 * Register a factory object by name;  if the factory's NULL, return the named factory
 */
PsfFactoryBase& Psf::_registry(std::string const& name, PsfFactoryBase* factory) {
    static std::map<std::string const, PsfFactoryBase *> psfRegistry;

    std::map<std::string const, PsfFactoryBase *>::iterator el = psfRegistry.find(name);

    if (el == psfRegistry.end()) {      // failed to find name
        if (factory) {
            psfRegistry[name] = factory;
        } else {
            throw LSST_EXCEPT(pexExcept::NotFoundException,
                              "Unable to lookup Psf variety \"" + name + "\"");
        }
    } else {
        if (!factory) {
            factory = (*el).second;
        } else {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                              "Psf variety \"" + name + "\" is already declared");
        }
    }

    return *factory;
}

/**
 * Declare a PsfFactory for a variety "name"
 *
 * @throws lsst::pex::exceptions::InvalidParameterException if name is already declared
 */
void Psf::declare(std::string name,          ///< name of variety
                  PsfFactoryBase* factory ///< Factory to make this sort of Psf
                 ) {
    (void)_registry(name, factory);
}

/**
 * Return the named PsfFactory
 *
 * @throws lsst::pex::exceptions::NotFoundException if name can't be found
 */
PsfFactoryBase& Psf::lookup(std::string name ///< desired variety
                                 ) {
    return _registry(name, NULL);
}

/************************************************************************************************************/
/**
 * Return a Psf of the requested variety
 *
 * @throws std::runtime_error if name can't be found
 */
Psf::Ptr createPsf(std::string const& name,       ///< desired variety
                   int width,                     ///< Number of columns in realisations of Psf
                   int height,                    ///< Number of rows in realisations of Psf
                   double p0,                     ///< Psf's 1st parameter
                   double p1,                     ///< Psf's 2nd parameter
                   double p2                      ///< Psf's 3rd parameter
            ) {
    return Psf::lookup(name).create(width, height, p0, p1, p2);
}

/**
 * Return a Psf of the requested variety
 *
 * @throws std::runtime_error if name can't be found
 */
Psf::Ptr createPsf(std::string const& name,             ///< desired variety
                   afwMath::Kernel::Ptr kernel          ///< Kernel specifying the Psf
                  ) {
    return Psf::lookup(name).create(kernel);
}

//
// We need to make an instance here so as to register it
//
// \cond
namespace {
    volatile bool isInstance =
        Psf::registerMe<KernelPsf, afwMath::Kernel::Ptr>("Kernel");
}
// \endcond
}}}

