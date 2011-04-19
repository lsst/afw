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
#include "lsst/afw/detection/Psf.h"

/************************************************************************************************************/

namespace pexExcept = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;

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
        bool normalizePeak              ///< normalize the image to have a maximum value of 1.0
                                    ) const {
    lsst::afw::image::Color color;
    afwGeom::Point2D const ccdXY = lsst::afw::geom::Point2D(0, 0);

    return doComputeImage(color, ccdXY, size, normalizePeak);
}

/** Return an Image of the PSF
 *
 * Evaluates the PSF at the specified point, and for neutrally coloured source
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        afwGeom::Point2D const& ccdXY, ///< Position in image where PSF should be created
        bool normalizePeak              ///< normalize the image to have a maximum value of 1.0
                                    ) const {
    lsst::afw::image::Color color;
    afwGeom::Extent2I const& size=lsst::afw::geom::Extent2I(0, 0);

    return doComputeImage(color, ccdXY, size, normalizePeak);
}

/** Return an Image of the PSF
 *
 * Unless otherwise specified, the image is of the "natural" size, and correct for the point (0,0);
 * a neutrally coloured source is assumed
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        afwGeom::Point2D const& ccdXY, ///< Position in image where PSF should be created
        afwGeom::Extent2I const& size, ///< Desired size of Image (overriding natural size of Kernel)
        bool normalizePeak              ///< normalize the image to have a maximum value of 1.0
                                    ) const {
    lsst::afw::image::Color color;
    return doComputeImage(color, ccdXY, size, normalizePeak);

}

/** Return an Image of the PSF
 *
 * Unless otherwise specified, the image is of the "natural" size, and correct for the point (0,0)
 *
 * \note The real work is done in the virtual function, Psf::doComputeImage
 */
Psf::Image::Ptr Psf::computeImage(
        lsst::afw::image::Color const& color, ///< Colour of source whose PSF is desired
        afwGeom::Point2D const& ccdXY, ///< Position in image where PSF should be created
        afwGeom::Extent2I const& size, ///< Desired size of Image (overriding natural size of Kernel)
        bool normalizePeak              ///< normalize the image to have a maximum value of 1.0
                            ) const {
    return doComputeImage(color, ccdXY, size, normalizePeak);
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
        bool normalizePeak                     ///< normalize the image to have a maximum value of 1.0
                                           ) const {
    afwMath::Kernel::ConstPtr kernel = getKernel(color);
    if (!kernel) {
        throw LSST_EXCEPT(pexExcept::NotFoundException, "Psf is unable to return a kernel");
    }
    int const width =  (size.getX() > 0) ? size.getX() : kernel->getWidth();
    int const height = (size.getY() > 0) ? size.getY() : kernel->getHeight();

    Psf::Image::Ptr im = boost::make_shared<Psf::Image>(
        geom::Extent2I(width, height)
    );
    kernel->computeImage(*im, !normalizePeak, ccdXY.getX(), ccdXY.getY());
    //
    // Do we want to normalize to the center being 1.0 (when centered in a pixel)?
    //
    if (normalizePeak) {
        double const centralPixelValue = (*im)(kernel->getCtrX(), kernel->getCtrY());
        *im /= centralPixelValue;
    }
    // "ir" : (integer, residual)
    std::pair<int, double> const ir_dx = lsst::afw::image::positionToIndex(ccdXY.getX(), true);
    std::pair<int, double> const ir_dy = lsst::afw::image::positionToIndex(ccdXY.getY(), true);
    
    if (ir_dx.second != 0.0 || ir_dy.second != 0.0) {
        im = lsst::afw::math::offsetImage(*im, ir_dx.second, ir_dy.second, "lanczos5");
    }
    im->setXY0(ir_dx.first - kernel->getCtrX() + (ir_dx.second <= 0.5 ? 0 : 1),
               ir_dy.first - kernel->getCtrY() + (ir_dy.second <= 0.5 ? 0 : 1));
    
    return im;
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

