// -*- LSST-C++ -*-
/*!
 * \brief Implementation of Psf code
 *
 * \file
 *
 * \ingroup algorithms
 */
#include <typeinfo>
#include <cmath>
#include "lsst/afw/image/ImagePca.h"
#include "lsst/afw/detection/Psf.h"

/************************************************************************************************************/

namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

namespace lsst {
namespace afw {
namespace detection {

Psf::Psf(int const width,               // desired width of Image realisations of the kernel
         int const height               // desired height of Image realisations of the kernel; default: width
        ) :  lsst::daf::data::LsstBase(typeid(this)),
             _kernel(afwMath::Kernel::Ptr()),
             _width(width), _height(height == 0 ? width : height) {}

Psf::Psf(lsst::afw::math::Kernel::Ptr kernel ///< The Kernel corresponding to this Psf
        ) : lsst::daf::data::LsstBase(typeid(this)),
            _kernel(kernel),
            _width(kernel.get()  == NULL ? 0 : kernel->getWidth()),
            _height(kernel.get() == NULL ? 0 : kernel->getHeight()) {}

///
/// Set the Psf's kernel
///
void Psf::setKernel(lsst::afw::math::Kernel::Ptr kernel) {
    _kernel = kernel;
}

///
/// Return the Psf's kernel
///
afwMath::Kernel::Ptr Psf::getKernel() {
    return _kernel;
}

///
/// Return the Psf's kernel
///
boost::shared_ptr<const afwMath::Kernel> Psf::getKernel() const {
    return boost::shared_ptr<const afwMath::Kernel>(_kernel);
}

/**
 * Return an Image of the the Psf at the point (x, y), setting the sum of all the Psf's pixels to 1.0
 *
 * The specified position is a floating point number, and the resulting image will
 * have a Psf with the correct fractional position, with the centre within pixel (width/2, height/2)
 * Specifically, fractional positions in [0, 0.5] will appear above/to the right of the center,
 * and fractional positions in (0.5, 1] will appear below/to the left (0.9999 is almost back at middle)
 *
 * @note If a fractional position is specified, the central pixel value may not be 1.0
 *
 * @note This is a virtual function; we expect that derived classes will do something
 * more useful than returning a NULL pointer
 */
afwImage::Image<Psf::Pixel>::Ptr Psf::getImage(double const, ///< column position in parent %image
                                                double const  ///< row position in parent %image
                                               ) const {
    return afwImage::Image<Psf::Pixel>::Ptr();
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
            throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException,
                              "Unable to lookup Psf variety \"" + name + "\"");
        }
    } else {
        if (!factory) {
            factory = (*el).second;
        } else {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
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
                   lsst::afw::math::Kernel::Ptr kernel ///< Kernel specifying the Psf
                  ) {
    return Psf::lookup(name).create(kernel);
}
    
}}}
