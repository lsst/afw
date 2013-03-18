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
#include "lsst/afw/math/offsetImage.h"

namespace lsst { namespace afw { namespace detection {

//-------- Psf member function implementations --------------------------------------------------------------

namespace {

// Comparison function that determines when we used the cached image instead of recomputing it.
// We'll probably want a tolerance for colors someday too, but they're just a placeholder right now
// so it's not worth the effort.
bool comparePsfEvalPoints(geom::Point2D const & a, geom::Point2D const & b) {
    // n.b. desired tolerance is actually sqrt(eps), so tolerance squared is eps.
    return (a - b).computeSquaredNorm() < std::numeric_limits<double>::epsilon();
}

} // anonymous

Psf::Psf(bool isFixed) : daf::base::Citizen(typeid(this)), _isFixed(isFixed) {}

PTR(image::Image<double>)
Psf::recenterKernelImage(
    PTR(Image) im, geom::Point2D const & xy, std::string const &warpAlgorithm, unsigned int warpBuffer
) {
    // "ir" : (integer, residual)
    std::pair<int,double> const irX = image::positionToIndex(xy.getX(), true);
    std::pair<int,double> const irY = image::positionToIndex(xy.getY(), true);

    if (irX.second != 0.0 || irY.second != 0.0) {
        im = math::offsetImage(*im, irX.second, irY.second, warpAlgorithm, warpBuffer);
    }

    im->setXY0(irX.first + im->getX0(), irY.first + im->getY0());
    return im;
}

PTR(Psf::Image) Psf::computeImage(
    geom::Point2D const& ccdXY, image::Color color, ImageOwnerEnum owner
) const {
    if (color.isIndeterminate()) color = getAverageColor();
    PTR(Psf::Image) result;
    if (_cachedImage && color == _cachedImageColor
        && comparePsfEvalPoints(ccdXY, _cachedImageCcdXY)
    ) {
        result = _cachedImage;
    } else {
        result = doComputeImage(ccdXY, color);
        _cachedImage = result;
        _cachedImageColor = color;
        _cachedImageCcdXY = ccdXY;
    }
    if (owner == COPY) {
        result = boost::make_shared<Image>(*result, true);
    }
    return result;
}

PTR(Psf::Image) Psf::computeKernelImage(
    geom::Point2D const& ccdXY, image::Color color, ImageOwnerEnum owner
) const {
    if (color.isIndeterminate()) color = getAverageColor();
    PTR(Psf::Image) result;
    if (_cachedKernelImage
        && (_isFixed ||
            (color == _cachedKernelImageColor && comparePsfEvalPoints(ccdXY, _cachedKernelImageCcdXY)))
    ) {
        result = _cachedKernelImage;
    } else {
        result = doComputeKernelImage(ccdXY, color);
        _cachedKernelImage = result;
        _cachedKernelImageColor = color;
        _cachedKernelImageCcdXY = ccdXY;
    }
    if (owner == COPY) {
        result = boost::make_shared<Image>(*result, true);
    }
    return result;
}

PTR(math::Kernel const) Psf::getLocalKernel(geom::Point2D const& ccdXY, image::Color color) const {
    if (color.isIndeterminate()) color = getAverageColor();
    // FixedKernel ctor will deep copy image, so we can use INTERNAL.
    PTR(Image) image = computeKernelImage(ccdXY, color, INTERNAL);
    return boost::make_shared<math::FixedKernel>(*image);
}

double Psf::computePeak(geom::Point2D const & ccdXY, image::Color color) const {
    PTR(Image) image = computeKernelImage(ccdXY, color, INTERNAL);
    return (*image)(-image->getX0(), -image->getY0());
}

PTR(Psf::Image) Psf::doComputeImage(geom::Point2D const& ccdXY, image::Color const& color) const {
    PTR(Psf::Image) im = computeKernelImage(ccdXY, color, COPY);
    return recenterKernelImage(im, ccdXY);
}

//-------- KernelPsf member function implementations --------------------------------------------------------

PTR(Psf::Image) KernelPsf::doComputeKernelImage(
    geom::Point2D const& ccdXY, image::Color const& color
) const {
    PTR(Psf::Image) im = boost::make_shared<Psf::Image>(_kernel->getDimensions());
    geom::Point2I ctr = _kernel->getCtr();
    _kernel->computeImage(*im, true, ccdXY.getX(), ccdXY.getY());
    im->setXY0(geom::Point2I(-ctr.getX(), -ctr.getY()));
    return im;
}

KernelPsf::KernelPsf(math::Kernel const & kernel) :
    Psf(!kernel.isSpatiallyVarying()), _kernel(kernel.clone()) {}

KernelPsf::KernelPsf(PTR(math::Kernel) kernel) :
    Psf(!kernel->isSpatiallyVarying()), _kernel(kernel) {}

PTR(Psf) KernelPsf::clone() const { return boost::make_shared<KernelPsf>(*this); }

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

bool KernelPsf::isPersistable() const { return _kernel->isPersistable(); }

std::string KernelPsf::getPersistenceName() const { return "KernelPsf"; }

void KernelPsf::write(OutputArchiveHandle & handle) const {
    static KernelPsfPersistenceHelper const & keys = KernelPsfPersistenceHelper::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    catalog.addNew()->set(keys.kernel, handle.put(_kernel));
    handle.saveCatalog(catalog);
}

}}} // namespace lsst::afw::detection
