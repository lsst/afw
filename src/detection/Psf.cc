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

PTR(Psf::Image) Psf::computeImage(geom::Point2D const& ccdXY) const {
    image::Color color;
    return doComputeImage(color, ccdXY);
}

PTR(Psf::Image) Psf::computeImage(image::Color const & color, geom::Point2D const& ccdXY) const {
    return doComputeImage(color, ccdXY);
}

PTR(Psf::Image) Psf::computeKernelImage(geom::Point2D const & ccdXY) const {
    image::Color color;
    return doComputeKernelImage(color, ccdXY);
}

PTR(Psf::Image) Psf::computeKernelImage(image::Color const & color, geom::Point2D const& ccdXY) const {
    return doComputeKernelImage(color, ccdXY);
}

PTR(math::Kernel const) Psf::getLocalKernel(geom::Point2D const & ccdXY) const {
    image::Color color;
    return getLocalKernel(color, ccdXY);
}

PTR(math::Kernel const) Psf::getLocalKernel(image::Color const & color, geom::Point2D const& ccdXY) const {
    PTR(Image) image = computeKernelImage(color, ccdXY);
    return boost::make_shared<math::FixedKernel>(*image);
}

PTR(Psf::Image) Psf::doComputeImage(image::Color const& color, geom::Point2D const& ccdXY) const {
    PTR(Psf::Image) im = doComputeKernelImage(color, ccdXY);
    return recenterKernelImage(im, ccdXY);
}

//-------- KernelPsf member function implementations --------------------------------------------------------

PTR(Psf::Image) KernelPsf::doComputeKernelImage(
    image::Color const& color, geom::Point2D const& ccdXY
) const {
    PTR(Psf::Image) im = boost::make_shared<Psf::Image>(_kernel->getDimensions());
    geom::Point2I ctr = _kernel->getCtr();
    _kernel->computeImage(*im, true, ccdXY.getX(), ccdXY.getY());
    im->setXY0(geom::Point2I(-ctr.getX(), -ctr.getY()));
    return im;
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
