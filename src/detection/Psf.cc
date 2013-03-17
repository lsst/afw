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

PTR(Psf::Image) Psf::computeImage(geom::Point2D const& ccdXY, bool normalizePeak) const {
    image::Color color;
    return doComputeImage(color, ccdXY, normalizePeak);
}

PTR(Psf::Image) Psf::computeImage(
    image::Color const & color, geom::Point2D const& ccdXY, bool normalizePeak
) const {
    return doComputeImage(color, ccdXY, normalizePeak);
}

PTR(Psf::Image) Psf::doComputeImage(
    image::Color const& color, geom::Point2D const& ccdXY, bool normalizePeak
) const {

    PTR(math::Kernel const) kernel = getLocalKernel(ccdXY, color);
    if (!kernel) {
        throw LSST_EXCEPT(pex::exceptions::NotFoundException, "Psf is unable to return a kernel");
    }

    int width =  kernel->getWidth();
    int height = kernel->getHeight();
    geom::Point2I ctr = kernel->getCtr();
    
    PTR(Psf::Image) im = boost::make_shared<Psf::Image>(geom::Extent2I(width, height));
    kernel->computeImage(*im, !normalizePeak, ccdXY.getX(), ccdXY.getY());
    
    //
    // Do we want to normalize to the center being 1.0 (when centered in a pixel)?
    //
    if (normalizePeak) {
	double const centralPixelValue = (*im)(ctr.getX(),ctr.getY());
        *im /= centralPixelValue;
    }
    
    return recenterKernelImage(im, ctr, ccdXY);
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
