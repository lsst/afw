// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include <sstream>
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

afwMath::DeltaFunctionKernel::DeltaFunctionKernel(
    int width,
    int height,
    afwGeom::Point2I const &point
) :
    Kernel(width, height, 0),
    _pixel(point)
{
    if (point.getX() < 0 || point.getX() >= width || point.getY() < 0 || point.getY() >= height) {
        std::ostringstream os;
        os << "point (" << point.getX() << ", " << point.getY() << ") lies outside "
            << width << "x" << height << " sized kernel";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}

PTR(afwMath::Kernel) afwMath::DeltaFunctionKernel::clone() const {
    PTR(afwMath::Kernel) retPtr(new afwMath::DeltaFunctionKernel(this->getWidth(), this->getHeight(),
        this->_pixel));
    retPtr->setCtr(this->getCtr());
    return retPtr;
}

std::string afwMath::DeltaFunctionKernel::toString(std::string const& prefix) const {
    const int pixelX = getPixel().getX(); // active pixel in Kernel
    const int pixelY = getPixel().getY();

    std::ostringstream os;            
    os << prefix << "DeltaFunctionKernel:" << std::endl;
    os << prefix << "Pixel (c,r) " << pixelX << "," << pixelY << ")" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

double afwMath::DeltaFunctionKernel::doComputeImage(
    afwImage::Image<Pixel> &image,
    bool
) const {
    const int pixelX = getPixel().getX(); // active pixel in Kernel
    const int pixelY = getPixel().getY();

    image = 0;
    *image.xy_at(pixelX, pixelY) = 1;

    return 1;
}

// ------ Persistence ---------------------------------------------------------------------------------------

namespace lsst { namespace afw { namespace math {

namespace {

struct DeltaFunctionKernelPersistenceHelper : public Kernel::PersistenceHelper, private boost::noncopyable {
    table::PointKey<int> pixel;

    static DeltaFunctionKernelPersistenceHelper const & get() {
        static DeltaFunctionKernelPersistenceHelper const instance;
        return instance;
    }

private:

    explicit DeltaFunctionKernelPersistenceHelper() :
        Kernel::PersistenceHelper(0),
        pixel(table::PointKey<int>::addFields(schema, "pixel", "position of nonzero pixel", "pixels"))
    {
       schema.getCitizen().markPersistent();
    }

};

} // anonymous

class DeltaFunctionKernel::Factory : public afw::table::io::PersistableFactory {
public:

    virtual PTR(afw::table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        DeltaFunctionKernelPersistenceHelper const & keys = DeltaFunctionKernelPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        afw::table::BaseRecord const & record = catalogs.front().front();
        PTR(DeltaFunctionKernel) result(
            new DeltaFunctionKernel(record.get(keys.dimensions.getX()), record.get(keys.dimensions.getY()),
                                    record.get(keys.pixel))
        );
        result->setCtr(record.get(keys.center));
        return result;
    }

    explicit Factory(std::string const & name) : afw::table::io::PersistableFactory(name) {}
};

namespace {

std::string getDeltaFunctionKernelPersistenceName() { return "DeltaFunctionKernel"; }

DeltaFunctionKernel::Factory registration(getDeltaFunctionKernelPersistenceName());

} // anonymous

std::string DeltaFunctionKernel::getPersistenceName() const {
    return getDeltaFunctionKernelPersistenceName();
}

void DeltaFunctionKernel::write(OutputArchiveHandle & handle) const {
    DeltaFunctionKernelPersistenceHelper const & keys = DeltaFunctionKernelPersistenceHelper::get();
    PTR(afw::table::BaseRecord) record = keys.write(handle, *this);
    record->set(keys.pixel, _pixel);
}

}}} // namespace lsst::afw::math
