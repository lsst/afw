// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "boost/make_shared.hpp"

#include "ndarray/eigen.h"

#include "lsst/afw/detection/GaussianPsf.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst { namespace afw { namespace detection {

namespace {

// Read-only singleton struct containing the schema and keys that a GaussianPsf is mapped
// to in record persistence.
struct GaussianPsfPersistenceHelper : private boost::noncopyable {
    afw::table::Schema schema;
    afw::table::PointKey<int> dimensions;
    afw::table::Key<double> sigma;

    static GaussianPsfPersistenceHelper const & get() {
        static GaussianPsfPersistenceHelper instance;
        return instance;
    }

private:
    GaussianPsfPersistenceHelper() :
        schema(),
        dimensions(
            afw::table::PointKey<int>::addFields(
                schema,
                "dimensions",
                "width/height of realization of Psf", "pixels"
            )
        ),
        sigma(schema.addField<double>("sigma", "radius of Gaussian", "pixels"))
    {
        schema.getCitizen().markPersistent();
    }
};

class GaussianPsfFactory : public afw::table::io::PersistableFactory {
public:

    virtual PTR(afw::table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        static GaussianPsfPersistenceHelper const & keys = GaussianPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        afw::table::BaseRecord const & record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        return boost::make_shared<GaussianPsf>(
            record.get(keys.dimensions.getX()),
            record.get(keys.dimensions.getY()),
            record.get(keys.sigma)
        );
    }

    GaussianPsfFactory(std::string const & name) : afw::table::io::PersistableFactory(name) {}

};

GaussianPsfFactory registration("GaussianPsf");

void checkDimensions(geom::Extent2I const & dimensions) {
    if (dimensions.getX() % 2 == 0 || dimensions.getY() % 2 == 2) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            "GaussianPsf dimensions must be odd"
        );
    }
}

} // anonymous

GaussianPsf::GaussianPsf(int width, int height, double sigma) :
    Psf(true), _dimensions(width, height), _sigma(sigma)
{
    checkDimensions(_dimensions);
}

GaussianPsf::GaussianPsf(geom::Extent2I const & dimensions, double sigma) :
    Psf(true), _dimensions(dimensions), _sigma(sigma)
{
    checkDimensions(_dimensions);
}

PTR(afw::detection::Psf) GaussianPsf::clone() const {
    return boost::make_shared<GaussianPsf>(_dimensions, _sigma);
}

std::string GaussianPsf::getPersistenceName() const { return "GaussianPsf"; }

std::string GaussianPsf::getPythonModule() const { return "lsst.afw.detection"; }

void GaussianPsf::write(OutputArchiveHandle & handle) const {
    static GaussianPsfPersistenceHelper const & keys = GaussianPsfPersistenceHelper::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(afw::table::BaseRecord) record = catalog.addNew();
    (*record)[keys.dimensions.getX()] = _dimensions.getX();
    (*record)[keys.dimensions.getY()] = _dimensions.getY();
    (*record)[keys.sigma] = getSigma();
    handle.saveCatalog(catalog);
}

PTR(GaussianPsf::Image) GaussianPsf::doComputeKernelImage(
    geom::Point2D const &, image::Color const &
) const {
    PTR(Image) r(new Image(_dimensions));
    Image::Array array = r->getArray();
    r->setXY0(geom::Point2I(-_dimensions / 2)); // integer truncation intentional
    double sum = 0.0;
    for (int yIndex = 0, y = r->getY0(); yIndex < _dimensions.getY(); ++yIndex, ++y) {
        Image::Array::Reference row = array[yIndex];
        for (int xIndex = 0, x = r->getX0(); xIndex < _dimensions.getX(); ++xIndex, ++x) {
            sum += row[xIndex] = std::exp(-0.5*(x*x + y*y)/(_sigma*_sigma));
        }
    }
    array.asEigen() /= sum;
    return r;
}

double GaussianPsf::doComputeApertureFlux(
    double radius, geom::Point2D const & position, image::Color const & color
) const {
    return 1.0 - std::exp(-0.5*radius*radius/(_sigma*_sigma));
}

geom::ellipses::Quadrupole GaussianPsf::doComputeShape(
    geom::Point2D const & position, image::Color const & color
) const {
    return geom::ellipses::Quadrupole(_sigma*_sigma, _sigma*_sigma, 0.0);
}

}}} // namespace lsst::afw::detection
