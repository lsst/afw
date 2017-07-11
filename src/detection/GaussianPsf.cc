// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include <memory>

#include "ndarray/eigen.h"

#include "lsst/afw/detection/GaussianPsf.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace detection {

namespace {

// Read-only singleton struct containing the schema and keys that a GaussianPsf is mapped
// to in record persistence.
struct GaussianPsfPersistenceHelper {
    afw::table::Schema schema;
    afw::table::PointKey<int> dimensions;
    afw::table::Key<double> sigma;

    static GaussianPsfPersistenceHelper const& get() {
        static GaussianPsfPersistenceHelper instance;
        return instance;
    }

    // No copying
    GaussianPsfPersistenceHelper(const GaussianPsfPersistenceHelper&) = delete;
    GaussianPsfPersistenceHelper& operator=(const GaussianPsfPersistenceHelper&) = delete;

    // No moving
    GaussianPsfPersistenceHelper(GaussianPsfPersistenceHelper&&) = delete;
    GaussianPsfPersistenceHelper& operator=(GaussianPsfPersistenceHelper&&) = delete;

private:
    GaussianPsfPersistenceHelper()
            : schema(),
              dimensions(afw::table::PointKey<int>::addFields(schema, "dimensions",
                                                              "width/height of realization of Psf", "pixel")),
              sigma(schema.addField<double>("sigma", "radius of Gaussian", "pixel")) {
        schema.getCitizen().markPersistent();
    }
};

class GaussianPsfFactory : public afw::table::io::PersistableFactory {
public:
    virtual std::shared_ptr<afw::table::io::Persistable> read(InputArchive const& archive,
                                                              CatalogVector const& catalogs) const {
        static GaussianPsfPersistenceHelper const& keys = GaussianPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        return std::make_shared<GaussianPsf>(record.get(keys.dimensions.getX()),
                                             record.get(keys.dimensions.getY()), record.get(keys.sigma));
    }

    GaussianPsfFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

GaussianPsfFactory registration("GaussianPsf");

void checkDimensions(geom::Extent2I const& dimensions) {
    if (dimensions.getX() % 2 == 0 || dimensions.getY() % 2 == 2) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "GaussianPsf dimensions must be odd");
    }
}

}  // anonymous

GaussianPsf::GaussianPsf(int width, int height, double sigma)
        : Psf(true), _dimensions(width, height), _sigma(sigma) {
    checkDimensions(_dimensions);
}

GaussianPsf::GaussianPsf(geom::Extent2I const& dimensions, double sigma)
        : Psf(true), _dimensions(dimensions), _sigma(sigma) {
    checkDimensions(_dimensions);
}

std::shared_ptr<afw::detection::Psf> GaussianPsf::clone() const {
    return std::make_shared<GaussianPsf>(_dimensions, _sigma);
}

std::shared_ptr<afw::detection::Psf> GaussianPsf::resized(int width, int height) const {
    return std::make_shared<GaussianPsf>(geom::Extent2I(width, height), _sigma);
}

std::string GaussianPsf::getPersistenceName() const { return "GaussianPsf"; }

std::string GaussianPsf::getPythonModule() const { return "lsst.afw.detection"; }

void GaussianPsf::write(OutputArchiveHandle& handle) const {
    static GaussianPsfPersistenceHelper const& keys = GaussianPsfPersistenceHelper::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<afw::table::BaseRecord> record = catalog.addNew();
    (*record)[keys.dimensions.getX()] = _dimensions.getX();
    (*record)[keys.dimensions.getY()] = _dimensions.getY();
    (*record)[keys.sigma] = getSigma();
    handle.saveCatalog(catalog);
}

std::shared_ptr<GaussianPsf::Image> GaussianPsf::doComputeKernelImage(geom::Point2D const&,
                                                                      image::Color const&) const {
    std::shared_ptr<Image> r(new Image(computeBBox()));
    Image::Array array = r->getArray();
    double sum = 0.0;
    for (int yIndex = 0, y = r->getY0(); yIndex < _dimensions.getY(); ++yIndex, ++y) {
        Image::Array::Reference row = array[yIndex];
        for (int xIndex = 0, x = r->getX0(); xIndex < _dimensions.getX(); ++xIndex, ++x) {
            sum += row[xIndex] = std::exp(-0.5 * (x * x + y * y) / (_sigma * _sigma));
        }
    }
    array.asEigen() /= sum;
    return r;
}

double GaussianPsf::doComputeApertureFlux(double radius, geom::Point2D const& position,
                                          image::Color const& color) const {
    return 1.0 - std::exp(-0.5 * radius * radius / (_sigma * _sigma));
}

geom::ellipses::Quadrupole GaussianPsf::doComputeShape(geom::Point2D const& position,
                                                       image::Color const& color) const {
    return geom::ellipses::Quadrupole(_sigma * _sigma, _sigma * _sigma, 0.0);
}

geom::Box2I GaussianPsf::doComputeBBox(geom::Point2D const& position, image::Color const& color) const {
    return geom::Box2I(geom::Point2I(-_dimensions / 2), _dimensions);  // integer truncation intentional
}
}
}
}  // namespace lsst::afw::detection
