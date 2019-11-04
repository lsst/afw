/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "lsst/afw/math/PixelAreaBoundedField.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

template std::shared_ptr<math::PixelAreaBoundedField>
    table::io::PersistableFacade<math::PixelAreaBoundedField>::dynamicCast(
        std::shared_ptr<table::io::Persistable> const&
    );

} // namespace io
} // namespace table

namespace math {

PixelAreaBoundedField::PixelAreaBoundedField(
    lsst::geom::Box2I const &bbox,
    std::shared_ptr<geom::SkyWcs const> skyWcs,
    lsst::geom::AngleUnit const & unit,
    double scaling
) : BoundedField(bbox),
    _skyWcs(skyWcs),
    _scaling(scaling)
{
    if (_skyWcs == nullptr) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            "SkyWcs passed to PixelAreaBoundedField is null."
        );
    }
    _scaling /= std::pow((1.0*unit).asRadians(), 2);
}

double PixelAreaBoundedField::evaluate(lsst::geom::Point2D const &position) const {
    return std::pow(_skyWcs->getPixelScale(position).asRadians(), 2) * _scaling;
}

ndarray::Array<double, 1, 1> PixelAreaBoundedField::evaluate(
    ndarray::Array<double const, 1> const & x,
    ndarray::Array<double const, 1> const & y
) const {
    if (x.getShape() != y.getShape()) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            (boost::format("Inconsistent shapes in evaluate; %s != %s.") % x.getShape() % y.getShape()).str()
        );
    }
    // Compute _skyWcs->pixelToSky at all of the given points in a single
    // vectorized call, along with points one pixel away in x and y.
    double constexpr side = 1.0;
    std::size_t const n = x.size();
    std::vector<lsst::geom::Point2D> pixPoints;
    pixPoints.reserve(n*3);
    for (std::size_t i = 0; i < n; ++i) {
        pixPoints.emplace_back(x[i], y[i]);
        pixPoints.emplace_back(x[i] + side, y[i]);
        pixPoints.emplace_back(x[i], y[i] + side);
    }
    auto skyPoints = _skyWcs->pixelToSky(pixPoints);
    // Work in 3-space to avoid RA wrapping and pole issues.
    ndarray::Array<double, 1, 1> z = ndarray::allocate(x.getShape());
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = i*3;
        auto skyLL = skyPoints[j].getVector();
        auto skyDx = skyPoints[j + 1].getVector() - skyLL;
        auto skyDy = skyPoints[j + 2].getVector() - skyLL;
        double skyAreaSq = skyDx.cross(skyDy).getSquaredNorm();
        z[i] = _scaling * std::sqrt(skyAreaSq) / (side*side);
    }
    return z;
}

bool PixelAreaBoundedField::isPersistable() const noexcept {
    return _skyWcs->isPersistable();
}

std::shared_ptr<BoundedField> PixelAreaBoundedField::operator*(double const scale) const {
    return std::make_shared<PixelAreaBoundedField>(getBBox(), _skyWcs, lsst::geom::radians, _scaling*scale);
}

bool PixelAreaBoundedField::operator==(BoundedField const &rhs) const {
    auto rhsCasted = dynamic_cast<PixelAreaBoundedField const *>(&rhs);
    if (!rhsCasted) return false;

    return getBBox() == rhsCasted->getBBox() && *_skyWcs == *rhsCasted->_skyWcs &&
        _scaling == rhsCasted->_scaling;
}

std::string PixelAreaBoundedField::toString() const {
    std::ostringstream os;
    os << "PixelAreaBoundedField(" << (*_skyWcs) << ", scaling=" << _scaling << ")";
    return os.str();
}


namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Box2IKey bbox;
    table::Key<int> wcs;
    table::Key<double> scaling;

    static PersistenceHelper const & get() {
        static PersistenceHelper const instance;
        return instance;
    }

private:
    PersistenceHelper() :
        schema(),
        bbox(table::Box2IKey::addFields(schema, "bbox", "Bounding box for field.", "pixel")),
        wcs(schema.addField<int>("wcs", "Archive ID for SkyWcs instance.")),
        scaling(schema.addField<double>("scaling",
                                        "Scaling factor (including any transformation from rad^2."))
    {}
    PersistenceHelper(PersistenceHelper const &) = delete;
    PersistenceHelper(PersistenceHelper &&) = delete;
    PersistenceHelper & operator=(PersistenceHelper const &) = delete;
    PersistenceHelper & operator=(PersistenceHelper &&) = delete;
    ~PersistenceHelper() noexcept = default;
};

class PixelAreaBoundedFieldFactory : public table::io::PersistableFactory {
public:
    explicit PixelAreaBoundedFieldFactory(std::string const& name)
            : afw::table::io::PersistableFactory(name) {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        table::BaseRecord const& record = catalogs.front().front();
        auto const & keys = PersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        lsst::geom::Box2I bbox(record.get(keys.bbox));
        auto wcs = archive.get<afw::geom::SkyWcs>(record.get(keys.wcs));
        double scaling = record.get(keys.scaling);
        return std::make_shared<PixelAreaBoundedField>(bbox, wcs, lsst::geom::radians, scaling);
    }
};

std::string getPixelAreaBoundedFieldPersistenceName() { return "PixelAreaBoundedField"; }

PixelAreaBoundedFieldFactory registration(getPixelAreaBoundedFieldPersistenceName());

}  // namespace

std::string PixelAreaBoundedField::getPersistenceName() const {
    return getPixelAreaBoundedFieldPersistenceName();
}

std::string PixelAreaBoundedField::getPythonModule() const { return "lsst.afw.math"; }

void PixelAreaBoundedField::write(OutputArchiveHandle& handle) const {
    auto const & keys = PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    record->set(keys.bbox, getBBox());
    record->set(keys.wcs, handle.put(_skyWcs));
    record->set(keys.scaling, _scaling);
    handle.saveCatalog(catalog);
}

}  // namespace math
}  // namespace afw
}  // namespace lsst
