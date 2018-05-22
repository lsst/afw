// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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

#include <cstdint>
#include <memory>
#include <string>

#include "ndarray/eigen.h"
#include "astshim.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/math/TransformBoundedField.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace math {

TransformBoundedField::TransformBoundedField(lsst::geom::Box2I const& bbox, Transform const& transform)
        : BoundedField(bbox), _transform(transform) {
    if (transform.getToEndpoint().getNAxes() != 1) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "To EndPoint of transform must have 1 axis but has " +
                                  std::to_string(transform.getToEndpoint().getNAxes()));
    }
}

double TransformBoundedField::evaluate(lsst::geom::Point2D const& position) const {
    return _transform.applyForward(position)[0];
}

ndarray::Array<double, 1, 1> TransformBoundedField::evaluate(ndarray::Array<double const, 1> const& x,
                                                             ndarray::Array<double const, 1> const& y) const {
    if (x.getSize<0>() != y.getSize<0>()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "x length " + std::to_string(x.getSize<0>()) + "!= y length " +
                                  std::to_string(x.getSize<0>()));
    }

    // TODO if Mapping.applyForward gains support for x, y (DM-11226) then use that instead of copying data
    int const nPoints = x.getSize<0>();
    ndarray::Array<double, 2, 2> xy = ndarray::allocate(ndarray::makeVector(2, nPoints));
    for (int col = 0; col < nPoints; ++col) {
        xy[0][col] = x[col];
        xy[1][col] = y[col];
    }

    auto res2D = _transform.getMapping()->applyForward(xy);

    // res2D has shape 1 x N; return a 1-D view with the extra dimension stripped
    auto resShape = ndarray::makeVector(nPoints);
    auto resStrides = ndarray::makeVector(1);
    return ndarray::external(res2D.getData(), resShape, resStrides, res2D);
}

// ------------------ persistence ---------------------------------------------------------------------------

namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Box2IKey bbox;
    // store the FrameSet as string encoded as a variable-length vector of bytes
    table::Key<table::Array<std::uint8_t>> frameSet;

    PersistenceHelper()
            : schema(),
              bbox(table::Box2IKey::addFields(schema, "bbox", "bounding box", "pixel")),
              frameSet(schema.addField<table::Array<std::uint8_t>>(
                      "frameSet", "FrameSet contained in the Transform", "", 0)) {}

    PersistenceHelper(table::Schema const& s)
            : schema(s), bbox(s["bbox"]), frameSet(s["frameSet"]) {}
};

class TransformBoundedFieldFactory : public table::io::PersistableFactory {
public:
    explicit TransformBoundedFieldFactory(std::string const& name)
            : afw::table::io::PersistableFactory(name) {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        table::BaseRecord const& record = catalogs.front().front();
        PersistenceHelper const keys(record.getSchema());
        auto bbox = record.get(keys.bbox);
        auto frameSetStr = formatters::bytesToString(record.get(keys.frameSet));
        auto transform =
                afw::geom::Transform<afw::geom::Point2Endpoint, afw::geom::GenericEndpoint>::readString(
                        frameSetStr);
        return std::make_shared<TransformBoundedField>(bbox, *transform);
    }
};

std::string getTransformBoundedFieldPersistenceName() { return "TransformBoundedField"; }

TransformBoundedFieldFactory registration(getTransformBoundedFieldPersistenceName());

}  // namespace

std::string TransformBoundedField::getPersistenceName() const {
    return getTransformBoundedFieldPersistenceName();
}

std::string TransformBoundedField::getPythonModule() const { return "lsst.afw.math"; }

void TransformBoundedField::write(OutputArchiveHandle& handle) const {
    PersistenceHelper const keys;
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    record->set(keys.bbox, getBBox());
    record->set(keys.frameSet, formatters::stringToBytes(getTransform().writeString()));
    handle.saveCatalog(catalog);
}

std::shared_ptr<BoundedField> TransformBoundedField::operator*(double const scale) const {
    auto zoomMap = ast::ZoomMap(1, scale);
    auto newMapping = getTransform().getMapping()->then(zoomMap);
    auto newTransform = Transform(newMapping);
    return std::make_shared<TransformBoundedField>(getBBox(), newTransform);
}

bool TransformBoundedField::operator==(BoundedField const& rhs) const {
    auto rhsCasted = dynamic_cast<TransformBoundedField const*>(&rhs);
    if (!rhsCasted) return false;

    return getBBox() == rhsCasted->getBBox() &&
           *(getTransform().getMapping()) == *(rhsCasted->getTransform().getMapping());
}

std::string TransformBoundedField::toString() const {
    return "TransformBoundedField";
}

}  // namespace math
}  // namespace afw
}  // namespace lsst
