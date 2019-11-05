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

#include <algorithm>

#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/math/ProductBoundedField.h"
#include "lsst/afw/math/ChebyshevBoundedField.h"

namespace lsst {
namespace afw {

template std::shared_ptr<math::ProductBoundedField> table::io::PersistableFacade<
        math::ProductBoundedField>::dynamicCast(std::shared_ptr<table::io::Persistable> const&);

namespace math {

namespace {

lsst::geom::Box2I checkAndExtractBBox(std::vector<std::shared_ptr<BoundedField const>> const & factors) {
    if (factors.size() < 1u) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            "ProductBoundedField requires at least one BoundedField factor."
        );
    }
    auto iter = factors.begin();
    auto bbox = (**iter).getBBox();
    ++iter;
    for (; iter != factors.end(); ++iter) {
        if ((**iter).getBBox() != bbox) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterError,
                (boost::format("Inconsistency in ProductBoundedField bboxes: %s != %s") %
                 bbox % (**iter).getBBox()).str()
            );
        }
    }
    return bbox;
}

} // anonymous

ProductBoundedField::ProductBoundedField(std::vector<std::shared_ptr<BoundedField const>> const & factors) :
    BoundedField(checkAndExtractBBox(factors)), _factors(factors)
{}

ProductBoundedField::ProductBoundedField(ProductBoundedField const&) = default;
ProductBoundedField::ProductBoundedField(ProductBoundedField&&) = default;
ProductBoundedField::~ProductBoundedField() = default;

double ProductBoundedField::evaluate(lsst::geom::Point2D const& position) const {
    double product = 1.0;
    for (auto const & field : _factors) {
        product *= field->evaluate(position);
    }
    return product;
}

ndarray::Array<double, 1, 1> ProductBoundedField::evaluate(
    ndarray::Array<double const, 1> const& x,
    ndarray::Array<double const, 1> const& y
) const {
    if (x.getShape() != y.getShape()) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            (boost::format("Inconsistent shapes: %s != %s") % x.getShape() % y.getShape()).str()
        );
    }
    ndarray::Array<double, 1, 1> z = ndarray::allocate(x.getShape());
    std::fill(z.begin(), z.end(), 1.0);
    for (auto const & field : _factors) {
        ndarray::asEigenArray(z) *= ndarray::asEigenArray(field->evaluate(x, y));
    }
    return z;
}

// ------------------ persistence ---------------------------------------------------------------------------

namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Key<int> id;

    static PersistenceHelper const & get() {
        static PersistenceHelper const instance;
        return instance;
    }

private:

    PersistenceHelper() :
        schema(),
        id(schema.addField<int>("id", "Archive ID of a BoundedField factor."))
    {}

    PersistenceHelper(PersistenceHelper const &) = delete;
    PersistenceHelper(PersistenceHelper &&) = delete;
    PersistenceHelper & operator=(PersistenceHelper const &) = delete;
    PersistenceHelper & operator=(PersistenceHelper &&) = delete;

    ~PersistenceHelper() noexcept = default;

};

class ProductBoundedFieldFactory : public table::io::PersistableFactory {
public:
    explicit ProductBoundedFieldFactory(std::string const& name)
            : afw::table::io::PersistableFactory(name) {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        auto const & keys = PersistenceHelper::get();
        auto const & cat = catalogs.front();
        std::vector<std::shared_ptr<BoundedField const>> factors;
        factors.reserve(cat.size());
        for (auto const & record : cat) {
            factors.push_back(archive.get<BoundedField>(record.get(keys.id)));
        }
        return std::make_shared<ProductBoundedField>(factors);
    }
};

std::string getProductBoundedFieldPersistenceName() { return "ProductBoundedField"; }

ProductBoundedFieldFactory registration(getProductBoundedFieldPersistenceName());

}  // namespace

bool ProductBoundedField::isPersistable() const noexcept {
    return std::all_of(_factors.begin(), _factors.end(),
                       [](auto const & field) { return field->isPersistable(); });
}

std::string ProductBoundedField::getPersistenceName() const {
    return getProductBoundedFieldPersistenceName();
}

std::string ProductBoundedField::getPythonModule() const { return "lsst.afw.math"; }

void ProductBoundedField::write(OutputArchiveHandle& handle) const {
    auto const & keys = PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    catalog.reserve(_factors.size());
    for (auto const & field : _factors) {
        catalog.addNew()->set(keys.id, handle.put(field));
    }
    handle.saveCatalog(catalog);
}

// ------------------ operators -----------------------------------------------------------------------------

std::shared_ptr<BoundedField> ProductBoundedField::operator*(double const scale) const {
    std::vector<std::shared_ptr<BoundedField const>> factors(_factors);
    bool multiplied = false;
    for (auto & field : factors) {
        try {
            field = (*field) * scale;
            multiplied = true;
            break;
        } catch (pex::exceptions::LogicError &) {}
    }
    if (!multiplied) {
        ndarray::Array<double, 2, 2> coefficients = ndarray::allocate(1, 1);
        coefficients[0][0] = scale;
        factors.push_back(std::make_shared<ChebyshevBoundedField>(getBBox(), coefficients));
    }
    return std::make_shared<ProductBoundedField>(factors);
}

bool ProductBoundedField::operator==(BoundedField const& rhs) const {
    auto rhsCasted = dynamic_cast<ProductBoundedField const*>(&rhs);
    if (!rhsCasted) return false;

    return (getBBox() == rhsCasted->getBBox()) &&
            std::equal(_factors.begin(), _factors.end(),
                       rhsCasted->_factors.begin(), rhsCasted->_factors.end(),
                       [](auto const & a, auto const & b) { return *a == *b; });
}

std::string ProductBoundedField::toString() const {
    std::ostringstream os;
    os << "ProductBoundedField([";
    for (auto const & field : _factors) {
        os << (*field) << ", ";
    }
    os << "])";
    return os.str();
}

}  // namespace math
}  // namespace afw
}  // namespace lsst
