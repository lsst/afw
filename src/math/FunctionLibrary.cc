// -*- lsst-c++ -*-

#include "boost/make_shared.hpp"
#include "boost/type_traits/is_same.hpp"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst { namespace afw { namespace math {

namespace {

struct FunctionSchema {
    table::Schema schema;
    table::Key< table::Array<double> > params;

    explicit FunctionSchema(int nParams) :
        schema(),
        params(schema.addField< table::Array<double> >("params", "function parameters", nParams))
    {}

    explicit FunctionSchema(table::Schema const & schema_) :
        schema(schema_),
        params(schema["params"])
    {}

};

struct Chebyshev1Function2Schema : public FunctionSchema {
    table::Key< table::Point<double> > min;
    table::Key< table::Point<double> > max;

    explicit Chebyshev1Function2Schema(int nParams) :
        FunctionSchema(nParams),
        min(schema.addField< table::Point<double> >("min", "minimum point for function's bbox")),
        max(schema.addField< table::Point<double> >("max", "maximum point for function's bbox"))
    {}

    explicit Chebyshev1Function2Schema(table::Schema const & schema_) :
        FunctionSchema(schema_),
        min(schema["min"]),
        max(schema["max"])
    {}

};

template <typename ReturnT>
class PolynomialFunction2Factory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        FunctionSchema const keys(catalogs.front().getSchema());
        return boost::make_shared< PolynomialFunction2<ReturnT> >(
            keys.params.extractVector(catalogs.front().front())
        );
    }

    PolynomialFunction2Factory(std::string const & name) : table::io::PersistableFactory(name) {}
};

template <typename ReturnT>
class Chebyshev1Function2Factory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        Chebyshev1Function2Schema keys(catalogs.front().getSchema());
        table::BaseRecord const & record = catalogs.front().front();
        geom::Box2D bbox(record.get(keys.min), record.get(keys.max));
        return boost::make_shared< Chebyshev1Function2<ReturnT> >(
            keys.params.extractVector(record),
            bbox
        );
    }

    Chebyshev1Function2Factory(std::string const & name) : table::io::PersistableFactory(name) {}
};

PolynomialFunction2Factory<float> registrationPoly2F("PolynomialFunction2F");
PolynomialFunction2Factory<double> registrationPoly2D("PolynomialFunction2D");

Chebyshev1Function2Factory<float> registrationCheb2F("Chebyshev1Function2F");
Chebyshev1Function2Factory<double> registrationCheb2D("Chebyshev1Function2D");

} // anonymous

template <typename ReturnT>
std::string PolynomialFunction2<ReturnT>::getPersistenceName() const {
    std::string result;
    if (boost::is_same<ReturnT,float>::value) {
        result = "PolynomialFunction2F";
    } else if (boost::is_same<ReturnT,double>::value) {
        result = "PolynomialFunction2D";
    }
    return result;
}

template <typename ReturnT>
std::string Chebyshev1Function2<ReturnT>::getPersistenceName() const {
    std::string result;
    if (boost::is_same<ReturnT,float>::value) {
        result = "Chebyshev1Function2F";
    } else if (boost::is_same<ReturnT,double>::value) {
        result = "Chebyshev1Function2D";
    }
    return result;
}

template <typename ReturnT>
void PolynomialFunction2<ReturnT>::write(table::io::OutputArchiveHandle & handle) const {
    FunctionSchema const keys(this->getNParameters());
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    keys.params.assignVector(*catalog.addNew(), this->getParameters());
    handle.saveCatalog(catalog);
}

template <typename ReturnT>
void Chebyshev1Function2<ReturnT>::write(table::io::OutputArchiveHandle & handle) const {
    Chebyshev1Function2Schema const keys(this->getNParameters());
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(table::BaseRecord) record = catalog.addNew();
    keys.params.assignVector(*record, this->getParameters());
    geom::Box2D bbox = getXYRange();
    record->set(keys.min, bbox.getMin());
    record->set(keys.max, bbox.getMax());
    handle.saveCatalog(catalog);
}

template class PolynomialFunction2<float>;
template class PolynomialFunction2<double>;

template class Chebyshev1Function2<float>;
template class Chebyshev1Function2<double>;

}}} // namespace lsst::afw::math
