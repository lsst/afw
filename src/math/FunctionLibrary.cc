// -*- lsst-c++ -*-

#include <memory>

#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst { namespace afw { namespace math {

namespace {

// Singleton persistence schema for 2-d Gaussians
struct GaussianFunction2PersistenceHelper {
    table::Schema schema;
    table::Key<double> sigma1;
    table::Key<double> sigma2;
    table::Key<double> angle;

    static GaussianFunction2PersistenceHelper const & get() {
        static GaussianFunction2PersistenceHelper instance;
        return instance;
    }

    // No copying
    GaussianFunction2PersistenceHelper (const GaussianFunction2PersistenceHelper&) = delete;
    GaussianFunction2PersistenceHelper& operator=(const GaussianFunction2PersistenceHelper&) = delete;

    // No moving
    GaussianFunction2PersistenceHelper (GaussianFunction2PersistenceHelper&&) = delete;
    GaussianFunction2PersistenceHelper& operator=(GaussianFunction2PersistenceHelper&&) = delete;

private:
    GaussianFunction2PersistenceHelper() :
        schema(),
        sigma1(schema.addField<double>("sigma1", "sigma along axis 1")),
        sigma2(schema.addField<double>("sigma2", "sigma along axis 2")),
        angle(schema.addField<double>("angle", "angle of axis 1 in rad (along x=0, y=pi/2)"))
    {
        schema.getCitizen().markPersistent();
    }
};

// Singleton persistence schema for 2-d circular DoubleGaussians
struct DoubleGaussianFunction2PersistenceHelper {
    table::Schema schema;
    table::Key<double> sigma1;
    table::Key<double> sigma2;
    table::Key<double> ampl2;

    static DoubleGaussianFunction2PersistenceHelper const & get() {
        static DoubleGaussianFunction2PersistenceHelper instance;
        return instance;
    }

    // No copying
    DoubleGaussianFunction2PersistenceHelper (const DoubleGaussianFunction2PersistenceHelper&) = delete;
    DoubleGaussianFunction2PersistenceHelper& operator=(const DoubleGaussianFunction2PersistenceHelper&) = delete;

    // No moving
    DoubleGaussianFunction2PersistenceHelper (DoubleGaussianFunction2PersistenceHelper&&) = delete;
    DoubleGaussianFunction2PersistenceHelper& operator=(DoubleGaussianFunction2PersistenceHelper&&) = delete;

private:
    DoubleGaussianFunction2PersistenceHelper() :
        schema(),
        sigma1(schema.addField<double>("sigma1", "sigma of first Gaussian")),
        sigma2(schema.addField<double>("sigma2", "sigma of second Gaussian")),
        ampl2(schema.addField<double>("ampl2", "peak of second Gaussian relative to peak of first"))
    {
        schema.getCitizen().markPersistent();
    }
};

// Persistence schema for 2-d polynomials; not a singleton because it depends on the order.
struct PolynomialFunction2PersistenceHelper {
    table::Schema schema;
    table::Key< table::Array<double> > coefficients;

    explicit PolynomialFunction2PersistenceHelper(int nCoefficients) :
        schema(),
        coefficients(
            schema.addField< table::Array<double> >(
                "coefficients",
                "polynomial coefficients, ordered (x,y) [0,0; 1,0; 0,1; 2,0; 1,1; 0,2; ...]",
                nCoefficients
            )
        )
    {}

    explicit PolynomialFunction2PersistenceHelper(table::Schema const & schema_) :
        schema(schema_),
        coefficients(schema["coefficients"])
    {}

};

// Persistance schema for 2-d Chebyshevs; not a singleton because it depends on the order.
struct Chebyshev1Function2PersistenceHelper : public PolynomialFunction2PersistenceHelper {
    table::PointKey<double> min;
    table::PointKey<double> max;

    explicit Chebyshev1Function2PersistenceHelper(int nCoefficients) :
        PolynomialFunction2PersistenceHelper(nCoefficients),
        min(table::PointKey<double>::addFields(schema, "min", "minimum point for function's bbox", "pixel")),
        max(table::PointKey<double>::addFields(schema, "max", "maximum point for function's bbox", "pixel"))
    {}

    explicit Chebyshev1Function2PersistenceHelper(table::Schema const & schema_) :
        PolynomialFunction2PersistenceHelper(schema_),
        min(schema["min"]),
        max(schema["max"])
    {}

};

template <typename ReturnT>
class GaussianFunction2Factory : public table::io::PersistableFactory {
public:

    virtual std::shared_ptr<table::io::Persistable>
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        GaussianFunction2PersistenceHelper const & keys = GaussianFunction2PersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema().contains(keys.schema));
        table::BaseRecord const & record = catalogs.front().front();
        return std::make_shared< GaussianFunction2<ReturnT> >(
            record.get(keys.sigma1), record.get(keys.sigma2), record.get(keys.angle)
        );
    }

    GaussianFunction2Factory(std::string const & name) : table::io::PersistableFactory(name) {}
};

template <typename ReturnT>
class DoubleGaussianFunction2Factory : public table::io::PersistableFactory {
public:

    virtual std::shared_ptr<table::io::Persistable>
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        DoubleGaussianFunction2PersistenceHelper const & keys = DoubleGaussianFunction2PersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema().contains(keys.schema));
        table::BaseRecord const & record = catalogs.front().front();
        return std::make_shared< DoubleGaussianFunction2<ReturnT> >(
            record.get(keys.sigma1), record.get(keys.sigma2), record.get(keys.ampl2)
        );
    }

    DoubleGaussianFunction2Factory(std::string const & name) : table::io::PersistableFactory(name) {}
};

template <typename ReturnT>
class PolynomialFunction2Factory : public table::io::PersistableFactory {
public:

    virtual std::shared_ptr<table::io::Persistable>
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        PolynomialFunction2PersistenceHelper const keys(catalogs.front().getSchema());
        return std::make_shared< PolynomialFunction2<ReturnT> >(
            keys.coefficients.extractVector(catalogs.front().front())
        );
    }

    PolynomialFunction2Factory(std::string const & name) : table::io::PersistableFactory(name) {}
};

template <typename ReturnT>
class Chebyshev1Function2Factory : public table::io::PersistableFactory {
public:

    virtual std::shared_ptr<table::io::Persistable>
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        Chebyshev1Function2PersistenceHelper keys(catalogs.front().getSchema());
        table::BaseRecord const & record = catalogs.front().front();
        geom::Box2D bbox(record.get(keys.min), record.get(keys.max));
        return std::make_shared< Chebyshev1Function2<ReturnT> >(
            keys.coefficients.extractVector(record),
            bbox
        );
    }

    Chebyshev1Function2Factory(std::string const & name) : table::io::PersistableFactory(name) {}
};

GaussianFunction2Factory<float> registrationGaussian2F("GaussianFunction2F");
GaussianFunction2Factory<double> registrationGaussian2D("GaussianFunction2D");

DoubleGaussianFunction2Factory<float> registrationDoubleGaussian2F("DoubleGaussianFunction2F");
DoubleGaussianFunction2Factory<double> registrationDoubleGaussian2D("DoubleGaussianFunction2D");

PolynomialFunction2Factory<float> registrationPoly2F("PolynomialFunction2F");
PolynomialFunction2Factory<double> registrationPoly2D("PolynomialFunction2D");

Chebyshev1Function2Factory<float> registrationCheb2F("Chebyshev1Function2F");
Chebyshev1Function2Factory<double> registrationCheb2D("Chebyshev1Function2D");

template <typename T> struct Suffix;
template <> struct Suffix<float> { static std::string get() { return "F"; } };
template <> struct Suffix<double> { static std::string get() { return "D"; } };

} // anonymous

template <typename ReturnT>
std::string GaussianFunction2<ReturnT>::getPersistenceName() const {
    return "GaussianFunction2" + Suffix<ReturnT>::get();
}

template <typename ReturnT>
std::string DoubleGaussianFunction2<ReturnT>::getPersistenceName() const {
    return "DoubleGaussianFunction2" + Suffix<ReturnT>::get();
}

template <typename ReturnT>
std::string PolynomialFunction2<ReturnT>::getPersistenceName() const {
    return "PolynomialFunction2" + Suffix<ReturnT>::get();
}

template <typename ReturnT>
std::string Chebyshev1Function2<ReturnT>::getPersistenceName() const {
    return "Chebyshev1Function2" + Suffix<ReturnT>::get();
}

template <typename ReturnT>
void GaussianFunction2<ReturnT>::write(table::io::OutputArchiveHandle & handle) const {
    GaussianFunction2PersistenceHelper const & keys = GaussianFunction2PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    record->set(keys.sigma1, this->getParameters()[0]);
    record->set(keys.sigma2, this->getParameters()[1]);
    record->set(keys.angle, this->getParameters()[2]);
    handle.saveCatalog(catalog);
}

template <typename ReturnT>
void DoubleGaussianFunction2<ReturnT>::write(table::io::OutputArchiveHandle & handle) const {
    DoubleGaussianFunction2PersistenceHelper const & keys = DoubleGaussianFunction2PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    record->set(keys.sigma1, this->getParameters()[0]);
    record->set(keys.sigma2, this->getParameters()[1]);
    record->set(keys.ampl2, this->getParameters()[2]);
    handle.saveCatalog(catalog);
}

template <typename ReturnT>
void PolynomialFunction2<ReturnT>::write(table::io::OutputArchiveHandle & handle) const {
    PolynomialFunction2PersistenceHelper const keys(this->getNParameters());
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    keys.coefficients.assignVector(*catalog.addNew(), this->getParameters());
    handle.saveCatalog(catalog);
}

template <typename ReturnT>
void Chebyshev1Function2<ReturnT>::write(table::io::OutputArchiveHandle & handle) const {
    Chebyshev1Function2PersistenceHelper const keys(this->getNParameters());
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    keys.coefficients.assignVector(*record, this->getParameters());
    geom::Box2D bbox = getXYRange();
    record->set(keys.min, bbox.getMin());
    record->set(keys.max, bbox.getMax());
    handle.saveCatalog(catalog);
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
    template class IntegerDeltaFunction1<TYPE>; \
    template class IntegerDeltaFunction2<TYPE>; \
    template class GaussianFunction1<TYPE>; \
    template class GaussianFunction2<TYPE>; \
    template class DoubleGaussianFunction2<TYPE>; \
    template class PolynomialFunction1<TYPE>; \
    template class PolynomialFunction2<TYPE>; \
    template class Chebyshev1Function1<TYPE>; \
    template class Chebyshev1Function2<TYPE>; \
    template class LanczosFunction1<TYPE>; \
    template class LanczosFunction2<TYPE>;

INSTANTIATE(float);
INSTANTIATE(double);

}}} // namespace lsst::afw::math
