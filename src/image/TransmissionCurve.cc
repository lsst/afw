/*
 * LSST Data Management System
 * Copyright 2017 LSST/AURA.
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
#include <memory>

#include "ndarray.h"

#include "gsl/gsl_interp.h"
#include "gsl/gsl_interp2d.h"
#include "gsl/gsl_errno.h"

#include "lsst/afw/image/TransmissionCurve.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

using namespace std::string_literals;

#define LSST_CHECK_GSL(type, status) \
    if (status) throw LSST_EXCEPT(type, "GSL error: "s + ::gsl_strerror(status))


namespace lsst {
namespace afw {
namespace image {

namespace {

/*
 * The TransmissionCurve implementation returned by TransmissionCurve::makeIdentity.
 *
 * This is zero-state singleton whose throughput is always exactly 1.
 */
class IdentityTransmissionCurve : public TransmissionCurve {
public:

    static constexpr char const * NAME = "IdentityTransmissionCurve";

    static std::shared_ptr<IdentityTransmissionCurve> get() {
        static std::shared_ptr<IdentityTransmissionCurve> instance(new IdentityTransmissionCurve());
        return instance;
    }

    std::pair<double,double> getWavelengthBounds() const override {
        constexpr double inf = std::numeric_limits<double>::infinity();
        return std::make_pair(-inf, inf);
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        return std::make_pair(1.0, 1.0);
    }

    void sampleAt(
        geom::Point2D const &,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {
        out.deep() = 1.0;
    }

    bool isPersistable() const override { return true; }

protected:

    // transforming an IdentityTransmissionCurve is a no-op
    std::shared_ptr<TransmissionCurve const> _transformedByImpl(
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const override {
        return shared_from_this();
    }

    // multiplying an IdentityTransmissionCurve always yields the other operand
    std::shared_ptr<TransmissionCurve const> _multipliedByImpl(
        std::shared_ptr<TransmissionCurve const> other
    ) const override {
        return other;
    }

    std::string getPersistenceName() const override { return NAME; }

    void write(OutputArchiveHandle& handle) const override {
        handle.saveEmpty();
    }

    class Factory : public table::io::PersistableFactory {
    public:

        std::shared_ptr<table::io::Persistable> read(
            InputArchive const& archive,
            CatalogVector const& catalogs
        ) const override {
            LSST_ARCHIVE_ASSERT(catalogs.empty());
            return get();
        }

        Factory() : table::io::PersistableFactory(NAME) {}
    };

    static Factory registration;

private:
    IdentityTransmissionCurve() = default;
};

IdentityTransmissionCurve::Factory IdentityTransmissionCurve::registration;


/*
 * InterpolatedTransmissionCurve: implements makeSpatiallyConstant and makeRadial.
 *
 * InterpolatedTransmissionCurve is templated on an implementation class
 * that does most of the work, letting it handle the boilerplate that is
 * common to both spatially-constant and radially-varying curves.
 *
 * We use two tricks to avoid repetition that bear some explanation:
 *
 *  - Even though the two implementations have different state, we can use
 *    the same PersistenceHelper template for both by using a std::vector
 *    to hold the keys for the arrays.  Impl2d has one extra array to save,
 *    so its vector is one element longer.
 *
 *  - The throughput array held by Impl2d is conceptually a 2-d array, but we
 *    actually store a flattened view into it.  That's okay because both GSL
 *    and the persistence layer only care about the pointer and the size. That
 *    requires going through some hoops in makeRadial, but from then on it's
 *    clean sailing.
 */

template <typename T>
using GslPtr = std::unique_ptr<T, void (*)(T*)>;

template <typename T>
GslPtr<T> makeGslPtr(T* p, void (*free)(T*)) {
    if (p == nullptr) {
        throw LSST_EXCEPT(
            pex::exceptions::MemoryError,
            "Could not allocate GSL object."
        );
    }
    return GslPtr<T>(p, free);
}

using ArrayKeyVector = std::vector<table::Key<table::Array<double>>>;

/*
 * Implementation object as a template parameter for the instantiation of
 * InterpolatedTransmissionCurve used by makeSpatiallyConstant.
 */
class Impl1d {
public:

    static constexpr bool isSpatiallyConstant = true;
    static constexpr char const * NAME = "SpatiallyConstantTransmissionCurve";

    // Initialize the GSL interpolator and take ownership of the given arrays.
    // Array size consistency checks are done by makeSpatiallyConstant,
    // so we don't need to worry about them here.
    Impl1d(ndarray::Array<double,1,1> const & throughput,
           ndarray::Array<double,1,1> const & wavelengths) :
        _throughput(throughput),
        _wavelengths(wavelengths),
        _interp(makeGslPtr(::gsl_interp_alloc(::gsl_interp_linear, _wavelengths.size()),
                           &::gsl_interp_free))
    {
        int status = ::gsl_interp_init(_interp.get(), _wavelengths.getData(), _throughput.getData(),
                                       _wavelengths.size());
        LSST_CHECK_GSL(pex::exceptions::LogicError, status);
    }

    std::pair<double, double> getWavelengthBounds() const {
        return std::make_pair(_wavelengths.front(), _wavelengths.back());
    }

    static void setupPersistence(table::Schema & schema, ArrayKeyVector & keys) {
        keys.push_back(
            schema.addField<table::Array<double>>(
                "throughput", "array of known throughput values", "", 0
            )
        );
        keys.push_back(
            schema.addField<table::Array<double>>(
                "wavelengths", "array of known wavelength values", "angstrom", 0
            )
        );
    }

    void persist(table::BaseRecord & record, ArrayKeyVector const & keys) const {
        record.set(keys[0], _throughput);
        record.set(keys[1], _wavelengths);
    }

    static Impl1d unpersist(table::BaseRecord & record, ArrayKeyVector const & keys) {
        return Impl1d(record[keys[0]], record[keys[1]]);
    }

    // A helper object constructed every time InterpolatedTransmissionCurve::sampleAt
    // is called, and then invoked at every iteration of the loop therein.
    struct Functor {

        Functor(Impl1d const &, geom::Point2D const &) :
            _accel(makeGslPtr(::gsl_interp_accel_alloc(), &gsl_interp_accel_free))
        {}

        double operator()(Impl1d const & parent, double wavelength) {
            double result = 0.0;
            int status = ::gsl_interp_eval_e(parent._interp.get(), parent._wavelengths.getData(),
                                             parent._throughput.getData(),
                                             wavelength, _accel.get(), &result);
            LSST_CHECK_GSL(pex::exceptions::RuntimeError, status);
            return result;
        }

    private:
        GslPtr<::gsl_interp_accel> _accel;
    };

private:
    ndarray::Array<double,1,1> _throughput;
    ndarray::Array<double,1,1> _wavelengths;
    GslPtr<::gsl_interp> _interp;
};

/*
 * Implementation object as a template parameter for the instantiation of
 * InterpolatedTransmissionCurve used by makeRadial.
 */
class Impl2d {
public:

    static constexpr bool isSpatiallyConstant = false;
    static constexpr char const * NAME = "RadialTransmissionCurve";

    Impl2d(ndarray::Array<double,1,1> const & throughput,
           ndarray::Array<double,1,1> const & wavelengths,
           ndarray::Array<double,1,1> const & radii) :
        _throughput(throughput),
        _wavelengths(wavelengths),
        _radii(radii),
        _interp(makeGslPtr(::gsl_interp2d_alloc(::gsl_interp2d_bilinear, _wavelengths.size(),
                                                _radii.size()),
                           &::gsl_interp2d_free))
    {
        int status = ::gsl_interp2d_init(_interp.get(), _wavelengths.getData(), _radii.getData(),
                                         _throughput.getData(), _wavelengths.size(), _radii.size());
        LSST_CHECK_GSL(pex::exceptions::LogicError, status);
    }

    std::pair<double, double> getWavelengthBounds() const {
        return std::make_pair(_wavelengths.front(), _wavelengths.back());
    }

    static void setupPersistence(table::Schema & schema, ArrayKeyVector & keys) {
        keys.push_back(
            schema.addField<table::Array<double>>(
                "throughput",
                "flattenned 2-d array of known throughput values, with radius dimension fastest", "", 0
            )
        );
        keys.push_back(
            schema.addField<table::Array<double>>(
                "wavelengths", "array of known wavelength values", "angstrom", 0
            )
        );
        keys.push_back(
            schema.addField<table::Array<double>>(
                "radii", "array of known radius values", "", 0
            )
        );
    }

    void persist(table::BaseRecord & record, ArrayKeyVector const & keys) const {
        record.set(keys[0], _throughput);
        record.set(keys[1], _wavelengths);
        record.set(keys[2], _radii);
    }

    static Impl2d unpersist(table::BaseRecord & record, ArrayKeyVector const & keys) {
        return Impl2d(record[keys[0]], record[keys[1]], record[keys[2]]);
    }

    // A helper object constructed every time InterpolatedTransmissionCurve::sampleAt
    // is called, and then invoked at every iteration of the loop therein.
    struct Functor {

        Functor(Impl2d const & parent, geom::Point2D const & point) :
            _radius(point.asEigen().norm()),
            _radiusAccel(makeGslPtr(::gsl_interp_accel_alloc(), &gsl_interp_accel_free)),
            _wavelengthAccel(makeGslPtr(::gsl_interp_accel_alloc(), &gsl_interp_accel_free))
        {
            _radius = std::max(_radius, parent._radii.front());
            _radius = std::min(_radius, parent._radii.back());
        }

        double operator()(Impl2d const & parent, double wavelength) {
            double result = 0.0;
            int status = ::gsl_interp2d_eval_e(parent._interp.get(), parent._wavelengths.getData(),
                                               parent._radii.getData(), parent._throughput.getData(),
                                               wavelength, _radius,
                                               _radiusAccel.get(), _wavelengthAccel.get(),
                                               &result);
            LSST_CHECK_GSL(pex::exceptions::RuntimeError, status);
            return result;
        }

    private:
        double _radius;
        GslPtr<::gsl_interp_accel> _radiusAccel;
        GslPtr<::gsl_interp_accel> _wavelengthAccel;
    };

private:
    ndarray::Array<double,1,1> _throughput;
    ndarray::Array<double,1,1> _wavelengths;
    ndarray::Array<double,1,1> _radii;
    GslPtr<::gsl_interp2d> _interp;
};


template <typename Impl>
class InterpolatedTransmissionCurve : public TransmissionCurve {
public:

    InterpolatedTransmissionCurve(Impl impl, std::pair<double, double> throughputAtBounds) :
        _atBounds(throughputAtBounds),
        _impl(std::move(impl))
    {
        if (!std::isfinite(_atBounds.first) || !std::isfinite(_atBounds.second)) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterError,
                "Throughput values at bounds must be finite"
            );
        }
    }

    std::pair<double,double> getWavelengthBounds() const override {
        return _impl.getWavelengthBounds();
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        return _atBounds;
    }

    void sampleAt(
        geom::Point2D const & point,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {
        LSST_THROW_IF_NE(
            wavelengths.getSize<0>(), out.getSize<0>(),
            pex::exceptions::LengthError,
            "Length of wavelength array (%d) does not match size of output array (%d)"
        );
        typename Impl::Functor functor(_impl, point);
        auto bounds = _impl.getWavelengthBounds();
        auto wlIter = wavelengths.begin();
        for (auto outIter = out.begin(); outIter != out.end(); ++outIter, ++wlIter) {
            double & y = *outIter;
            if (*wlIter < bounds.first) {
                y = _atBounds.first;
            } else if (*wlIter > bounds.second) {
                y = _atBounds.second;
            } else {
                y = functor(_impl, *wlIter);
            }
        }
    }

    bool isPersistable() const override { return true; }

protected:

    std::shared_ptr<TransmissionCurve const> _transformedByImpl(
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const override {
        if (_impl.isSpatiallyConstant) {
            return shared_from_this();
        } else {
            return TransmissionCurve::_transformedByImpl(transform);
        }
    }

    std::string getPersistenceName() const override { return Impl::NAME; }

    struct PersistenceHelper {
        table::Schema schema;
        table::Key<double> throughputAtMin;
        table::Key<double> throughputAtMax;
        ArrayKeyVector arrays;

        static PersistenceHelper const & get() {
            static PersistenceHelper const instance;
            return instance;
        }

    private:

        PersistenceHelper() :
            schema(),
            throughputAtMin(schema.addField<double>(
                "throughputAtMin", "throughput below minimum wavelength")),
            throughputAtMax(schema.addField<double>(
                "throughputAtMax", "throughput above minimum wavelength"))
        {
            Impl::setupPersistence(schema, arrays);
            schema.getCitizen().markPersistent();
        }

    };

    void write(OutputArchiveHandle& handle) const override {
        auto const & keys = PersistenceHelper::get();
        auto catalog = handle.makeCatalog(keys.schema);
        auto record = catalog.addNew();
        record->set(keys.throughputAtMin, _atBounds.first);
        record->set(keys.throughputAtMax, _atBounds.second);
        _impl.persist(*record, keys.arrays);
        handle.saveCatalog(catalog);
    }

    class Factory : public table::io::PersistableFactory {
    public:

        std::shared_ptr<table::io::Persistable> read(
            InputArchive const& archive,
            CatalogVector const& catalogs
        ) const override {
            auto const& keys = PersistenceHelper::get();
            LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
            LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
            auto & record = catalogs.front().front();
            return std::make_shared<InterpolatedTransmissionCurve>(
                Impl::unpersist(record, keys.arrays),
                std::make_pair(record.get(keys.throughputAtMin), record.get(keys.throughputAtMax))
            );
        }

        Factory() : table::io::PersistableFactory(Impl::NAME) {}
    };

    static Factory registration;

private:
    std::pair<double,double> _atBounds;
    Impl _impl;
};

template <typename Impl>
typename InterpolatedTransmissionCurve<Impl>::Factory InterpolatedTransmissionCurve<Impl>::registration;

template class InterpolatedTransmissionCurve<Impl1d>;
template class InterpolatedTransmissionCurve<Impl2d>;


/*
 * ProductTransmissionCurve: default for TransmissionCurve::multipliedBy().
 *
 * This is a straightforward lazy-evaluation object.  Its only state is the
 * two operands it delegates to.
 */
class ProductTransmissionCurve : public TransmissionCurve {
public:

    static constexpr char const * NAME = "ProductTransmissionCurve";

    ProductTransmissionCurve(
        std::shared_ptr<TransmissionCurve const> a,
        std::shared_ptr<TransmissionCurve const> b
    ) : _a(std::move(a)), _b(std::move(b)) {}

    std::pair<double, double> getWavelengthBounds() const override {
        auto aWavelengthBounds = _a->getWavelengthBounds();
        auto bWavelengthBounds = _b->getWavelengthBounds();
        auto aThroughputAtBounds = _a->getThroughputAtBounds();
        auto bThroughputAtBounds = _b->getThroughputAtBounds();

        auto determineWavelengthBound = [](
            double aWavelength, double bWavelength,
            double aThroughput, double bThroughput,
            auto isFirstOuter
        ) -> double {
            // Use the outermost wavelength bound only if its throughput
            // values are not being multiplied by zeros from the operand with
            // the innermost wavelength bound.
            if (isFirstOuter(aWavelength, bWavelength)) {
                return (bThroughput == 0.0) ? bWavelength : aWavelength;
            } else {
                return (aThroughput == 0.0) ? aWavelength : bWavelength;
            }
        };

        return std::make_pair(
            determineWavelengthBound(aWavelengthBounds.first, bWavelengthBounds.first,
                                     aThroughputAtBounds.first, bThroughputAtBounds.first,
                                     std::less<double>()),
            determineWavelengthBound(aWavelengthBounds.second, bWavelengthBounds.second,
                                     aThroughputAtBounds.second, bThroughputAtBounds.second,
                                     std::greater<double>())
        );
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        auto aAtBounds = _a->getThroughputAtBounds();
        auto bAtBounds = _b->getThroughputAtBounds();
        return std::make_pair(aAtBounds.first * bAtBounds.first,
                              aAtBounds.second * bAtBounds.second);
    }

    void sampleAt(
        geom::Point2D const & position,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {
        _a->sampleAt(position, wavelengths, out);
        ndarray::Array<double,1,1> tmp = ndarray::allocate(wavelengths.getSize<0>());
        _b->sampleAt(position, wavelengths, tmp);
        out.deep() *= tmp;
    }

    bool isPersistable() const override { return _a->isPersistable() && _b->isPersistable(); }

protected:

    std::string getPersistenceName() const override { return NAME; }

    struct PersistenceHelper {
        table::Schema schema;
        table::Key<int> a;
        table::Key<int> b;

        static PersistenceHelper const & get() {
            static PersistenceHelper const instance;
            return instance;
        }

    private:

        PersistenceHelper() :
            schema(),
            a(schema.addField<int>("a", "archive ID of first operand")),
            b(schema.addField<int>("b", "archive ID of second operand"))
        {
            schema.getCitizen().markPersistent();
        }

    };

    class Factory : public table::io::PersistableFactory {
    public:

        std::shared_ptr<table::io::Persistable> read(
            InputArchive const& archive,
            CatalogVector const& catalogs
        ) const override {
            auto const& keys = PersistenceHelper::get();
            LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
            LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
            auto const & record = catalogs.front().front();
            return std::make_shared<ProductTransmissionCurve>(
                archive.get<TransmissionCurve>(record.get(keys.a)),
                archive.get<TransmissionCurve>(record.get(keys.b))
            );
        }

        Factory() : table::io::PersistableFactory(NAME) {}
    };

    void write(OutputArchiveHandle & handle) const override {
        auto const & keys = PersistenceHelper::get();
        auto catalog = handle.makeCatalog(keys.schema);
        auto record = catalog.addNew();
        record->set(keys.a, handle.put(_a));
        record->set(keys.b, handle.put(_b));
        handle.saveCatalog(catalog);
    }

    static Factory registration;

private:
    std::shared_ptr<TransmissionCurve const> _a;
    std::shared_ptr<TransmissionCurve const> _b;
};

ProductTransmissionCurve::Factory ProductTransmissionCurve::registration;


/*
 * TransformedTransmissionCurve: default for TransmissionCurve::transform.
 *
 * This is a another straightforward lazy-evaluation object.  Its only state
 * is the two operands it delegates to.
 */
class TransformedTransmissionCurve : public TransmissionCurve {
public:

    static constexpr char const * NAME = "TransformedTransmissionCurve";

    TransformedTransmissionCurve(
        std::shared_ptr<TransmissionCurve const> nested,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) : _nested(std::move(nested)), _transform(std::move(transform)) {}

    std::pair<double,double> getWavelengthBounds() const override {
        return _nested->getWavelengthBounds();
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        return _nested->getThroughputAtBounds();
    }

    void sampleAt(
        geom::Point2D const & position,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {
        return _nested->sampleAt(_transform->applyInverse(position), wavelengths, out);
    }

    bool isPersistable() const override { return _nested->isPersistable() && _transform->isPersistable(); }

protected:

    // transforming a TransformedTransmissionCurve composes the transforms
    std::shared_ptr<TransmissionCurve const> _transformedByImpl(
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const override {
        return std::make_shared<TransformedTransmissionCurve>(
            _nested,
            transform->then(*_transform)
        );
    }

    std::string getPersistenceName() const override { return NAME; }

    struct PersistenceHelper {
        table::Schema schema;
        table::Key<int> nested;
        table::Key<int> transform;

        static PersistenceHelper const & get() {
            static PersistenceHelper const instance;
            return instance;
        }

    private:

        PersistenceHelper() :
            schema(),
            nested(schema.addField<int>("nested", "archive ID of the nested TransmissionCurve")),
            transform(schema.addField<int>("transform", "archive ID of the coordinate transform"))
        {
            schema.getCitizen().markPersistent();
        }

    };

    void write(OutputArchiveHandle & handle) const override {
        auto const & keys = PersistenceHelper::get();
        auto catalog = handle.makeCatalog(keys.schema);
        auto record = catalog.addNew();
        record->set(keys.nested, handle.put(_nested));
        record->set(keys.transform, handle.put(_transform));
        handle.saveCatalog(catalog);
    }

    class Factory : public table::io::PersistableFactory {
    public:

        std::shared_ptr<table::io::Persistable> read(
            InputArchive const& archive,
            CatalogVector const& catalogs
        ) const override {
            auto const& keys = PersistenceHelper::get();
            LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
            LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
            auto const & record = catalogs.front().front();
            return std::make_shared<TransformedTransmissionCurve>(
                archive.get<TransmissionCurve>(record.get(keys.nested)),
                archive.get<geom::TransformPoint2ToPoint2>(record.get(keys.transform))
            );
        }

        Factory() : table::io::PersistableFactory(NAME) {}
    };

    static Factory registration;

private:
    std::shared_ptr<TransmissionCurve const> _nested;
    std::shared_ptr<geom::TransformPoint2ToPoint2> _transform;
};

TransformedTransmissionCurve::Factory TransformedTransmissionCurve::registration;

}  // namespace



/*
 * TransmissionCurve itself
 */

std::shared_ptr<TransmissionCurve const> TransmissionCurve::makeIdentity() {
    return IdentityTransmissionCurve::get();
}

std::shared_ptr<TransmissionCurve const> TransmissionCurve::makeSpatiallyConstant(
    ndarray::Array<double const,1> const & throughput,
    ndarray::Array<double const,1> const & wavelengths,
    double throughputAtMin, double throughputAtMax
) {
    ::gsl_set_error_handler_off();
    LSST_THROW_IF_NE(
        wavelengths.getSize<0>(), throughput.getSize<0>(),
        pex::exceptions::LengthError,
        "Length of wavelength array (%d) does not match size of throughput array (%d)"
    );
    return std::make_shared<InterpolatedTransmissionCurve<Impl1d>>(
        Impl1d(ndarray::copy(throughput), ndarray::copy(wavelengths)),
        std::make_pair(throughputAtMin, throughputAtMax)
    );
}

std::shared_ptr<TransmissionCurve const> TransmissionCurve::makeRadial(
    ndarray::Array<double const,2> const & throughput,
    ndarray::Array<double const,1> const & wavelengths,
    ndarray::Array<double const,1> const & radii,
    double throughputAtMin, double throughputAtMax
) {
    ::gsl_set_error_handler_off();
    LSST_THROW_IF_NE(
        wavelengths.getSize<0>(), throughput.getSize<0>(),
        pex::exceptions::LengthError,
        "Length of wavelength array (%d) does not match first dimension of of throughput array (%d)"
    );
    LSST_THROW_IF_NE(
        radii.getSize<0>(), throughput.getSize<1>(),
        pex::exceptions::LengthError,
        "Length of radii array (%d) does not match second dimension of of throughput array (%d)"
    );
    // GSL wants a column major array (Array<T,2,-2>).  But ndarray can only flatten row-major arrays
    // (Array<T,2,2>).  So we allocate a row-major array, assign the caller's throughput array to a
    // transposed view of it, and then flatten the row-major array.
    ndarray::Array<double,2,2> throughputTransposed = ndarray::allocate(throughput.getShape().reverse());
    throughputTransposed.transpose() = throughput;
    ndarray::Array<double,1,1> throughputFlat = ndarray::flatten<1>(throughputTransposed);
    return std::make_shared<InterpolatedTransmissionCurve<Impl2d>>(
        Impl2d(throughputFlat, ndarray::copy(wavelengths), ndarray::copy(radii)),
        std::make_pair(throughputAtMin, throughputAtMax)
    );
}

std::shared_ptr<TransmissionCurve const> TransmissionCurve::multipliedBy(
    TransmissionCurve const & other
) const {
    auto a = shared_from_this();
    auto b = other.shared_from_this();
    auto result = a->_multipliedByImpl(b);
    if (result == nullptr) {
        result = b->_multipliedByImpl(a);
    }
    if (result == nullptr) {
        result = std::make_shared<ProductTransmissionCurve>(std::move(a), std::move(b));
    }
    return result;
}

std::shared_ptr<TransmissionCurve const> TransmissionCurve::transformedBy(
    std::shared_ptr<geom::TransformPoint2ToPoint2> transform
) const {
    return _transformedByImpl(std::move(transform));
}

ndarray::Array<double,1,1> TransmissionCurve::sampleAt(
    geom::Point2D const & position,
    ndarray::Array<double const,1,1> const & wavelengths
) const {
    ndarray::Array<double,1,1> out = ndarray::allocate(wavelengths.getSize<0>());
    sampleAt(position, wavelengths, out);
    return out;
}

std::shared_ptr<TransmissionCurve const> TransmissionCurve::_transformedByImpl(
    std::shared_ptr<geom::TransformPoint2ToPoint2> transform
) const {
    return std::make_shared<TransformedTransmissionCurve>(shared_from_this(), std::move(transform));
}

std::shared_ptr<TransmissionCurve const> TransmissionCurve::_multipliedByImpl(
    std::shared_ptr<TransmissionCurve const> other
) const {
    return nullptr;
}

std::string TransmissionCurve::getPythonModule() const {
    return "lsst.afw.image";
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
