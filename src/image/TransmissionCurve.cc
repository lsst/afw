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

#define LSST_CHECK_GSL(type, status) \
    if (status) throw LSST_EXCEPT(type, std::string("GSL error: ") + ::gsl_strerror(status))


namespace lsst {
namespace afw {
namespace image {

namespace {

template <typename T>
using CPtr = std::unique_ptr<T, void (*)(T*)>;


double computeNaturalSamplingSize(ndarray::Array<double,1,1> const & wavelengths) {
    double last = -1.0;
    double minSpacing = -1.0;
    for (auto wl : wavelengths) {
        if (last < 0) {
            last = wl;
        } else {
            double spacing = wl - last;
            if (spacing < 0) {
                throw LSST_EXCEPT(
                    pex::exceptions::InvalidParameterError,
                    "Input array of wavelengths is not monotonically increasing."
                );
            }
            if (minSpacing < 0 || spacing < minSpacing) {
                minSpacing = spacing;
            }
            last = wl;
        }
    }
    // We round instead of (conservatively) using ceil to correctly handle the case where the given
    // wavelength sampling *is* even, but round-off error makes minSpacing *slightly* too small
    // for exact division.
    return 1 + std::round((wavelengths.back() - wavelengths.front())/minSpacing);
}


class IdentityTransmissionCurve : public TransmissionCurve {
public:

    // IdentityTransmissionCurve is a singleton.
    static std::shared_ptr<IdentityTransmissionCurve> get() {
        static std::shared_ptr<IdentityTransmissionCurve> instance(new IdentityTransmissionCurve());
        return instance;
    }

    SampleDef getNaturalSampling() const override {
        return SampleDef();
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
    std::shared_ptr<TransmissionCurve> _transformImpl(
        std::shared_ptr<TransmissionCurve> self,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const override {
        return self;
    }

    // multiplying an IdentityTransmissionCurve always yields the other operand
    std::shared_ptr<TransmissionCurve> _multiplyImpl(
        std::shared_ptr<TransmissionCurve> self,
        std::shared_ptr<TransmissionCurve> other
    ) const {
        return other;
    }

    std::string getPersistenceName() const override { return "IdentityTransmissionCurve"; }

    std::string getPythonModule() const override { return "lsst.afw.image"; }

    void write(OutputArchiveHandle& handle) const override;

private:

    IdentityTransmissionCurve() {}

};

void IdentityTransmissionCurve::write(OutputArchiveHandle & handle) const {
    handle.saveEmpty();
}

class IdentityFactory : public table::io::PersistableFactory {
public:

    std::shared_ptr<table::io::Persistable> read(
        InputArchive const& archive,
        CatalogVector const& catalogs
    ) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 0u);
        return IdentityTransmissionCurve::get();
    }

    IdentityFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

IdentityFactory identityRegistration("IdentityTransmissionCurve");


class ConstantTransmissionCurve : public TransmissionCurve {
public:

    ConstantTransmissionCurve(
        ndarray::Array<double const,1> const & throughput,
        ndarray::Array<double const,1> const & wavelengths,
        double throughputAtMin, double throughputAtMax
    ) :
        _atBounds(throughputAtMin, throughputAtMax),
        _throughput(ndarray::copy(throughput)),
        _wavelengths(ndarray::copy(wavelengths)),
        _naturalSamplingSize(computeNaturalSamplingSize(_wavelengths)),
        _interp(::gsl_interp_alloc(::gsl_interp_linear, _throughput.getSize<0>()), &::gsl_interp_free)
    {
        LSST_THROW_IF_NE(
            wavelengths.getSize<0>(), throughput.getSize<0>(),
            pex::exceptions::LengthError,
            "Length of wavelength array (%d) does not match size of throughput array (%d)"
        );
        int status = ::gsl_interp_init(_interp.get(), _wavelengths.getData(), _throughput.getData(),
                                       _throughput.getSize<0>());
        LSST_CHECK_GSL(pex::exceptions::LogicError, status);
    }

    SampleDef getNaturalSampling() const override {
        return SampleDef(_wavelengths.front(), _wavelengths.back(), _naturalSamplingSize);
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        return _atBounds;
    }

    void sampleAt(
        geom::Point2D const &,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {
        LSST_THROW_IF_NE(
            wavelengths.getSize<0>(), out.getSize<0>(),
            pex::exceptions::LengthError,
            "Length of wavelength array (%d) does not match size of output array (%d)"
        );
        CPtr<::gsl_interp_accel> accel(::gsl_interp_accel_alloc(), &gsl_interp_accel_free);
        auto wlIter = wavelengths.begin();
        for (auto outIter = out.begin(); outIter != out.end(); ++outIter, ++wlIter) {
            double & y = *outIter;
            if (*wlIter < _wavelengths.front()) {
                y = _atBounds.first;
            } else if (*wlIter > _wavelengths.back()) {
                y = _atBounds.second;
            } else {
                int status = ::gsl_interp_eval_e(_interp.get(), _wavelengths.getData(), _throughput.getData(),
                                                 *wlIter, accel.get(), &y);
                if (status) {
                    throw LSST_EXCEPT(
                        pex::exceptions::InvalidParameterError,
                        (boost::format("Invalid wavelength value: %g") % (*wlIter)).str()
                    );
                }
            }
        }
    }

    bool isPersistable() const override { return true; }

protected:

    // transforming a ConstantTransmissionCurve is a no-op
    std::shared_ptr<TransmissionCurve> _transformImpl(
        std::shared_ptr<TransmissionCurve> self,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const override {
        return self;
    }

    std::string getPersistenceName() const override { return "ConstantTransmissionCurve"; }

    std::string getPythonModule() const override { return "lsst.afw.image"; }

    void write(OutputArchiveHandle& handle) const override;

private:
    std::pair<double,double> _atBounds;
    ndarray::Array<double,1,1> _throughput;
    ndarray::Array<double,1,1> _wavelengths;
    int _naturalSamplingSize;
    CPtr<::gsl_interp> _interp;
};

struct ConstantPersistenceHelper {
    table::Schema schema;
    table::Key<double> throughputAtMin;
    table::Key<double> throughputAtMax;
    table::Key<table::Array<double>> throughput;
    table::Key<table::Array<double>> wavelengths;

    static ConstantPersistenceHelper const & get() {
        static ConstantPersistenceHelper const instance;
        return instance;
    }

private:

    ConstantPersistenceHelper() :
        schema(),
        throughputAtMin(schema.addField<double>(
            "throughputAtMin", "throughput below minimum wavelength")),
        throughputAtMax(schema.addField<double>(
            "throughputAtMax", "throughput above minimum wavelength")),
        throughput(schema.addField<table::Array<double>>(
            "throughput", "array of known throughput values", "", 0)),
        wavelengths(schema.addField<table::Array<double>>(
            "wavelengths", "array of wavelengths corresponding to known throughputs", "angstrom", 0))
    {
        schema.getCitizen().markPersistent();
    }

};

void ConstantTransmissionCurve::write(OutputArchiveHandle & handle) const {
    auto const & keys = ConstantPersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    auto record = catalog.addNew();
    record->set(keys.throughputAtMin, _atBounds.first);
    record->set(keys.throughputAtMax, _atBounds.second);
    record->set(keys.throughput, _throughput);
    record->set(keys.wavelengths, _wavelengths);
    handle.saveCatalog(catalog);
}

class ConstantFactory : public table::io::PersistableFactory {
public:

    std::shared_ptr<table::io::Persistable> read(
        InputArchive const& archive,
        CatalogVector const& catalogs
    ) const override {
        auto const& keys = ConstantPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        auto const & record = catalogs.front().front();
        return std::make_shared<ConstantTransmissionCurve>(
            record.get(keys.throughput),
            record.get(keys.wavelengths),
            record.get(keys.throughputAtMin), record.get(keys.throughputAtMax)
        );
    }

    ConstantFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

ConstantFactory constantRegistration("ConstantTransmissionCurve");


class RadialTransmissionCurve : public TransmissionCurve {
public:

    RadialTransmissionCurve(
        ndarray::Array<double const,2> const & throughput,
        ndarray::Array<double const,1> const & wavelengths,
        ndarray::Array<double const,1> const & radii,
        double throughputAtMin, double throughputAtMax
    ) :
        _atBounds(throughputAtMin, throughputAtMax),
        _throughput(ndarray::allocate(throughput.getShape())),
        _wavelengths(ndarray::copy(wavelengths)),
        _radii(ndarray::copy(radii)),
        _naturalSamplingSize(computeNaturalSamplingSize(_wavelengths)),
        _interp(
            ::gsl_interp2d_alloc(::gsl_interp2d_bilinear, _throughput.getSize<0>(), _throughput.getSize<1>()),
            &::gsl_interp2d_free
        )
    {
        // We can't use ndarray::copy() above because GSL needs a column-major array (contradicting their
        // docs, which claim that it wants a row-major array).
        _throughput.deep() = throughput;
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
        int status = ::gsl_interp2d_init(_interp.get(), _wavelengths.getData(), _radii.getData(),
                                         _throughput.getData(),
                                         _throughput.getSize<0>(), _throughput.getSize<1>());
        LSST_CHECK_GSL(pex::exceptions::LogicError, status);
    }

    SampleDef getNaturalSampling() const override {
        return SampleDef(_wavelengths.front(), _wavelengths.back(), _naturalSamplingSize);
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
        CPtr<::gsl_interp_accel> xAccel(::gsl_interp_accel_alloc(), &gsl_interp_accel_free);
        CPtr<::gsl_interp_accel> yAccel(::gsl_interp_accel_alloc(), &gsl_interp_accel_free);
        auto wlIter = wavelengths.begin();
        // transform the point to a radius, and limit to min/max values.
        double r = point.asEigen().norm();
        r = std::max(r, _radii.front());
        r = std::min(r, _radii.back());
        // iterate over wavelength values and interpolate in 2-d
        for (auto outIter = out.begin(); outIter != out.end(); ++outIter, ++wlIter) {
            double & z = *outIter;
            if (*wlIter < _wavelengths.front()) {
                z = _atBounds.first;
            } else if (*wlIter > _wavelengths.back()) {
                z = _atBounds.second;
            } else {
                int status = ::gsl_interp2d_eval_e(_interp.get(), _wavelengths.getData(), _radii.getData(),
                                                   _throughput.getData(),
                                                   *wlIter, r, xAccel.get(), yAccel.get(), &z);
                if (status) {
                    throw LSST_EXCEPT(
                        pex::exceptions::InvalidParameterError,
                        (boost::format("Invalid wavelength value: %g") % (*wlIter)).str()
                    );
                }
            }
        }
    }

    bool isPersistable() const override { return true; }

protected:

    std::string getPersistenceName() const override { return "RadialTransmissionCurve"; }

    std::string getPythonModule() const override { return "lsst.afw.image"; }

    void write(OutputArchiveHandle& handle) const override;

private:
    std::pair<double,double> _atBounds;
    ndarray::Array<double,2,-2> _throughput;
    ndarray::Array<double,1,1> _wavelengths;
    ndarray::Array<double,1,1> _radii;
    int _naturalSamplingSize;
    CPtr<::gsl_interp2d> _interp;
};


struct RadialPersistenceHelper {
    table::Schema schema;
    table::Key<double> throughputAtMin;
    table::Key<double> throughputAtMax;
    table::Key<table::Array<double>> throughput;
    table::Key<table::Array<double>> wavelengths;
    table::Key<table::Array<double>> radii;

    static RadialPersistenceHelper const & get() {
        static RadialPersistenceHelper const instance;
        return instance;
    }

private:

    RadialPersistenceHelper() :
        schema(),
        throughputAtMin(schema.addField<double>(
            "throughputAtMin", "throughput below minimum wavelength")),
        throughputAtMax(schema.addField<double>(
            "throughputAtMax", "throughput above minimum wavelength")),
        throughput(schema.addField<table::Array<double>>(
            "throughput", "array of known throughput values", "", 0)),
        wavelengths(schema.addField<table::Array<double>>(
            "wavelengths", "array of wavelengths corresponding to known throughputs", "angstrom", 0)),
        radii(schema.addField<table::Array<double>>(
            "radii", "array of radii corresponding to known throughputs", "", 0))
    {
        schema.getCitizen().markPersistent();
    }

};

void RadialTransmissionCurve::write(OutputArchiveHandle & handle) const {
    auto const & keys = RadialPersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    auto record = catalog.addNew();
    record->set(keys.throughputAtMin, _atBounds.first);
    record->set(keys.throughputAtMax, _atBounds.second);
    ndarray::Array<double,1,1> flat = ndarray::flatten<1>(_throughput.transpose());
    record->set(keys.throughput, flat);
    record->set(keys.wavelengths, _wavelengths);
    record->set(keys.radii, _radii);
    handle.saveCatalog(catalog);
}

class RadialFactory : public table::io::PersistableFactory {
public:

    std::shared_ptr<table::io::Persistable> read(
        InputArchive const& archive,
        CatalogVector const& catalogs
    ) const override {
        auto const& keys = RadialPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        auto const & record = catalogs.front().front();
        auto wavelengths = record.get(keys.wavelengths);
        auto radii = record.get(keys.radii);
        ndarray::Array<double,2,-2> throughputs = ndarray::allocate(
            wavelengths.getSize<0>(),
            radii.getSize<0>()
        );
        ndarray::flatten<1>(throughputs.transpose()) = record.get(keys.throughput);
        return std::make_shared<RadialTransmissionCurve>(
            throughputs,
            wavelengths,
            radii,
            record.get(keys.throughputAtMin), record.get(keys.throughputAtMax)
        );
    }

    RadialFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

RadialFactory radialRegistration("RadialTransmissionCurve");


class ProductTransmissionCurve : public TransmissionCurve {
public:

    ProductTransmissionCurve(
        std::shared_ptr<TransmissionCurve> a,
        std::shared_ptr<TransmissionCurve> b
    ) : _a(std::move(a)), _b(std::move(b)) {}

    SampleDef getNaturalSampling() const override {
        auto aSampling = _a->getNaturalSampling();
        auto bSampling = _b->getNaturalSampling();
        auto aAtBounds = _a->getThroughputAtBounds();
        auto bAtBounds = _b->getThroughputAtBounds();
        double min, max;
        // If an operand goes to zero or NaN at its min [max] wavelength,
        // any values from the other operand below [above] that wavelength
        // are meaningless, and we want the new min [max] wavelength to
        // be the greater [lesser] of the mins [maxes] of the two operands
        // to exclude that region.
        if (aSampling.min < bSampling.min) {
            if (bAtBounds.first > 0.0) { // not zero *or* NaN
                min = aSampling.min;
            } else {
                min = bSampling.min;
            }
        } else {
            if (aAtBounds.first > 0.0) {
                min = bSampling.min;
            } else {
                min = aSampling.min;
            }
        }
        if (aSampling.max > bSampling.max) {
            if (bAtBounds.second > 0.0) { // not zero *or* NaN
                max = aSampling.max;
            } else {
                max = bSampling.max;
            }
        } else {
            if (aAtBounds.second > 0.0) {
                max = bSampling.max;
            } else {
                max = aSampling.max;
            }
        }
        double spacing = std::min(aSampling.getSpacing(), bSampling.getSpacing());
        int size = 1 + std::ceil(max - min)/spacing;
        return SampleDef(min, max, size);
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        auto aAtBounds = _a->getThroughputAtBounds();
        auto bAtBounds = _b->getThroughputAtBounds();
        auto func = [](double a, double b) -> double { // multiply, but let 0*NaN = 0
            if (a == 0.0 || b == 0.0) {
                return 0.0;
            }
            return a*b;
        };
        return std::make_pair(func(aAtBounds.first, bAtBounds.first),
                              func(aAtBounds.second, bAtBounds.second));
    }

    void sampleAt(
        geom::Point2D const & position,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {

        _a->sampleAt(position, wavelengths, out);
        ndarray::Array<double,1,1> tmp = ndarray::allocate(wavelengths.getSize<0>());
        _b->sampleAt(position, wavelengths, tmp);
        for (auto outIter = out.begin(), tmpIter = tmp.begin(); outIter != out.end(); ++outIter, ++tmpIter) {
            if (*tmpIter == 0.0) {
                *outIter = 0.0;
            } else if (*outIter != 0.0) {
                *outIter *= *tmpIter;
            }
        }
    }

    bool isPersistable() const override { return _a->isPersistable() && _b->isPersistable(); }

protected:

    std::string getPersistenceName() const override { return "ProductTransmissionCurve"; }

    std::string getPythonModule() const override { return "lsst.afw.image"; }

    void write(OutputArchiveHandle& handle) const override;

private:
    std::shared_ptr<TransmissionCurve> _a;
    std::shared_ptr<TransmissionCurve> _b;
};

struct ProductPersistenceHelper {
    table::Schema schema;
    table::Key<int> a;
    table::Key<int> b;

    static ProductPersistenceHelper const & get() {
        static ProductPersistenceHelper const instance;
        return instance;
    }

private:

    ProductPersistenceHelper() :
        schema(),
        a(schema.addField<int>("a", "archive ID of first operand")),
        b(schema.addField<int>("b", "archive ID of second operand"))
    {
        schema.getCitizen().markPersistent();
    }

};

void ProductTransmissionCurve::write(OutputArchiveHandle & handle) const {
    auto const & keys = ProductPersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    auto record = catalog.addNew();
    record->set(keys.a, handle.put(_a));
    record->set(keys.b, handle.put(_b));
    handle.saveCatalog(catalog);
}

class ProductFactory : public table::io::PersistableFactory {
public:

    std::shared_ptr<table::io::Persistable> read(
        InputArchive const& archive,
        CatalogVector const& catalogs
    ) const override {
        auto const& keys = ProductPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        auto const & record = catalogs.front().front();
        return std::make_shared<ProductTransmissionCurve>(
            archive.get<TransmissionCurve>(record.get(keys.a)),
            archive.get<TransmissionCurve>(record.get(keys.b))
        );
    }

    ProductFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

ProductFactory productRegistration("ProductTransmissionCurve");


class TransformedTransmissionCurve : public TransmissionCurve {
public:

    TransformedTransmissionCurve(
        std::shared_ptr<TransmissionCurve> nested,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) : _nested(std::move(nested)), _transform(std::move(transform)) {}

    SampleDef getNaturalSampling() const override {
        return _nested->getNaturalSampling();
    }

    std::pair<double,double> getThroughputAtBounds() const override {
        return _nested->getThroughputAtBounds();
    }

    void sampleAt(
        geom::Point2D const & position,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const override {
        return _nested->sampleAt(_transform->applyForward(position), wavelengths, out);
    }

    bool isPersistable() const override { return _nested->isPersistable() && _transform->isPersistable(); }

protected:

    // transforming a TransformedTransmissionCurve composes the transforms
    std::shared_ptr<TransmissionCurve> _transformImpl(
        std::shared_ptr<TransmissionCurve>,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const override {
        return std::make_shared<TransformedTransmissionCurve>(
            _nested,
            transform->then(*_transform)
        );
    }

    std::string getPersistenceName() const override { return "TransformedTransmissionCurve"; }

    std::string getPythonModule() const override { return "lsst.afw.image"; }

    void write(OutputArchiveHandle& handle) const override;

private:
    std::shared_ptr<TransmissionCurve> _nested;
    std::shared_ptr<geom::TransformPoint2ToPoint2> _transform;
};

struct TransformedPersistenceHelper {
    table::Schema schema;
    table::Key<int> nested;
    table::Key<int> transform;

    static TransformedPersistenceHelper const & get() {
        static TransformedPersistenceHelper const instance;
        return instance;
    }

private:

    TransformedPersistenceHelper() :
        schema(),
        nested(schema.addField<int>("nested", "archive ID of the nested TransmissionCurve")),
        transform(schema.addField<int>("transform", "archive ID of the coordinate transform"))
    {
        schema.getCitizen().markPersistent();
    }

};

void TransformedTransmissionCurve::write(OutputArchiveHandle & handle) const {
    auto const & keys = TransformedPersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    auto record = catalog.addNew();
    record->set(keys.nested, handle.put(_nested));
    record->set(keys.transform, handle.put(_transform));
    handle.saveCatalog(catalog);
}

class TransformedFactory : public table::io::PersistableFactory {
public:

    std::shared_ptr<table::io::Persistable> read(
        InputArchive const& archive,
        CatalogVector const& catalogs
    ) const override {
        auto const& keys = TransformedPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        auto const & record = catalogs.front().front();
        return std::make_shared<TransformedTransmissionCurve>(
            archive.get<TransmissionCurve>(record.get(keys.nested)),
            archive.get<geom::TransformPoint2ToPoint2>(record.get(keys.transform))
        );
    }

    TransformedFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

TransformedFactory transformedRegistration("TransformedTransmissionCurve");


} // anonymous


TransmissionCurve::SampleDef::SampleDef() :
    min(-std::numeric_limits<double>::infinity()),
    max(std::numeric_limits<double>::infinity()),
    size(0)
{}

ndarray::Array<double,1,1> TransmissionCurve::SampleDef::makeArray() const {
    ndarray::Array<double,1,1> result = ndarray::allocate(this->size);
    double const step = (this->max - this->min)/(this->size - 1);
    double current = this->min;
    for (auto & element : result) {
        element = current;
        current += step;
    }
    result.back() = this->max; // make sure round-off error doesn't push us too far
    return result;
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::makeIdentity() {
    return IdentityTransmissionCurve::get();
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::makeConstant(
    ndarray::Array<double const,1> const & throughput,
    ndarray::Array<double const,1> const & wavelengths,
    double throughputAtMin, double throughputAtMax
) {
    ::gsl_set_error_handler_off();
    return std::make_shared<ConstantTransmissionCurve>(throughput, wavelengths,
                                                       throughputAtMin, throughputAtMax);
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::makeRadial(
    ndarray::Array<double const,2> const & throughput,
    ndarray::Array<double const,1> const & wavelengths,
    ndarray::Array<double const,1> const & radii,
    double throughputAtMin, double throughputAtMax
) {
    ::gsl_set_error_handler_off();
    return std::make_shared<RadialTransmissionCurve>(throughput, wavelengths, radii,
                                                     throughputAtMin, throughputAtMax);
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::multiply(
    std::shared_ptr<TransmissionCurve> a,
    std::shared_ptr<TransmissionCurve> b
) {
    auto result = a->_multiplyImpl(a, b);
    if (result == nullptr) {
        result = b->_multiplyImpl(b, a);
    }
    if (result == nullptr) {
        result = std::make_shared<ProductTransmissionCurve>(std::move(a), std::move(b));
    }
    return result;
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::transform(
    std::shared_ptr<TransmissionCurve> base,
    std::shared_ptr<geom::TransformPoint2ToPoint2> transform
) {
    return base->_transformImpl(base, std::move(transform));
}

void TransmissionCurve::sampleAt(
    geom::Point2D const & position,
    SampleDef const & wavelengths,
    ndarray::Array<double,1,1> const & out
) const {
    return sampleAt(position, wavelengths.makeArray(), out);
}

ndarray::Array<double,1,1> TransmissionCurve::sampleAt(
    geom::Point2D const & position,
    ndarray::Array<double const,1,1> const & wavelengths
) const {
    ndarray::Array<double,1,1> out = ndarray::allocate(wavelengths.getSize<0>());
    sampleAt(position, wavelengths, out);
    return out;
}

ndarray::Array<double,1,1> TransmissionCurve::sampleAt(
    geom::Point2D const & position,
    SampleDef const & wavelengths
) const {
    return sampleAt(position, wavelengths.makeArray());
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::_transformImpl(
    std::shared_ptr<TransmissionCurve> self,
    std::shared_ptr<geom::TransformPoint2ToPoint2> transform
) const {
    return std::make_shared<TransformedTransmissionCurve>(self, std::move(transform));
}

std::shared_ptr<TransmissionCurve> TransmissionCurve::_multiplyImpl(
    std::shared_ptr<TransmissionCurve> self,
    std::shared_ptr<TransmissionCurve> other
) const {
    return nullptr;
}

}}} // lsst::afw::image
