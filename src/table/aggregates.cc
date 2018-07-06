// -*- lsst-c++ -*-
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

#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/geom/Box.h"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst {
namespace afw {
namespace table {

//============ PointKey =====================================================================================

template <typename T>
PointKey<T> PointKey<T>::addFields(Schema &schema, std::string const &name, std::string const &doc,
                                   std::string const &unit) {
    Key<T> xKey = schema.addField<T>(schema.join(name, "x"), doc, unit);
    Key<T> yKey = schema.addField<T>(schema.join(name, "y"), doc, unit);
    return PointKey<T>(xKey, yKey);
}

template <typename T>
lsst::geom::Point<T, 2> PointKey<T>::get(BaseRecord const &record) const {
    return lsst::geom::Point<T, 2>(record.get(_x), record.get(_y));
}

template <typename T>
void PointKey<T>::set(BaseRecord &record, lsst::geom::Point<T, 2> const &value) const {
    record.set(_x, value.getX());
    record.set(_y, value.getY());
}

template class PointKey<int>;
template class PointKey<double>;

//============ BoxKey =====================================================================================

template <typename Box>
BoxKey<Box> BoxKey<Box>::addFields(Schema &schema, std::string const &name, std::string const &doc,
                                   std::string const &unit) {
    auto minKey = PointKey<Element>::addFields(schema, schema.join(name, "min"), doc + " (minimum)", unit);
    auto maxKey = PointKey<Element>::addFields(schema, schema.join(name, "max"), doc + " (maximum)", unit);
    return BoxKey<Box>(minKey, maxKey);
}

template <typename Box>
Box BoxKey<Box>::get(BaseRecord const &record) const {
    return Box(record.get(_min), record.get(_max), /*invert=*/false);
}

template <typename Box>
void BoxKey<Box>::set(BaseRecord &record, Box const &value) const {
    _min.set(record, value.getMin());
    _max.set(record, value.getMax());
}

template class BoxKey<lsst::geom::Box2I>;
template class BoxKey<lsst::geom::Box2D>;

//============ CoordKey =====================================================================================

CoordKey CoordKey::addFields(Schema &schema, std::string const &name, std::string const &doc) {
    Key<lsst::geom::Angle> ra = schema.addField<lsst::geom::Angle>(schema.join(name, "ra"), doc);
    Key<lsst::geom::Angle> dec = schema.addField<lsst::geom::Angle>(schema.join(name, "dec"), doc);
    return CoordKey(ra, dec);
}

lsst::geom::SpherePoint CoordKey::get(BaseRecord const &record) const {
    return lsst::geom::SpherePoint(record.get(_ra), record.get(_dec));
}

void CoordKey::set(BaseRecord &record, lsst::geom::SpherePoint const &value) const {
    record.set(_ra, value.getLongitude());
    record.set(_dec, value.getLatitude());
}

//============ QuadrupoleKey ================================================================================

QuadrupoleKey QuadrupoleKey::addFields(Schema &schema, std::string const &name, std::string const &doc,
                                       CoordinateType coordType) {
    std::string unit = coordType == CoordinateType::PIXEL ? "pixel^2" : "rad^2";

    Key<double> xxKey = schema.addField<double>(schema.join(name, "xx"), doc, unit);
    Key<double> yyKey = schema.addField<double>(schema.join(name, "yy"), doc, unit);
    Key<double> xyKey = schema.addField<double>(schema.join(name, "xy"), doc, unit);
    return QuadrupoleKey(xxKey, yyKey, xyKey);
}

geom::ellipses::Quadrupole QuadrupoleKey::get(BaseRecord const &record) const {
    return geom::ellipses::Quadrupole(record.get(_ixx), record.get(_iyy), record.get(_ixy));
}

void QuadrupoleKey::set(BaseRecord &record, geom::ellipses::Quadrupole const &value) const {
    record.set(_ixx, value.getIxx());
    record.set(_iyy, value.getIyy());
    record.set(_ixy, value.getIxy());
}

//============ EllipseKey ================================================================================

EllipseKey EllipseKey::addFields(Schema &schema, std::string const &name, std::string const &doc,
                                 std::string const &unit) {
    QuadrupoleKey qKey = QuadrupoleKey::addFields(schema, name, doc, CoordinateType::PIXEL);
    PointKey<double> pKey = PointKey<double>::addFields(schema, name, doc, unit);
    return EllipseKey(qKey, pKey);
}

geom::ellipses::Ellipse EllipseKey::get(BaseRecord const &record) const {
    return geom::ellipses::Ellipse(record.get(_qKey), record.get(_pKey));
}

void EllipseKey::set(BaseRecord &record, geom::ellipses::Ellipse const &value) const {
    _qKey.set(record, value.getCore());
    _pKey.set(record, value.getCenter());
}

//============ CovarianceMatrixKey ==========================================================================

template <typename T, int N>
CovarianceMatrixKey<T, N> CovarianceMatrixKey<T, N>::addFields(Schema &schema, std::string const &prefix,
                                                               NameArray const &names,
                                                               std::string const &unit, bool diagonalOnly) {
    NameArray units(names.size(), unit);
    return addFields(schema, prefix, names, units, diagonalOnly);
}

template <typename T, int N>
CovarianceMatrixKey<T, N> CovarianceMatrixKey<T, N>::addFields(Schema &schema, std::string const &prefix,
                                                               NameArray const &names, NameArray const &units,
                                                               bool diagonalOnly) {
    if (N != Eigen::Dynamic) {
        LSST_THROW_IF_NE(names.size(), std::size_t(N), pex::exceptions::LengthError,
                         "Size of names array (%d) does not match template argument (%d)");
        LSST_THROW_IF_NE(units.size(), std::size_t(N), pex::exceptions::LengthError,
                         "Size of units array (%d) does not match template argument (%d)");
    }
    SigmaKeyArray sigma;
    CovarianceKeyArray cov;
    sigma.reserve(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        sigma.push_back(schema.addField<T>(schema.join(prefix, names[i] + "Sigma"),
                                           "1-sigma uncertainty on " + names[i], units[i]));
    }
    if (!diagonalOnly) {
        cov.reserve((names.size() * (names.size() - 1)) / 2);
        for (std::size_t i = 0; i < names.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                // We iterate over the lower-triangular part of the matrix in row-major order,
                // but we use the upper-triangular names (i.e. we switch the order of i and j, below).
                // That puts the elements in the order expected by the constructor we call below,
                // while creating the field names users would expect from the ordering of their name
                // vector (i.e. _a_b_Cov instead of _b_a_Cov if names=[a, b]).
                cov.push_back(schema.addField<T>(
                        schema.join(prefix, names[j], names[i], "Cov"),
                        "uncertainty covariance between " + names[j] + " and " + names[i],
                        units[j] + (units[j].empty() || units[i].empty() ? "" : " ") + units[i]));
            }
        }
    }
    return CovarianceMatrixKey<T, N>(sigma, cov);
}

template <typename T, int N>
CovarianceMatrixKey<T, N>::CovarianceMatrixKey() {}

template <typename T, int N>
CovarianceMatrixKey<T, N>::CovarianceMatrixKey(SigmaKeyArray const &sigma, CovarianceKeyArray const &cov)
        : _sigma(sigma), _cov(cov) {
    if (N != Eigen::Dynamic) {
        LSST_THROW_IF_NE(sigma.size(), std::size_t(N), pex::exceptions::LengthError,
                         "Size of sigma array (%d) does not match template argument (%d)");
    }
    if (!cov.empty()) {
        LSST_THROW_IF_NE(cov.size(), sigma.size() * (sigma.size() - 1) / 2, pex::exceptions::LengthError,
                         "Size of cov array (%d) is does not match with size inferred from sigma array (%d)");
        bool haveCov = false;
        for (typename CovarianceKeyArray::const_iterator i = _cov.begin(); i != _cov.end(); ++i) {
            if (i->isValid()) haveCov = true;
        }
        if (!haveCov) _cov.resize(0);
    }
}

template <typename T, int N>
CovarianceMatrixKey<T, N>::CovarianceMatrixKey(SubSchema const &s, NameArray const &names)
        : _sigma(names.size()), _cov(names.size() * (names.size() - 1) / 2) {
    int const n = names.size();
    int k = 0;
    bool haveCov = false;
    for (int i = 0; i < n; ++i) {
        _sigma[i] = s[names[i] + "Sigma"];
        for (int j = 0; j < i; ++j, ++k) {
            try {
                _cov[k] = s[names[i] + "_" + names[j] + "_Cov"];
                haveCov = true;
            } catch (pex::exceptions::NotFoundError &) {
                try {
                    _cov[k] = s[names[j] + "_" + names[i] + "_Cov"];
                    haveCov = true;
                } catch (pex::exceptions::NotFoundError &) {
                }
            }
        }
    }
    if (!haveCov) _cov.resize(0);
}

template <typename T, int N>
CovarianceMatrixKey<T, N>::CovarianceMatrixKey(CovarianceMatrixKey const &) = default;
template <typename T, int N>
CovarianceMatrixKey<T, N>::CovarianceMatrixKey(CovarianceMatrixKey &&) = default;
template <typename T, int N>
CovarianceMatrixKey<T, N> &CovarianceMatrixKey<T, N>::operator=(CovarianceMatrixKey const &) = default;
template <typename T, int N>
CovarianceMatrixKey<T, N> &CovarianceMatrixKey<T, N>::operator=(CovarianceMatrixKey &&) = default;
template <typename T, int N>
CovarianceMatrixKey<T, N>::~CovarianceMatrixKey() noexcept = default;

// these are workarounds for the fact that Eigen has different constructors for
// dynamic-sized matrices and fixed-size matrices, but we don't want to have to
// partial-specialize the entire template just to change one line
namespace {

template <typename T, int N>
Eigen::Matrix<T, N, N> makeZeroMatrix(int n, CovarianceMatrixKey<T, N> const *) {
    return Eigen::Matrix<T, N, N>::Zero();
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> makeZeroMatrix(
        int n, CovarianceMatrixKey<T, Eigen::Dynamic> const *) {
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
}

}  // namespace

template <typename T, int N>
Eigen::Matrix<T, N, N> CovarianceMatrixKey<T, N>::get(BaseRecord const &record) const {
    Eigen::Matrix<T, N, N> value = makeZeroMatrix(_sigma.size(), this);
    int const n = _sigma.size();
    int k = 0;
    for (int i = 0; i < n; ++i) {
        T sigma = record.get(_sigma[i]);
        value(i, i) = sigma * sigma;
        if (!_cov.empty()) {
            for (int j = 0; j < i; ++j, ++k) {
                if (_cov[k].isValid()) {
                    value(i, j) = value(j, i) = record.get(_cov[k]);
                }
            }
        }
    }
    return value;
}

template <typename T, int N>
void CovarianceMatrixKey<T, N>::set(BaseRecord &record, Eigen::Matrix<T, N, N> const &value) const {
    int const n = _sigma.size();
    int k = 0;
    for (int i = 0; i < n; ++i) {
        record.set(_sigma[i], std::sqrt(value(i, i)));
        if (!_cov.empty()) {
            for (int j = 0; j < i; ++j, ++k) {
                if (_cov[k].isValid()) {
                    record.set(_cov[k], value(i, j));
                }
            }
        }
    }
}

template <typename T, int N>
bool CovarianceMatrixKey<T, N>::isValid() const noexcept {
    int const n = _sigma.size();
    if (n < 1) return false;
    for (int i = 0; i < n; ++i) {
        if (!_sigma[i].isValid()) return false;
    }
    return true;
}

template <typename T, int N>
bool CovarianceMatrixKey<T, N>::operator==(CovarianceMatrixKey const &other) const noexcept {
    if (_sigma.size() != other._sigma.size()) {
        return false;
    }
    if (_cov.size() != other._cov.size()) {
        return false;
    }
    int const n = _sigma.size();
    int k = 0;
    for (int i = 0; i < n; ++i) {
        if (_sigma[i] != other._sigma[i]) {
            return false;
        }
        if (!_cov.empty()) {
            for (int j = 0; j < i; ++j, ++k) {
                if (_cov[k] != other._cov[k]) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T, int N>
T CovarianceMatrixKey<T, N>::getElement(BaseRecord const &record, int i, int j) const {
    if (i == j) {
        T sigma = record.get(_sigma[i]);
        return sigma * sigma;
    }
    if (_cov.empty()) {
        return 0.0;
    }
    Key<T> key = (i < j) ? _cov[j * (j - 1) / 2 + i] : _cov[i * (i - 1) / 2 + j];
    return key.isValid() ? record.get(key) : 0.0;
}

template <typename T, int N>
void CovarianceMatrixKey<T, N>::setElement(BaseRecord &record, int i, int j, T value) const {
    if (i == j) {
        record.set(_sigma[i], std::sqrt(value));
    } else {
        if (_cov.empty()) {
            throw LSST_EXCEPT(
                    pex::exceptions::LogicError,
                    (boost::format("Cannot set covariance element %d,%d; no fields for covariance") % i % j)
                            .str());
        }
        Key<T> key = (i < j) ? _cov[j * (j - 1) / 2 + i] : _cov[i * (i - 1) / 2 + j];
        if (!key.isValid()) {
            throw LSST_EXCEPT(
                    pex::exceptions::LogicError,
                    (boost::format("Cannot set covariance element %d,%d; no field for this element") % i % j)
                            .str());
        }
        record.set(key, value);
    }
}

template class CovarianceMatrixKey<float, 2>;
template class CovarianceMatrixKey<float, 3>;
template class CovarianceMatrixKey<float, 4>;
template class CovarianceMatrixKey<float, 5>;
template class CovarianceMatrixKey<float, Eigen::Dynamic>;
template class CovarianceMatrixKey<double, 2>;
template class CovarianceMatrixKey<double, 3>;
template class CovarianceMatrixKey<double, 4>;
template class CovarianceMatrixKey<double, 5>;
template class CovarianceMatrixKey<double, Eigen::Dynamic>;
}  // namespace table
}  // namespace afw
}  // namespace lsst
