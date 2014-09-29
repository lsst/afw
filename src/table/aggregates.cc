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
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table {

//============ PointKey =====================================================================================

template <typename T>
PointKey<T> PointKey<T>::addFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc,
    std::string const & unit
) {
    Key<T> xKey = schema.addField<T>(schema.join(name, "x"), doc, unit);
    Key<T> yKey = schema.addField<T>(schema.join(name, "y"), doc, unit);
    return PointKey<T>(xKey, yKey);
}

template <typename T>
geom::Point<T,2> PointKey<T>::get(BaseRecord const & record) const {
    return geom::Point<T,2>(record.get(_x), record.get(_y));
}

template <typename T>
void PointKey<T>::set(BaseRecord & record, geom::Point<T,2> const & value) const {
    record.set(_x, value.getX());
    record.set(_y, value.getY());
}

template class PointKey<int>;
template class PointKey<double>;

//============ QuadrupoleKey ================================================================================

QuadrupoleKey QuadrupoleKey::addFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc,
    std::string const & unit
) {
    Key<double> xxKey = schema.addField<double>(schema.join(name, "xx"), doc, unit);
    Key<double> yyKey = schema.addField<double>(schema.join(name, "yy"), doc, unit);
    Key<double> xyKey = schema.addField<double>(schema.join(name, "xy"), doc, unit);
    return QuadrupoleKey(xxKey, yyKey, xyKey);
}

geom::ellipses::Quadrupole QuadrupoleKey::get(BaseRecord const & record) const {
    return geom::ellipses::Quadrupole(record.get(_ixx), record.get(_iyy), record.get(_ixy));
}

void QuadrupoleKey::set(BaseRecord & record, geom::ellipses::Quadrupole const & value) const {
    record.set(_ixx, value.getIxx());
    record.set(_iyy, value.getIyy());
    record.set(_ixy, value.getIxy());
}

//============ CovarianceMatrixKey ==========================================================================

template <typename T, int N>
CovarianceMatrixKey<T,N>::CovarianceMatrixKey() : _isDiagonalVariance(false) {}

template <typename T, int N>
CovarianceMatrixKey<T,N>::CovarianceMatrixKey(
    SigmaKeyArray const & sigma,
    CovarianceKeyArray const & cov
) : _isDiagonalVariance(false), _sigma(sigma), _cov(cov)
{
    if (N != Eigen::Dynamic) {
        LSST_THROW_IF_NE(
            sigma.size(), std::size_t(N),
            pex::exceptions::LengthError,
            "Size of sigma array (%d) does not match template argument (%d)"
        );
    }
    if (!cov.empty()) {
        LSST_THROW_IF_NE(
            cov.size(), sigma.size()*(sigma.size() - 1)/2,
            pex::exceptions::LengthError,
            "Size of cov array (%d) is does not match with size inferred from sigma array (%d)"
        );
        bool haveCov = false;
        for (typename CovarianceKeyArray::const_iterator i = _cov.begin(); i != _cov.end(); ++i) {
            if (i->isValid()) haveCov = true;
        }
        if (!haveCov) _cov.resize(0);
    }
}

template <typename T, int N>
template <typename U>
CovarianceMatrixKey<T,N>::CovarianceMatrixKey(
    Key< Covariance<U> > const & other
) : _isDiagonalVariance(true), _sigma(other.getSize()), _cov(other.getSize() * (other.getSize() - 1)/2)
{
    int const n = _sigma.size();
    int k = 0;
    for (int i = 0; i < n; ++i) {
        _sigma[i] = other(i, i);
        for (int j = 0; j < i; ++j, ++k) {
            _cov[k] = other(i, j);
        }
    }
}

template <typename T, int N>
CovarianceMatrixKey<T,N>::CovarianceMatrixKey(SubSchema const & s, NameArray const & names) :
    _isDiagonalVariance(false), _sigma(names.size()), _cov(names.size()*(names.size() - 1)/2)
{
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
                } catch (pex::exceptions::NotFoundError &) {}
            }
        }
    }
    if (!haveCov) _cov.resize(0);
}

// these are workarounds for the fact that Eigen has different constructors for
// dynamic-sized matrices and fixed-size matrices, but we don't want to have to
// partial-specialize the entire template just to change one line
namespace {

template <typename T, int N>
Eigen::Matrix<T,N,N> makeZeroMatrix(
    int n, CovarianceMatrixKey<T,N> const *
) {
    return Eigen::Matrix<T,N,N>::Zero();
}

template <typename T>
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> makeZeroMatrix(
    int n, CovarianceMatrixKey<T,Eigen::Dynamic> const *
) {
    return Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Zero(n, n);
}

} // anonymous

template <typename T, int N>
Eigen::Matrix<T,N,N> CovarianceMatrixKey<T,N>::get(BaseRecord const & record) const {
    Eigen::Matrix<T,N,N> value = makeZeroMatrix(_sigma.size(), this);
    int const n = _sigma.size();
    int k = 0;
    for (int i = 0; i < n; ++i) {
        T sigma = record.get(_sigma[i]);
        value(i, i) = (_isDiagonalVariance) ? sigma : sigma*sigma;
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
void CovarianceMatrixKey<T,N>::set(BaseRecord & record, Eigen::Matrix<T,N,N> const & value) const {
    int const n = _sigma.size();
    int k = 0;
    for (int i = 0; i < n; ++i) {
        record.set(_sigma[i], (_isDiagonalVariance) ? value(i, i) : std::sqrt(value(i, i)));
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
bool CovarianceMatrixKey<T,N>::isValid() const {
    int const n = _sigma.size();
    if (n < 1) return false;
    for (int i = 0; i < n; ++i) {
        if (!_sigma[i].isValid()) return false;
    }
    return true;
}

template <typename T, int N>
bool CovarianceMatrixKey<T,N>::operator==(CovarianceMatrixKey const & other) const {
    if (_sigma.size() != other._sigma.size()) {
        return false;
    }
    if (_cov.size() != other._cov.size()) {
        return false;
    }
    if (_isDiagonalVariance != other._isDiagonalVariance) {
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

template class CovarianceMatrixKey<float,2>;
template class CovarianceMatrixKey<float,3>;
template class CovarianceMatrixKey<float,4>;
template class CovarianceMatrixKey<float,Eigen::Dynamic>;
template class CovarianceMatrixKey<double,2>;
template class CovarianceMatrixKey<double,3>;
template class CovarianceMatrixKey<double,4>;
template class CovarianceMatrixKey<double,Eigen::Dynamic>;

template CovarianceMatrixKey<float,Eigen::Dynamic>::CovarianceMatrixKey(Key< Covariance<float> > const &);
template CovarianceMatrixKey<float,2>::CovarianceMatrixKey(Key< Covariance<Point<float> > > const &);
template CovarianceMatrixKey<float,3>::CovarianceMatrixKey(Key< Covariance<Moments<float> > > const &);

}}} // namespace lsst::afw::table
