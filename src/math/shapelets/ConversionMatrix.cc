// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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

#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray/eigen.h"

#include <boost/format.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include <complex>
#include <vector>

namespace shapelets = lsst::afw::math::shapelets;
namespace nd = lsst::ndarray;

namespace {

typedef shapelets::Pixel Pixel;

inline std::complex<Pixel> iPow(int z) {
    switch (z % 4) {
    case 0:
        return std::complex<Pixel>(1.0, 0.0);
    case 1:
        return std::complex<Pixel>(0.0, 1.0);
    case 2:
        return std::complex<Pixel>(-1.0, 0.0);
    case 3:
        return std::complex<Pixel>(0.0, -1.0);
    };
}

class ConversionSingleton {
public:

    typedef std::vector<Eigen::MatrixXd> BlockVec;

    Eigen::MatrixXd const & getBlockH2L(int n) const { return _h2l[n]; }

    Eigen::MatrixXd const & getBlockL2H(int n) const { return _l2h[n]; }

    BlockVec const & getH2L() const { return _h2l; }

    BlockVec const & getL2H() const { return _l2h; }

    void ensure(int order) {
        if (order > _max_order) {
            _h2l.reserve(order + 1);
            _l2h.reserve(order + 1);
            for (int i = _max_order + 1; i <= order; ++i) {
                _h2l.push_back(makeBlockH2L(i));
                _h2l.push_back(makeBlockL2H(i, _h2l.back()));
            }
        }
    }

    static Eigen::MatrixXd makeBlockH2L(int n) {
        Eigen::MatrixXd block = Eigen::MatrixXd::Zero(n + 1, n + 1);
        for (int x = 0, y = n; x <= n; ++x, --y) {
            Pixel v_xy = std::sqrt(boost::math::factorial<Pixel>(x) * boost::math::factorial<Pixel>(y));
            for (int p = n, q = 0; q <= p; --p, ++q) {
                std::complex<Pixel> v_rs(0.0, 0.0);
                for (int r = 0; r <= p; ++r) {
                    for (int s = 0; s <= q; ++s) {
                        if (r + s == x) {
                            v_rs += iPow(r - s) * boost::math::binomial_coefficient<Pixel>(p, r) 
                                * boost::math::binomial_coefficient<Pixel>(q, s);
                        }
                    }
                }
                std::complex<Pixel> v = v_xy * v_rs * iPow(p - q) * std::pow(2.0, -0.5 * (p + q)) 
                    / std::sqrt(boost::math::factorial<Pixel>(p) * boost::math::factorial<Pixel>(q));
                if (q != p) {
                    block(2*q, x) = v.real();
                    block(2*q+1, x) = v.imag();
                } else {
                    block(2*q, x) = v.real();
                }
            }
        }
        return block;
    }

    static Eigen::MatrixXd makeBlockL2H(int n, Eigen::MatrixXd const & h2l) {
        Eigen::MatrixXd l2h = h2l.transpose();
        for (int p = n, q = 0; q < p; --p, ++q) {
            l2h.col(2*q + 1) *= -1;
        }
        return l2h;
    }

    static ConversionSingleton & get() {
        static ConversionSingleton instance;
        return instance;
    }

private:
    int _max_order;
    BlockVec _h2l;
    BlockVec _l2h;
};

} // anonymous

Eigen::MatrixXd shapelets::ConversionMatrix::getBlock(int n) const { 
    if (_input == _output) return Eigen::MatrixXd::Identity(n + 1, n + 1);
    if (_input == HERMITE)
        return ConversionSingleton::get().getBlockH2L(n);
    else
        return ConversionSingleton::get().getBlockL2H(n);
}

Eigen::MatrixXd shapelets::ConversionMatrix::buildDenseMatrix() const { 
    int const size = computeSize(_order);
    if (_input == _output) return Eigen::MatrixXd::Identity(size, size);
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(size, size);
    if (_input == HERMITE) {
        for (int n = 0, offset = 0; n <= _order; offset += ++n) {
            r.block(offset, offset, n + 1, n + 1) = ConversionSingleton::get().getBlockH2L(n);
        }
    } else {
        for (int n = 0, offset = 0; n <= _order; offset += ++n) {
            r.block(offset, offset, n + 1, n + 1) = ConversionSingleton::get().getBlockL2H(n);
        }
    }
    return r;
}

void shapelets::ConversionMatrix::multiplyOnLeft(nd::Array<Pixel,1> const & array) const {
    if (array.getSize<0>() != computeSize(_order)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            (boost::format(
                "Array for multiplyOnLeft has incorrect size (%n, should be %n)."
            ) % array.getSize<0>() % computeSize(_order)).str()
        );
    }
    if (_input == _output) return;
    ConversionSingleton::BlockVec::const_iterator i;
    if (_input == HERMITE) {
        i = ConversionSingleton::get().getH2L().begin();
    } else {
        i = ConversionSingleton::get().getL2H().begin();
    }
    nd::EigenView<Pixel,1,0> vector(array);
    for (int offset = 0; offset < vector.size(); ++i, offset += i->rows()) {
        vector.segment(offset, offset + i->rows()) *= i->transpose();
    }
}

void shapelets::ConversionMatrix::multiplyOnRight(nd::Array<Pixel,1> const & array) const {
    if (array.getSize<0>() != computeSize(_order)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            (boost::format(
                "Array for multiplyOnRight has incorrect size (%n, should be %n)."
            ) % array.getSize<0>() % computeSize(_order)).str()
        );
    }
    if (_input == _output) return;
    ConversionSingleton::BlockVec::const_iterator i;
    if (_input == HERMITE) {
        i = ConversionSingleton::get().getH2L().begin();
    } else {
        i = ConversionSingleton::get().getL2H().begin();
    }
    nd::EigenView<Pixel,1,0> vector(array);
    for (int offset = 0; offset < vector.size(); ++i, offset += i->rows()) {
        vector.segment(offset, offset + i->rows()) *= (*i);
    }
}

shapelets::ConversionMatrix::ConversionMatrix(BasisTypeEnum input, BasisTypeEnum output, int order) :
    _order(order), _input(input), _output(output)
{
    ConversionSingleton::get().ensure(_order);
}

void shapelets::ConversionMatrix::convertCoefficientVector(
    nd::Array<shapelets::Pixel,1> const & array,
    shapelets::BasisTypeEnum input, shapelets::BasisTypeEnum output, int order
) {
    if (input == output) return;
    ConversionMatrix m(input, output, order);
    m.multiplyOnLeft(array);
}

void shapelets::ConversionMatrix::convertOperationVector(
    nd::Array<shapelets::Pixel,1> const & array,
    shapelets::BasisTypeEnum input, shapelets::BasisTypeEnum output, int order
) {
    if (input == output) return;
    ConversionMatrix m(output, input, order);
    m.multiplyOnRight(array);
}
