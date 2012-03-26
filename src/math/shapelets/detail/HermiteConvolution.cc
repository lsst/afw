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

#include "lsst/afw/math/shapelets/ShapeletFunction.h"
#include "lsst/afw/math/shapelets/detail/HermiteConvolution.h"
#include "lsst/afw/geom/Angle.h"
#include "ndarray/eigen.h"

namespace lsst { namespace afw { namespace math { namespace shapelets { namespace detail {

namespace {

static double const NORMALIZATION = std::pow(geom::PI, -0.25);

class TripleProductIntegral {
public:
    
    ndarray::Array<double,3,3> const asArray() const { return _array; }

    static ndarray::Array<double,3,3> make1d(int const order1, int const order2, int const order3);

    TripleProductIntegral(int order1, int order2, int order3);

    ndarray::Vector<int,3> const & getOrders() const { return _orders; }

private:

    ndarray::Vector<int,3> _orders;
    ndarray::Array<double,3,3> _array;
};

class Binomial {
public:

    Binomial(int const n, double a, double b);

    void reset(double a, double b);

    double const operator[](int const k) const { return _workspace[k] * _coefficients[k]; }

    int const getOrder() const { return _coefficients.size()-1; }

private:
    Eigen::VectorXd _coefficients;
    Eigen::VectorXd _workspace;
};

TripleProductIntegral::TripleProductIntegral(int order1, int order2, int order3) :
    _orders(ndarray::makeVector(order1, order2, order3)),
    _array(
        ndarray::allocate(
            ndarray::makeVector(
                computeOffset(order1+1), 
                computeOffset(order2+1),
                computeOffset(order3+1)
            )
        )
    )
{
    _array.deep() = 0.0;
    ndarray::Array<double,3,3> a1d = make1d(order1, order2, order3);
    ndarray::Array<double,3,3>::Index o, i, x, y, n;
    for (n[0] = o[0] = 0; n[0] <= _orders[0]; o[0] += ++n[0]) {
        for (n[2] = o[2] = 0; n[2] <= _orders[2]; o[2] += ++n[2]) {
            int n0n2 = n[0] + n[2];
            int max = (order2 < n0n2) ? _orders[1] : n0n2;
            for (o[1] = n[1] = bool(n0n2 % 2); n[1] <= max; (o[1] += ++n[1]) += ++n[1]) {
                for (i[0] = o[0], x[0] = 0, y[0] = n[0]; x[0] <= n[0]; ++x[0], --y[0], ++i[0]) {
                    for (i[2] = o[2], x[2] = 0, y[2] = n[2]; x[2] <= n[2]; ++x[2], --y[2], ++i[2]) {
                        for (i[1] = o[1], x[1] = 0, y[1] = n[1]; x[1] <= n[1]; ++x[1], --y[1], ++i[1]) {
                            _array[i] = a1d[x] * a1d[y];
                        }
                    }
                }
            }
        }
    }
}

ndarray::Array<double,3,3>
TripleProductIntegral::make1d(int order1, int order2, int order3) {
    ndarray::Array<double,3,3> array = ndarray::allocate(ndarray::makeVector(order1+1, order2+1, order3+1));
    array.deep() = 0.0;
    double first = array[ndarray::makeVector(0,0,0)] = NORMALIZATION * M_SQRT1_2;
    if (order1 < 1) {
        if (order3 < 1) return array;
        // n1=0, n2=0, n3=even
        double previous = first;
        for (int n3 = 2; n3 <= order3; n3 += 2) {
            previous = array[ndarray::makeVector(0,0,n3)] = -std::sqrt((0.25 * (n3 - 1)) / n3) * previous;
        }
        // n1=0, n2>0, n3>0
        for (int n2 = 1; n2 <= order2; ++n2) {
            for (int n3 = n2; n3 <= order3; n3 += 2) {
                array[ndarray::makeVector(0,n2,n3)]
                    = std::sqrt((0.5 * n3) / n2) * array[ndarray::makeVector(0, n2-1, n3-1)];
            }
        }
        return array;
    }

    if (order3 < 1) {
        // n1=even, n2=0, n3=even
        double previous = first;
        for (int n1 = 2; n1 <= order1; n1 += 2) {
            previous = array[ndarray::makeVector(0,0,n1)] = -std::sqrt((0.25 * (n1 - 1)) / n1) * previous;
        }
        // n1=0, n2>0, n3>0
        for (int n2 = 1; n2 <= order2; ++n2) {
            for (int n1 = n2; n1 <= order1; n1 += 2) {
                array[ndarray::makeVector(n1,n2,0)]
                    = std::sqrt((0.5 * n1) / n2) * array[ndarray::makeVector(n1-1, n2-1, 0)];
            }
        }
        return array;
    }

    // n1=any, n2=0, n3=(0,1)
    {
        double previous = first;
        int n1 = 0;
        while (true) {
            if (++n1 > order1) break;
            array[ndarray::makeVector(n1, 0, 1)] = std::sqrt(0.25 * n1) * previous;
            if (++n1 > order1) break;
            previous = array[ndarray::makeVector(n1, 0, 0)] = -std::sqrt((0.25 * (n1 - 1)) / n1) * previous;
        }
    }

    // n1=(0,1), n2=0, n3=any
    {
        double previous = first;
        int n3 = 0;
        while (true) {
            if (++n3 > order3) break;
            array[ndarray::makeVector(1, 0, n3)] = std::sqrt(0.25 * n3) * previous;
            if (++n3 > order3) break;
            previous = array[ndarray::makeVector(0, 0, n3)] = -std::sqrt((0.25 * (n3 - 1)) / n3) * previous;
        }
    }

    // n1>1, n2=0, n3>1
    for (int n1 = 2; n1 <= order1; ++n1) {
        double f1 = -std::sqrt((0.25 * (n1 - 1)) / n1);
        for (int n3 = (n1 % 2) ? 3:2; n3 <= order3; n3 += 2) {
            array[ndarray::makeVector(n1,0,n3)]
                = f1 * array[ndarray::makeVector(n1-2, 0, n3)]
                + std::sqrt((0.25 * n3) / n1) * array[ndarray::makeVector(n1-1, 0, n3-1)];
        }
    }
    
    for (int n2 = 1; n2 <= order2; ++n2) {
        // n1>=n2, n3=0
        for (int n1 = n2; n1 <= order1; n1 += 2) {
            array[ndarray::makeVector(n1,n2,0)]
                = std::sqrt((0.5 * n1) / n2) * array[ndarray::makeVector(n1-1, n2-1, 0)];
        }
        // n1=0, n3>=n2
        for (int n3 = n2; n3 <= order3; n3 += 2) {
            array[ndarray::makeVector(0,n2,n3)]
                = std::sqrt((0.5 * n3) / n2) * array[ndarray::makeVector(0, n2-1, n3-1)];
        }
        // n1>0, n3>0
        for (int n1 = 1; n1 <= order1; ++n1) {
            for (int n3 = ((n1+n2) % 2) ? 1:2; n3 <= order3; n3 += 2) {
                array[ndarray::makeVector(n1,n2,n3)]
                    = std::sqrt((0.5 * n1) / n2) * array[ndarray::makeVector(n1-1, n2-1, n3)]
                    + std::sqrt((0.5 * n3) / n2) * array[ndarray::makeVector(n1, n2-1, n3-1)];
            }
        }
    }

    return array;
}

Binomial::Binomial(int const n, double a, double b) : _coefficients(n+1), _workspace(n+1) {
    _coefficients[0] = _coefficients[n] = 1.0;
    int const mid = n/2;
    for (int k = 1; k <= mid; ++k) {
        _coefficients[k] = _coefficients[k-1] * (n - k + 1.0) / k;
    }
    for (int k = mid+1; k < n; ++k) {
        _coefficients[k] = _coefficients[n-k];
    }
    reset(a, b);
}

void Binomial::reset(double a, double b) {
    int const n = getOrder();
    double v = 1;
    for (int k = 0; k <= n; ++k) {
        _workspace[k] = v;
        v *= b;
    }
    v = 1;
    for (int nk = n; nk >= 0; --nk) {
        _workspace[nk] *= v;
        v *= a;
    }    
}

} // anonymous

class HermiteConvolution::Impl {
public:

    Impl(int colOrder, ShapeletFunction const & psf);

    ndarray::Array<Pixel const,2,2> evaluate(afw::geom::ellipses::Ellipse & ellipse) const;

    int getColOrder() const { return _colOrder; }

    int getRowOrder() const { return _rowOrder; }

    Eigen::MatrixXd computeHermiteTransformMatrix(
        int order, afw::geom::LinearTransform const & transform
    ) const;

private:
    int _rowOrder;
    int _colOrder;
    ShapeletFunction _psf;
    ndarray::Array<Pixel,2,2> _result;
    TripleProductIntegral _tpi;
    Eigen::MatrixXd _monomialFwd;
    Eigen::MatrixXd _monomialInv;
};

HermiteConvolution::Impl::Impl(
    int colOrder, ShapeletFunction const & psf
) : 
    _rowOrder(colOrder + psf.getOrder()), _colOrder(colOrder), _psf(psf),
    _result(ndarray::allocate(computeSize(_rowOrder), computeSize(_colOrder))),
    _tpi(psf.getOrder(), _rowOrder, _colOrder),
    _monomialFwd(
        Eigen::MatrixXd::Zero(
            computeSize(_rowOrder),
            computeSize(_rowOrder)
        )
    ),
    _monomialInv(Eigen::MatrixXd::Identity(_monomialFwd.rows(), _monomialFwd.cols()))
{
    _psf.changeBasisType(HERMITE);
    _monomialFwd(0, 0) = NORMALIZATION;
    if (_rowOrder >= 1) {
        _monomialFwd(1, 1) = _monomialFwd(0, 0) * M_SQRT2;
    }
    for (int n = 2; n <= _rowOrder; ++n) {
        _monomialFwd(n, 0) = -_monomialFwd(n-2, 0) * std::sqrt((n - 1.0) / n);
        for (int m = (n % 2) ? 1:2; m <= n; m += 2) {
            _monomialFwd(n, m) 
                = _monomialFwd(n-1, m-1) * std::sqrt(2.0 / n) 
                - _monomialFwd(n-2, m) * std::sqrt((n - 1.0) / n);
        }
    }
    _monomialFwd.triangularView<Eigen::Lower>().solveInPlace(_monomialInv);
}

ndarray::Array<Pixel const,2,2> HermiteConvolution::Impl::evaluate(
    afw::geom::ellipses::Ellipse & ellipse
) const {
    ndarray::EigenView<double,2,2> result(_result);
    ndarray::EigenView<double const,1,1> psf_coeff(_psf.getCoefficients());

    afw::geom::LinearTransform psf_gt = _psf.getEllipse().getCore().getGridTransform().invert();
    afw::geom::LinearTransform model_gt = ellipse.getCore().getGridTransform().invert();
    ellipse.convolve(_psf.getEllipse()).inPlace();
    afw::geom::LinearTransform convolved_gt_inv = ellipse.getCore().getGridTransform();
    afw::geom::LinearTransform psf_htm_arg(
        Eigen::Matrix2d(M_SQRT2 * (convolved_gt_inv.getMatrix() * psf_gt.getMatrix()).transpose())
    );
    afw::geom::LinearTransform model_htm_arg(
        Eigen::Matrix2d(M_SQRT2 * (convolved_gt_inv.getMatrix() * model_gt.getMatrix()).transpose())
    );
    int const psfOrder = _psf.getOrder();
    Eigen::MatrixXd psf_htm = computeHermiteTransformMatrix(psfOrder, psf_htm_arg);
    Eigen::MatrixXd model_htm = computeHermiteTransformMatrix(_colOrder, model_htm_arg);
        
    // [kq]_m = \sum_m i^{n+m} [psf_htm]_{m,n} [psf]_n
    // kq is zero unless {n+m} is even
    Eigen::VectorXd kq = Eigen::VectorXd::Zero(psf_htm.size());
    for (int m = 0, mo = 0; m <= psfOrder; mo += ++m) {
        for (int n = m, no = mo; n <= psfOrder; (no += ++n) += ++n) {
            if ((n + m) % 4) { // (n + m) % 4 is always 0 or 2
                kq.segment(mo, m+1) -= psf_htm.block(mo, no, m+1, n+1) * psf_coeff.segment(no, n+1);
            } else {
                kq.segment(mo, m+1) += psf_htm.block(mo, no, m+1, n+1) * psf_coeff.segment(no, n+1);
            }
        }
    }
    kq *= 4.0 * geom::PI * (
        std::fabs(convolved_gt_inv.computeDeterminant()
                  * psf_gt.computeDeterminant()
                  * model_gt.computeDeterminant())
    );
        
    // [kqb]_{m,n} = \sum_l i^{m-n-l} [kq]_l [tpi]_{l,m,n}
    Eigen::MatrixXd kqb = Eigen::MatrixXd::Zero(result.rows(), result.cols());
    {
        ndarray::Array<double,3,3> b = _tpi.asArray();
        ndarray::Vector<int,3> n, o, x;
        ndarray::Vector<int,3> strides = b.getStrides();
        x[0] = 0;
        for (n[1] = o[1] = 0; n[1] <= _rowOrder; o[1] += ++n[1]) {
            for (n[2] = o[2] = 0; n[2] <= _colOrder; o[2] += ++n[2]) {
                for (n[0] = o[0] = bool((n[1]+n[2])%2); n[0] <= psfOrder; (o[0] += ++n[0]) += ++n[0]) {
                    if (n[0] + n[2] < n[1]) continue; // b is triangular
                    double factor = ((n[2] - n[0] - n[1]) % 4) ? -1.0 : 1.0;
                    for (x[1] = 0; x[1] <= n[1]; ++x[1]) {
                        for (x[2] = 0; x[2] <= n[2]; ++x[2]) {
                            // TODO: optimize this inner loop - it's sparse
                            double & out = kqb(o[1] + x[1], o[2] + x[2]);
                            for (x[0] = 0; x[0] <= n[0]; ++x[0]) {
                                out += factor * kq[o[0] + x[0]] * b[o + x];
                            }
                        }
                    }
                }
            }
        }
    }
        
    result.setZero();
    for (int m = 0, mo = 0; m <= _rowOrder; mo += ++m) {
        for (int n = 0, no = 0; n <= _colOrder; no += ++n) {
            int jo = bool(n % 2);
            for (int j = jo; j <= n; (jo += ++j) += ++j) {
                if ((n - j) % 4) { // (n - j) % 4 is always 0 or 2
                    result.block(mo, no, m+1, n+1) 
                        -= kqb.block(mo, jo, m+1, j+1) * model_htm.block(jo, no, j+1, n+1);
                } else {
                    result.block(mo, no, m+1, n+1)
                        += kqb.block(mo, jo, m+1, j+1) * model_htm.block(jo, no, j+1, n+1);
                }
            }
        }
    }

    return _result;
}

Eigen::MatrixXd HermiteConvolution::Impl::computeHermiteTransformMatrix(
    int order, geom::LinearTransform const & transform
) const {
    int const size = computeSize(order);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(size, size);
    for (int jn=0, joff=0; jn <= order; joff += (++jn)) {
        for (int kn=jn, koff=joff; kn <= order; (koff += (++kn)) += (++kn)) {
            for (int jx=0,jy=jn; jx <= jn; ++jx,--jy) {
                for (int kx=0,ky=kn; kx <= kn; ++kx,--ky) {
                    double & element = result(joff+jx, koff+kx);
                    for (int m = 0; m <= order; ++m) {
                        int const order_minus_m = order - m;
                        Binomial binomial_m(
                            m, 
                            transform[geom::LinearTransform::XX],
                            transform[geom::LinearTransform::XY]
                        );
                        for (int p = 0; p <= m; ++p) {
                            for (int n = 0; n <= order_minus_m; ++n) {
                                Binomial binomial_n(
                                    n, 
                                    transform[geom::LinearTransform::YX], 
                                    transform[geom::LinearTransform::YY]
                                );
                                for (int q = 0; q <= n; ++q) {
                                    element +=
                                        _monomialFwd(kx, m) * _monomialFwd(ky, n) *
                                        _monomialInv(m+n-p-q, jx) * _monomialInv(p+q, jy) *
                                        binomial_m[p] * binomial_n[q];
                                } // q
                            } // n
                        } // p
                    } // m
                } // kx,ky
            } // jx,jy
        } // kn
    } // jn
    return result;
}

int HermiteConvolution::getRowOrder() const { return _impl->getRowOrder(); }

int HermiteConvolution::getColOrder() const { return _impl->getColOrder(); }

ndarray::Array<Pixel const,2,2>
HermiteConvolution::evaluate(
    geom::ellipses::Ellipse & ellipse
) const {
    return _impl->evaluate(ellipse);
}

HermiteConvolution::HermiteConvolution(
    int colOrder, 
    ShapeletFunction const & psf
) : _impl(new Impl(colOrder, psf)) {}

HermiteConvolution::~HermiteConvolution() {}

}}}}} // namespace lsst::afw::math::detail
