/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#include <iostream>
#include <cmath>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageAlgorithm.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
template <typename T>
struct erase : public afwImage::pixelOp0<T> {
    T operator()() const override { return 0; }
};

template <typename T>
struct setVal
        : public afwImage::pixelOp0<T> {  // don't call it fill as people like to say using namespace std
    explicit setVal(T val) : _val(val) {}
    T operator()() const override { return _val; }

private:
    T _val;
};

template <typename T>
struct addOne : public afwImage::pixelOp1<T> {
    T operator()(T val) const override { return val + 1; }
};

template <typename T1, typename T2>
struct divide : public afwImage::pixelOp2<T1, T2> {
    T1 operator()(T1 lhs, T2 rhs) const override { return lhs / rhs; }
};

template <typename T>
struct Gaussian : public afwImage::pixelOp1XY<T> {
    Gaussian(float a, float xc, float yc, float alpha) : _a(a), _xc(xc), _yc(yc), _alpha(alpha) {}
    T operator()(int x, int y, T val) const override {
        float const dx = x - _xc;
        float const dy = y - _yc;
        return val + _a * ::exp(-(dx * dx + dy * dy) / (2 * _alpha * _alpha));
    }

private:
    float _a, _xc, _yc, _alpha;
};

using namespace std;

int main() {
    afwImage::Image<float> img1(afwGeom::Extent2I(10, 6));
    afwImage::Image<int> img2(img1.getDimensions());
    // set img1 to 0 (actually, the constructor already did this)
    lsst::afw::image::for_each_pixel(img1, erase<float>());

    // Set img2 to 10
    lsst::afw::image::for_each_pixel(img2, setVal<int>(10));
    cout << img1(0, 0) << " " << img2(0, 0) << endl;

    // Set img1 += 1
    lsst::afw::image::for_each_pixel(img1, addOne<float>());
    cout << img1(0, 0) << " " << img2(0, 0) << endl;

    // Set img1 = img2 + 1
    lsst::afw::image::for_each_pixel(img1, img2, addOne<int>());
    cout << img1(0, 0) << " " << img2(0, 0) << endl;

    // Set img1 = 10, img2 = 3 then img1 /= img2
    lsst::afw::image::for_each_pixel(img1, setVal<float>(10));
    lsst::afw::image::for_each_pixel(img2, setVal<int>(3));

    lsst::afw::image::for_each_pixel(img1, img2, divide<float, int>());
    cout << img1(0, 0) << " " << img2(0, 0) << endl;

    // Set img1 = 10 + Gaussian()
    float const peak = 1000.0;  // peak value
    float const xc = 5.0;       // center of Gaussian
    float const yc = 3.0;       //
    float const alpha = 1.5;    // "sigma" for Gaussian

    lsst::afw::image::for_each_pixel(img1, setVal<float>(10));
    lsst::afw::image::for_each_pixel(img1, Gaussian<float>(peak, xc, yc, alpha));
    cout << img1(0, 0) << " " << img1(xc, yc) << endl;
}
