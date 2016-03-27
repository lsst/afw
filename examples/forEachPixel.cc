/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <cmath>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageAlgorithm.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
template<typename T>
struct erase : public afwImage::pixelOp0<T> {
    T operator()() const {
        return 0;
    }
};

template<typename T>
struct setVal : public afwImage::pixelOp0<T> { // don't call it fill as people like to say using namespace std
    setVal(T val) : _val(val) {}
    T operator()() const {
        return _val;
    }
private:
    T _val;
};

template<typename T>
struct addOne : public afwImage::pixelOp1<T> {
    T operator()(T val) const {
        return val + 1;
    }
};

template<typename T1, typename T2>
struct divide : public afwImage::pixelOp2<T1, T2> {
    T1 operator()(T1 lhs, T2 rhs) const {
        return lhs/rhs;
    }
};

template<typename T>
struct Gaussian : public afwImage::pixelOp1XY<T> {
    Gaussian(float a, float xc, float yc, float alpha) : _a(a), _xc(xc), _yc(yc), _alpha(alpha) {}
    T operator()(int x, int y, T val) const {
        float const dx = x - _xc;
        float const dy = y - _yc;
        return val + _a*::exp(-(dx*dx + dy*dy)/(2*_alpha*_alpha));
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
    cout << img1(0,0) << " " << img2(0,0) << endl;

    // Set img1 += 1
    lsst::afw::image::for_each_pixel(img1, addOne<float>());
    cout << img1(0,0) << " " << img2(0,0) << endl;

    // Set img1 = img2 + 1
    lsst::afw::image::for_each_pixel(img1, img2, addOne<int>());
    cout << img1(0,0) << " " << img2(0,0) << endl;

    // Set img1 = 10, img2 = 3 then img1 /= img2
    lsst::afw::image::for_each_pixel(img1, setVal<float>(10));
    lsst::afw::image::for_each_pixel(img2, setVal<int>(3));

    lsst::afw::image::for_each_pixel(img1, img2, divide<float, int>());
    cout << img1(0,0) << " " << img2(0,0) << endl;

    // Set img1 = 10 + Gaussian()
    float const peak = 1000.0;          // peak value
    float const xc = 5.0;               // center of
    float const yc = 3.0;               //           Gaussian
    float const alpha = 1.5;            // "sigma" for Gaussian
    
    lsst::afw::image::for_each_pixel(img1, setVal<float>(10));
    lsst::afw::image::for_each_pixel(img1, Gaussian<float>(peak, xc, yc, alpha));
    cout << img1(0,0) << " " << img1(xc, yc) << endl;
}
