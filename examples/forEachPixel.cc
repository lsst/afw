#include <iostream>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageFunctional.h"

namespace afwImage = lsst::afw::image;

template<typename T>
struct setVal : afwImage::pixelOp0<T> {
    setVal(T val) : _val(val) {}
    T operator()() const {
        return _val;
    }
private:
    T _val;
};

template<typename T>
struct addOne : afwImage::pixelOp1<T> {
    T operator()(T val) const {
        return val + 1;
    }
};

template<typename T1, typename T2>
struct divide : afwImage::pixelOp2<T1, T2> {
    T1 operator()(T1 lhs, T2 rhs) const {
        return lhs/rhs;
    }
};

using namespace std;

int main() {
    afwImage::Image<float> img1(10, 6);
    afwImage::Image<int> img2(10, 6);
    // Set img2 to 10
    lsst::afw::image::for_each_pixel(img2, setVal<int>(10));
    cout << img1(0,0) << " " << img2(0,0) << endl;

    // Set img1 = img2 + 1
    lsst::afw::image::for_each_pixel(img1, img2, addOne<int>());
    cout << img1(0,0) << " " << img2(0,0) << endl;

    // Set img1 = 10, img2 = 3 then img1 /= img2
    lsst::afw::image::for_each_pixel(img1, setVal<float>(10));
    lsst::afw::image::for_each_pixel(img2, setVal<int>(3));

    lsst::afw::image::for_each_pixel(img1, img2, divide<float, int>());
    cout << img1(0,0) << " " << img2(0,0) << endl;
}
