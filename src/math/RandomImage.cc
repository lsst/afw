// -*- lsst-c++ -*-

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

/*
 * Fill Images with Random numbers
 */
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageAlgorithm.h"
#include "lsst/afw/math/Random.h"

namespace lsst {
namespace afw {
namespace math {

namespace {

template <typename T>
struct do_random : public lsst::afw::image::pixelOp0<T> {
    explicit do_random(Random &rand) : _rand(rand) {}

protected:
    Random &_rand;
};

template <typename T>
struct do_uniform : public do_random<T> {
    explicit do_uniform(Random &rand) : do_random<T>(rand) {}

    virtual T operator()() const { return do_random<T>::_rand.uniform(); }
};

template <typename T>
struct do_uniformPos : public do_random<T> {
    explicit do_uniformPos(Random &rand) : do_random<T>(rand) {}

    virtual T operator()() const { return do_random<T>::_rand.uniformPos(); }
};

template <typename T>
struct do_uniformInt : public do_random<T> {
    do_uniformInt(Random &rand, unsigned long n) : do_random<T>(rand), _n(n) {}

    virtual T operator()() const { return do_random<T>::_rand.uniformInt(_n); }

private:
    unsigned long _n;
};

template <typename T>
struct do_flat : public do_random<T> {
    do_flat(Random &rand, double const a, double const b) : do_random<T>(rand), _a(a), _b(b) {}

    virtual T operator()() const { return do_random<T>::_rand.flat(_a, _b); }

private:
    double const _a;
    double const _b;
};

template <typename T>
struct do_gaussian : public do_random<T> {
    explicit do_gaussian(Random &rand) : do_random<T>(rand) {}

    virtual T operator()() const { return do_random<T>::_rand.gaussian(); }
};

template <typename T>
struct do_chisq : public do_random<T> {
    do_chisq(Random &rand, double nu) : do_random<T>(rand), _nu(nu) {}

    virtual T operator()() const { return do_random<T>::_rand.chisq(_nu); }

private:
    double const _nu;
};

template <typename T>
struct do_poisson : public do_random<T> {
    do_poisson(Random &rand, double mu) : do_random<T>(rand), _mu(mu) {}

    virtual T operator()() const { return do_random<T>::_rand.poisson(_mu); }

private:
    double const _mu;
};
}  // namespace

template <typename ImageT>
void randomUniformImage(ImageT *image, Random &rand) {
    lsst::afw::image::for_each_pixel(*image, do_uniform<typename ImageT::Pixel>(rand));
}

template <typename ImageT>
void randomUniformPosImage(ImageT *image, Random &rand) {
    lsst::afw::image::for_each_pixel(*image, do_uniformPos<typename ImageT::Pixel>(rand));
}

template <typename ImageT>
void randomUniformIntImage(ImageT *image, Random &rand, unsigned long n) {
    lsst::afw::image::for_each_pixel(*image, do_uniformInt<typename ImageT::Pixel>(rand, n));
}

template <typename ImageT>
void randomFlatImage(ImageT *image, Random &rand, double const a, double const b) {
    lsst::afw::image::for_each_pixel(*image, do_flat<typename ImageT::Pixel>(rand, a, b));
}

template <typename ImageT>
void randomGaussianImage(ImageT *image, Random &rand) {
    lsst::afw::image::for_each_pixel(*image, do_gaussian<typename ImageT::Pixel>(rand));
}

template <typename ImageT>
void randomChisqImage(ImageT *image, Random &rand, double const nu) {
    lsst::afw::image::for_each_pixel(*image, do_chisq<typename ImageT::Pixel>(rand, nu));
}

template <typename ImageT>
void randomPoissonImage(ImageT *image, Random &rand, double const mu) {
    lsst::afw::image::for_each_pixel(*image, do_poisson<typename ImageT::Pixel>(rand, mu));
}

//
// Explicit instantiations
//
/// @cond
#define INSTANTIATE(T)                                                                                     \
    template void randomUniformImage(lsst::afw::image::Image<T> *image, Random &rand);                     \
    template void randomUniformPosImage(lsst::afw::image::Image<T> *image, Random &rand);                  \
    template void randomUniformIntImage(lsst::afw::image::Image<T> *image, Random &rand, unsigned long n); \
    template void randomFlatImage(lsst::afw::image::Image<T> *image, Random &rand, double const a,         \
                                  double const b);                                                         \
    template void randomGaussianImage(lsst::afw::image::Image<T> *image, Random &rand);                    \
    template void randomChisqImage(lsst::afw::image::Image<T> *image, Random &rand, double const nu);      \
    template void randomPoissonImage(lsst::afw::image::Image<T> *image, Random &rand, double const mu);

INSTANTIATE(double)
INSTANTIATE(float)
/// @endcond
}  // namespace math
}  // namespace afw
}  // namespace lsst
