// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 

/**
 * @file
 * @brief Fill Images with Random numbers
 * @ingroup afw
 */
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageAlgorithm.h"
#include "lsst/afw/math/Random.h"

namespace lsst {
namespace afw {
namespace math {

namespace {

    template<typename T>
    struct do_random : public lsst::afw::image::pixelOp0<T>{
        do_random(Random &rand) : _rand(rand) {}
    protected:
        Random &_rand;
    };

    template<typename T>
    struct do_uniform  : public do_random<T> {
        do_uniform(Random &rand) : do_random<T>(rand) {}

        virtual T operator()() const { return do_random<T>::_rand.uniform(); }
    };

    template<typename T>
    struct do_uniformPos  : public do_random<T> {
        do_uniformPos(Random &rand) : do_random<T>(rand) {}

        virtual T operator()() const { return do_random<T>::_rand.uniformPos(); }
    };

    template<typename T>
    struct do_uniformInt  : public do_random<T> {
        do_uniformInt(Random &rand, unsigned long n) : do_random<T>(rand), _n(n) {}

        virtual T operator()() const { return do_random<T>::_rand.uniformInt(_n); }
    private:
        unsigned long _n;
    };

    template<typename T>
    struct do_flat : public do_random<T> {
        do_flat(Random &rand, double const a, double const b) : do_random<T>(rand), _a(a), _b(b) {}

        virtual T operator()() const { return do_random<T>::_rand.flat(_a, _b); }
    private:
        double const _a;
        double const _b;
    };

    template<typename T>
    struct do_gaussian : public do_random<T> {
        do_gaussian(Random &rand) : do_random<T>(rand) {}

        virtual T operator()() const { return do_random<T>::_rand.gaussian(); }
    };

    template<typename T>
    struct do_chisq : public do_random<T> {
        do_chisq(Random &rand, double nu) : do_random<T>(rand), _nu(nu) {}

        virtual T operator()() const { return do_random<T>::_rand.chisq(_nu); }
    private:
        double const _nu;
    };

    template<typename T>
    struct do_poisson : public do_random<T> {
        do_poisson(Random &rand, double mu) : do_random<T>(rand), _mu(mu) {}

        virtual T operator()() const { return do_random<T>::_rand.poisson(_mu); }
    private:
        double const _mu;
    };
}

/************************************************************************************************************/
/**
 * Set image to random numbers uniformly distributed in the range [0, 1)
 */
template<typename ImageT>
void randomUniformImage(ImageT *image,  ///< The image to set
                        Random &rand    ///< definition of random number algorithm, seed, etc.
                       ) {
    lsst::afw::image::for_each_pixel(*image, do_uniform<typename ImageT::Pixel>(rand));
}

/**
 * Set image to random numbers uniformly distributed in the range (0, 1)
 */
template<typename ImageT>
void randomUniformPosImage(ImageT *image,  ///< The image to set
                           Random &rand    ///< definition of random number algorithm, seed, etc.
                          ) {
    lsst::afw::image::for_each_pixel(*image, do_uniformPos<typename ImageT::Pixel>(rand));
}

/**
 * Set image to random integers uniformly distributed in the range 0 ... n - 1
 */
template<typename ImageT>
void randomUniformIntImage(ImageT *image,  ///< The image to set
                           Random &rand,   ///< definition of random number algorithm, seed, etc.
                           unsigned long n ///< (exclusive) upper limit for random variates
                          ) {
    lsst::afw::image::for_each_pixel(*image, do_uniformInt<typename ImageT::Pixel>(rand, n));
}

/**
 * Set image to random numbers uniformly distributed in the range [a, b)
 */
template<typename ImageT>
void randomFlatImage(ImageT *image,     ///< The image to set
                     Random &rand,      ///< definition of random number algorithm, seed, etc.
                     double const a,    ///< (inclusive) lower limit for random variates
                     double const b     ///< (exclusive) upper limit for random variates
                    ) {
    lsst::afw::image::for_each_pixel(*image, do_flat<typename ImageT::Pixel>(rand, a, b));
}

/**
 * Set image to random numbers with a gaussian N(0, 1) distribution
 */
template<typename ImageT>
void randomGaussianImage(ImageT *image,  ///< The image to set
                         Random &rand    ///< definition of random number algorithm, seed, etc.
                        ) {
    lsst::afw::image::for_each_pixel(*image, do_gaussian<typename ImageT::Pixel>(rand));
}

/**
 * Set image to random numbers with a chi^2_{nu} distribution
 */
template<typename ImageT>
void randomChisqImage(ImageT *image,    ///< The image to set
                      Random &rand,     ///< definition of random number algorithm, seed, etc.
                      double const nu   ///< number of degrees of freedom
                     ) {
    lsst::afw::image::for_each_pixel(*image, do_chisq<typename ImageT::Pixel>(rand, nu));
}


/**
 * Set image to random numbers with a Poisson distribution with mean mu (n.b. not per-pixel)
 */
template<typename ImageT>
void randomPoissonImage(ImageT *image,    ///< The image to set
                      Random &rand,     ///< definition of random number algorithm, seed, etc.
                      double const mu   ///< mean of distribution
                     ) {
    lsst::afw::image::for_each_pixel(*image, do_poisson<typename ImageT::Pixel>(rand, mu));
}

/************************************************************************************************************/
//
// Explicit instantiations
//
/// \cond
#define INSTANTIATE(T) \
    template void randomUniformImage(lsst::afw::image::Image<T> *image, Random &rand); \
    template void randomUniformPosImage(lsst::afw::image::Image<T> *image, Random &rand); \
    template void randomUniformIntImage(lsst::afw::image::Image<T> *image, Random &rand, unsigned long n); \
    template void randomFlatImage(lsst::afw::image::Image<T> *image, \
                                  Random &rand, double const a, double const b); \
    template void randomGaussianImage(lsst::afw::image::Image<T> *image, Random &rand); \
    template void randomChisqImage(lsst::afw::image::Image<T> *image, Random &rand, double const nu); \
    template void randomPoissonImage(lsst::afw::image::Image<T> *image, Random &rand, double const mu);
    
INSTANTIATE(double)
INSTANTIATE(float)
/// \endcond

}}}
