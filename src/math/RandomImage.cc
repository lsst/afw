// -*- lsst-c++ -*-

/**
 * @file
 * @brief Fill Images with Random numbers
 * @ingroup afw
 */
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Random.h"

namespace lsst {
namespace afw {
namespace math {

template<typename ImageT, typename FunctionT>
FunctionT for_each_pixel(ImageT *image, FunctionT func) {
    for (int y = 0; y != image->getHeight(); ++y) {
        for (typename ImageT::x_iterator ptr = image->row_begin(y), end = image->row_end(y); ptr != end; ++ptr) {
            *ptr = func();
        }
    }

    return func;
}

namespace {
    struct do_random {
        do_random(Random &rand) : _rand(rand) {}
        virtual ~do_random() {}

        virtual double operator()() = 0;
    protected:
        Random &_rand;
    };

    struct do_uniform  : public do_random {
        do_uniform(Random &rand) : do_random(rand) {}

        virtual double operator()() { return _rand.uniform(); }
    };

    struct do_uniformPos  : public do_random {
        do_uniformPos(Random &rand) : do_random(rand) {}

        virtual double operator()() { return _rand.uniformPos(); }
    };

    struct do_uniformInt  : public do_random {
        do_uniformInt(Random &rand, unsigned long n) : do_random(rand), _n(n) {}

        virtual double operator()() { return _rand.uniformInt(_n); }
    private:
        unsigned long _n;
    };

    struct do_flat : public do_random {
        do_flat(Random &rand, double const a, double const b) : do_random(rand), _a(a), _b(b) {}

        virtual double operator()() { return _rand.flat(_a, _b); }
    private:
        double const _a;
        double const _b;
    };

    struct do_gaussian : public do_random {
        do_gaussian(Random &rand) : do_random(rand) {}

        virtual double operator()() { return _rand.gaussian(); }
    };

    struct do_chisq : public do_random {
        do_chisq(Random &rand, double nu) : do_random(rand), _nu(nu) {}

        virtual double operator()() { return _rand.chisq(_nu); }
    private:
        double const _nu;
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
    for_each_pixel(image, do_uniform(rand));
}

/**
 * Set image to random numbers uniformly distributed in the range (0, 1)
 */
template<typename ImageT>
void randomUniformPosImage(ImageT *image,  ///< The image to set
                           Random &rand    ///< definition of random number algorithm, seed, etc.
                          ) {
    for_each_pixel(image, do_uniformPos(rand));
}

/**
 * Set image to random integers uniformly distributed in the range 0 ... n - 1
 */
template<typename ImageT>
void randomUniformIntImage(ImageT *image,   ///< The image to set
                           Random &rand,    ///< definition of random number algorithm, seed, etc.
                           unsigned long n  ///< maximum value (exclusive)
                          ) {
    for_each_pixel(image, do_uniformInt(rand, n));
}

/**
 * Set image to random numbers uniformly distributed in the range [a, b)
 */
template<typename ImageT>
void randomFlatImage(ImageT *image,     ///< The image to set
                     Random &rand,      ///< definition of random number algorithm, seed, etc.
                     double const a,    ///< minimum value (inclusive)
                     double const b     ///< maximum value (exclusive)
                    ) {
    for_each_pixel(image, do_flat(rand, a, b));
}

/**
 * Set image to random numbers with a gaussian N(0, 1) distribution
 */
template<typename ImageT>
void randomGaussianImage(ImageT *image,  ///< The image to set
                         Random &rand    ///< definition of random number algorithm, seed, etc.
                        ) {
    for_each_pixel(image, do_gaussian(rand));
}

/**
 * Set image to random numbers with a gaussian N(0, 1) distribution
 */
template<typename ImageT>
void randomChisqImage(ImageT *image,    ///< The image to set
                      Random &rand,     ///< definition of random number algorithm, seed, etc.
                      double const nu   ///< number of degrees of freedom
                     ) {
    for_each_pixel(image, do_chisq(rand, nu));
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(T) \
    template void randomUniformImage(lsst::afw::image::Image<T> *image, Random &rand); \
    template void randomUniformPosImage(lsst::afw::image::Image<T> *image, Random &rand); \
    template void randomUniformIntImage(lsst::afw::image::Image<T> *image, Random &rand, unsigned long n); \
    template void randomFlatImage(lsst::afw::image::Image<T> *image, Random &rand, double const a, double const b); \
    template void randomGaussianImage(lsst::afw::image::Image<T> *image, Random &rand); \
    template void randomChisqImage(lsst::afw::image::Image<T> *image, Random &rand, double const nu);
    
INSTANTIATE(float);
INSTANTIATE(double);

}}}
