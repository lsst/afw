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
 * Random number generator class.
 */

#ifndef LSST_AFW_MATH_RANDOM_H
#define LSST_AFW_MATH_RANDOM_H

#include <memory>

#include "gsl/gsl_rng.h"

#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"


namespace lsst { namespace afw { namespace math {

/**
 * A class that can be used to generate sequences of random numbers according to a number
 * of different algorithms. Support for generating random variates from the uniform,  Gaussian, Poisson,
 * and chi-squared distributions is provided.
 *
 * This class is a thin wrapper for the random number generation facilities of
 * <a href="http://www.gnu.org/software/gsl/">GSL</a>, which supports many
 * additional distributions that can easily be added to this class as the
 * need arises.
 *
 * To enable reproducibility, factory functions which determine the algorithm type and seed value
 * to used based on the the `LSST_RNG_ALGORITHM` and `LSST_RNG_SEED` environment variables (or the
 * "rngAlgorithm" and "rngSeed" keys in a policy) are provided.
 *
 * @see <a href="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Generation.html">Random number generation in GSL</a>
 * @see <a href="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">Random number distributions in GSL</a>
 */
class Random {
public:

    /** Identifiers for the list of supported algorithms. */
    enum Algorithm {
        /** The `MT19937` "Mersenne Twister" generator of Makoto Matsumoto and Takuji Nishimura. */
        MT19937 = 0,
        /** Second-generation version of the RANLUX algorithm of Lüscher, 24-bit output, luxury level 0 (weakest) */
        RANLXS0,
        /** Second-generation version of the RANLUX algorithm of Lüscher, 24-bit output, luxury level 1 (stronger) */
        RANLXS1,
        /** Second-generation version of the RANLUX algorithm of Lüscher, 24-bit output, luxury level 2 (strongest) */
        RANLXS2,
        /** Double precision (48-bit) output using the `RANLXS` algorithm, luxury level 1 (weakest). */
        RANLXD1,
        /** Double precision (48-bit) output using the `RANLXS` algorithm, luxury level 2 (strongest). */
        RANLXD2,
        /** Original version of the RANLUX algorithm, 24-bit output. */
        RANLUX,
        /** Original version of the RANLUX algorithm, 24-bit output (all bits are decorrelated). */
        RANLUX389,
        /** Combined multiple recursive generator by L'Ecuyer. */
        CMRG,
        /** Fifth-order multiple recursive generator by L'Ecuyer, Blouin, and Coutre. */
        MRG,
        /** A maximally equidistributed combined Tausworthe generator by L'Ecuyer. */
        TAUS,
        /** A maximally equidistributed combined Tausworthe generator by L'Ecuyer with improved seeding relative to TAUS. */
        TAUS2,
        /** A fifth-order multiple recursive generator by L'Ecuyer, Blouin, and Coutre. */
        GFSR4,
        /** Number of supported algorithms */
        NUM_ALGORITHMS
    };

    // -- Constructor --------
    /**
     * Creates a random number generator that uses the given algorithm to produce random numbers,
     * and seeds it with the specified value. Passing a seed-value of zero will cause the
     * generator to be seeded with an algorithm specific default value. The default value for
     * `algorithm` is MT19937, corresponding to the "Mersenne Twister" algorithm by
     * Makoto Matsumoto and Takuji Nishimura.
     *
     * @param[in] algorithm     the algorithm to use for random number generation
     * @param[in] seed          the seed value to initialize the generator with
     *
     * @throws lsst::pex::exceptions::InvalidParameterError
     *      Thrown if the requested algorithm is not supported or a seed value of zero
     *      (corresponding to an algorithm specific seed) is chosen.
     * @throws lsst::pex::exceptions::MemoryError
     *      Thrown if memory allocation for internal generator state fails.
     */
    explicit Random(Algorithm algorithm = MT19937, unsigned long seed = 1);
    /**
     * Creates a random number generator that uses the algorithm with the given name to produce
     * random numbers, and seeds it with the specified value. Passing a seed-value of zero will
     * cause the generator to be seeded with an algorithm specific default value.
     *
     * @param[in] algorithm     the name of the algorithm to use for random number generation
     * @param[in] seed          the seed value to initialize the generator with
     *
     * @throws lsst::pex::exceptions::InvalidParameterError
     *      Thrown if the requested algorithm is not supported or a seed value of zero
     *      (corresponding to an algorithm specific seed) is chosen.
     * @throws lsst::pex::exceptions::MemoryError
     *      Thrown if memory allocation for internal generator state fails.
     */
    explicit Random(std::string const & algorithm, unsigned long seed = 1);
    /**
     * Creates a random number generator using the algorithm and seed specified
     * in the given policy. The algorithm name and seed are expected to be specified
     * in string-valued keys named "rngAlgorithm" and "rngSeed" respectively. The
     * "rngSeed" value is expected to be convertible to an unsigned long integer
     * and must not be positive.
     *
     * @param[in] policy    policy which contains the algorithm and seed to
     *                      to use for random number generation
     * @returns              a newly created random number generator
     *
     * @throws lsst::pex::exceptions::InvalidParameterError
     *      Thrown if the requested algorithm is not supported.
     * @throws lsst::pex::exceptions::MemoryError
     *      Thrown if memory allocation for internal generator state fails.
     * @throws lsst::pex::exceptions::RuntimeError
     *      Thrown if the "rngSeed" policy value cannot be converted to an unsigned long int.
     */
    explicit Random(std::shared_ptr<lsst::pex::policy::Policy> const policy);
    // Use compiler generated destructor and shallow copy constructor/assignment operator

    /**
     * Creates a deep copy of this random number generator. Both this random number
     * and its copy will subsequently produce an identical stream of random numbers.
     *
     * @returns  a deep copy of this random number generator
     *
     * @throws lsst::pex::exceptions::MemoryError
     *      Thrown if memory allocation for internal generator state fails.
     */
    Random deepCopy() const;

    //@{
    /**
     *  Accessors for the opaque state of the random number generator.
     *
     *  These may be used to save the state and restore it later, possibly after persisting.
     *  The state is algorithm-dependent, and possibly platform/architecture dependent; it
     *  should only be used for debugging perposes.
     *
     *  We use string here because it's a format Python and afw::table understand; the actual
     *  value is a binary blob that is not expected to be human-readable.
     */
    typedef std::string State;
    State getState() const;
    void setState(State const & state);
    std::size_t getStateSize() const;
    //@}

    // -- Accessors --------
    /**
     * @returns  The algorithm in use by this random number generator.
     */
    Algorithm getAlgorithm() const;
    /**
     * @returns  The name of the algorithm in use by this random number generator.
     */
    std::string getAlgorithmName() const;
    /**
     * @returns  The list of names of supported random number generation algorithms.
     */
    static std::vector<std::string> const & getAlgorithmNames();
    /**
     * @returns  The integer this random number generator was seeded with.
     * @note    A seed value of 0 indicates that the random number generator
     *          was seeded with an algorithm specific default value.
     */
    unsigned long getSeed() const;

    // -- Modifiers: generating random numbers --------
    /**
     * Returns a uniformly distributed random double precision floating point number from the
     * generator. The random number will be in the range [0, 1); the range includes 0.0 but
     * excludes 1.0. Note that some algorithms will not produce randomness across all mantissa
     * bits - choose an algorithm that produces double precisions results (such as
     * Random::RANLXD1, Random::TAUS, or
     * Random::MT19937) if this is important.
     *
     * @returns  a uniformly distributed random double precision floating point
     *          number in the range [0, 1).
     * @see uniformPositiveDouble()
     */
    double uniform();
    /**
     * Returns a uniformly distributed random double precision floating point number from the
     * generator. The random number will be in the range (0, 1); the range excludes both 0.0
     * and 1.0. Note that some algorithms will not produce randomness across all mantissa
     * bits - choose an algorithm that produces double precisions results (such as
     * Random::RANLXD1, Random::TAUS, or
     * Random::MT19937) if this is important.
     *
     * @returns  a uniformly distributed random double precision floating point
     *          number in the range (0, 1).
     */
    double uniformPos();
    /**
     * Returns a uniformly distributed random integer from `0` to `n-1`.
     *
     * This function is not intended to generate values across the full range
     * of unsigned integer values [0, 2^32 - 1]. If this is necessary, use
     * a high precision algorithm like Random::RANLXD1, Random::TAUS,
     * or Random::MT19937 with a minimum value of zero and call get() directly.
     *
     * @param[in] n     specifies the range of allowable return values (`0` to `n-1`)
     * @returns          a uniformly distributed random integer
     *
     * @throws lsst::pex::exceptions::RangeError
     *      Thrown if `n` is larger than the algorithm specific range of the generator.
     *
     * @see get()
     * @see getMin()
     * @see getMax()
     */
    unsigned long uniformInt(unsigned long n);

    // -- Modifiers: computing random variates for various distributions --------
    /**
     * Returns a random variate from the flat (uniform) distribution on [`a`, `b`).
     *
     * @param[in] a     lower endpoint of uniform distribution range (inclusive)
     * @param[in] b     upper endpoint of uniform distribution range (exclusive)
     * @returns          a uniform random variate.
     */
    double flat(double const a, double const b);
    /**
     * Returns a gaussian random variate with mean `0` and standard deviation `1`
     *
     * @returns          a gaussian random variate
     *
     * @note    The implementation uses the
     *          <a href="http://en.wikipedia.org/wiki/Ziggurat_algorithm">Ziggurat algorithm</a>.
     */
    double gaussian();
    /**
     * Returns a random variate from the chi-squared distribution with `nu` degrees of freedom.
     *
     * @param[in] nu    the number of degrees of freedom in the chi-squared distribution
     * @returns          a random variate from the chi-squared distribution
     */
    double chisq(double const nu);
    /**
     * Returns a random variate from the poisson distribution with mean `mu`.
     *
     * @param mu        desired mean (and variance)
     * @returns          a random variate from the Poission distribution
     */
    double poisson(double const mu);

private:
    std::shared_ptr< ::gsl_rng> _rng;
    unsigned long _seed;
    Algorithm _algorithm;

    static ::gsl_rng_type const * const _gslRngTypes[NUM_ALGORITHMS];
    static char const * const _algorithmNames[NUM_ALGORITHMS];
    static char const * const _algorithmEnvVarName;
    static char const * const _seedEnvVarName;

    /**
     * Initializes the underlying GSL random number generator.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError
     *      Thrown if a seed value of zero (corresponding to an algorithm specific seed) is chosen.
     */
    void initialize();
    /**
     * Initializes the underlying GSL random number generator.
     *
     * @param[in] algorithm     the algorithm to use for random number generation
     *
     * @throws lsst::pex::exceptions::InvalidParameterError
     *      Thrown if the requested algorithm is not supported or a seed value of zero
     *      (corresponding to an algorithm specific seed) is chosen.
     */
    void initialize(std::string const &);
};

/*
 * Create Images containing random numbers
 */
/**
 * Set image to random numbers uniformly distributed in the range [0, 1)
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 */
template<typename ImageT>
void randomUniformImage(ImageT *image, Random &rand);

/**
 * Set image to random numbers uniformly distributed in the range (0, 1)
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 */
template<typename ImageT>
void randomUniformPosImage(ImageT *image, Random &rand);

/**
 * Set image to random integers uniformly distributed in the range 0 ... n - 1
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 * @param[in] n (exclusive) upper limit for random variates
 */
template<typename ImageT>
void randomUniformIntImage(ImageT *image, Random &rand, unsigned long n);

/**
 * Set image to random numbers uniformly distributed in the range [a, b)
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 * @param[in] a (inclusive) lower limit for random variates
 * @param[in] b (exclusive) upper limit for random variates
 */
template<typename ImageT>
void randomFlatImage(ImageT *image, Random &rand, double const a, double const b);

/**
 * Set image to random numbers with a gaussian N(0, 1) distribution
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 */
template<typename ImageT>
void randomGaussianImage(ImageT *image, Random &rand);

/**
 * Set image to random numbers with a chi^2_{nu} distribution
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 * @param[in] nu number of degrees of freedom
 */
template<typename ImageT>
void randomChisqImage(ImageT *image, Random &rand, double const nu);

/**
 * Set image to random numbers with a Poisson distribution with mean mu (n.b. not per-pixel)
 *
 * @param[out] image The image to set
 * @param[in, out] rand definition of random number algorithm, seed, etc.
 * @param[in] mu mean of distribution
 */
template<typename ImageT>
void randomPoissonImage(ImageT *image, Random &rand, double const mu);


}}} // end of namespace lsst::afw::math

#endif // LSST_AFW_MATH_RANDOM_H

