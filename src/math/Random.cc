// -*- lsst-c++ -*-

/**
 * @file
 * @brief Random number generator implementaion.
 * @ingroup afw
 */

#include "gsl/gsl_errno.h"
#include "gsl/gsl_randist.h"

#include "lsst/pex/exceptions.h"

#include "lsst/afw/math/Random.h"

namespace ex = lsst::pex::exceptions;
namespace math = lsst::afw::math;


// -- Private helper functions --------

/**
 * @internal
 * @brief   Initializes the underlying GSL random number generator.
 */
void math::RandomNumberGenerator::initialize() {
    static ::gsl_rng_type const * const supportedAlgorithms[] = {
        ::gsl_rng_mt19937,
        ::gsl_rng_ranlxs0,
        ::gsl_rng_ranlxs1,
        ::gsl_rng_ranlxs2,
        ::gsl_rng_ranlxd1,
        ::gsl_rng_ranlxd2,
        ::gsl_rng_ranlux,
        ::gsl_rng_ranlux389,
        ::gsl_rng_cmrg,
        ::gsl_rng_mrg,
        ::gsl_rng_taus,
        ::gsl_rng_taus2,
        ::gsl_rng_gfsr4
    };
    if (_algorithm < MT19937 || _algorithm > GFSR4) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Invalid random number generation algorithm");
    }
    _rng = ::gsl_rng_alloc(supportedAlgorithms[_algorithm]);
    if (_rng == 0) {
        throw LSST_EXCEPT(ex::MemoryException, "gsl_rng_alloc() failed");
    }
    ::gsl_rng_set(_rng, _seed);
}

/**
 * @internal
 * @brief   Frees memory used by the underlying GSL random number generator.
 */
void math::RandomNumberGenerator::cleanup() {
    if (_rng) {
        ::gsl_rng_free(_rng);
        _rng = 0;
    }
}


// -- Constructors and assignment --------

/**
 * Creates a random number generator that uses the MT19937 "Mersenne Twister" algorithm by
 * Makoto Matsumoto and Takuji Nishimura. The second revision of the seeding procedure
 * published by the two authors above in 2002 is used; the default seed value is 4357.
 *
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if sufficient memory to hold internal generator state cannot be allocated. 
 */
math::RandomNumberGenerator::RandomNumberGenerator() : _seed(0), _algorithm(MT19937) {
    initialize();
}

/**
 * Creates a random number generator that uses the given algorithm to produce random numbers,
 * and seeds it with the specified value. Passing a seed-value of zero will cause the
 * generator to be seeded with an algorithm specific default value.
 *
 * @param[in] algorithm     the algorithm to use for random number generation
 * @param[in] seed          the seed value to initialize the generator with
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *      Thrown if @a algorithm is not a supported algorithm. 
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if sufficient memory to hold internal generator state cannot be allocated. 
 */
math::RandomNumberGenerator::RandomNumberGenerator(Algorithm const algorithm, unsigned long seed)
    : _seed(seed), _algorithm(algorithm)
{
    initialize();
}

math::RandomNumberGenerator::~RandomNumberGenerator() {
    cleanup();
}

/**
 * Creates a copy of the given random number generator, including @b all of its internal state.
 * Both random number generators will subsequently produce an identical stream of random numbers.
 * 
 * @param[in] rng   the random number generator to copy
 */
math::RandomNumberGenerator::RandomNumberGenerator(RandomNumberGenerator const & rng)
    : _seed(rng._seed), _algorithm(rng._algorithm)
{
    _rng = ::gsl_rng_clone(rng._rng);
    if (_rng == 0) {
        throw LSST_EXCEPT(ex::MemoryException, "gsl_rng_clone() failed");
    }
}

/**
 * Copies the state of the given random number generator to this generator. Both random
 * number generators will subsequently produce an identical stream of random numbers.
 * 
 * @param[in] rng   the random number generator to copy
 *
 * @throw lsst::pex::exceptions::RuntimeErrorException
 *      Thrown if the @c gsl_rng_memcpy() function fails.
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if sufficient memory to hold a copy of internal generator state cannot be allocated. 
 */
math::RandomNumberGenerator & math::RandomNumberGenerator::operator=(
    RandomNumberGenerator const & rng
) {
    if (&rng != this) {
        if (rng._algorithm == _algorithm) {
            int status = ::gsl_rng_memcpy(_rng, rng._rng);
            if (status != 0) {
                throw LSST_EXCEPT(ex::RuntimeErrorException,
                                  std::string("gsl_rng_memcpy() failed: ") + ::gsl_strerror(status));
            }
        } else {
            cleanup();
            _algorithm = rng._algorithm;
            _rng = ::gsl_rng_clone(rng._rng);
            if (_rng == 0) {
                throw LSST_EXCEPT(ex::MemoryException, "gsl_rng_clone() failed");
            }
        }
        _seed = rng._seed;
    }
}


// -- Accessors --------

/**
 * @return  The algorithm in use by this random number generator.
 */
math::RandomNumberGenerator::Algorithm math::RandomNumberGenerator::getAlgorithm() const {
    return _algorithm;
}

/**
 * @return  The name of the algorithm in use by this random number generator.
 */
std::string math::RandomNumberGenerator::getAlgorithmName() const {
    return std::string(::gsl_rng_name(_rng));
}

/**
 * @return  The integer this random number generator was seeded with.
 * @note    A seed value of 0 indicates that the random number generator
 *          was seeded with an algorithm specific default value.
 */
unsigned long math::RandomNumberGenerator::getSeed() const {
    return _seed;
}

/**
 * @return  The algorithm specific minimum value (inclusive) that
 *          will be returned by calls to get()
 */
unsigned long math::RandomNumberGenerator::getMin() const {
    return ::gsl_rng_min(_rng);
}

/**
 * @return  The algorithm specific maximum value (inclusive) that
 *          will be returned by calls to get()
 */
unsigned long math::RandomNumberGenerator::getMax() const {
    return ::gsl_rng_max(_rng);
}


// -- Mutators: generating random numbers --------

/**
 * Returns a random integer from the generator. The minimum and maximum values depend on the
 * algorithm used, but all integers in the range [getMin(),getMax()] are equally likely.
 *
 * @return  a uniformly distributed random integer covering an algorithm specific range.
 */
unsigned long math::RandomNumberGenerator::get() {
    return ::gsl_rng_get(_rng);
}

/**
 * Returns a uniformly distributed random double precision floating point number from the
 * generator. The random number will be in the range [0, 1); the range includes 0.0 but
 * excludes 1.0. Note that some algorithms will not produce randomness across all mantissa
 * bits - choose an algorithm that produces double precisions results (such as
 * RandomNumberGenerator::RANLXD1, RandomNumberGenerator::TAUS, or
 * RandomNumberGenerator::MT19937) if this is important.
 *
 * @return  a uniformly distributed random double precision floating point
 *          number in the range [0, 1).
 * @sa uniformPositiveDouble()
 */
double math::RandomNumberGenerator::uniformDouble() {
    return ::gsl_rng_uniform(_rng);
}

/**
 * Returns a uniformly distributed random double precision floating point number from the
 * generator. The random number will be in the range (0, 1); the range excludes both 0.0
 * and 1.0. Note that some algorithms will not produce randomness across all mantissa
 * bits - choose an algorithm that produces double precisions results (such as
 * RandomNumberGenerator::RANLXD1, RandomNumberGenerator::TAUS, or
 * RandomNumberGenerator::MT19937) if this is important.
 *
 * @return  a uniformly distributed random double precision floating point
 *          number in the range (0, 1).
 */
double math::RandomNumberGenerator::uniformPositiveDouble() {
    return ::gsl_rng_uniform_pos(_rng);
}

/**
 * Returns a uniformly distributed random integer from 0 to @a n-1.
 *
 * This function is not intended to generate values across the full range
 * of unsigned integer values [0, 2^32 - 1]. If this is necessary, use
 * a high precision algorithm like RandomNumberGenerator::RANLXD1, RandomNumberGenerator::TAUS,
 * or RandomNumberGenerator::MT19937 with a minimum value of zero and call get() directly.
 *
 * @param[in] n     specifies the range of allowable return values (0 to @a n-1)
 * @return          a uniformly distributed random integer
 *
 * @throw lsst::pex::exceptions::RangeErrorException
 *      Thrown if @a n is larger than the algorithm specific range of the generator.
 *
 * @sa get()
 * @sa getMin()
 * @sa getMax()
 */
unsigned long math::RandomNumberGenerator::uniformInt(unsigned long n) {
    if (n > getMax() - getMin()) {
        throw LSST_EXCEPT(ex::RangeErrorException,
                          "Desired random number range exceeds generator range");
    }
    return ::gsl_rng_uniform_int(_rng, n);
}

// -- Mutators: computing random variates for various distributions --------

/**
 * Returns a random variate from the flat (uniform) distribution on [@a a, @a b).
 *
 * @param[in] a     lower endpoint of uniform distribution range (inclusive)
 * @param[in] b     upper endpoint of uniform distribution range (exclusive)
 * @return          a uniform random variate.
 *
 * @note    The implementation uses the
 *          <a href="http://en.wikipedia.org/wiki/Ziggurat_algorithm">Ziggurat algorithm</a>.
 */
double math::RandomNumberGenerator::flat(double const a, double const b) {
    return ::gsl_ran_flat(_rng, a, b);
}

/**
 * Returns a gaussian random variate with mean @a mu and standard deviation @a sigma.
 *
 * @param[in] mu    the mean of the gaussian distribution
 * @param[in] sigma the standard deviation of the gaussian distribution
 * @return          a gaussian random variate
 *
 * @note    The implementation uses the
 *          <a href="http://en.wikipedia.org/wiki/Ziggurat_algorithm">Ziggurat algorithm</a>.
 */
double math::RandomNumberGenerator::gaussian(double mu, double sigma) {
    return ::gsl_ran_gaussian_ziggurat(_rng, sigma) + mu;
}

/**
 * Returns a random variate from the chi-squared distribution with @a nu degrees of freedom.
 *
 * @param[in] nu    the number of degrees of freedom in the chi-squared distribution
 * @return          a random variate from the chi-squared distribution
 */
double math::RandomNumberGenerator::chisq(double nu) {
    return ::gsl_ran_chisq(_rng, nu);
}

