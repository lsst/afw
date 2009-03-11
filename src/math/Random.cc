// -*- lsst-c++ -*-

/**
 * @file
 * @brief Random number generator implementaion.
 * @ingroup afw
 */

#include <cstdlib>

#include "boost/format.hpp"
#include "boost/lexical_cast.hpp"

#include "gsl/gsl_errno.h"
#include "gsl/gsl_randist.h"

#include "lsst/pex/exceptions.h"

#include "lsst/afw/math/Random.h"

using lsst::pex::policy::Policy;

namespace ex = lsst::pex::exceptions;
namespace math = lsst::afw::math;


// -- Static data --------

::gsl_rng_type const * const math::Random::_gslRngTypes[math::Random::NUM_ALGORITHMS] = {
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

char const * const math::Random::_algorithmNames[math::Random::NUM_ALGORITHMS] = {
    "MT19937",
    "RANLXS0",
    "RANLXS1",
    "RANLXS2",
    "RANLXD1",
    "RANLXD2",
    "RANLUX",
    "RANLUX389",
    "CMRG",
    "MRG",
    "TAUS",
    "TAUS2",
    "GFSR4"
};

char const * const math::Random::_algorithmEnvVarName = "LSST_RNG_ALGORITHM";
char const * const math::Random::_seedEnvVarName = "LSST_RNG_SEED";


// -- Private helper functions --------

/**
 * @internal
 * @brief   Initializes the underlying GSL random number generator.
 */
void math::Random::initialize() {
    ::gsl_rng * rng = ::gsl_rng_alloc(_gslRngTypes[_algorithm]);
    if (rng == 0) {
        throw LSST_EXCEPT(ex::MemoryException, "gsl_rng_alloc() failed");
    }
    ::gsl_rng_set(rng, _seed);
    _rng.reset(rng, ::gsl_rng_free);
}


// -- Constructor --------

/**
 * Creates a random number generator that uses the given algorithm to produce random numbers,
 * and seeds it with the specified value. Passing a seed-value of zero will cause the
 * generator to be seeded with an algorithm specific default value. The default value for
 * @a algorithm is MT19937, corresponding to the "Mersenne Twister" algorithm by
 * Makoto Matsumoto and Takuji Nishimura.
 *
 * @param[in] algorithm     the algorithm to use for random number generation
 * @param[in] seed          the seed value to initialize the generator with
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *      Thrown if the requested algorithm is not supported.
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if memory allocation for internal generator state fails.
 */
math::Random::Random(Algorithm const algorithm, unsigned long seed)
    : _rng(), _seed(seed), _algorithm(algorithm)
{
    if (_algorithm < 0 || _algorithm >= NUM_ALGORITHMS) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Invalid RNG algorithm");
    }
    initialize();
}


/**
 * Creates a random number generator that uses the algorithm with the given name to produce
 * random numbers, and seeds it with the specified value. Passing a seed-value of zero will
 * cause the generator to be seeded with an algorithm specific default value.
 *
 * @param[in] algorithm     the name of the algorithm to use for random number generation
 * @param[in] seed          the seed value to initialize the generator with
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *      Thrown if the requested algorithm is not supported.
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if memory allocation for internal generator state fails.
 */
math::Random::Random(std::string const & algorithm, unsigned long seed)
    : _rng(), _seed(seed)
{
    // linear search (the number of algorithms is small)
    for (int i = 0; i < NUM_ALGORITHMS; ++i) {
        if (_algorithmNames[i] == algorithm) {
            _algorithm = static_cast<Algorithm>(i);
            initialize();
            return;
        }
    }
    throw LSST_EXCEPT(ex::InvalidParameterException, "RNG algorithm " +
                      algorithm + " is not supported");
}


/**
 * Creates a deep copy of this random number generator. Both this random number
 * and its copy will subsequently produce an identical stream of random numbers.
 * 
 * @return  a deep copy of this random number generator
 *
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if memory allocation for internal generator state fails.
 */
math::Random math::Random::deepCopy() const {
    Random rng = *this;
    rng._rng.reset(::gsl_rng_clone(_rng.get()), ::gsl_rng_free);
    if (!rng._rng) {
        throw LSST_EXCEPT(ex::MemoryException, "gsl_rng_clone() failed");
    }
    return rng;
}


// -- Accessors --------

/**
 * @return  The algorithm in use by this random number generator.
 */
math::Random::Algorithm math::Random::getAlgorithm() const {
    return _algorithm;
}

/**
 * @return  The name of the algorithm in use by this random number generator.
 */
std::string math::Random::getAlgorithmName() const {
    return std::string(_algorithmNames[_algorithm]);
}

/**
 * @return  The list of names of supported random number generation algorithms.
 */
std::vector<std::string> const & math::Random::getAlgorithmNames() {
    static std::vector<std::string> names;
    if (names.size() == 0) {
        for (int i = 0; i < NUM_ALGORITHMS; ++i) {
            names.push_back(_algorithmNames[i]);
        }
    }
    return names;
}

/**
 * @return  The integer this random number generator was seeded with.
 * @note    A seed value of 0 indicates that the random number generator
 *          was seeded with an algorithm specific default value.
 */
unsigned long math::Random::getSeed() const {
    return _seed;
}

/**
 * @return  The algorithm specific minimum value (inclusive) that
 *          will be returned by calls to get()
 */
unsigned long math::Random::getMin() const {
    return ::gsl_rng_min(_rng.get());
}

/**
 * @return  The algorithm specific maximum value (inclusive) that
 *          will be returned by calls to get()
 */
unsigned long math::Random::getMax() const {
    return ::gsl_rng_max(_rng.get());
}


// -- Mutators: generating random numbers --------

/**
 * Returns a random integer from the generator. The minimum and maximum values depend on the
 * algorithm used, but all integers in the range [getMin(),getMax()] are equally likely.
 *
 * @return  a uniformly distributed random integer covering an algorithm specific range.
 */
unsigned long math::Random::get() {
    return ::gsl_rng_get(_rng.get());
}

/**
 * Returns a uniformly distributed random double precision floating point number from the
 * generator. The random number will be in the range [0, 1); the range includes 0.0 but
 * excludes 1.0. Note that some algorithms will not produce randomness across all mantissa
 * bits - choose an algorithm that produces double precisions results (such as
 * Random::RANLXD1, Random::TAUS, or
 * Random::MT19937) if this is important.
 *
 * @return  a uniformly distributed random double precision floating point
 *          number in the range [0, 1).
 * @sa uniformPositiveDouble()
 */
double math::Random::uniform() {
    return ::gsl_rng_uniform(_rng.get());
}

/**
 * Returns a uniformly distributed random double precision floating point number from the
 * generator. The random number will be in the range (0, 1); the range excludes both 0.0
 * and 1.0. Note that some algorithms will not produce randomness across all mantissa
 * bits - choose an algorithm that produces double precisions results (such as
 * Random::RANLXD1, Random::TAUS, or
 * Random::MT19937) if this is important.
 *
 * @return  a uniformly distributed random double precision floating point
 *          number in the range (0, 1).
 */
double math::Random::uniformPos() {
    return ::gsl_rng_uniform_pos(_rng.get());
}

/**
 * Returns a uniformly distributed random integer from 0 to @a n-1.
 *
 * This function is not intended to generate values across the full range
 * of unsigned integer values [0, 2^32 - 1]. If this is necessary, use
 * a high precision algorithm like Random::RANLXD1, Random::TAUS,
 * or Random::MT19937 with a minimum value of zero and call get() directly.
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
unsigned long math::Random::uniformInt(unsigned long n) {
    if (n > getMax() - getMin()) {
        throw LSST_EXCEPT(ex::RangeErrorException,
                          "Desired random number range exceeds generator range");
    }
    return ::gsl_rng_uniform_int(_rng.get(), n);
}

// -- Mutators: computing random variates for various distributions --------

/**
 * Returns a random variate from the flat (uniform) distribution on [@a a, @a b).
 *
 * @param[in] a     lower endpoint of uniform distribution range (inclusive)
 * @param[in] b     upper endpoint of uniform distribution range (exclusive)
 * @return          a uniform random variate.
 */
double math::Random::flat(double const a, double const b) {
    return ::gsl_ran_flat(_rng.get(), a, b);
}

/**
 * Returns a gaussian random variate with mean @a mu and standard deviation @a sigma.
 *
 * @param[in] sigma the standard deviation of the gaussian distribution
 * @param[in] mu    the mean of the gaussian distribution
 * @return          a gaussian random variate
 *
 * @note    The implementation uses the
 *          <a href="http://en.wikipedia.org/wiki/Ziggurat_algorithm">Ziggurat algorithm</a>.
 */
double math::Random::gaussian(double sigma, double mu) {
    return ::gsl_ran_gaussian_ziggurat(_rng.get(), sigma) + mu;
}

/**
 * Returns a random variate from the chi-squared distribution with @a nu degrees of freedom.
 *
 * @param[in] nu    the number of degrees of freedom in the chi-squared distribution
 * @return          a random variate from the chi-squared distribution
 */
double math::Random::chisq(double nu) {
    return ::gsl_ran_chisq(_rng.get(), nu);
}


// -- Factory functions --------

/**
 * Creates a random number generator using the given algorithm and seed, both of which may
 * be overriden by policy as well as environment variables. The actual algorithm and seed
 * value are determined as follows:
 *
 *   - First, @a policy is checked for string-valued keys named "rngAlgorithm" and "rngSeed".
 *     If both keys exist, the corresponding policy values are used to create the generator.
 *   - If @a policy doesn't contain these keys, the environment is checked for the
 *     @c LSST_RNG_ALGORITHM and @c LSST_RNG_SEED environment variables. If both exist,
 *     their values are used in place of the user specified algorithm and seed.
 *
 * Note that the "rngSeed" policy value and @c LSST_RNG_SEED environment variable value
 * must be convertible to an unsigned long integer.
 *
 * @param[in] policy        policy which can potentially override the values
 *                          of @a algorithm and @a seed
 * @param[in] algorithm     the algorithm to use for random number generation
 * @param[in] seed          the seed value to initialize the generator with
 * @return                  a newly created random number generator
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *      Thrown if the requested algorithm is not supported.
 * @throw lsst::pex::exceptions::MemoryException
 *      Thrown if memory allocation for internal generator state fails.
 * @throw lsst::pex::exceptions::RuntimeErrorException
 *      Thrown if the "rngSeed" policy value or @c LSST_RNG_SEED environment variable
 *      value cannot be converted to an unsigned long int.
 */
math::Random math::Random::create(
    lsst::pex::policy::Policy::Ptr policy,
    Algorithm algorithm,
    unsigned long seed
) {
    if (algorithm < 0 || algorithm >= NUM_ALGORITHMS) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Invalid RNG algorithm");
    }
    return create(policy, _algorithmNames[algorithm], seed);
}

/**
 * @copydoc create(lsst::pex::Policy::Ptr, Algorithm, unsigned long)
 */
math::Random math::Random::create(
    lsst::pex::policy::Policy::Ptr policy,
    std::string const & algorithm,
    unsigned long seed
) {
    std::string rngAlgorithm = algorithm;
    unsigned long rngSeed = seed;

    if (policy && policy->exists("rngAlgorithm") && policy->exists("rngSeed")) {
        std::string const seedString(policy->getString("rngSeed"));
        rngAlgorithm = policy->getString("rngAlgorithm");
        try {
            rngSeed = boost::lexical_cast<unsigned long>(seedString);
        } catch(boost::bad_lexical_cast & b) {
            throw LSST_EXCEPT(ex::RuntimeErrorException,
                (boost::format("Invalid \"rngSeed\" policy value: \"%1%\"") % seedString).str());
        }
    } else {
        char * envAlg = std::getenv(_algorithmEnvVarName);
        char * envSeed = std::getenv(_seedEnvVarName);
        if (envAlg != 0 && envSeed != 0) {
            rngAlgorithm = envAlg;
            try {
                rngSeed = boost::lexical_cast<unsigned long>(envSeed);
            } catch(boost::bad_lexical_cast & b) {
                throw LSST_EXCEPT(ex::RuntimeErrorException,
                    (boost::format("Invalid \"%1%\" environment variable value: \"%2%\"") %
                        _seedEnvVarName % envSeed).str());
            }
        }
    }
    return Random(rngAlgorithm, rngSeed);
}

