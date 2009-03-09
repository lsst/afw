// -*- lsst-c++ -*-

/**
 * @file
 * @brief   Random number generator class.
 * @ingroup afw
 */

#ifndef LSST_AFW_MATH_RANDOM_H
#define LSST_AFW_MATH_RANDOM_H

#include "boost/shared_ptr.hpp"

#include "gsl/gsl_rng.h"

#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"


namespace lsst { namespace afw { namespace math {

/**
 * A class that can be used to generate sequences of random numbers according to a number
 * of different algorithms. Support for generating random variates from the uniform, gaussian,
 * and chi-squared distributions is provided.
 *
 * This class is a thin wrapper for the random number generation facilities of
 * <a href="http://www.gnu.org/software/gsl/">GSL</a>, which supports many
 * additional distributions that can easily be added to this class as the
 * need arises.
 *
 * To enable reproducibility, factory functions which determine the algorithm type and seed value
 * to used based on the the @c LSST_RNG_ALGORITHM and @c LSST_RNG_SEED environment variables (or the
 * "rngAlgorithm" and "rngSeed" keys in a policy) are provided.
 *
 * @see <a href="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Generation.html">Random number generation in GSL</a>
 * @see <a href="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">Random number distributions in GSL</a>
 */
class Random {
public:

    /** Identifiers for the list of supported algorithms. */
    enum Algorithm {
        /** The @c MT19937 "Mersenne Twister" generator of Makoto Matsumoto and Takuji Nishimura. */
        MT19937 = 0,
        /** Second-generation version of the RANLUX algorithm of Lüscher, 24-bit output, luxury level 0 (weakest) */
        RANLXS0,
        /** Second-generation version of the RANLUX algorithm of Lüscher, 24-bit output, luxury level 1 (stronger) */
        RANLXS1,
        /** Second-generation version of the RANLUX algorithm of Lüscher, 24-bit output, luxury level 2 (strongest) */
        RANLXS2,
        /** Double precision (48-bit) output using the @c RANLXS algorithm, luxury level 1 (weakest). */
        RANLXD1,
        /** Double precision (48-bit) output using the @c RANLXS algorithm, luxury level 2 (strongest). */
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
    explicit Random(Algorithm algorithm = MT19937, unsigned long seed = 0);
    explicit Random(std::string const & algorithm, unsigned long seed = 0);
    // Use compiler generated destructor and shallow copy constructor/assignment operator

    Random deepCopy() const;

    // -- Accessors --------
    Algorithm getAlgorithm() const;
    std::string getAlgorithmName() const;
    static std::vector<std::string> const & getAlgorithmNames();
    unsigned long getSeed() const;
    unsigned long getMin() const;
    unsigned long getMax() const;

    // -- Modifiers: generating random numbers --------
    unsigned long get();
    double uniform();
    double uniformPos();
    unsigned long uniformInt(unsigned long n);

    // -- Modifiers: computing random variates for various distributions --------
    double flat(double const a, double const b);
    double gaussian(double const sigma = 1.0, double const mu = 0.0);
    double chisq(double const nu);

    // -- Factory functions --------

    // create RNGs, allowing policy/environment variables to override the algorithm and seed.
    static Random create(
        lsst::pex::policy::Policy::Ptr policy,
        Algorithm algorithm = MT19937,
        unsigned long seed = 0
    );
    static Random create(
        lsst::pex::policy::Policy::Ptr policy,
        std::string const & algorithm,
        unsigned long seed = 0
    );

    // create RNGs, allowing environment variables to override the algorithm and seed.
    static inline Random create(Algorithm algorithm = MT19937, unsigned long seed = 0);
    static inline Random create(std::string const & algorithm, unsigned long seed = 0);

private:
    boost::shared_ptr< ::gsl_rng> _rng;
    unsigned long _seed;
    Algorithm _algorithm;

    static ::gsl_rng_type const * const _gslRngTypes[NUM_ALGORITHMS];
    static char const * const _algorithmNames[NUM_ALGORITHMS];
    static char const * const _algorithmEnvVarName;
    static char const * const _seedEnvVarName;

    void initialize();
};


// -- Inline function implementations --------

/**
 * Equivalent to calling @c create() with a null (or empty) policy and
 * the given algorithm/seed. 
 *
 * @copydoc create(lsst::pex::Policy::Ptr, Algorithm, unsigned long)
 * @sa create(lsst::pex::Policy::Ptr, Algorithm, unsigned long)
 */
Random Random::create(Random::Algorithm algorithm, unsigned long seed) {
    return create(lsst::pex::policy::Policy::Ptr(), algorithm, seed);
}

/**
 * Equivalent to calling @c create() with a null (or empty) policy and
 * the given algorithm/seed. 
 *
 * @copydoc create(lsst::pex::Policy::Ptr, std::string const &, unsigned long)
 * @sa create(lsst::pex::Policy::Ptr, std::string const &, unsigned long)
 */
Random Random::create(std::string const & algorithm, unsigned long seed) {
    return create(lsst::pex::policy::Policy::Ptr(), algorithm, seed);
}

}}} // end of namespace lsst::afw::math

#endif // LSST_AFW_MATH_RANDOM_H

