// -*- lsst-c++ -*-

/**
 * @file
 * @brief Random number generation
 * @ingroup afw
 */

#ifndef LSST_AFW_MATH_RANDOM_H
#define LSST_AFW_MATH_RANDOM_H

#include "gsl/gsl_rng.h"


namespace lsst { namespace afw { namespace math {

/**
 * An object that can be used to generate sequences of random numbers according to a number
 * of different algorithms. Support for generating random variates from the uniform, gaussian,
 * and chi-squared distributions is provided. This class is a thin wrapper for the random number
 * generation facilities of <a href="http://www.gnu.org/software/gsl/">GSL</a>, which supports
 * a very large number of additional distributions that can easily be added to this class as the
 * need arises.
 *
 * @see <a href="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Generation.html">Random number generation in GSL</a>
 * @see <a href="http://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distributions.html">Random number distributions in GSL</a>
 */
class RandomNumberGenerator {
public:

    /**
     * An enumeration of the supported random number generation algorithms.
     * For details and references on the algorithms, see
     * http://www.gnu.org/software/gsl/manual/html_node/Random-number-generator-algorithms.html
     */
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
        GFSR4
    };

    // -- Constructors and assignment --------
    RandomNumberGenerator();
    explicit RandomNumberGenerator(Algorithm algorithm, unsigned long seed = 0);

    ~RandomNumberGenerator();

    RandomNumberGenerator(RandomNumberGenerator const &);
    RandomNumberGenerator & operator=(RandomNumberGenerator const &);

    // -- Accessors --------
    Algorithm getAlgorithm() const;
    std::string getAlgorithmName() const;
    unsigned long getSeed() const;
    unsigned long getMin() const;
    unsigned long getMax() const;

    // -- Mutators: generating random numbers --------
    unsigned long get();
    double uniform();
    double uniformPos();
    unsigned long uniformInt(unsigned long n);

    // -- Mutators: computing random variates for various distributions --------
    double flat(double const a, double const b);
    double gaussian(double const mu = 0.0, double const sigma = 1.0);
    double chisq(double const nu);

private:
    ::gsl_rng * _rng;
    unsigned long _seed;
    Algorithm _algorithm;
    
    void initialize();
    void cleanup();
};

}}} // end of namespace lsst::afw::math

#endif // LSST_AFW_MATH_RANDOM_H

