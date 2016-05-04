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
 

/**
 * @file
 * @brief   Random number generator class.
 * @ingroup afw
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
    explicit Random(Algorithm algorithm = MT19937, unsigned long seed = 1);
    explicit Random(std::string const & algorithm, unsigned long seed = 1);
    explicit Random(lsst::pex::policy::Policy::Ptr const policy);
    // Use compiler generated destructor and shallow copy constructor/assignment operator

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
    Algorithm getAlgorithm() const;
    std::string getAlgorithmName() const;
    static std::vector<std::string> const & getAlgorithmNames();
    unsigned long getSeed() const;

    // -- Modifiers: generating random numbers --------
    double uniform();
    double uniformPos();
    unsigned long uniformInt(unsigned long n);

    // -- Modifiers: computing random variates for various distributions --------
    double flat(double const a, double const b);
    double gaussian();
    double chisq(double const nu);
    double poisson(double const nu);

private:
    std::shared_ptr< ::gsl_rng> _rng;
    unsigned long _seed;
    Algorithm _algorithm;

    static ::gsl_rng_type const * const _gslRngTypes[NUM_ALGORITHMS];
    static char const * const _algorithmNames[NUM_ALGORITHMS];
    static char const * const _algorithmEnvVarName;
    static char const * const _seedEnvVarName;

    void initialize();
    void initialize(std::string const &);
};

/************************************************************************************************************/
/*
 * Create Images containing random numbers
 */
template<typename ImageT>
void randomUniformImage(ImageT *image, Random &rand);

template<typename ImageT>
void randomUniformPosImage(ImageT *image, Random &rand);

template<typename ImageT>
void randomUniformIntImage(ImageT *image, Random &rand, unsigned long n);

template<typename ImageT>
void randomFlatImage(ImageT *image, Random &rand, double const a, double const b);

template<typename ImageT>
void randomGaussianImage(ImageT *image, Random &rand);

template<typename ImageT>
void randomChisqImage(ImageT *image, Random &rand, double const nu);

template<typename ImageT>
void randomPoissonImage(ImageT *image, Random &rand, double const mu);

            
}}} // end of namespace lsst::afw::math

#endif // LSST_AFW_MATH_RANDOM_H

