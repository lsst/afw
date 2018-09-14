// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016 LSST Corporation.
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
 * Random number generator implementaion.
 */

#include <cstdlib>
#include <limits>

#include "boost/format.hpp"

#include "gsl/gsl_errno.h"
#include "gsl/gsl_randist.h"

#include "lsst/pex/exceptions.h"

#include "lsst/afw/math/Random.h"

namespace ex = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {

// -- Static data --------

::gsl_rng_type const *const Random::_gslRngTypes[Random::NUM_ALGORITHMS] = {
        ::gsl_rng_mt19937, ::gsl_rng_ranlxs0, ::gsl_rng_ranlxs1,   ::gsl_rng_ranlxs2, ::gsl_rng_ranlxd1,
        ::gsl_rng_ranlxd2, ::gsl_rng_ranlux,  ::gsl_rng_ranlux389, ::gsl_rng_cmrg,    ::gsl_rng_mrg,
        ::gsl_rng_taus,    ::gsl_rng_taus2,   ::gsl_rng_gfsr4};

char const *const Random::_algorithmNames[Random::NUM_ALGORITHMS] = {
        "MT19937",   "RANLXS0", "RANLXS1", "RANLXS2", "RANLXD1", "RANLXD2", "RANLUX",
        "RANLUX389", "CMRG",    "MRG",     "TAUS",    "TAUS2",   "GFSR4"};

char const *const Random::_algorithmEnvVarName = "LSST_RNG_ALGORITHM";
char const *const Random::_seedEnvVarName = "LSST_RNG_SEED";

// -- Private helper functions --------

void Random::initialize() {
    ::gsl_rng *rng = ::gsl_rng_alloc(_gslRngTypes[_algorithm]);
    if (rng == 0) {
        throw std::bad_alloc();
    }
    // This seed is guaranteed to be non-zero.
    // We want to give a non-zero seed to GSL to avoid it choosing its own.
    unsigned long int useSeed = _seed == 0 ? std::numeric_limits<unsigned long int>::max() : _seed;
    ::gsl_rng_set(rng, useSeed);
    _rng.reset(rng, ::gsl_rng_free);
}

void Random::initialize(std::string const &algorithm) {
    // linear search (the number of algorithms is small)
    for (int i = 0; i < NUM_ALGORITHMS; ++i) {
        if (_algorithmNames[i] == algorithm) {
            _algorithm = static_cast<Algorithm>(i);
            initialize();
            return;
        }
    }
    throw LSST_EXCEPT(ex::InvalidParameterError, "RNG algorithm " + algorithm + " is not supported");
}

// -- Constructor --------

Random::Random(Algorithm const algorithm, unsigned long seed) : _rng(), _seed(seed), _algorithm(algorithm) {
    if (_algorithm < 0 || _algorithm >= NUM_ALGORITHMS) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Invalid RNG algorithm");
    }
    initialize();
}

Random::Random(std::string const &algorithm, unsigned long seed) : _rng(), _seed(seed) {
    initialize(algorithm);
}

Random Random::deepCopy() const {
    Random rng = *this;
    rng._rng.reset(::gsl_rng_clone(_rng.get()), ::gsl_rng_free);
    if (!rng._rng) {
        throw std::bad_alloc();
    }
    return rng;
}

Random::State Random::getState() const {
    return State(static_cast<char *>(::gsl_rng_state(_rng.get())), getStateSize());
}

void Random::setState(State const &state) {
    if (state.size() != getStateSize()) {
        throw LSST_EXCEPT(
                pex::exceptions::LengthError,
                (boost::format("Size of given state vector (%d) does not match expected size (%d)") %
                 state.size() % getStateSize())
                        .str());
    }
    std::copy(state.begin(), state.end(), static_cast<char *>(::gsl_rng_state(_rng.get())));
}

std::size_t Random::getStateSize() const { return ::gsl_rng_size(_rng.get()); }

// -- Accessors --------

Random::Algorithm Random::getAlgorithm() const { return _algorithm; }

std::string Random::getAlgorithmName() const { return std::string(_algorithmNames[_algorithm]); }

std::vector<std::string> const &Random::getAlgorithmNames() {
    static std::vector<std::string> names;
    if (names.size() == 0) {
        for (int i = 0; i < NUM_ALGORITHMS; ++i) {
            names.push_back(_algorithmNames[i]);
        }
    }
    return names;
}

unsigned long Random::getSeed() const { return _seed; }

// -- Mutators: generating random numbers --------

double Random::uniform() { return ::gsl_rng_uniform(_rng.get()); }

double Random::uniformPos() { return ::gsl_rng_uniform_pos(_rng.get()); }

unsigned long Random::uniformInt(unsigned long n) {
    if (n > ::gsl_rng_max(_rng.get()) - ::gsl_rng_min(_rng.get())) {
        throw LSST_EXCEPT(ex::RangeError, "Desired random number range exceeds generator range");
    }
    return ::gsl_rng_uniform_int(_rng.get(), n);
}

// -- Mutators: computing random variates for various distributions --------

double Random::flat(double const a, double const b) { return ::gsl_ran_flat(_rng.get(), a, b); }

double Random::gaussian() { return ::gsl_ran_gaussian_ziggurat(_rng.get(), 1.0); }

double Random::chisq(double nu) { return ::gsl_ran_chisq(_rng.get(), nu); }

double Random::poisson(double mu) { return ::gsl_ran_poisson(_rng.get(), mu); }
}  // namespace math
}  // namespace afw
}  // namespace lsst
