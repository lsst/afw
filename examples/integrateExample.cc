// -*- LSST-C++ -*-

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

//
// Here are some concrete examples of the use of int1d and int2d.
//
// First, I include the examples from the comment at the beginning of
// the file Int.h.
//
// Next, I include an astronomically useful calculation of coordinate distance
// as a function of redshit.
//
// If you want more details, see the comment at the beginning of Integrate.h
//

#include <iostream>
#include <stdexcept>
#include <limits>

#include "lsst/afw/math/Integrate.h"

namespace math = lsst::afw::math;

// A simple Gaussian, parametrized by its center (mu) and size (sigma).
class Gauss : public std::unary_function<double, double> {
public:
    explicit Gauss(double mu, double sig) : _mu(mu), _sig(sig), _sigsq(sig * sig) {}

    double operator()(double x) const {
        double const SQRTTWOPI = 2.50662827463;
        return exp(-pow(x - _mu, 2) / 2.0 / _sigsq) / SQRTTWOPI / _sig;
    }

private:
    double _mu, _sig, _sigsq;
};

// In the file Int.h, I present this as a class Integrand.
// Here I do it as a function to show how that can work just as well.
double foo(double x, double y) {
    // A simple function:
    // f(x,y) = x*(3*x+y) + y
    return x * (3.0 * x + y) + y;
}

// This is stripped down from a more complete Cosmology class that
// calculates all kinds of things, including power spectra and such.
// The simplest integration calculation is the w(z) function, so that's
// all that is replicated here.
struct Cosmology {
    Cosmology(double omMxx, double omVxx, double wxx, double waxx)
            : omM(omMxx), omV(omVxx), w(wxx), wa(waxx) {}

    double calcW(double z);
    // calculate coordinate distance (in units of c/Ho) as a function of z.

    double omM, omV, w, wa;
};

struct W_Integrator : public std::unary_function<double, double> {
    explicit W_Integrator(Cosmology const &c) : _c(c) {}
    double operator()(double a) const {
        // First calculate H^2 according to:
        //
        // H^2 = H0^2 * [ Om_M a^-3 + Om_k a^-2 +
        //                Om_DE exp (-3 [ (1+w+wa)*lna + wa*(1-a) ] ) ]
        // Ignore the H0^2 scaling
        double lna = log(a);
        double omK = 1.0 - _c.omM - _c.omV;
        double hsq = _c.omM * std::exp(-3.0 * lna) + omK * std::exp(-2.0 * lna) +
                     _c.omV * std::exp(-3.0 * ((1.0 + _c.w + _c.wa) * lna + _c.wa * (1.0 - a)));

        if (hsq <= 0.0) {
            // This can happen for very strange w, wa values with non-flat
            // cosmologies so do something semi-graceful if it does.
            std::cerr << "Invalid hsq for a = " << a << ".  hsq = " << hsq << std::endl;
            throw std::runtime_error("Negative hsq found.");
        }

        // w = int( 1/sqrt(H(z)) dz ) = int( 1/sqrt(H(a)) 1/a^2 da )
        // So we return the integrand.
        return 1.0 / (std::sqrt(hsq) * (a * a));
    }

private:
    Cosmology const &_c;
};

double Cosmology::calcW(double z) {
    // w = int( 1/sqrt(H(z)) dz ) = int( 1/sqrt(H(a)) 1/a^2 da )
    // H(a) = H0 sqrt( Om_m a^-3 + Om_k a^-2 +
    //                 Om_de exp(3 int(1+w(a') dln(a'), a'=a..1) ) )
    // For w(a) = w0 + wa(1-a), we can do the internal integral:
    // ... Om_de exp( -3(1+w0+wa) ln(a) - 3 wa(1-a) )

    math::IntRegion<double> intreg(1.0 / (1.0 + z), 1);
    W_Integrator winteg(*this);

    return int1d(winteg, intreg);
}

int main() {
    // First some integrations of a Gaussian:

    math::IntRegion<double> reg1(-1.0, 1.0);
    math::IntRegion<double> reg2(-2.0, 2.0);
    math::IntRegion<double> reg3(0.0, math::MOCK_INF);

    Gauss g01(0.0, 1.0);  // mu = 0, sigma = 1.
    Gauss g02(0.0, 2.0);  // mu = 0, sigma = 2.

    std::cout << "int(Gauss(0.0, 1.0) , -1..1) = " << int1d(g01, reg1) << std::endl;
    std::cout << "int(Gauss(0.0, 2.0) , -1..1) = " << int1d(g02, reg1) << std::endl;

    std::cout << "int(Gauss(0.0, 1.0) , -2..2) = " << int1d(g01, reg2) << std::endl;
    std::cout << "int(Gauss(0.0, 2.0) , -2..2) = " << int1d(g02, reg2) << std::endl;

    std::cout << "int(Gauss(0.0, 1.0) , 0..inf) = " << int1d(g01, reg3) << std::endl;
    std::cout << "int(Gauss(0.0, 2.0) , 0..inf) = " << int1d(g02, reg3) << std::endl;

    math::IntRegion<double> reg4(0.0, 1.0);
    std::cout << "\nint(x*(3*x+y)+y, 0..1, 0..1) = " << int2d(std::ptr_fun(foo), reg4, reg4) << std::endl;

    std::cout << "\nIn a universe with:\n\n";
    std::cout << "Omega_m = 0.3\n";
    std::cout << "Omega_v = 0.65\n";
    std::cout << "w = -0.9\n";
    std::cout << "wa = 0.2\n";
    std::cout << "\nThe w(z) relation is:\n\n";
    std::cout << "z\tw\n\n";
    Cosmology c(0.3, 0.65, -0.9, 0.2);
    for (double z = 0.0; z < 5.01; z += 0.2) {
        std::cout << z << "\t" << c.calcW(z) << std::endl;
    }
}
