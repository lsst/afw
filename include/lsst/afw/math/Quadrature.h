// -*- lsst-c++ -*-
#if !defined(LSST_AFW_MATH_QUADRATURE_H)
#define LSST_AFW_MATH_QUADRATURE_H 1

#include <iostream>
#include <functional>
#include <string>
#include <sstream>
#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"

namespace ex = lsst::pex::exceptions;

namespace lsst { namespace afw { namespace math {

/**
 * @file    Quadrature.h
 * @brief   Compute romberg integral and double integral.
 * @ingroup afw/math
 * @author  Steve Bickerton
 * @date    May 25, 2009
 *
 * Outline:
 *
 * The Quadrature::romberg2D() function wraps the function to be integrated in a
 * function object which calls Quadrature::romberg().  romberg2D() then calls romberg()
 * with the wrapped function object.  In essence, romberg() is calling itself
 * recusively to compute the double integral.  To operate this way, the
 * integrand must have a common format, as defined in this IntegrandBase base
 * class.
 *
 */


/**
 *
 * This polynomial interpolator is based the Lagrange formula described
 * in NR 3rd ed. p 118.  It uses no actual NR code, but does not take advantage
 * of neville's algorithm.
 * I was included to see if extrapolation of the romberg solutions improved efficiency
 * but romberg's algorithm usually converged in fewer than 5 terms (suitable for a 4th order polynom).
 *
 */

/*
 *double poly_interp(double x, int const order, std::vector<double> xx, std::vector<double> yy, int const i0) {
 *  
 *  std::vector<double> coeffs(order + 1, 1.0);
 *
 *  double result = 0.0;
 *    for(int i = 0; i < order + 1; ++i) {
 *	for (int j = 0; j < order + 1; ++j) {
 *	    if ( j != i ){
 *		coeffs[i] *= (x - xx[i0 + j])/(xx[i0 + i]-xx[i0 + j]);
 *	    }
 *	}
 *	result += coeffs[i] * yy[i0+i];
 *	printf("%.4f %.4f\n", xx[i0+i], yy[i0+i]);
 *    }
 *    return result;
 *}
 */

// =============================================================
/**
 * @brief The 1D Romberg integrator
 *
 * @note This code is adapted from the standard romberg algorithm,
 *       a perfectly adequate description of which can be found at:
 *       http://en.wikipedia.org/wiki/Romberg%27s_method
 *
 */
template<typename UnaryFunctionT>
typename UnaryFunctionT::result_type romberg(UnaryFunctionT func,
					     typename UnaryFunctionT::argument_type const a,
					     typename UnaryFunctionT::argument_type const b,
					     double eps=1.0e-6)  {

    typedef typename UnaryFunctionT::argument_type Argtype;

    int const max_steps = 20;
    //int const poly_order = 5;  // can include if we want to extrapolate

    // create an R matrix to hold the successive trapezoids and romberg expansions
    std::vector<std::vector<Argtype> > R(1, std::vector<Argtype>(1));

    // these two vectors could be constants but 
    // vectors will allow the solution to be extrapolate to stepsize = 0
    // (a currently unused functionality)
    std::vector<Argtype> h(max_steps);
    std::vector<Argtype> results(max_steps);

    h[0] = b - a;
    R[0][0] = 0.5*h[0]*(func(a) + func(b));

    Argtype tolerance = 1.0;  // a relative tolerance
    //Argtype extrapolated_result_prev = func(a);
    int j = 1, m = 1;
    while ( j < max_steps && eps < tolerance ) {

	// push on a new vector to hold the next row
	R.push_back(std::vector<Argtype>(j, 0.0));
	
	// do the trapezoid to get 0-entry for the j-1 row
	h[j] = 0.5*h[j-1];
	Argtype sum = 0.0;
	for (int k = 0; k < m; ++k) {
	    sum += func(a + h[j]*(2*(k+1) - 1));
	}
	R[j][0] = 0.5*R[j - 1][0] + h[j]*sum;
	m = 2*m;

	// do the romberg summation
	for (int k = 0; k < j-1; ++k) {
	    R[j][k+1] = R[j][k] + (R[j][k] - R[j-1][k]) / (std::pow(4.0, k+1) - 1);
	}
	results[j-1] = R[j][j-1];

	// could extrapolate to stepsize here, but not currently implemented
	//Argtype extrapolated_result = results[j-1];
	//if (j > poly_order + 2) {
	//    extrapolated_result = poly_interp(0.0, poly_order, h, results, j-poly_order-2);
	//}
	//tolerance = std::fabs(extrapolated_result - extrapolated_result_prev);
	//extrapolated_result_prev = extrapolated_result;

	tolerance = std::fabs(R[j][j-1] - R[j-1][j-2]);

	j++;
    }
    
    if (j >= max_steps) {
        throw LSST_EXCEPT(ex::RuntimeErrorException,"Exceed max number of steps in Romberg integration.");
    }
    return results[j-2];
    //return extrapolated_result_prev;

}



namespace details {            


/**
 * @class FunctionWrapper
 *
 * @brief Wrap an integrand in a call to a 1D integrator: romberg()
 *
 * When romberg2D() is called, it wraps the integrand it was given
 * in a FunctionWrapper functor.  This wrapper calls romberg() on the integrand
 * to get a 1D (along the x-coord, for constant y) result .
 * romberg2D() then calls romberg() with the FunctionWrapper functor as an
 * integrand.
 *
 * @author S. Bickerton (adapted from RHL's SDSS C code)
 */
template<typename BinaryFunctionT>
class FunctionWrapper :
    public std::unary_function<typename BinaryFunctionT::second_argument_type, typename BinaryFunctionT::result_type> {
public:
    FunctionWrapper(BinaryFunctionT func,
                    typename BinaryFunctionT::first_argument_type const x1,
                    typename BinaryFunctionT::first_argument_type const x2,
                    double const eps=1.0e-6) :
        _func(func), _x1(x1), _x2(x2), _eps(eps) {}
    typename BinaryFunctionT::result_type operator() (typename BinaryFunctionT::second_argument_type const y) {
        return romberg(std::bind2nd(_func, y), _x1, _x2, _eps);
    }
private:
    BinaryFunctionT _func;
    typename BinaryFunctionT::first_argument_type _x1, _x2;
    double _eps;
};

} // end of namespace afw::math::details





// =============================================================
/**
 * @brief The 2D Romberg integrator
 *
 * @note Adapted from RHL's SDSS code
 *
 */

template<typename BinaryFunctionT>
typename BinaryFunctionT::result_type romberg2D(BinaryFunctionT func,
                                       typename BinaryFunctionT::first_argument_type const x1,
                                       typename BinaryFunctionT::first_argument_type const x2,
                                       typename BinaryFunctionT::second_argument_type const y1,
                                       typename BinaryFunctionT::second_argument_type const y2,
                                       double eps=1.0e-6) {
    using namespace details;
    
    // note the more stringent eps requirement to ensure the requested limit
    // can be reached.
    FunctionWrapper<BinaryFunctionT> fwrap(func, x1, x2, 1.0e-2*eps);
    return romberg(fwrap, y1, y2, eps);
}

}}}
    
#endif
