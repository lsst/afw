// -*- lsst-c++ -*-
/**
 * \file
 * \brief A boolean pair class used to express the output of spatial predicates on Point and Extent.
 */
#ifndef LSST_AFW_GEOM_COORDINATEEXPR_H
#define LSST_AFW_GEOM_COORDINATEEXPR_H

#include "lsst/afw/geom/CoordinateBase.h"

namespace lsst { namespace afw { namespace geom {

/**
 *  \brief A boolean coordinate.
 *
 *  CoordinateExpr is intended to be used as a temporary in coordinate comparisons:
 *  \code
 *  Point2D a(3.5,1.2);
 *  Point2D b(-1.5,4.3);
 *  std::cout << all(a < b) << std::endl;  // false
 *  std::cout << any(a < b) << std::endl;  // true
 *  \endcode
 *  
 *  CoordinateExpr is not a true lazy-evaluation expression template, as that seems unnecessary when
 *  the object is only two bools large (smaller than the raw pointers necessary to implement a lazy
 *  solution).  The consequence is that there's no short-circuiting of logical operators, but I don't
 *  think that will even remotely matter for most use cases.  The any() and all() functions do support
 *  short-circuiting.
 */
template<int N>
class CoordinateExpr : public CoordinateBase<CoordinateExpr<N>,bool,N> {
public:

    /**
     *  \brief Constructors
     *
     *  See the CoordinateBase constructors for more discussion.
     */
    //@{
    explicit CoordinateExpr(bool val=false);

    template <typename Vector>
    explicit CoordinateExpr(Eigen::MatrixBase<Vector> const & vector);
    //@}

    /**
     *  @name Equality comparison
     *
     *  Unlike other boolean operators, I propose returning a scalar here; testing for equality
     *  generally implies testing for complete equality, not element-wise equality.
     */
    //@{
    bool operator==(const CoordinateExpr& rhs) const;
    bool operator!=(const CoordinateExpr& rhs) const;
    //@}

    /**
     *  @name Logical operators
     *
     *  Interopability with scalars for these operators, if desired, should probably be provided by a
     *  non-explicit constructor from scalar, since that's really what operator interopability
     *  implies.
     */
    //@{
    CoordinateExpr operator&&(CoordinateExpr const & rhs) const;
    CoordinateExpr operator||(CoordinateExpr const & rhs) const;
    CoordinateExpr operator!() const;
    //@}

    /**
     *  @name Reductions
     *
     *  I propose these as free functions rather than member functions both to match the numpy interface
     *  and because I find the syntax 'all(a > b)' much nicer than '(a > b).all()'.  Either one works,
     *  of course.
     */
    friend inline bool all(CoordinateExpr const & expr);
    friend inline bool any(CoordinateExpr const & expr);

};

}}}

#endif
