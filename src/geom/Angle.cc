
#include <iostream>

#include "boost/static_assert.hpp"

#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace geom {

template<typename T>
double operator /(T const lhs, Angle const rhs) {
    BOOST_STATIC_ASSERT_MSG((sizeof(T) == 0), "You may not divide by an Angle");
    return 0.0;
}

std::ostream& operator<<(std::ostream &s, ///< The output stream
                         Angle const a    ///< The angle
						 ) {
    return s << static_cast<double>(a) << " rad";
}

}}}
