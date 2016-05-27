
#include <iostream>

#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace geom {

template<typename T>
double operator /(T const lhs, Angle const rhs) {
    static_assert((sizeof(T) == 0), "You may not divide by an Angle");
    return 0.0;
}

std::ostream& operator<<(std::ostream &s, ///< The output stream
                         Angle const a    ///< The angle
						 ) {
    return s << static_cast<double>(a) << " rad";
}

}}}
