
#include <iostream>

#include "lsst/afw/geom/Angle.h"

namespace lsst {
namespace afw {
namespace geom {

template <typename T>
double operator/(T const lhs, Angle rhs) {
    static_assert((sizeof(T) == 0), "You may not divide by an Angle");
    return 0.0;
}

std::ostream& operator<<(std::ostream& s, Angle a) { return s << static_cast<double>(a) << " rad"; }
}
}
}
