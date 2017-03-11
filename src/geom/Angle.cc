
#include <iostream>

#include "lsst/afw/geom/Angle.h"

namespace lsst {
namespace afw {
namespace geom {

std::ostream& operator<<(std::ostream& s, Angle a) { return s << static_cast<double>(a) << " rad"; }
}
}
}
