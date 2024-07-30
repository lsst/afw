#include "pybind11/pybind11.h"
#include <lsst/cpputils/python.h>

namespace lsst {
namespace afw {
namespace math {
namespace detail {
void wrapConvolve(lsst::cpputils::python::WrapperCollection &);
void wrapSpline(lsst::cpputils::python::WrapperCollection &);

PYBIND11_MODULE(_detail, mod) {
    lsst::cpputils::python::WrapperCollection wrappers(mod, "lsst.afw.math.detail");
    wrapConvolve(wrappers);
    wrapSpline(wrappers);
    wrappers.finish();
}
}  // namespace detail
}  // namespace math
}  // namespace afw
}  // namespace lsst