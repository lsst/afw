#include "pybind11/pybind11.h"
#include <lsst/utils/python.h>

namespace lsst {
namespace afw {
namespace math {
namespace detail {
void wrapConvolve(lsst::utils::python::WrapperCollection &);
void wrapSpline(lsst::utils::python::WrapperCollection &);

PYBIND11_MODULE(_detail, mod) {
    lsst::utils::python::WrapperCollection wrappers(mod, "lsst.afw.math.detail");
    wrapConvolve(wrappers);
    wrapSpline(wrappers);
    wrappers.finish();
}
}  // namespace detail
}  // namespace math
}  // namespace afw
}  // namespace lsst