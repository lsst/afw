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

#if !defined(LSST_AFW_MATH_KERNEL_TRAITS_H)
#define LSST_AFW_MATH_KERNEL_TRAITS_H 1

#include "boost/mpl/bool.hpp"
/*
 * Traits to describe kernels, allowing for compile-time optimisation
 */
namespace lsst {
namespace afw {
namespace math {
class AnalyticKernel;

/// traits class to determine if a Kernel is represented as an analytic function
template <typename KernelT>
struct is_analyticKernel {
    BOOST_STATIC_CONSTANT(bool, value = false);
};

template <typename KernelT>
struct is_analyticKernel<KernelT *> : public is_analyticKernel<KernelT> {};

template <typename KernelT>
struct is_analyticKernel<std::shared_ptr<KernelT> > : public is_analyticKernel<KernelT> {};

template <>
struct is_analyticKernel<AnalyticKernel> {
    BOOST_STATIC_CONSTANT(bool, value = true);
};

/// Tags carrying information about Kernels
struct generic_kernel_tag {
    generic_kernel_tag() {}
};  ///< Kernel with no special properties
struct deltafunction_kernel_tag : public generic_kernel_tag {
    deltafunction_kernel_tag() {}
};  ///< Kernel has only one non-zero pixel

/// template trait class with information about Kernels
template <typename KernelT>
struct kernel_traits {
    typedef typename KernelT::kernel_fill_factor kernel_fill_factor;  ///< Fraction of non-zero pixels
};

extern generic_kernel_tag generic_kernel_tag_;
extern deltafunction_kernel_tag deltafunction_kernel_tag_;
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
