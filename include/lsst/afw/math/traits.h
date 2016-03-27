/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#if !defined(LSST_AFW_MATH_KERNEL_TRAITS_H)
#define LSST_AFW_MATH_KERNEL_TRAITS_H 1

#include "boost/mpl/bool.hpp"
/**
 * @file
 * @brief Traits to describe kernels, allowing for compile-time optimisation
 */
namespace lsst {
namespace afw {
namespace math {
    class AnalyticKernel;

    //! \brief traits class to determine if a Kernel is represented as an analytic function
    template<typename KernelT>
    struct is_analyticKernel {
        BOOST_STATIC_CONSTANT(bool, value=false);
    };

    template<typename KernelT>
    struct is_analyticKernel<KernelT *> : public is_analyticKernel<KernelT> {
    };

    template<typename KernelT>
    struct is_analyticKernel<boost::shared_ptr<KernelT> > : public is_analyticKernel<KernelT> {
    };

    template<>
    struct is_analyticKernel<AnalyticKernel> {
        BOOST_STATIC_CONSTANT(bool, value=true);
    };

/************************************************************************************************************/

    /// \brief Tags carrying information about Kernels
    struct generic_kernel_tag {generic_kernel_tag(){}};        ///< Kernel with no special properties
    struct deltafunction_kernel_tag : public generic_kernel_tag {deltafunction_kernel_tag(){}}; ///< Kernel has only one non-zero pixel

    /// \brief template trait class with information about Kernels
    template<typename KernelT>
    struct kernel_traits {
        typedef typename KernelT::kernel_fill_factor kernel_fill_factor; ///< Fraction of non-zero pixels
    };

    extern generic_kernel_tag generic_kernel_tag_;
    extern deltafunction_kernel_tag deltafunction_kernel_tag_;
}}} // lsst::afw::math

#endif
