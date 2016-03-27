// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief CPU and GPU convolution shared code
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
	namespace afw {
		namespace math {
			namespace detail {

    template <typename OutImageT, typename InImageT>
    void assertDimensionsOK(
            OutImageT const &convolvedImage,
            InImageT const &inImage,
            lsst::afw::math::Kernel const &kernel
                                   );


}}}} //lsst::afw::math::detail


