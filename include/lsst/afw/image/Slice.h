// -*- LSST-C++ -*-
#if !defined(LSST_AFW_IMAGE_SLICE_H)
#define LSST_AFW_IMAGE_SLICE_H
/**
 * @file Slice.h
 * @brief Define a single column or row of an Image
 * @ingroup afw
 * @author Steve Bickerton
 *
 */

#include "lsst/afw/image/Image.h"

namespace math = lsst::afw::math;

namespace lsst {
namespace afw {
namespace image {
            
/**
 * @brief A class to specify a slice of an image
 * @ingroup afw
 *
 */
template<typename PixelT>
class Slice : public Image<PixelT> {
public:
    explicit Slice(Image<PixelT> &img);
    ~Slice() {}
    void operator+=(Image<PixelT> &rhs);
    typename Image<PixelT>::Ptr operator+(Image<PixelT> &rhs);
// private:
};
            
/********************************************************************
 *
 * row, column Operators
 *
 *********************************************************************/

template<typename ImageT, typename SliceT>
typename ImageT::Ptr sliceOperate(ImageT const &image, SliceT const &slice,
				  std::string sliceType, char op, bool deep=true);

    
}}}

#endif
