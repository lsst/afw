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


    //namespace {

    template<typename PixelT>
    class Slice;

enum SliceType {ROW, COLUMN};

template<typename PixelT>
struct Plus  { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix + slcPix; } };
template<typename PixelT>
struct Minus { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix - slcPix; } };
template<typename PixelT>
struct Mult  { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix * slcPix; } };
template<typename PixelT>
struct Div   { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix / slcPix; } };

template<typename OperatorT, typename PixelT>
void operate(Image<PixelT> &img, Slice<PixelT> const &slc, 
	     SliceType sliceType) {
    
    OperatorT op;
    
    if (sliceType == ROW) {
	for (int y = 0; y < img.getHeight(); ++y) {
	    typename Slice<PixelT>::x_iterator pSlc = slc.row_begin(0);
	    for (typename Image<PixelT>::x_iterator pImg = img.row_begin(y), end = img.row_end(y);
		 pImg != end; ++pImg, ++pSlc) {
		*pImg = op(*pImg, *pSlc);
	    }
	}
    } else if (sliceType == COLUMN) {

	typename Slice<PixelT>::y_iterator pSlc = slc.col_begin(0);
	for (int y = 0; y < img.getHeight(); ++y, ++pSlc) {
	    for (typename Image<PixelT>::x_iterator pImg = img.row_begin(y), end = img.row_end(y);
		 pImg != end; ++pImg) {
		*pImg = op(*pImg, *pSlc);
	    }
	}
    }
    
}



            
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
    void operator+=(Image<PixelT> &img);
    void operator-=(Image<PixelT> &img);
    void operator*=(Image<PixelT> &img);
    void operator/=(Image<PixelT> &img);
    //    typename Image<PixelT>::Ptr operator+(Image<PixelT> &img);
    typename Image<PixelT>::Ptr operator-(Image<PixelT> &img);
    typename Image<PixelT>::Ptr operator*(Image<PixelT> &img);
    typename Image<PixelT>::Ptr operator/(Image<PixelT> &img);

    
    friend typename Image<PixelT>::Ptr operator+(Image<PixelT> &img, Slice<PixelT> &slc) {
	typename Image<PixelT>::Ptr retImg(new Image<PixelT>(img, true));
	operate<Plus<PixelT> >(*retImg, slc, slc.getSliceType());
	return retImg;
    }
    friend typename Image<PixelT>::Ptr operator+(Slice<PixelT> &slc, Image<PixelT> &img) {
	typename Image<PixelT>::Ptr retImg(new Image<PixelT>(img, true));
	operate<Plus<PixelT> >(*retImg, slc, slc.getSliceType());
	return retImg;
    }

    
    SliceType getSliceType() { return _sliceType; }
private:
    SliceType _sliceType;

};

#if 0
template<typename PixelT>
typename Image<PixelT>::Ptr operator+(Image<PixelT> &img, Slice<PixelT> &slc);
template<typename PixelT>
typename Image<PixelT>::Ptr operator+(Slice<PixelT> &slc, Image<PixelT> &img);
#endif

/********************************************************************
 *
 * row, column Operators
 *
 *********************************************************************/

template<typename ImageT>
typename ImageT::Ptr sliceOperate(ImageT &image, ImageT &slice,
				  std::string sliceType, char op, bool deep=true);

    
}}}

#endif
