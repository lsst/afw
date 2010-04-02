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
    enum SliceType {ROW, COLUMN};
    
    explicit Slice(Image<PixelT> &img);
    ~Slice() {}
    SliceType getSliceType() { return _sliceType; }
    
private:
    SliceType _sliceType;
};
    

    
namespace details {


template<typename PixelT>
struct Plus     { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix + slcPix; } };
template<typename PixelT>
struct Minus    { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix - slcPix; } };
template<typename PixelT>
struct MinusInv { PixelT operator()(PixelT imgPix, PixelT slcPix) { return slcPix - imgPix; } };
template<typename PixelT>
struct Mult     { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix * slcPix; } };
template<typename PixelT>
struct Div      { PixelT operator()(PixelT imgPix, PixelT slcPix) { return imgPix / slcPix; } };
template<typename PixelT>
struct DivInv   { PixelT operator()(PixelT imgPix, PixelT slcPix) { return slcPix / imgPix; } };
    
template<typename OperatorT, typename PixelT>
void operate(Image<PixelT> &img, Slice<PixelT> const &slc,
             typename Slice<PixelT>::SliceType sliceType) {
    
    OperatorT op;
    
    if (sliceType == Slice<PixelT>::ROW) {
	for (int y = 0; y < img.getHeight(); ++y) {
	    typename Slice<PixelT>::x_iterator pSlc = slc.row_begin(0);
	    for (typename Image<PixelT>::x_iterator pImg = img.row_begin(y), end = img.row_end(y);
		 pImg != end; ++pImg, ++pSlc) {
		*pImg = op(*pImg, *pSlc);
	    }
	}
    } else if (sliceType == Slice<PixelT>::COLUMN) {

	typename Slice<PixelT>::y_iterator pSlc = slc.col_begin(0);
	for (int y = 0; y < img.getHeight(); ++y, ++pSlc) {
	    for (typename Image<PixelT>::x_iterator pImg = img.row_begin(y), end = img.row_end(y);
		 pImg != end; ++pImg) {
		*pImg = op(*pImg, *pSlc);
	    }
	}
    }
    
}
}


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// overload +
template<typename PixelT>    
typename Image<PixelT>::Ptr operator+(Image<PixelT> &img, Slice<PixelT> &slc);

template<typename PixelT>    
typename Image<PixelT>::Ptr operator+(Slice<PixelT> &slc, Image<PixelT> &img);

template<typename PixelT>    
void operator+=(Image<PixelT> &img, Slice<PixelT> &slc);


// -----------------------------------------------------------------
// overload -
template<typename PixelT>    
typename Image<PixelT>::Ptr operator-(Image<PixelT> &img, Slice<PixelT> &slc);

template<typename PixelT>
void operator-=(Image<PixelT> &img, Slice<PixelT> &src);

    
// ******************************************************************
// overload *
template<typename PixelT>    
typename Image<PixelT>::Ptr operator*(Image<PixelT> &img, Slice<PixelT> &slc);
    
template<typename PixelT>    
typename Image<PixelT>::Ptr operator*(Slice<PixelT> &slc, Image<PixelT> &img);

template<typename PixelT>    
void operator*=(Image<PixelT> &img, Slice<PixelT> &slc);


// ///////////////////////////////////////////////////////////////////
// overload /
template<typename PixelT>    
typename Image<PixelT>::Ptr operator/(Image<PixelT> &img, Slice<PixelT> &slc);

template<typename PixelT>    
void operator/=(Image<PixelT> &img, Slice<PixelT> &slc);

        
}}}

#endif
