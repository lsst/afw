// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  MaskedImage.h
//  Implementation of the Class MaskedImage
//  Created on:      23-Feb-2007 16:23:06
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_MASKEDIMAGE_H
#define LSST_MASKEDIMAGE_H

#include <vw/Image.h>
#include <vw/Math/BBox.h>
#include <boost/shared_ptr.hpp>
#include <list>
#include <map>
#include <string>

#include "Mask.h"
#include "Image.h"

namespace lsst {

    template<typename ImagePixelT, typename MaskPixelT> class PixelProcessingFunc;

    template<class ImagePixelT, class MaskPixelT>
    class MaskedImage
    {
        
    public:
        typedef Image<ImagePixelT> ImageT;
        typedef Mask<MaskPixelT> MaskT;
        typedef boost::shared_ptr<Image<ImagePixelT> > ImagePtrT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
        typedef boost::shared_ptr<MaskedImage<ImagePixelT, MaskPixelT> > MaskedImagePtrT;
        typedef MaskedImage<ImagePixelT, MaskPixelT> MaskedImageT;

	MaskedImage();
            
	MaskedImage(ImagePtrT image, MaskPtrT mask);

        MaskedImage(int nCols, int nRows);

	MaskedImage returnProcessedPixels(MaskPixelBooleanFunc<MaskPixelT> selectionFunc, PixelProcessingFunc<ImagePixelT, MaskPixelT> processingFunc);
            
	void processPixels(MaskPixelBooleanFunc<MaskPixelT> selectionFunc, PixelProcessingFunc<ImagePixelT, MaskPixelT> processingFunc);

	MaskedImagePtrT getSubImage(BBox2i region);

	void replaceSubImage(BBox2i region, MaskedImage sImage, bool replaceMask, bool replaceAstro);

	MaskedImage<ImagePixelT, MaskPixelT>& operator+=( MaskedImageT& maskedImageInput);

	MaskedImage<ImagePixelT, MaskPixelT>& operator-=( MaskedImage & maskedImageInput);

	MaskedImage<ImagePixelT, MaskPixelT>& operator*=( MaskedImage & maskedImageInput);

	MaskedImage<ImagePixelT, MaskPixelT>& operator/=( MaskedImage & maskedImageInput);

	ImagePtrT getImage();

	MaskPtrT getMask();

	virtual ~MaskedImage();
        
    private:
	ImagePtrT _imagePtr;
	MaskPtrT _maskPtr;
        ImageT & _image;
        MaskT & _mask;
    };
    
#include "MaskedImage.cc"

} // namespace lsst
#endif //  LSST_MASKEDIMAGE_H
