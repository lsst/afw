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
#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <list>
#include <map>
#include <string>

#include "LsstBase.h"
#include "Mask.h"
#include "Image.h"

namespace lsst {
    template<class ImagePixelT, class MaskPixelT> class MaskedImage;

    template<typename PixelT> class PixelLocator : public vw::PixelIterator<vw::ImageView<PixelT> > {
    public:
        PixelLocator(vw::ImageView<PixelT>*, vw::PixelIterator<vw::ImageView<PixelT> >);
        PixelLocator<PixelT>& advance(int dx, int dy);
    private:
        unsigned _cols;
        unsigned _rows;
        unsigned _planes;
        unsigned _cstride;
        unsigned _rstride;
        unsigned _pstride;
    };


    template<typename ImagePixelT, typename MaskPixelT> class PixelProcessingFunc : 
        public std::unary_function<boost::tuple<ImagePixelT&, MaskPixelT&>&, void>
    {
    public:
        typedef boost::tuple<ImagePixelT&, MaskPixelT&> TupleT;
        typedef boost::shared_ptr<Image<ImagePixelT> > ImagePtrT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
        typedef PixelLocator<ImagePixelT> ImageIteratorT;
        typedef PixelLocator<MaskPixelT> MaskIteratorT;
        typedef vw::ImageView<ImagePixelT> * ImageViewPtrT;
        typedef vw::ImageView<MaskPixelT> * MaskViewPtrT;

        PixelProcessingFunc(MaskedImage<ImagePixelT, MaskPixelT>& m) : 
            _imagePtr(m.getImage()),
            _maskPtr(m.getMask()),
            _imageViewPtr(_imagePtr->getIVwPtr().get()),
            _maskViewPtr(_maskPtr->getIVwPtr().get()),
            _imageLocatorBegin(_imageViewPtr, _imageViewPtr->begin()),
            _imageLocatorEnd(_imageViewPtr, _imageViewPtr->end()),
            _maskLocatorBegin(_maskViewPtr, _maskViewPtr->begin()),
            _maskLocatorEnd(_maskViewPtr, _maskViewPtr->end()) {
        }
        virtual void operator () (ImageIteratorT&, MaskIteratorT&);
        virtual ~PixelProcessingFunc() {};
        ImageIteratorT & getImagePixelLocatorBegin() { return _imageLocatorBegin; }
        ImageIteratorT & getImagePixelLocatorEnd() { return _imageLocatorEnd; }
        MaskIteratorT & getMaskPixelLocatorBegin() { return _maskLocatorBegin; }
        MaskIteratorT & getMaskPixelLocatorEnd() { return _maskLocatorEnd; }
    protected:
        ImagePtrT _imagePtr;
        MaskPtrT _maskPtr;
        ImageViewPtrT _imageViewPtr;
        MaskViewPtrT _maskViewPtr;
        ImageIteratorT _imageLocatorBegin;
        ImageIteratorT _imageLocatorEnd;
        MaskIteratorT _maskLocatorBegin;
        MaskIteratorT _maskLocatorEnd;
    };

    template<class ImagePixelT, class MaskPixelT>
    class MaskedImage : private fw::LsstBase {
        
    public:
        typedef Image<ImagePixelT> ImageT;
        typedef Mask<MaskPixelT> MaskT;
        typedef boost::shared_ptr<Image<ImagePixelT> > ImagePtrT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
        typedef boost::shared_ptr<MaskedImage<ImagePixelT, MaskPixelT> > MaskedImagePtrT;
        typedef MaskedImage<ImagePixelT, MaskPixelT> MaskedImageT;

        // Constructors
	MaskedImage();
        MaskedImage(ImagePtrT image, MaskPtrT mask);
        MaskedImage(int nCols, int nRows);

        virtual ~MaskedImage();

        // Processing functions
	void processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc,
            MaskedImageT &);
            
	void processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc);

	void processPixels(PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc);

        // SubImage functions
#if 0
	MaskedImagePtrT getSubImage(BBox2i region);

	void replaceSubImage(BBox2i region, MaskedImage &sImage, bool replaceMask, bool replaceAstro);
#endif
        // Operators
	MaskedImage& operator+=( MaskedImageT& maskedImageInput);
	MaskedImage& operator-=( MaskedImage & maskedImageInput);
	MaskedImage& operator*=( MaskedImage & maskedImageInput);
	MaskedImage& operator/=( MaskedImage & maskedImageInput);
        
        // IO functions
        void readFits(std::string baseName);

        // Getters
	ImagePtrT getImage();
	MaskPtrT getMask();
    private:
        MaskedImage(const MaskedImage &);
        MaskedImage& operator=(const MaskedImage &);

	ImagePtrT _imagePtr;
	MaskPtrT _maskPtr;
        ImageT & _image;
        MaskT & _mask;
        int _imageRows;
        int _imageCols;

    };
    
#include "MaskedImage.cc"

} // namespace lsst
#endif //  LSST_MASKEDIMAGE_H
