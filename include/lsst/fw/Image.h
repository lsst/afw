// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Image.h
//  Implementation of the Class Image
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_IMAGE_H
#define LSST_IMAGE_H

#include <vw/Image.h>
#include <vw/Math/BBox.h>
#include <boost/shared_ptr.hpp>
#include <list>
#include <map>
#include <string>

#include "lsst/fw/LsstBase.h"
#include "lsst/fw/DataProperty.h"
#include "lsst/fw/LSSTFitsResource.h"



namespace lsst {

    using namespace vw;
    using namespace std;

    template<typename ImagePixelT>
    class Image : private fw::LsstBase {
    public:
        typedef typename PixelChannelType<ImagePixelT>::type ImageChannelT;
        typedef ImageView<ImagePixelT> ImageIVwT;
        typedef boost::shared_ptr<Image<ImagePixelT> > ImagePtrT;
        typedef boost::shared_ptr<ImageIVwT> ImageIVwPtrT;

        
        Image();

        Image(ImageIVwPtrT image);

        Image(int ncols, int nrows);
        
        void readFits(const string& fileName, int hdu=0);

        void writeFits(const string& fileName);

#if 0                                   // not implemented
        ImagePtrT getSubImage(BBox2i maskRegion);

        void replaceSubImage(BBox2i maskRegion, Image<ImagePixelT>& insertImage);
        ImageChannelT operator ()(int x, int y) const;

        bool operator ()(int x, int y, int plane) const;
#endif

        Image<ImagePixelT>& operator += (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator -= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator *= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator /= (const Image<ImagePixelT>& inputImage);

#if 0                                   // not implemented
        int getImageCols() const;

        int getImageRows() const;
#endif

        ImageIVwPtrT getIVwPtr() const;

//         virtual ~Image();

    private:
        ImageIVwPtrT _imagePtr;
        ImageIVwT& _image;
        int _imageRows;
        int _imageCols;
        DataProperty::DataPropertyPtrT _metaData;
    };
  
#include "Image.cc"

} // namespace lsst

#endif // LSST_Image_H


