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

using namespace vw;
using namespace std;

namespace lsst {

    template<typename ImagePixelT>
    class Image
    {
    public:
        typedef typename PixelChannelType<ImagePixelT>::type ImageChannelT;
        typedef ImageView<ImagePixelT> ImageIVwT;
        typedef boost::shared_ptr<Image<ImagePixelT> > ImagePtrT;
        typedef boost::shared_ptr<ImageIVwT> ImageIVwPtrT;

        
        Image();

        Image(ImageIVwPtrT image);

        Image(int ncols, int nrows);
        
        ImagePtrT getSubImage(BBox2i maskRegion);

        void replaceSubImage(BBox2i maskRegion, Image<ImagePixelT>& insertImage);

        ImageChannelT operator ()(int x, int y) const;

        bool operator ()(int x, int y, int plane) const;

        Image<ImagePixelT>& operator += (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator -= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator *= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator /= (const Image<ImagePixelT>& inputImage);

        int getImageCols() const;

        int getImageRows() const;

        ImageIVwPtrT getIVwPtr() const;

//         virtual ~Image();

    private:
        ImageIVwPtrT _imagePtr;
        ImageIVwT& _image;
        int _imageRows;
        int _imageCols;
    };
  
#include "Image.cc"

} // namespace lsst

#endif // LSST_Image_H


