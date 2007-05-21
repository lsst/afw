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

    namespace fw {
        
        using namespace vw;
        using namespace std;
        
        template<typename ImagePixelT>
        class Image : private LsstBase {
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
            
            DataPropertyPtrT getMetaData();
            
            ImagePtrT getSubImage(const BBox2i imageRegion) const;
            
            void replaceSubImage(const BBox2i imageRegion, ImagePtrT insertImage);

            ImageChannelT operator ()(int x, int y) const;
            
            Image<ImagePixelT>& operator += (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator -= (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator *= (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator /= (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator += (const ImagePixelT scalar);
            Image<ImagePixelT>& operator -= (const ImagePixelT scalar);
            Image<ImagePixelT>& operator *= (const ImagePixelT scalar);
            Image<ImagePixelT>& operator /= (const ImagePixelT scalar);
            
            int getImageCols() const;
            int getImageRows() const;
            
            ImageIVwPtrT getIVwPtr() const;
            
            ImageIVwT& getIVw() const;

            double getGain() const;
            
//         virtual ~Image();
            
        private:
            ImageIVwPtrT _imagePtr;
            ImageIVwT& _image;
            DataPropertyPtrT _metaData;
        };
  
#include "Image.cc"
        
    } // namespace fw

} // namespace lsst

#endif // LSST_Image_H


