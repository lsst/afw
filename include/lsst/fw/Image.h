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
            typedef typename vw::ImageView<ImagePixelT>::pixel_accessor pixel_accessor;
            
            Image();
            
            Image(ImageIVwPtrT image);
            
            Image(int ncols, int nrows);

            Image& operator=(const Image& image);

            void readFits(const string& fileName, int hdu=0);
            
            void writeFits(const string& fileName) const;
            
            DataPropertyPtrT getMetaData() const;
            
            ImagePtrT getSubImage(const BBox2i imageRegion) const;
            
            void replaceSubImage(const BBox2i imageRegion, ImagePtrT insertImage);

            ImageChannelT operator ()(int x, int y) const;

            pixel_accessor origin() const { return getIVwPtr()->origin(); }
            
            Image<ImagePixelT>& operator += (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator -= (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator *= (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator /= (const Image<ImagePixelT>& inputImage);
            Image<ImagePixelT>& operator += (const ImagePixelT scalar);
            Image<ImagePixelT>& operator -= (const ImagePixelT scalar);
            Image<ImagePixelT>& operator *= (const ImagePixelT scalar);
            Image<ImagePixelT>& operator /= (const ImagePixelT scalar);
            
            unsigned int getCols() const;
            unsigned int getRows() const;
            unsigned int getOffsetCols() const;
            unsigned int getOffsetRows() const;
            
            ImageIVwPtrT getIVwPtr() const;
            
            ImageIVwT& getIVw() const;

            double getGain() const;
            
//         virtual ~Image();
            
        private:
            ImageIVwPtrT _imagePtr;
            ImageIVwT& _image;
            DataPropertyPtrT _metaData;
            unsigned int _offsetRows;
            unsigned int _offsetCols;

            void setOffsetRows(unsigned int offset);
            void setOffsetCols(unsigned int offset);

        };
  
#include "Image.cc"
        
    } // namespace fw

} // namespace lsst

#endif // LSST_Image_H


