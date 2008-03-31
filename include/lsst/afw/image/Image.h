// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Image.h
//  Implementation of the Class Image
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_IMAGE_IMAGE_H
#define LSST_IMAGE_IMAGE_H

#include <list>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>
#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include <lsst/daf/base.h>
#include <lsst/daf/data/LsstBase.h>

namespace lsst {
namespace afw {
    namespace formatters {
        template <typename ImagePixelT> class ImageFormatter;
    }
namespace image {
    template<typename ImagePixelT>
    class Image : public lsst::daf::base::Persistable,
                  public lsst::daf::data::LsstBase {
    public:
        typedef typename vw::PixelChannelType<ImagePixelT>::type ImageChannelT;
        typedef vw::ImageView<ImagePixelT> ImageIVwT;
        typedef boost::shared_ptr<Image<ImagePixelT> > ImagePtrT;
        typedef boost::shared_ptr<ImageIVwT> ImageIVwPtrT;
        typedef typename vw::ImageView<ImagePixelT>::pixel_accessor pixel_accessor;
        
        Image();
        
        Image(ImageIVwPtrT image);
        
        Image(int ncols, int nrows);

        Image& operator=(const Image& image);

        void readFits(const std::string& fileName, int hdu=0);
        
        void writeFits(const std::string& fileName) const;
        
        lsst::daf::base::DataProperty::PtrType getMetaData() const;
        
        ImagePtrT getSubImage(const vw::BBox2i imageRegion) const;
        
        void replaceSubImage(const vw::BBox2i imageRegion, ImagePtrT insertImage);

        inline ImageChannelT operator ()(int x, int y) const;

        inline pixel_accessor origin() const;
        
        Image<ImagePixelT>& operator += (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator -= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator *= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator /= (const Image<ImagePixelT>& inputImage);
        Image<ImagePixelT>& operator += (const ImagePixelT scalar);
        Image<ImagePixelT>& operator -= (const ImagePixelT scalar);
        Image<ImagePixelT>& operator *= (const ImagePixelT scalar);
        Image<ImagePixelT>& operator /= (const ImagePixelT scalar);
        
        inline unsigned int getCols() const;
        inline unsigned int getRows() const;
        inline unsigned int getOffsetCols() const;
        inline unsigned int getOffsetRows() const;
        
        inline ImageIVwPtrT getIVwPtr() const;
        
        inline ImageIVwT& getIVw() const;

        double getGain() const;
        
//        virtual ~Image();
        
    private:
        LSST_PERSIST_FORMATTER(formatters::ImageFormatter<ImagePixelT>);
        ImageIVwPtrT _vwImagePtr;
        lsst::daf::base::DataProperty::PtrType _metaData;
        unsigned int _offsetRows;
        unsigned int _offsetCols;

        inline void setOffsetRows(unsigned int offset);
        inline void setOffsetCols(unsigned int offset);

    };

}}}  // lsst::afw::image

// Included definitions for templated and inline member functions
#ifndef SWIG // don't bother SWIG with .cc files
#include "Image.cc"
#endif

#endif // LSST_IMAGE_IMAGE_H
