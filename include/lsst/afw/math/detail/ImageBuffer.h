// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/**
 * @file
 *
 * @brief contains ImageBuffer class (for simple handling of images)
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#include "convCUDA.h"

namespace lsst {
    namespace afw {
        namespace math {
            namespace detail {

/**
 * @brief Class for representing an image buffer
 *
 * Allocates width*height pixels memory for image. Automatically allocates and
 * releases memory for buffer (this class is the owner of the buffer).
 *
 * Can be uninitialized. Only uninitialized image buffer can be copied.
 * For copying initialized image buffers, use CopyFromBuffer member function.
 * Provides access to pixels and lines in image
 * Can be copied to and from the Image.
 *
 * @ingroup afw
 */
template <typename PixelT>
class ImageBuffer
{
public:
        typedef lsst::afw::image::Image<PixelT>  ImageT;

    PixelT*     img;
    int         width;
    int         height;

    ImageBuffer() : img(NULL) {}

    //copying is not allowed except for uninitialized image buffers
    ImageBuffer(const ImageBuffer& x) {
        assert(x.img==NULL);
        img=NULL;
        };

    void Init(ImageT const& image)
    {
        assert(img==NULL);
        this->width=image.getWidth();
        this->height=image.getHeight();
        try {
            img = new PixelT [width*height];
            }
        catch(...) {
            throw LSST_EXCEPT(pexExcept::MemoryException, "ImageBuffer:Init - not enough memory");
            }

        //copy input image data to buffer
        for (int i = 0; i < height; ++i) {
            typename ImageT::x_iterator inPtr = image.x_at(0, i);
            PixelT*                  imageDataPtr = img + i*width;

            for (typename ImageT::x_iterator cnvEnd = inPtr + width; inPtr != cnvEnd;
                    ++inPtr, ++imageDataPtr){
                    *imageDataPtr = *inPtr;
                }
            }
    }

    void Init(int width, int height) {
        assert(img==NULL);
        this->width=width;
        this->height=height;
        try {
            img = new PixelT [width*height];
            }
        catch(...) {
            throw LSST_EXCEPT(pexExcept::MemoryException, "ImageBuffer:Init - not enough memory");
            }
        }

    ImageBuffer(ImageT const& image){
        img=NULL;
        Init(image);
        }

    ImageBuffer(int width, int height){
        img=NULL;
        Init(width,height);
        }

    ~ImageBuffer(){
        delete[] img;
        }

    int Size() const { return width*height; }

    PixelT* GetImgLinePtr(int y) {
        assert(img!=NULL);
        assert(y>=0 && y<height);
        return &img[width*y];
        }
    const PixelT* GetImgLinePtr(int y) const {
        assert(img!=NULL);
        assert(y>=0 && y<height);
        return &img[width*y];
        }
    PixelT& Pixel(int x, int y){
        assert(img!=NULL);
        assert(x>=0 && x<width);
        assert(y>=0 && y<height);
        return img[x+ y*width];
        }
    const PixelT& Pixel(int x, int y) const {
        assert(img!=NULL);
        assert(x>=0 && x<width);
        assert(y>=0 && y<height);
        return img[x+ y*width];
        }

    void CopyFromBuffer(ImageBuffer<PixelT>& buffer, int startX, int startY)
    {
        assert(img!=NULL);
        for (int i = 0; i < height; ++i) {
            PixelT* inPtr = startX + buffer.GetImgLinePtr(i+startY);
            PixelT* outPtr = buffer.GetImgLinePtr(i);
            for (int j=0; j<width;j++) {
                *outPtr = *inPtr;
                inPtr++;
                outPtr++;
                }
            }
    }

    void CopyToImage(ImageT outImage, int startX, int startY)
    {
        assert(img!=NULL);
        for (int i = 0; i < height; ++i) {
            PixelT*  outPtrImg = &img[width*i];

            for (typename ImageT::x_iterator cnvPtr = outImage.x_at(startX, i + startY),
                    cnvEnd = cnvPtr + width;    cnvPtr != cnvEnd;    ++cnvPtr )
                {
                *cnvPtr = *outPtrImg;
                ++outPtrImg;
                }
            }
       }

};

}}}} //namespace lsst::afw::math::detail ends

