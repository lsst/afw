// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief contains GpuBuffer2D class (for simple handling of images or 2D arrays)
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

/**
 * @brief Class for representing an image or 2D array in general)
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
class GpuBuffer2D
{
public:
    typedef lsst::afw::image::Image<PixelT>  ImageT;

    PixelT*     img;
    int         width;
    int         height;

    GpuBuffer2D() : img(NULL) {}

    //copying is not allowed except for uninitialized image buffers
    GpuBuffer2D(const GpuBuffer2D& x) {
        assert(x.img == NULL);
        img = NULL;
    };

    void Init(const ImageT& image)
    {
        assert(img == NULL);
        this->width = image.getWidth();
        this->height = image.getHeight();
        try {
            img = new PixelT [width*height];
        } catch(...) {
            throw LSST_EXCEPT(pexExcept::MemoryError, "GpuBuffer2D:Init - not enough memory");
        }

        //copy input image data to buffer
        for (int i = 0; i < height; ++i) {
            typename ImageT::x_iterator inPtr = image.x_at(0, i);
            PixelT*                  imageDataPtr = img + i * width;

            for (typename ImageT::x_iterator cnvEnd = inPtr + width; inPtr != cnvEnd;
                    ++inPtr, ++imageDataPtr) {
                *imageDataPtr = *inPtr;
            }
        }
    }

    void Init(int width, int height) {
        assert(img == NULL);
        this->width = width;
        this->height = height;
        try {
            img = new PixelT [width*height];
        } catch(...) {
            throw LSST_EXCEPT(pexExcept::MemoryError, "GpuBuffer2D:Init - not enough memory");
        }
    }

    GpuBuffer2D(const ImageT& image) {
        img = NULL;
        Init(image);
    }

    GpuBuffer2D(int width, int height) {
        img = NULL;
        Init(width, height);
    }

    ~GpuBuffer2D() {
        delete[] img;
    }

    int Size() const {
        return width * height;
    }

    PixelT* GetImgLinePtr(int y) {
        assert(img != NULL);
        assert(y >= 0 && y < height);
        return &img[width*y];
    }
    const PixelT* GetImgLinePtr(int y) const {
        assert(img != NULL);
        assert(y >= 0 && y < height);
        return &img[width*y];
    }
    PixelT& Pixel(int x, int y) {
        assert(img != NULL);
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return img[x+ y*width];
    }
    const PixelT& Pixel(int x, int y) const {
        assert(img != NULL);
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return img[x+ y*width];
    }

    void CopyFromBuffer(const GpuBuffer2D<PixelT>& buffer, int startX, int startY)
    {
        assert(img != NULL);
        for (int i = 0; i < height; ++i) {
            PixelT* inPtr = startX + buffer.GetImgLinePtr(i + startY);
            PixelT* outPtr = buffer.GetImgLinePtr(i);
            for (int j = 0; j < width; j++) {
                *outPtr = *inPtr;
                inPtr++;
                outPtr++;
            }
        }
    }

    void CopyToImage(ImageT outImage, int startX, int startY)
    {
        assert(img != NULL);
        for (int i = 0; i < height; ++i) {
            PixelT*  outPtrImg = &img[width*i];

            for (typename ImageT::x_iterator cnvPtr = outImage.x_at(startX, i + startY),
                    cnvEnd = cnvPtr + width;    cnvPtr != cnvEnd;    ++cnvPtr ) {
                *cnvPtr = *outPtrImg;
                ++outPtrImg;
            }
        }
    }

};

}
}
}
} //namespace lsst::afw::math::detail ends

