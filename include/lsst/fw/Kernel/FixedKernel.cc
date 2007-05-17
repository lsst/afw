// -*- LSST-C++ -*-
#include <stdexcept>

// #include <lsst/fw/Kernel.h>

namespace lsst {
namespace fw {

    /**
     * Construct a blank FixedKernel of size 0x0
     */
    template<typename PixelT>
    FixedKernel<PixelT>::FixedKernel(
    ) :
        LsstBase(typeid(this)),
        Kernel<PixelT>(),
        _image()
    { }

    /**
     * Construct a FixedKernel from an image
     */
    template<typename PixelT>
    FixedKernel<PixelT>::FixedKernel(
        const Image<PixelT> &image
    ) :
        LsstBase(typeid(this)),
        Kernel<PixelT>(image.getCols(), image.getRows(), 0),
        _image(image)
    { }

    /**
     * Return the image of the kernel
     */
    template<typename PixelT>
    const Image<PixelT>
    FixedKernel<PixelT>::getImage(PixelT x, PixelT y) {
        return _image;
    }

    template<typename PixelT>
    inline const std::vector<PixelT>
    FixedKernel<PixelT>::getCurrentKernelParameters() const {
        return std::vector<PixelT>(0);
    }

    template<typename PixelT>
    inline void
    setKernelParameters(const std::vector<PixelT> params) {
        if (params.size() > 0) {
            throw std::invalid_argument("FixedKernel has no kernel parameters");
        }
    }

}   // namespace fw
}   // namespace lsst
