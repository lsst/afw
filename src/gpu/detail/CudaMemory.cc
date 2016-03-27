// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Functions to help managing setup for GPU kernels
 *
 * Functions in this file are used to query GPu device, allocate necessary buffers,
 * transfer data from and to GPU memory, and to set up and perform convolution.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifdef GPU_BUILD

#include <cuda.h>
#include <cuda_runtime.h>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/detail/GpuBuffer2D.h"
#include "lsst/afw/gpu/detail/CudaMemory.h"

using namespace std;

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

template<typename T>
int GpuMemOwner<T>::TransferFromImageBase(const lsst::afw::image::ImageBase<T>& img)
{
    const T* imgPtr=img.getArray().getData();
    int const imgStride=img.getArray().getStrides()[0] ;
    int const imgMemSize=imgStride * img.getHeight();
    Transfer(imgPtr,imgMemSize);
    return imgStride;
}

template<typename T>
int GpuMemOwner<T>::AllocImageBaseBuffer(const lsst::afw::image::ImageBase<T>& img)
{
    int const imgStride=img.getArray().getStrides()[0] ;
    int const imgMemSize=imgStride * img.getHeight();
    Alloc(imgMemSize);
    return imgStride;
}

template<typename T>
void GpuMemOwner<T>::CopyToImageBase(lsst::afw::image::ImageBase<T>& img) const
{
    T* imgPtr=img.getArray().getData();
    int const imgStride=img.getArray().getStrides()[0] ;
    int const imgMemSize=imgStride * img.getHeight();
    assert(imgMemSize==size);
    CopyFromGpu(imgPtr);
}

//
// Explicit instantiations
//
/// \cond
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(PIXELT) \
    template class GpuMemOwner<PIXELT>;

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(boost::uint16_t)
/// \endcond


}}}} // namespace lsst::afw::gpu::detail ends

#endif

