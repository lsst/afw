// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008 - 2012 LSST Corporation.
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

#include <cstdint>

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
INSTANTIATE(std::uint16_t)
/// \endcond


}}}} // namespace lsst::afw::gpu::detail ends

#endif

