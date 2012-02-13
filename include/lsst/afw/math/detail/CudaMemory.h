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
 * @brief Functions to help allocating GPU global memory and transferring data to and from a GPU
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

/* requires:
#include <cuda.h>
#include <cuda_runtime.h>
#include "lsst/afw/math/detail/ImageBuffer.h"
*/

#ifdef GPU_BUILD 

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

template<typename T>
T* AllocOnGpu(int size)
{
    T* dataGpu;
    cudaError_t cudaError = cudaMalloc((void**)&dataGpu, size * sizeof(T));
    if (cudaError != cudaSuccess) {
        return NULL;
    }
    return dataGpu;
}
template<typename T>
void CopyFromGpu(T* destCpu, T* sourceGpu, int size)
{
    cudaError_t cudaError = cudaMemcpy(
                                /* Desination:*/     destCpu,
                                /* Source:    */     sourceGpu,
                                /* Size in bytes: */ size * sizeof(T),
                                /* Direction   */    cudaMemcpyDeviceToHost
                            );
    if (cudaError != cudaSuccess)
        throw LSST_EXCEPT(GpuMemoryException, "CopyFromGpu: failed");
}
template<typename T>
void CopyToGpu(T* destGpu, T* sourceCpu, int size)
{
    cudaError_t cudaError;
    cudaError = cudaMemcpy(
                    /* Desination:*/     destGpu,
                    /* Source:    */     sourceCpu,
                    /* Size in bytes: */ size * sizeof(T),
                    /* Direction   */    cudaMemcpyHostToDevice
                );
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuMemoryException, "CopyToGpu: failed");
    }
}

template<typename T>
T* TransferToGpu(const T* sourceCpu, int size)
{
    T* dataGpu;
    cudaError_t cudaError = cudaMalloc((void**)&dataGpu, size * sizeof(T));
    if (cudaError != cudaSuccess) {
        return NULL;
    }
    cudaError = cudaMemcpy(
                    /* Desination:*/     dataGpu,
                    /* Source:    */     sourceCpu,
                    /* Size in bytes: */ size * sizeof(T),
                    /* Direction   */    cudaMemcpyHostToDevice
                );
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuMemoryException, "TransferToGpu: transfer failed");
    }
    return dataGpu;
}
/**
    A class for handling GPU memory managment and copying data to and from GPU

    Automatically releases GPU memory on destruction, simplifying GPU memory management
*/

template<typename T>
class GpuMemOwner
{
public:
    T* ptr;
    int size;
    GpuMemOwner() : ptr(NULL) {}

    T* Transfer(const T* source, int size_p) {
        assert(ptr == NULL);
        size = size_p;
        ptr = TransferToGpu(source, size);
        return ptr;
    }
    T* Transfer(const ImageBuffer<T>& source) {
        assert(ptr == NULL);
        size = source.Size();
        ptr = TransferToGpu(source.img, size);
        return ptr;
    }
    T* TransferVec(const std::vector<T>& source) {
        assert(ptr == NULL);
        size = int(source.size());
        ptr = TransferToGpu(&source[0], size);
        return ptr;
    }
    T* Alloc(int size_p)  {
        assert(ptr == NULL);
        size = size_p;
        ptr = AllocOnGpu<T>(size);
        return ptr;
    }
    T* CopyToGpu(ImageBuffer<T>& source) const {
        assert(ptr != NULL);
        assert(source.Size() == size);
        gpu::CopyToGpu(ptr, source.img, size);
        return ptr;
    }
    T* CopyFromGpu(ImageBuffer<T>& dest) const {
        assert(ptr != NULL);
        assert(dest.Size() == size);
        gpu::CopyFromGpu(dest.img, ptr, size);
        return ptr;
    }
    T* CopyFromGpu(T* dest) const {
        assert(ptr != NULL);
        gpu::CopyFromGpu(dest, ptr, size);
        return ptr;
    }

    int TransferFromImageBase(const lsst::afw::image::ImageBase<T>& img);
    int AllocImageBaseBuffer(const lsst::afw::image::ImageBase<T>& img);
    void CopyToImageBase(lsst::afw::image::ImageBase<T>& img) const;

    ~GpuMemOwner() {
        if (ptr != NULL) cudaFree(ptr);
    }
};

}}}}} // namespace lsst::afw::math::detail::gpu ends

#endif

