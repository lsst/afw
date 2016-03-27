// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Functions and a class to help allocating GPU global memory and transferring data to and from a GPU
 *
 * @author Kresimir Cosic
 *
 * Contents of this file are available only when GPU_BUILD is defined (i.e. when cuda_toolkit package is setup)
 *
 * requires:
 *   - include <cuda.h>
 *   - include <cuda_runtime.h>
 *   - include "lsst/afw/gpu/detail/GpuBuffer2D.h"
 *
 * @ingroup afw
 */

#ifdef GPU_BUILD

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

/**
 * \brief Allocates memory on GPU for an array
 *
 * \return Pointer to allocated GPU memory, or NULL on failure
 *
 * \arg size: specifies the number of elements of array that will be allocated
 *
 * Allocates a continuos block of GPU memory, and returns a GPU pointer to the allocated memory.
 * If GPU memory allocation fails, returns NULL, so the return value should always be verified.
 * Use function CopyToGpu() to copy data to the GPU into the allocated block, or CopyFromGpu()
 * to copy data from the allocated GPU memory to local buffers.
 *
 * It is obligatory to free the allocated GPU memory with cudaFree() (available from cuda_runtime.h)
 */
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

/**
 * \brief Copies an array of elements from GPU memory to an array in main memory
 * \arg destCpu: pointer to the destination array (in main memory)
 * \arg sourceGpu: pointer to the source array, in GPU memory
 * \arg size: number of elements of the array to copy
 *
 * Copies data from GPU to CPU memory. destCpu should be a valid pointer into any allocated CPU memory,
 * and sourceGpu should be a valid parameter into any allocated GPU memory.
 *
 * A failure to copy is generaly not expected. It could be due to invalid arguments,
 * or due to CUDA or GPu device not being initialized properly, or not in a correct state.
 *
 * \throw lsst::afw::math::detail::GpuMemoryError if data copying fails
 */
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
        throw LSST_EXCEPT(GpuMemoryError, "CopyFromGpu: failed");
}

/**
 * \brief Copies an array of elements from main memory to GPU memory
 * \arg destGpu: pointer to the destination array, in GPU memory
 * \arg sourceCpu: pointer to the source array (in main memory)
 * \arg size: number of elements of the array to copy
 *
 * Copies data from CPU to GPU memory. destGpu should be a valid pointer into any allocated GPU memory,
 * and sourceCpu should be a valid parameter into any allocated CPU memory.
 *
 * A failure to copy is generaly not expected. It could be due to invalid arguments,
 * or due to CUDA or GPU device not being initialized properly, or not in a correct state.
 *
 * \throw lsst::afw::math::GpuMemoryError if data copying fails
 */
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
        throw LSST_EXCEPT(GpuMemoryError, "CopyToGpu: failed");
    }
}

/**
 * \brief Transfers data from an array to the GPU
 * \arg sourceCpu: pointer to the source array (in CPU main memory)
 * \arg size: number of elements of the array to copy
 * \return Pointer to allocated GPU memory where data was transferred int, or NULL on GPU memory allocation failure
 *
 * Allocates a continuos block of GPU memory which will contain the transferred data,
 * and returns a GPU pointer to the allocated memory. Then copies data from CPU to GPU memory.
 * sourceCpu should be a valid parameter into any allocated CPU memory.
 * If GPU memory allocation fails, returns NULL, so the return value should always be verified.
 * Use function CopyFromGpu() to copy data from the allocated GPU memory to local buffers.
 *
 * It is obligatory to free the allocated GPU memory with cudaFree() (available from cuda_runtime.h)
 *
 * Throws exception if data copying fails. A failure to copy is generaly not expected. It could be due to
 * invalid arguments, or due to CUDA or GPU device not being initialized properly, or not in a correct state.
 *
 * \throw lsst::afw::math::GpuMemoryError if data copying fails
 */
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
        throw LSST_EXCEPT(GpuMemoryError, "TransferToGpu: transfer failed");
    }
    return dataGpu;
}

/** \brief A class for simplified GPU memory managment and copying data to and from a GPU

    Automatically releases GPU memory on destruction, simplifying GPU memory management

    The objects of this class have two states: free (when this->ptr == NULL) and bound (otherwise)
         free - this object is not bound to any GPU memory
         bound - a block of GPU memory has been allocated by this object and it is bound to it
    Some of the available functions change the state of this object from free to bound.
    Some of the functions can be called only in a particular state.
    When this object is destroyed, if it was in a bound state, it will free the allocated GPU memory block.

    The objects of this class are non-copyable
*/
template<typename T>
class GpuMemOwner
{
private:
    void operator=(const GpuMemOwner& rhs);

public:
    GpuMemOwner(const GpuMemOwner& rhs) {
	    assert(rhs.getPtr() == NULL);
        ptr=NULL;
	}

    T* ptr;   ///> pointer to the allocated GPU memory block, or NULL
    int size; ///> size (as element count) of the allocated GPU memory block, valid whenever this->ptr != NULL

    /** \brief Creates a GpuMemOwner object in a free state (see GpuMemOwner class description) */
    GpuMemOwner() : ptr(NULL) {}

    /** \brief returns the value of this->ptr */
    T* getPtr() const
    {
    	return ptr;
    }

    /** \brief returns the value of this->size */
    int getSize() const
    {
    	return size;
    }

    /**
     * \brief Transfers data from an array to the GPU
     * \arg source: pointer to the source array (in CPU main memory)
     * \arg size: number of elements of the array to copy
     * \return Pointer to allocated GPU memory where data was transferred into, or NULL on GPU mem. allocation failure
     *
     * \pre this->ptr must be NULL (this object must be in the free state, see class description)
     *
     * If transfer succeeds, this->ptr will be set to point to the GPU memory block where data was transferred into.
     * The same value (equal to this->ptr will be returned). It will change the
     * state of this object to bound. Sets this->size to the value of the argument size_p
     * If GPU memory allocation fails, returns NULL, and sets this->ptr to NULL, so the return value should
     * always be verified.
     *
     * Use member function CopyFromGpu() to copy data from the bound GPU memory.
     *
     * Also see function lsst:afw::gpu::detail::TransferToGpu().
     *
     * Throws exception if data copying fails. A failure to copy is generaly not expected. It could be due to invalid
     * arguments, or due to CUDA or GPU device not being initialized properly, or not in a correct state.
     *
     * \throw lsst::afw::math::GpuMemoryError if data copying fails
     */
    T* Transfer(const T* source, int size_p) {
        assert(ptr == NULL);
        size = size_p;
        ptr = TransferToGpu(source, size);
        return ptr;
    }

    /**
     * \brief Transfers data from an GpuBuffer2D to the GPU
     * \arg source: source GpuBuffer2D (in CPU main memory)
     * \return Pointer to allocated GPU memory where data was transferred into, or NULL on GPU mem. allocation failure
     *
     * \pre this->ptr must be NULL (this object must be in the free state, see class description)
     *
     * \copydetails GpuMemOwner::Transfer(const T* source, int size_p)
     *
     * \throw lsst::afw::math::GpuMemoryError if data copying fails
     */
    T* Transfer(const GpuBuffer2D<T>& source) {
        assert(ptr == NULL);
        size = source.Size();
        ptr = TransferToGpu(source.img, size);
        return ptr;
    }

    /**
     * \brief Transfers data from a vector to the GPU
     * \arg source: source vector (in CPU main memory)
     * \return Pointer to allocated GPU memory where data was transferred into, or NULL on GPU mem. allocation failure
     *
     * \pre this->ptr must be NULL (this object must be in the free state, see class description)
     *
     * \copydetails GpuMemOwner::Transfer(const T* source, int size_p)
     *
     * \throw lsst::afw::math::GpuMemoryError if data copying fails
     */
    T* TransferVec(const std::vector<T>& source) {
        assert(ptr == NULL);
        size = int(source.size());
        ptr = TransferToGpu(&source[0], size);
        return ptr;
    }

    /**
     * \brief Allocates GPU memory for an array
     * \arg size: number of elements of the array
     * \return Pointer to allocated GPU memory, or NULL on GPU memory allocation failure
     *
     * \pre this->ptr must be NULL (this object must be in the free state, see class description)
     *
     * If allocation succeeds, this->ptr will be set to point to the allocated GPU memory block.
     * The same value (equal to this->ptr will be returned). It will change the
     * state of this object to bound. Sets this->size to the value of the argument size_p
     * If GPU memory allocation fails, returns NULL, and sets this->ptr to NULL, so the return value should
     * always be verified.
     *
     * Use member function CopyFromGpu() to copy data from the bound GPU memory.
     *
     * Also see function lsst:afw::gpu::detail::AllocOnGpu().
     */
    T* Alloc(int size_p)  {
        assert(ptr == NULL);
        size = size_p;
        ptr = AllocOnGpu<T>(size);
        return ptr;
    }

    /** \brief Copies GpuBuffer2D data to bound GPU memory
     *
     * Also see lsst::afw::math::detail::CopyToGpu, which this function delegates to.
     */
    T* CopyToGpu(detail::GpuBuffer2D<T>& source) const {
        assert(ptr != NULL);
        assert(source.Size() == size);
        lsst::afw::gpu::detail::CopyToGpu(ptr, source.img, size);
        return ptr;
    }

    /** \brief Copies GpuBuffer2D data from bound GPU memory
     *
     * Also see lsst::afw::math::detail::CopyFromGpu, which this function delegates to.
     */
    T* CopyFromGpu(detail::GpuBuffer2D<T>& dest) const {
        assert(ptr != NULL);
        assert(dest.Size() == size);
        lsst::afw::gpu::detail::CopyFromGpu(dest.img, ptr, size);
        return ptr;
    }

    /** \brief Copies data from bound GPU memory to CPU memory
     *
     * Also see lsst::afw::gpu::detail::CopyFromGpu, which this function delegates to.
     */
    T* CopyFromGpu(T* dest) const {
        assert(ptr != NULL);
        lsst::afw::gpu::detail::CopyFromGpu(dest, ptr, size);
        return ptr;
    }

    /** \brief Transfers data from an ImageBase to the GPU. The stride from ImageBase is retained.
     *
     * Also see ImageBase::Transfer() function, which this function delegates to.
     */
    int TransferFromImageBase(const lsst::afw::image::ImageBase<T>& img);

    /** \brief Allocates a block of GPU memory into which ImageBases' data can fit into.
     *
     *  The stride required for data of a given ImageBase is taken into account.
     *
     * Also see ImageBase::Alloc() function, which this function delegates to.
     */
    int AllocImageBaseBuffer(const lsst::afw::image::ImageBase<T>& img);

    /** \brief Copies data from bound GPU memory into an ImageBase.
     *
     * The stride of ImageBase must be taken into account, and the easiest way to assure that
     * is to bind GPU memory with ImageBase::TransferFromImageBase() or ImageBase::AllocImageBaseBuffer() .
     *
     * Also see ImageBase::CopyFromGpu() function, which this function delegates to.
     */
    void CopyToImageBase(lsst::afw::image::ImageBase<T>& img) const;

    /**
     * \brief Frees bound GPU memory block
     *
     * If this object is in bound state, frees bound GPU memory. Otherwise does nothing.
     * Also see GpuMemOwner class description
     */
    ~GpuMemOwner() {
        if (ptr != NULL) cudaFree(ptr);
    }
};

}}}} // namespace lsst::afw::gpu::detail ends

#endif //IS_GPU_BUILD

