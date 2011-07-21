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
 * @brief Set up for convolution, calls GPU convolution kernels
 *
 * Functions in this file are used to allocate necessary buffers,
 * transfer data from and to GPU memory, and to set up and perform convolution.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifndef GPU_BUILD //build this file only if requested

#include <stdio.h>

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

void PrintCudaDeviceInfo() {
    printf("Afw not compiled with GPU support\n");
}
void TestGpuKernel(int& ret1, int& ret2) {
    ret1=0;
    ret2=0;
}

}
}
}
}
}

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/math/Kernel.h"

#include "lsst/afw/math/detail/ImageBuffer.h"
#include "lsst/afw/math/detail/cudaConvWrapper.h"

using namespace std;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetailGpu = lsst::afw::math::detail::gpu;

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

void PrintDeviceProperties(int id,cudaDeviceProp deviceProp)
{
    printf("Name : %s  |", deviceProp.name );
    printf("  CUDA Capable SM %d.%d hardware, %d multiproc.\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    printf("   Clock rate:       %6.2f GHz \t", deviceProp.clockRate/(1000.0*1000));
    printf("   Memory on device: %6zu MiB\n", deviceProp.totalGlobalMem/(1<<20) );
    printf("   Multiprocessors:  %6d\n", deviceProp.multiProcessorCount);
    printf("       Warp size:    %6d \t",  deviceProp.warpSize );
    printf("       Shared memory:%6zu KiB\n", deviceProp.sharedMemPerBlock/(1<<10) );
    printf("       Registers:    %6d \t", deviceProp.regsPerBlock );
    printf("       Max threads:  %6d \n", deviceProp.maxThreadsPerBlock );

    printf("   Compute mode (device sharing) : ");
    if (deviceProp.computeMode==cudaComputeModeDefault)
        printf("Default - shared between threads\n" );
    if (deviceProp.computeMode==cudaComputeModeExclusive)
        printf("Exclusive - only one thread at a time\n" );
    if (deviceProp.computeMode==cudaComputeModeProhibited)
        printf("Prohibited - cannot use this device\n" );

    printf("   Timeout enabled: %3s  ", deviceProp.kernelExecTimeoutEnabled==1 ? "Yes" : "No" );
    printf("   Overlapped copying: %3s  ", deviceProp.deviceOverlap==1 ? "Yes" : "No" );
    printf("   Intergrated on MB: %3s\n", deviceProp.integrated==1 ? "Yes" : "No" );
    printf("   Memory pitch: %12zu \t", deviceProp.memPitch );
    printf("   Constant memory: %6zu kiB \n", deviceProp.totalConstMem/(1<<10) );
}

void PrintCudaErrorInfo(cudaError_t cudaError, const char* errorStr)
{
    printf("\nSupplied error string: %s\n",errorStr);
    printf(  "CUDA error           : %d\n",cudaError);
    printf(  "CUDA error string    : %s\n",cudaGetErrorString(cudaError));
    exit(0);
}

void PrintCudaDeviceInfo()
{
    fflush(stdout);

    cudaError_t cudaError;

    int driverVersion;
    cudaError=cudaDriverGetVersion(&driverVersion);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"Could not get CUDA driver version");
    printf("Driver ver.: %d.%d   ",driverVersion/1000, driverVersion%1000);
    fflush(stdout);

    int runtimeVersion;
    cudaError=cudaRuntimeGetVersion(&runtimeVersion);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"Could not get CUDA runtime version");
    printf("Runtime ver.: %d.%d   ",runtimeVersion/1000, runtimeVersion%1000);
    fflush(stdout);

    //int preferredDeviceId = 0;

    int cudaDevicesN=0;
    cudaError=cudaGetDeviceCount(&cudaDevicesN);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"Could not get CUDA device count");

    printf("Device count: %d   ",cudaDevicesN);
    fflush(stdout);
    if(cudaDevicesN<1) {
        printf("Your system does not have a CUDA capable device\n");
        exit(0);
    }

    int curDevId;
    cudaError=cudaGetDevice(&curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"Could not get CUDA device id");
    printf("Info for device %d\n",curDevId);
    fflush(stdout);

    cudaDeviceProp deviceProp;
    cudaError=cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"Could not get CUDA device properties");
    PrintDeviceProperties(curDevId,deviceProp);
    fflush(stdout);

    for (int i=0; i<79; i++) printf("-");
    printf("\n");
    fflush(stdout);
}

int GetPreferredCudaDevice()
{
    const char *devStr = getenv("CUDA_DEVICE");
    if (devStr == NULL)
        return -2;
    else
        return atoi(devStr);
}

void SelectPreferredCudaDevice()
{
    static bool isDeviceSet=false;
    if (isDeviceSet) return;
    isDeviceSet=true;

    int cudaDevicesN=0;
    cudaGetDeviceCount(&cudaDevicesN);
    if (cudaDevicesN==0)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "No CUDA capable GPUs found");

    int devId=GetPreferredCudaDevice();

    //printf("DEVICE ID %d\n", devId);

    if (devId>=0) {
        cudaError_t err = cudaSetDevice(devId);
        if (err!= cudaSuccess) {
            cudaGetLastError(); //clear error
            char errorStr[1000];
            sprintf(errorStr, "Error selecting device %d:\n %s\n", devId, cudaGetErrorString(err));
            throw LSST_EXCEPT(pexExcept::RuntimeErrorException, errorStr);
        }
        return;
    }

    if (devId!=-2)
        return;

    cudaDeviceProp prop;
    char errorStr[1000];
    memset(&prop, 1, sizeof(prop));

    //min sm 1.3
    prop.major=1;
    prop.minor=3;

    prop.maxGridSize[0]=128;
    prop.maxThreadsDim[0]=256;

    prop.multiProcessorCount=2;
    prop.clockRate = 700.0 * 1000 ; // 700 MHz
    prop.warpSize = 32 ;
    prop.sharedMemPerBlock = 32 * (1<<10); //23 KiB
    prop.regsPerBlock = 256 * 60 ;
    prop.maxThreadsPerBlock = 256;
    prop.totalGlobalMem = 500 * 1024 * 1024;

    cudaError_t cudaError= cudaChooseDevice(&devId, &prop);
    //printf("Error device %d:\n %s\n", devId, cudaGetErrorString(err));
    if (cudaError!= cudaSuccess)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Error choosing device automatically");

    cudaError = cudaSetDevice(devId);
    if (cudaError == cudaErrorSetOnActiveProcess) {
        cudaGetDevice(&devId);
    }
    else if (cudaError!= cudaSuccess) {
        cudaGetLastError(); //clear error
        sprintf(errorStr, "Error automatically selecting device %d:\n %s\n",
                devId, cudaGetErrorString(cudaError));
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, errorStr);
    }

    cudaDeviceProp deviceProp;
    cudaError=cudaGetDeviceProperties(&deviceProp, devId);
    if (cudaError!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Could not get CUDA device properties");

    if (deviceProp.major < prop.major ||
            (deviceProp.major == prop.major && deviceProp.minor < prop.minor)
       ) {
        sprintf(errorStr,"Only SM %d.%d or better GPU devices are currently allowed",prop.major,prop.minor);
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, errorStr );
    }

    if (deviceProp.major == prop.major && deviceProp.minor < prop.minor)

        if (deviceProp.totalGlobalMem < prop.totalGlobalMem)
            throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Not enough global memory on GPU");

    if (deviceProp.sharedMemPerBlock < 16 * 1000)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Not enough shared memory on GPU");

    if (deviceProp.regsPerBlock < prop.regsPerBlock)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Not enough registers per block available on GPU");

    if (deviceProp.maxThreadsPerBlock < prop.maxThreadsPerBlock)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Not enough threads per block available on GPU");
}

int GetCudaDeviceId()
{
    int curDevId;
    cudaError_t cudaError=cudaGetDevice(&curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"GetCudaDeviceId> Could not get CUDA device id");
    return curDevId;
}

int GetCudaCurSMSharedMemorySize()
{
    int curDevId=GetCudaDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError=cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"GetCudaSMSharedMemorySize> Could not get CUDA device properties");

    return deviceProp.sharedMemPerBlock;
}

int GetCudaCurGlobalMemorySize()
{
    int curDevId=GetCudaDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError=cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"GetCudaCurGlobalMemorySize> Could not get CUDA device properties");

    return deviceProp.totalGlobalMem;
}

int GetCudaCurSMRegisterCount()
{
    int curDevId=GetCudaDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError=cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"GetCudaSMRegisterCount> Could not get CUDA device properties");

    return deviceProp.regsPerBlock;
}

int GetCudaCurSMCount()
{
    int curDevId=GetCudaDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError=cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"GetCudaSMCount> Could not get CUDA device properties");

    return deviceProp.multiProcessorCount;
}

bool GetCudaCurIsDoublePrecisionSupported()
{
    int curDevId=GetCudaDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError=cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"GetCudaIsDoublePrecisionSupported> Could not get CUDA device properties");

    return deviceProp.major>=2 || (deviceProp.major==1 && deviceProp.minor>=3);
}

void SetCudaDevice(int devId)
{
    cudaError_t cudaError=cudaSetDevice(devId);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"SetCudaDevice> unsucessfull");
}

void CudaReserveDevice()
{
    int* dataGpu;
    cudaError_t cudaError=cudaMalloc((void**)&dataGpu, 256*sizeof(int));
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"CudaReserveDevice> Could not reserve device by calling cudaMalloc");

    cudaError=cudaFree(dataGpu);
    if (cudaError!=cudaSuccess) PrintCudaErrorInfo(cudaError,"CudaReserveDevice> Could not release memory by calling cudaFree");
}

template<typename T>
T* AllocOnGpu(int size)
{
    T* dataGpu;
    cudaError_t cudaError=cudaMalloc((void**)&dataGpu, size*sizeof(T));
    if (cudaError!=cudaSuccess)
        return NULL;

    return dataGpu;
}
template<typename T>
void CopyFromGpu(T* destCpu, T* sourceGpu,int size)
{
    cudaError_t cudaError=cudaMemcpy(
                              /* Desination:*/     destCpu,
                              /* Source:    */     sourceGpu,
                              /* Size in bytes: */ size*sizeof(T),
                              /* Direction   */    cudaMemcpyDeviceToHost
                          );
    if (cudaError!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::MemoryException, "CopyFromGpu: failed");
}
template<typename T>
void CopyToGpu(T* destGpu, T* sourceCpu,int size)
{
    cudaError_t cudaError;
    cudaError=cudaMemcpy(
                  /* Desination:*/     destGpu,
                  /* Source:    */     sourceCpu,
                  /* Size in bytes: */ size*sizeof(T),
                  /* Direction   */    cudaMemcpyHostToDevice
              );
    if (cudaError!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::MemoryException, "CopyToGpu: failed");
}

template<typename T>
T* TransferToGpu(const T* sourceCpu,int size)
{
    T* dataGpu;
    cudaError_t cudaError=cudaMalloc((void**)&dataGpu, size*sizeof(T));
    if (cudaError!=cudaSuccess) {
        return NULL;
    }
    cudaError=cudaMemcpy(
                  /* Desination:*/     dataGpu,
                  /* Source:    */     sourceCpu,
                  /* Size in bytes: */ size*sizeof(T),
                  /* Direction   */    cudaMemcpyHostToDevice
              );
    if (cudaError!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::MemoryException, "TransferToGpu: transfer failed");

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
        assert(ptr==NULL);
        size = size_p;
        ptr = TransferToGpu(source,size);
        return ptr;
    }
    T* Transfer(const ImageBuffer<T>& source) {
        assert(ptr==NULL);
        size = source.Size();
        ptr=TransferToGpu(source.img, size);
        return ptr;
    }
    T* Alloc(int size_p)  {
        assert(ptr==NULL);
        size = size_p;
        ptr=AllocOnGpu<T>(size);
        return ptr;
    }
    T* CopyToGpu(ImageBuffer<T>& source) {
        assert(ptr!=NULL);
        assert(source.Size() == size);
        gpu::CopyToGpu(ptr,source.img,size);
        return ptr;
    }

    ~GpuMemOwner() {
        if (ptr!=NULL)
            cudaFree(ptr);
    }
};

// Returns true if there is sufficient shared memory for loading an image block,
// where image block includes including filter frame.
bool IsSufficientSharedMemoryAvailable(int filterW, int filterH, int pixSize)
{
    int shMemSize=GetCudaCurSMSharedMemorySize();
    int bufferSize=(filterW+blockSizeX-1)*(filterH+blockSizeY-1)*pixSize;

    return shMemSize-90-bufferSize > 0;
}

// Returns true if there is sufficient shared memory for loading an image block,
// and acommpanying block of mask data (mask image block),
// where image block and mask image block include including filter frame.
bool IsSufficientSharedMemoryAvailableImgAndMask(int filterW, int filterH, int pixSize)
{
    int shMemSize=GetCudaCurSMSharedMemorySize();
    int imgBufferSize=(filterW+blockSizeX-1)*(filterH+blockSizeY-1)*pixSize;
    int mskBufferSize=(filterW+blockSizeX-1)*(filterH+blockSizeY-1)*sizeof(MskPixel);

    int memRemaining=shMemSize-200-imgBufferSize-mskBufferSize ;

    return memRemaining > 0;
}

// This function decides on the best GPU block count
// uses simple heuristics (not to much blocks and not too many)
// but guarantees that number of blocks will be a multiple of number of multiprocessors
int CalcBlockCount(int multiprocCount)
{
    if (multiprocCount<12)
        return multiprocCount*4;
    if (multiprocCount<24)
        return multiprocCount*2;
    return multiprocCount;
}



// calls test gpu kernel
// should return 5 and 8 in ret1 and ret2
void TestGpuKernel(int& ret1, int& ret2)
{
    int res[2];

    int* resGpu=gpu::AllocOnGpu<int>(2);

    CallTestGpuKernel(resGpu);

    gpu::CopyFromGpu(res,resGpu,2);

    ret1=res[0];
    ret2=res[1];
}

} // namespace lsst::afw::math::detail::gpu ends

namespace {

//calculates sum of each image in 'images' vector
template <typename T>
vector<T> SumsOfImages(const vector< ImageBuffer<T> >&  images)
{
    int n=int(images.size());
    vector<T> sum(n);
    for (int i=0; i<n; i++) {
        KerPixel totalSum=0;
        int h=images[i].height;
        int w=images[i].width;

        for (int y=0; y<h; y++) {
            KerPixel rowSum=0;
            for (int x=0; x<w; x++)
                rowSum+= images[i].Pixel(x,y);
            totalSum+=rowSum;
        }
        sum[i]=totalSum;
    }
    return sum;
}

/**
    Calculates values of some spatial functions (given by sfn) and normalization.

    Normalization is added-up, and a reciprocal value is calculated on final pass.

    @arg i          - first spatial function to calculate values of
    @arg curKernelN - number of spatial function to calculate values of
    @arg isLastPass - on final pass the reciprocal for normalization will be calculated
    @arg doNormalize - if false, the normalitatoin will not be calculated
    @arg colPos    - x coodinate for spatial functions of a given image column
    @arg rowPos    - y coodinate for spatial functions of a given image row
    @arg x       - x cordinate in img, top left corner of rectangle to be copied
    @arg sFnVal    - output buffers, contains calculated values of spatial functions
           the first output buffer to be used is given by sFnValBeg
*/
inline bool CalculateSpatialFunctionsValuesAndNormalization(
    const int i, const int curKernelN, const bool isLastPass,
    const bool doNormalize,
    const vector<double>& colPos,
    const vector<double>& rowPos,
    const vector< afwMath::Kernel::SpatialFunctionPtr >& sFn,
    const vector< KerPixel >&   basisKernelSum,
    ImageBuffer<double>& norm,       // accumulator and result, input and output
    vector< ImageBuffer<double> >& sFnVal, //output buffers
    int sFnValBeg
)
{
    const bool isFirstPass= i==0;
    bool normalizationError=false;
    int outW = sFnVal[0].width;
    int outH = sFnVal[0].height;

    assert(norm.width==outW);
    assert(norm.height==outH);

    // calculate spatial function values for processed kernels
    int sfnKstart=0;
    if (doNormalize && isFirstPass) {
        sfnKstart=1;
        for (int y=0; y<outH; y++)
            for (int x=0; x<outW; x++) {
                double val=(*sFn[0])(colPos[x],rowPos[y]);
                sFnVal[sFnValBeg].Pixel(x,y)=val;
                norm.Pixel(x,y) = basisKernelSum[0] * val;
            }
    }

    if (doNormalize)
        for (int kernelI=sfnKstart; kernelI<curKernelN; kernelI++)
            for (int y=0; y<outH; y++)
                for (int x=0; x<outW; x++) {
                    double val=(*sFn[i+kernelI])(colPos[x],rowPos[y]);
                    sFnVal[sFnValBeg+kernelI].Pixel(x,y)=val;
                    norm.Pixel(x,y) += basisKernelSum[i+kernelI] * val;
                }
    else
        for (int kernelI=sfnKstart; kernelI<curKernelN; kernelI++)
            for (int y=0; y<outH; y++)
                for (int x=0; x<outW; x++) {
                    double val=(*sFn[i+kernelI])(colPos[x],rowPos[y]);
                    sFnVal[sFnValBeg+kernelI].Pixel(x,y)=val;
                }

    if (doNormalize && isLastPass)
        for (int y=0; y<outH; y++)
            for (int x=0; x<outW; x++) {
                double kernelSum=norm.Pixel(x,y);
                if (kernelSum==0)
                    normalizationError=true;
                else
                    norm.Pixel(x,y) = 1.0 / kernelSum;
            }

    return normalizationError;
}

/**
    Combine output of convolution by each kernel and spatial function values
    to get output image.

    Result is added up in outImage.

    @arg curKernelN - number of kernels and buffers with spatial function values
    @arg sFnValBeg  - first spatial function values buffer to use
    @arg outBuf - (input) results of convolution by each kernel
*/
template <typename OutPixelT>
void CombineOutputsAndSfnIntoResult(
    int curKernelN, bool isFirstPass,
    const vector< ImageBuffer<double> >&    sFnVal,
    int sFnValBeg,
    const vector< ImageBuffer<double> >& outBuf,
    ImageBuffer<OutPixelT>&             outImage   //accumulator and result, input and output
)
{
    for (int kernelI=0; kernelI<curKernelN; kernelI++)
        if (isFirstPass && kernelI==0)
            for (int y=0; y<outImage.height; y++) {
                OutPixelT* outPtr          = outImage.GetImgLinePtr(y);
                const double* outBufPtr    = outBuf[kernelI].GetImgLinePtr(y);
                const double* sFnValPtr    = sFnVal[sFnValBeg+kernelI].GetImgLinePtr(y);
                for (int x=0; x<outImage.width; x++) {
                    *outPtr = *outBufPtr * *sFnValPtr;
                    outPtr++;
                    outBufPtr++;
                    sFnValPtr++;
                }
            }
        else
            for (int y=0; y<outImage.height; y++) {
                OutPixelT* outPtr          = outImage.GetImgLinePtr(y);
                const double* outBufPtr = outBuf[kernelI].GetImgLinePtr(y);
                const double* sFnValPtr    = sFnVal[sFnValBeg+kernelI].GetImgLinePtr(y);
                for (int x=0; x<outImage.width; x++) {
                    *outPtr += *outBufPtr * *sFnValPtr;
                    outPtr++;
                    outBufPtr++;
                    sFnValPtr++;
                }
            }
}

/**
    Convolves given inImage with multiple kernels.

    Calculates:
        - the result of convolution with LinearCombination coefficients (given
            by sFn) in out image
        - the sFn (spatial function) values, output in sFnVal
        - if doNormalize is true, will also compute and apply normalization.
            normalization coefficients will be placed in norm buffer.

    Basis kernels, given by basisKernelsListGPU, must have been transfered to GPU memory previously.
*/
template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_MultipleKernels(
    const ImageBuffer<InPixelT>& inImage,
    const vector<double>& colPos,
    const vector<double>& rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr >& sFn,
    vector< ImageBuffer<double> >&   sFnVal, //output
    bool                             reuseSfnValBuffers,
    int maxSimKernels,
    ImageBuffer<OutPixelT>&  outImage, //output
    KerPixel*   basisKernelsListGPU,
    int kernelW, int kernelH,
    const vector< KerPixel >&   basisKernelSum,
    ImageBuffer<double>& norm,  //output
    bool doNormalize
)
{
    int kernelN=sFn.size();

    gpu::GpuMemOwner<InPixelT> inImageGPU;
    inImageGPU.Transfer(inImage);
    if (inImageGPU.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for input image");

    int shMemSize=gpu::GetCudaCurSMSharedMemorySize()-90;

    maxSimKernels = min(maxSimKernels,kernelN);

    // allocate array of output images on GPU   (one output image per kernel)
    vector< double* > outImageGPUPtr(kernelN);
    vector< gpu::GpuMemOwner<double> > outImageGPU_Owner(kernelN);

    int simultaneousKernelN=0;
    for (int i=0; i<maxSimKernels; i++) {
        outImageGPUPtr[i]=outImageGPU_Owner[i].Alloc( outImage.Size());
        if (outImageGPUPtr[i]==NULL) break;
        simultaneousKernelN=i+1;
    }
    if (simultaneousKernelN==0)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for any kernels");

    double** outImageGPU   =gpu::TransferToGpu(&outImageGPUPtr[0],kernelN);

    // allocate convolution output buffers on CPU  (one per kernel)
    vector< ImageBuffer<double> > outBuf(simultaneousKernelN);
    for (int i=0; i<simultaneousKernelN; i++)
        outBuf[i].Init(outImage.width, outImage.height);

    int blockN=gpu::CalcBlockCount( gpu::GetCudaCurSMCount());
    bool normalizationError=false;

    int curKernelN = kernelN%simultaneousKernelN;
    if (curKernelN==0) curKernelN=simultaneousKernelN;
    bool prevIsFirstPass=false;
    int prevCurKernelN=curKernelN;

    int sFnValBeg=0;

    cudaGetLastError(); //clear error status

    for (int i=0; i<kernelN; i+=prevCurKernelN)
    {
        bool isLastPass  = i+simultaneousKernelN>=kernelN;
        bool isFirstPass = i==0;

        mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<double,InPixelT>(
            inImageGPU.ptr, inImage.width, inImage.height,
            basisKernelsListGPU+(i*kernelW*kernelH),
            curKernelN,
            kernelW, kernelH,
            outImageGPU,
            blockN,
            shMemSize
        );
        //the following part until CopyFromGpu executes in parallel with GPU

        //add up result from previous calculation
        if (!isFirstPass)
            CombineOutputsAndSfnIntoResult(prevCurKernelN, prevIsFirstPass, sFnVal, sFnValBeg, outBuf, outImage);

        sFnValBeg=i;
        if (reuseSfnValBuffers)
            sFnValBeg=0;

        normalizationError = normalizationError ||
                             CalculateSpatialFunctionsValuesAndNormalization(
                                 i, curKernelN, isLastPass, doNormalize,
                                 colPos, rowPos, sFn, basisKernelSum,
                                 norm, sFnVal, sFnValBeg
                             );

        cudaThreadSynchronize();
        if (cudaGetLastError()!=cudaSuccess)
            throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "GPU calculation failed to run");

        for (int kernelI=0; kernelI<curKernelN; kernelI++)
            gpu::CopyFromGpu(outBuf[kernelI].img, outImageGPUPtr[kernelI], outImage.Size() );

        prevIsFirstPass=isFirstPass;
        prevCurKernelN=curKernelN;
        curKernelN=simultaneousKernelN;
    }
    CombineOutputsAndSfnIntoResult(prevCurKernelN, prevIsFirstPass, sFnVal, sFnValBeg, outBuf, outImage);

    if (doNormalize) {
        if (normalizationError)
            throw LSST_EXCEPT(pexExcept::OverflowErrorException, "Cannot normalize; kernel sum is 0");

        for (int y=0; y<outImage.height; y++)
            for (int x=0; x<outImage.width; x++)
                outImage.Pixel(x,y) *= norm.Pixel(x,y);
    }
}

} //local namespace ends

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_LinearCombinationKernel(
    ImageBuffer<InPixelT>& inImage,
    vector<double> colPos,
    vector<double> rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn,
    ImageBuffer<OutPixelT>&                outImage,
    std::vector< ImageBuffer<KerPixel> >&  basisKernels,
    bool doNormalize
)
{
    assert(basisKernels.size() == sFn.size());

    int kernelN=sFn.size();
    int kernelW=basisKernels[0].width;
    int kernelH=basisKernels[0].height;
    int kernelSize=kernelW*kernelH;

    for (int i=0; i<kernelN; i++) {
        assert(kernelW==basisKernels[i].width);
        assert(kernelH==basisKernels[i].height);
    }

    //calculate kernel sums
    std::vector< KerPixel > basisKernelSum(kernelN);
    if (doNormalize)
        basisKernelSum=SumsOfImages(basisKernels);

    int maxSimKernels = 6;    // heuristic - more than 7 kernels can not provide increase in speed
    maxSimKernels = min(maxSimKernels,kernelN);

    // allocate array of basis kernels on GPU
    gpu::GpuMemOwner<KerPixel > basisKernelsGPU;
    basisKernelsGPU.Alloc(kernelSize * kernelN);

    for (int i=0; i<kernelN; i++) {
        KerPixel* kernelBeg=basisKernelsGPU.ptr + (kernelSize * i);
        gpu::CopyToGpu(kernelBeg,
                       basisKernels[i].img,
                       kernelSize
                      );
    }

    // allocate array of spatial function values on CPU  (one per kernel)
    vector< ImageBuffer<double> > sFnVal(maxSimKernels);
    for (int i=0; i<maxSimKernels; i++)
        sFnVal[i].Init(outImage.width, outImage.height);

    //buffer for normalization coefficients
    ImageBuffer<double> norm;
    norm.width = outImage.width;
    norm.height = outImage.height; //squelch warnings
    if (doNormalize)
        norm.Init(outImage.width, outImage.height);

    GPU_ConvolutionImage_MultipleKernels<OutPixelT,InPixelT>(
        inImage,
        colPos,
        rowPos,
        sFn,
        sFnVal, //output
        true,  //reuseSfnValBuffers,
        maxSimKernels,
        outImage,  //output
        basisKernelsGPU.ptr,
        kernelW, kernelH,
        basisKernelSum,
        norm,  //output
        doNormalize
    );

}

#define INSTANTIATE_GPU_ConvolutionImage_LinearCombinationKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionImage_LinearCombinationKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>& inImage, \
                    vector<double> colPos, \
                    vector<double> rowPos, \
                    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn, \
                    ImageBuffer<OutPixelT>&                outImage, \
                    std::vector< ImageBuffer<KerPixel> >&  basisKernels, \
                    bool doNormalize \
                    );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_LinearCombinationKernel(
    ImageBuffer<InPixelT>& inImageImg,
    ImageBuffer<VarPixel>& inImageVar,
    ImageBuffer<MskPixel>& inImageMsk,
    vector<double> colPos,
    vector<double> rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn,
    ImageBuffer<OutPixelT>&                outImageImg,
    ImageBuffer<VarPixel>&                 outImageVar,
    ImageBuffer<MskPixel>&                 outImageMsk,
    std::vector< ImageBuffer<KerPixel> >&  basisKernels,
    bool doNormalize
)
{
    int blockN=gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

    assert(basisKernels.size() == sFn.size());
    assert(outImageImg.width==outImageVar.width);
    assert(outImageImg.width==outImageMsk.width);
    assert(outImageImg.height==outImageVar.height);
    assert(outImageImg.height==outImageMsk.height);

    int outWidth=outImageImg.width;
    int outHeight=outImageImg.height;

    const int kernelN=sFn.size();
    const int kernelW=basisKernels[0].width;
    const int kernelH=basisKernels[0].height;
    const int kernelSize = kernelW*kernelH;

    for (int i=0; i<kernelN; i++) {
        assert(kernelW==basisKernels[i].width);
        assert(kernelH==basisKernels[i].height);
    }

    //calculate kernel sums
    std::vector< KerPixel > basisKernelSum(kernelN);
    if (doNormalize)
        basisKernelSum=SumsOfImages(basisKernels);

    // allocate array of basis kernels on GPU
    gpu::GpuMemOwner<KerPixel > basisKernelsGPU;
    basisKernelsGPU.Alloc(kernelSize * kernelN);

    for (int i=0; i<kernelN; i++) {
        KerPixel* kernelBeg=basisKernelsGPU.ptr + (kernelSize * i);
        gpu::CopyToGpu(kernelBeg,
                       basisKernels[i].img,
                       kernelSize
                      );
    }

    // allocate array of spatial function values on CPU and GPU
    vector< ImageBuffer<double> >         sFnVal(kernelN);
    vector< double* >                     sFnValGPUPtr(kernelN);
    vector< gpu::GpuMemOwner<double > >   sFnValGPU_Owner(kernelN);

    for (int i=0; i<kernelN; i++)
        sFnVal[i].Init(outWidth, outHeight);

    //buffer for normalization coefficients
    ImageBuffer<double> norm;
    norm.width = outWidth;
    norm.height = outHeight; //squelch warnings
    if (doNormalize)
        norm.Init(outWidth, outHeight);


    {   //==================== image plane ===================
        int maxSimKernels = 6;    // heuristic - more than 7 kernels can not provide increase in speed
        maxSimKernels = min(maxSimKernels,kernelN);

        GPU_ConvolutionImage_MultipleKernels<OutPixelT,InPixelT>(
            inImageImg,
            colPos,
            rowPos,
            sFn,
            sFnVal, //output
            false,  //reuseSfnValBuffers,
            maxSimKernels,
            outImageImg,  //output
            basisKernelsGPU.ptr,
            kernelW, kernelH,
            basisKernelSum,
            norm,  //output
            doNormalize
        );
    } //==================== image plane ends===============

    //transfer sFn values
    for (int i=0; i<kernelN; i++) {
        sFnValGPUPtr[i]=sFnValGPU_Owner[i].Transfer(sFnVal[i]);
        if (sFnValGPUPtr[i]==NULL)
            throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for spatial function values");
    }
    double**    sFnValGPU      =gpu::TransferToGpu(&sFnValGPUPtr[0],kernelN);

    //transfer normalization coeficients
    gpu::GpuMemOwner<double > normGPU_Owner;
    if (doNormalize)
        normGPU_Owner.Transfer(norm);

    //transfer input image planes to GPU
    gpu::GpuMemOwner<VarPixel> inImageGPUVar;
    inImageGPUVar.Transfer(inImageVar);
    if (inImageGPUVar.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for input variance");
    gpu::GpuMemOwner<MskPixel> inImageGPUMsk;
    inImageGPUMsk.Transfer(inImageMsk);
    if (inImageGPUMsk.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for input mask");

    // allocate output image planes on GPU
    gpu::GpuMemOwner<VarPixel > outImageGPUVar;
    outImageGPUVar.Alloc( outImageVar.Size());
    if (outImageGPUVar.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for output variance");
    gpu::GpuMemOwner<MskPixel > outImageGPUMsk;
    outImageGPUMsk.Alloc( outImageMsk.Size());
    if (outImageGPUMsk.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for output mask");

    int shMemSize=gpu::GetCudaCurSMSharedMemorySize() - 200;

    cudaGetLastError(); //clear error status
    mathDetailGpu::Call_ConvolutionKernel_LC_Var(
        inImageGPUVar.ptr, inImageVar.width, inImageVar.height,
        inImageGPUMsk.ptr,
        basisKernelsGPU.ptr, kernelN,
        kernelW, kernelH,
        sFnValGPU,
        normGPU_Owner.ptr,
        outImageGPUVar.ptr,
        outImageGPUMsk.ptr,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError()!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "GPU calculation failed to run");

    gpu::CopyFromGpu(outImageVar.img, outImageGPUVar.ptr, outImageVar.Size() );
    gpu::CopyFromGpu(outImageMsk.img, outImageGPUMsk.ptr, outImageMsk.Size() );
}

#define INSTANTIATE_GPU_ConvolutionMI_LinearCombinationKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionMI_LinearCombinationKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>& inImageImg, \
                    ImageBuffer<VarPixel>& inImageVar, \
                    ImageBuffer<MskPixel>& inImageMsk, \
                    vector<double> colPos, \
                    vector<double> rowPos, \
                    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn, \
                    ImageBuffer<OutPixelT>&                outImageImg, \
                    ImageBuffer<VarPixel>&                 outImageVar, \
                    ImageBuffer<MskPixel>&                 outImageMsk, \
                    std::vector< ImageBuffer<KerPixel> >&  basisKernels, \
                    bool doNormalize  \
                    );


template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_SpatiallyInvariantKernel(
    ImageBuffer<InPixelT>&    inImage,
    ImageBuffer<OutPixelT>&   outImage,
    ImageBuffer<KerPixel>&    kernel
)
{
    int kernelW=kernel.width;
    int kernelH=kernel.height;

    gpu::GpuMemOwner<InPixelT> inImageGPU;
    inImageGPU.Transfer(inImage);
    if (inImageGPU.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU for input image");

    int shMemSize=gpu::GetCudaCurSMSharedMemorySize()-90;

    if (!gpu::IsSufficientSharedMemoryAvailable(kernelW,kernelH,sizeof(KerPixel)))
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough shared GPU memory available");

    // allocate array of kernels on GPU
    gpu::GpuMemOwner<KerPixel > basisKernelGPU;
    basisKernelGPU.Transfer(kernel);
    if (basisKernelGPU.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for kernel");

    // allocate array of output images on GPU   (one output image per kernel)
    vector< OutPixelT* > outImageGPUPtr(1);
    vector< gpu::GpuMemOwner<OutPixelT> > outImageGPU_Owner(1);

    outImageGPUPtr[0]=outImageGPU_Owner[0].Alloc( outImage.Size());
    if (outImageGPUPtr[0]==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for output image");

    OutPixelT** outImageGPU   =gpu::TransferToGpu(&outImageGPUPtr[0],1);

    int blockN=gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

    cudaGetLastError(); //clear error status
    mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<OutPixelT,InPixelT>(
        inImageGPU.ptr, inImage.width, inImage.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPU,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError()!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "GPU calculation failed to run");

    gpu::CopyFromGpu(outImage.img, outImageGPUPtr[0], outImage.Size() );
}

#define INSTANTIATE_GPU_ConvolutionImage_SpatiallyInvariantKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionImage_SpatiallyInvariantKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>&    inImage, \
                    ImageBuffer<OutPixelT>&   outImage, \
                    ImageBuffer<KerPixel>&    kernel  \
                    );


template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_SpatiallyInvariantKernel(
    ImageBuffer<InPixelT>&    inImageImg,
    ImageBuffer<VarPixel>&    inImageVar,
    ImageBuffer<MskPixel>&    inImageMsk,
    ImageBuffer<OutPixelT>&   outImageImg,
    ImageBuffer<VarPixel>&    outImageVar,
    ImageBuffer<MskPixel>&    outImageMsk,
    ImageBuffer<KerPixel>&    kernel
)
{
    int kernelW=kernel.width;
    int kernelH=kernel.height;

    gpu::GpuMemOwner<InPixelT> inImageGPUImg;
    inImageGPUImg.Transfer(inImageImg);
    if (inImageGPUImg.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for input image");
    gpu::GpuMemOwner<VarPixel> inImageGPUVar;
    inImageGPUVar.Transfer(inImageVar);
    if (inImageGPUVar.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for input variance");
    gpu::GpuMemOwner<MskPixel> inImageGPUMsk;
    inImageGPUMsk.Transfer(inImageMsk);
    if (inImageGPUMsk.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for input mask");

    int shMemSize=gpu::GetCudaCurSMSharedMemorySize()-90;

    if (!gpu::IsSufficientSharedMemoryAvailable(kernelW,kernelH,sizeof(KerPixel)))
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough shared memory on GPU for required kernel size");

    //allocate kernel on GPU
    gpu::GpuMemOwner<KerPixel > basisKernelGPU;
    basisKernelGPU.Transfer(kernel);
    if (basisKernelGPU.ptr==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for kernel");

    // allocate array of output image planes on GPU
    vector< OutPixelT* > outImageGPUPtrImg(1);
    vector< VarPixel*  > outImageGPUPtrVar(1);
    vector< MskPixel*  > outImageGPUPtrMsk(1);

    vector< gpu::GpuMemOwner<OutPixelT> > outImageGPU_OwnerImg(1);
    vector< gpu::GpuMemOwner<VarPixel > > outImageGPU_OwnerVar(1);
    vector< gpu::GpuMemOwner<MskPixel > > outImageGPU_OwnerMsk(1);

    outImageGPUPtrImg[0]=outImageGPU_OwnerImg[0].Alloc( outImageImg.Size());
    if (outImageGPUPtrImg[0]==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for output image");
    outImageGPUPtrVar[0]=outImageGPU_OwnerVar[0].Alloc( outImageVar.Size());
    if (outImageGPUPtrVar[0]==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for output variance");
    outImageGPUPtrMsk[0]=outImageGPU_OwnerMsk[0].Alloc( outImageMsk.Size());
    if (outImageGPUPtrMsk[0]==NULL)
        throw LSST_EXCEPT(pexExcept::MemoryException, "Not enough memory on GPU available for output mask");

    OutPixelT** outImageGPUImg   =gpu::TransferToGpu(&outImageGPUPtrImg[0],1);
    VarPixel**  outImageGPUVar   =gpu::TransferToGpu(&outImageGPUPtrVar[0],1);
    MskPixel**  outImageGPUMsk   =gpu::TransferToGpu(&outImageGPUPtrMsk[0],1);

    int blockN=gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

    mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<OutPixelT,InPixelT>(
        inImageGPUImg.ptr, inImageImg.width, inImageImg.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPUImg,
        blockN,
        shMemSize
    );
    //square kernel
    for (int y=0; y<kernelH; y++)
        for (int x=0; x<kernelW; x++)
            kernel.Pixel(x,y) *= kernel.Pixel(x,y);

    gpu::CopyFromGpu(outImageImg.img, outImageGPUPtrImg[0], outImageImg.Size() );

    basisKernelGPU.CopyToGpu(kernel);

    cudaGetLastError(); //clear last error

    mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<VarPixel,VarPixel>(
        inImageGPUVar.ptr, inImageVar.width, inImageVar.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPUVar,
        blockN,
        shMemSize
    );

    cudaThreadSynchronize();
    if (cudaGetLastError()!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "GPU variance calculation failed to run");

    mathDetailGpu::Call_SpatiallyInvariantMaskConvolutionKernel(
        inImageGPUMsk.ptr, inImageMsk.width, inImageMsk.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPUMsk,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError()!=cudaSuccess)
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "GPU mask calculation failed to run");

    gpu::CopyFromGpu(outImageVar.img, outImageGPUPtrVar[0], outImageVar.Size() );
    gpu::CopyFromGpu(outImageMsk.img, outImageGPUPtrMsk[0], outImageMsk.Size() );
}

#define INSTANTIATE_GPU_ConvolutionMI_SpatiallyInvariantKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionMI_SpatiallyInvariantKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>&    inImageImg,  \
                    ImageBuffer<VarPixel>&    inImageVar,  \
                    ImageBuffer<MskPixel>&    inImageMsk,  \
                    ImageBuffer<OutPixelT>&   outImageImg, \
                    ImageBuffer<VarPixel>&    outImageVar, \
                    ImageBuffer<MskPixel>&    outImageMsk, \
                    ImageBuffer<KerPixel>&    kernel   \
                    );

/*
 * Explicit instantiation
 */
/// \cond

#define INSTANTIATE(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionImage_LinearCombinationKernel(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionMI_LinearCombinationKernel(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionImage_SpatiallyInvariantKernel(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionMI_SpatiallyInvariantKernel(OutPixelT,InPixelT)


INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
/// \endcond

}
}
}
} //namespace lsst::afw::math::detail ends

#endif //GPU_BUILD



