// -*- lsst-c++ -*-

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
* @brief Times the speedup of GPU accelerated convolution
*
* @author Kresimir Cosic
*
* @ingroup afw
*/


#include <typeinfo>
#include <cstdio>

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <ctime>

#include "lsst/utils/Utils.h"
#include "lsst/utils/ieee.h"
#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
//Just for PrintCudaDeviceInfo
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"


using namespace std;
namespace pexEx = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;
namespace afwMath  = lsst::afw::math;
namespace afwGeom  = lsst::afw::geom;

typedef int TestResult;
typedef afwMath::Kernel::Pixel  KerPixel;
typedef afwImage::VariancePixel VarPixel;
typedef afwImage::MaskPixel     MskPixel;

// returns time difference in seconds
double DiffTime(clock_t start, clock_t end)
{
    double tm;
    tm = (double)difftime(end, start);
    tm = abs(tm);
    tm /= CLOCKS_PER_SEC;
    return tm;
}

//Calculates relative RMSD (coefficient of variation of the root-mean-square deviation)
template <typename T1, typename T2>
double CvRmsd(afwImage::Image<T1>& imgA, afwImage::Image<T2>& imgB)
{
    const int dimX = imgA.getWidth();
    const int dimY = imgA.getHeight();

    if (dimX != imgB.getWidth() || dimY != imgB.getHeight()) return NAN;

    double sqSum = 0;
    double avgSum = 0;
    int cnt = 0;

    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {
            const double valA = imgA(x, y);
            const double valB = imgB(x, y);
            if (lsst::utils::isnan(valA) && lsst::utils::isnan(valB)) continue;
            if (lsst::utils::isinf(valA) && lsst::utils::isinf(valB)) continue;

            cnt++;
            avgSum += (valA + valB) / 2;
            const double diff = valA - valB;
            sqSum += diff * diff;
        }
    }
    double rmsd = sqrt(sqSum / cnt);
    double avg = avgSum / cnt;
    return rmsd / avg;
}


//Returns number of different values
template <typename T>
double DiffCnt(afwImage::Mask<T>& imgA, afwImage::Mask<T>& imgB)
{
    typedef long long unsigned int Bitint;

    const int dimX = imgA.getWidth();
    const int dimY = imgA.getHeight();

    if (dimX != imgB.getWidth() || dimY != imgB.getHeight()) return NAN;

    int cnt = 0;

    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {
            const T valA = imgA(x, y);
            const T valB = imgB(x, y);

            if (valA != valB) cnt++;
        }
    }
    return cnt;
}


void PrintSeparator()
{
    for (int i = 0; i < 79; i++)
        cout << "=";
    cout << endl;
}

typedef afwMath::LinearCombinationKernel LinearCombinationKernel;

afwMath::LinearCombinationKernel::Ptr  ConstructLinearCombinationKernel(
    const unsigned kernelW,
    const unsigned kernelH,
    const int basisKernelN,
    const int order,
    int sizeX,
    int sizeY,
    const bool isPolynomial = false
)
{
    double const MinSigma = 1.5;
    double const MaxSigma = 4.5;

    // construct basis kernels
    const int kernelN = basisKernelN;
    afwMath::KernelList kernelList;
    for (int ii = 0; ii < kernelN; ++ii) {
        double majorSigma = (ii == 1) ? MaxSigma : MinSigma;
        double minorSigma = (ii == 2) ? MinSigma : MaxSigma;
        double angle = 0.0;
        if (ii > 2) angle = ii / 10;
        afwMath::GaussianFunction2<afwMath::Kernel::Pixel> gaussFunc(majorSigma, minorSigma, angle);
        afwMath::Kernel::Ptr basisKernelPtr(
            new afwMath::AnalyticKernel(kernelW, kernelH, gaussFunc)
        );

        kernelList.push_back(basisKernelPtr);
    }

    // construct spatially varying linear combination kernel
    afwMath::LinearCombinationKernel::Ptr kernelPtr;
    if (isPolynomial) {
        afwMath::PolynomialFunction2<double> polyFunc(order);
        kernelPtr = afwMath::LinearCombinationKernel::Ptr(
                        new LinearCombinationKernel(kernelList, polyFunc)
                    );
    } else {
        afwMath::Chebyshev1Function2<double> chebyFunc(order,
                lsst::afw::geom::Box2D(
                    lsst::afw::geom::Point2D(-sizeX / 3.0, -sizeY / 4.0),
                    lsst::afw::geom::Point2D( sizeX + 100.0,  sizeY + 200.0)
                )
                                                      );
        kernelPtr = afwMath::LinearCombinationKernel::Ptr(
                        new LinearCombinationKernel(kernelList, chebyFunc)
                    );
    }
    LinearCombinationKernel& kernel = *kernelPtr;

    // Get copy of spatial parameters (all zeros), set and feed back to the kernel
    vector<vector<double> > polyParams = kernel.getSpatialParameters();

    // Set spatial parameters for basis kernel 0
    polyParams[0][0] =  1.0;
    polyParams[0][1] = -0.5 / 100;
    polyParams[0][2] = -0.5 / 100;
    if (order > 1 ) {
        polyParams[0][3] = -0.1 / sizeX;
        polyParams[0][4] = -0.2 / sizeY;
        polyParams[0][5] =  0.4 / sizeX;
    }
    for (int i = 1; i < kernelN; i++) {
        // Set spatial function parameters of other basis kernels
        polyParams[i][0] = polyParams[0][0] - i * 0.2;
        polyParams[i][1] = polyParams[0][1] + i * 0.1;
        polyParams[i][2] = polyParams[0][2] + i * 0.1;

        for (int jj = 3; jj < (order + 1)*(order + 2) / 2; jj++) {
            polyParams[i][jj] = i * jj * 0.1 / 600;
        }
    }

    kernel.setSpatialParameters(polyParams);
    return kernelPtr;
}

string GetInputImagePath(int argc, char **argv)
{
    string inImagePath;
    if (argc < 2) {
        try {
            string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/med.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            cerr << "Usage: convolveGPU [fitsFile]" << endl;
            cerr << "fitsFile is the path to a masked image" << endl;
            cerr << "\nError: setup afwdata or specify fitsFile.\n" << endl;
            exit(EXIT_FAILURE);
        }
    }
    else {
        inImagePath = string(argv[1]);
    }
    return inImagePath;
}

afwMath::FixedKernel::Ptr ConstructKernel(
    const double kernelCols,
    const double kernelRows,
    const double majorSigma = 2.5,
    const double minorSigma = 2.0,
    const double angle = 0.5,
    const double denormalizationFactor = 47.3
)
{
    afwMath::GaussianFunction2<KerPixel> gaussFunc(majorSigma, minorSigma, angle);
    afwMath::AnalyticKernel analyticKernel(kernelCols, kernelRows, gaussFunc);
    lsst::afw::image::Image<KerPixel> analyticImage(analyticKernel.getDimensions());
    (void)analyticKernel.computeImage(analyticImage, true);
    analyticImage *= denormalizationFactor;
    afwMath::FixedKernel::Ptr fixedKernel(new afwMath::FixedKernel(analyticImage));
    return fixedKernel;
}


string Sel(bool b, const char* onTrue, const char* onFalse)
{
    return b ? string(onTrue) : string(onFalse);
}

string DecimalPlaces(int places, double val)
{
    stringstream ss;
    ss << fixed << showpoint << setprecision(places) << val;
    return ss.str();
}

template<typename T>
void TimeOneKernelMI(
    const afwImage::MaskedImage<T>  inImg,
    afwMath::Kernel::Ptr   kernel,
    bool                   doNormalizeKernel = true
)
{
    const int kernelSizeX = kernel->getWidth();
    const int kernelSizeY = kernel->getHeight();

    const afwImage::MaskedImage<T>  inMI = inImg;

    afwMath::ConvolutionControl cctrlXGpu  (doNormalizeKernel, false, 0, lsst::afw::gpu::AUTO);
    afwMath::ConvolutionControl cctrlXCpu  (doNormalizeKernel, false, 0, lsst::afw::gpu::USE_CPU);

    afwImage::MaskedImage<T> resMI   (inMI.getDimensions());
    afwImage::MaskedImage<T> resMIGpu(inMI.getDimensions());

    const int GPUrep = 4;
    const int sizeRepMax = 2;

    int repCpu = sizeRepMax * 18 / kernelSizeX;
    int repGpu = GPUrep * sizeRepMax * 18 / kernelSizeX;

    // convolve masked image
    time_t maskedImgCpuStart = clock();
    for (int i = 0; i < repCpu; i++)
        afwMath::convolve(resMI   , inMI, *kernel, cctrlXCpu);
    double maskedImgCpuTime = DiffTime(maskedImgCpuStart, clock()) / repCpu;

    time_t maskedImgGpuStart = clock();
    for (int i = 0; i < repGpu; i++)
        afwMath::convolve(resMIGpu, inMI, *kernel, cctrlXGpu);
    double maskedImgGpuTime = DiffTime(maskedImgGpuStart, clock()) / repGpu;

    double diffMIImg = CvRmsd(*resMI.getImage()   , *resMIGpu.getImage());
    double diffMIVar = CvRmsd(*resMI.getVariance(), *resMIGpu.getVariance());
    double diffMIMsk = DiffCnt(*resMI.getMask()   , *resMIGpu.getMask());

    cout << setw(2) << setfill('0') << kernelSizeX << "x" << setw(2) << kernelSizeY << setfill(' ')
         << setw(8) << DecimalPlaces(3, maskedImgCpuTime) << " s "
         << setw(8) << DecimalPlaces(4, maskedImgGpuTime) << " s "
         << setw(6) << DecimalPlaces(1, maskedImgCpuTime / maskedImgGpuTime) << "x "
         << setw(16) << diffMIImg
         << setw(16) << diffMIVar
         << setw(9) << diffMIMsk
         << endl;
}

template<typename T>
void TimeOneKernelPI(
    const afwImage::MaskedImage<T>  inImg,
    afwMath::Kernel::Ptr   kernel,
    const int sizeRepMax,
    bool                   doNormalizeKernel = false
)
{
    const int kernelSizeX = kernel->getWidth();
    const int kernelSizeY = kernel->getHeight();

    const afwImage::MaskedImage<T>  inMI = inImg;
    const afwImage::Image<T> inPI = *inMI.getImage();

    afwMath::ConvolutionControl cctrlXGpu  (doNormalizeKernel, false, 0, lsst::afw::gpu::AUTO);
    afwMath::ConvolutionControl cctrlXCpu  (doNormalizeKernel, false, 0, lsst::afw::gpu::USE_CPU);

    afwImage::Image<T>       resPI   (inMI.getDimensions());
    afwImage::Image<T>       resPIGpu(inMI.getDimensions());

    const int GPUrep = 4;
    int repCpu = sizeRepMax * 18 / kernelSizeX;
    int repGpu = GPUrep * sizeRepMax * 18 / kernelSizeX;


    // convolve plain image
    time_t plainImgCpuStart = clock();
    for (int i = 0; i < repCpu; i++)
        afwMath::convolve(resPI   , inPI, *kernel, cctrlXCpu);
    double plainImgCpuTime = DiffTime(plainImgCpuStart, clock()) / repCpu;

    time_t plainImgGpuStart = clock();
    for (int i = 0; i < repGpu; i++)
        afwMath::convolve(resPIGpu, inPI, *kernel, cctrlXGpu);
    double plainImgGpuTime = DiffTime(plainImgGpuStart, clock()) / repGpu;

    double diffPI = CvRmsd(resPI, resPIGpu);

    cout << setw(2) << setfill('0') << kernelSizeX << "x" << setw(2) << kernelSizeY << setfill(' ');
    cout << setw(8) <<  DecimalPlaces(3, plainImgCpuTime) << " s " ;
    cout << setw(8) <<  DecimalPlaces(4, plainImgGpuTime) << " s " ;
    cout << setw(6) <<  DecimalPlaces(1, plainImgCpuTime / plainImgGpuTime) << "x " ;
    cout << setw(16) << diffPI ;
    cout << endl;
}

template < typename T>
void TestLCChebKernel(const afwImage::MaskedImage<T>  inImg,
                      const int order,
                      const int basisKernelCount,
                      const int skipSize = 2,
                      const int gpuRep = 1
                     )
{
    const int sizeX = inImg.getWidth();
    const int sizeY = inImg.getHeight();

    const afwImage::Image<T> inPIDbl = *inImg.getImage();


    cout << endl;
    string typeStr = sizeof(T) == sizeof(float) ? string("float") : string("double");
    cout << "        Plain Image<" << typeStr << ">, LC kernel with Chebyshev spatial function" << endl;


    cout << "     Number of basis kernels: " << basisKernelCount << endl;
    cout << "               Spatial order: " << order << endl;
    cout << " Size   CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 6; i < 19; i += skipSize) {
        LinearCombinationKernel::Ptr linCoKernelCheb = ConstructLinearCombinationKernel(
                    i, i,  //size
                    basisKernelCount, //basis kernel count
                    order,       //order
                    sizeX, sizeY,  //image size
                    false //chebyshev
                );

        TimeOneKernelPI(inImg, linCoKernelCheb, gpuRep * 2);
    }

    cout << endl;
    cout << "        Masked Image<" << typeStr << ">, LC kernel with Chebyshev spatial function" << endl;
    cout << "     Number of basis kernels: " << basisKernelCount << endl;
    cout << "               Spatial order: " << order << endl;
    cout << " Size   CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 6; i < 19; i += skipSize) {
        LinearCombinationKernel::Ptr linCoKernelCheb = ConstructLinearCombinationKernel(
                    i, i,  //size
                    basisKernelCount,       //basis kernel count
                    order,       //order
                    sizeX, sizeY,  //image size
                    false //chebyshev
                );

        TimeOneKernelMI(inImg, linCoKernelCheb, gpuRep);
    }
}

void TestConvGpu(
    const afwImage::MaskedImage<double>  inImgDbl,
    const afwImage::MaskedImage<float>   inImgFlt
)
{

    const afwImage::MaskedImage<double>  inMIDbl = inImgDbl;
    const afwImage::MaskedImage<float>   inMIFlt = inImgFlt;
    const int sizeX = inMIDbl.getWidth();
    const int sizeY = inMIDbl.getHeight();

    {
        // do one convolution and discard the result
        // because first convolution has to initialize GPU, thus using aditional time
        afwMath::FixedKernel::Ptr     fixedKernel = ConstructKernel(7, 7);
        afwMath::ConvolutionControl   cctrlXGpu  (false, false, 0, lsst::afw::gpu::AUTO);
        afwImage::MaskedImage<float>  resMI   (inMIFlt.getDimensions());
        afwMath::convolve(resMI, inMIFlt, *fixedKernel, cctrlXGpu);
    }

    const afwImage::Image<double> inPIDbl = *inMIDbl.getImage();
    const afwImage::Image<float>  inPIFlt = *inMIFlt.getImage();

    cout << "Image size: " << sizeX << " x " << sizeY << endl;

    cout << endl;
    cout << "        Plain Image<float>, fixed kernel" << endl;
    cout << " Size   CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 6; i < 19; i++) {
        afwMath::FixedKernel::Ptr fixedKernel = ConstructKernel(i, i);
        TimeOneKernelPI(inMIFlt, fixedKernel, 4);
    }

    cout << endl;
    cout << "        Masked Image<float>, fixed kernel" << endl;
    cout << " Size   CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 6; i < 19; i++) {
        afwMath::FixedKernel::Ptr fixedKernel = ConstructKernel(i, i);
        TimeOneKernelMI(inMIFlt, fixedKernel, 4);
    }

    TestLCChebKernel(inMIFlt, 2, 1);
    TestLCChebKernel(inMIFlt, 2, 4);
    TestLCChebKernel(inMIFlt, 3, 4);
    TestLCChebKernel(inMIDbl, 3, 4);
}

void TimeGpu(int argc, char**argv)
{
    string inImagePath = GetInputImagePath(argc, argv);

    afwImage::MaskedImage<float>    inImgFlt(inImagePath);
    afwImage::MaskedImage<double>   inImgDbl(inImagePath);

    TestConvGpu(inImgDbl, inImgFlt);
}

int main(int argc, char **argv)
{
    int status = EXIT_SUCCESS;

    if (lsst::afw::gpu::isGpuBuild()) {
        lsst::afw::gpu::detail::PrintCudaDeviceInfo();
    } else {
        cout << "AFW not compiled with GPU support. Exiting." << endl;
        return EXIT_SUCCESS;
    }

    PrintSeparator();
    cout << endl;
    cout << "Note: Dev =  coefficient of variation of RMSD" << endl;
    cout << endl;

    TimeGpu(argc, argv);

    // Check for memory leaks
    if (lsst::daf::base::Citizen::census(0) == 0) {
        cerr << "No leaks detected" << endl;
    } else {
        cerr << "Leaked memory blocks:" << endl;
        lsst::daf::base::Citizen::census(cerr);
        status = EXIT_FAILURE;
    }
    return status;
}


