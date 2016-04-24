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
* @brief Tests GPU accelerated convolution
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

#include "lsst/daf/base.h"
#include "lsst/utils/ieee.h"
#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
#include "lsst/afw/gpu/DevicePreference.h"
//Just for PrintCudaDeviceInfo
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"



using namespace std;
using lsst::pex::logging::Trace;
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

string NumToStr(double num)
{
    if (lsst::utils::isnan(num))       return string("NAN!!!");
    else if (lsst::utils::isinf(num))  return string("INF!!!");
    else {
        stringstream ss;
        ss << num;
        return ss.str();
    }
}

void PrintSeparator()
{
    for (int i = 0; i < 79; i++)
        cout << "=";
    cout << endl;
}

void PrintKernelSize(afwMath::Kernel::Ptr kernel)
{
    cout << "                 Kernel size: " << kernel->getWidth() << " x " << kernel->getHeight() << endl;
}

typedef afwMath::LinearCombinationKernel LinearCombinationKernel;

afwMath::LinearCombinationKernel::Ptr  ConstructLinearCombinationKernel(
    const unsigned kernelW,
    const unsigned kernelH,
    const int basisKernelN,
    const int order,
    int sizeX,
    int sizeY,
    const bool isPolynomial = false,
    const bool isGaussian = false
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
    }
    if (isGaussian) {
        afwMath::GaussianFunction2<double> gaussFunc(1.3, 2.3, 23.0);
        kernelPtr = afwMath::LinearCombinationKernel::Ptr(
                        new LinearCombinationKernel(kernelList, gaussFunc)
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
    vector<std::vector<double> > polyParams = kernel.getSpatialParameters();

    // Set spatial parameters for basis kernel 0
    polyParams[0][0] =  1.0;
    polyParams[0][1] = -0.5 / 100;
    polyParams[0][2] = -0.5 / 100;
    if (order > 1 && !isGaussian) {
        polyParams[0][3] = -0.1 / sizeX;
        polyParams[0][4] = -0.2 / sizeY;
        polyParams[0][5] =  0.4 / sizeX;
    }
    for (int i = 1; i < kernelN; i++) {
        // Set spatial function parameters of other basis kernels
        polyParams[i][0] = polyParams[0][0] - i * 0.2;
        polyParams[i][1] = polyParams[0][1] + i * 0.1;
        polyParams[i][2] = polyParams[0][2] + i * 0.1;

        if (isGaussian) continue;

        for (int jj = 3; jj < (order + 1)*(order + 2) / 2; jj++) {
            polyParams[i][jj] = i * jj * 0.1 / 600;
        }
    }

    kernel.setSpatialParameters(polyParams);

    PrintKernelSize(kernelPtr);
    cout << "     Number of basis kernels: " << kernel.getNBasisKernels() << endl;
    cout << "               Spatial order: " << order << endl;
    cout << "Coef. count per basis kernel: " << polyParams[0].size() << endl;

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
            cerr << "Warning: tests not run! Setup afwdata if you wish to use the default fitsFile." << endl;
            exit(EXIT_SUCCESS);
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

bool IsErrorAcceptable(double val, double limit)
{
    if (lsst::utils::isnan(val)) return false;
    if (lsst::utils::isinf(val)) return false;
    return val < limit;
}

bool TestConvGpu(
    const afwImage::MaskedImage<double>  inImgDbl,
    const afwImage::MaskedImage<float>   inImgFlt,
    afwMath::Kernel::Ptr   kernel,
    string                 kernelStr,
    bool                   doNormalizeKernel
)
{
    const afwImage::MaskedImage<double>  inMIDbl = inImgDbl;
    const afwImage::MaskedImage<float>   inMIFlt = inImgFlt;
    const int sizeX = inMIDbl.getWidth();
    const int sizeY = inMIDbl.getHeight();

    const afwImage::Image<double> inPIDbl = *inMIDbl.getImage();
    const afwImage::Image<float>  inPIFlt = *inMIFlt.getImage();

    cout << "Image size: " << sizeX << " x " << sizeY;
    cout << "        Kernel normalization: " << Sel(doNormalizeKernel, "on", "off") << endl;

    afwMath::ConvolutionControl cctrlXGpu  (doNormalizeKernel, false, 0, lsst::afw::gpu::USE_GPU);
    afwMath::ConvolutionControl cctrlXCpu  (doNormalizeKernel, false, 0, lsst::afw::gpu::USE_CPU);

    afwImage::MaskedImage<double> resMIDbl(inMIDbl.getDimensions());
    afwImage::MaskedImage<float>  resMIFlt(inMIDbl.getDimensions());
    afwImage::MaskedImage<double> resMIDblGpu(inMIDbl.getDimensions());
    afwImage::MaskedImage<float>  resMIFltGpu(inMIDbl.getDimensions());
    afwImage::Image<double>       resPIDbl(inMIDbl.getDimensions());
    afwImage::Image<float>        resPIFlt(inMIDbl.getDimensions());
    afwImage::Image<double>       resPIDblGpu(inMIDbl.getDimensions());
    afwImage::Image<float>        resPIFltGpu(inMIDbl.getDimensions());

    for (int i = 0; i < int(kernelStr.size()); i++)  cout << " ";
    cout << "       Planes             Image         Variance   Mask" << endl;
    for (int i = 0; i < 79; i++)  cout << "-";
    cout << endl;

    bool isSuccess = true;

    // convolve
    afwMath::convolve(resPIDbl   , inPIDbl, *kernel, cctrlXCpu);
    afwMath::convolve(resPIDblGpu, inPIDbl, *kernel, cctrlXGpu);
    afwMath::convolve(resPIFlt   , inPIFlt, *kernel, cctrlXCpu);
    afwMath::convolve(resPIFltGpu, inPIFlt, *kernel, cctrlXGpu);

    const double errDbl = 5e-16;
    const double errFlt = 5e-7;

    double diffPIDbl = CvRmsd(resPIDbl, resPIDblGpu);
    double diffPIFlt = CvRmsd(resPIFlt, resPIFltGpu);

    if ( !IsErrorAcceptable(diffPIDbl, errDbl) ) isSuccess = false;
    if ( !IsErrorAcceptable(diffPIFlt, errFlt) ) isSuccess = false;

    cout << " Plain Image Dbl " << kernelStr << " Dev: " << setw(11) << diffPIDbl
         << Sel(diffPIDbl > errDbl, "*", " ") << endl;
    cout << " Plain Image Flt " << kernelStr << " Dev: " << setw(11) << diffPIFlt
         << Sel(diffPIFlt > errFlt, "*", " ") << endl;

    afwMath::convolve(resMIDbl   , inMIDbl, *kernel, cctrlXCpu);
    afwMath::convolve(resMIDblGpu, inMIDbl, *kernel, cctrlXGpu);
    afwMath::convolve(resMIFlt   , inMIFlt, *kernel, cctrlXCpu);
    afwMath::convolve(resMIFltGpu, inMIFlt, *kernel, cctrlXGpu);

    double diffMIImgDbl = CvRmsd(*resMIDbl.getImage()   , *resMIDblGpu.getImage());
    double diffMIVarDbl = CvRmsd(*resMIDbl.getVariance(), *resMIDblGpu.getVariance());
    double diffMIMskDbl = DiffCnt(*resMIDbl.getMask()   , *resMIDblGpu.getMask());

    double diffMIImgFlt = CvRmsd(*resMIFlt.getImage()   , *resMIFltGpu.getImage());
    double diffMIVarFlt = CvRmsd(*resMIFlt.getVariance(), *resMIFltGpu.getVariance());
    double diffMIMskFlt = DiffCnt(*resMIFlt.getMask()   , *resMIFltGpu.getMask());

    if ( !IsErrorAcceptable(diffMIImgDbl, errDbl) ) isSuccess = false;
    if ( !IsErrorAcceptable(diffMIVarDbl, errFlt) ) isSuccess = false;
    if ( diffMIMskDbl > 0) isSuccess = false;
    if ( !IsErrorAcceptable(diffMIImgFlt, errFlt) ) isSuccess = false;
    if ( !IsErrorAcceptable(diffMIVarFlt, errFlt) ) isSuccess = false;
    if ( diffMIMskFlt > 0) isSuccess = false;

    cout << "Masked Image Dbl " << kernelStr << " Dev: "
         << setw(11) << diffMIImgDbl << Sel(diffMIImgDbl > errDbl, "*  ", "   ")
         << setw(11) << diffMIVarDbl << Sel(diffMIVarDbl > errFlt, "*  ", "   ")
         << setw( 6) << diffMIMskDbl << Sel(diffMIMskDbl > 0     , "*  ", "   ")
         << endl;
    cout << "Masked Image Flt " << kernelStr << " Dev: "
         << setw(11) << diffMIImgFlt << Sel(diffMIImgFlt > errFlt, "*  ", "   ")
         << setw(11) << diffMIVarFlt << Sel(diffMIVarFlt > errFlt, "*  ", "   ")
         << setw( 6) << diffMIMskFlt << Sel(diffMIMskFlt > 0     , "*  ", "   ")
         << endl;

    if (!isSuccess) {
        cout << "ERROR: Unacceptaby large deviation found!" << endl;
        cout << "The failed tests are marked with *" << endl;
    }

    return isSuccess;
}

bool GpuTestAccuracy(string imgFileName)
{
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    afwGeom::Box2I inputBBox(afwGeom::Point2I(52, 574), afwGeom::Extent2I(76, 80));

    afwImage::MaskedImage<float>    inImgFlt(imgFileName);
    afwImage::MaskedImage<double>   inImgDbl(imgFileName);
    const int sizeX = inImgFlt.getWidth();
    const int sizeY = inImgFlt.getHeight();

    if (sizeX < 500 || sizeY < 700) {
        std::cerr << "Minimum image size is 500 x 700" << endl;
        exit(EXIT_FAILURE);
    }

    bool isSuccess = true;

    for (int i = 0; i < 4; i++) {

        bool doNormalizeKernel = i % 2 == 0;
        bool isBoundedBoxKernel = i / 2 == 0;

        const afwImage::MaskedImage<double>  inMIDblBoxed(inImgDbl, inputBBox, afwImage::LOCAL, true);
        const afwImage::MaskedImage<float>   inMIFltBoxed(inImgFlt, inputBBox, afwImage::LOCAL, true);

        const afwImage::MaskedImage<double> inMIDbl =
            isBoundedBoxKernel ? inMIDblBoxed : inImgDbl;;
        const afwImage::MaskedImage<float>  inMIFlt =
            isBoundedBoxKernel ? inMIFltBoxed : inImgFlt;

        PrintSeparator();
        LinearCombinationKernel::Ptr linCoKernelCheb = ConstructLinearCombinationKernel(
                    4, 5,  //size
                    4,       //basis kernel count
                    3,       //order
                    sizeX, sizeY,  //image size
                    false //chebyshev
                );
        const bool isSuccessCheb = TestConvGpu(inMIDbl, inMIFlt, linCoKernelCheb,
                                               "cheby LC kernel", doNormalizeKernel);
        PrintSeparator();
        LinearCombinationKernel::Ptr linCoKernelPoly = ConstructLinearCombinationKernel(
                    6, 4,  //size
                    3,       //basis kernel count
                    2,       //order
                    sizeX, sizeY,  //image size
                    true //polynomial
                );
        const bool isSuccessPoly = TestConvGpu(inMIDbl, inMIFlt, linCoKernelPoly,
                                               " poly  LC kernel", doNormalizeKernel);
        PrintSeparator();
        LinearCombinationKernel::Ptr linCoKernelCheb11 = ConstructLinearCombinationKernel(
                    3, 3,  //size
                    1,       //basis kernel count
                    1,       //order
                    sizeX, sizeY,  //image size
                    false //chebyshev
                );
        const bool isSuccessCheb11 = TestConvGpu(inMIDbl, inMIFlt, linCoKernelCheb11,
                                     " cheby LC kernel", doNormalizeKernel);
        PrintSeparator();
        afwMath::FixedKernel::Ptr fixedKernel1 = ConstructKernel(5, 7);
        PrintKernelSize(fixedKernel1);
        const bool isSuccessFixed1 = TestConvGpu(inMIDbl, inMIFlt, fixedKernel1,
                                     "invariant kernel", doNormalizeKernel);
        PrintSeparator();
        afwMath::FixedKernel::Ptr fixedKernel2 = ConstructKernel(8, 3, 3.5, 1.3, 9.0, 11);
        PrintKernelSize(fixedKernel2);
        const bool isSuccessFixed2 = TestConvGpu(inMIDbl, inMIFlt, fixedKernel2,
                                     "invariant kernel", doNormalizeKernel);

        isSuccess = isSuccess && isSuccessCheb && isSuccessPoly && isSuccessCheb11
                    && isSuccessFixed1 && isSuccessFixed2;
    }
    return isSuccess;
}

bool GpuTestExceptions(const string imgFileName)
{
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    afwImage::MaskedImage<double> inImg(imgFileName);
    afwImage::MaskedImage<double> resImg(inImg.getDimensions());

    bool isSuccess = true;

    PrintSeparator();

    afwMath::ConvolutionControl cctrlFGpu   (false, false, 0, lsst::afw::gpu::USE_GPU);
    afwMath::ConvolutionControl cctrlFCpu   (false, false, 0, lsst::afw::gpu::USE_CPU);
    afwMath::ConvolutionControl cctrlThrw  (false, false, 0, lsst::afw::gpu::AUTO);
    afwMath::ConvolutionControl cctrlSafe  (false, false, 0, lsst::afw::gpu::AUTO_WITH_CPU_FALLBACK);

    afwMath::FixedKernel::Ptr fixedKernel = ConstructKernel(5, 7);
    afwMath::GaussianFunction2<KerPixel> gaussFunc(0.8, 0.7, 30);
    afwMath::AnalyticKernel analyticKernel(7, 7, gaussFunc);
    afwMath::DeltaFunctionKernel deltaFKernel(11, 17, lsst::afw::geom::Point2I(1));
    LinearCombinationKernel::Ptr linCoKernelGauss = ConstructLinearCombinationKernel(
                4, 5,  //size
                4,       //basis kernel count
                3,       //order
                1, 1,  //image size
                false,  //not polynomial
                true    //gaussian
            );

    cout << endl;
    bool isThrown;
    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, *fixedKernel, cctrlThrw);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with fixed kernel with AUTO "
             << "should have not thrown an exception" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlFGpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (lsst::afw::gpu::isGpuEnabled() && !isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with USE_GPU"
             << "should have thrown an exception because it's acceleration is not supported" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlSafe);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with AUTO_WITH_CPU_FALLBACK"
             << "should not have thrown an exception because it should have fell back to CPU code path" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlThrw);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with AUTO"
             << "should not have thrown an exception because it should have fell back to CPU code path" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlFCpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with USE_CPU"
             << "should not have thrown an exception because it should have used CPU code path" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, analyticKernel, cctrlFGpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with analytic kernel with USE_GPU "
             << "should have not thrown an exception because this specific kernel was spatially invariant" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, *linCoKernelGauss, cctrlFGpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (lsst::afw::gpu::isGpuEnabled() && !isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with linear combination kernel "
             << "with gaussian spatial function with USE_GPU "
             << "should have thrown an exception because gaussian spatial function is not supported" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, *linCoKernelGauss, cctrlThrw);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with linear combination kernel "
             << "with gaussian spatial function with AUTO "
             << "should not have thrown an exception because it should have fell back to CPU execution" << endl;
    }

    if (isSuccess) {
        cout << "All GPU exception tests passed." << endl;
    }
    cout << endl;

    return isSuccess;
}

bool CpuTestExceptions(const string imgFileName)
{
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    afwImage::MaskedImage<double> inImg(imgFileName);
    afwImage::MaskedImage<double> resImg(inImg.getDimensions());

    bool isSuccess = true;

    PrintSeparator();

    afwMath::ConvolutionControl cctrlFGpu   (false, false, 0, lsst::afw::gpu::USE_GPU);
    afwMath::ConvolutionControl cctrlFCpu   (false, false, 0, lsst::afw::gpu::USE_CPU);
    afwMath::ConvolutionControl cctrlThrw  (false, false, 0, lsst::afw::gpu::AUTO);
    afwMath::ConvolutionControl cctrlSafe  (false, false, 0, lsst::afw::gpu::AUTO_WITH_CPU_FALLBACK);

    afwMath::FixedKernel::Ptr fixedKernel = ConstructKernel(5, 7);
    afwMath::GaussianFunction2<KerPixel> gaussFunc(0.8, 0.7, 30);
    afwMath::AnalyticKernel analyticKernel(7, 7, gaussFunc);
    afwMath::DeltaFunctionKernel deltaFKernel(11, 17, lsst::afw::geom::Point2I(1));
    LinearCombinationKernel::Ptr linCoKernelGauss = ConstructLinearCombinationKernel(
                4, 5,  //size
                4,       //basis kernel count
                3,       //order
                1, 1,  //image size
                false,  //not polynomial
                true    //gaussian
            );

    cout << endl;
    bool isThrown;
    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, *fixedKernel, cctrlThrw);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with fixed kernel with AUTO "
             << "should have not thrown an exception" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlFGpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (lsst::afw::gpu::isGpuEnabled() && !isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with USE_GPU"
             << "should have thrown an exception because AFW was not compiled with GPU support" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlSafe);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with AUTO_WITH_CPU_FALLBACK"
             << "should not have thrown an exception because it should have used to CPU execution" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlThrw);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with AUTO"
             << "should not have thrown an exception because it should have fell back to CPU execution" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, deltaFKernel, cctrlFCpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with delta function kernel with USE_CPU"
             << "should not have thrown an exception because it should have used CPU execution" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, analyticKernel, cctrlFGpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (lsst::afw::gpu::isGpuEnabled() && !isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with analytic kernel with USE_GPU "
             << "should have thrown an exception because AFW was not compiled with GPU support" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, *linCoKernelGauss, cctrlFGpu);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (lsst::afw::gpu::isGpuEnabled() && !isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with linear combination kernel "
             << "with gaussian spatial function with USE_GPU "
             << "should have thrown an exception because AFW was not compiled with GPU support" << endl;
    }

    isThrown = false;
    try {
        afwMath::convolve(resImg, inImg, *linCoKernelGauss, cctrlThrw);
    } catch(pexEx::Exception) {
        isThrown = true;
    }

    if (isThrown) {
        isSuccess = false;
        cout << "ERROR: GPU convolution with linear combination kernel "
             << "with gaussian spatial function with AUTO "
             << "should not have thrown an exception because it should have used CPU execution" << endl;
    }

    if (isSuccess) {
        cout << "All GPU exception tests passed (but AFW was not compiled with GPU support!!)." << endl;
        cout << "Additional tests will be performed when GPU acceleration is available." << endl;
    }
    cout << endl;

    return isSuccess;
}

TestResult TestGpu(int argc, char**argv)
{
    lsst::afw::gpu::detail::PrintCudaDeviceInfo();
    string inImageName = GetInputImagePath(argc, argv);

    const bool isSuccess1 = GpuTestAccuracy(inImageName);
    const bool isSuccess2 = GpuTestExceptions(inImageName);
    const bool isSuccess = isSuccess1 && isSuccess2;

    return isSuccess ? EXIT_SUCCESS : EXIT_FAILURE;
}

TestResult TestCpu(int argc, char**argv)
{
    string inImageName = GetInputImagePath(argc, argv);

    const bool isSuccess = CpuTestExceptions(inImageName);

    return isSuccess ? EXIT_SUCCESS : EXIT_FAILURE;
}


int main(int argc, char **argv)
{
    cout << endl;
    cout << "Note: Dev =  coefficient of variation of RMSD" << endl;
    cout << endl;

    GetInputImagePath(argc, argv); // if afwdata not setup then exit

    int status = EXIT_SUCCESS;
    try {
        if (lsst::afw::gpu::isGpuBuild()) {
            status = TestGpu(argc, argv);
        } else {
            status = TestCpu(argc, argv);
        }
    } catch (pexEx::Exception &e) {
        clog << e.what() << endl;
        status = EXIT_FAILURE;
    }

    PrintSeparator();
    cout << endl;
    if (status == EXIT_FAILURE) {
        cout << "Some tests have failed." << endl;
    } else {
        cout << "All tests passed OK." << endl;
    }

    cout << endl;

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


