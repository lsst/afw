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
#include <math.h>
#include <time.h>

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
//Just for PrintCudaDeviceInfo
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"

const int defaultInterpLen = 20;

using namespace std;
using lsst::pex::logging::Trace;
namespace pexEx = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;
namespace afwMath  = lsst::afw::math;
namespace afwGeom  = lsst::afw::geom;

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
double CvRmsd(const afwImage::Image<T1>& imgA, const afwImage::Image<T2>& imgB)
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
            if (isnan(valA) && isnan(valB)) continue;
            if (isinf(valA) && isinf(valB)) continue;

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
double DiffCnt(const afwImage::Mask<T>& imgA, const afwImage::Mask<T>& imgB)
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

string GetInputFileName(int argc, char **argv)
{
    string imgBaseFileName;
    if (argc < 2) {
        string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "Usage: convolveGPU fitsFile" << endl;
            std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << endl;
            std::cerr << "I can take a default file from AFWDATA_DIR, but it's not defined." << endl;
            std::cerr << "Is afwdata set up?\n" << endl;
            exit(EXIT_FAILURE);
        }
        else {
            imgBaseFileName = afwdata + "/data/med";
            //imgBaseFileName = afwdata + "/data/medsub";
            //imgBaseFileName = afwdata + "/data/871034p_1_MI";
            cout << "Using image: " << imgBaseFileName << endl;
        }
    }
    else {
        imgBaseFileName = string(argv[1]);
    }
    return imgBaseFileName;
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
    const int order,
    const afwImage::Wcs::Ptr destWcs,
    const afwImage::Wcs::Ptr srcWcs,
    const int interpLen
)
{
    const lsst::afw::gpu::DevicePreference selCPU = lsst::afw::gpu::USE_CPU;
    const lsst::afw::gpu::DevicePreference selGPU = lsst::afw::gpu::AUTO;

    afwMath::LanczosWarpingKernel lanKernel(order);

    afwImage::MaskedImage<T>       resMI   (inImg.getDimensions());
    afwImage::MaskedImage<T>       resMIGpu(inImg.getDimensions());

    const int sizeRepMax = 1;
    const int GPUrepMul = 20;
    const int repCpu = sizeRepMax * 5 / order;
    const int repGpu = GPUrepMul * sizeRepMax * 5 / order;

    int numGoodPixels;
    int numGoodPixelsGpu;

    // warp masked image
    time_t maskedImgCpuStart = clock();
    for (int i = 0; i < repCpu; i++) {
        numGoodPixels = warpImage(resMI, *destWcs, inImg, *srcWcs, lanKernel, interpLen, selCPU);
    }
    double maskedImgCpuTime = DiffTime(maskedImgCpuStart, clock()) / repCpu;

    time_t maskedImgGpuStart = clock();
    for (int i = 0; i < repGpu; i++) {
        numGoodPixelsGpu = warpImage(resMIGpu, *destWcs, inImg, *srcWcs, lanKernel, interpLen, selGPU);
    }
    double maskedImgGpuTime = DiffTime(maskedImgGpuStart, clock()) / repGpu;

    double diffMIImg = CvRmsd(*resMI.getImage()   , *resMIGpu.getImage());
    double diffMIVar = CvRmsd(*resMI.getVariance(), *resMIGpu.getVariance());
    double diffMIMsk = DiffCnt(*resMI.getMask()   , *resMIGpu.getMask());

    cout << "  " << setw(2) << setfill('0') << order << setfill(' ')
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
    const afwImage::Image<T>  inImg,
    const int order,
    const afwImage::Wcs::Ptr destWcs,
    const afwImage::Wcs::Ptr srcWcs,
    const int interpLen
)
{
    const lsst::afw::gpu::DevicePreference selCPU = lsst::afw::gpu::USE_CPU;
    const lsst::afw::gpu::DevicePreference selGPU = lsst::afw::gpu::AUTO;

    afwMath::LanczosWarpingKernel lanKernel(order);

    afwImage::Image<T>       resPI   (inImg.getDimensions());
    afwImage::Image<T>       resPIGpu(inImg.getDimensions());

    const int sizeRepMax = 1;
    const int GPUrepMul = 20;
    const int repCpu = sizeRepMax * 5 / order;
    const int repGpu = GPUrepMul * sizeRepMax * 5 / order;

    int numGoodPixels;
    int numGoodPixelsGpu;

    // warp plain image
    time_t plainImgCpuStart = clock();
    for (int i = 0; i < repCpu; i++) {
        numGoodPixels = warpImage(resPI, *destWcs, inImg, *srcWcs, lanKernel, interpLen, selCPU);
    }
    double plainImgCpuTime = DiffTime(plainImgCpuStart, clock()) / repCpu;

    time_t plainImgGpuStart = clock();
    for (int i = 0; i < repGpu; i++) {
        numGoodPixelsGpu = warpImage(resPIGpu, *destWcs, inImg, *srcWcs, lanKernel, interpLen, selGPU);
    }
    double plainImgGpuTime = DiffTime(plainImgGpuStart, clock()) / repGpu;

    double diffPI = CvRmsd(resPI, resPIGpu);

    cout << "  " << setw(2) << setfill('0') << order << setfill(' ');
    cout << setw(8) <<  DecimalPlaces(3, plainImgCpuTime) << " s " ;
    cout << setw(8) <<  DecimalPlaces(4, plainImgGpuTime) << " s " ;
    cout << setw(6) <<  DecimalPlaces(1, plainImgCpuTime / plainImgGpuTime) << "x " ;
    cout << setw(16) << diffPI ;
    cout << endl;
}

void TestWarpGpu(
    const afwImage::MaskedImage<double>  inImgDbl,
    const afwImage::MaskedImage<float>   inImgFlt
)
{

    const afwImage::MaskedImage<double>  inMIDbl = inImgDbl;
    const afwImage::MaskedImage<float>   inMIFlt = inImgFlt;
    const int sizeX = inMIDbl.getWidth();
    const int sizeY = inMIDbl.getHeight();

    const afwImage::Image<double> inPIDbl = *inMIDbl.getImage();
    const afwImage::Image<float>  inPIFlt = *inMIFlt.getImage();

    cout << "Image size: " << sizeX << " x " << sizeY << endl;

    Eigen::Matrix2d  m1(2, 2);
    Eigen::Matrix2d  m2(2, 2);
    m1 << 1 , 0, 0, 1;
    m2 <<  0.88 , 0.2, -0.4, 1.12;
    const afwGeom::Point2D origin(0, 0);
    afwImage::Wcs wcs1(origin, origin, m1);
    afwImage::Wcs wcs2(origin, origin, m2);

    {
        // do one warp and discard the result
        // because first warp has to initialize GPU, thus using aditional time
        lsst::afw::gpu::DevicePreference selGPU = lsst::afw::gpu::USE_GPU;
        afwMath::LanczosWarpingKernel lanKernel(2);
        afwImage::MaskedImage<float>       resGpu(15, 15);
        warpImage(resGpu, wcs1, inImgFlt, wcs2, lanKernel, 40, selGPU);
    }

    cout << endl;
    cout << "        Plain Image<float>, Lanczos kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        TimeOneKernelPI(inPIFlt, i, wcs1.clone(), wcs2.clone(), defaultInterpLen);
    }

    cout << endl;
    cout << "        Plain Image<double>, Lanczos kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        TimeOneKernelPI(inPIDbl, i, wcs1.clone(), wcs2.clone(), defaultInterpLen);
    }

    cout << endl;
    cout << "        Masked Image<float>, Lanczos kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        TimeOneKernelMI(inMIFlt, i, wcs1.clone(), wcs2.clone(), defaultInterpLen);
    }

    cout << endl;
    cout << "        Masked Image<double>, Lanczos kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        TimeOneKernelMI(inMIDbl, i, wcs1.clone(), wcs2.clone(), defaultInterpLen);
    }
}

void TimeGpu(int argc, char**argv)
{
    string baseFileName = GetInputFileName(argc, argv);

    const afwImage::MaskedImage<float>    inImgFlt(baseFileName);
    const afwImage::MaskedImage<double>   inImgDbl(baseFileName);

    TestWarpGpu(inImgDbl, inImgFlt);
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
    cout << "Note: Interpolation length set to " << defaultInterpLen << endl;
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


