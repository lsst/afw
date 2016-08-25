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
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
//Just for PrintCudaDeviceInfo
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"

int const defaultInterpLen = 20;

using namespace std;
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
    int const dimX = imgA.getWidth();
    int const dimY = imgA.getHeight();

    if (dimX != imgB.getWidth() || dimY != imgB.getHeight()) return NAN;

    double sqSum = 0;
    double avgSum = 0;
    int cnt = 0;

    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {
            const double valA = imgA(x, y);
            const double valB = imgB(x, y);
            if (std::isnan(valA) && std::isnan(valB)) continue;
            if (std::isinf(valA) && std::isinf(valB)) continue;

            cnt++;
            avgSum += (valA + valB) / 2;
            double const diff = valA - valB;
            sqSum += diff * diff;
        }
    }
    double rmsd = sqrt(sqSum / cnt);
    double avg = avgSum / cnt;
    return rmsd / avg;
}


//Returns number of different values
template <typename T>
double DiffCnt(afwImage::Mask<T> const& imgA, afwImage::Mask<T> const& imgB)
{
    typedef long long unsigned int Bitint;

    int const dimX = imgA.getWidth();
    int const dimY = imgA.getHeight();

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
typename T::SinglePixel const GetEdgePixel(T& x)
{
    return afwMath::edgePixel< T >( typename afwImage::detail::image_traits< T >::image_category() );
}

template<typename T>
void TimeOneKernelMI(
    const afwImage::MaskedImage<T>  inImg,
    const afwImage::Wcs::Ptr destWcs,
    const afwImage::Wcs::Ptr srcWcs,
    afwMath::WarpingControl wctrlCPU,
    afwMath::WarpingControl wctrlGPU,
    int order            // redundant, but easier this way
)
{
    afwImage::MaskedImage<T>       resMI   (inImg.getDimensions());
    afwImage::MaskedImage<T>       resMIGpu(inImg.getDimensions());

    int const sizeRepMax = 1;
    int const GPUrepMul = 20;
    int const repCpu = sizeRepMax * 5 / order;
    int const repGpu = GPUrepMul * sizeRepMax * 5 / order;

    // warp masked image
    time_t maskedImgCpuStart = clock();
    for (int i = 0; i < repCpu; i++) {
        warpImage(resMI, *destWcs, inImg, *srcWcs, wctrlCPU);
    }
    double maskedImgCpuTime = DiffTime(maskedImgCpuStart, clock()) / repCpu;

    time_t maskedImgGpuStart = clock();
    for (int i = 0; i < repGpu; i++) {
        warpImage(resMIGpu, *destWcs, inImg, *srcWcs, wctrlGPU);
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
    int const order,
    const afwImage::Wcs::Ptr destWcs,
    const afwImage::Wcs::Ptr srcWcs,
    int const interpLen
)
{
    const lsst::afw::gpu::DevicePreference selCPU = lsst::afw::gpu::USE_CPU;
    const lsst::afw::gpu::DevicePreference selGPU = lsst::afw::gpu::AUTO;

    std::ostringstream os;
    os << "lanczos" << order;
    auto lanczosKernelName = os.str();

    afwMath::WarpingControl lanCPU(lanczosKernelName, "", 0, interpLen, selCPU);
    afwMath::WarpingControl lanGPU(lanczosKernelName, "", 0, interpLen, selGPU);

    afwImage::Image<T>       resPI   (inImg.getDimensions());
    afwImage::Image<T>       resPIGpu(inImg.getDimensions());

    int const sizeRepMax = 1;
    int const GPUrepMul = 20;
    int const repCpu = sizeRepMax * 5 / order;
    int const repGpu = GPUrepMul * sizeRepMax * 5 / order;

    // warp plain image
    time_t plainImgCpuStart = clock();
    for (int i = 0; i < repCpu; i++) {
        warpImage(resPI, *destWcs, inImg, *srcWcs, lanCPU);
    }
    double plainImgCpuTime = DiffTime(plainImgCpuStart, clock()) / repCpu;

    time_t plainImgGpuStart = clock();
    for (int i = 0; i < repGpu; i++) {
        warpImage(resPIGpu, *destWcs, inImg, *srcWcs, lanGPU);
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
    const lsst::afw::gpu::DevicePreference selCPU = lsst::afw::gpu::USE_CPU;
    const lsst::afw::gpu::DevicePreference selGPU = lsst::afw::gpu::AUTO;

    auto const lanczosKernelName = "lanczos2";

    const afwImage::MaskedImage<double>  inMIDbl = inImgDbl;
    const afwImage::MaskedImage<float>   inMIFlt = inImgFlt;
    int const sizeX = inMIDbl.getWidth();
    int const sizeY = inMIDbl.getHeight();

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
        afwMath::LanczosWarpingKernel lanKernel(2);
        afwImage::MaskedImage<float>       resGpu(15, 15);
        afwMath::WarpingControl lanGPU(lanczosKernelName, "", 0, 40, lsst::afw::gpu::USE_GPU);
        warpImage(resGpu, wcs1, inImgFlt, wcs2, lanGPU);
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
	    afwMath::LanczosWarpingKernel lanKernel(i);
	    afwMath::WarpingControl wctrlCPU(lanczosKernelName, "", 0, defaultInterpLen, selCPU);
        afwMath::WarpingControl wctrlGPU(lanczosKernelName, "", 0, defaultInterpLen, selGPU);
        TimeOneKernelMI(inMIFlt, wcs1.clone(), wcs2.clone(), wctrlCPU, wctrlGPU, i);
    }

    cout << endl;
    cout << "        Masked Image<double>, Lanczos kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        afwMath::LanczosWarpingKernel lanKernel(i);
        afwMath::WarpingControl wctrlCPU(lanczosKernelName, "", 0, defaultInterpLen, selCPU);
        afwMath::WarpingControl wctrlGPU(lanczosKernelName, "", 0, defaultInterpLen, selGPU);
        TimeOneKernelMI(inMIDbl, wcs1.clone(), wcs2.clone(), wctrlCPU, wctrlGPU, i);
    }

    cout << endl;
    cout << "        Masked Image<float>, Lanczos kernel, bilinear mask kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        char kernelNameBuf[30];
        sprintf(kernelNameBuf, "lanczos%d",i);
	    afwMath::WarpingControl wctrlCPU( kernelNameBuf, "bilinear", 0, defaultInterpLen, selCPU);
        afwMath::WarpingControl wctrlGPU( kernelNameBuf, "bilinear", 0, defaultInterpLen, selGPU);
        TimeOneKernelMI(inMIFlt, wcs1.clone(), wcs2.clone(), wctrlCPU, wctrlGPU, i);
    }

    cout << endl;
    cout << "        Masked Image<double>, Lanczos kernel, bilinear mask kernel" << endl;
    cout << "Order  CPU time  GPU time  Speedup     Image Dev      Variance Dev   Mask Diff" << endl;
    PrintSeparator();

    for (int i = 2; i < 6; i++) {
        char kernelNameBuf[30];
        sprintf(kernelNameBuf, "lanczos%d",i);
        afwMath::WarpingControl wctrlCPU( kernelNameBuf, "bilinear", 0, defaultInterpLen, selCPU);
        afwMath::WarpingControl wctrlGPU( kernelNameBuf, "bilinear", 0, defaultInterpLen, selGPU);
        TimeOneKernelMI(inMIDbl, wcs1.clone(), wcs2.clone(), wctrlCPU, wctrlGPU, i);
    }
}

void TimeGpu(int argc, char**argv)
{
    string inImagePath = GetInputImagePath(argc, argv);

    const afwImage::MaskedImage<float>    inImgFlt(inImagePath);
    const afwImage::MaskedImage<double>   inImgDbl(inImagePath);

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


