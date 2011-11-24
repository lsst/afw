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


#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <time.h>

#include "lsst/daf/base.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"

using namespace std;
const string outFileMasked   ("resultMasked");
const string outFileMaskedGPU("resultMaskedGPU");
const string outFilePlain    ("resultPlain_img.fits");
const string outFilePlainGPU ("resultPlainGPU_img.fits");;

#define CALC_PLAIN_IMG
#define CALC_MASKED_IMG

namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

typedef afwMath::Kernel::Pixel KerPixel;
typedef float  InPixel;
typedef double  OutPixel;

double DiffTime (clock_t start, clock_t end)
{
	double tm;
	tm=(double)difftime(end,start);
	if (tm<0) tm=-tm;
	tm/=CLOCKS_PER_SEC;
	return tm;
}

template <typename T1, typename T2>
double OutputDiffStDev(afwImage::Image<T1>& imgA, afwImage::Image<T2>& imgB)
{
    int dimX=imgA.getWidth();
    int dimY=imgA.getHeight();

    int dimBX=imgB.getWidth();
    int dimBY=imgB.getHeight();

    if (dimX!=dimBX || dimY!=dimBY)
        return NAN;

    double var=0;
    for (int x=0;x<dimX;x++)
        for (int y=0;y<dimY;y++) {
            double a=imgA(x,y);
            double b=imgB(x,y);

            double diff=a-b;

            if (isnan(a) && isnan(b))
                continue;

            if (diff!=0){
                int i;
                i++;
                }

            var+=diff*diff;
            }
    return sqrt(var/(dimX*dimY));
}

template <typename T1>
void IsAll0(afwImage::Image<T1>& imgA)
{
    int dimX=imgA.getWidth();
    int dimY=imgA.getHeight();

    int nonZeroCnt=0;

    double var=0;
    for (int x=0;x<dimX;x++)
        for (int y=0;y<dimY;y++) {
            double a=imgA(x,y);

            //if (isnan(a) && isnan(b))
            //    continue;

            if (a!=0)
                nonZeroCnt++;
            }

    printf("Non zero cnt: %d\n", nonZeroCnt);
}

struct PlanesVar
{
    double img;
    double msk;
    double var;
};

template <typename T1, typename T2>
PlanesVar OutputDiffStDev(afwImage::MaskedImage<T1>& imgA, afwImage::MaskedImage<T2>& imgB)
{
    int dimX=imgA.getWidth();
    int dimY=imgA.getHeight();

    int dimBX=imgB.getWidth();
    int dimBY=imgB.getHeight();

    if (dimX!=dimBX || dimY!=dimBY) {
        PlanesVar ret={NAN,NAN,NAN};
        return ret;
        }

    PlanesVar var={0,0,0};

    for (int x=0;x<dimX;x++)
        for (int y=0;y<dimY;y++) {
            bool aMsk=(*imgA.at(x,y)).mask();
            bool bMsk=(*imgB.at(x,y)).mask();

            if (aMsk!=bMsk) {
                var.msk+=1;
                //continue;
                }

            double aImg=(*imgA.at(x,y)).image();
            double bImg=(*imgB.at(x,y)).image();
            double aVar=(*imgA.at(x,y)).variance();
            double bVar=(*imgB.at(x,y)).variance();

            double diffImg=aImg-bImg;
            double diffVar=aVar-bVar;

            if (!isnan(aImg) || !isnan(bImg))
                var.img+=diffImg*diffImg;
            if (!isinf(aVar) || !isinf(bVar))
                var.var+=diffVar*diffVar;
            }
    var.img=sqrt(var.img/(dimX*dimY));
    var.var=sqrt(var.var/(dimX*dimY));

    return var;
}

template <typename T1>
void IsAll0(afwImage::MaskedImage<T1>& imgA)
{
    int dimX=imgA.getWidth();
    int dimY=imgA.getHeight();

    int nonZeroCnt=0;

    double var=0;
    for (int x=0;x<dimX;x++)
        for (int y=0;y<dimY;y++) {
            double aImg=(*imgA.at(x,y)).image();

            if (isnan(aImg))
                continue;

            if (aImg!=0)
                nonZeroCnt++;
            }

    printf("Non zero cnt: %d\n", nonZeroCnt);
}


void PrintNanOrNum(double num)
{
    if (isnan(num))
        cout << "NAN !!!";
    else if (isinf(num))
        cout << "INF !!!";
    else
        cout << num;

}

void PrintVar(const char* text, double var)
{
    cout << text << " " ;
    PrintNanOrNum(var);
    cout << endl;
}

void PrintVar(const char* text, PlanesVar var)
{
    cout << text << " " ;
    cout << "  Img: ";
    PrintNanOrNum(var.img);
    cout << "  Msk: ";
    PrintNanOrNum(var.msk);
    cout << "  Var: ";
    PrintNanOrNum(var.var);

    cout << endl;
}



typedef afwMath::LinearCombinationKernel LinearCombinationKernel;

LinearCombinationKernel& ConstructLinearCombinationKernel(
            unsigned int const KernelCols,
            unsigned int const KernelRows,
            int imageWidthCoeff,
            int imageHeightCoeff
            )
{
        double const MinSigma = 1.5;
        double const MaxSigma = 4.5;

        // construct basis kernels
        const int kernelN=3;
        afwMath::KernelList kernelList;
        for (int ii = 0; ii < kernelN; ++ii) {
            double majorSigma = (ii == 1) ? MaxSigma : MinSigma;
            double minorSigma = (ii == 2) ? MinSigma : MaxSigma;
            double angle = 0.0;
            if (ii>2) angle=ii/10;
            afwMath::GaussianFunction2<afwMath::Kernel::Pixel> gaussFunc(majorSigma, minorSigma, angle);
            afwMath::Kernel::Ptr basisKernelPtr(
                new afwMath::AnalyticKernel(KernelCols, KernelRows, gaussFunc)
            );
            kernelList.push_back(basisKernelPtr);
        }

        // construct spatially varying linear combination kernel
        int const polyOrder = 2;
        afwMath::PolynomialFunction2<double> polyFunc(polyOrder);
        afwMath::Chebyshev1Function2<double> chebyFunc(polyOrder,lsst::afw::geom::Box2D(
                                    lsst::afw::geom::Point2D(-1000, -1000),
                                    lsst::afw::geom::Point2D( 1000,  1000)));

        LinearCombinationKernel* kernelPtr=new LinearCombinationKernel(kernelList, polyFunc);
        LinearCombinationKernel& kernel=*kernelPtr;

        // Get copy of spatial parameters (all zeros), set and feed back to the kernel
        std::vector<std::vector<double> > polyParams = kernel.getSpatialParameters();
        //std::vector<std::vector<double> > polyParams(3, std::vector<double>(6));

        printf("Poly dim: %d %d\n", int(polyParams.size()), int(polyParams[0].size()));

        // Set spatial parameters for basis kernel 0
        polyParams[0][0] =  1.0;
        polyParams[0][1] = -0.5;
        polyParams[0][2] = -0.2;
        if (polyOrder>1) {
            polyParams[0][3] = -0.1 / static_cast<double>(imageHeightCoeff);
            polyParams[0][4] = -0.2 / static_cast<double>(imageHeightCoeff);
            polyParams[0][5] = 0.4 / static_cast<double>(imageHeightCoeff);
            }
        for (int i=1;i<kernelN;i++) {
            // Set spatial function parameters other basis kernels
            polyParams[i][0] = polyParams[0][0] - i*0.2;
            polyParams[i][1] = polyParams[0][1] + i*0.1;
            polyParams[i][2] = polyParams[0][2] + i*0.1;
            if (polyOrder>1) {
                polyParams[i][3] = polyParams[0][3] + i*0.1;
                polyParams[i][4] = polyParams[0][4] + i*0.1;
                polyParams[i][5] = polyParams[0][5] - i*0.1;
                }
            }

        kernel.setSpatialParameters(polyParams);

        std::cout << "Kernel size: " << KernelCols << " x " << KernelRows << std::endl;
        std::cout << "Number of basis kernels: " << kernel.getNBasisKernels() << std::endl;
        std::cout << "Spatial order: " << polyOrder << std::endl;

        //afwMath::Kernel::SpatialFunctionPtr sf=kernel.getSpatialFunctionList()[0];

        /*double _scaleX = 2.0 / (1000 - (-1000));
        double _scaleY = 2.0 / (1000 - (-1000));
        double _offsetX = -((-1000) + 1000) * 0.5;
        double _offsetY = -((-1000) + 1000) * 0.5;

        double vMy=MyCheby(polyParams[0], 2, _offsetX, _offsetY, _scaleX,  _scaleY,
                        5,6);

        double vOr=(*sf)(5,6);
        printf("%f  %f\n", vMy, vOr);

        time_t myChebyStart=clock();
        double sumMy=0;
        for (double x=0;x<900;x+=1.0/16)
            for (int y=0;y<900;y++)
                sumMy+=MyCheby(polyParams[0], 2, _offsetX, _offsetY, _scaleX,  _scaleY,
                        x,y);
        double myChebyTime=DiffTime(myChebyStart, clock());

        time_t orChebyStart=clock();
        double sumOr=0;
        for (double x=0;x<900;x+=1.0/16)
            for (int y=0;y<900;y++)
                sumOr+=(*sf)(x,y);
        double orChebyTime=DiffTime(orChebyStart, clock());

        printf("%f  %f\n", sumMy, sumOr);

        printf("My cheby time: %7.5f  Or time: %7.5f\n",
                    myChebyTime, orChebyTime);*/
/*
        printf("Values:\n");

        for (int x=0;x<400;x+=40)
            {
            printf("\n");
            for (int y=0;y<400;y+=40){
                double a= (*(kernel.getSpatialFunctionList())[0]) (x,y);
                printf("%7.2f ",a);
                }
            }
        printf("\n");
*/

    return kernel;
}

namespace lsst {
    namespace afw {
        namespace math {
            namespace detail {
                namespace gpu {

    void TestGpuKernel(int& ret1, int& ret2);
    void PrintCudaDeviceInfo();

}}}}}

namespace mathDetailGpu = lsst::afw::math::detail::gpu;

int main(int argc, char **argv) {

    unsigned int kernelCols = 18;
    unsigned int kernelRows = 18;

    #ifdef _DEBUG
        int debugFac=1;
    #else
        int debugFac=1;
    #endif

    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    string imgBaseFileName;
    if (argc < 2) {

        string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "Usage: convolveGPU fitsFile [sigma]" << endl;
            std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << endl;
            std::cerr << "I can take a default file from AFWDATA_DIR, but it's not defined." << endl;
            std::cerr << "Is afwdata set up?\n" << endl;
            exit(EXIT_FAILURE);
            }
        else{
            imgBaseFileName = afwdata + "/med";
            //imgBaseFileName = afwdata + "/medsub";
            //imgBaseFileName = afwdata + "/imsim_t1"; //just plain
            //imgBaseFileName = afwdata + "/871034p_1_MI";
            cerr << "Using " << imgBaseFileName << endl;
            }
        }
    else{
        imgBaseFileName = string(argv[1]);
        }

    //mathDetailGpu::PrintCudaDeviceInfo(); 

    /*int a,b;
    mathDetailGpu::TestGpuKernel(a,b);
    printf("GpuTest: %d %d \n", a,b);
    double val=1770.9779052734375000;
    double kval=0.0214927946025038792143835309  ;
    printf("val %.20f   kval%.20f   var%.20f\n",val,kval,val*kval*kval);
*/
    printf("===============================================\n");


    { // block in which to allocate and deallocate memory

        int sizeX;
        int sizeY;

        // read in fits file
        #ifdef CALC_MASKED_IMG
            afwImage::MaskedImage<InPixel> maskedImage(imgBaseFileName);
            afwImage::MaskedImage<double>  maskedImageDbl(imgBaseFileName);
            afwImage::MaskedImage<float>  maskedImageFlt(imgBaseFileName);
            sizeX=maskedImage.getWidth();
            sizeY=maskedImage.getHeight();
        #endif

        #ifdef CALC_PLAIN_IMG
            afwImage::Image<InPixel>       plainImage(imgBaseFileName+"_img.fits");
            sizeX=plainImage.getWidth();
            sizeY=plainImage.getHeight();
        #endif

        printf("Image size: %d x %d\n", sizeX, sizeY);

        // construct kernels
        double majorSigma = 2.5;
        double minorSigma = 2.0;
        double angle = 0.5;
        afwMath::GaussianFunction2<KerPixel> gaussFunc(majorSigma, minorSigma, angle);
        afwMath::AnalyticKernel analyticKernel(kernelCols, kernelRows, gaussFunc);

        afwMath::DeltaFunctionKernel deltaFKernel(1,1,lsst::afw::geom::Point2I(0, 0));

        LinearCombinationKernel& linCoKernel=ConstructLinearCombinationKernel(
                                                                kernelCols,kernelRows,
                                                                sizeX,sizeY
                                                                );

        lsst::afw::image::Image<KerPixel> analyticImage(analyticKernel.getDimensions());
        (void)analyticKernel.computeImage(analyticImage, true);
        analyticImage *= 47.3; // denormalize by some arbitrary factor
        lsst::afw::math::FixedKernel fixedKernel(analyticImage);

        afwMath::ConvolutionControl convCtrlCPU(true,false,0, afwMath::ConvolutionControl::FORCE_CPU);
	afwMath::ConvolutionControl convCtrlGPU(true,false,0);


        //resultant images
        #ifdef CALC_MASKED_IMG
            afwImage::MaskedImage<OutPixel> resMaskedImage(maskedImage.getDimensions());
            afwImage::MaskedImage<double> resMaskedImageDbl(maskedImage.getDimensions());
            afwImage::MaskedImage<OutPixel> resMaskedImageGPU(maskedImage.getDimensions());

            afwImage::MaskedImage<OutPixel> resMaskedImageLC(maskedImage.getDimensions());
            afwImage::MaskedImage<double> resMaskedImageLCDbl(maskedImage.getDimensions());
            afwImage::MaskedImage<OutPixel> resMaskedImageLCGPU(maskedImage.getDimensions());
        #endif

        #ifdef CALC_PLAIN_IMG
            afwImage::Image<OutPixel>       resPlainImage(plainImage.getDimensions());
            afwImage::Image<double>         resPlainImageDbl(plainImage.getDimensions());
            afwImage::Image<OutPixel>       resPlainImageGPU(plainImage.getDimensions());

            afwImage::Image<OutPixel>       resPlainImageLC(plainImage.getDimensions());
            afwImage::Image<double>         resPlainImageLCDbl(plainImage.getDimensions());
            afwImage::Image<OutPixel>       resPlainImageLCGPU(plainImage.getDimensions());
        #endif

	//init gpu	
	afwMath::convolve(resMaskedImageGPU, maskedImage, fixedKernel, convCtrlGPU);

        // convolve
        #ifdef CALC_MASKED_IMG
            //afwMath::convolve   (resMaskedImageDbl   , maskedImageDbl, fixedKernel, convCtrlCPU);
            time_t maskedImgCpuStart=clock();
            for (int i=0;i<debugFac;i++)
                afwMath::convolve(resMaskedImage   , maskedImage, fixedKernel, convCtrlCPU);
            double maskedImgCpuTime=DiffTime(maskedImgCpuStart, clock())/debugFac;
            time_t maskedImgGpuStart=clock();
            for (int i=0;i<5*debugFac;i++)
                afwMath::convolve(resMaskedImageGPU, maskedImage, fixedKernel, convCtrlGPU);
            double maskedImgGpuTime=DiffTime(maskedImgGpuStart, clock())/(5*debugFac);
            //afwMath::convolve   (resMaskedImageLCDbl   , maskedImageDbl, linCoKernel, convCtrlCPU);
            time_t maskedImgLCCpuStart=clock();
            for (int i=0;i<1;i++)
                afwMath::convolve(resMaskedImageLC   , maskedImage, linCoKernel, convCtrlCPU);
            double maskedImgLCCpuTime=DiffTime(maskedImgLCCpuStart, clock())/1;
            time_t maskedImgLCGpuStart=clock();
            for (int i=0;i<5*debugFac;i++)
                afwMath::convolve(resMaskedImageLCGPU, maskedImage, linCoKernel, convCtrlGPU);
            double maskedImgLCGpuTime=DiffTime(maskedImgLCGpuStart, clock())/(5*debugFac);
        #endif

        #ifdef CALC_PLAIN_IMG
            //afwMath::convolve   (resPlainImageDbl   , plainImage, fixedKernel, convCtrlCPU);
            time_t plainImgCpuStart=clock();
            for (int i=0;i<debugFac;i++)
                afwMath::convolve(resPlainImage   , plainImage, fixedKernel, convCtrlCPU);
            double plainImgCpuTime=DiffTime(plainImgCpuStart, clock())/debugFac;
            time_t plainImgGpuStart=clock();
            for (int i=0;i<5*debugFac;i++)
                afwMath::convolve(resPlainImageGPU, plainImage, fixedKernel, convCtrlGPU);
            double plainImgGpuTime=DiffTime(plainImgGpuStart, clock())/(5*debugFac);

            //afwMath::convolve   (resPlainImageLCDbl   , plainImage, linCoKernel, convCtrlCPU);
            time_t plainImgLCCpuStart=clock();
            for (int i=0;i<debugFac;i++)
                afwMath::convolve(resPlainImageLC   , plainImage, linCoKernel, convCtrlCPU);
            double plainImgLCCpuTime=DiffTime(plainImgLCCpuStart, clock())/debugFac;
            time_t plainImgLCGpuStart=clock();
            for (int i=0;i<5*debugFac;i++)
                afwMath::convolve(resPlainImageLCGPU, plainImage, linCoKernel, convCtrlGPU);
            double plainImgLCGpuTime=DiffTime(plainImgLCGpuStart, clock())/(5*debugFac);
        #endif

        // write results
        #ifdef CALC_MASKED_IMG
            printf("=============== MASKED IMAGE ==================================== \n");
            //PrintVar("Masked Image Dbl StDev: ", OutputDiffStDev(resMaskedImageDbl,resMaskedImageGPU));
            PrintVar("Masked Image Ot  StDev: ", OutputDiffStDev(resMaskedImage,resMaskedImageGPU));
            printf("CPU time: %7.5f  GPU time: %7.5f  Speedup: %6.2f\n",
                    maskedImgCpuTime, maskedImgGpuTime, maskedImgCpuTime/maskedImgGpuTime);

            //PrintVar("Masked Image Dbl StDev: ", OutputDiffStDev(resMaskedImageDbl,resMaskedImageGPU));
            PrintVar("Masked Image LC  StDev: ", OutputDiffStDev(resMaskedImageLC,resMaskedImageLCGPU));
            printf("CPU time: %7.5f  GPU time: %7.5f  Speedup: %6.2f\n",
                    maskedImgLCCpuTime, maskedImgLCGpuTime, maskedImgLCCpuTime/maskedImgLCGpuTime);
        #endif
        #ifdef CALC_PLAIN_IMG
            printf("=============== PLAIN IMAGE ==================================== \n");
            //PrintVar(       "Image Dbl StDev: ", OutputDiffStDev(resPlainImageDbl,resPlainImageGPU));
            PrintVar(       "Image Ot  StDev: ", OutputDiffStDev(resPlainImage,resPlainImageGPU));
            printf("CPU time: %7.5f  GPU time: %7.5f  Speedup: %6.2f\n",
                    plainImgCpuTime, plainImgGpuTime, plainImgCpuTime/plainImgGpuTime);

            //PrintVar(       "Image Dbl StDev: ", OutputDiffStDev(resPlainImageLCDbl,resPlainImageLCGPU));
            PrintVar(       "Image LC  StDev: ", OutputDiffStDev(resPlainImageLC,resPlainImageLCGPU));
            printf("CPU time: %7.5f  GPU time: %7.5f  Speedup: %6.2f\n",
                    plainImgLCCpuTime, plainImgLCGpuTime, plainImgLCCpuTime/plainImgLCGpuTime);
        #endif

        #ifdef CALC_MASKED_IMG
            resMaskedImageLC   .writeFits(outFileMasked);
            resMaskedImageLCGPU.writeFits(outFileMaskedGPU);

            //IsAll0(resMaskedImage);
            //IsAll0(resMaskedImageGPU);
        #endif

        #ifdef CALC_PLAIN_IMG
            resPlainImageLC   .writeFits(outFilePlain);
            resPlainImageLCGPU.writeFits(outFilePlainGPU);

            //IsAll0(resPlainImage);
            //IsAll0(resPlainImageGPU);
        #endif

        delete &linCoKernel;
    }


     //
     // Check for memory leaks
     //
     if (lsst::daf::base::Citizen::census(0) != 0) {
         std::cerr << "Leaked memory blocks:" << std::endl;
         lsst::daf::base::Citizen::census(std::cerr);
     }

     //system("PAUSE");

}
