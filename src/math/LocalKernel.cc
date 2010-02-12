#include <lsst/afw/math/LocalKernel.h>
#include <fftw3.h>
#include <cassert>

namespace afwMath = lsst::afw::math;

afwMath::FourierCutout::Ptr afwMath::FftLocalKernel::getFourierImage() const {        
    if(_fourierStack.getStackDepth() > 0)
        return _fourierStack.getCutout(0);
   
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
            "Must previously call FourierLocalKernel::setDimensions");
}

std::vector<afwMath::FourierCutout::Ptr> 
afwMath::FftLocalKernel::getFourierDerivatives() const {
    if(_fourierStack.getStackDepth() > 1) {
        //has already been transformed. return output of latest transform
        return _fourierStack.getCutoutVector(1);
    }
    else if(getNParameters() == 0) {
        //no derivative info
        return std::vector<FourierCutout::Ptr>();
    }

    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
            "Must previously call FourierLocalKernel::setDimensions");
}


void afwMath::FftLocalKernel::copyImage(
        Pixel * dest, 
        Image::Ptr image, 
        int const &destWidth
) {
    if(!image || !dest)
        return;

    double * destIter = dest;    
    double * innerIter;
    for(int y = 0; y < image->getHeight(); ++y, destIter += destWidth) {
        Image::const_x_iterator rowIter(image->row_begin(y));
        Image::const_x_iterator const rowEnd(image->row_end(y));
        for(innerIter = destIter; rowIter != rowEnd; ++rowIter, ++innerIter) {
            (*innerIter) = (*rowIter);
        }
    }
}

void afwMath::FftLocalKernel::fillImageStack(
        Pixel * imageStack, 
        int const & imageSize,
        int const & imageWidth
) {
    Pixel * rowPtr = imageStack;
    assert(rowPtr != 0);
    copyImage(rowPtr, _imageKernel.getImage(), imageWidth);

    rowPtr += imageSize;    
    ImagePtrList derivatives = _imageKernel.getDerivatives();
    ImagePtrList::const_iterator i(derivatives.begin());
    ImagePtrList::const_iterator const end(derivatives.end());
    for( ; i != end ; ++i, rowPtr += imageSize) {
        copyImage(rowPtr, *i, imageWidth);   
    }
}


void afwMath::FftLocalKernel::setDimensions(
    int const & width, int const &height, bool normalize
) {
    int kernelWidth = _imageKernel.getWidth();
    int kernelHeight = _imageKernel.getHeight();
    if(kernelWidth > width || kernelHeight > height) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException, 
            (boost::format(
                "Requested dimensions (%1%, %2%) must be at least as large"
                " as input dimensions (%3%, %4%)"
            ) %  width % height % kernelWidth % kernelHeight).str()
        );
    }

    int stackDepth = getNParameters() + 1;
    //construct imageStack (the input for fftw)
    int imageSize = width*height;
    boost::scoped_array<Pixel> imageStack(new Pixel[stackDepth * imageSize]);
   
    FourierCutoutStack temp(width, height, stackDepth);
    int cutoutSize = temp.getCutoutSize();
    std::pair<int, int> dimensions =std::make_pair(height, width);

    //construct a forward-transform plan
    fftw_plan plan = fftw_plan_many_dft_r2c(
            2, //rank: these are 2 dimensional images
            &dimensions.first, //image dimensions
            stackDepth, //number of images to transform
            imageStack.get(), //input ptr
            NULL, //embeded input image dimensions
            1, //input images are contiguous
            imageSize, //input stack is contiguous
            reinterpret_cast<fftw_complex*>(temp.getData().get()), //output ptr
            NULL, //embeded output image dimensions
            1, //output images are contiguous
            cutoutSize, //output stack is contiguous
            FFTW_MEASURE | FFTW_DESTROY_INPUT | FFTW_UNALIGNED //flags
    );
   

    //only fill after the plan has been constructed
    //this is because planner can destroy input data
    fillImageStack(imageStack.get(), imageSize, width);
   
    //finally execute the plan
    fftw_execute(plan);

    fftw_destroy_plan(plan);

    _fourierStack.swap(temp);
    
    //shift and if necessary, normalize each fft
    lsst::afw::geom::Point2I const & center = _imageKernel.getCenter();
    int dx = -center.getX();
    int dy = - center.getY();    
    double fftScale = 1.0/cutoutSize;
    
    FourierCutout::Ptr cutoutPtr;
    for(int i = 0; i < stackDepth; ++i) {
        cutoutPtr = _fourierStack.getCutout(i);
        cutoutPtr->shift(dx, dy);
        if(normalize) 
            (*cutoutPtr) *= fftScale;
    }
}
