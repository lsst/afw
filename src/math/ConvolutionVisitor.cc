#include <lsst/afw/math/ConvolutionVisitor.h>
#include <fftw3.h>

namespace afwMath = lsst::afw::math;

void afwMath::FourierConvolutionVisitor::copyImage(
        PixelT * dest, 
        FourierConvolutionVisitor::Image::Ptr image, 
        int const &destWidth
) {
    double * destIter = dest;    
    int rowStep = destWidth - image->getWidth();

    for(int y = 0; y < image->getHeight(); ++y, destIter += rowStep) {
        Image::const_x_iterator rowIter(image->row_begin(y));
        Image::const_x_iterator const rowEnd(image->row_end(y));
        for( ; rowIter != rowEnd; ++rowIter, ++destIter) {
            (*destIter) = *rowIter;            
        }
    }
}

void afwMath::FourierConvolutionVisitor::fillImageStack(
        PixelT * imageStack, 
        std::pair<int,int> const & imageDimensions
) {
    int imageSize = imageDimensions.first*imageDimensions.second;
    PixelT * rowPtr = imageStack;
    copyImage(rowPtr, _imageVisitor.getImage(), imageDimensions.first);

    rowPtr += imageSize;    
    ImagePtrList derivative = _imageVisitor.getDerivative();
    ImagePtrList::const_iterator i(derivative.begin());
    ImagePtrList::const_iterator const end(derivative.end());
    for( ; i != end ; ++i, rowPtr += imageSize) {
        copyImage(rowPtr, *i, imageDimensions.first);   
    }
}


void afwMath::FourierConvolutionVisitor::fft(
        std::pair<int,int> const & imageDimensions
) {
    //construct imageStack (the input for fftw)
    int imageSize = imageDimensions.first*imageDimensions.second;
    boost::scoped_ptr<PixelT> imageStack(
            new PixelT[(getNParameters() + 1) * imageSize]
    );
    
    _fourierStack = FourierCutoutStack(getNParameters()+1, imageDimensions);
    //construct a forward-transform plan
    fftw_plan plan = fftw_plan_many_dft_r2c(
            2, //rank: these are 2 dimensional images
            &imageDimensions.first, //image dimensions
            getNParameters() + 1, //number of images to transform
            imageStack.get(), //input ptr
            NULL, //embeded input image dimensions
            1, //input images are contiguous
            imageSize, //input stack is contiguous
            reinterpret_cast<fftw_complex*>(_fourierStack.getData().get()), //output ptr
            NULL, //embeded output image dimensions
            1, //output images are contiguous
            _fourierStack.getCutoutSize(), //output stack is contiguous
            FFTW_MEASURE | FFTW_DESTROY_INPUT //flags
    );
    
    //only fill after the plan has been constructed
    //this is because planner can destroy input data
    fillImageStack(imageStack.get(), imageDimensions);
    
    //finally execute the plan
    fftw_execute(plan);
}
