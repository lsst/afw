#if !defined(LSST_MATH_COPY_EDGES_H)
#define LSST_MATH_COPY_EDGES_H 1
/*
 * Private functions to copy the border of an image.  Declared inline as this file is
 * currently included in Convolve(Masked)?Image.cc which is #included;  when this is
 * fixed, these functions should no longer be declared inline
 */
/**
 * @brief Private function to copy a rectangular region from one (Masked)?Image to another, setting 0 or more bits
 */

template<typename OutPixelT, typename InPixelT, typename MaskPixelT, typename VariancePixelT>
inline void _copyRegion(lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT, VariancePixelT> &outImage,	///< destination MaskedImage
                        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT, VariancePixelT> const &inImage, ///< source MaskedImage
                        lsst::afw::image::Bbox const &region,                   ///< region to copy
                        int orMask                                              ///< data to | into the mask pixels
                       ) {
    lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT, VariancePixelT> outPatch(outImage, region); 
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT, VariancePixelT>   inPatch(inImage, region);
    outPatch <<= inPatch;
    *outPatch.getMask() |= orMask;
}

template<typename OutPixelT, typename InPixelT>
inline void _copyRegion(lsst::afw::image::Image<OutPixelT> &outImage,     ///< destination Image
                        lsst::afw::image::Image<InPixelT> const &inImage, ///< source Image
                        lsst::afw::image::Bbox const &region,               ///< region to copy
                        int
                       ) {
    lsst::afw::image::Image<OutPixelT> outPatch(outImage, region); 
    lsst::afw::image::Image<InPixelT>   inPatch(inImage, region);
    outPatch <<= inPatch;
}

template <typename OutImageT, typename InImageT>
inline void _copyBorder(
    OutImageT& convolvedImage,                           ///< convolved image
    InImageT const& inImage,                             ///< image to convolve
    lsst::afw::math::Kernel const &kernel,               ///< convolution kernel
    int edgeBit                         ///< mask bit to indicate border pixel;  if negative then no bit is set
) {
    const unsigned int imWidth = inImage.getWidth();
    const unsigned int imHeight = inImage.getHeight();
    const unsigned int kWidth = kernel.getWidth();
    const unsigned int kHeight = kernel.getHeight();
    const unsigned int kCtrX = kernel.getCtrX();
    const unsigned int kCtrY = kernel.getCtrY();
    
    using lsst::afw::image::Bbox;
    using lsst::afw::image::PointI;
    Bbox bottomEdge(PointI(0, 0), imWidth, kCtrY);
    _copyRegion(convolvedImage, inImage, bottomEdge, edgeBit);
    
    int numHeight = kHeight - (1 + kCtrY);
    Bbox topEdge(PointI(0, imHeight - numHeight), imWidth, numHeight);
    _copyRegion(convolvedImage, inImage, topEdge, edgeBit);

    Bbox leftEdge(PointI(0, kCtrY), kCtrX, imHeight + 1 - kHeight);
    _copyRegion(convolvedImage, inImage, leftEdge, edgeBit);
    
    int numWidth = kWidth - (1 + kCtrX);
    Bbox rightEdge(PointI(imWidth - numWidth, kCtrY), numWidth, imHeight + 1 - kHeight);
    _copyRegion(convolvedImage, inImage, rightEdge, edgeBit);
}

#endif
