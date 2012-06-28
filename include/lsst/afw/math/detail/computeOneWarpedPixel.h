namespace lsst {
namespace afw {
namespace math {
namespace detail {

    /**
     * @brief A class to manage a warping kernel and optional mask warping kernel
     */
    class WarpingKernelInfo {
    public:
        WarpingKernelInfo(
            afwMath::SeparableKernel::Ptr kernelPtr,            ///< warping kernel (required)
            afwMath::SeparableKernel::Ptr maskKernelPtr = 0,    ///< mask warping kernel (optional)
        ) :
            _kernelPtr(kernelPtr),
            _maskKernelPtr(maskKernelPtr),
            _xList(kernelPtr->getWidth(),
            _yList(kernelPtr->getHeight(),
            _maskXList(maskKernelPtr ? maskKernelPtr->getWidth() : 0),
            _maskYList(maskKernelPtr ? maskKernelPtr->getHeight() : 0),
        ) { }
        
        /**
         * Set fractional index in kernel parameters
         *
         * @return sum of kernel
         */
        double setFracIndex(double xFrac, double yFrac) {
            std::pair<double, double> srcFracInd(srcIndFracX.second, srcIndFracY.second);
            _kernelPtr->setKernelParameters(srcFracInd);
            double kSum = _kernelPtr->computeVectors(_xList, _yList, false);
            
            if (_maskKernelPtr) {
                _maskKernelPtr->setKernelParameters(srcFracInd);
                _maskKernelPtr->computeVectors(_maskXList, _maskYList, false);
            }
        }
        
        afwGeom::Point2I getKernelCtr() const {
            return _kernelPtr->getCtr();
        }
        afwGeom::Point2I getMaskKernelCtr() const {
            if (not _maskKernelPtr) {
                throw error
            }
            return _kernelPtr->getCtr();
        }
        
        bool hasMaskKernel() const { return bool(_maskKernelPtr); }
        
        std::vector<double> const & getXList() const { return _xList; }
        std::vector<double> const & getYList() const { return _xList; }
        std::vector<double> const & getMaskXList() const { return _maskXList; }
        std::vector<double> const & getMaskYList() const { return _maskYList; }
    
    private:
        afwMath::SeparableKernel::Ptr _kernelPtr;
        afwMath::SeparableKernel::Ptr _maskKernelPtr;
        std::vector<double> _xList;
        std::vector<double> _yList;
        std::vector<double> _maskXList;
        std::vector<double> _maskYList;
    };
    
    
    /**
     * @brief Compute one warped pixel, Image version
     *
     * This is the Image version; it ignores the mask kernel.
     */
    template<ToPixelT, FromPixelT>
    void computeOneWarpedPixel(
        typename afwImage::Image<ToPixelT>::x_iterator &destXIter,  ///< output pixel as an x iterator
        WarpingKernelInfo &kernelInfo,  ///< information about the warping kernel
        afwImage::Image<FromPixelT> const &srcImage,    ///< source image
        afwGeom::Point2D const &srcPos, ///< pixel position on source image at which to compute output
        double relativeArea,    ///< output/input area a pixel covers on the sky
        afwImage::Image<ToPixelT>::SinglePixel const &edgePixel
            ///< result if warped pixel is undefined (off the edge edge)
    );
    
    /**
     * @brief Compute one warped pixel, MaskedImage version
     *
     * This is the MaskedImage version; it uses the mask kernel, if present, to compute the mask pixel.
     */
    template<ToPixelT, FromPixelT>
    void computeOneWarpedPixel(
        typename afwImage::MaskedImage<ToPixelT>::x_iterator &destXIter,  ///< output pixel as an x iterator
        WarpingKernelInfo &kernelInfo,  ///< information about the warping kernel
        afwImage::MaskedImage<FromPixelT> const &srcImage,    ///< source image
        afwGeom::Point2D const &srcPos, ///< pixel position on source image at which to compute output
        double relativeArea,    ///< output/input area a pixel covers on the sky
        afwImage::MaskedImage<ToPixelT>::SinglePixel const &edgePixel
            ///< result if warped pixel is undefined (off the edge edge)
    );

}}}} // lsst::afw::math::detail
