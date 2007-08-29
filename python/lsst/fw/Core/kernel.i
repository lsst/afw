%{
#include "lsst/fw/Kernel.h"
#include "lsst/fw/KernelFunctions.h"
%}

%newobject lsst::fw::kernel::convolve;

%include "lsst/fw/Kernel.h"
%include "lsst/fw/KernelFunctions.h"

%template(KernelF)              lsst::fw::Kernel<float>;
%template(FixedKernelF)         lsst::fw::FixedKernel<float>;
%template(AnalyticKernelF)      lsst::fw::AnalyticKernel<float>;
%template(LinearCombinationKernelF) lsst::fw::LinearCombinationKernel<float>;

%template(Function2PtrTypeF)    boost::shared_ptr<lsst::fw::function::Function2<float> >;
%template(KernelPtrTypeF)       boost::shared_ptr<lsst::fw::Kernel<float> >;

%template(printKernelF)         lsst::fw::kernel::printKernel<float>;

%template(convolveF)            lsst::fw::kernel::convolve<ImagePixelType, MaskPixelType, float>;

%template(KernelD)              lsst::fw::Kernel<double>;
%template(FixedKernelD)         lsst::fw::FixedKernel<double>;
%template(AnalyticKernelD)      lsst::fw::AnalyticKernel<double>;
%template(LinearCombinationKernelD) lsst::fw::LinearCombinationKernel<double>;

%template(Function2PtrTypeD)    boost::shared_ptr<lsst::fw::function::Function2<double> >;
%template(KernelPtrTypeD)       boost::shared_ptr<lsst::fw::Kernel<double> >;

%template(printKernelD)         lsst::fw::kernel::printKernel<double>;

// define kernel-related vectors
%template(kernelPtrVectorF)     std::vector<boost::shared_ptr<lsst::fw::Kernel<float> > >;
%template(kernelPtrVectorD)     std::vector<boost::shared_ptr<lsst::fw::Kernel<double> > >;

%inline %{
    template <typename ImageT>
    vw::ImageView<ImageT> copyImageView(vw::ImageView<ImageT> &src) {
        return src;
    }

    template <typename ImageT>
    lsst::fw::Image<ImageT> copyImage(lsst::fw::Image<ImageT> &src) {
        return src;
    }

    template <typename MaskT>
    lsst::fw::Mask<MaskT> copyMask(lsst::fw::Mask<MaskT> &src) {
        return src;
    }

    template <typename ImageT, typename MaskT>
    lsst::fw::MaskedImage<ImageT, MaskT> copyMaskedImage(lsst::fw::MaskedImage<ImageT, MaskT> &src) {
        return src;
    }
%}

//%template(copyImageViewF) copyImageView<ImagePixelType>;
//%template(copyImageF) copyImage<ImagePixelType>;
//%template(copyMaskF) copyMask<MaskPixelType>; // fails to build if Mask's copy constructor is explicit; why?
//%template(copyMaskedImageF) copyMaskedImage<ImagePixelType, MaskPixelType>;
