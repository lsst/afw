// -*- LSST-C++ -*-
#ifndef LLST_FW_Kernel_H
#define LLST_FW_Kernel_H
/**
 * \file
 * \ingroup fw
 *
 * Kernels are used for convolution with MaskedImages
 * and (eventually) Images
 *
 * To do:
 * - Fix the inline functions. These presently are not part of the header,
 *   so will not be properly inlined. The issue of how to separate definition
 *   implementation is presently being debated so I don't want to do this
 *   until that is finished.
 *
 * - Fix LinearCombinationKernel::getImage to handle a spatial model.
 *
 * - Fix warning: Kernel.h:132: warning: inline function `void lsst::fw::Kernel<PixelT>::setKernelParameters(
 *   std::vector<T, std::allocator<_CharT> >) [with PixelT = double]' used but never defined
 *
 * - Perhaps make getImage const (i.e. does not change the class).
 *   That will require moving the implementation of setKernelParameters
 *   to a protected or private version of the function that is const
 *   (because the public setKernelParameters must not be const).
 *
 * \author Russell Owen
 */
#include <vector>

#include <vw/Image.h>

#include <lsst/fw/LsstBase.h>
#include <lsst/fw/Function.h>
#include <lsst/fw/Image.h>

namespace lsst {
namespace fw {

    /**
     * A convolution kernel.
     *
     * This is a virtual base class; it cannot be instantiated.
     */
    template<class PixelT>
    class Kernel : private LsstBase
    {
    
    public:
        typedef boost::shared_ptr<Function2<PixelT> > Function2PtrType;

        Kernel();
        
        Kernel(
            unsigned cols,
            unsigned rows,
            unsigned nKernelParams
        );
        
        Kernel(
            unsigned cols,
            unsigned rows,
            const unsigned nKernelParams,
            Function2PtrType spatialFunction
        );
        
        virtual ~Kernel() {};
    
        virtual const unsigned getCols() const;
        
        virtual const unsigned getRows() const;
        
        virtual const unsigned getCtrCol() const;

        virtual const unsigned getCtrRow() const;
            
        /**
         * Get an image (pixellized representation) of the kernel
         *
         * x, y are ignored if there is no spatial function.
         */
        virtual const Image<PixelT>
        getImage(PixelT x = 0, PixelT y = 0) = 0;
        
        virtual inline const std::vector<PixelT>
        getKernelParameters(PixelT x = 0, PixelT y = 0) const;
    
        virtual inline const unsigned
        getNKernelParameters() const;
    
        virtual inline const unsigned
        getNSpatialParameters() const;
    
        virtual inline std::vector<std::vector<PixelT> >
        getSpatialParameters() const;
            
        virtual inline const bool
        isSpatiallyVarying() const;    
    
        /**
         * Set the kernel parameters of a spatially invariant kernel.
         *
         * Throws an exception if:
         * - the kernel has a spatial function
         * - the params vector is the wrong length
         */
        virtual inline void
        setKernelParameters(const std::vector<PixelT> params) = 0;
    
        virtual void
        setSpatialParameters(const std::vector<std::vector<PixelT> > params);
    
    protected:
        virtual const std::vector<PixelT>
        getKernelParametersFromSpatialModel(PixelT x, PixelT y) const;
                
        /**
         * Get the current kernel parameters.
         *
         * Assumes there is no spatial model.
         */
        virtual inline const std::vector<PixelT>
        getCurrentKernelParameters() const = 0;
        
        virtual void
        setKernelParametersFromSpatialModel(PixelT x, PixelT y);
           
    private:
        unsigned _cols;
        unsigned _rows;
        unsigned _ctrCol;
        unsigned _ctrRow;
        unsigned _nKernelParams;
        bool _isSpatiallyVarying;
        Function2PtrType _spatialFunctionPtr;
        std::vector<std::vector<PixelT> > _spatialParams;
    };
    
    
    /**
     * A kernel created from an Image
     * (a boost::shared_ptr to an lsst::fw::Image, to be precise).
     * It has no adjustable parameters and so cannot be spatially varying.
     */
    template<class PixelT>
    class FixedKernel : public Kernel<PixelT>
    {
    
    public:
        FixedKernel();

        FixedKernel(
            const Image<PixelT> &image
        );
        
        virtual ~FixedKernel();
    
        virtual const Image<PixelT>
        getImage(PixelT x = 0, PixelT y = 0);
            
    protected:
        /**
         * Return kernel parameters if there is no spatial model.
         */
        virtual inline const std::vector<PixelT>
        getCurrentKernelParameters() const;

        virtual inline void
        setKernelParameters(const std::vector<PixelT> params);
            
    private:
        Image<PixelT> _image;
    };
    
    
    /**
     * A kernel described by a function (an lsst::fw::Function2<PixelT> to be precise).
     */
    template<class PixelT>
    class AnalyticKernel : public Kernel<PixelT>
    {
    public:
        AnalyticKernel();

        AnalyticKernel(
            typename Kernel<PixelT>::Function2PtrType kernelFunction,
            unsigned cols,
            unsigned rows
        );
        
        AnalyticKernel(
            typename Kernel<PixelT>::Function2PtrType kernelFunction,
            unsigned cols,
            unsigned rows,
            typename Kernel<PixelT>::Function2PtrType spatialFunction
        );
        
        AnalyticKernel(
            typename Kernel<PixelT>::Function2PtrType kernelFunction,
            unsigned cols,
            unsigned rows,
            typename Kernel<PixelT>::Function2PtrType spatialFunction,
            const std::vector<std::vector<PixelT> > spatialParameters
        );
        
        virtual ~AnalyticKernel() {};
    
        virtual const Image<PixelT>
        getImage(PixelT x = 0, PixelT y = 0);
    
        virtual inline typename Kernel<PixelT>::Function2PtrType
        getKernelFunction() const;
    
        virtual inline void
        setKernelParameters(const std::vector<PixelT> params);
            
    protected:
        virtual inline const std::vector<PixelT>
        getCurrentKernelParameters() const;

    private:
        typename Kernel<PixelT>::Function2PtrType _kernelFunctionPtr;
    };
    
    
    /**
     * A kernel that is a linear combination of fixed basis kernels
     * (see definition of KernelListType for the specific format).
     * 
     * Convolution may be performed by first convolving the image
     * with each fixed kernel, then adding the resulting images using the (possibly
     * spatially varying) kernel coefficients.
     */
    template<class PixelT>
    class LinearCombinationKernel : public Kernel<PixelT>
    {
    
    public:
        typedef std::vector<boost::shared_ptr<Kernel<PixelT> > > KernelListType;

        LinearCombinationKernel();

        LinearCombinationKernel(
            const KernelListType kernelList,
            const std::vector<PixelT> kernelParameters
        );
        
        LinearCombinationKernel(
            const KernelListType kernelList,
            typename Kernel<PixelT>::Function2PtrType spatialFunction
        );
        
        LinearCombinationKernel(
            const KernelListType kernelList,
            typename Kernel<PixelT>::Function2PtrType spatialFunction,
            const std::vector<std::vector<PixelT> > spatialParameters
        );
        
        virtual ~LinearCombinationKernel() {};
    
        virtual const Image<PixelT>
        getImage(PixelT x = 0, PixelT y = 0);
                
        /**
         * Get the fixed basis kernels
         */
        virtual inline const KernelListType &
        getKernelList() const;
        
        /**
         * Check that all kernels have the same size and center
         * and that none are spatially varying
         */
        void checkKernelList(const KernelListType &kernelList) const;
        
        virtual inline void
        setKernelParameters(const std::vector<PixelT> params);
            
    protected:
        virtual inline const std::vector<PixelT>
        getCurrentKernelParameters() const;

    private:
        KernelListType _kernelList;
        std::vector<PixelT> _kernelParams;
    };
    
    // Included templated implementation
    #include <lsst/fw/Kernel/Kernel.cc>
    #include <lsst/fw/Kernel/FixedKernel.cc>
    #include <lsst/fw/Kernel/AnalyticKernel.cc>
    #include <lsst/fw/Kernel/LinearCombinationKernel.cc>

    // Explicitly instantiate the available type(s)
    extern template class Kernel<float>;
    extern template class FixedKernel<float>;
    extern template class AnalyticKernel<float>;
    extern template class LinearCombinationKernel<float>;

}   // namespace fw
}   // namespace lsst

#endif // !defined(LLST_FW_Kernel_H)
