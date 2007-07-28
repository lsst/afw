// -*- LSST-C++ -*-
#ifndef LSST_FW_Kernel_H
#define LSST_FW_Kernel_H
/**
 * \file
 *
 * \brief Declare the Kernel class and subclasses.
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <vector>

#include <boost/shared_ptr.hpp>
#include <vw/Image.h>

#include <lsst/mwi/data/LsstBase.h>
#include <lsst/fw/Function.h>
#include <lsst/fw/Image.h>

namespace lsst {
namespace fw {

    /**
     * \brief Kernels are used for convolution with MaskedImages and (eventually) Images
     *
     * Kernel is a virtual base class; it cannot be instantiated. The following notes apply to
     * Kernel and to its subclasses.
     *
     * The template type should usually be float or double; integer kernels
     * should be used with caution because they do not normalize well.
     *
     * The center pixel of a Kernel is at index: (cols-1)/2, (rows-1)/2. Thus it is centered along
     * columns/rows if the kernel has an odd number of columns/rows and shifted 1/2 pixel towards 0 otherwise.
     * A kernel should have an odd number of columns and rows unless it is intended to shift an image.
     *
     * <b>Spatially Varying Kernels</b>
     *
     * Kernels may optionally vary spatially (so long as they have any kernel parameters).
     * To make a spatially varying kernel, specify a spatial function at construction
     * (you cannot change your mind once the kernel is constructed).
     * You must also specify a set of spatial parameters, and you may do this at construction
     * and/or later by calling setSpatialParameters. The spatial parameters are a vector
     * (one per kernel function parameter) of spatial function parameters. In other words
     * the spatial parameters are a vector of vectors indexed as [kernel parameter][spatial parameter].
     * The one spatial function is used to compute the kernel parameters at a given spatial position
     * by computing each kernel parameter using its associated vector of spatial function parameters.
     *
     * One may only specify a single spatial function to describe the variation of all kernel parameters
     * (though the spatial parameters can, and usually will, be different for each kernel parameter).
     * The only use cases we had used the same spatial model for all parameters, and I felt it
     * would be a burden to the user to have to create one function per kernel parameter.
     * (I would gladly change this design if wanted, since it would simplify the internal code.)
     * You still can assign a different spatial model to each kernel parameter by defining
     * one function that is the sum of all the desired spatial models. Then for each kernel parameter,
     * then zero all but the few spatial parameters that control the spatial model for that kernel parameter.
     *
     * When determining the parameters for a spatial function or a kernel function,
     * please keep the LSST convention for pixel position vs. pixel index in mind.
     * See lsst/fw/ImageUtils.h for the convention.
     *
     * Note that if a kernel is spatially varying then you may not set the kernel parameters directly;
     * that is the job of the spatial function! However, you may change the spatial parameters at any time.
     *
     * <b>Design Notes</b>
     *
     * The basic design is to use the same kernel class for both spatially varying and spatially invariant
     * kernels. The user either does or does not supply a function describing the spatial variation 
     * at creation time. In addition, analytic kernels are described by a user-supplied function
     * of the same basic type as the spatial variation function.
     * 
     * Several other designs were considered, including:
     * A) Use different classes for spatially varying and spatially invariant versions of each kernel.
     * Thus instead of three basic kernel classes (FixedKernel, AnalyticKernel and LinearCombinationKernel)
     * we would have five (since FixedKernel cannot be spatially varying). Robert Lupton argued that
     * was a needless expansion of the class hiearchy and I agreed.
     * B) Construct analytic kernels by defining a subclass of AnalyticKernel that is specific to the
     * desired functional (e.g. GaussianAnalyticKernel). If spatial models are handled the same way
     * then this creates a serious proliferation of kernel classes (even if we only have two different
     * spatial models, e.g. polynomial and Chebyshev polynomial). I felt it made more sense to define
     * the spatial model by some kind of function class (often called a "functor"), and since we needed
     * such a class, I chose to use it for the analytic kernel as well.
     *
     * However, having a separate function class does introduce some potential inefficiencies.
     * If a function is part of the class it can potentially be evaluated more quickly than calling
     * a function for each pixel or spatial position.
     *
     * A possible variant on the current design is to define the spatial model and analytic kernel
     * by specifying the functions as template parameters. This has the potential to regain some efficiency
     * in evaluating the functions. However, it would be difficult or impossible to pre-instantiate
     * the desired template classes, a requirement of the LSST coding standards.
     *
     * \ingroup fw
     */
    template<typename PixelT>
    class Kernel : private lsst::mwi::data::LsstBase {
    
    public:
        typedef boost::shared_ptr<lsst::fw::function::Function2<PixelT> > KernelFunctionPtrType;
        typedef boost::shared_ptr<lsst::fw::function::Function2<double> > SpatialFunctionPtrType;

        explicit Kernel();
        
        explicit Kernel(
            unsigned int cols,
            unsigned int rows,
            unsigned int nKernelParams
        );
        
        explicit Kernel(
            unsigned int cols,
            unsigned int rows,
            unsigned int nKernelParams,
            SpatialFunctionPtrType spatialFunction
        );
        
        explicit Kernel(
            unsigned int cols,
            unsigned int rows,
            unsigned int nKernelParams,
            SpatialFunctionPtrType spatialFunction,
            std::vector<std::vector<double> > const &spatialParameters
        );

        virtual ~Kernel() {};
        
        Image<PixelT> computeNewImage(
            PixelT &imSum,
            double x = 0.0,
            double y = 0.0,
            bool doNormalize = true
        ) const;        
            
        /**
         * \brief Compute an image (pixellized representation of the kernel) in place
         *
         * \return the sum of the image pixels. Warning: the sum will be wrong
         * if doNormalize is true and the kernel is of integer type.
         *
         * x, y are ignored if there is no spatial function.
         *
         * \throw std::invalid_argument if the image is the wrong size
         */
        virtual void computeImage(
            Image<PixelT> &image,   ///< image whose pixels are to be set
            PixelT &imSum,  ///< sum of image pixels
            double x = 0.0, ///< x (column position) at which to compute spatial function
            double y = 0.0, ///< y (row position) at which to compute spatial function
            bool doNormalize = true ///< normalize the image (so sum is 1)?
        ) const = 0;
    
        inline unsigned int getCols() const;
        
        inline unsigned int getRows() const;
        
        inline unsigned int getCtrCol() const;

        inline unsigned int getCtrRow() const;
        
        virtual std::vector<double> getKernelParameters(double x = 0, double y = 0) const;
    
        inline unsigned int getNKernelParameters() const;
    
        inline unsigned int getNSpatialParameters() const;
    
        inline std::vector<std::vector<double> > getSpatialParameters() const;
            
        inline bool isSpatiallyVarying() const;    
    
        inline void setKernelParameters(std::vector<double> const &params);
        
        virtual void setSpatialParameters(const std::vector<std::vector<double> > params);
    
    protected:
        /**
         * \brief Set the kernel parameters.
         *
         * Use with caution. This function is const but setting the kernel parameters
         * is not always a const operation. For example if the kernel has a spatial model
         * then it is a const operation if part of computing the kernel at a particular position.
         * But if there is no spatial model then it is not const.
         *
         * \throw std::invalid_argument if the params vector is the wrong length
         */
        virtual void basicSetKernelParameters(std::vector<double> const &params) const = 0;
                
        /**
         * \brief Get the current kernel parameters.
         *
         * Assumes there is no spatial model.
         */
        virtual std::vector<double> getCurrentKernelParameters() const = 0;
        
        /**
         * \brief Compute the kernel parameters at a particular spatial position.
         *
         * \throw std::invalid_argument if the params vector is the wrong length
         */
        virtual std::vector<double> getKernelParametersFromSpatialModel(double x, double y) const;
        
        virtual void setKernelParametersFromSpatialModel(double x, double y) const;
           
    private:
        unsigned int _cols;
        unsigned int _rows;
        unsigned int _ctrCol;
        unsigned int _ctrRow;
        unsigned int _nKernelParams;
        bool _isSpatiallyVarying;
        SpatialFunctionPtrType _spatialFunctionPtr;
        std::vector<std::vector<double> > _spatialParams;
    };
    
    
    /**
     * \brief A kernel created from an Image
     *
     * It has no adjustable parameters and so cannot be spatially varying.
     *
     * \ingroup fw
     */
    template<typename PixelT>
    class FixedKernel : public Kernel<PixelT> {
    public:
        explicit FixedKernel();

        explicit FixedKernel(
            Image<PixelT> const &image
        );
        
        virtual ~FixedKernel() {};
    
        virtual void computeImage(
            Image<PixelT> &image,
            PixelT &imSum,
            double x = 0.0,
            double y = 0.0,
            bool doNormalize = true
        ) const;
            
    protected:
        virtual void basicSetKernelParameters(std::vector<double> const &params) const;
            
        virtual std::vector<double> getCurrentKernelParameters() const;

    private:
        Image<PixelT> _image;
        PixelT _sum;
    };
    
    
    /**
     * \brief A kernel described by a function.
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * \ingroup fw
     */
    template<typename PixelT>
    class AnalyticKernel : public Kernel<PixelT> {
    public:
        explicit AnalyticKernel();

        explicit AnalyticKernel(
            typename Kernel<PixelT>::KernelFunctionPtrType kernelFunction,
            unsigned int cols,
            unsigned int rows
        );
        
        explicit AnalyticKernel(
            typename Kernel<PixelT>::KernelFunctionPtrType kernelFunction,
            unsigned int cols,
            unsigned int rows,
            typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction
        );
        
        explicit AnalyticKernel(
            typename Kernel<PixelT>::KernelFunctionPtrType kernelFunction,
            unsigned int cols,
            unsigned int rows,
            typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction,
            std::vector<std::vector<double> > const &spatialParameters
        );
        
        virtual ~AnalyticKernel() {};
    
        virtual void computeImage(
            Image<PixelT> &image,
            PixelT &imSum,
            double x = 0.0,
            double y = 0.0,
            bool doNormalize = true
        ) const;
    
        virtual typename Kernel<PixelT>::KernelFunctionPtrType getKernelFunction() const;
            
    protected:
        virtual void basicSetKernelParameters(std::vector<double> const &params) const;

        virtual std::vector<double> getCurrentKernelParameters() const;
    
    private:
        typename Kernel<PixelT>::KernelFunctionPtrType _kernelFunctionPtr;
    };
    
    
    /**
     * \brief A kernel that is a linear combination of fixed basis kernels.
     * 
     * Convolution may be performed by first convolving the image
     * with each fixed kernel, then adding the resulting images using the (possibly
     * spatially varying) kernel coefficients.
     *
     * \ingroup fw
     */
    template<typename PixelT>
    class LinearCombinationKernel : public Kernel<PixelT> {
    
    public:
        typedef std::vector<boost::shared_ptr<Kernel<PixelT> > > KernelListType;

        explicit LinearCombinationKernel();

        explicit LinearCombinationKernel(
            KernelListType const &kernelList,
            std::vector<double> const &kernelParameters
        );
        
        explicit LinearCombinationKernel(
            KernelListType const &kernelList,
            typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction
        );
        
        explicit LinearCombinationKernel(
            KernelListType const &kernelList,
            typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction,
            std::vector<std::vector<double> > const &spatialParameters
        );
        
        virtual ~LinearCombinationKernel() {};
    
        virtual void computeImage(
            Image<PixelT> &image,
            PixelT &imSum,
            double x = 0.0,
            double y = 0.0,
            bool doNormalize = true
        ) const;
                
        /**
         * \brief Get the fixed basis kernels
         */
        virtual KernelListType const &getKernelList() const;
        
        /**
         * \brief Check that all kernels have the same size and center and that none are spatially varying
         *
         * \throw std::invalid_argument if the check fails
         */
        void checkKernelList(const KernelListType &kernelList) const;
        
    protected:
        virtual void basicSetKernelParameters(std::vector<double> const &params) const;

        virtual std::vector<double> getCurrentKernelParameters() const;
    
    private:
        KernelListType _kernelList;
        mutable std::vector<double> _kernelParams;
    };

}   // namespace fw
}   // namespace lsst
    
// Included definitions for templated and inline member functions
#include <lsst/fw/Kernel/Kernel.cc>

#endif // !defined(LSST_FW_Kernel_H)
