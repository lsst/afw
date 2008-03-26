// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_KERNEL_H
#define LSST_AFW_MATH_KERNEL_H
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

#include <boost/mpl/or.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>
#include <vw/Image.h>

#include <lsst/mwi/data/LsstBase.h>
#include <lsst/afw/math/Function.h>
#include <lsst/afw/image/Image.h>
#include <lsst/afw/image/Kernel/traits.h>

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
    class Kernel : public lsst::mwi::data::LsstBase {
    
    public:
        typedef double PixelT;
        typedef boost::shared_ptr<Kernel> PtrT;
        typedef boost::shared_ptr<lsst::fw::function::Function2<PixelT> > KernelFunctionPtrType;
        typedef boost::shared_ptr<lsst::fw::function::Function2<double> > SpatialFunctionPtrType;
        // Traits values for this class of Kernel
        typedef generic_kernel_tag kernel_fill_factor;

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
         * \throw lsst::mwi::exceptions::InvalidParameter if the image is the wrong size
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

        void computeKernelParametersFromSpatialModel(std::vector<double> &kernelParams, double x, double y) const;
    
        virtual std::string toString(std::string prefix = "") const {
            std::ostringstream os;
            os << prefix << "Kernel:" << std::endl;
            os << prefix << "..rows, cols: " << _rows << ", " << _cols << std::endl;
            os << prefix << "..ctrRow, Col: " << _ctrRow << ", " << _ctrCol << std::endl;
            os << prefix << "..isSpatiallyVarying: " << (_isSpatiallyVarying ? "True" : "False") << std::endl;
            os << prefix << "..spatialFunction: " << (_spatialFunctionPtr ? _spatialFunctionPtr->toString() : "None") << std::endl;
            os << prefix << "..nKernelParams: " << _nKernelParams << std::endl;
            os << prefix << "..spatialParams:" << std::endl;
            for (std::vector<std::vector<double> >::const_iterator i = _spatialParams.begin(); i != _spatialParams.end(); ++i) {
                os << prefix << "....[ ";
                for (std::vector<double>::const_iterator j = i->begin(); j != i->end(); ++j) {
                    if (j != i->begin()) os << ", ";
                    os << *j;
                }
                os << " ]" << std::endl;
            }
            return os.str();
        };

    protected:
        /**
         * \brief Set the kernel parameters.
         *
         * Use with caution. This function is const but setting the kernel parameters
         * is not always a const operation. For example if the kernel has a spatial model
         * then it is a const operation if part of computing the kernel at a particular position.
         * But if there is no spatial model then it is not const.
         *
         * \throw lsst::mwi::exceptions::InvalidParameter if the params vector is the wrong length
         */
        virtual void basicSetKernelParameters(std::vector<double> const &params) const {};
                
        /**
         * \brief Get the current kernel parameters.
         *
         * Assumes there is no spatial model.
         */
        virtual std::vector<double> getCurrentKernelParameters() const { return std::vector<double>(); }
           
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
     * \brief A list of Kernels
     *
     * This is basically a wrapper for an stl container, but defines
     * a conversion from KernelList<K1> to KernelList<K2> providing
     * that K1 is derived from K2 (or that K1 == K2)
     *
     * \ingroup fw
     */
    template<typename _KernelT=Kernel>
    class KernelList : public std::vector<typename _KernelT::PtrT> {
    public:
        typedef _KernelT KernelT;

        KernelList() { }
        
        template<typename Kernel2T>
        KernelList(const KernelList<Kernel2T>& k2l) :
            std::vector<typename KernelT::PtrT>(k2l.size())
            {
#if !defined(SWIG)
                BOOST_STATIC_ASSERT((
                                     boost::mpl::or_<
                                     boost::is_same<KernelT, Kernel2T>,
                                     boost::is_base_and_derived<KernelT, Kernel2T>
                                     >::value
                                    ));
#endif
                copy(k2l.begin(), k2l.end(), this->begin());
            }
    };
    
    /**
     * \brief A kernel created from an Image
     *
     * It has no adjustable parameters and so cannot be spatially varying.
     *
     * \ingroup fw
     */
    class FixedKernel : public Kernel {
    public:
        typedef boost::shared_ptr<FixedKernel> PtrT;

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
            
        virtual std::string toString(std::string prefix = "") const {
            std::ostringstream os;
            os << prefix << "FixedKernel:" << std::endl;
            os << prefix << "..sum: " << _sum << std::endl;
            os << Kernel::toString(prefix + "\t");
            return os.str();
        };

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
     * The function's x, y arguments are:
     * * -getCtrCol(), -getCtrRow() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getCols() - 1) - getCtrCol(), (getRows() - 1) - getCtrRow() for the upper right pixel
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * \ingroup fw
     */
    class AnalyticKernel : public Kernel {
    public:
        typedef boost::shared_ptr<AnalyticKernel> PtrT;
        
        explicit AnalyticKernel();

        explicit AnalyticKernel(
            Kernel::KernelFunctionPtrType kernelFunction,
            unsigned int cols,
            unsigned int rows
        );
        
        explicit AnalyticKernel(
            Kernel::KernelFunctionPtrType kernelFunction,
            unsigned int cols,
            unsigned int rows,
            Kernel::SpatialFunctionPtrType spatialFunction
        );
        
        explicit AnalyticKernel(
            Kernel::KernelFunctionPtrType kernelFunction,
            unsigned int cols,
            unsigned int rows,
            Kernel::SpatialFunctionPtrType spatialFunction,
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
    
        virtual Kernel::KernelFunctionPtrType getKernelFunction() const;
            
        virtual std::string toString(std::string prefix = "") const {
            std::ostringstream os;
            os << prefix << "AnalyticKernel:" << std::endl;
            os << prefix << "..function: " << (_kernelFunctionPtr ? _kernelFunctionPtr->toString() : "None") << std::endl;
            os << Kernel::toString(prefix + "\t");
            return os.str();
        };

    protected:
        virtual void basicSetKernelParameters(std::vector<double> const &params) const;

        virtual std::vector<double> getCurrentKernelParameters() const;
    
    private:
        Kernel::KernelFunctionPtrType _kernelFunctionPtr;
    };
    
    
    /**
     * \brief A kernel that has only one non-zero pixel
     *
     * The function's x, y arguments are:
     * * -getCtrCol(), -getCtrRow() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getCols() - 1) - getCtrCol(), (getRows() - 1) - getCtrRow() for the upper right pixel
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * \ingroup fw
     */
    class DeltaFunctionKernel : public Kernel {
    public:
        typedef boost::shared_ptr<DeltaFunctionKernel> PtrT;
        // Traits values for this class of Kernel
        typedef deltafunction_kernel_tag kernel_fill_factor;

        explicit DeltaFunctionKernel(int pixelCol,
                                     int pixelRow,
                                     unsigned int cols,
                                     unsigned int rows);

        virtual void computeImage(
            Image<PixelT> &image,
            PixelT &imSum,
            double x = 0.0,
            double y = 0.0,
            bool doNormalize = true
        ) const;

        std::pair<int, int> getPixel() const { return _pixel; }

        virtual std::string toString(std::string prefix = "") const {
            const int pixelCol = getPixel().first; // active pixel in Kernel
            const int pixelRow = getPixel().second;

            std::ostringstream os;            
            os << prefix << "DeltaFunctionKernel:" << std::endl;
            os << prefix << "Pixel (c,r) " << pixelCol << "," << pixelRow << ")" << std::endl;
            os << Kernel::toString(prefix + "\t");
            return os.str();
        };

    private:
        std::pair<int, int> _pixel;
    };
    

    /**
     * \brief A kernel that is a linear combination of fixed basis kernels.
     * 
     * Convolution may be performed by first convolving the image
     * with each fixed kernel, then adding the resulting images using the (possibly
     * spatially varying) kernel coefficients.
     *
     * Warnings:
     * - This class does not normalize the individual basis kernels; they are used "as is".
     * - The kernels are assumed to be invariant; do not try to modify the basis kernels
     *   while using LinearCombinationKernel.
     *
     * \ingroup fw
     */
    class LinearCombinationKernel : public Kernel {
    public:
        typedef boost::shared_ptr<LinearCombinationKernel> PtrT;
        typedef KernelList<Kernel> KernelListType;

        explicit LinearCombinationKernel();

        explicit LinearCombinationKernel(
            KernelListType const &kernelList,
            std::vector<double> const &kernelParameters
        );
        
        explicit LinearCombinationKernel(
            KernelListType const &kernelList,
            Kernel::SpatialFunctionPtrType spatialFunction
        );
        
        explicit LinearCombinationKernel(
            KernelListType const &kernelList,
            Kernel::SpatialFunctionPtrType spatialFunction,
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
                
        virtual KernelListType const &getKernelList() const;
        
        void checkKernelList(const KernelListType &kernelList) const;
        
        virtual std::string toString(std::string prefix = "") const {
            std::ostringstream os;
            os << prefix << "LinearCombinationKernel:" << std::endl;
            os << prefix << "..Kernels:" << std::endl;
            for (KernelListType::const_iterator i = _kernelList.begin(); i != _kernelList.end(); ++i) {
                os << (*i)->toString(prefix + "\t");
            }
            os << "..parameters: [ ";
            for (std::vector<double>::const_iterator i = _kernelParams.begin(); i != _kernelParams.end(); ++i) {
                if (i != _kernelParams.begin()) os << ", ";
                os << *i;
            }
            os << " ]" << std::endl;
            os << Kernel::toString(prefix + "\t");
            return os.str();
        };

    protected:
        virtual void basicSetKernelParameters(std::vector<double> const &params) const;

        virtual std::vector<double> getCurrentKernelParameters() const;
    
    private:
        void _computeKernelImageList();
        KernelListType _kernelList;
        std::vector<boost::shared_ptr<Image<PixelT> > > _kernelImagePtrList;
        mutable std::vector<double> _kernelParams;
    };
}}   // lsst:fw
    
// Included definitions for templated and inline member functions
#ifndef SWIG // don't bother SWIG with .cc files
#include <lsst/afw/math/Kernel.cc>
#endif

#endif // !defined(LSST_AFW_MATH_KERNEL_H)
