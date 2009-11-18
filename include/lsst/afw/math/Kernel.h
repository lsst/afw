// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_KERNEL_H
#define LSST_AFW_MATH_KERNEL_H
/**
 * @file
 *
 * @brief Declare the Kernel class and subclasses.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <vector>

#include "boost/mpl/or.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_same.hpp"
#include "boost/type_traits/is_base_and_derived.hpp"

#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/export.hpp"

#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/traits.h"
#include "lsst/afw/math/ConvolutionVisitor.h"

namespace lsst {
namespace afw {

namespace formatters {
class KernelFormatter;
}

namespace math {

#ifndef SWIG
using boost::serialization::make_nvp;
#endif

    /**
     * @brief Kernels are used for convolution with MaskedImages and (eventually) Images
     *
     * Kernel is a virtual base class; it cannot be instantiated. The following notes apply to
     * Kernel and to its subclasses.
     *
     * The template type should usually be float or double; integer kernels
     * should be used with caution because they do not normalize well.
     *
     * The center pixel of a Kernel is at index: (width-1)/2, (height-1)/2. Thus it is centered along
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
     * The convolve function computes the spatial function at the pixel position (not index) of the image.
     * See the convolve function for details.
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
     * @ingroup afw
     */
    class Kernel : public lsst::daf::data::LsstBase, public lsst::daf::base::Persistable {

    public:
        typedef double Pixel;
        typedef boost::shared_ptr<Kernel> Ptr;
        typedef boost::shared_ptr<const Kernel> ConstPtr;
        typedef boost::shared_ptr<lsst::afw::math::Function2<double> > SpatialFunctionPtr;
        typedef lsst::afw::math::Function2<double> SpatialFunction;
        typedef lsst::afw::math::NullFunction2<double> NullSpatialFunction;

        // Traits values for this class of Kernel
        typedef generic_kernel_tag kernel_fill_factor;

        explicit Kernel();

        explicit Kernel(int width, int height, unsigned int nKernelParams,
                        SpatialFunction const &spatialFunction=NullSpatialFunction());
        explicit Kernel(int width, int height, const std::vector<SpatialFunctionPtr> spatialFunctionList);

        virtual ~Kernel() {};

        /**
         * @brief Return a pointer to a deep copy of this kernel
         *
         * This kernel exists instead of a copy constructor
         * so one can obtain a copy of an actual kernel
         * instead of a useless copy of the base class.
         *
         * Every kernel subclass must override this method.
         *
         * @return a pointer to a deep copy of the kernel
         */
        virtual Kernel::Ptr clone() const = 0;

        /**
         * @brief Compute an image (pixellized representation of the kernel) in place
         *
         * x, y are ignored if there is no spatial function.
         *
         * @note computeNewImage has been retired; it doesn't need to be a member
         *
         * @throw lsst::pex::exceptions::InvalidParameterException if the image is the wrong size
         */
        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,   ///< image whose pixels are to be set (output)
            bool doNormalize,   ///< normalize the image (so sum is 1)?
            double x = 0.0, ///< x (column position) at which to compute spatial function
            double y = 0.0  ///< y (row position) at which to compute spatial function
        ) const = 0;

        virtual ImageConvolutionVisitor::Ptr computeImageConvolutionVisitor(
            lsst::afw::image::PointD const & location
        ) const;


        virtual FourierConvolutionVisitor::Ptr computeFourierConvolutionVisitor(
           lsst::afw::image::PointD const & location
        ) const;

        /**
        * @brief Return the Kernel's dimensions (width, height)
        */
        std::pair<int, int> const getDimensions() const {
            return std::pair<int, int>(_width, _height); }

        /**
         * @brief Return the Kernel's width
         */
        inline int getWidth() const {
            return _width;
        };

        /**
         * @brief Return the Kernel's height
         */
        inline int getHeight() const {
            return _height;
        };

        /**
         * @brief Return index of the center column
         */
        inline int getCtrX() const {
            return _ctrX;
        };

        /**
         * @brief Return index of the center row
         */
        inline int getCtrY() const {
            return _ctrY;
        };

        /**
         * @brief Return the number of kernel parameters (0 if none)
         */
        inline unsigned int getNKernelParameters() const {
            return _nKernelParams;
        };

        /**
         * @brief Return the number of spatial parameters (0 if not spatially varying)
         */
        inline int getNSpatialParameters() const {
            return this->isSpatiallyVarying() ? _spatialFunctionList[0]->getNParameters() : 0;
        };

        SpatialFunctionPtr getSpatialFunction(unsigned int index) const;

        std::vector<SpatialFunctionPtr> getSpatialFunctionList() const;

        virtual std::vector<double> getKernelParameters() const;

        /**
        * @brief Set the center index, x axis
        */
        inline void setCtrX(int ctrX) {
            _ctrX = ctrX;
        };

        /**
        * @brief Set the center index, y axis
        */
        inline void setCtrY(int ctrY) {
            _ctrY = ctrY;
        };

        /**
         * @brief Return the spatial parameters parameters (an empty vector if not spatially varying)
         */
        inline std::vector<std::vector<double> > getSpatialParameters() const {
            std::vector<std::vector<double> > spatialParams;
            std::vector<SpatialFunctionPtr>::const_iterator spFuncIter = _spatialFunctionList.begin();
            for ( ; spFuncIter != _spatialFunctionList.end(); ++spFuncIter) {
                spatialParams.push_back((*spFuncIter)->getParameters());
            }
            return spatialParams;
        };

        /**
         * @brief Return true iff the kernel is spatially varying (has a spatial function)
         */
        inline bool isSpatiallyVarying() const {
            return _spatialFunctionList.size() != 0;
        };

        /**
         * @brief Set the kernel parameters of a spatially invariant kernel.
         *
         * @throw lsst::pex::exceptions::RuntimeErrorException if the kernel has a spatial function
         * @throw lsst::pex::exceptions::InvalidParameterException if the params vector is the wrong length
         */
        inline void setKernelParameters(std::vector<double> const &params) {
            if (this->isSpatiallyVarying()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                    "Kernel is spatially varying");
            }
            const unsigned int nParams = this->getNKernelParameters();
            if (nParams != params.size()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  (boost::format("Number of parameters is wrong, saw %d expected %d") %
                                   nParams % params.size()).str());
            }
            for (unsigned int ii = 0; ii < nParams; ++ii) {
                this->setKernelParameter(ii, params[ii]);
            }
        };

        /**
         * @brief Set the kernel parameters of a 2-component spatially invariant kernel.
         *
         * @warning This is a low-level method intended for maximum efficiency when using warping kernels.
         * No error checking is performed. Use the std::vector<double> form if you want safety.
         */
        inline void setKernelParameters(std::pair<double, double> const& params) {
            this->setKernelParameter(0, params.first);
            this->setKernelParameter(1, params.second);
        };

        void setSpatialParameters(const std::vector<std::vector<double> > params);

        void computeKernelParametersFromSpatialModel(
            std::vector<double> &kernelParams, double x, double y) const;

        virtual std::string toString(std::string prefix = "") const;

        virtual void toFile(std::string fileName) const;

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;

        void setKernelParametersFromSpatialModel(double x, double y) const;

        std::vector<SpatialFunctionPtr> _spatialFunctionList;

    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::KernelFormatter);

        int _width;
        int _height;
        int _ctrX;
        int _ctrY;
        unsigned int _nKernelParams;
    };

    typedef std::vector<Kernel::Ptr> KernelList;

    /**
     * @brief A kernel created from an Image
     *
     * It has no adjustable parameters and so cannot be spatially varying.
     *
     * @ingroup afw
     */
    class FixedKernel : public Kernel {
    public:
        typedef boost::shared_ptr<FixedKernel> Ptr;
        typedef boost::shared_ptr<const FixedKernel> ConstPtr;

        explicit FixedKernel();

        explicit FixedKernel(
            lsst::afw::image::Image<Pixel> const &image
        );

        virtual ~FixedKernel() {};

        virtual Kernel::Ptr clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual std::string toString(std::string prefix = "") const;

    private:
        lsst::afw::image::Image<Pixel> _image;
        Pixel _sum;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k",
                        boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("img", _image);
                ar & make_nvp("sum", _sum);
            };
    };


    /**
     * @brief A kernel described by a function.
     *
     * The function's x, y arguments are as follows:
     * * -getCtrX(), -getCtrY() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getWidth() - 1) - getCtrX(), (getHeight() - 1) - getCtrY() for the upper right pixel
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * @ingroup afw
     */
    class AnalyticKernel : public Kernel {
    public:
        typedef boost::shared_ptr<AnalyticKernel> Ptr;
        typedef boost::shared_ptr<const AnalyticKernel> ConstPtr;
        typedef lsst::afw::math::Function2<Pixel> KernelFunction;
        typedef boost::shared_ptr<lsst::afw::math::Function2<Pixel> > KernelFunctionPtr;

        explicit AnalyticKernel();

        explicit AnalyticKernel(
            int width,
            int height,
            KernelFunction const &kernelFunction,
            Kernel::SpatialFunction const &spatialFunction=NullSpatialFunction()
        );

        explicit AnalyticKernel(
            int width,
            int height,
            KernelFunction const &kernelFunction,
            std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList
        );

        virtual ~AnalyticKernel() {};

        virtual Kernel::Ptr clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual std::vector<double> getKernelParameters() const;

        virtual KernelFunctionPtr getKernelFunction() const;

        virtual std::string toString(std::string prefix = "") const;

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;

        KernelFunctionPtr _kernelFunctionPtr;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k",
                        boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("fn", _kernelFunctionPtr);
            };
    };


    /**
     * @brief A kernel that has only one non-zero pixel (of value 1)
     *
     * It has no adjustable parameters and so cannot be spatially varying.
     *
     * @ingroup afw
     */
    class DeltaFunctionKernel : public Kernel {
    public:
        typedef boost::shared_ptr<DeltaFunctionKernel> Ptr;
        typedef boost::shared_ptr<const DeltaFunctionKernel> ConstPtr;
        // Traits values for this class of Kernel
        typedef deltafunction_kernel_tag kernel_fill_factor;

        explicit DeltaFunctionKernel(
            int width,
            int height,
            lsst::afw::image::PointI const &point
        );

        virtual ~DeltaFunctionKernel() {};

        virtual Kernel::Ptr clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        lsst::afw::image::PointI getPixel() const { return _pixel; }

        virtual std::string toString(std::string prefix = "") const;

    private:
        lsst::afw::image::PointI _pixel;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
            boost::serialization::void_cast_register<
                DeltaFunctionKernel, Kernel>(
                    static_cast<DeltaFunctionKernel*>(0),
                    static_cast<Kernel*>(0));
        };
    };


    /**
     * @brief A kernel that is a linear combination of fixed basis kernels.
     *
     * Convolution may be performed by first convolving the image
     * with each fixed kernel, then adding the resulting images using the (possibly
     * spatially varying) kernel coefficients.
     *
     * The basis kernels are cloned (deep copied) so you may safely modify your own copies.
     *
     * Warnings:
     * - This class does not normalize the individual basis kernels; they are used "as is".
     *
     * @ingroup afw
     */
    class LinearCombinationKernel : public Kernel {
    public:
        typedef boost::shared_ptr<LinearCombinationKernel> Ptr;
        typedef boost::shared_ptr<const LinearCombinationKernel> ConstPtr;

        explicit LinearCombinationKernel();

        explicit LinearCombinationKernel(
            KernelList const &kernelList,
            std::vector<double> const &kernelParameters
        );

        explicit LinearCombinationKernel(
            KernelList const &kernelList,
            Kernel::SpatialFunction const &spatialFunction
        );

        explicit LinearCombinationKernel(
            KernelList const &kernelList,
            std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList
        );

        virtual ~LinearCombinationKernel() {};

        virtual Kernel::Ptr clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual ImageConvolutionVisitor::Ptr computeImageConvolutionVisitor(
            lsst::afw::image::PointD const & location
        ) const;

        virtual std::vector<double> getKernelParameters() const;

        virtual KernelList const &getKernelList() const;

        std::vector<double> getKernelSumList() const;

        void checkKernelList(const KernelList &kernelList) const;

        virtual std::string toString(std::string prefix = "") const;

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;

    private:
        void _setKernelList(KernelList const &kernelList);
        
        KernelList _kernelList; ///< basis kernels
        std::vector<boost::shared_ptr<lsst::afw::image::Image<Pixel> > > _kernelImagePtrList;
            ///< image of each basis kernel (a cache)
        std::vector<double> _kernelSumList; ///< sum of each basis kernel (a cache)
        mutable std::vector<double> _kernelParams;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k", boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("klist", _kernelList);
                ar & make_nvp("kimglist", _kernelImagePtrList);
                ar & make_nvp("ksumlist", _kernelSumList);
                ar & make_nvp("params", _kernelParams);
            };
    };

    /**
     * @brief A kernel described by a pair of functions: func(x, y) = colFunc(x) * rowFunc(y)
     *
     * The function's x, y arguments are as follows:
     * * -getCtrX(), -getCtrY() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getWidth() - 1) - getCtrX(), (getHeight() - 1) - getCtrY() for the upper right pixel
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * @ingroup afw
     */
    class SeparableKernel : public Kernel {
    public:
        typedef boost::shared_ptr<SeparableKernel> Ptr;
        typedef boost::shared_ptr<const SeparableKernel> ConstPtr;
        typedef lsst::afw::math::Function1<Pixel> KernelFunction;
        typedef boost::shared_ptr<KernelFunction> KernelFunctionPtr;

        explicit SeparableKernel();

        explicit SeparableKernel(
            int width, int height,
            KernelFunction const& kernelColFunction,
            KernelFunction const& kernelRowFunction,
            Kernel::SpatialFunction const& spatialFunction=NullSpatialFunction()
        );

        explicit SeparableKernel(int width, int height,
                                 KernelFunction const& kernelColFunction,
                                 KernelFunction const& kernelRowFunction,
                                 std::vector<Kernel::SpatialFunctionPtr> const& spatialFunctionList);
        virtual ~SeparableKernel() {};

        virtual Kernel::Ptr clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        double computeVectors(
            std::vector<Pixel> &colList,
            std::vector<Pixel> &rowList,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual std::vector<double> getKernelParameters() const;

        KernelFunctionPtr getKernelColFunction() const;

        KernelFunctionPtr getKernelRowFunction() const;

        virtual std::string toString(std::string prefix = "") const;

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;

    private:
        double basicComputeVectors(
            std::vector<Pixel> &colList,
            std::vector<Pixel> &rowList,
            bool doNormalize
        ) const;

        KernelFunctionPtr _kernelColFunctionPtr;
        KernelFunctionPtr _kernelRowFunctionPtr;
        mutable std::vector<Pixel> _localColList;  // used by computeImage
        mutable std::vector<Pixel> _localRowList;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k",
                    boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("colfn", _kernelColFunctionPtr);
                ar & make_nvp("rowfn", _kernelRowFunctionPtr);
                ar & make_nvp("cols", _localColList);
                ar & make_nvp("rows", _localRowList);
            };
    };

}}}   // lsst:afw::math

namespace boost {
namespace serialization {

template <class Archive>
inline void save_construct_data(
    Archive& ar, lsst::afw::math::DeltaFunctionKernel const* k,
    unsigned int const file_version) {
    int width = k->getWidth();
    int height = k->getHeight();
    int x = k->getPixel().getX();
    int y = k->getPixel().getY();
    ar << make_nvp("width", width);
    ar << make_nvp("height", height);
    ar << make_nvp("pixX", x);
    ar << make_nvp("pixY", y);
};

template <class Archive>
inline void load_construct_data(
    Archive& ar, lsst::afw::math::DeltaFunctionKernel* k,
    unsigned int const file_version) {
    int width;
    int height;
    int x;
    int y;
    ar >> make_nvp("width", width);
    ar >> make_nvp("height", height);
    ar >> make_nvp("pixX", x);
    ar >> make_nvp("pixY", y);
    ::new(k) lsst::afw::math::DeltaFunctionKernel(
        width, height, lsst::afw::image::PointI(x, y));
};

}}

#endif // !defined(LSST_AFW_MATH_KERNEL_H)
