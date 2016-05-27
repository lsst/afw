// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "boost/mpl/or.hpp"

#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/export.hpp"

#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/traits.h"

#include "lsst/afw/table/io/Persistable.h"

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
    class Kernel : public lsst::daf::base::Citizen, public lsst::daf::base::Persistable,
                   public afw::table::io::PersistableFacade<Kernel>,
                   public afw::table::io::Persistable
    {

    public:
        typedef double Pixel;
        typedef PTR(Kernel) Ptr;
        typedef CONST_PTR(Kernel) ConstPtr;
        typedef PTR(lsst::afw::math::Function2<double>) SpatialFunctionPtr;
        typedef lsst::afw::math::Function2<double> SpatialFunction;
        typedef lsst::afw::math::NullFunction2<double> NullSpatialFunction;

        // Traits values for this class of Kernel
        typedef generic_kernel_tag kernel_fill_factor;

        /**
         * @brief Construct a null Kernel of size 0,0.
         *
         * A null constructor is primarily intended for persistence.
         */
        explicit Kernel();

        /**
         * @brief Construct a spatially invariant Kernel or a spatially varying Kernel with one spatial function
         * that is duplicated as needed.
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if a spatial function is specified
         * and the kernel has no parameters.
         * @throw lsst::pex::exceptions::InvalidParameterError if a width or height < 1
         */
        explicit Kernel(
            int width,                      ///< number of columns
            int height,                     ///< number of height
            unsigned int nKernelParams,     ///< number of kernel parameters
            SpatialFunction const &spatialFunction=NullSpatialFunction()
                                            ///< spatial function, or NullSpatialFunction() if none specified
        );

        /**
         * @brief Construct a spatially varying Kernel with a list of spatial functions (one per kernel parameter)
         *
         * Note: if the list of spatial functions is empty then the kernel is not spatially varying.
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if a width or height < 1
         */
        explicit Kernel(
            int width,  ///< number of columns
            int height, ///< number of height
            const std::vector<SpatialFunctionPtr> spatialFunctionList
                        ///< list of spatial function, one per kernel parameter
        );

        virtual ~Kernel() {}

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
        virtual PTR(Kernel) clone() const = 0;

        /**
         * @brief Compute an image (pixellized representation of the kernel) in place
         *
         * x, y are ignored if there is no spatial function.
         *
         * @return The kernel sum
         *
         * @note computeNewImage has been retired; it doesn't need to be a member
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the image is the wrong size
         * @throw lsst::pex::exceptions::OverflowError if doNormalize is true and the kernel sum is
         * exactly 0
         */
        double computeImage(
            lsst::afw::image::Image<Pixel> &image,   ///< image whose pixels are to be set (output);
                ///< xy0 of the image will be set to -kernel.getCtr()
            bool doNormalize,   ///< normalize the image (so sum is 1)?
            double x = 0.0, ///< x (column position) at which to compute spatial function
            double y = 0.0  ///< y (row position) at which to compute spatial function
        ) const;

        /**
        * @brief Return the Kernel's dimensions (width, height)
        */
        geom::Extent2I const getDimensions() const {
            return geom::Extent2I(_width, _height); }

        void setDimensions(geom::Extent2I dims) {
            _width = dims.getX();
            _height = dims.getY();
            
        }
        inline void setWidth(int width) { _width = width; }
        inline void setHeight(int height) { _height = height; }

        /**
         * @brief Return the Kernel's width
         */
        inline int getWidth() const {
            return _width;
        }

        /**
         * @brief Return the Kernel's height
         */
        inline int getHeight() const {
            return _height;
        }
        
        /**
         * @brief Return index of kernel's center
         */
        inline lsst::afw::geom::Point2I getCtr() const {
            return lsst::afw::geom::Point2I(_ctrX, _ctrY);
        }

        /**
         * @brief Return x index of kernel's center
         *
         * @deprecated Use getCtr instead
         */
        inline int getCtrX() const {
            return _ctrX;
        }

        /**
         * @brief Return y index of kernel's center
         *
         * @deprecated Use getCtr instead
         */
        inline int getCtrY() const {
            return _ctrY;
        }
        
        /**
         * @brief return parent bounding box, with XY0 = -center
         */
        inline lsst::afw::geom::Box2I getBBox() const {
            return lsst::afw::geom::Box2I(
                lsst::afw::geom::Point2I(-_ctrX, -_ctrY),
                lsst::afw::geom::Extent2I(_width, _height)
            );
        }

        /**
         * @brief Return the number of kernel parameters (0 if none)
         */
        inline unsigned int getNKernelParameters() const {
            return _nKernelParams;
        }

        /**
         * @brief Return the number of spatial parameters (0 if not spatially varying)
         */
        inline int getNSpatialParameters() const {
            return this->isSpatiallyVarying() ? _spatialFunctionList[0]->getNParameters() : 0;
        }

        /**
         * @brief Return a clone of the specified spatial function (one component of the spatial model)
         *
         * @return a shared pointer to a spatial function. The function is a deep copy, so setting its parameters
         * has no effect on the kernel.
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if kernel not spatially varying
         * @throw lsst::pex::exceptions::InvalidParameterError if index out of range
         */
        SpatialFunctionPtr getSpatialFunction(
            unsigned int index  ///< index of desired spatial function;
                                ///< must be in range [0, number spatial parameters - 1]
        ) const;

        /**
         * @brief Return a list of clones of the spatial functions.
         *
         * @return a list of shared pointers to spatial functions. The functions are deep copies,
         * so setting their parameters has no effect on the kernel.
         */
        std::vector<SpatialFunctionPtr> getSpatialFunctionList() const;

        /// Return a particular Kernel Parameter (no bounds checking).  This version is slow,
        /// but specialisations may be faster
        virtual double getKernelParameter(unsigned int i) const {
            return getKernelParameters()[i];
        }
        
        /**
         * @brief Return the current kernel parameters
         *
         * If the kernel is spatially varying then the parameters are those last computed.
         * See also computeKernelParametersFromSpatialModel.
         * If there are no kernel parameters then returns an empty vector.
         */
        virtual std::vector<double> getKernelParameters() const;
        
        /**
         * Given a bounding box for pixels one wishes to compute by convolving an image with this kernel,
         * return the bounding box of pixels that must be accessed on the image to be convolved.
         * Thus the box shifted by -kernel.getCtr() and its size is expanded by kernel.getDimensions()-1.
         *
         * @return the bbox expanded by the kernel. 
         */
        lsst::afw::geom::Box2I growBBox(lsst::afw::geom::Box2I const &bbox) const;
        
        /**
         * Given a bounding box for an image one wishes to convolve with this kernel,
         * return the bounding box for the region of pixels that can be computed.
         * Thus the box shifted by kernel.getCtr() and its size is reduced by kernel.getDimensions()-1.
         *
         * @return the bbox shrunk by the kernel.
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the resulting box would have
         * dimension < 1 in either axis
         */
        lsst::afw::geom::Box2I shrinkBBox(lsst::afw::geom::Box2I const &bbox) const;

        /**
         * @brief Set index of kernel's center
         */
        inline void setCtr(lsst::afw::geom::Point2I ctr) {
            _ctrX = ctr.getX();
            _ctrY = ctr.getY();
            _setKernelXY();
        }

        /**
         * @brief Set x index of kernel's center
         *
         * @deprecated Use setCtr instead
         */
        inline void setCtrX(int ctrX) {
            _ctrX = ctrX;
            _setKernelXY();
        }

        /**
         * @brief Set y index of kernel's center
         *
         * @deprecated Use setCtr instead
         */
        inline void setCtrY(int ctrY) {
            _ctrY = ctrY;
            _setKernelXY();
        }

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
        }

        /**
         * @brief Return true iff the kernel is spatially varying (has a spatial function)
         */
        inline bool isSpatiallyVarying() const {
            return _spatialFunctionList.size() != 0;
        }

        /**
         * @brief Set the kernel parameters of a spatially invariant kernel.
         *
         * @throw lsst::pex::exceptions::RuntimeError if the kernel has a spatial function
         * @throw lsst::pex::exceptions::InvalidParameterError if the params vector is the wrong length
         */
        inline void setKernelParameters(std::vector<double> const &params) {
            if (this->isSpatiallyVarying()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                    "Kernel is spatially varying");
            }
            const unsigned int nParams = this->getNKernelParameters();
            if (nParams != params.size()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                                  (boost::format("Number of parameters is wrong, saw %d expected %d") %
                                   nParams % params.size()).str());
            }
            for (unsigned int ii = 0; ii < nParams; ++ii) {
                this->setKernelParameter(ii, params[ii]);
            }
        }

        /**
         * @brief Set the kernel parameters of a 2-component spatially invariant kernel.
         *
         * @warning This is a low-level method intended for maximum efficiency when using warping kernels.
         * No error checking is performed. Use the std::vector<double> form if you want safety.
         */
        inline void setKernelParameters(std::pair<double, double> const& params) {
            this->setKernelParameter(0, params.first);
            this->setKernelParameter(1, params.second);
        }

        /**
         * @brief Set the parameters of all spatial functions
         *
         * Params is indexed as [kernel parameter][spatial parameter]
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if params is the wrong shape
         *  (if this exception is thrown then no parameters are changed)
         */
        void setSpatialParameters(const std::vector<std::vector<double> > params);

        /**
         * @brief Compute the kernel parameters at a specified point
         *
         * Warning: this is a low-level function that assumes kernelParams is the right length.
         * It will fail in unpredictable ways if that condition is not met.
         */
        void computeKernelParametersFromSpatialModel(
            std::vector<double> &kernelParams, double x, double y) const;

        /**
         * @brief Return a string representation of the kernel
         */
        virtual std::string toString(std::string const& prefix="") const;

        /**
         * @brief Compute a cache of Kernel values, if desired
         *
         * @warning: few kernel classes actually support this,
         * in which case this is a no-op and getCacheSize always returns 0.
         */
        virtual void computeCache(
            int const   ///< desired cache size
        ) {}
        
        /**
         * @brief Get the current size of the kernel cache (0 if none or if caches not supported)
         */
        virtual int getCacheSize() const { return 0; };
        
#if 0                                   // fails to compile with icc; is it actually used?
        virtual void toFile(std::string fileName) const;
#endif

        struct PersistenceHelper;

    protected:

        virtual std::string getPythonModule() const;

        /**
         * @brief Set one kernel parameter
         *
         * Classes that have kernel parameters must subclass this function.
         *
         * This function is marked "const", despite modifying unimportant internals,
         * so that computeImage can be const.
         *
         * @throw lsst::pex::exceptions::InvalidParameterError always (unless subclassed)
         */
        virtual void setKernelParameter(unsigned int ind, double value) const;

        /**
         * @brief Set the kernel parameters from the spatial model (if any).
         *
         * This function has no effect if there is no spatial model.
         *
         * This function is marked "const", despite modifying unimportant internals,
         * so that computeImage can be const.
         */
        void setKernelParametersFromSpatialModel(double x, double y) const;
        
        /**
         * @brief Low-level version of computeImage
         *
         * Before this is called the image dimensions are checked, the image's xy0 is set
         * and the kernel's parameters are set.
         * This routine sets the pixels, including normalization if requested.
         *
         * @return The kernel sum
         */
        virtual double doComputeImage(
            lsst::afw::image::Image<Pixel> &image,   ///< image whose pixels are to be set (output)
            bool doNormalize    ///< normalize the image (so sum is 1)?
        ) const = 0;

        std::vector<SpatialFunctionPtr> _spatialFunctionList;

    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::KernelFormatter)

        int _width;
        int _height;
        int _ctrX;
        int _ctrY;
        unsigned int _nKernelParams;
        
        // prevent copying and assignment (to avoid problems from type slicing)
        Kernel(const Kernel&);
        Kernel& operator=(const Kernel&);
        // Set the Kernel's ideas about the x- and y- coordinates
        virtual void _setKernelXY() {}
    };

    typedef std::vector<PTR(Kernel)> KernelList;

    /**
     * @brief A kernel created from an Image
     *
     * It has no adjustable parameters and so cannot be spatially varying.
     *
     * @ingroup afw
     */
    class FixedKernel : public afw::table::io::PersistableFacade<FixedKernel>, public Kernel {
    public:
        typedef PTR(FixedKernel) Ptr;
        typedef CONST_PTR(FixedKernel) ConstPtr;

        /**
         * @brief Construct an empty FixedKernel of size 0x0
         */
        explicit FixedKernel();

        /**
         * @brief Construct a FixedKernel from an image
         */
        explicit FixedKernel(
            lsst::afw::image::Image<Pixel> const &image     ///< image for kernel
        );

        /**
         * @brief Construct a FixedKernel from a generic Kernel
         */
        explicit FixedKernel(
            lsst::afw::math::Kernel const& kernel,      ///< Kernel to convert to Fixed
            lsst::afw::geom::Point2D const& pos         ///< desired position 
        );

        virtual ~FixedKernel() {}

        virtual PTR(Kernel) clone() const;

        virtual std::string toString(std::string const& prefix = "") const;

        virtual Pixel getSum() const {
            return _sum;
        }

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:
        double doComputeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize
        ) const;

        virtual std::string getPersistenceName() const;

        virtual void write(OutputArchiveHandle & handle) const;

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
            }
    };


    /**
     * @brief A kernel described by a function.
     *
     * The function's x, y arguments are as follows:
     * * -getCtr() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getDimensions() - 1) - getCtr() for the upper right pixel
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * @ingroup afw
     */
    class AnalyticKernel : public afw::table::io::PersistableFacade<AnalyticKernel>, public Kernel {
    public:
        typedef PTR(AnalyticKernel) Ptr;
        typedef CONST_PTR(AnalyticKernel) ConstPtr;
        typedef lsst::afw::math::Function2<Pixel> KernelFunction;
        typedef PTR(lsst::afw::math::Function2<Pixel>) KernelFunctionPtr;

        /**
         * @brief Construct an empty spatially invariant AnalyticKernel of size 0x0
         */
        explicit AnalyticKernel();

        /**
         * @brief Construct a spatially invariant AnalyticKernel,
         * or a spatially varying AnalyticKernel where the spatial model
         * is described by one function (that is cloned to give one per analytic function parameter).
         */
        explicit AnalyticKernel(
            int width,  ///< width of kernel
            int height, ///< height of kernel
            KernelFunction const &kernelFunction,   ///< kernel function; a deep copy is made
            Kernel::SpatialFunction const &spatialFunction=NullSpatialFunction()  ///< spatial function;
                ///< one deep copy is made for each kernel function parameter;
                ///< if omitted or set to Kernel::NullSpatialFunction() then the kernel is spatially invariant
        );

        /**
         * @brief Construct a spatially varying AnalyticKernel, where the spatial model
         * is described by a list of functions (one per analytic function parameter).
         *
         * @throw lsst::pex::exceptions::InvalidParameterError
         *        if the length of spatialFunctionList != # kernel function parameters.
         */
        explicit AnalyticKernel(
            int width,  ///< width of kernel
            int height, ///< height of kernel
            KernelFunction const &kernelFunction,   ///< kernel function; a deep copy is made
            std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList ///< list of spatial functions,
                    ///< one per kernel function parameter; a deep copy is made of each function
        );

        virtual ~AnalyticKernel() {}

        virtual PTR(Kernel) clone() const;

        
        /**
         * @brief Compute an image (pixellized representation of the kernel) in place
         *
         * This special version accepts any size image (though you can get in trouble
         * if the image is large enough that the image is evaluated outside its domain).
         *
         * x, y are ignored if there is no spatial function.
         *
         * @return The kernel sum
         *
         * @note computeNewImage has been retired; it doesn't need to be a member
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the image is the wrong size
         * @throw lsst::pex::exceptions::OverflowError if doNormalize is true and the kernel sum is
         * exactly 0
         */
        double computeImage(
            lsst::afw::image::Image<Pixel> &image,   ///< image whose pixels are to be set (output)
                ///< xy0 of the image will be set to -kernel.getCtr() - border,
                ///< where border = (image.getDimensions() - kernel.getDimensions()) / 2
            bool doNormalize,   ///< normalize the image (so sum is 1)?
            double x = 0.0, ///< x (column position) at which to compute spatial function
            double y = 0.0  ///< y (row position) at which to compute spatial function
        ) const;

        virtual std::vector<double> getKernelParameters() const;

        /**
         * @brief Get a deep copy of the kernel function
         */
        virtual KernelFunctionPtr getKernelFunction() const;

        virtual std::string toString(std::string const& prefix="") const;

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:
        virtual double doComputeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize
        ) const;

        virtual std::string getPersistenceName() const;

        virtual void write(OutputArchiveHandle & handle) const;

    protected:
        virtual void setKernelParameter(unsigned int ind, double value) const;

        KernelFunctionPtr _kernelFunctionPtr;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k",
                        boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("fn", _kernelFunctionPtr);
            }
    };


    /**
     * @brief A kernel that has only one non-zero pixel (of value 1)
     *
     * It has no adjustable parameters and so cannot be spatially varying.
     *
     * @ingroup afw
     */
    class DeltaFunctionKernel : public afw::table::io::PersistableFacade<DeltaFunctionKernel>,
                                public Kernel
    {
    public:
        typedef PTR(DeltaFunctionKernel) Ptr;
        typedef CONST_PTR(DeltaFunctionKernel) ConstPtr;
        // Traits values for this class of Kernel
        typedef deltafunction_kernel_tag kernel_fill_factor;

        /**
         * @brief Construct a spatially invariant DeltaFunctionKernel
         *
         * @throw pexExcept::InvalidParameterError if active pixel is off the kernel
         */
        explicit DeltaFunctionKernel(
            int width,              ///< kernel size (columns)
            int height,             ///< kernel size (rows)
            lsst::afw::geom::Point2I const &point   ///< index of active pixel (where 0,0 is the lower left corner)
        );

        virtual ~DeltaFunctionKernel() {}

        virtual PTR(Kernel) clone() const;

        lsst::afw::geom::Point2I getPixel() const { return _pixel; }

        virtual std::string toString(std::string const& prefix="") const;

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:
        virtual double doComputeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize
        ) const;

        virtual std::string getPersistenceName() const;

        virtual void write(OutputArchiveHandle & handle) const;

    private:
        lsst::afw::geom::Point2I _pixel;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
            boost::serialization::void_cast_register<
                DeltaFunctionKernel, Kernel>(
                    static_cast<DeltaFunctionKernel*>(0),
                    static_cast<Kernel*>(0));
        }
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
    class LinearCombinationKernel : public afw::table::io::PersistableFacade<LinearCombinationKernel>,
                                    public Kernel
    {
    public:
        typedef PTR(LinearCombinationKernel) Ptr;
        typedef CONST_PTR(LinearCombinationKernel) ConstPtr;

        /**
         * @brief Construct an empty LinearCombinationKernel of size 0x0
         */
        explicit LinearCombinationKernel();

        /**
         * @brief Construct a spatially invariant LinearCombinationKernel
         */
        explicit LinearCombinationKernel(
            KernelList const &kernelList,    ///< list of (shared pointers to const) basis kernels
            std::vector<double> const &kernelParameters ///< kernel coefficients
        );

        /**
         * @brief Construct a spatially varying LinearCombinationKernel, where the spatial model
         * is described by one function (that is cloned to give one per basis kernel).
         */
        explicit LinearCombinationKernel(
            KernelList const &kernelList,    ///< list of (shared pointers to const) basis kernels
            Kernel::SpatialFunction const &spatialFunction  ///< spatial function;
                ///< one deep copy is made for each basis kernel
        );

        /**
         * @brief Construct a spatially varying LinearCombinationKernel, where the spatial model
         * is described by a list of functions (one per basis kernel).
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the length of spatialFunctionList != # kernels
         */
        explicit LinearCombinationKernel(
            KernelList const &kernelList,    ///< list of (shared pointers to const) kernels
            std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList
                ///< list of spatial functions, one per basis kernel
        );

        virtual ~LinearCombinationKernel() {}

        virtual PTR(Kernel) clone() const;

        virtual std::vector<double> getKernelParameters() const;

        /**
         * @brief Get the fixed basis kernels
         */
        virtual KernelList const &getKernelList() const;

        /**
        * @brief Get the sum of the pixels of each fixed basis kernel
        */
        std::vector<double> getKernelSumList() const;
        
        /**
         * @brief Get the number of basis kernels
         */
        int getNBasisKernels() const { return static_cast<int>(_kernelList.size()); };

        /**
         * @brief Check that all kernels have the same size and center and that none are spatially varying
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if the check fails
         */
        void checkKernelList(const KernelList &kernelList) const;
        
        /**
         * Return true if all basis kernels are instances of DeltaFunctionKernel
         */
        bool isDeltaFunctionBasis() const { return _isDeltaFunctionBasis; };
        
        /**
         * @brief Refactor the kernel as a linear combination of N bases where N is the number of parameters
         * for the spatial model.
         *
         * Refactoring is only possible if all of the following are true:
         *  * Kernel is spatially varying
         *  * The spatial functions are a linear combination of coefficients (return isLinearCombination() true).
         *  * The spatial functions all are the same class (and so have the same functional form)
         * Refactoring produces a kernel that is faster to compute only if the number of basis kernels
         * is greater than the number of parameters in the spatial model.
         *
         * Details:
         * A spatially varying LinearCombinationKernel consisting of M basis kernels
         * and using a spatial model that is a linear combination of N coefficients can be expressed as:
         * K(x,y) =   K0 (C00 F0(x,y) + C10 F1(x,y) + C20 F2(x,y) + ... + CN0 FN(x,y))
         *          + K1 (C01 F0(x,y) + C11 F1(x,y) + C21 F2(x,y) + ... + CN1 FN(x,y))
         *          + K2 (C02 F0(x,y) + C12 F1(x,y) + C22 F2(x,y) + ... + CN2 FN(x,y))
         *          + ...
         *          + KM (C0M F0(x,y) + C1M F1(x,y) + C2M F2(x,y) + ... + CNM FN(x,y))
         *
         * This is equivalent to the following linear combination of N basis kernels:
         *
         *         =      K0' F0(x,y) + K1' F1(x,y) + K2' F2(x,y) + ... + KN' FN(x,y)
         *
         *           where Ki' = sum over j of Kj Cij
         *
         * This is what refactor returns provided the required conditions are met. However, the spatial functions
         * for the refactored kernel are the same as those for the original kernel (for generality and simplicity)
         * with all coefficients equal to 0 except one that is set to 1; hence they are not computed optimally.
         *
         * Thanks to Kresimir Cosic for inventing or reinventing this useful technique.
         *
         * @return a shared pointer to new kernel, or empty pointer if refactoring not possible
         */
        PTR(Kernel) refactor() const;

        virtual std::string toString(std::string const& prefix="") const;

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:
        virtual double doComputeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize
        ) const;

        virtual std::string getPersistenceName() const;

        virtual void write(OutputArchiveHandle & handle) const;

        virtual void setKernelParameter(unsigned int ind, double value) const;

    private:
        /**
         * @brief Set _kernelList by cloning each input kernel and update the kernel image cache.
         */
        void _setKernelList(KernelList const &kernelList);
        
        KernelList _kernelList; ///< basis kernels
        std::vector<PTR(lsst::afw::image::Image<Pixel>)> _kernelImagePtrList;
            ///< image of each basis kernel (a cache)
        std::vector<double> _kernelSumList; ///< sum of each basis kernel (a cache)
        mutable std::vector<double> _kernelParams;
        bool _isDeltaFunctionBasis;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k", boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("klist", _kernelList);
                ar & make_nvp("kimglist", _kernelImagePtrList);
                ar & make_nvp("ksumlist", _kernelSumList);
                ar & make_nvp("params", _kernelParams);
                if (version > 0) {
                    ar & make_nvp("deltaBasis", _isDeltaFunctionBasis);
                }
                else if (Archive::is_loading::value) {
                    _isDeltaFunctionBasis = false;
                }
            }
    };

    /**
     * @brief A kernel described by a pair of functions: func(x, y) = colFunc(x) * rowFunc(y)
     *
     * The function's x, y arguments are as follows:
     * * -getCtr() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getDimensions() - 1) - getCtr() for the upper right pixel
     *
     * Note: each pixel is set to the value of the kernel function at the center of the pixel
     * (rather than averaging the function over the area of the pixel).
     *
     * @ingroup afw
     */
    class SeparableKernel : public afw::table::io::PersistableFacade<SeparableKernel>, public Kernel {
    public:
        typedef PTR(SeparableKernel) Ptr;
        typedef CONST_PTR(SeparableKernel) ConstPtr;
        typedef lsst::afw::math::Function1<Pixel> KernelFunction;
        typedef PTR(KernelFunction) KernelFunctionPtr;

        /**
         * @brief Construct an empty spatially invariant SeparableKernel of size 0x0
         */
        explicit SeparableKernel();

        /**
         * @brief Construct a spatially invariant SeparableKernel, or a spatially varying SeparableKernel
         * that uses the same functional form to model each function parameter.
         */
        explicit SeparableKernel(
            int width,  ///< width of kernel
            int height, ///< height of kernel
            KernelFunction const& kernelColFunction,    ///< kernel column function
            KernelFunction const& kernelRowFunction,    ///< kernel row function
            Kernel::SpatialFunction const& spatialFunction=NullSpatialFunction()    ///< spatial function;
                ///< one deep copy is made for each kernel column and row function parameter;
                ///< if omitted or set to Kernel::NullSpatialFunction then the kernel is spatially invariant
        );

        /**
         * @brief Construct a spatially varying SeparableKernel
         *
         * @throw lsst::pex::exceptions::InvalidParameterError
         *  if the length of spatialFunctionList != # kernel function parameters.
         */
        explicit SeparableKernel(
            int width,  ///< width of kernel
            int height, ///< height of kernel
             KernelFunction const& kernelColFunction,    ///< kernel column function
             KernelFunction const& kernelRowFunction,    ///< kernel row function
             std::vector<Kernel::SpatialFunctionPtr> const& spatialFunctionList ///< list of spatial funcs,
                ///< one per kernel column and row function parameter; a deep copy is made of each function
        );
        virtual ~SeparableKernel() {}

        virtual PTR(Kernel) clone() const;

        /**
         * @brief Compute the column and row arrays in place, where kernel(col, row) = colList(col) * rowList(row)
         *
         * x, y are ignored if there is no spatial function.
         *
         * @return the kernel sum (1.0 if doNormalize true)
         *
         * @throw lsst::pex::exceptions::InvalidParameterError if colList or rowList is the wrong size
         * @throw lsst::pex::exceptions::OverflowError if doNormalize is true and the kernel sum is
         * exactly 0
         */
        double computeVectors(
            std::vector<Pixel> &colList,    ///< column vector
            std::vector<Pixel> &rowList,    ///< row vector
            bool doNormalize,   ///< normalize the image (so sum of each is 1)?
            double x = 0.0,     ///< x (column position) at which to compute spatial function
            double y = 0.0      ///< y (row position) at which to compute spatial function
        ) const;

        virtual double getKernelParameter(unsigned int i) const {
            unsigned int const ncol = _kernelColFunctionPtr->getNParameters();
            if (i < ncol) {
                return _kernelColFunctionPtr->getParameter(i);
            } else {
                i -= ncol;
                return _kernelRowFunctionPtr->getParameter(i);
            }
        }
        virtual std::vector<double> getKernelParameters() const;

        /**
         * @brief Get a deep copy of the col kernel function
         */
        KernelFunctionPtr getKernelColFunction() const;

        /**
         * @brief Get a deep copy of the row kernel function
         */
        KernelFunctionPtr getKernelRowFunction() const;

        virtual std::string toString(std::string const& prefix="") const;

        /***
         * @brief Compute a cache of values for the x and y kernel functions
         *
         * A value of 0 disables the cache for maximum accuracy.
         * 10,000 typically results in a warping error of a fraction of a count.
         * 100,000 typically results in a warping error of less than 0.01 count.
         */
        virtual void computeCache(
            int const cacheSize ///< cache size (number of double precision array elements in the x and y caches)
        );
        
        /**
         * @brief Get the current cache size (0 if none)
         */
        virtual int getCacheSize() const;

    protected:
        virtual double doComputeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize
        ) const;

        virtual void setKernelParameter(unsigned int ind, double value) const;

    private:
        /**
         * @brief Compute the column and row arrays in place, where kernel(col, row) = colList(col) * rowList(row)
         *
         * @return the kernel sum (1.0 if doNormalize true)
         *
         * Warning: the length of colList and rowList are not verified!
         *
         * @throw lsst::pex::exceptions::OverflowError if doNormalize is true and the kernel sum is
         * exactly 0
         */
        double basicComputeVectors(
            std::vector<Pixel> &colList,    ///< column vector
            std::vector<Pixel> &rowList,    ///< row vector
            bool doNormalize                ///< normalize the arrays (so sum of each is 1)?
        ) const;

        KernelFunctionPtr _kernelColFunctionPtr;
        KernelFunctionPtr _kernelRowFunctionPtr;
        mutable std::vector<Pixel> _localColList;  // used by doComputeImage
        mutable std::vector<Pixel> _localRowList;
        mutable std::vector<double> _kernelX; // used by SeparableKernel::basicComputeVectors
        mutable std::vector<double> _kernelY;
        //
        // Cached values of the row- and column- kernels
        //
        mutable std::vector<std::vector<double> > _kernelRowCache;
        mutable std::vector<std::vector<double> > _kernelColCache;

        friend class boost::serialization::access;
        template <class Archive>
            void serialize(Archive& ar, unsigned int const version) {
                ar & make_nvp("k",
                    boost::serialization::base_object<Kernel>(*this));
                ar & make_nvp("colfn", _kernelColFunctionPtr);
                ar & make_nvp("rowfn", _kernelRowFunctionPtr);
                ar & make_nvp("cols", _localColList);
                ar & make_nvp("rows", _localRowList);
                ar & make_nvp("kernelX", _kernelX);
                ar & make_nvp("kernelY", _kernelY);
            }

        virtual void _setKernelXY() {
            lsst::afw::geom::Extent2I const dim = getDimensions();
            lsst::afw::geom::Point2I const ctr = getCtr();

            assert (dim[0] == static_cast<int>(_kernelX.size()));
            for (int i = 0; i != dim.getX(); ++i) {
                _kernelX[i] = i - ctr.getX();
            }

            assert (dim[1] == static_cast<int>(_kernelY.size()));
            for (int i = 0; i != dim.getY(); ++i) {
                _kernelY[i] = i - ctr.getY();
            }
        }
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
}

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
        width, height, lsst::afw::geom::Point2I(x, y));
}

}}

#ifndef SWIG
BOOST_CLASS_VERSION(lsst::afw::math::LinearCombinationKernel, 1)
#endif

#endif // !defined(LSST_AFW_MATH_KERNEL_H)
