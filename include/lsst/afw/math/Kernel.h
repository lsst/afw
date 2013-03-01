// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
#include <utility>
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

        explicit Kernel();

        explicit Kernel(int width, int height, unsigned int nKernelParams,
                        SpatialFunction const &spatialFunction=NullSpatialFunction());
        explicit Kernel(int width, int height, const std::vector<SpatialFunctionPtr> spatialFunctionList);

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
         * @throw lsst::pex::exceptions::InvalidParameterException if the image is the wrong size
         * @throw lsst::pex::exceptions::OverflowErrorException if doNormalize is true and the kernel sum is
         * exactly 0
         */
        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,   ///< image whose pixels are to be set (output)
            bool doNormalize,   ///< normalize the image (so sum is 1)?
            double x = 0.0, ///< x (column position) at which to compute spatial function
            double y = 0.0  ///< y (row position) at which to compute spatial function
        ) const = 0;

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

        SpatialFunctionPtr getSpatialFunction(unsigned int index) const;

        std::vector<SpatialFunctionPtr> getSpatialFunctionList() const;

        /// Return a particular Kernel Parameter (no bounds checking).  This version is slow,
        /// but specialisations may be faster
        virtual double getKernelParameter(unsigned int i) const {
            return getKernelParameters()[i];
        }
        
        virtual std::vector<double> getKernelParameters() const;
        
        lsst::afw::geom::Box2I growBBox(lsst::afw::geom::Box2I const &bbox) const;
        
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

        void setSpatialParameters(const std::vector<std::vector<double> > params);

        void computeKernelParametersFromSpatialModel(
            std::vector<double> &kernelParams, double x, double y) const;

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

        virtual void setKernelParameter(unsigned int ind, double value) const;

        void setKernelParametersFromSpatialModel(double x, double y) const;

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

        explicit FixedKernel();

        explicit FixedKernel(
            lsst::afw::image::Image<Pixel> const &image
        );

        explicit FixedKernel(
            lsst::afw::math::Kernel const& kernel,
            lsst::afw::geom::Point2D const& pos
        );

        virtual ~FixedKernel() {}

        virtual PTR(Kernel) clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual std::string toString(std::string const& prefix = "") const;

        virtual Pixel getSum() const {
            return _sum;
        }

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:

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
     * * -getCtrX(), -getCtrY() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getWidth() - 1) - getCtrX(), (getHeight() - 1) - getCtrY() for the upper right pixel
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

        virtual ~AnalyticKernel() {}

        virtual PTR(Kernel) clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual std::vector<double> getKernelParameters() const;

        virtual KernelFunctionPtr getKernelFunction() const;

        virtual std::string toString(std::string const& prefix="") const;

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:

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

        explicit DeltaFunctionKernel(
            int width,
            int height,
            lsst::afw::geom::Point2I const &point
        );

        virtual ~DeltaFunctionKernel() {}

        virtual PTR(Kernel) clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        lsst::afw::geom::Point2I getPixel() const { return _pixel; }

        virtual std::string toString(std::string const& prefix="") const;

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:

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

        virtual ~LinearCombinationKernel() {}

        virtual PTR(Kernel) clone() const;

        virtual double computeImage(
            lsst::afw::image::Image<Pixel> &image,
            bool doNormalize,
            double x = 0.0,
            double y = 0.0
        ) const;

        virtual std::vector<double> getKernelParameters() const;

        virtual KernelList const &getKernelList() const;

        std::vector<double> getKernelSumList() const;
        
        /**
         * @brief Get the number of basis kernels
         */
        int getNBasisKernels() const { return static_cast<int>(_kernelList.size()); };

        void checkKernelList(const KernelList &kernelList) const;
        
        /**
         * Return true if all basis kernels are instances of DeltaFunctionKernel
         */
        bool isDeltaFunctionBasis() const { return _isDeltaFunctionBasis; };
        
        PTR(Kernel) refactor() const;

        virtual std::string toString(std::string const& prefix="") const;

        virtual bool isPersistable() const { return true; }

        class Factory;

    protected:

        virtual std::string getPersistenceName() const;

        virtual void write(OutputArchiveHandle & handle) const;

        virtual void setKernelParameter(unsigned int ind, double value) const;

    private:
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
     * * -getCtrX(), -getCtrY() for the lower left corner pixel
     * * 0, 0 for the center pixel
     * * (getWidth() - 1) - getCtrX(), (getHeight() - 1) - getCtrY() for the upper right pixel
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
        virtual ~SeparableKernel() {}

        virtual PTR(Kernel) clone() const;

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

        KernelFunctionPtr getKernelColFunction() const;

        KernelFunctionPtr getKernelRowFunction() const;

        virtual std::string toString(std::string const& prefix="") const;

        virtual void computeCache(int const cacheSize);
        
        virtual int getCacheSize() const;

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
            assert (getWidth() == static_cast<int>(_kernelX.size()));
            for (int i = 0; i != getWidth(); ++i) {
                _kernelX[i] = i - getCtrX();
            }

            assert (getHeight() == static_cast<int>(_kernelY.size()));
            for (int i = 0; i != getHeight(); ++i) {
                _kernelY[i] = i - getCtrY();
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
