// -*- LSST-C++ -*-
#if !defined(LSST_AFW_DETECTION_PSF_H)
#define LSST_AFW_DETECTION_PSF_H
//!
// Describe an image's PSF
//
#include <string>
#include <typeinfo>
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace cameraGeom {
    class Detector;
}
namespace detection {

class PsfFormatter;
class PsfFactoryBase;

/**
 * Create a particular sort of Psf.
 *
 * PsfT: the Psf class that we're going to instantiate
 * PsfFactorySignatureT: The signature of the Psf constructor
 *
 * \note We do NOT define the unspecialised type, as only a specific set of signatures are supported.  To add
 * another, you'll have to instantiate PsfFactory with the correct signature, add a suitable virtual member to
 * PsfFactoryBase, and add a matching createPsf function.
 */
template<typename PsfT, typename PsfFactorySignatureT> class PsfFactory;

/*!
 * \brief Represent an image's Point Spread Function
 *
 * \note A polymorphic base class for Psf%s
 */
class Psf : public lsst::daf::base::Citizen, public lsst::daf::base::Persistable,
            public afw::table::io::PersistableFacade<Psf>, public afw::table::io::Persistable
 {
public:
    typedef boost::shared_ptr<Psf> Ptr;            ///< shared_ptr to a Psf
    typedef boost::shared_ptr<const Psf> ConstPtr; ///< shared_ptr to a const Psf

    typedef lsst::afw::math::Kernel::Pixel Pixel; ///< Pixel type of Image returned by computeImage
    typedef lsst::afw::image::Image<Pixel> Image; ///< Image type returned by computeImage

    /// ctor
    Psf() : lsst::daf::base::Citizen(typeid(this)), _detector() {}
    virtual ~Psf() {}

    virtual Ptr clone() const = 0;

    // accessors for distortion
    void setDetector(PTR(lsst::afw::cameraGeom::Detector) det) {
        _detector = det;
    }
    PTR(lsst::afw::cameraGeom::Detector) getDetector() {
        return _detector;
    }
    CONST_PTR(lsst::afw::cameraGeom::Detector) getDetector() const {
        return _detector;
    }

    PTR(Image) computeImage(lsst::afw::geom::Extent2I const& size, bool normalizePeak=true,
                            bool distort=true) const;

    PTR(Image) computeImage(lsst::afw::geom::Point2D const& ccdXY, bool normalizePeak,
                            bool distort=true) const;

    PTR(Image) computeImage(lsst::afw::geom::Point2D const& ccdXY=lsst::afw::geom::Point2D(0, 0),
                            lsst::afw::geom::Extent2I const& size=lsst::afw::geom::Extent2I(0, 0),
                            bool normalizePeak=true,
                            bool distort=true) const;

    PTR(Image) computeImage(lsst::afw::image::Color const& color,
                            lsst::afw::geom::Point2D const& ccdXY=lsst::afw::geom::Point2D(0, 0),
                            lsst::afw::geom::Extent2I const& size=lsst::afw::geom::Extent2I(0, 0),
                            bool normalizePeak=true,
                            bool distort=true) const;

    lsst::afw::math::Kernel::Ptr getKernel(lsst::afw::image::Color const&
                                           color=lsst::afw::image::Color()) {
        return doGetKernel(color);
    }
    lsst::afw::math::Kernel::ConstPtr getKernel(lsst::afw::image::Color const&
                                                color=lsst::afw::image::Color()) const {
        return doGetKernel(color);
    }
    lsst::afw::math::Kernel::Ptr getLocalKernel(
        lsst::afw::geom::Point2D const& ccdXY=lsst::afw::geom::Point2D(0, 0),
        lsst::afw::image::Color const& color=lsst::afw::image::Color()) {
        return doGetLocalKernel(ccdXY, color);
    }
    lsst::afw::math::Kernel::ConstPtr getLocalKernel(
        lsst::afw::geom::Point2D const& ccdXY=lsst::afw::geom::Point2D(0, 0),
        lsst::afw::image::Color const& color=lsst::afw::image::Color()) const {
        return doGetLocalKernel(ccdXY, color);
    }
    /**
     * Return the average Color of the stars used to construct the Psf
     *
     * \note this the Color used to return a Psf if you don't specify a Color
     */
    lsst::afw::image::Color getAverageColor() const {
        return lsst::afw::image::Color();
    }
    /**
     * Register a factory that builds a type of Psf
     *
     * \note This function returns bool so that it can be used in an initialisation
     * at file scope to do the actual registration
     */
    template<typename PsfT, typename PsfFactorySignatureT>
    static bool registerMe(std::string const& name) {
        static bool _registered = false;
        
        if (!_registered) {
            PsfFactory<PsfT, PsfFactorySignatureT> *factory = new PsfFactory<PsfT, PsfFactorySignatureT>();
            factory->markPersistent();
            
            Psf::declare(name, factory);
            _registered = true;
        }

        return true;
    }

    /**
     * Helper function for Psf::computeImage(): takes a kernel image \c src, with central pixel \c ctr 
     * (presumably equal to kernel->getCtr()) and stuffs it into an output image \c dst, which need not 
     * have the same dimensions as \c src.  Returns the central pixel for the output image.
     *
     * The image xy0 fields are ignored, since these are generally not meaningful for the output
     * of Kernel::computeImage() anyway (this is generally true throughout the kernel API).
     */
    static lsst::afw::geom::Point2I resizeKernelImage(Image &dst, const Image &src, 
                                                      const lsst::afw::geom::Point2I &ctr);

    /**
     * Helper function for Psf::computeImage(): converts a kernel image (i.e. xy0 not meaningful; 
     * center given by parameter \c ctr) to a psf image (i.e. xy0 is meaningful)
     *
     * \c warpAlgorithm is passed to afw::math::makeWarpingKernel() and can be "nearest", "bilinear", 
     * or "lanczosN"
     *
     * \c warpBuffer zero-pads the image before recentering.  Recommended value is 1 for bilinear, 
     * N for lanczosN (note that it would be cleaner to infer this value from the warping algorithm
     * but this would require mild API changes; same issue occurs in e.g. afw::math::offsetImage())
     *
     * The point with integer coordinates \c ctr in the source image corresponds to the point
     * \c xy in the destination image.  If \c xy is not integer-valued then we will need to fractionally
     * shift the image using interpolation (lanczos5 currently hardcoded)
     *
     * Note: if fractional recentering is performed, then a new image will be allocated and returned.
     * If not, then the original image will be returned (after setting XY0)
     */
    static PTR(Image) recenterKernelImage(PTR(Image) im, const lsst::afw::geom::Point2I &ctr, 
                                          const lsst::afw::geom::Point2D &xy,
                                          std::string const &warpAlgorithm = "lanczos5", 
                                          unsigned int warpBuffer = 5);

protected:
    PTR(lsst::afw::cameraGeom::Detector) _detector;

    virtual std::string getPythonModule() const;
    
    virtual Image::Ptr doComputeImage(lsst::afw::image::Color const& color,
                                      lsst::afw::geom::Point2D const& ccdXY,
                                      lsst::afw::geom::Extent2I const& size,
                                      bool normalizePeak,
                                      bool distort
                                     ) const;

    virtual lsst::afw::math::Kernel::Ptr doGetKernel(lsst::afw::image::Color const&) {
        return lsst::afw::math::Kernel::Ptr();
    }
        
    virtual lsst::afw::math::Kernel::ConstPtr doGetKernel(lsst::afw::image::Color const&) const {
        return lsst::afw::math::Kernel::Ptr();
    }
        
    virtual lsst::afw::math::Kernel::Ptr doGetLocalKernel(lsst::afw::geom::Point2D const&,
                                                          lsst::afw::image::Color const&) {
        return lsst::afw::math::Kernel::Ptr();
    }
        
    virtual lsst::afw::math::Kernel::ConstPtr doGetLocalKernel(lsst::afw::geom::Point2D const&,
                                                               lsst::afw::image::Color const&) const {
        return lsst::afw::math::Kernel::Ptr();
    }

        
private:
    LSST_PERSIST_FORMATTER(PsfFormatter)
    /*
     * Support for Psf factories
     */
protected:
#if !defined(SWIG)
    friend Psf::Ptr createPsf(std::string const& name,
                              int const width, int const height, double p0, double p1, double p2);
    friend Psf::Ptr createPsf(std::string const& name,
                              lsst::afw::math::Kernel::Ptr kernel);
#endif

    static void declare(std::string name, PsfFactoryBase* factory = NULL);
    static PsfFactoryBase& lookup(std::string name);
private:
    static PsfFactoryBase& _registry(std::string const& name, PsfFactoryBase * factory = NULL);
};

/************************************************************************************************************/
/**
 * A Psf built from a Kernel
 */
class KernelPsf : public afw::table::io::PersistableFacade<KernelPsf>, public Psf {
public:
    KernelPsf(
        lsst::afw::math::Kernel::Ptr kernel=lsst::afw::math::Kernel::Ptr() ///< This PSF's Kernel
             ) : Psf(), _kernel(kernel) {}

protected:
    /**
     * Return the Psf's kernel
     */
    virtual lsst::afw::math::Kernel::Ptr
    doGetKernel(lsst::afw::image::Color const&) {
        return _kernel;
    }
    /**
     * Return the Psf's kernel
     */
    virtual lsst::afw::math::Kernel::ConstPtr
    doGetKernel(lsst::afw::image::Color const&) const {
        return lsst::afw::math::Kernel::ConstPtr(_kernel);
    }
    /**
     * Return the Psf's kernel instantiated at a point
     */
    virtual lsst::afw::math::Kernel::Ptr doGetLocalKernel(lsst::afw::geom::Point2D const& pos,
                                                          lsst::afw::image::Color const&) {
        return boost::make_shared<lsst::afw::math::FixedKernel>(*_kernel, pos);
    }
    /**
     * Return the Psf's kernel instantiated at a point
     */
    virtual lsst::afw::math::Kernel::ConstPtr doGetLocalKernel(lsst::afw::geom::Point2D const& pos,
                                                               lsst::afw::image::Color const&) const {
        return boost::make_shared<lsst::afw::math::FixedKernel>(*_kernel, pos);
    }

    /// Clone a KernelPsf
    virtual Ptr clone() const { return boost::make_shared<KernelPsf>(*this); }

    /// Whether this object is persistable; just delegates to the kernel.
    virtual bool isPersistable() const { return _kernel->isPersistable(); }

protected:

    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle & handle) const;

    void setKernel(lsst::afw::math::Kernel::Ptr kernel) { _kernel = kernel; }
    
private:
    lsst::afw::math::Kernel::Ptr _kernel; // Kernel that corresponds to the Psf
};

/************************************************************************************************************/
/**
 * A polymorphic base class for Psf factories
 */
class PsfFactoryBase : public lsst::daf::base::Citizen {
public:
    PsfFactoryBase() : lsst::daf::base::Citizen(typeid(this)) {}
    virtual ~PsfFactoryBase() {}
    virtual Psf::Ptr create(int = 0, int = 0, double = 0, double = 0, double = 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException,
                          "This Psf type doesn't have an (int, int, double, double, double) constructor");
    };
    virtual Psf::Ptr create(lsst::afw::math::Kernel::Ptr) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException,
                          "This Psf type doesn't have a (lsst::afw::math::Kernel::Ptr) constructor");
    };
};
 
/**
 * Create a particular sort of Psf with signature (int, int, double, double, double)
 */
template<typename PsfT>
class PsfFactory<PsfT, boost::tuple<int, int, double, double, double> > : public PsfFactoryBase {
public:
    /**
     * Return a (shared_ptr to a) new PsfT
     */
    virtual Psf::Ptr create(int width = 0, int height = 0, double p0 = 0, double p1 = 0, double p2 = 0) {
        return typename PsfT::Ptr(new PsfT(width, height, p0, p1, p2));
    }
    /*
     * Call the other (non-implemented) create method to make icc happy
     */
    virtual Psf::Ptr create(lsst::afw::math::Kernel::Ptr ptr) {
        return PsfFactoryBase::create(ptr);
    };

};

/**
 * Create a particular sort of Psf with signature (lsst::afw::math::Kernel::Ptr)
 */
template<typename PsfT>
class PsfFactory<PsfT, lsst::afw::math::Kernel::Ptr> : public PsfFactoryBase {
public:
    /*
     * Call the other (non-implemented) create method to make icc happy
     */
    virtual Psf::Ptr create(int width = 0, int height = 0, double p0 = 0, double p1 = 0, double p2 = 0) {
        return PsfFactoryBase::create(width, height, p0, p1, p2);
    }
    /**
     * Return a (shared_ptr to a) new PsfT
     */
    virtual Psf::Ptr create(lsst::afw::math::Kernel::Ptr kernel) {
        return typename PsfT::Ptr(new PsfT(kernel));
    }
};

/************************************************************************************************************/
/**
 * Factory functions to return a Psf
 */
/**
 * Create a named sort of Psf with signature (int, int, double, double, double)
 */
Psf::Ptr createPsf(std::string const& type, int width = 0, int height = 0,
                   double = 0, double = 0, double = 0);

/**
 * Create a named sort of Psf with signature (lsst::afw::math::Kernel::Ptr)
 */
Psf::Ptr createPsf(std::string const& type, lsst::afw::math::Kernel::Ptr kernel);
}}}
#endif
