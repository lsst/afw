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
#include "lsst/afw/math.h"
#include "lsst/afw/image/Color.h"

namespace lsst {
namespace afw {
namespace detection {

class LocalPsf;
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
class Psf : public lsst::daf::base::Citizen, public lsst::daf::base::Persistable {
public:
    typedef boost::shared_ptr<Psf> Ptr;            ///< shared_ptr to a Psf
    typedef boost::shared_ptr<const Psf> ConstPtr; ///< shared_ptr to a const Psf

    typedef lsst::afw::math::Kernel::Pixel Pixel; ///< Pixel type of Image returned by computeImage
    typedef lsst::afw::image::Image<Pixel> Image; ///< Image type returned by computeImage

    /// ctor
    Psf() : lsst::daf::base::Citizen(typeid(this)) {}
    virtual ~Psf() {}

    virtual Ptr clone() const = 0;

    /// Return true iff Psf is valid
    operator bool() const { return getKernel().get() != NULL; }

    PTR(Image) computeImage(lsst::afw::geom::Extent2I const& size, bool normalizePeak=true) const;

    PTR(Image) computeImage(lsst::afw::geom::Point2D const& ccdXY, bool normalizePeak) const;

    PTR(Image) computeImage(lsst::afw::geom::Point2D const& ccdXY=lsst::afw::geom::Point2D(0, 0),
                            lsst::afw::geom::Extent2I const& size=lsst::afw::geom::Extent2I(0, 0),
                            bool normalizePeak=true) const;

    PTR(Image) computeImage(lsst::afw::image::Color const& color,
                            lsst::afw::geom::Point2D const& ccdXY=lsst::afw::geom::Point2D(0, 0),
                            lsst::afw::geom::Extent2I const& size=lsst::afw::geom::Extent2I(0, 0),
                            bool normalizePeak=true) const;
   
    PTR(LocalPsf) getLocalPsf(
        lsst::afw::geom::Point2D const & ccdXY,
        lsst::afw::image::Color const & color=lsst::afw::image::Color()
    ) const {
        return doGetLocalPsf(ccdXY, color);
    }

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
protected:
    virtual Image::Ptr doComputeImage(lsst::afw::image::Color const& color,
                                      lsst::afw::geom::Point2D const& ccdXY,
                                      lsst::afw::geom::Extent2I const& size,
                                      bool normalizePeak) const;

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

    virtual PTR(LocalPsf) doGetLocalPsf(
        lsst::afw::geom::Point2D const&, 
        lsst::afw::image::Color const&
    ) const {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Functionality not implemented"
        );
    };

        
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
class KernelPsf : public Psf {
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
    virtual lsst::afw::math::Kernel::Ptr doGetLocalKernel(lsst::afw::geom::Point2D const&,
                                                          lsst::afw::image::Color const&) {
        return _kernel;
    }
    /**
     * Return the Psf's kernel instantiated at a point
     */
    virtual lsst::afw::math::Kernel::ConstPtr doGetLocalKernel(lsst::afw::geom::Point2D const&,
                                                               lsst::afw::image::Color const&) const {
        return lsst::afw::math::Kernel::ConstPtr(_kernel);
    }

    /// Clone a KernelPsf
    virtual Ptr clone() const { return boost::make_shared<KernelPsf>(*this); }

protected:
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
