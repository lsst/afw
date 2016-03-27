// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#ifndef LSST_AFW_DETECTION_GaussianPsf_h_INCLUDED
#define LSST_AFW_DETECTION_GaussianPsf_h_INCLUDED

#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/detection/Psf.h"

namespace lsst { namespace afw { namespace detection {

/**
 *  @brief A circularly symmetric Gaussian Psf class with no spatial variation, intended mostly for
 *         testing purposes.
 *
 *  This class is essentially an alternate implementation of meas::algorithms::SingleGaussianPsf;
 *  While SingleGaussianPsf inherits from ImagePsf and KernelPsf, and hence delegates to those
 *  various operations relating to the PSF model image (e.g. computeShape()), GaussianPsf computes
 *  these analytically.
 */
class GaussianPsf : public afw::table::io::PersistableFacade<GaussianPsf>, public Psf {
public:

    /**
     *  @brief Constructor for a GaussianPsf
     *
     *  @param[in] width   Number of columns in realizations of the PSF at a point.
     *  @param[in] height  Number of rows in realizations of the PSF at a point.
     *  @param[in] sigma   Radius of the Gaussian.
     */
    GaussianPsf(int width, int height, double sigma);

    /**
     *  @brief Constructor for a GaussianPsf
     *
     *  @param[in] dimensions     Number of columns, rows in realizations of the PSF at a point.
     *  @param[in] sigma   Radius of the Gaussian.
     */
    GaussianPsf(geom::Extent2I const & dimensions, double sigma);

    /// Polymorphic deep copy; should usually be unnecessary because Psfs are immutable.
    virtual PTR(afw::detection::Psf) clone() const;

    /// Return the dimensions of the images returned by computeImage()
    geom::Extent2I getDimensions() const { return _dimensions; }

    /// Return the radius of the Gaussian.
    double getSigma() const { return _sigma; }

    /// Whether the Psf is persistable; always true.
    virtual bool isPersistable() const { return true; }

protected:

    virtual std::string getPersistenceName() const;

    virtual std::string getPythonModule() const;

    virtual void write(OutputArchiveHandle & handle) const;

private:

#if 0 // We could reimplement this more efficiently than what's in the base class,
      // but it's tricky to get the position right in all corner cases, and it's
      // not actually performance-critical, so we should just wait for #3116.
    virtual PTR(Image) doComputeImage(
        geom::Point2D const & position, image::Color const& color
    ) const;
#endif

    virtual PTR(Image) doComputeKernelImage(
        geom::Point2D const & position, image::Color const & color
    ) const;

    virtual double doComputeApertureFlux(
        double radius, geom::Point2D const & position, image::Color const & color
    ) const;

    virtual geom::ellipses::Quadrupole doComputeShape(
        geom::Point2D const & position, image::Color const & color
    ) const;

    geom::Extent2I _dimensions;
    double _sigma;
};

}}} // namespace lsst::afw::detection

#endif // !LSST_AFW_DETECTION_Psf_h_INCLUDED
