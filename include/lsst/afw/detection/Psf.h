// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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
#ifndef LSST_AFW_DETECTION_Psf_h_INCLUDED
#define LSST_AFW_DETECTION_Psf_h_INCLUDED

#include <string>
#include <typeinfo>

#include "boost/shared_ptr.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw {

namespace cameraGeom {
class Detector;
} // namespace cameraGeom

namespace detection {

class PsfFormatter;

/// A polymorphic base class for representing an image's Point Spread Function
class Psf : public daf::base::Citizen, public daf::base::Persistable,
            public afw::table::io::PersistableFacade<Psf>, public afw::table::io::Persistable
{
public:
    typedef boost::shared_ptr<Psf> Ptr;            ///< @deprecated shared_ptr to a Psf
    typedef boost::shared_ptr<const Psf> ConstPtr; ///< @deprecated shared_ptr to a const Psf

    typedef math::Kernel::Pixel Pixel; ///< Pixel type of Image returned by computeImage
    typedef image::Image<Pixel> Image; ///< Image type returned by computeImage

    virtual ~Psf() {}

    /// Polymorphic deep-copy.
    virtual PTR(Psf) clone() const = 0;

    // accessors for distortion
    void setDetector(PTR(lsst::afw::cameraGeom::Detector) det) {
        _detector = det;
    }
    PTR(cameraGeom::Detector) getDetector() {
        return _detector;
    }
    PTR(cameraGeom::Detector const) getDetector() const {
        return _detector;
    }

    //@{
    /**
     *  @brief Return an Image of the PSF
     *
     * The specified position is a floating point number, and the resulting image will
     * have a Psf with the correct fractional position, with the centre within pixel (width/2, height/2)
     * Specifically, fractional positions in [0, 0.5] will appear above/to the right of the center,
     * and fractional positions in (0.5, 1] will appear below/to the left (0.9999 is almost back at middle)
     *
     * The image's (X0, Y0) will be set correctly to reflect this 
     *
     * @note If a fractional position is specified, the calculated central pixel value may be less than 1.
     *  Evaluates the PSF at the specified point and [optional] color
     *
     *  @note The real work is done in the virtual function, Psf::doComputeImage
     */
    PTR(Image) computeImage(
        geom::Point2D const& ccdXY=geom::Point2D(),
        bool normalizePeak=true, bool distort=true
    ) const;

    PTR(Image) computeImage(
        image::Color const& color,
        geom::Point2D const& ccdXY=geom::Point2D(0, 0),
        bool normalizePeak=true, bool distort=true
    ) const;
    //@}

    PTR(math::Kernel const) getKernel(image::Color const& color=image::Color()) const {
        return doGetKernel(color);
    }

    PTR(math::Kernel const) getLocalKernel(
        geom::Point2D const& ccdXY=geom::Point2D(0, 0),
        image::Color const& color=image::Color()
    ) const {
        return doGetLocalKernel(ccdXY, color);
    }
    /**
     * Return the average Color of the stars used to construct the Psf
     *
     * \note this the Color used to return a Psf if you don't specify a Color
     */
    image::Color getAverageColor() const {
        return image::Color();
    }

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
    static PTR(Image) recenterKernelImage(
        PTR(Image) im, geom::Point2I const &ctr,
        geom::Point2D const &xy,
        std::string const &warpAlgorithm = "lanczos5",
        unsigned int warpBuffer = 5
    );

protected:

    Psf() : daf::base::Citizen(typeid(this)), _detector() {}

    PTR(cameraGeom::Detector) _detector;

    virtual std::string getPythonModule() const;

    virtual PTR(Image) doComputeImage(
        image::Color const& color,
        geom::Point2D const& ccdXY,
        bool normalizePeak,
        bool distort
    ) const;

    virtual PTR(math::Kernel const) doGetKernel(image::Color const&) const {
        return PTR(math::Kernel const)();
    }

    virtual PTR(math::Kernel const) doGetLocalKernel(geom::Point2D const&, image::Color const&) const {
        return PTR(math::Kernel const)();
    }

private:
    LSST_PERSIST_FORMATTER(PsfFormatter)
};

/**
 * A Psf built from a Kernel
 */
class KernelPsf : public afw::table::io::PersistableFacade<KernelPsf>, public Psf {
public:
    KernelPsf(PTR(math::Kernel) kernel=PTR(math::Kernel)()) : Psf(), _kernel(kernel) {}

protected:
    /**
     * Return the Psf's kernel
     */
    virtual PTR(math::Kernel const)
    doGetKernel(image::Color const&) const {
        return PTR(math::Kernel const)(_kernel);
    }
    /**
     * Return the Psf's kernel instantiated at a point
     */
    virtual PTR(math::Kernel const) doGetLocalKernel(geom::Point2D const& pos, image::Color const&) const {
        return boost::make_shared<math::FixedKernel>(*_kernel, pos);
    }

    /// Clone a KernelPsf
    virtual Ptr clone() const { return boost::make_shared<KernelPsf>(*this); }

    /// Whether this object is persistable; just delegates to the kernel.
    virtual bool isPersistable() const { return _kernel->isPersistable(); }

protected:

    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle & handle) const;

    void setKernel(PTR(math::Kernel) kernel) { _kernel = kernel; }

private:
    PTR(math::Kernel) _kernel;
};

}}} // namespace lsst::afw::detection

#endif // !LSST_AFW_DETECTION_Psf_h_INCLUDED
