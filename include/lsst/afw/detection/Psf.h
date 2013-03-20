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
#include <limits>

#include "boost/shared_ptr.hpp"

#include "lsst/daf/base.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace detection {

class PsfFormatter;

/**
 *  @brief A polymorphic base class for representing an image's Point Spread Function
 *
 *  Most of a Psf's functionality involves its evaluation at a position and color, either
 *  or both of which may be unspecified (which will result in evaluation at some average
 *  position or color).  Unlike the closely-related Kernel class, there is no requirement
 *  that a Psf have a well-defined spatial function or any parameters.  Psfs are not
 *  necessarily continuous, and the dimensions of image of the Psf at a point may not be
 *  fixed.
 *
 *  Psfs are immutable - derived classes should have no non-const methods, and hence
 *  should be fully-defined after construction.  This allows shared_ptrs to Psfs to be
 *  passed around and shared between objects without concern for whether they will be
 *  unexpectedly modified.
 *
 *  In most cases, Psf derived classes should inherit from meas::algorithms::ImagePsf
 *  or meas::algorithms::KernelPsf, as these will provide default implementions for
 *  several member functions.
 */
class Psf : public daf::base::Citizen, public daf::base::Persistable,
            public afw::table::io::PersistableFacade<Psf>, public afw::table::io::Persistable
{
    static geom::Point2D makeNullPoint() {
        return geom::Point2D(std::numeric_limits<double>::quiet_NaN());
    }
public:
    typedef boost::shared_ptr<Psf> Ptr;            ///< @deprecated shared_ptr to a Psf
    typedef boost::shared_ptr<const Psf> ConstPtr; ///< @deprecated shared_ptr to a const Psf

    typedef math::Kernel::Pixel Pixel; ///< Pixel type of Image returned by computeImage
    typedef image::Image<Pixel> Image; ///< Image type returned by computeImage

    /// Enum passed to computeImage and computeKernelImage to determine image ownership.
    enum ImageOwnerEnum {
        COPY=0,     ///< The image will be copied before returning; caller will own it.
        INTERNAL=1  /**< An internal image will be returned without copying.  The caller must not modify
                     *   it, and it may be invalidated the next time a Psf member function is called with
                     *   different color and/or position.
                     */
    };

    virtual ~Psf() {}

    /**
     *  @brief Polymorphic deep-copy.
     *
     *  Because Psfs are immutable, clones should generally be unnecessary, but they may
     *  be useful in allowing Psfs to maintain separate caches for their most recently
     *  returned images.
     */
    virtual PTR(Psf) clone() const = 0;

    /**
     *  @brief Return an Image of the PSF, in a form that can be compared directly with star images.
     *
     *  The specified position is a floating point number, and the resulting image will have a Psf
     *  with the correct fractional position, with the centre within pixel (width/2, height/2)
     *  Specifically, fractional positions in [0, 0.5] will appear above/to the right of the center,
     *  and fractional positions in (0.5, 1] will appear below/to the left (0.9999 is almost back at
     *  the middle).
     *
     *  The image's (X0, Y0) will be set correctly to reflect this, such that the returned image can
     *  be directly compared to a star at the given position.
     *
     *  The returned image is normalized to sum to unity.
     *
     *  @param[in]  position     Position to evaluate the PSF at; defaults to getAveragePosition().
     *  @param[in]  color        Color of the source for which to evaluate the PSF; defaults to
     *                           getAverageColor().
     *  @param[in]  owner        Whether to copy the return value or return an internal image that
     *                           must be handled with care (see ImageOwnerEnum).
     *
     *  The Psf class caches the most recent return value of computeImage, so repeated calls
     *  with the same arguments will be highly optimized.
     *
     *  @note The real work is done in the virtual private member function Psf::doComputeImage;
     *        computeImage only handles caching and default arguments.
     */
    PTR(Image) computeImage(
        geom::Point2D position=makeNullPoint(),
        image::Color color=image::Color(),
        ImageOwnerEnum owner=COPY
    ) const;

    /**
     *  @brief Return an Image of the PSF, in a form suitable for convolution.
     *
     *  While the position need not be an integer, the center of the PSF image returned by
     *  computeKernelImage will in the center of the center pixel of the image, which will be
     *  (0,0) when the Image's xy0 is taken into account.
     *
     *  This is similar to the image returned by a Kernel, but with the image's xy0 set such that
     *  the center is at (0,0) (but see #2620, which proposes using the same convention for Kernel).
     *
     *  The returned image is normalized to sum to unity.
     *
     *  @param[in]  position     Position to evaluate the PSF at; defaults to getAveragePosition().
     *  @param[in]  color        Color of the source for which to evaluate the PSF; defaults to
     *                           getAverageColor().
     *  @param[in]  owner        Whether to copy the return value or return an internal image that
     *                           must be handled with care (see ImageOwnerEnum).
     *
     *  The Psf class caches the most recent return value of computeKernelImage, so repeated calls
     *  with the same arguments will be highly optimized.
     *
     *  @note The real work is done in the virtual private member function Psf::doComputeKernelImage;
     *        computeKernelImage only handles caching and default arguments.
     */
    PTR(Image) computeKernelImage(
        geom::Point2D position=makeNullPoint(),
        image::Color color=image::Color(),
        ImageOwnerEnum owner=COPY
    ) const;

    /**
     *  @brief  Return the peak value of the PSF image.
     *
     *  @param[in]  position     Position to evaluate the PSF at; defaults to getAveragePosition().
     *  @param[in]  color        Color of the source for which to evaluate the PSF; defaults to
     *                           getAverageColor().
     *
     *  This calls computeKernelImage internally, but because this will usually be cached, it shouldn't
     *  be expensive (but be careful not to accidentally call it with no arguments when you actually
     *  want to call it with the same arguments just used to call computeImage or computeKernelImage).
     */
    double computePeak(
        geom::Point2D position=makeNullPoint(),
        image::Color color=image::Color()
    ) const;

    /**
     *  @brief Compute the "flux" of the Psf model within a circular aperture of the given radius.
     *
     *  @param[in]  radius       Radius of the aperture to measure.
     *  @param[in]  position     Position to evaluate the PSF at; defaults to getAveragePosition().
     *  @param[in]  color        Color of the source for which to evaluate the PSF; defaults to
     *                           getAverageColor().
     *
     *  The flux is relative to a Psf image that has been normalized to unit integral, and the radius
     *  is in pixels.
     */
    double computeApertureFlux(
        double radius,
        geom::Point2D position=makeNullPoint(),
        image::Color color=image::Color()
    ) const;

    /**
     *  @brief Compute the ellipse corresponding to the second moments of the Psf.
     *
     *  @param[in]  position     Position to evaluate the PSF at; defaults to getAveragePosition().
     *  @param[in]  color        Color of the source for which to evaluate the PSF; defaults to
     *                           getAverageColor().
     *
     *  The algorithm used to compute the moments is up to the derived class, and hence this
     *  method should not be used when a particular algorithm or weight function is required.
     */
    geom::ellipses::Quadrupole computeShape(
        geom::Point2D position=makeNullPoint(),
        image::Color color=image::Color()
    ) const;

    /**
     *  @brief Return a FixedKernel corresponding to the Psf image at the given point.
     *
     *  This is implemented by calling computeKernelImage, and is simply provided for
     *  convenience.
     */
    PTR(math::Kernel const) getLocalKernel(
        geom::Point2D position=makeNullPoint(),
        image::Color color=image::Color()
    ) const;

    /**
     *  @brief Return the average Color of the stars used to construct the Psf
     *
     *  This is also the Color used to return an image if you don't specify a Color.
     */
    image::Color getAverageColor() const { return image::Color(); }

    /**
     *  @brief Return the average position of the stars used to construct the Psf.
     *
     *  This is also the position used to return an image if you don't specify a position.
     */
    virtual geom::Point2D getAveragePosition() const;

    /**
     * Helper function for Psf::doComputeImage(): converts a kernel image (centered at (0,0) when xy0
     * is taken into account) to an image centered at position when xy0 is taken into account.
     *
     * @c warpAlgorithm is passed to afw::math::makeWarpingKernel() and can be "nearest", "bilinear",
     * or "lanczosN"
     *
     * @c warpBuffer zero-pads the image before recentering.  Recommended value is 1 for bilinear,
     * N for lanczosN (note that it would be cleaner to infer this value from the warping algorithm
     * but this would require mild API changes; same issue occurs in e.g. afw::math::offsetImage()).
     *
     * The point with integer coordinates @c (0,0) in the source image (with xy0 taken into account)
     * corresponds to the point @c position in the destination image.  If @c position is not
     * integer-valued then we will need to fractionally shift the image using interpolation.
     *
     * Note: if fractional recentering is performed, then a new image will be allocated and returned.
     * If not, then the original image will be returned (after setting XY0).
     */
    static PTR(Image) recenterKernelImage(
        PTR(Image) im,
        geom::Point2D const & position,
        std::string const & warpAlgorithm = "lanczos5",
        unsigned int warpBuffer = 5
    );

protected:
 
    /**
     *  Main constructor for subclasses.
     *
     *  @param[in] isFixed  Should be true for Psf for which doComputeKernelImage always returns
     *                      the same image, regardless of color or position arguments.
     */
    explicit Psf(bool isFixed=false);

private:

    //@{
    /**
     *  These virtual member functions are private, not protected, because we only want derived classes
     *  to implement them, not call them; they should call the corresponding compute*Image member
     *  functions instead so as to let the Psf base class handle caching properly.
     *
     *  Derived classes are responsible for ensuring that returned images sum to one.
     */
    virtual PTR(Image) doComputeImage(
        geom::Point2D const & position, image::Color const& color
    ) const;
    virtual PTR(Image) doComputeKernelImage(
        geom::Point2D const & position, image::Color const & color
    ) const = 0;
    virtual double doComputeApertureFlux(
        double radius, geom::Point2D const & position, image::Color const & color
    ) const = 0;
    virtual geom::ellipses::Quadrupole doComputeShape(
        geom::Point2D const & position, image::Color const & color
    ) const = 0;
    //@}

    bool const _isFixed;
    mutable PTR(Image) _cachedImage;
    mutable PTR(Image) _cachedKernelImage;
    mutable image::Color _cachedImageColor;
    mutable image::Color _cachedKernelImageColor;
    mutable geom::Point2D _cachedImagePosition;
    mutable geom::Point2D _cachedKernelImagePosition;

    LSST_PERSIST_FORMATTER(PsfFormatter)
};

}}} // namespace lsst::afw::detection

#endif // !LSST_AFW_DETECTION_Psf_h_INCLUDED
