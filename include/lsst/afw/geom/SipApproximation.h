// -*- lsst-c++ -*-
/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_AFW_GEOM_SipApproximation_h_INCLUDED
#define LSST_AFW_GEOM_SipApproximation_h_INCLUDED

#include <memory>
#include <optional>
#include <vector>

#include "Eigen/Core"

#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/geom/Box.h"
#include "lsst/geom/LinearTransform.h"

namespace lsst { namespace afw { namespace geom {

/**
 *  A fitter and results class for approximating a general Transform in a form compatible with
 *  FITS WCS persistence.
 *
 *  The Simple Imaging Polynomial (SIP) convention (Shupe et al 2005) adds
 *  forward and reverse polynomial mappings to a standard projection FITS WCS
 *  projection (e.g. "TAN" for gnomonic) that relate Intermediate World
 *  Coordinates (see Calabretta & Greisen 2002) to image pixel coordinates.
 *  The SIP "forward" transform is defined by polynomial coefficients @f$A@f$
 *  and @f$B@f$ that map pixel coordinates @f$(u, v)@f$ to Intermediate World
 *  Coordinates @f$(x, y)@f$ via
 *  @f[
 *     \boldsymbol{S}\left[\begin{array}{c}
 *        x \\
 *        y
 *     \end{array}\right]
 *     \equiv
 *     \left[\begin{array}{c}
 *        x_s \\
 *        y_s
 *     \end{array}\right]
 *     =
 *     \left[\begin{array}{c}
 *        (u - u_0) + \displaystyle\sum_{p,q}^{0 \le p + q \le N} \mathrm{A}_{p,q} (u - u_0)^p (v - v_0)^q \\
 *        (v - v_0) + \displaystyle\sum_{p,q}^{0 \le p + q \le N} \mathrm{B}_{p,q} (u - u_0)^p (v - v_0)^q
 *     \end{array}\right]
 *  @f]
 *  The reverse transform has essentially the same form:
 *  @f[
 *     \left[\begin{array}{c}
 *        u - u_0 \\
 *        v - v_0
 *     \end{array}\right]
 *     =
 *     \left[\begin{array}{c}
 *        x_s + \displaystyle\sum_{p,q}^{0 \le p + q \le N} \mathrm{AP}_{p,q} x_s^p y_s^q \\
 *        y_s + \displaystyle\sum_{p,q}^{0 \le p + q \le N} \mathrm{BP}_{p,q} x_s^p y_s^q
 *     \end{array}\right]
 *  @f]
 *  In both cases, @f$(u_0, v_0)@f$ is the pixel origin (CRPIX in FITS WCS) and
 *  @f$\boldsymbol{S}@f$ is the *inverse* of the Jacobian "CD" matrix.
 *
 *  @note In the implementation, we typically refer to @f$(u-u_0, v-v_0)@f$ as
 *  @c dpix (for "pixel delta"), and @f$(x_s, y_s)@f$ as @c iwc (for 
 *  intermediate world coordinates").
 *
 *  While LSST WCSs are in general too complex to be described exactly in FITS
 *  WCS, they can generally be closely approximated by standard FITS WCS
 *  projection with additional SIP distortions.  This class fits such an
 *  approximation.
 */
class SipApproximation final {
public:

    /**
     *  Construct a new approximation by fitting on a grid of points.
     *
     *  @param[in]  target          The true WCS to approximate.
     *  @param[in]  bbox            Pixel-coordinate bounding box over which the
     *                              approximation should be valid.  Used to construct
     *                              the grid of points to fit.
     *  @param[in]  gridShape       Number of points in x and y for the grid of points.
     *  @param[in]  order           Order of the polynomial (same for forward and
     *                              reverse transforms).
     *  @param[in]  pixelOrigin     Pixel reference point.  If not provided, the center
     *                              of `bbox` is used.  Must lie within `bbox`.  Note
     *                              that this is 0-indexed, and is hence one less than
     *                              the FITS CRPIX value.
     *  @param[in]  svdThreshold    Fraction of the largest singular value at which to
     *                              declare smaller singular values zero in the least
     *                              squares solution.  Negative values use Eigen's
     *                              internal default.
     *
     *  @throws lsst::pex::exceptions::InvalidParameterError Thrown if order is negative
     *      or gridShape is non-positive.
     *
     *  @exceptsafe strong
     */
    SipApproximation(
        SkyWcs const & target,
        lsst::geom::Box2D const & bbox,
        lsst::geom::Extent2I const & gridShape,
        int order = 5,
        std::optional<lsst::geom::Point2D> const & pixelOrigin = std::nullopt,
        double svdThreshold=-1
    );

    // No copies just because they'd be a pain to implement and probably aren't necessary
    SipApproximation(SipApproximation const &) = delete;
    SipApproximation & operator=(SipApproximation const &) = delete;

    // Moves are fine.
    SipApproximation(SipApproximation &&) noexcept = default;
    SipApproximation & operator=(SipApproximation &&) noexcept = default;

    // Needs to be defined in .cc, where compiler can see the definitions of forward-declared
    // private implementation classes.
    ~SipApproximation() noexcept;

    /// Return the polynomial order of the current solution (same for forward and reverse).
    int getOrder() const noexcept;

    /**
     * Return a coefficient of the forward transform polynomial.
     *
     * Out-of-bounds arguments yields undefined behavior.
     *
     * @exceptsafe strong
     */
    double getA(int p, int q) const;

    /**
     * Return a coefficient of the forward transform polynomial.
     *
     * Out-of-bounds arguments yields undefined behavior.
     *
     * @exceptsafe strong
     */
    double getB(int p, int q) const;

    /// Return the coefficients of the forward transform polynomial.
    Eigen::MatrixXd getA() const;

    /// Return the coefficients of the forward transform polynomial.
    Eigen::MatrixXd getB() const;

    /// Return the pixel-coordinate bounding box over which the approximation should be valid.
    lsst::geom::Box2D getBBox() const noexcept { return _bbox; }

    /// Return the sky-coordinate origin of the WCS.
    lsst::geom::SpherePoint getSkyOrigin() const noexcept { return _skyOrigin; }

    /// Return the pixel origin of the WCS.
    lsst::geom::Point2D getPixelOrigin() const noexcept { return lsst::geom::Point2D(_pixelOrigin); }

    /// Return the CD matrix of the WCS (in degrees).
    Eigen::Matrix2d getCdMatrix() const noexcept;

    /// Return a TAN-SIP WCS object that evaluates the approximation.
    std::shared_ptr<SkyWcs> getWcs() const noexcept;

    /**
     *  Return the maximum deviations of the solution from the exact transform
     *  on a grid offset from the one used in the fit.
     *
     *  The returned quantities are the maxima of:
     *  ```
     *      Angle delta1 = target.pixelToSky(p).separation(approx.pixelToSky(p));
     *      double delta2 = (target.skyToPixel(s) - approx.skyToPixel(s)).computeNorm();
     *  ```
     *  at all pixel points `p` and sky points `s` on the offset grid.
     */
    std::pair<lsst::geom::Angle, double> computeDeltas() const;

private:

    struct FitGrid;
    struct ValidationGrid;
    struct Solution;

    lsst::geom::Box2D _bbox;
    lsst::geom::Point2D _pixelOrigin;
    lsst::geom::SpherePoint _skyOrigin;
    std::unique_ptr<FitGrid const> _fitGrid;
    std::unique_ptr<ValidationGrid const> _validationGrid;
    std::unique_ptr<Solution const> _solution;
    std::shared_ptr<SkyWcs> _wcs;
};

}}}  // namespace lsst::afw::geom

#endif // !LSST_AFW_GEOM_SipApproximation_h_INCLUDED
