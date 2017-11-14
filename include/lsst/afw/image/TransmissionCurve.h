/*
 * LSST Data Management System
 * Copyright 2017 LSST/AURA.
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

#ifndef LSST_AFW_IMAGE_TransmissionCurve_h_INCLUDED
#define LSST_AFW_IMAGE_TransmissionCurve_h_INCLUDED

#include "ndarray_fwd.h"

#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace image {


/**
 *  A spatially-varying transmission curve as a function of wavelength.
 *
 *  TransmissionCurve can only be evaluated at discrete user-provided positions;
 *  it does not provide an interface for computing average transmission over
 *  regions or computing spatially-varying scalars from integrals over the
 *  wavelength dimension.
 *
 *  TransmissionCurves are immutable and are expected to be passed and held
 *  by shared_ptr.  As such they are neither copyable nor movable.
 *
 *  All wavelength values should be in Angstroms.
 *
 *  The flux units and overall normalization of TransmissionCurves is
 *  unspecified by the class, but their normalization and units should always
 *  be consistent throughout the spatial area over which they are defined
 *  (an implementation should not e.g. re-normalize to unit bolometric flux
 *  at each position it is evaluated at).  Other classes and functions using
 *  TransmissionCurves should of course document the flux units and/or
 *  normalization expected/provided.
 */
class TransmissionCurve : public table::io::PersistableFacade<TransmissionCurve>,
                          public table::io::Persistable
{
public:

    /**
     *  A simple struct used to specify the wavelength grid for evaluation.
     */
    struct SampleDef {

        /// Construct with infinite bounds and zero size.
        SampleDef();

        /// Construct with the given bounds and size.
        SampleDef(double min_, double max_, int size_) : min(min_), max(max_), size(size_) {}

        double min; ///< minimum wavelength (inclusive; Angstroms)
        double max; ///< maximum wavelength (inclusive; Angstroms)
        int size;   ///< number of evenly-spaced (in wavelength) sample points

        /// Return the spacing between sample points, or zero if size is zero.
        double getSpacing() const { return size == 0 ? 0 : (max - min)/size; }

        /// Create an array containing the full set of wavelength sample points.
        ndarray::Array<double,1,1> makeArray() const;
    };

    /**
     *  Create a new TranmissionCurve that has unit thoughput at all wavelengths everywhere.
     */
    static std::shared_ptr<TransmissionCurve> makeIdentity();

    /**
     *  Create a new TransmissionCurve with spatially-constant throughput.
     *
     *  @param[in]  throughput      an Array of throughput values with the same
     *                              size as the wavelengths Array (will be copied).
     *  @param[in]  wavelengths     an Array of wavelengths in Angstroms (will be copied).
     *                              Must be monotonically increasing.
     *  @param[in]  throughputAtMin the throughput value used for wavelengths
     *                              below wavelengths.front().
     *  @param[in]  throughputAtMax the throughput value used for wavelengths
     *                              above wavelengths.back().
     */
    static std::shared_ptr<TransmissionCurve> makeConstant(
        ndarray::Array<double const,1> const & throughput,
        ndarray::Array<double const,1> const & wavelengths,
        double throughputAtMin=0.0, double throughputAtMax=0.0
    );

    /**
     *  Create a new TransmissionCurve with throughput varying as function of radius.
     *
     *  @param[in]  throughput      an Array of throughput values with shape
     *                              (wavelengths.size(), radii.size()).  Will be
     *                              copied.
     *  @param[in]  wavelengths     an Array of wavelengths in Angstroms (will be copied).
     *                              Must be monotonically increasing.
     *  @param[in]  radii           an Array of radii (will be copied).
     *                              Must be monotonically increasing.
     *  @param[in]  throughputAtMin the throughput value used for wavelengths
     *                              below wavelengths.front().
     *  @param[in]  throughputAtMax the throughput value used for wavelengths
     *                              above wavelengths.back().
     *
     *  Throughput at radii below radii.front() or above radii.back() is assumed
     *  to be constant.
     */
    static std::shared_ptr<TransmissionCurve> makeRadial(
        ndarray::Array<double const,2> const & throughput,
        ndarray::Array<double const,1> const & wavelengths,
        ndarray::Array<double const,1> const & radii,
        double throughputAtMin=0.0, double throughputAtMax=0.0
    );

    /**
     *  Return a new TransmissionCurve that simply multiplies the values of two others.
     *
     *  The product is computed lazily (the returned object samples from each
     *  operand, and then multiplies those samples).
     *
     *  @note This is mapped to __mul__ in Python (overriding operator* in C++
     *        would be problematic due to the use of shared_ptr).
     */
    static std::shared_ptr<TransmissionCurve> multiply(
        std::shared_ptr<TransmissionCurve> a,
        std::shared_ptr<TransmissionCurve> b
    );

    /**
     *  Return a view of a TransmissionCurve in a different coordinate system.
     *
     *  The transform is computed lazily (input points are transformed and then
     *  the original TransmissionCurve is evaluated at the transformed points).
     *
     *  @note This is wrapped as a bound method in Python (in C++ this
     *        would be problematic due to the use of shared_ptr for base).
     */
    static std::shared_ptr<TransmissionCurve> transform(
        std::shared_ptr<TransmissionCurve> base,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    );

    // TransmissionCurve is not copyable.
    TransmissionCurve(TransmissionCurve const &) = delete;

    // TransmissionCurve is not movable.
    TransmissionCurve(TransmissionCurve &&) = delete;

    // TransmissionCurve is not copy-assignable.
    TransmissionCurve & operator=(TransmissionCurve const &) = delete;

    // TransmissionCurve is not move-assignable.
    TransmissionCurve & operator=(TransmissionCurve &&) = delete;

    // TransmissionCurve is polymorphic.
    virtual ~TransmissionCurve() {}

    /**
     *  Return the bounds and spacing recommended for full evaluation.
     *
     *  Values within the returned min and max (inclusive) are guaranteed to
     *  not yield NaN results when passed to sampleAt.
     *
     *  Min and/or max values may be infinite and size may be zero to indicate
     *  an analytic curve with no wavelength bounds.
     */
    virtual SampleDef getNaturalSampling() const = 0;

    /**
     *  Return the throughput value that will be returned for wavelengths
     *  below and above getNaturalSampling().min and .max (respectively).
     *
     *  Results may be NaN to indicate that the throughput is unknown, but must
     *  otherwise be finite.
     */
    virtual std::pair<double,double> getThroughputAtBounds() const = 0;

    //@{
    /**
     *  Evaluate the throughput at a position into a provided output array.
     *
     *  @param[in]  position     Spatial position at which to evaluate.
     *  @param[in]  wavelengths  Wavelengths at which to evaluate.
     *
     *  @param[in,out]  out      Computed throughput values.  Must be pre-
     *                           allocated to the same size as the wavelengths
     *                           array.  May include NaN values if the
     *                           requested wavelengths are beyond bounds
     *                           reported by getWavelengthBounds(). Out-of-
     *                           bounds wavelengths may also yield zero
     *                           throughput values if throughput is known to
     *                           be limited to those bounds.
     *
     *  @throws pex::exceptions::DomainError if the given position is outside
     *          the region for which the TransmissionCurve is defined.
     */
    virtual void sampleAt(
        geom::Point2D const & position,
        ndarray::Array<double const,1,1> const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const = 0;

    void sampleAt(
        geom::Point2D const & position,
        SampleDef const & wavelengths,
        ndarray::Array<double,1,1> const & out
    ) const;
    //@}


    //@{
    /**
     *  Evaluate the throughput at a position into a new array.
     *
     *  @param[in]  position     Spatial position at which to evaluate.
     *  @param[in]  wavelengths  Wavelengths at which to evaluate.
     *
     *  @return  Computed throughput values, in an array with the same size as
     *           wavelengths.  May include NaN values if the requested
     *           wavelengths are beyond bounds reported by
     *           getWavelengthBounds().  Out-of-bounds wavelengths may also
     *           yield zero throughput values if throughput is known to be
     *           limited to those bounds.
     *
     *  @throws pex::exceptions::DomainError if the given position is outside
     *          the region for which the TransmissionCurve is defined.
     */
    ndarray::Array<double,1,1> sampleAt(
        geom::Point2D const & position,
        ndarray::Array<double const,1,1> const & wavelengths
    ) const;

    ndarray::Array<double,1,1> sampleAt(
        geom::Point2D const & position,
        SampleDef const & wavelengths
    ) const;
    //@}

protected:

    /**
     *  Polymorphic implementation for transform().
     *
     *  The default implementation of this method creates a new
     *  TransmissionCurve that lazily applies the given transform to points
     *  before evaluating the original TransmissionCurve, which should be
     *  appropriate for nearly all concrete TransmissionCurve subclases.
     *
     *  @param[in] self       A shared_ptr to this.
     *  @param[in] transform  A transform to that maps the coordinate system
     *                        of the returned transform to that of self.
     */
    virtual std::shared_ptr<TransmissionCurve> _transformImpl(
        std::shared_ptr<TransmissionCurve> self,
        std::shared_ptr<geom::TransformPoint2ToPoint2> transform
    ) const;

    /**
     *  One-way polymorphic implementation for multiply().
     *
     *  The default implementation simply returns nullptr, indicating that
     *  the other operand should be given an opportunity to specialize (if
     *  it hasn't already).  If neither operand recognizes the other, a
     *  lazy-evaluation product will be constructed.
     *
     *  @param[in] self       A shared_ptr to this.
     *  @param[in] other      The other TransmissionCurve to multiply with self.
     */
    virtual std::shared_ptr<TransmissionCurve> _multiplyImpl(
        std::shared_ptr<TransmissionCurve> self,
        std::shared_ptr<TransmissionCurve> other
    ) const;

    TransmissionCurve() {}

};


}}} // lsst::afw::image

#endif // !LSST_AFW_IMAGE_TransmissionCurve_h_INCLUDED