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
#ifndef LSST_AFW_CAMERAGEOM_AMPLIFIER_H_INCLUDED
#define LSST_AFW_CAMERAGEOM_AMPLIFIER_H_INCLUDED

#include <string>

#include "lsst/afw/table/fwd.h"
#include "lsst/geom/Box.h"
#include "lsst/geom/Extent.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Readout corner, in the frame of reference of the assembled image
 */
enum class ReadoutCorner {
    LL,
    LR,
    UR,
    UL,
};

/**
 * Assembly state of the amplifier, used to identify bounding boxes and component existence.
 */
enum class AssemblyState {
    RAW,
    SCIENCE,
};

/**
 *  Geometry and electronic information about raw amplifier images
 *
 *  The Amplifier class itself is an abstract base class that provides no
 *  mutation or copy interfaces.  Typically Amplifiers are constructed via the
 *  Builder subclass, which can produce a shared_ptr to an immutable Amplifier
 *  instance.
 *
 * Here is a pictorial example showing the meaning of flipX and flipY:
 *
 @verbatim
     CCD with 4 amps        Desired assembled output      Use these parameters

     --x         x--            y
    |  amp1    amp2 |           |                               flipX       flipY
    y               y           |                       amp1    False       True
                                | CCD image             amp2    True        True
    y               y           |                       amp3    False       False
    |  amp3    amp4 |           |                       amp4    True        False
     --x         x--             ----------- x
 @endverbatim
 * @note
 * * All bounding boxes are parent boxes with respect to the raw image.
 * * The overscan and prescan bounding boxes represent the full regions;
 *   unusable regions are set via ISR configuration parameters.
 * * xyOffset is not used for instrument signature removal (ISR); it is intended for use by display
 *   utilities. It supports construction of a raw CCD image in the case that raw data is provided as
 *   individual amplifier images (which is uncommon):
 *   * Use 0,0 for cameras that supply raw data as a raw CCD image (most cameras)
 *   * Use nonzero for cameras that supply raw data as separate amplifier images with xy0=0,0 (LSST)
 * * This design assumes assembled X is always +/- raw X, which we require for CCDs (so that bleed trails
 *   are always along the Y axis).
 */
class Amplifier {
public:

    class Builder;

    /// Return the schema used in the afw.table representation of amplifiers.
    static table::Schema getRecordSchema();

    virtual ~Amplifier() noexcept = default;

    /**
     * Copy the Amplifier's fields into the given record.
     *
     * @param[out] record   Record to modify.
     *                      `record.getSchema().contains(this->getRecordSchema())` must be true.
     */
    void toRecord(table::BaseRecord & record) const;

    /**
     * Return a Builder object initialized with the fields of this.
     *
     * This is simply a shortcut for `Amplifier::Builder(*this)`.
     */
    Builder rebuild() const;

    /// Name of the amplifier.
    std::string getName() const { return getFields().name; }

    /// Bounding box of amplifier pixels in the trimmed, assembled image.
    lsst::geom::Box2I getBBox() const { return getFields().bbox; }

    /// Amplifier gain in e-/ADU.
    double getGain() const { return getFields().gain; }

    /// Amplifier read noise, in e-.
    double getReadNoise() const { return getFields().readNoise; }

    /**
     *  Level in ADU above which pixels are considered saturated; use `nan` if
     *  no such level applies.
     */
    double getSaturation() const { return getFields().saturation; }

    /**
     *  Level in ADU above which pixels are considered suspicious, meaning
     *  they may be affected by unknown systematics; for example if
     *  non-linearity corrections above a certain level are unstable then that
     *  would be a useful value for suspectLevel. Use `nan` if no such level
     *  applies.
     */
    double getSuspectLevel() const { return getFields().suspectLevel; }

    /// Readout corner in the trimmed, assembled image.
    ReadoutCorner getReadoutCorner() const { return getFields().readoutCorner; }

    /// Vector of linearity coefficients.
    ndarray::Array<double const, 1, 1> getLinearityCoeffs() const { return getFields().linearityCoeffs; }

    /// Name of linearity parameterization.
    std::string getLinearityType() const { return getFields().linearityType; }

    /// # TODO! NEVER DOCUMENTED!
    double getLinearityThreshold() const { return getFields().linearityThreshold; }

    /// # TODO! NEVER DOCUMENTED!
    double getLinearityMaximum() const { return getFields().linearityMaximum; }

    /// # TODO! NEVER DOCUMENTED!
    std::string getLinearityUnits() const { return getFields().linearityUnits; }

    /// Does this table have raw amplifier information?
    [[deprecated("Amplifier objects always have raw information; will be removed after 19.0.")]] // DM-21711
    bool getHasRawInfo() const { return true; }

    /**
     *  Bounding box of all amplifier pixels on untrimmed, assembled raw
     *  image.
     */
    lsst::geom::Box2I getRawBBox() const { return getFields().rawBBox; }

    /**
     *  Bounding box of amplifier image pixels on untrimmed, assembled raw
     *  image.
     */
    lsst::geom::Box2I getRawDataBBox() const { return getFields().rawDataBBox; }

    /**
     *  Flip row order in transformation from untrimmed, assembled raw image
     *  to trimmed, assembled post-ISR image?
     */
    bool getRawFlipX() const { return getFields().rawFlipX; }

    /**
     *  Flip column order in transformation from untrimmed, assembled raw
     *  image to trimmed, assembled post-ISR image?
     */
    bool getRawFlipY() const { return getFields().rawFlipY; }

    /**
     *  Offset in transformation from pre-raw, unassembled image to trimmed,
     *  assembled post-ISR image: final xy0 - pre-raw xy0.
     */
    lsst::geom::Extent2I getRawXYOffset() const { return getFields().rawXYOffset; }

    /**
     * The bounding box of horizontal overscan pixels in the assembled,
     * untrimmed raw image.
     */
    lsst::geom::Box2I getRawHorizontalOverscanBBox() const { return getFields().rawHorizontalOverscanBBox; }

    /**
     * The bounding box of vertical overscan pixels in the assembled,
     * untrimmed raw image.
     */
    lsst::geom::Box2I getRawVerticalOverscanBBox() const { return getFields().rawVerticalOverscanBBox; }

    /**
     * The bounding box of (horizontal) prescan pixels in the assembled,
     * untrimmed raw image.
     */
    lsst::geom::Box2I getRawPrescanBBox() const { return getFields().rawPrescanBBox; }

protected:

    struct Fields {
        std::string name;
        lsst::geom::Box2I bbox;
        double gain = 0.0;
        double readNoise = 0.0;
        double saturation = 0.0;
        double suspectLevel = 0.0;
        ReadoutCorner readoutCorner = ReadoutCorner::LL;
        ndarray::Array<double const, 1, 1> linearityCoeffs;
        std::string linearityType;
        double linearityThreshold;
        double linearityMaximum;
        std::string linearityUnits;
        lsst::geom::Box2I rawBBox;
        lsst::geom::Box2I rawDataBBox;
        bool rawFlipX = false;
        bool rawFlipY = false;
        lsst::geom::Extent2I rawXYOffset;
        lsst::geom::Box2I rawHorizontalOverscanBBox;
        lsst::geom::Box2I rawVerticalOverscanBBox;
        lsst::geom::Box2I rawPrescanBBox;
    };

    // Amplifier construction and assignment is protected to avoid type-slicing
    // but permit derived classes to implement safe versions of these.

    Amplifier() = default;

    Amplifier(Amplifier const &) = default;
    Amplifier(Amplifier &&) = default;
    Amplifier & operator=(Amplifier const &) = default;
    Amplifier & operator=(Amplifier &&) = default;

    virtual Fields const & getFields() const = 0;

};

/**
 *  A mutable Amplifier subclass class that can be used to incrementally
 *  construct or modify Amplifiers.
 */
class Amplifier::Builder final : public Amplifier {
public:

    /**
     * Construct a new Builder object from the fields in the given
     * record.
     *
     * @param[in] record   Record to copy fields from.
     */
    static Builder fromRecord(table::BaseRecord const & record);

    /// Construct a Builder with default values for all fields.
    Builder() = default;

    /// Standard copy constructor.
    Builder(Builder const &) = default;

    /// Standard move constructor.
    Builder(Builder &&) = default;

    /// Construct a Builder with values initialized from the given Amplifier.
    Builder(Amplifier const & other);

    /// Standard copy assignment.
    Builder & operator=(Builder const &) = default;

    /// Standard move assignment.
    Builder & operator=(Builder &&) = default;

    /// Set the Builder's fields to those of the given Amplifier.
    Builder & operator=(Amplifier const & other);

    ~Builder() noexcept override = default;

    /**
     *  Construct an immutable Amplifier with the same values as the Builder.
     *
     *  The derived type of the return instance is unspecified, and should be
     *  considered an implementation detail.
     */
    std::shared_ptr<Amplifier const> finish() const;

    /// @copydoc Amplifier::getName
    void setName(std::string const &name) { _fields.name = name; }

    /// @copydoc Amplifier::getBBox
    void setBBox(lsst::geom::Box2I const &bbox) { _fields.bbox = bbox; }

    /// @copydoc Amplifier::getGain
    void setGain(double gain) { _fields.gain = gain; }

    /// @copydoc Amplifier::getReadNoise
    void setReadNoise(double readNoise) { _fields.readNoise = readNoise; }

    /// @copydoc Amplifier::getSaturation
    void setSaturation(double saturation) { _fields.saturation = saturation; }

    /// @copydoc Amplifier::getSuspectLevel
    void setSuspectLevel(double suspectLevel) { _fields.suspectLevel = suspectLevel; }

    /// @copydoc Amplifier::getReadoutCorner
    void setReadoutCorner(ReadoutCorner readoutCorner) { _fields.readoutCorner = readoutCorner; }

    /// @copydoc Amplifier::getLinearityCoeffs
    void setLinearityCoeffs(ndarray::Array<double const, 1, 1> const & coeffs) {
        _fields.linearityCoeffs = coeffs;
    }

    /// @copydoc Amplifier::getLinearityType
    void setLinearityType(std::string const & type) { _fields.linearityType = type; }

    /// @copydoc Amplifier::getLinearityThreshold
    void setLinearityThreshold(double threshold) { _fields.linearityThreshold = threshold; }

    /// @copydoc Amplifier::getLinearityMaximum
    void setLinearityMaximum(double maximum) { _fields.linearityMaximum = maximum; }

    /// @copydoc Amplifier::getLinearityUnits
    void setLinearityUnits(std::string const & units) { _fields.linearityUnits = units; }

    /// @copydoc Amplifier::getRawBBox
    void setRawBBox(lsst::geom::Box2I const &bbox) { _fields.rawBBox = bbox; }

    /// @copydoc Amplifier::getRawDataBBox
    void setRawDataBBox(lsst::geom::Box2I const &bbox) { _fields.rawDataBBox = bbox; }

    /// @copydoc Amplifier::getRawFlipX
    void setRawFlipX(bool rawFlipX) { _fields.rawFlipX = rawFlipX; }

    /// @copydoc Amplifier::getRawFlipY
    void setRawFlipY(bool rawFlipY) { _fields.rawFlipY = rawFlipY; }

    /// @copydoc Amplifier::getRawXYOffset
    void setRawXYOffset(lsst::geom::Extent2I const &xy) { _fields.rawXYOffset = xy; }

    /// @copydoc Amplifier::getRawHorizontalOverscanBBox
    void setRawHorizontalOverscanBBox(lsst::geom::Box2I const &bbox) {
        _fields.rawHorizontalOverscanBBox = bbox;
    }

    /// @copydoc Amplifier::getRawVerticalOverscanBBox
    void setRawVerticalOverscanBBox(lsst::geom::Box2I const &bbox) {
        _fields.rawVerticalOverscanBBox = bbox;
    }

    /// @copydoc Amplifier::getRawPrescanBBox
    void setRawPrescanBBox(lsst::geom::Box2I const &bbox) {
        _fields.rawPrescanBBox = bbox;
    }

protected:
    Fields const & getFields() const override { return _fields; }
private:
    Fields _fields;
};



}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_CAMERAGEOM_AMPLIFIER_H_INCLUDED
