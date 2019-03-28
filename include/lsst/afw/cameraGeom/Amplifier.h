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
 *  Geometry and electronic information about raw amplifier images
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
 * * The overscan and underscan bounding boxes are regions containing USABLE data,
 *   NOT the entire underscan and overscan region. These bounding boxes should exclude areas
 *   with weird electronic artifacts. Each bounding box can be empty (0 extent) if the corresponding
 *   region is not used for data processing.
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

    /// Return the schema used in the afw.table representation of amplifiers.
    static table::Schema getRecordSchema();

    /**
     * Construct a new Amplifier object from the fields in the given record.
     *
     * @param[in] record   Record to copy fields from.
     */
    static std::shared_ptr<Amplifier const> fromRecord(table::BaseRecord const & record);

    Amplifier() = default;

    Amplifier(Amplifier const &) = delete;
    Amplifier(Amplifier &&) = delete;
    Amplifier &operator=(Amplifier const &) = delete;
    Amplifier &operator=(Amplifier &&) = delete;
    ~Amplifier() noexcept = default;

    /**
     * Copy the Amplifier's fields into the given record.
     *
     * @param[in,out] record   Record to modify.
     *                         `record.getSchema().contains(this->getRecordSchema())` must be true.
     */
    void toRecord(table::BaseRecord & record) const;

    //@{
    /// Name of the amplifier.
    std::string getName() const { return _name; }
    void setName(std::string const &name) { _name = name; }
    //@}

    //@{
    /// Bounding box of amplifier pixels in the trimmed, assembled image.
    lsst::geom::Box2I getBBox() const { return _bbox; }
    void setBBox(lsst::geom::Box2I const &bbox) { _bbox = bbox; }
    //@}

    //@{
    /// Amplifier gain in e-/ADU.
    double getGain() const { return _gain; }
    void setGain(double gain) { _gain = gain; }
    //@}

    //@{
    /// Amplifier read noise, in e-.
    double getReadNoise() const { return _readNoise; }
    void setReadNoise(double readNoise) { _readNoise = readNoise; }
    //@}

    //@{
    /**
     *  Level in ADU above which pixels are considered saturated; use `nan` if
     *  no such level applies.
     */
    double getSaturation() const { return _saturation; }
    void setSaturation(double saturation) { _saturation = saturation; }
    //@}

    //@{
    /**
     *  Level in ADU above which pixels are considered suspicious, meaning
     *  they may be affected by unknown systematics; for example if
     *  non-linearity corrections above a certain level are unstable then that
     *  would be a useful value for suspectLevel. Use `nan` if no such level
     *  applies.
     */
    double getSuspectLevel() const { return _suspectLevel; }
    void setSuspectLevel(double suspectLevel) { _suspectLevel = suspectLevel; }
    //@}

    //@{
    /// Readout corner in the trimmed, assembled image.
    ReadoutCorner getReadoutCorner() const { return _readoutCorner; }
    void setReadoutCorner(ReadoutCorner readoutCorner) { _readoutCorner = readoutCorner; }
    //@}

    //@{
    /// Vector of linearity coefficients.
    ndarray::Array<double const, 1, 1> getLinearityCoeffs() const { return _linearityCoeffs; }
    void setLinearityCoeffs(ndarray::Array<double const, 1, 1> const & coeffs) { _linearityCoeffs = coeffs; }
    //@}

    //@{
    /// Name of linearity parameterization.
    std::string getLinearityType() const { return _linearityType; }
    void setLinearityType(std::string const & type) { _linearityType = type; }
    //@}

    //@{
    /// # TODO! NEVER DOCUMENTED!
    double getLinearityThreshold() const { return _linearityThreshold; }
    void setLinearityThreshold(double threshold) { _linearityThreshold = threshold; }
    //@}

    //@{
    /// # TODO! NEVER DOCUMENTED!
    double getLinearityMaximum() const { return _linearityMaximum; }
    void setLinearityMaximum(double maximum) { _linearityMaximum = maximum; }
    //@}

    //@{
    /// # TODO! NEVER DOCUMENTED!
    std::string getLinearityUnits() const { return _linearityUnits; }
    void setLinearityUnits(std::string const & units) { _linearityUnits = units; }
    //@}

    //@{
    /// Does this table have raw amplifier information?
    [[deprecated("Amplifier objects always have raw information.")]]
    bool getHasRawInfo() const { return true; }
    //@}

    //@{
    /**
     *  Bounding box of all amplifier pixels on untrimmed, assembled raw
     *  image.
     */
    lsst::geom::Box2I getRawBBox() const { return _rawBBox; }
    void setRawBBox(lsst::geom::Box2I const &bbox) { _rawBBox = bbox; }
    //@}

    //@{
    /**
     *  Bounding box of amplifier image pixels on untrimmed, assembled raw
     *  image.
     */
    lsst::geom::Box2I getRawDataBBox() const { return _rawDataBBox; }
    void setRawDataBBox(lsst::geom::Box2I const &bbox) { _rawDataBBox = bbox; }
    //@}

    //@{
    /**
     *  Flip row order in transformation from untrimmed, assembled raw image
     *  to trimmed, assembled post-ISR image?
     */
    bool getRawFlipX() const { return _rawFlipX; }
    void setRawFlipX(bool rawFlipX) { _rawFlipX = rawFlipX; }
    //@}

    //@{
    /**
     *  Flip column order in transformation from untrimmed, assembled raw
     *  image to trimmed, assembled post-ISR image?
     */
    bool getRawFlipY() const { return _rawFlipY; }
    void setRawFlipY(bool rawFlipY) { _rawFlipY = rawFlipY; }
    //@}

    //@{
    /**
     *  Offset in transformation from pre-raw, unassembled image to trimmed,
     *  assembled post-ISR image: final xy0 - pre-raw xy0.
     */
    lsst::geom::Extent2I getRawXYOffset() const { return _rawXYOffset; }
    void setRawXYOffset(lsst::geom::Extent2I const &xy) { _rawXYOffset = xy; }
    //@}

    //@{
    /**
     * The bounding box of usable horizontal overscan pixels in the assembled,
     * untrimmed raw image.
     */
    lsst::geom::Box2I getRawHorizontalOverscanBBox() const { return _rawHorizontalOverscanBBox; }
    void setRawHorizontalOverscanBBox(lsst::geom::Box2I const &bbox) { _rawHorizontalOverscanBBox = bbox; }
    //@}

    //@{
    /**
     * The bounding box of usable vertical overscan pixels in the assembled,
     * untrimmed raw image.
     */
    lsst::geom::Box2I getRawVerticalOverscanBBox() const { return _rawVerticalOverscanBBox; }
    void setRawVerticalOverscanBBox(lsst::geom::Box2I const &bbox) { _rawVerticalOverscanBBox = bbox; }
    //@}

    //@{
    /**
     * The bounding box of usable (horizontal) prescan pixels in the assembled,
     * untrimmed raw image.
     */
    lsst::geom::Box2I getRawPrescanBBox() const { return _rawPrescanBBox; }
    void setRawPrescanBBox(lsst::geom::Box2I const &bbox) { _rawPrescanBBox = bbox; }
    //@}

private:
    std::string _name;
    lsst::geom::Box2I _bbox;
    double _gain = 0.0;
    double _readNoise = 0.0;
    double _saturation = 0.0;
    double _suspectLevel = 0.0;
    ReadoutCorner _readoutCorner = ReadoutCorner::LL;
    ndarray::Array<double const, 1, 1> _linearityCoeffs;
    std::string _linearityType;
    double _linearityThreshold;
    double _linearityMaximum;
    std::string _linearityUnits;
    lsst::geom::Box2I _rawBBox;
    lsst::geom::Box2I _rawDataBBox;
    bool _rawFlipX = false;
    bool _rawFlipY = false;
    lsst::geom::Extent2I _rawXYOffset;
    lsst::geom::Box2I _rawHorizontalOverscanBBox;
    lsst::geom::Box2I _rawVerticalOverscanBBox;
    lsst::geom::Box2I _rawPrescanBBox;
};

}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_CAMERAGEOM_AMPLIFIER_H_INCLUDED
