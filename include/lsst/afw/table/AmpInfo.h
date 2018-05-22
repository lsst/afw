// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
#ifndef AFW_TABLE_AmpInfo_h_INCLUDED
#define AFW_TABLE_AmpInfo_h_INCLUDED

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace table {

/**
 * Readout corner, in the frame of reference of the assembled image
 */
enum ReadoutCorner {
    LL,
    LR,
    UR,
    UL,
};

class AmpInfoRecord;
class AmpInfoTable;

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
 *   are always along the Y axis). If you must swap X/Y then add a doTranspose flag.
 */
class AmpInfoRecord : public BaseRecord {
public:
    typedef AmpInfoTable Table;
    typedef ColumnViewT<AmpInfoRecord> ColumnView;
    typedef CatalogT<AmpInfoRecord> Catalog;
    typedef CatalogT<AmpInfoRecord const> ConstCatalog;

    AmpInfoRecord(AmpInfoRecord const &) = delete;
    AmpInfoRecord(AmpInfoRecord &&) = delete;
    AmpInfoRecord & operator=(AmpInfoRecord const &) = delete;
    AmpInfoRecord & operator=(AmpInfoRecord &&) = delete;
    ~AmpInfoRecord();

    std::shared_ptr<AmpInfoTable const> getTable() const {
        return std::static_pointer_cast<AmpInfoTable const>(BaseRecord::getTable());
    }

    //@{
    /// Convenience accessors for the keys in the minimal reference schema.
    std::string getName() const;
    void setName(std::string const &name);  ///< name of amplifier location in camera

    lsst::geom::Box2I getBBox() const;
    void setBBox(lsst::geom::Box2I const &bbox);  ///< bounding box of amplifier pixels in assembled image

    double getGain() const;
    void setGain(double gain);  ///< amplifier gain in e-/ADU

    double getReadNoise() const;
    void setReadNoise(double readNoise);  ///< amplifier read noise, in e-

    double getSaturation() const;
    void setSaturation(double saturation);  ///< level in ADU above which pixels are considered saturated;
    ///< use `nan` if no such level applies",

    double getSuspectLevel() const;
    void setSuspectLevel(
            double suspectLevel);  ///< level in ADU above which pixels are considered suspicious,
    ///< meaning they may be affected by unknown systematics; for example if non-linearity corrections
    ///< above a certain level are unstable then that would be a useful value for suspectLevel;
    ///< use `nan` if no such level applies",

    ReadoutCorner getReadoutCorner() const;
    void setReadoutCorner(ReadoutCorner val);  ///< set readout corner

    std::vector<double> getLinearityCoeffs() const;
    void setLinearityCoeffs(std::vector<double> const &coeffs);  ///< vector of linearity coefficients

    std::string getLinearityType() const;
    void setLinearityType(std::string const &type);  ///< name of linearity parameterization

    bool getHasRawInfo() const;
    void setHasRawInfo(bool hasRawInfo);  ///< does this table have raw amplifier information?

    lsst::geom::Box2I getRawBBox() const;
    void setRawBBox(lsst::geom::Box2I const &bbox);  ///< bounding box of all amplifier pixels on raw image

    lsst::geom::Box2I getRawDataBBox() const;
    void setRawDataBBox(
            lsst::geom::Box2I const &bbox);  ///< bounding box of amplifier image pixels on raw image

    bool getRawFlipX() const;
    void setRawFlipX(bool rawFlipX);  ///< flip row order to make assembled image?

    bool getRawFlipY() const;
    void setRawFlipY(bool rawFlipY);  ///< flip column order to make an assembled image?

    lsst::geom::Extent2I getRawXYOffset() const;
    void setRawXYOffset(lsst::geom::Extent2I const
                                &xy);  ///< offset for assembling a raw CCD image: desired xy0 - raw xy0

    lsst::geom::Box2I getRawHorizontalOverscanBBox() const;
    void setRawHorizontalOverscanBBox(
            lsst::geom::Box2I const &bbox);  ///< bounding box of usable horizontal overscan pixels

    lsst::geom::Box2I getRawVerticalOverscanBBox() const;
    void setRawVerticalOverscanBBox(
            lsst::geom::Box2I const &bbox);  ///< bounding box of usable vertical overscan pixels

    lsst::geom::Box2I getRawPrescanBBox() const;
    void setRawPrescanBBox(lsst::geom::Box2I const &bbox);  ///< bounding box of usable (horizontal) prescan
                                                            ///< pixels on raw image

    //@}

protected:
    friend class AmpInfoTable;

    explicit AmpInfoRecord(std::shared_ptr<AmpInfoTable> const &table);
};

/**
 *  Table of amplifier information (AmpInfoRecord records)
 *
 *  @copydetails AmpInfoRecord
 */
class AmpInfoTable : public BaseTable {
public:
    typedef AmpInfoRecord Record;
    typedef ColumnViewT<AmpInfoRecord> ColumnView;
    typedef CatalogT<Record> Catalog;
    typedef CatalogT<Record const> ConstCatalog;
    static int const MAX_NAME_LENGTH = 64;            // max length for amplifier name
    static int const MAX_LINEARITY_COEFFS = 4;        // max number of linearity coefficients
    static int const MAX_LINEARITY_TYPE_LENGTH = 64;  // max length for linearity type

    AmpInfoTable & operator=(AmpInfoTable const &) = delete;
    AmpInfoTable & operator=(AmpInfoTable &&) = delete;
    ~AmpInfoTable();

    /**
     *  Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     */
    static std::shared_ptr<AmpInfoTable> make(Schema const &schema);

    /**
     *  Return a minimal schema for AmpInfo tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on AmpInfoRecords will assume that at least the fields
     *  provided by this routine are present.
     */
    static Schema makeMinimalSchema() {
        Schema r = getMinimalSchema().schema;
        r.disconnectAliases();
        return r;
    }

    /**
     *  Return true if the given schema is a valid AmpInfoTable schema.
     *
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(Schema const &other) { return other.contains(getMinimalSchema().schema); }

    //@{
    /**
     *  Get keys for standard fields shared by all references.
     *
     *  These keys are used to implement getters and setters on AmpInfoRecord.
     */
    static Key<std::string> getNameKey() { return getMinimalSchema().name; }
    static PointKey<int> getBBoxMinKey() { return getMinimalSchema().bboxMin; }
    static PointKey<int> getBBoxExtentKey() { return getMinimalSchema().bboxExtent; }
    static Key<double> getGainKey() { return getMinimalSchema().gain; }
    static Key<double> getReadNoiseKey() { return getMinimalSchema().readNoise; }
    static Key<double> getSaturationKey() { return getMinimalSchema().saturation; }
    static Key<double> getSuspectLevelKey() { return getMinimalSchema().suspectLevel; }
    static Key<int> getReadoutCornerKey() { return getMinimalSchema().readoutCorner; }
    static Key<Array<double> > getLinearityCoeffsKey() { return getMinimalSchema().linearityCoeffs; }
    static Key<std::string> getLinearityTypeKey() { return getMinimalSchema().linearityType; }
    static Key<Flag> getHasRawInfoKey() { return getMinimalSchema().hasRawInfo; }
    static PointKey<int> getRawBBoxMinKey() { return getMinimalSchema().rawBBoxMin; }
    static PointKey<int> getRawBBoxExtentKey() { return getMinimalSchema().rawBBoxExtent; }
    static PointKey<int> getRawDataBBoxMinKey() { return getMinimalSchema().rawDataBBoxMin; }
    static PointKey<int> getRawDataBBoxExtentKey() { return getMinimalSchema().rawDataBBoxExtent; }
    static Key<Flag> getRawFlipXKey() { return getMinimalSchema().rawFlipX; }
    static Key<Flag> getRawFlipYKey() { return getMinimalSchema().rawFlipY; }
    static PointKey<int> getRawXYOffsetKey() { return getMinimalSchema().rawXYOffset; }
    static PointKey<int> getRawHorizontalOverscanBBoxMinKey() {
        return getMinimalSchema().rawHorizontalOverscanBBoxMin;
    }
    static PointKey<int> getRawHorizontalOverscanBBoxExtentKey() {
        return getMinimalSchema().rawHorizontalOverscanBBoxExtent;
    }
    static PointKey<int> getRawVerticalOverscanBBoxMinKey() {
        return getMinimalSchema().rawVerticalOverscanBBoxMin;
    }
    static PointKey<int> getRawVerticalOverscanBBoxExtentKey() {
        return getMinimalSchema().rawVerticalOverscanBBoxExtent;
    }
    static PointKey<int> getRawPrescanBBoxMinKey() { return getMinimalSchema().rawPrescanBBoxMin; }
    static PointKey<int> getRawPrescanBBoxExtentKey() { return getMinimalSchema().rawPrescanBBoxExtent; }
    //@}

    /// @copydoc BaseTable::clone
    std::shared_ptr<AmpInfoTable> clone() const { return std::static_pointer_cast<AmpInfoTable>(_clone()); }

    /// @copydoc BaseTable::makeRecord
    std::shared_ptr<AmpInfoRecord> makeRecord() {
        return std::static_pointer_cast<AmpInfoRecord>(_makeRecord());
    }

    /// @copydoc BaseTable::copyRecord
    std::shared_ptr<AmpInfoRecord> copyRecord(BaseRecord const &other) {
        return std::static_pointer_cast<AmpInfoRecord>(BaseTable::copyRecord(other));
    }

    /// @copydoc BaseTable::copyRecord
    std::shared_ptr<AmpInfoRecord> copyRecord(BaseRecord const &other, SchemaMapper const &mapper) {
        return std::static_pointer_cast<AmpInfoRecord>(BaseTable::copyRecord(other, mapper));
    }

protected:
    explicit AmpInfoTable(Schema const &schema);

    explicit AmpInfoTable(AmpInfoTable const &other);
    explicit AmpInfoTable(AmpInfoTable &&other);

    std::shared_ptr<BaseTable> _clone() const override;

    std::shared_ptr<BaseRecord> _makeRecord() override;

private:
    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        Schema schema;
        Key<std::string> name;
        PointKey<int> bboxMin;
        PointKey<int> bboxExtent;
        Key<double> gain;
        Key<double> readNoise;
        Key<double> saturation;
        Key<double> suspectLevel;
        Key<int> readoutCorner;
        Key<Array<double> > linearityCoeffs;
        Key<std::string> linearityType;
        Key<Flag> hasRawInfo;
        PointKey<int> rawBBoxMin;
        PointKey<int> rawBBoxExtent;
        PointKey<int> rawDataBBoxMin;
        PointKey<int> rawDataBBoxExtent;
        Key<Flag> rawFlipX;
        Key<Flag> rawFlipY;
        PointKey<int> rawXYOffset;
        PointKey<int> rawHorizontalOverscanBBoxMin;
        PointKey<int> rawHorizontalOverscanBBoxExtent;
        PointKey<int> rawVerticalOverscanBBoxMin;
        PointKey<int> rawVerticalOverscanBBoxExtent;
        PointKey<int> rawPrescanBBoxMin;
        PointKey<int> rawPrescanBBoxExtent;
        MinimalSchema();
    };

    // Return the singleton minimal schema.
    static MinimalSchema &getMinimalSchema();

    friend class io::FitsWriter;

    // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    std::shared_ptr<io::FitsWriter> makeFitsWriter(fits::Fits *fitsfile, int flags) const override;
};
}
}
}  // namespace lsst::afw::table

#endif  // !AFW_TABLE_AmpInfo_h_INCLUDED
