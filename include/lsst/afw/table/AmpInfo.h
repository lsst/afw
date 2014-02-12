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
#include "lsst/afw/table/IdFactory.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/SortedCatalog.h"

namespace lsst { namespace afw { namespace table {

class AmpInfoRecord;
class AmpInfoTable;

/**
 *  @brief Record class that must contain a unique ID field and *** some other things for amps.
 *
 *  AmpInfoTable / AmpInfoRecord are intended to be the base class for records containing information about 
 *  amplifiers in detectors.  
 */
class AmpInfoRecord : public BaseRecord {
public:

    typedef AmpInfoTable Table;
    typedef ColumnViewT<AmpInfoRecord> ColumnView;
    typedef SortedCatalogT<AmpInfoRecord> Catalog;
    typedef SortedCatalogT<AmpInfoRecord const> ConstCatalog;

    CONST_PTR(AmpInfoTable) getTable() const {
        return boost::static_pointer_cast<AmpInfoTable const>(BaseRecord::getTable());
    }

    //@{
    /// @brief Convenience accessors for the keys in the minimal reference schema.
    RecordId getId() const;
    void setId(RecordId id);

    std::string getName() const;
    void setName(std::string const &name); ///< name of amplifier location in camera

    geom::Box2I getTrimmedBbox() const;
    void setTrimmedBbox(geom::Box2I const &bbox); ///< bounding box of amplifier pixels in assembled image

    double getGain() const;
    void setGain(double gain); ///< amplifier gain in e-/ADU
    
    double getReadNoise() const;
    void setReadNoise(double readNoise); ///< amplifier read noise, in e-

    std::vector<double> getLinearityCoeffs() const;
    void setLinearityCoeffs(std::vector<double> const &coeffs); ///< vector of linearity coefficients

    std::string getLinearityType() const;
    void setLinearityType(std::string const &type); ///< name of linearity parameterization

    bool getHasRawAmplifier() const;
    void setHasRawAmplifier(bool hasRawAmplifier); ///< does this table have raw amplifier information?

    geom::Box2I getRawBbox() const;
    void setRawBbox(geom::Box2I const &bbox); ///< bounding box of all amplifier pixels on raw image

    geom::Box2I getDataBbox() const;
    void setDataBbox(geom::Box2I const &bbox); ///< bounding box of amplifier image pixels on raw image

    geom::Box2I getHorizontalOverscanBbox() const;
    void setHorizontalOverscanBbox(geom::Box2I const &bbox); ///< bounding box of usable horizontal overscan pixels

    geom::Box2I getVerticalOverscanBbox() const;
    void setVerticalOverscanBbox(geom::Box2I const &bbox); ///< bounding box of usable vertical overscan pixels

    geom::Box2I getPrescanBbox() const;
    void setPrescanBbox(geom::Box2I const &bbox); ///< bounding box of usable (horizontal) prescan pixels on raw image

/**
 * Geometry and electronic information about raw amplifier images
 *
 * Here is a pictorial example showing the meaning of flipX and flipY:
 *
 *    CCD with 4 amps        Desired assembled output      Use these parameters
 *   
 *    --x         x--            y                       
 *   |  amp1    amp2 |           |                               flipX       flipY
 *   y               y           |                       amp1    False       True
 *                               | CCD image             amp2    True        True
 *   y               y           |                       amp3    False       False
 *   |  amp3    amp4 |           |                       amp4    True        False
 *    --x         x--             ----------- x
 *   
 * @note:
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

    bool getFlipX() const;
    void setFlipX(bool doFlipX);  ///< flip row order to make assembled image?

    bool getFlipY() const;
    void setFlipY(bool doFlipY); ///< flip column order to make an assembled image?

    geom::Extent2I getRawXYOffset() const;
    void setRawXYOffset(geom::Extent2I const &xy); ///< offset for assembling a raw CCD image: desired xy0 - raw xy0
    
    //@}

protected:

    AmpInfoRecord(PTR(AmpInfoTable) const & table);

};

/**
 *  @brief Table class that must contain a unique ID field and *** some other things for amps..
 *
 *  @copydetails AmpInfoRecord
 */
class AmpInfoTable : public BaseTable {
public:

    typedef AmpInfoRecord Record;
    typedef ColumnViewT<AmpInfoRecord> ColumnView;
    typedef SortedCatalogT<Record> Catalog;
    typedef SortedCatalogT<Record const> ConstCatalog;

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If null, record IDs will default to zero.
     *
     *  Note that not passing an IdFactory at all will call the other override of make(), which will
     *  set the ID factory to IdFactory::makeSimple().
     */
    static PTR(AmpInfoTable) make(Schema const & schema, PTR(IdFactory) const & idFactory);

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *
     *  This overload sets the ID factory to IdFactory::makeSimple().
     */
    static PTR(AmpInfoTable) make(Schema const & schema) { return make(schema, IdFactory::makeSimple()); }

    /**
     *  @brief Return a minimal schema for AmpInfo tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on AmpInfoRecords will assume that at least the fields
     *  provided by this routine are present.
     */
    static Schema makeMinimalSchema() { return getMinimalSchema().schema; }

    /**
     *  @brief Return true if the given schema is a valid AmpInfoTable schema.
     *  
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(Schema const & other) {
        return other.contains(getMinimalSchema().schema);
    }

    /// @brief Return the object that generates IDs for the table (may be null).
    PTR(IdFactory) getIdFactory() { return _idFactory; }

    /// @brief Return the object that generates IDs for the table (may be null).
    CONST_PTR(IdFactory) getIdFactory() const { return _idFactory; }

    /// @brief Switch to a new IdFactory -- object that generates IDs for the table (may be null).
    void setIdFactory(PTR(IdFactory) f) { _idFactory = f; }

    //@{
    /**
     *  Get keys for standard fields shared by all references.
     *
     *  These keys are used to implement getters and setters on AmpInfoRecord.
     */
    /// @brief Key for the unique ID.
    static Key<RecordId> getIdKey() { return getMinimalSchema().id; }
    static Key<std::string> getNameKey() { return getMinimalSchema().name; }
    static Key< Point<int> > getTrimmedBboxLLKey() { return getMinimalSchema().trimmedbbox_ll; }
    static Key< Point<int> > getTrimmedBboxURKey() { return getMinimalSchema().trimmedbbox_ur; }
    static Key<double> getGainKey() { return getMinimalSchema().gain; }
    static Key<double> getReadNoiseKey() { return getMinimalSchema().readnoise; }
    static Key< Array<double> > getLinearityCoeffsKey() { return getMinimalSchema().linearitycoeffs; }
    static Key<std::string> getLinearityTypeKey() { return getMinimalSchema().linearitytype; }
    static Key<Flag> getHasRawAmplifierKey() { return getMinimalSchema().hasrawamplifier; }
    static Key< Point<int> > getRawBboxLLKey() { return getMinimalSchema().rawbbox_ll; }
    static Key< Point<int> > getRawBboxURKey() { return getMinimalSchema().rawbbox_ur; }
    static Key< Point<int> > getDataBboxLLKey() { return getMinimalSchema().databbox_ll; }
    static Key< Point<int> > getDataBboxURKey() { return getMinimalSchema().databbox_ur; }
    static Key< Point<int> > getHorizontalOverscanBboxLLKey() { return getMinimalSchema().horizontaloverscanbbox_ll; }
    static Key< Point<int> > getHorizontalOverscanBboxURKey() { return getMinimalSchema().horizontaloverscanbbox_ur; }
    static Key< Point<int> > getVerticalOverscanBboxLLKey() { return getMinimalSchema().verticaloverscanbbox_ll; }
    static Key< Point<int> > getVerticalOverscanBboxURKey() { return getMinimalSchema().verticaloverscanbbox_ur; }
    static Key< Point<int> > getPrescanBboxLLKey() { return getMinimalSchema().prescanbbox_ll; }
    static Key< Point<int> > getPrescanBboxURKey() { return getMinimalSchema().prescanbbox_ur; }
    static Key<Flag> getFlipXKey() { return getMinimalSchema().flipx; }
    static Key<Flag> getFlipYKey() { return getMinimalSchema().flipy; }
    static Key< Point<int> > getRawXYOffsetKey() { return getMinimalSchema().rawxyoffset; }
    //@}

    /// @copydoc BaseTable::clone
    PTR(AmpInfoTable) clone() const { return boost::static_pointer_cast<AmpInfoTable>(_clone()); }

    /// @copydoc BaseTable::makeRecord
    PTR(AmpInfoRecord) makeRecord() { return boost::static_pointer_cast<AmpInfoRecord>(_makeRecord()); }

    /// @copydoc BaseTable::copyRecord
    PTR(AmpInfoRecord) copyRecord(BaseRecord const & other) {
        return boost::static_pointer_cast<AmpInfoRecord>(BaseTable::copyRecord(other));
    }

    /// @copydoc BaseTable::copyRecord
    PTR(AmpInfoRecord) copyRecord(BaseRecord const & other, SchemaMapper const & mapper) {
        return boost::static_pointer_cast<AmpInfoRecord>(BaseTable::copyRecord(other, mapper));
    }

protected:

    AmpInfoTable(Schema const & schema, PTR(IdFactory) const & idFactory);

    AmpInfoTable(AmpInfoTable const & other);

private:

    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        Schema schema;
        Key<RecordId> id;
        Key<std::string> name;
        Key< Point<int> > trimmedbbox_ll;
        Key< Point<int> > trimmedbbox_ur;
        Key<double> gain;
        Key<double> readnoise;
        Key< Array<double> > linearitycoeffs;
        Key<std::string> linearitytype;
        Key<Flag> hasrawamplifier;
        Key< Point<int> > rawbbox_ll;
        Key< Point<int> > rawbbox_ur;
        Key< Point<int> > databbox_ll;
        Key< Point<int> > databbox_ur;
        Key< Point<int> > horizontaloverscanbbox_ll;
        Key< Point<int> > horizontaloverscanbbox_ur;
        Key< Point<int> > verticaloverscanbbox_ll;
        Key< Point<int> > verticaloverscanbbox_ur;
        Key< Point<int> > prescanbbox_ll;
        Key< Point<int> > prescanbbox_ur;
        Key<Flag> flipx;
        Key<Flag> flipy;
        Key< Point<int> > rawxyoffset;
        MinimalSchema();
    };
    
    // Return the singleton minimal schema.
    static MinimalSchema & getMinimalSchema();

    friend class io::FitsWriter;

     // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    virtual PTR(io::FitsWriter) makeFitsWriter(fits::Fits * fitsfile, int flags) const;

    PTR(IdFactory) _idFactory;        // generates IDs for new records
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_AmpInfo_h_INCLUDED
