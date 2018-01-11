// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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
#ifndef AFW_DETECTION_Peak_h_INCLUDED
#define AFW_DETECTION_Peak_h_INCLUDED

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/IdFactory.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/BaseColumnView.h"

namespace lsst {
namespace afw {
namespace detection {

class PeakRecord;
class PeakTable;

/**
 *  Record class that represents a peak in a Footprint
 */
class PeakRecord : public afw::table::BaseRecord {
public:
    typedef PeakTable Table;
    typedef afw::table::ColumnViewT<PeakRecord> ColumnView;
    typedef afw::table::CatalogT<PeakRecord> Catalog;
    typedef afw::table::CatalogT<PeakRecord const> ConstCatalog;

    virtual ~PeakRecord() = default;
    PeakRecord(PeakRecord const&) = delete;
    PeakRecord(PeakRecord&&) = delete;
    PeakRecord& operator=(PeakRecord const&) = delete;
    PeakRecord& operator=(PeakRecord&&) = delete;

    std::shared_ptr<PeakTable const> getTable() const {
        return std::static_pointer_cast<PeakTable const>(afw::table::BaseRecord::getTable());
    }

    //@{
    /// Convenience accessors for the keys in the minimal schema.
    afw::table::RecordId getId() const;
    void setId(afw::table::RecordId id);

    int getIx() const;
    int getIy() const;
    void setIx(int ix);
    void setIy(int iy);
    afw::geom::Point2I getI() const { return afw::geom::Point2I(getIx(), getIy()); }
    afw::geom::Point2I getCentroid(bool) const { return getI(); }

    float getFx() const;
    float getFy() const;
    void setFx(float fx);
    void setFy(float fy);
    afw::geom::Point2D getF() const { return afw::geom::Point2D(getFx(), getFy()); }
    afw::geom::Point2D getCentroid() const { return getF(); }

    float getPeakValue() const;
    void setPeakValue(float peakValue);
    //@}

protected:
    explicit PeakRecord(std::shared_ptr<PeakTable> const& table);

private:
    friend class PeakTable;
};

/**
 *  Table class for Peaks in Footprints.
 */
class PeakTable : public afw::table::BaseTable {
public:
    typedef PeakRecord Record;
    typedef afw::table::ColumnViewT<PeakRecord> ColumnView;
    typedef afw::table::CatalogT<Record> Catalog;
    typedef afw::table::CatalogT<Record const> ConstCatalog;

    virtual ~PeakTable();
    PeakTable& operator=(PeakTable const&) = delete;
    PeakTable& operator=(PeakTable&&) = delete;

    /**
     *  Obtain a table that can be used to create records with given schema
     *
     *  @param[in] schema     Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] forceNew   If true, guarantee that the returned PeakTable will be a new one, rather
     *                        than attempting to reuse an existing PeakTable with the same Schema.
     *
     *  If a PeakTable already exists that uses this Schema, that PeakTable will be returned instead
     *  of creating a new one.  This is different from how most Record/Table classes work, but it is
     *  an important memory optimization for Peaks, for which we expect to have very few distinct
     *  Schemas as well as many catalogs (one per Footprint) with a small number of Peaks; we don't want
     *  to have a different PeakTable for each one of those catalogs if they all share the same Schema.
     *  This behavior can be disabled by setting forceNewTable=true or by cloning an existing table
     *  (in both of these cases, the new table will not be reused in the future, either)
     */
    static std::shared_ptr<PeakTable> make(afw::table::Schema const& schema, bool forceNew = false);

    /**
     *  Return a minimal schema for Peak tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on PeakRecords will assume that at least the fields
     *  provided by this routine are present.
     */
    static afw::table::Schema makeMinimalSchema() { return getMinimalSchema().schema; }

    /**
     *  Return true if the given schema is a valid PeakTable schema.
     *
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(afw::table::Schema const& other) {
        return other.contains(getMinimalSchema().schema);
    }

    /// Return the object that generates IDs for the table (may be null).
    std::shared_ptr<afw::table::IdFactory> getIdFactory() { return _idFactory; }

    /// Return the object that generates IDs for the table (may be null).
    std::shared_ptr<afw::table::IdFactory const> getIdFactory() const { return _idFactory; }

    /// Switch to a new IdFactory -- object that generates IDs for the table (may be null).
    void setIdFactory(std::shared_ptr<afw::table::IdFactory> f) { _idFactory = f; }

    //@{
    /**
     *  Get keys for standard fields shared by all peaks.
     *
     *  These keys are used to implement getters and setters on PeakRecord.
     */
    static afw::table::Key<afw::table::RecordId> getIdKey() { return getMinimalSchema().id; }
    static afw::table::Key<int> getIxKey() { return getMinimalSchema().ix; }
    static afw::table::Key<int> getIyKey() { return getMinimalSchema().iy; }
    static afw::table::Key<float> getFxKey() { return getMinimalSchema().fx; }
    static afw::table::Key<float> getFyKey() { return getMinimalSchema().fy; }
    static afw::table::Key<float> getPeakValueKey() { return getMinimalSchema().peakValue; }
    //@}

    /// @copydoc table::BaseTable::clone
    std::shared_ptr<PeakTable> clone() const { return std::static_pointer_cast<PeakTable>(_clone()); }

    /// @copydoc table::BaseTable::makeRecord
    std::shared_ptr<PeakRecord> makeRecord() { return std::static_pointer_cast<PeakRecord>(_makeRecord()); }

    /// @copydoc table::BaseTable::copyRecord
    std::shared_ptr<PeakRecord> copyRecord(afw::table::BaseRecord const& other) {
        return std::static_pointer_cast<PeakRecord>(afw::table::BaseTable::copyRecord(other));
    }

    /// @copydoc table::BaseTable::copyRecord
    std::shared_ptr<PeakRecord> copyRecord(afw::table::BaseRecord const& other,
                                           afw::table::SchemaMapper const& mapper) {
        return std::static_pointer_cast<PeakRecord>(afw::table::BaseTable::copyRecord(other, mapper));
    }

protected:
    PeakTable(afw::table::Schema const& schema, std::shared_ptr<afw::table::IdFactory> const& idFactory);

    PeakTable(PeakTable const& other);
    PeakTable(PeakTable&& other);

    std::shared_ptr<afw::table::BaseTable> _clone() const override;

    std::shared_ptr<afw::table::BaseRecord> _makeRecord() override;

private:
    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        afw::table::Schema schema;
        afw::table::Key<afw::table::RecordId> id;
        afw::table::Key<float> fx;
        afw::table::Key<float> fy;
        afw::table::Key<int> ix;
        afw::table::Key<int> iy;
        afw::table::Key<float> peakValue;

        MinimalSchema();
    };

    // Return the singleton minimal schema.
    static MinimalSchema& getMinimalSchema();

    friend class afw::table::io::FitsWriter;

    // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    std::shared_ptr<afw::table::io::FitsWriter> makeFitsWriter(fits::Fits* fitsfile,
                                                               int flags) const override;

    std::shared_ptr<afw::table::IdFactory> _idFactory;  // generates IDs for new records
};

std::ostream& operator<<(std::ostream& os, PeakRecord const& record);

inline afw::table::RecordId PeakRecord::getId() const { return get(PeakTable::getIdKey()); }
inline void PeakRecord::setId(afw::table::RecordId id) { set(PeakTable::getIdKey(), id); }

inline int PeakRecord::getIx() const { return get(PeakTable::getIxKey()); }
inline int PeakRecord::getIy() const { return get(PeakTable::getIyKey()); }
inline void PeakRecord::setIx(int ix) { set(PeakTable::getIxKey(), ix); }
inline void PeakRecord::setIy(int iy) { set(PeakTable::getIyKey(), iy); }

inline float PeakRecord::getFx() const { return get(PeakTable::getFxKey()); }
inline float PeakRecord::getFy() const { return get(PeakTable::getFyKey()); }
inline void PeakRecord::setFx(float fx) { set(PeakTable::getFxKey(), fx); }
inline void PeakRecord::setFy(float fy) { set(PeakTable::getFyKey(), fy); }

inline float PeakRecord::getPeakValue() const { return get(PeakTable::getPeakValueKey()); }
inline void PeakRecord::setPeakValue(float peakValue) { set(PeakTable::getPeakValueKey(), peakValue); }

typedef afw::table::ColumnViewT<PeakRecord> PeakColumnView;
typedef afw::table::CatalogT<PeakRecord> PeakCatalog;
typedef afw::table::CatalogT<PeakRecord const> ConstPeakCatalog;
}
}
}  // namespace lsst::afw::detection

#endif  // !AFW_DETECTION_Peak_h_INCLUDED
