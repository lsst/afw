// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   DiaSourceFormatters.cc
//! \brief  Implementation of persistence for DiaSource instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include <lsst/pex/exceptions.h>
#include <lsst/daf/persistence/BoostStorage.h>
#include <lsst/daf/persistence/DbStorage.h>
#include <lsst/daf/persistence/DbTsvStorage.h>
#include <lsst/daf/persistence/FormatterImpl.h>

#include <boost/any.hpp>
#include <boost/format.hpp>

#include <lsst/afw/detection/DiaSource.h>
#include <lsst/afw/formatters/DiaSourceFormatters.h>
#include <lsst/afw/formatters/Utils.h>

using namespace lsst::afw::formatters;
namespace ex = lsst::pex::exceptions;


// -- DiaSourceVectorFormatter ----------------

DiaSourceVectorFormatter::DiaSourceVectorFormatter(Policy::Ptr const & policy) :
    Formatter(typeid(*this)),
    _policy(policy)
{}


DiaSourceVectorFormatter::~DiaSourceVectorFormatter() {}


Formatter::Ptr DiaSourceVectorFormatter::createInstance(Policy::Ptr policy) {
    return Formatter::Ptr(new DiaSourceVectorFormatter(policy));
}


FormatterRegistration DiaSourceVectorFormatter::registration(
    "DiaSourceVector",
    typeid(DiaSourceVector),
    createInstance
);


/*!
    \internal   Generates a unique identifier for a DiaSource given the id of the
                originating visit, the id of the originating ccd, and the sequence
                number of the DiaSource within that slice.
 */
inline static int64_t generateDiaSourceId(unsigned short seqNum, int ccdId, int64_t visitId) {
    return (visitId << 24) + (ccdId << 16) + seqNum;
}


/*!
    Inserts a single DiaSource into a database table using \a db
    (an instance of lsst::daf::persitence::DbStorage or subclass thereof).
 */
template <typename T>
void DiaSourceVectorFormatter::insertRow(T & db, DiaSource const & d) {

    db.template setColumn<int64_t>("diaSourceId",      d._diaSourceId);
    db.template setColumn<int64_t>("ccdExposureId",    d._ccdExposureId);
    db.template setColumn<double> ("colc",             d._colc);
    db.template setColumn<double> ("rowc",             d._rowc);
    db.template setColumn<double> ("dcol",             d._dcol);
    db.template setColumn<double> ("drow",             d._drow);
    db.template setColumn<double> ("ra",               d._ra);
    db.template setColumn<double> ("decl",             d._decl);
    db.template setColumn<double> ("raErr4detection",  d._raErr4detection);
    db.template setColumn<double> ("decErr4detection", d._decErr4detection);
    db.template setColumn<double> ("cx",               d._cx);
    db.template setColumn<double> ("cy",               d._cy);
    db.template setColumn<double> ("cz",               d._cz);
    db.template setColumn<double> ("taiMidPoint",      d._taiMidPoint);
    db.template setColumn<double> ("taiRange",         d._taiRange);
    db.template setColumn<double> ("flux",             d._flux);
    db.template setColumn<double> ("fluxErr",          d._fluxErr);
    db.template setColumn<double> ("psfMag",           d._psfMag);
    db.template setColumn<double> ("psfMagErr",        d._psfMagErr);
    db.template setColumn<double> ("apMag",            d._apMag);
    db.template setColumn<double> ("apMagErr",         d._apMagErr);
    db.template setColumn<double> ("modelMag",         d._modelMag);
    db.template setColumn<double> ("modelMagErr",      d._modelMagErr);
    db.template setColumn<float>  ("colcErr",          d._colcErr);
    db.template setColumn<float>  ("rowcErr",          d._rowcErr);
    db.template setColumn<float>  ("fwhmA",            d._fwhmA);
    db.template setColumn<float>  ("fwhmB",            d._fwhmB);
    db.template setColumn<float>  ("fwhmTheta",        d._fwhmTheta);
    db.template setColumn<float>  ("snr",              d._snr);
    db.template setColumn<float>  ("chi2",             d._chi2);
    db.template setColumn<int32_t>("scId",             d._scId);
    db.template setColumn<char>   ("filterId",         static_cast<char>(d._filterId));
    db.template setColumn<char>   ("_dataSource",      static_cast<char>(d._dataSource));

    if (!d.isNull(DiaSource::OBJECT_ID)) {
        db.template setColumn<int64_t>("objectId", d._objectId);
    } else {
        db.setColumnToNull("objectId");
    }
    if (!d.isNull(DiaSource::MOVING_OBJECT_ID)) {
        db.template setColumn<int64_t>("movingObjectId", d._movingObjectId);
    } else {
        db.setColumnToNull("movingObjectId");
    }
    if (!d.isNull(DiaSource::RA_ERR_4_WCS)) {
        db.template setColumn<double>("raErr4wcs", d._raErr4wcs);
    } else {
        db.setColumnToNull("raErr4wcs");
    }
    if (!d.isNull(DiaSource::DEC_ERR_4_WCS)) {
        db.template setColumn<double>("decErr4wcs", d._decErr4wcs);
    } else {
        db.setColumnToNull("decErr4wcs");
    }
    if (!d.isNull(DiaSource::AP_DIA)) {
        db.template setColumn<float>("apDia", d._apDia);
    } else {
        db.setColumnToNull("apDia");
    }
    if (!d.isNull(DiaSource::IXX)) {
        db.template setColumn<float>("Ixx", d._ixx);
    } else {
        db.setColumnToNull("Ixx");
    }
    if (!d.isNull(DiaSource::IXX_ERR)) {
        db.template setColumn<float>("IxxErr", d._ixxErr);
    } else {
        db.setColumnToNull("IxxErr");
    }
    if (!d.isNull(DiaSource::IYY)) {
        db.template setColumn<float>("Iyy", d._iyy);
    } else {
        db.setColumnToNull("Iyy");
    }
    if (!d.isNull(DiaSource::IYY_ERR)) {
        db.template setColumn<float>("IyyErr", d._iyyErr);
    } else {
        db.setColumnToNull("IyyErr");
    }
    if (!d.isNull(DiaSource::IXY)) {
        db.template setColumn<float>("Ixy", d._ixy);
    } else {
        db.setColumnToNull("Ixy");
    }
    if (!d.isNull(DiaSource::IXY_ERR)) {
        db.template setColumn<float>("IxyErr", d._ixyErr);
    } else {
        db.setColumnToNull("IxyErr");
    }
    if (!d.isNull(DiaSource::FLAG_4_ASSOCIATION)) {
        db.template setColumn<int16_t>("flag4association", d._flag4association);
    } else {
        db.setColumnToNull("flag4association");
    }
    if (!d.isNull(DiaSource::FLAG_4_DETECTION)) {
        db.template setColumn<int16_t>("flag4detection", d._flag4detection);
    } else {
        db.setColumnToNull("flag4detection");
    }
    if (!d.isNull(DiaSource::FLAG_4_WCS)) {
        db.template setColumn<int16_t>("flag4wcs", d._flag4wcs);
    } else {
        db.setColumnToNull("flag4wcs");
    }
    db.insertRow();
}

//! \cond
template void DiaSourceVectorFormatter::insertRow<DbStorage>   (DbStorage &,    DiaSource const &);
template void DiaSourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, DiaSource const &);
//! \endcond


/*! Prepares for reading DiaSource instances from a database table. */
void DiaSourceVectorFormatter::setupFetch(DbStorage & db, DiaSource & d) {
    db.outParam("diaSourceId",      &(d._diaSourceId));
    db.outParam("ccdExposureId",    &(d._ccdExposureId));
    db.outParam("filterId",         reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",         &(d._objectId));
    db.outParam("movingObjectId",   &(d._movingObjectId));
    db.outParam("scId",             &(d._scId));
    db.outParam("colc",             &(d._colc));
    db.outParam("colcErr",          &(d._colcErr));
    db.outParam("rowc",             &(d._rowc));
    db.outParam("rowcErr",          &(d._rowcErr));
    db.outParam("dcol",             &(d._dcol));
    db.outParam("drow",             &(d._drow));
    db.outParam("ra",               &(d._ra));
    db.outParam("decl",             &(d._decl));
    db.outParam("raErr4detection",  &(d._raErr4detection));
    db.outParam("decErr4detection", &(d._decErr4detection));
    db.outParam("raErr4wcs",        &(d._raErr4wcs));
    db.outParam("decErr4wcs",       &(d._decErr4wcs));
    db.outParam("cx",               &(d._cx));
    db.outParam("cy",               &(d._cy));
    db.outParam("cz",               &(d._cz));
    db.outParam("taiMidPoint",      &(d._taiMidPoint));
    db.outParam("taiRange",         &(d._taiRange));
    db.outParam("fwhmA",            &(d._fwhmA));
    db.outParam("fwhmB",            &(d._fwhmB));
    db.outParam("fwhmTheta",        &(d._fwhmTheta));
    db.outParam("flux",             &(d._flux));
    db.outParam("fluxErr",          &(d._fluxErr));
    db.outParam("psfMag",           &(d._psfMag));
    db.outParam("psfMagErr",        &(d._psfMagErr));
    db.outParam("apMag",            &(d._apMag));
    db.outParam("apMagErr",         &(d._apMagErr));
    db.outParam("modelMag",         &(d._modelMag));
    db.outParam("modelMagErr",      &(d._modelMagErr));
    db.outParam("apDia",            &(d._apDia));
    db.outParam("Ixx",              &(d._ixx));
    db.outParam("IxxErr",           &(d._ixxErr));
    db.outParam("Iyy",              &(d._iyy));
    db.outParam("IyyErr",           &(d._iyyErr));
    db.outParam("Ixy",              &(d._ixy));
    db.outParam("IxyErr",           &(d._ixyErr));
    db.outParam("snr",              &(d._snr));
    db.outParam("chi2",             &(d._chi2));
    db.outParam("flag4association", &(d._flag4association));
    db.outParam("flag4detection",   &(d._flag4detection));
    db.outParam("flag4wcs",         &(d._flag4wcs));
    db.outParam("_dataSource",      reinterpret_cast<char *>(&(d._dataSource)));
}


template <class Archive>
void DiaSourceVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const version,
    Persistable *      persistable
) {
    DiaSourceVector * p = dynamic_cast<DiaSourceVector *>(persistable);
    archive & boost::serialization::base_object<Persistable>(*p);
    DiaSourceVector::size_type sz;

    if (Archive::is_loading::value) {
        DiaSource data;
        archive & sz;
        p->reserve(sz);
        for (; sz > 0; --sz) {
            archive & data;
            p->push_back(data);
        }
    } else {
        sz = p->size();
        archive & sz;
        DiaSourceVector::iterator const end(p->end());
        for (DiaSourceVector::iterator i = p->begin(); i != end; ++i) {
            archive & *i;
        }
    }
}

template void DiaSourceVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void DiaSourceVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);
//template void DiaSourceVectorFormatter::delegateSerialize<boost::archive::binary_oarchive>(
//    boost::archive::binary_oarchive &, unsigned int const, Persistable *
//);
//template void DiaSourceVectorFormatter::delegateSerialize<boost::archive::binary_iarchive>(
//    boost::archive::binary_iarchive &, unsigned int const, Persistable *
//);


void DiaSourceVectorFormatter::write(
    Persistable const *   persistable,
    Storage::Ptr          storage,
    DataProperty::PtrType additionalData
) {
    if (persistable == 0) {
        throw ex::InvalidParameter("No Persistable provided");
    }
    if (!storage) {
        throw ex::InvalidParameter("No Storage provided");
    }

    DiaSourceVector const * p = dynamic_cast<DiaSourceVector const *>(persistable);
    if (p == 0) {
        throw ex::Runtime("Persistable was not of concrete type DiaSourceVector");
    }

    // Assume all have ids or none do.
    if (p->begin()->_diaSourceId == 0 &&
        (!_policy || !_policy->exists("GenerateIds") ||
         _policy->getBool("GenerateIds"))) {
        DiaSourceVector* v    = const_cast<DiaSourceVector*>(p);
        unsigned short seq    = 1;
        int64_t visitId       = extractVisitId(additionalData);
        int64_t ccdExposureId = extractCcdExposureId(additionalData);
        int     ccdId         = extractCcdId(additionalData);
        if (ccdId < 0 || ccdId >= 256) {
            throw ex::InvalidParameter("ccdId out of range");
        }
        for (DiaSourceVector::iterator i = v->begin(); i != v->end(); ++i) {
            i->_diaSourceId    = generateDiaSourceId(seq, ccdId, visitId);
            i->_ccdExposureId  = ccdExposureId;
            ++seq;
            if (seq == 0) { // Overflowed
                throw ex::Runtime("Too many DiaSources");
            }
        }
    }

    if (typeid(*storage) == typeid(BoostStorage)) {
        BoostStorage * bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw ex::Runtime("Didn't get BoostStorage");
        }
        bs->getOArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        std::string itemName(getItemName(additionalData));
        std::string name(getVisitSliceTableName(_policy, additionalData));
        std::string model = extractPolicyString(
            _policy,
            itemName + ".templateTableName",
            itemName + "Template"
        );
        bool mayExist = !extractOptionalFlag(additionalData, itemName + ".isPerSliceTable");
        if (typeid(*storage) == typeid(DbStorage)) {
            DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            DiaSourceVector::const_iterator const end(p->end());
            for (DiaSourceVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbStorage>(*db, *i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            DiaSourceVector::const_iterator const end(p->end());
            for (DiaSourceVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbTsvStorage>(*db, *i);
            }
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported"); 
    }
}


Persistable* DiaSourceVectorFormatter::read(
    Storage::Ptr          storage,
    DataProperty::PtrType additionalData
) {
    std::auto_ptr<DiaSourceVector> p(new DiaSourceVector);

    if (typeid(*storage) == typeid(BoostStorage)) {
        BoostStorage* bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw ex::Runtime("Didn't get BoostStorage");
        }
        bs->getIArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
        if (db == 0) {
            throw ex::Runtime("Didn't get DbStorage");
        }
        std::vector<std::string> tables;
        getAllVisitSliceTableNames(tables, _policy, additionalData);

        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator const end = tables.end();
        for (std::vector<std::string>::const_iterator i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            DiaSource data;
            setupFetch(*db, data);
            db->query();
            data.setNotNull();
            while (db->next()) {
                if (db->columnIsNull( 0)) { throw ex::Runtime("null column \"diaSourceId\"");      }
                if (db->columnIsNull( 1)) { throw ex::Runtime("null column \"exposureId\"");       }
                if (db->columnIsNull( 2)) { throw ex::Runtime("null column \"filterId\"");         }
                if (db->columnIsNull( 3)) { data.setNull(DiaSource::OBJECT_ID);                    }
                if (db->columnIsNull( 4)) { data.setNull(DiaSource::MOVING_OBJECT_ID);             }
                if (db->columnIsNull( 5)) { throw ex::Runtime("null column \"scId\"");             }
                if (db->columnIsNull( 6)) { throw ex::Runtime("null column \"colc\"");             }
                if (db->columnIsNull( 7)) { throw ex::Runtime("null column \"colcErr\"");          }
                if (db->columnIsNull( 8)) { throw ex::Runtime("null column \"rowc\"");             }
                if (db->columnIsNull( 9)) { throw ex::Runtime("null column \"rowcErr\"");          }
                if (db->columnIsNull(10)) { throw ex::Runtime("null column \"dcol\"");             }
                if (db->columnIsNull(11)) { throw ex::Runtime("null column \"drow\"");             }
                if (db->columnIsNull(12)) { throw ex::Runtime("null column \"ra\"");               }
                if (db->columnIsNull(13)) { throw ex::Runtime("null column \"decl\"");             }
                if (db->columnIsNull(14)) { throw ex::Runtime("null column \"raErr4detection\"");  }
                if (db->columnIsNull(15)) { throw ex::Runtime("null column \"decErr4detection\""); }
                if (db->columnIsNull(16)) { data.setNull(DiaSource::RA_ERR_4_WCS);                 }
                if (db->columnIsNull(17)) { data.setNull(DiaSource::DEC_ERR_4_WCS);                }
                if (db->columnIsNull(18)) { throw ex::Runtime("null column \"cx\"");               }
                if (db->columnIsNull(19)) { throw ex::Runtime("null column \"cy\"");               }
                if (db->columnIsNull(20)) { throw ex::Runtime("null column \"cz\"");               }
                if (db->columnIsNull(21)) { throw ex::Runtime("null column \"taiMidPoint\"");      }
                if (db->columnIsNull(22)) { throw ex::Runtime("null column \"taiRange\"");         }
                if (db->columnIsNull(23)) { throw ex::Runtime("null column \"fwhmA\"");            }
                if (db->columnIsNull(24)) { throw ex::Runtime("null column \"fwhmB\"");            }
                if (db->columnIsNull(25)) { throw ex::Runtime("null column \"fwhmTheta\"");        }
                if (db->columnIsNull(26)) { throw ex::Runtime("null column \"flux\"");             }
                if (db->columnIsNull(27)) { throw ex::Runtime("null column \"fluxErr\"");          }
                if (db->columnIsNull(28)) { throw ex::Runtime("null column \"psfMag\"");           }
                if (db->columnIsNull(29)) { throw ex::Runtime("null column \"psfMagErr\"");        }
                if (db->columnIsNull(30)) { throw ex::Runtime("null column \"apMag\"");            }
                if (db->columnIsNull(31)) { throw ex::Runtime("null column \"apMagErr\"");         }
                if (db->columnIsNull(32)) { throw ex::Runtime("null column \"modelMag\"");         }
                if (db->columnIsNull(33)) { throw ex::Runtime("null column \"modelMagErr\"");      }
                if (db->columnIsNull(34)) { data.setNull(DiaSource::AP_DIA);                       }
                if (db->columnIsNull(35)) { data.setNull(DiaSource::IXX);                          }
                if (db->columnIsNull(36)) { data.setNull(DiaSource::IXX_ERR);                      }
                if (db->columnIsNull(37)) { data.setNull(DiaSource::IYY);                          }
                if (db->columnIsNull(38)) { data.setNull(DiaSource::IYY_ERR);                      }
                if (db->columnIsNull(39)) { data.setNull(DiaSource::IXY);                          }
                if (db->columnIsNull(40)) { data.setNull(DiaSource::IXY_ERR);                      }
                if (db->columnIsNull(41)) { throw ex::Runtime("null column \"snr\"");              }
                if (db->columnIsNull(42)) { throw ex::Runtime("null column \"chi2\"");             }
                if (db->columnIsNull(43)) { data.setNull(DiaSource::FLAG_4_ASSOCIATION);           }
                if (db->columnIsNull(44)) { data.setNull(DiaSource::FLAG_4_DETECTION);             }
                if (db->columnIsNull(45)) { data.setNull(DiaSource::FLAG_4_WCS);                   }
                if (db->columnIsNull(46)) { throw ex::Runtime("null column \"_dataSource\"");      }
                p->push_back(data);
                data.setNotNull(); // clear out null markers
            }
            db->finishQuery();
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported");
    }
    return p.release();
}


void DiaSourceVectorFormatter::update(Persistable*, Storage::Ptr, DataProperty::PtrType) {
    throw ex::Runtime("DiaSourceVectorFormatter: updates not supported");
}
