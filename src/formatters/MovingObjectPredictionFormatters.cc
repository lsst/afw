// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   MovingObjectPredictionFormatters.cc
//! \brief  Implementation of persistence for MovingObjectPrediction instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include <lsst/mwi/exceptions.h>
#include <lsst/mwi/persistence/BoostStorage.h>
#include <lsst/mwi/persistence/DbStorage.h>
#include <lsst/mwi/persistence/DbTsvStorage.h>
#include <lsst/mwi/persistence/FormatterImpl.h>

#include <boost/any.hpp>
#include <boost/format.hpp>
//#include <boost/serialization/export.hpp>

#include "lsst/fw/MovingObjectPrediction.h"
#include "lsst/fw/formatters/MovingObjectPredictionFormatters.h"
#include "lsst/fw/formatters/Utils.h"


//BOOST_CLASS_EXPORT(lsst::fw::MovingObjectPrediction)
//BOOST_CLASS_EXPORT(lsst::fw::MovingObjectPredictionVector)


namespace lsst {
namespace fw {
namespace formatters {

namespace ex = lsst::mwi::exceptions;


// -- MovingObjectPredictionVectorFormatter ----------------

MovingObjectPredictionVectorFormatter::MovingObjectPredictionVectorFormatter(Policy::Ptr const & policy) :
    Formatter(typeid(*this)),
    _policy(policy)
{}


MovingObjectPredictionVectorFormatter::~MovingObjectPredictionVectorFormatter() {}


FormatterRegistration MovingObjectPredictionVectorFormatter::registration(
    "MovingObjectPredictionVector",
    typeid(MovingObjectPredictionVector),
    createInstance
);


Formatter::Ptr MovingObjectPredictionVectorFormatter::createInstance(Policy::Ptr policy) {
    return Formatter::Ptr(new MovingObjectPredictionVectorFormatter(policy));
}


template <typename T>
void MovingObjectPredictionVectorFormatter::insertRow(T & db, MovingObjectPrediction const & p) {
    db.template setColumn<int64_t>("orbit_id", p._orbitId);
    db.template setColumn<double> ("ra_deg",   p._ra);
    db.template setColumn<double> ("dec_deg",  p._dec);
    db.template setColumn<double> ("mjd",      p._mjd);
    db.template setColumn<double> ("smia",     p._smia);
    db.template setColumn<double> ("smaa",     p._smaa);
    db.template setColumn<double> ("pa",       p._pa);
    db.template setColumn<double> ("mag",      p._mag);
    db.insertRow();
}

//! \cond
template void MovingObjectPredictionVectorFormatter::insertRow<DbStorage>   (DbStorage &,    MovingObjectPrediction const &);
template void MovingObjectPredictionVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, MovingObjectPrediction const &);
//! \endcond

void MovingObjectPredictionVectorFormatter::setupFetch(DbStorage & db, MovingObjectPrediction & p) {
    db.outParam("orbit_id", &(p._orbitId));
    db.outParam("ra_deg",   &(p._ra));
    db.outParam("dec_deg",  &(p._dec));
    db.outParam("mjd",      &(p._mjd));
    db.outParam("smia",     &(p._smia));
    db.outParam("smaa",     &(p._smaa));
    db.outParam("pa",       &(p._pa));
    db.outParam("mag",      &(p._mag));
}


template <class Archive>
void MovingObjectPredictionVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const version,
    Persistable *      persistable
) {
    MovingObjectPredictionVector * p = dynamic_cast<MovingObjectPredictionVector *>(persistable);
    archive & boost::serialization::base_object<Persistable>(*p);
    MovingObjectPredictionVector::size_type sz;

    if (Archive::is_loading::value) {
        MovingObjectPrediction data;
        archive & sz;
        p->reserve(sz);
        for (; sz > 0; --sz) {
            archive & data;
            p->push_back(data);
        }
    } else {
        sz = p->size();
        archive & sz;
        MovingObjectPredictionVector::iterator const end(p->end());
        for (MovingObjectPredictionVector::iterator i = p->begin(); i != end; ++i) {
            archive & *i;
        }
    }
}

template void MovingObjectPredictionVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void MovingObjectPredictionVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);
//template void MovingObjectPredictionVectorFormatter::delegateSerialize<boost::archive::binary_oarchive>(
//    boost::archive::binary_oarchive &, unsigned int const, Persistable *
//);
//template void MovingObjectPredictionVectorFormatter::delegateSerialize<boost::archive::binary_iarchive>(
//    boost::archive::binary_iarchive &, unsigned int const, Persistable *
//);


void MovingObjectPredictionVectorFormatter::write(
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

    MovingObjectPredictionVector const * p = dynamic_cast<MovingObjectPredictionVector const *>(persistable);
    if (p == 0) {
        throw ex::Runtime("Persistable was not of concrete type MovingObjectPredictionVector");
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
            MovingObjectPredictionVector::const_iterator const end(p->end());
            for (MovingObjectPredictionVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbStorage>(*db, *i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            MovingObjectPredictionVector::const_iterator const end(p->end());
            for (MovingObjectPredictionVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbTsvStorage>(*db, *i);
            }
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported"); 
    }
}


Persistable* MovingObjectPredictionVectorFormatter::read(
    Storage::Ptr          storage,
    DataProperty::PtrType additionalData
) {
    std::auto_ptr<MovingObjectPredictionVector> p(new MovingObjectPredictionVector);

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
            MovingObjectPrediction data;
            setupFetch(*db, data);
            db->query();
            while (db->next()) {
                if (db->columnIsNull(0)) { throw ex::Runtime("null column \"orbit_id\""); }
                if (db->columnIsNull(1)) { throw ex::Runtime("null column \"ra_deg\"");   }
                if (db->columnIsNull(2)) { throw ex::Runtime("null column \"dec_deg\"");  }
                if (db->columnIsNull(3)) { throw ex::Runtime("null column \"mjd\"");      }
                if (db->columnIsNull(4)) { throw ex::Runtime("null column \"smia\"");     }
                if (db->columnIsNull(5)) { throw ex::Runtime("null column \"smaa\"");     }
                if (db->columnIsNull(6)) { throw ex::Runtime("null column \"pa\"");       }
                if (db->columnIsNull(7)) { throw ex::Runtime("null column \"mag\"");      }
                p->push_back(data);
            }
            db->finishQuery();
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported");
    }
    return p.release();
}


void MovingObjectPredictionVectorFormatter::update(Persistable*, Storage::Ptr, DataProperty::PtrType) {
    throw ex::Runtime("MovingObjectPredictionVectorFormatter: updates not supported");
}


}}} // end of namespace lsst::fw::formatters

