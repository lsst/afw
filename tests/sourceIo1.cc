// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Testing of IO via the persistence framework for Source and SourceSet.
//
//##====----------------                                ----------------====##/

#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include <stdexcept>


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SourceIO

#include "boost/test/unit_test.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/formatters/Utils.h"

using lsst::daf::base::PropertySet;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::LogicalLocation;
using lsst::daf::persistence::Persistence;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;

namespace afwFormatters = lsst::afw::formatters;
using namespace lsst::afw::detection;


#define Assert(pred, msg) do { if (!(pred)) { throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, (msg)); } } while (false)


static std::string const makeTempFile() {
    char name[32];
    std::strncpy(name, "Source_XXXXXX", 31);
    name[31] = 0;
    int const fd = ::mkstemp(name);
    Assert(fd != -1, "Failed to create temporary file");
    ::close(fd);
    return std::string(name);
}


static void initTestData(SourceSet & v, int sliceId = 0) {
    v.clear();
    v.reserve(NUM_SOURCE_NULLABLE_FIELDS + 2);
    Source data;
    for (int i = 0; i != NUM_SOURCE_NULLABLE_FIELDS + 2; ++i) {
        // make sure each field (other than angles) has a different value, and that IO for
        // each nullable field is tested.  Note: Source ids are generated in ascending order
        int j = i*NUM_SOURCE_NULLABLE_FIELDS;
        data.setSourceId(j + sliceId*(NUM_SOURCE_NULLABLE_FIELDS + 2)*66 + 1);
        data.setAmpExposureId(static_cast<int64_t>(j +  2));
        data.setFilterId(-1);
        data.setObjectId(static_cast<int64_t>(j +  4));
        data.setMovingObjectId(static_cast<int64_t>(j +  5));
        data.setProcHistoryId(-1);
        data.setRa(0);
        data.setRaErrForDetection(0);
        data.setRaErrForWcs(0);
        data.setDec(0);
        data.setDecErrForDetection(0);
        data.setDecErrForWcs(0);
        data.setXFlux(static_cast<double>(j + 14));
        data.setXFluxErr(static_cast<float>(j + 15));
        data.setYFlux(static_cast<double>(j + 16));
        data.setYFluxErr(static_cast<float>(j + 17));
        data.setRaFlux(0);
        data.setRaFluxErr(0);
        data.setDecFlux(0);
        data.setDecFluxErr(0);
        data.setXPeak(static_cast<double>(j + 22));
        data.setYPeak(static_cast<double>(j + 23));
        data.setRaPeak(0);
        data.setDecPeak(0);
        data.setXAstrom(static_cast<double>(j + 26));
        data.setXAstromErr(static_cast<float>(j + 27));
        data.setYAstrom(static_cast<double>(j + 28));
        data.setYAstromErr(static_cast<float>(j + 29));        
        data.setRaAstrom(0);
        data.setRaAstromErr(0);
        data.setDecAstrom(0);
        data.setDecAstromErr(0);                
        data.setTaiMidPoint(static_cast<double>(j + 34));
        data.setTaiRange(static_cast<double>(j + 35));
        data.setPsfFlux(static_cast<double>(j + 39));
        data.setPsfFluxErr(static_cast<float>(j + 40));
        data.setApFlux(static_cast<double>(j + 41));
        data.setApFluxErr(static_cast<float>(j + 42));
        data.setModelFlux(static_cast<double>(j + 43));
        data.setModelFluxErr(static_cast<float>(j + 44));
        data.setPetroFlux(static_cast<double>(j + 45));
        data.setPetroFluxErr(static_cast<float>(j + 46));
        data.setInstFlux(static_cast<double>(j + 47));
        data.setInstFluxErr(static_cast<float>(j + 48));
        data.setNonGrayCorrFlux(static_cast<double>(j + 49));
        data.setNonGrayCorrFluxErr(static_cast<float>(j + 50));
        data.setAtmCorrFlux(static_cast<double>(j + 51));
        data.setAtmCorrFluxErr(static_cast<float>(j + 52));
        data.setApDia(static_cast<float>(j + 53));
        data.setSnr(static_cast<float>(j + 54));
        data.setChi2(static_cast<float>(j + 55));
        data.setSky(static_cast<float>(j + 56));
        data.setSkyErr(static_cast<float>(j + 57));
        data.setRaObject(0);
        data.setDecObject(0);
        data.setFlagForAssociation(1);
        data.setFlagForDetection(2);
        data.setFlagForWcs(3);

        if (i < NUM_SOURCE_NULLABLE_FIELDS) {
            data.setNotNull();
            data.setNull(i);
        } else if ((i & 1) == 0) {
            data.setNotNull();
        } else {
            data.setNull();
        }
        lsst::afw::geom::ellipses::Separable<
            lsst::afw::geom::ellipses::Distortion, 
            lsst::afw::geom::ellipses::DeterminantRadius
        > core(0,0, i);
        lsst::afw::geom::Point2D point(data.getRa(), data.getDec());
        lsst::afw::geom::ellipses::Ellipse ellipse(core, point);
        Footprint::Ptr fp(new Footprint(ellipse));
        data.setFootprint(fp);

        Source::Ptr sourcePtr(new Source(data));        
        v.push_back(sourcePtr);
    }
}

// Make at least a token attempt at generating a unique visit id
// (in-db table name collisions could cause spurious testcase failures)
static int createVisitId() {
    struct timeval tv;
    ::gettimeofday(&tv, 0);
    return abs(static_cast<int>(tv.tv_sec));
}

static PropertySet::Ptr createDbTestProps(
    int const sliceId,
    int const numSlices,
    std::string const & itemName
) {
    Assert(sliceId < numSlices && numSlices > 0, "invalid slice parameters");

    PropertySet::Ptr props(new PropertySet);

    if (numSlices > 1) {
        props->add("Source.isPerSliceTable", true);
        props->add("Source.numSlices", numSlices);
    }
    int visitId = createVisitId();
    props->add("visitId",  visitId);
    props->add("exposureId", static_cast<int64_t>(visitId*2));
    props->add("ampExposureId", static_cast<int64_t>(visitId)*32);
    props->add("universeSize", numSlices);
    props->add("sliceId",  sliceId);
    props->add("itemName", itemName);
    return props;
}

static void testBoost(void) {
    // Create a blank Policy and PropertySet.
    Policy::Ptr      policy(new Policy);
    PropertySet::Ptr props = createDbTestProps(0,1,"Source");
    props->add("doFootprints", true);
    // Setup test location
    LogicalLocation loc(makeTempFile());

    // Intialize test data
    SourceSet dsv;
    initTestData(dsv);
    PersistableSourceVector::Ptr persistPtr(new PersistableSourceVector(dsv));    
    Persistence::Ptr pers = Persistence::getPersistence(policy);

    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage("BoostStorage", loc));
        pers->persist(*persistPtr, storageList, props);
    }

    // read in data
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage("BoostStorage", loc));
        Persistable::Ptr p = pers->retrieve("PersistableSourceVector", 
            storageList, props);
        BOOST_CHECK_MESSAGE(p.get() != 0, "Failed to retrieve Persistable");
        PersistableSourceVector::Ptr persistVec =
           boost::dynamic_pointer_cast<PersistableSourceVector, Persistable>(p);
        BOOST_CHECK_MESSAGE(persistVec.get() != 0, 
            "Couldn't cast to PersistableSourceVector");
        BOOST_CHECK_MESSAGE(*persistVec == dsv, 
            "persist()/retrieve() resulted in PersistableSourceVector corruption");

        for(int i =0; i < dsv.size(); ++i) {
            boost::shared_ptr<const Footprint> dsvFp = dsv[i]->getFootprint();
            boost::shared_ptr<const Footprint> persistFp = dsv[i]->getFootprint();
            for(int j = 0; j < dsvFp->getSpans().size(); ++j) {
                BOOST_CHECK_EQUAL(dsvFp->getSpans()[j]->getY(), persistFp->getSpans()[j]->getY());
                BOOST_CHECK_EQUAL(dsvFp->getSpans()[j]->getX0(), persistFp->getSpans()[j]->getX0());
                BOOST_CHECK_EQUAL(dsvFp->getSpans()[j]->getX1(), persistFp->getSpans()[j]->getX1());
            }
            BOOST_CHECK_EQUAL(dsvFp->getArea(), persistFp->getArea());
            BOOST_CHECK_EQUAL(dsvFp->getArea(), persistFp->getArea());
            lsst::afw::geom::BoxI dsvBBox = dsvFp->getBBox();
            lsst::afw::geom::BoxI persistBBox = persistFp->getBBox();
            BOOST_CHECK_EQUAL(dsvBBox.getMinX(),persistBBox.getMinX());
            BOOST_CHECK_EQUAL(dsvBBox.getMinY(),persistBBox.getMinY());
            BOOST_CHECK_EQUAL(dsvBBox.getMaxX(),persistBBox.getMaxX());
            BOOST_CHECK_EQUAL(dsvBBox.getMaxY(),persistBBox.getMaxY());
            dsvBBox = dsvFp->getRegion();
            persistBBox = persistFp->getRegion();
            BOOST_CHECK_EQUAL(dsvBBox.getMinX(),persistBBox.getMinX());
            BOOST_CHECK_EQUAL(dsvBBox.getMinY(),persistBBox.getMinY());
            BOOST_CHECK_EQUAL(dsvBBox.getMaxX(),persistBBox.getMaxX());
            BOOST_CHECK_EQUAL(dsvBBox.getMaxY(),persistBBox.getMaxY());
        }
    }
    ::unlink(loc.locString().c_str());
}





// comparison operator used to sort Source in id order
struct SourceLessThan {
    bool operator()(Source::Ptr const & d1, Source::Ptr const & d2) {
        return d1->getId() < d2->getId();
    }
};


static void testDb(std::string const & storageType) {
    // Create the required Policy and PropertySet
    Policy::Ptr policy(new Policy);
    std::string policyRoot("Formatter.PersistableSourceVector");
    policy->set(policyRoot + ".Source.templateTableName", "Source");
    policy->set(policyRoot + ".Source.tableNamePattern", "_tmp_test_Source_%(visitId)");
    Policy::Ptr nested(policy->getPolicy(policyRoot));

    PropertySet::Ptr props = createDbTestProps(0, 1, "Source");
    Persistence::Ptr pers = Persistence::getPersistence(policy);
    LogicalLocation loc("mysql://lsst10.ncsa.uiuc.edu:3306/test_source_v2");
    
    // 1. Test on a single Source
    Source::Ptr ds(new Source());
    ds->setId(2);
    SourceSet dsv;
    dsv.push_back(ds);
    PersistableSourceVector::Ptr persistPtr(new PersistableSourceVector(dsv));
    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(*persistPtr, storageList, props);
    }
    // and read it back in (in a SourceSet)
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("PersistableSourceVector", 
            storageList, props);
        BOOST_CHECK_MESSAGE(p.get() != 0, "Failed to retrieve Persistable");
        PersistableSourceVector::Ptr persistVec = 
            boost::dynamic_pointer_cast<PersistableSourceVector, Persistable>(p);
        BOOST_CHECK_MESSAGE(persistVec.get() != 0, 
            "Couldn't cast to PersistableSourceVector");
        SourceSet vec = persistVec->getSources();
        BOOST_CHECK_MESSAGE(*vec.at(0) == *dsv[0], 
            "persist()/retrieve() resulted in corruption");
    }
    afwFormatters::dropAllSliceTables(loc, nested, props);

    // 2. Test on a SourceSet
    dsv.clear();
    initTestData(dsv);
    persistPtr->setSources(dsv);
    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(*persistPtr, storageList, props);
    }
    // and read it back in
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("PersistableSourceVector", 
            storageList, props);
        BOOST_CHECK_MESSAGE(p.get() != 0, 
            "Failed to retrieve Persistable");
        PersistableSourceVector::Ptr persistVec = 
            boost::dynamic_pointer_cast<PersistableSourceVector, Persistable>(p);
        BOOST_CHECK_MESSAGE(persistVec.get() != 0, 
            "Couldn't cast to PersistableSourceVector");
        SourceSet vec(persistVec->getSources());
        
        // sort in ascending id order 
        // (database does not give any ordering guarantees
        // in the absence of an ORDER BY clause)
        std::sort(vec.begin(), vec.end(), SourceLessThan());
        BOOST_CHECK_MESSAGE(vec.size() == dsv.size(),
            "persist()/retrieve() resulted in corruption");
    
        for (size_t i =0; i<vec.size();i++){
            if (*vec[i] != *dsv[i]){
                BOOST_ERROR("persist()/retrieve() resulted in corruption");
                break;
            }
        }
    }
    afwFormatters::dropAllSliceTables(loc, nested, props);
}


static void testDb2(std::string const & storageType) {
    // Create the required Policy and PropertySet
    Policy::Ptr policy(new Policy);
    std::string policyRoot("Formatter.PersistableSourceVector");
    policy->set(policyRoot + ".Source.templateTableName", "Source");
    policy->set(policyRoot + ".Source.tableNamePattern", "_tmp_test_Source_v%(visitId)_s%(sliceId)");

    Policy::Ptr nested(policy->getPolicy(policyRoot));

    Persistence::Ptr pers = Persistence::getPersistence(policy);
    LogicalLocation loc("mysql://lsst10.ncsa.uiuc.edu:3306/test_source_v2");

    SourceSet all;
    int const numSlices = 3;
    PropertySet::Ptr props = createDbTestProps(0, numSlices, "Source");

    // 1. Write out each slice table seperately
    for (int sliceId = 0; sliceId < numSlices; ++sliceId) {
        props->set("sliceId", sliceId);
        props->set("ampExposureId", static_cast<int64_t>(sliceId));
        SourceSet dsv;
        initTestData(dsv, sliceId);

        all.insert(all.end(), dsv.begin(), dsv.end());
        PersistableSourceVector persistVec;
        persistVec.setSources(dsv);
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(persistVec, storageList, props);
    }

    // 2. Read in all slice tables - simulates association pipeline
    //    gathering the results of numSlices image processing pipeline slices
    Storage::List storageList;
    storageList.push_back(pers->getRetrieveStorage(storageType, loc));
    Persistable::Ptr p = pers->retrieve("PersistableSourceVector", 
        storageList, props);
    BOOST_CHECK_MESSAGE(p.get() != 0, "Failed to retrieve Persistable");
    PersistableSourceVector::Ptr persistPtr = 
        boost::dynamic_pointer_cast<PersistableSourceVector, Persistable>(p);
    BOOST_CHECK_MESSAGE(persistPtr.get() != 0, 
        "Couldn't cast to PersistableSourceVector");
    
    // sort in ascending id order 
    // (database does not give any ordering guarantees
    // in the absence of an ORDER BY clause)
    SourceSet vec = persistPtr->getSources();
    std::sort(vec.begin(), vec.end(), SourceLessThan());
    BOOST_CHECK_MESSAGE(vec.size() == all.size(),
        "persist()/retrieve() resulted in corruption");

    for (size_t i =0; i< vec.size();i++){
        if (*vec[i] != *all[i]){
            BOOST_ERROR("persist()/retrieve() resulted in corruption");
            break;
        }            
    }    
    afwFormatters::dropAllSliceTables(loc, nested, props);
}

BOOST_AUTO_TEST_CASE(SourceEquality) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    Source::Ptr a(new Source), b(new Source);
    a->setId(3);
    BOOST_CHECK_MESSAGE(*a != *b && *b != *a, "field equality fails");
    b->setId(3);
    BOOST_CHECK_MESSAGE(*a == *b && *b == *a, "field equality fails");
    a->setNotNull(lsst::afw::detection::MOVING_OBJECT_ID);
    BOOST_CHECK_MESSAGE(*a != *b && *b != *a, "field equality fails");    
    a->setMovingObjectId(5);
    BOOST_CHECK_MESSAGE(*a != *b && *b != *a, "field equality fails");    
    b->setNotNull(lsst::afw::detection::MOVING_OBJECT_ID);
    BOOST_CHECK_MESSAGE(*a != *b && *b != *a, "field equality fails");    
    b->setMovingObjectId(5);    
    BOOST_CHECK_MESSAGE(*a == *b && *b == *a, "field equality fails");
    
    SourceSet av, bv;
    av.push_back(a);
    PersistableSourceVector apv(av);
    BOOST_CHECK(apv.getSources()[0]->isNull(0) == a->isNull(0));
    BOOST_CHECK(apv.getSources()[0]->isNull(1) == a->isNull(1));
    bv.push_back(b);
    
}

BOOST_AUTO_TEST_CASE(SourceIO) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    try {
        testBoost();
        if (lsst::daf::persistence::DbAuth::available("lsst10.ncsa.uiuc.edu", "3306")) {
            testDb("DbStorage");
            testDb("DbTsvStorage");
            testDb2("DbStorage");
            testDb2("DbTsvStorage");
        }
        else BOOST_TEST_MESSAGE("Skipping DB tests");
        BOOST_CHECK_MESSAGE(lsst::daf::base::Citizen::census(0) == 0, 
            "Detected memory leaks");
    } catch(std::exception const & ex) {
        BOOST_FAIL(ex.what());
    }
}
