// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Testing of IO via the persistence framework for DiaSource and DiaSourceSet.
//
//##====----------------                                ----------------====##/

#include <stdexcept>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <cstring>


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SourceIO

#include "boost/test/unit_test.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"

#include "lsst/afw/detection/DiaSource.h"
#include "lsst/afw/formatters/Utils.h"

using lsst::daf::base::PropertySet;
using lsst::daf::base::Persistable;
using lsst::pex::policy::Policy;
using lsst::daf::persistence::LogicalLocation;
using lsst::daf::persistence::Persistence;
using lsst::daf::persistence::Storage;
using lsst::afw::detection::DiaSource;
using lsst::afw::detection::DiaSourceSet;
using lsst::afw::detection::PersistableDiaSourceVector;

namespace afwFormatters = lsst::afw::formatters;
namespace afwDet = lsst::afw::detection;

#define Assert(pred, msg) do { if (!(pred)) { throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, (msg)); } } while (false)


static std::string const makeTempFile() {
    char name[32];
    std::strncpy(name, "DiaSource_XXXXXX", 31);
    name[31] = 0;
    int const fd = ::mkstemp(name);
    Assert(fd != -1, "Failed to create temporary file");
    ::close(fd);
    return std::string(name);
}

//not all fields are defined in DC3a
//this array is a hack to facilitate testing, and nothing else
//use this until DC3b when fields are reintroduced to schema
int DiaSourceFieldsToTest[] = {
        afwDet::OBJECT_ID,
        afwDet::MOVING_OBJECT_ID,
        afwDet::RA_ERR_FOR_DETECTION,
        afwDet::DEC_ERR_FOR_DETECTION,
        afwDet::X_ASTROM_ERR,
        afwDet::Y_ASTROM_ERR,
        afwDet::AP_DIA,
        afwDet::IXX,
        afwDet::IXX_ERR,
        afwDet::IYY,
        afwDet::IYY_ERR,
        afwDet::IXY,
        afwDet::IXY_ERR,
        afwDet::FLAG_FOR_DETECTION,   
        afwDet::DIA_SOURCE_TO_ID,
        afwDet::RA_ERR_FOR_WCS,
        afwDet::DEC_ERR_FOR_WCS,
        afwDet::PSF_FLUX_ERR,
        afwDet::AP_FLUX_ERR,
        afwDet::MODEL_FLUX_ERR,
        afwDet::INST_FLUX_ERR,
        afwDet::FLAG_CLASSIFICATION,        
};
int const NUM_FIELDS = 22;

static void initTestData(DiaSourceSet & v, int sliceId = 0) {
    v.clear();
    v.reserve(NUM_FIELDS + 2);
    DiaSource data;
    for (int i = 0; i != NUM_FIELDS + 2; ++i) {
        // make sure each field has a different value, and that IO for each nullable field is tested
        int j = i*NUM_FIELDS;
        data.setNull();
        data.setDiaSourceId(j + sliceId*(NUM_FIELDS + 2)*64 + 1);
        data.setAmpExposureId(static_cast<int64_t>(j + 2));
        data.setDiaSourceToId(j + sliceId*(NUM_FIELDS + 2)*64 + 3);
        data.setFilterId(-1);
        data.setObjectId(static_cast<int64_t>(j + 4));
        data.setMovingObjectId(static_cast<int64_t>(j + 5));
        data.setRa(static_cast<double>(j + 8));
        data.setRaErrForDetection(static_cast<float>(j + 9));
        data.setRaErrForWcs(static_cast<float>(j + 10));
        data.setDec(static_cast<double>(j + 11));
        data.setDecErrForDetection(static_cast<float>(j + 12));
        data.setDecErrForWcs(static_cast<float>(j + 13));
        data.setXAstrom(static_cast<double>(j + 26));
        data.setXAstromErr(static_cast<float>(j + 27));
        data.setYAstrom(static_cast<double>(j + 28));
        data.setYAstromErr(static_cast<float>(j + 29));                   
        data.setTaiMidPoint(static_cast<double>(j + 34));
        data.setTaiRange(static_cast<double>(j + 35));
        data.setPsfFlux(static_cast<double>(j + 39));
        data.setPsfFluxErr(static_cast<float>(j + 40));
        data.setApFlux(static_cast<double>(j + 41));
        data.setApFluxErr(static_cast<float>(j + 42));
        data.setModelFlux(static_cast<double>(j + 43));
        data.setModelFluxErr(static_cast<float>(j + 44));
        data.setInstFlux(static_cast<double>(j + 47));
        data.setInstFluxErr(static_cast<float>(j + 48));
        data.setApDia(static_cast<float>(j + 53));
        data.setIxx(static_cast<float> (j + 53));
        data.setIxxErr(static_cast<float> (j + 53));
        data.setIyy(static_cast<float> (j + 53));
        data.setIyyErr(static_cast<float> (j + 53));
        data.setIxy(static_cast<float> (j + 53));
        data.setIxyErr(static_cast<float> (j + 53));        
        data.setSnr(static_cast<float>(j + 54));
        data.setChi2(static_cast<float>(j + 55));
        data.setFlagClassification(4);
        data.setFlagForDetection(2);
       
        #if 0
        //The following are not defined in DC3a Schema
        data.setProcHistoryId(-1);
        data.setScId(j + 6);
        data.setSsmId(static_cast<int64_t>(j + 7));       
        data.setXFlux(static_cast<double>(j + 14));
        data.setXFluxErr(static_cast<float>(j + 15));
        data.setYFlux(static_cast<double>(j + 16));
        data.setYFluxErr(static_cast<float>(j + 17));
        data.setRaFlux(static_cast<double>(j + 18));
        data.setRaFluxErr(static_cast<float>(j + 19));
        data.setDecFlux(static_cast<double>(j + 20));
        data.setDecFluxErr(static_cast<float>(j + 21));
        data.setXPeak(static_cast<double>(j + 22));
        data.setYPeak(static_cast<double>(j + 23));
        data.setRaPeak(static_cast<double>(j + 24));
        data.setDecPeak(static_cast<double>(j + 25));
        data.setRaAstrom(static_cast<double>(j + 30));
        data.setRaAstromErr(static_cast<float>(j + 31));
        data.setDecAstrom(static_cast<double>(j + 32));
        data.setDecAstromErr(static_cast<float>(j + 33));     
        data.setLengthDeg(static_cast<double>(j + 39));       
        data.setNonGrayCorrFlux(static_cast<double>(j + 49));
        data.setNonGrayCorrFluxErr(static_cast<float>(j + 50));
        data.setAtmCorrFlux(static_cast<double>(j + 51));
        data.setAtmCorrFluxErr(static_cast<float>(j + 52));
        data.setRefFlux(static_cast<float> (j + 53));
        data.setValX1(static_cast<float>(j + 56));
        data.setValX2(static_cast<float>(j + 57));
        data.setValY1(static_cast<float>(j + 58));
        data.setValY2(static_cast<float>(j + 59));
        data.setValXY(static_cast<float>(j + 60));                
        data.setObsCode(1);
        data.setIsSynthetic(2);
        data.setMopsStatus(3);
        data.setFlagForAssociation(1);
        data.setFlagForWcs(3);        
        #endif
        
        if (i < NUM_FIELDS) {
            data.setNull(DiaSourceFieldsToTest[i]);
        } else if ((i & 1) == 0) {
            for(int j =0; j < NUM_FIELDS; j++)
                data.setNotNull(DiaSourceFieldsToTest[j]);
        } else {
            for(int j =0; j < NUM_FIELDS; j++)
                data.setNull(DiaSourceFieldsToTest[j]);
        }
        
        DiaSource::Ptr sourcePtr(new DiaSource(data));
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
    int const sliceId,  int const numSlices,
    std::string const & itemName
) {
    Assert(sliceId < numSlices && numSlices > 0, "invalid slice parameters");

    PropertySet::Ptr props(new PropertySet);

    if (numSlices > 1) {
        props->add("DIASource.isPerSliceTable", true);
        props->add("DIASource.numSlices", numSlices);
    }
    int visitId = createVisitId();
    props->add("visitId",  visitId);
    props->add("exposureId", static_cast<int64_t>(visitId)*2);
    props->add("ampExposureId", static_cast<int64_t>(visitId)*32);
    props->add("universeSize", numSlices);
    props->add("sliceId",  sliceId);
    props->add("itemName", itemName);
    return props;
}

static void testBoost(void) {
    // Create a blank Policy and PropertySet.
    Policy::Ptr      policy(new Policy);
    PropertySet::Ptr props= createDbTestProps(0,1,"DIASource");

    // Setup test location
    LogicalLocation loc(makeTempFile());

    // Intialize test data
    DiaSourceSet dsv;
    initTestData(dsv);
    PersistableDiaSourceVector::Ptr persistPtr(
        new PersistableDiaSourceVector(dsv)
    );    
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
        Persistable::Ptr p = pers->retrieve(
            "PersistableDiaSourceVector", storageList, props);
        BOOST_CHECK_MESSAGE(p.get() != 0, "Failed to retrieve Persistable");
        PersistableDiaSourceVector::Ptr persistVec =
            boost::dynamic_pointer_cast<PersistableDiaSourceVector, Persistable>(p);
        BOOST_CHECK_MESSAGE(persistVec.get() != 0, 
            "Couldn't cast to PersistableDiaSourceVector");
        BOOST_CHECK_MESSAGE(*persistVec == dsv, 
            "persist()/retrieve() resulted in corruption");
    }
    ::unlink(loc.locString().c_str());
}

// comparison operator used to sort DiaSource in id order
struct SourceLessThan {
    bool operator()(DiaSource::Ptr const & d1, DiaSource::Ptr const & d2) {
        return d1->getId() < d2->getId();
    }
};


static void testDb(std::string const & storageType) {
    // Create the required Policy and DataProperty
    Policy::Ptr policy(new Policy);
    std::string policyRoot = "Formatter.PersistableDiaSourceVector";
    policy->set(policyRoot + ".DIASource.templateTableName", "DIASource");
    policy->set(policyRoot + ".DIASource.tableNamePattern", "_tmp_test_DIASource_%(visitId)");
    Policy::Ptr nested = policy->getPolicy(policyRoot);
    PropertySet::Ptr props = createDbTestProps(0, 1, "DIASource");

    Persistence::Ptr pers = Persistence::getPersistence(policy);
    LogicalLocation loc("mysql://lsst10.ncsa.uiuc.edu:3306/test_diasource");

    // 1. Test on a single DiaSource
    DiaSource::Ptr ds(new DiaSource);
    ds->setId(2);
    DiaSourceSet dsv;
    dsv.push_back(ds);
    PersistableDiaSourceVector::Ptr persistPtr(
        new PersistableDiaSourceVector(dsv)
    );

    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(*persistPtr, storageList, props);
    }
    // and read it back in (in a DiaSourceSet)
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("PersistableDiaSourceVector", 
            storageList, props);
        BOOST_CHECK_MESSAGE(p != 0, "Failed to retrieve Persistable");
        PersistableDiaSourceVector::Ptr persistVec = 
            boost::dynamic_pointer_cast<PersistableDiaSourceVector, Persistable>(p);
        BOOST_CHECK_MESSAGE(persistVec.get() != 0, 
            "Couldn't cast to PersistableDiaSourceVector");
        DiaSourceSet vec = persistVec->getSources();
        BOOST_CHECK_MESSAGE(*vec.at(0) == *dsv[0], 
            "persist()/retrieve() resulted in corruption");
    }
    afwFormatters::dropAllSliceTables(loc, nested, props);

    // 2. Test on a DiaSourceSet
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
        Persistable::Ptr p = pers->retrieve("PersistableDiaSourceVector", 
            storageList, props);
        BOOST_CHECK_MESSAGE(p != 0, "Failed to retrieve Persistable");
        PersistableDiaSourceVector::Ptr persistVec = 
            boost::dynamic_pointer_cast<PersistableDiaSourceVector, Persistable>(p);
        BOOST_CHECK_MESSAGE(persistVec.get() != 0, 
            "Couldn't cast to PersistableDiaSourceVector");
        DiaSourceSet vec(persistVec->getSources());

        // sort in ascending id order 
        // (database does not give any ordering guarantees
        // in the absence of an ORDER BY clause)
        std::sort(vec.begin(), vec.end(), SourceLessThan());
        BOOST_CHECK_MESSAGE(vec.size() == dsv.size(), 
            "persist()/retrieve() resulted in corruption");
    
        for (size_t i =0; i<vec.size();i++){
            if (*vec[i]!=*dsv[i])
                BOOST_ERROR("persist()/retrieve() resulted in corruption");
        }
    }
    afwFormatters::dropAllSliceTables(loc, nested, props);
}

BOOST_AUTO_TEST_CASE(DiaSourceEquality) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    DiaSource::Ptr a(new DiaSource), b(new DiaSource);
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
    
    DiaSourceSet av, bv;
    av.push_back(a);
    PersistableDiaSourceVector apv(av);
    BOOST_CHECK(apv.getSources()[0]->isNull(0) == a->isNull(0));
    BOOST_CHECK(apv.getSources()[0]->isNull(1) == a->isNull(1));
}

BOOST_AUTO_TEST_CASE(DiaSourceIO) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    try {
        testBoost();
        if (lsst::daf::persistence::DbAuth::available("lsst10.ncsa.uiuc.edu", "3306")) {
            BOOST_TEST_MESSAGE("Skipping DB tests");
            testDb("DbStorage");
            testDb("DbTsvStorage");
        }
        else BOOST_TEST_MESSAGE("Skipping DB tests");
        BOOST_CHECK_MESSAGE(lsst::daf::base::Citizen::census(0) == 0, 
            "Detected memory leaks");
    } catch (std::exception & ex) {
        BOOST_FAIL(ex.what());
    }
}
