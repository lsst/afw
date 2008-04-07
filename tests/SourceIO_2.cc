// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   SourceIo_1.cc
//! \brief  Testing of IO via the persistence framework for Source and SourceVector.
//
//##====----------------                                ----------------====##/

#include <stdexcept>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <cstring>

#include <boost/cstdint.hpp>

#include <lsst/daf/base.h>
#include <lsst/daf/persistence.h>
#include <lsst/pex/exceptions.h>
#include <lsst/pex/policy/Policy.h>

#include <lsst/afw/detection/Source.h>
#include <lsst/afw/formatters/Utils.h>

using boost::int64_t;

using lsst::daf::base::DataProperty;
using lsst::daf::base::Persistable;
using lsst::pex::policy::Policy;
using lsst::daf::persistence::LogicalLocation;
using lsst::daf::persistence::Persistence;
using lsst::daf::persistence::Storage;

namespace afwFormatters = lsst::afw::formatters;
using namespace lsst::afw::detection;


#define Assert(pred, msg) do { if (!(pred)) { doThrow((msg), __LINE__); } } while(false)

static void doThrow(std::string const & msg, int line) {
    std::ostringstream oss;
    oss << __FILE__ << ':' << line << ":\n" << msg << std::ends;
    throw std::runtime_error(oss.str());
}


static std::string const makeTempFile() {
    char name[32];
    std::strncpy(name, "Source_XXXXXX", 31);
    name[31] = 0;
    int const fd = ::mkstemp(name);
    Assert(fd != -1, "Failed to create temporary file");
    ::close(fd);
    return std::string(name);
}


static void initTestData(SourceVector & v, int sliceId = 0) {
    v.reserve(Source::NUM_NULLABLE_FIELDS + 2);
    for (int i = 0; i < Source::NUM_NULLABLE_FIELDS + 2; ++i) {
        Source data;
        // make sure each field has a different value, and that IO for each nullable field is tested
        int j = i*64;
        data.setId              (0);
        data.setCcdExposureId   (j +  1);
        data.setObjectId        (j +  2);
        data.setMovingObjectId  (j +  3);
        data.setColc            (static_cast<double>(j +  4));
        data.setRowc            (static_cast<double>(j +  5));
        data.setDcol            (static_cast<double>(j +  6));
        data.setDrow            (static_cast<double>(j +  7));
        data.setRa              (static_cast<double>(j +  8));
        data.setDec             (static_cast<double>(j +  9));
        data.setRaErr4detection (static_cast<double>(j + 10));
        data.setDecErr4detection(static_cast<double>(j + 11));
        data.setRaErr4wcs       (static_cast<double>(j + 12));
        data.setDecErr4wcs      (static_cast<double>(j + 13));
        data.setCx              (static_cast<double>(j + 14));
        data.setCy              (static_cast<double>(j + 15));
        data.setCz              (static_cast<double>(j + 16));
        data.setTaiMidPoint     (static_cast<double>(j + 17));
        data.setTaiRange        (static_cast<double>(j + 18));
        data.setFlux            (static_cast<double>(j + 19));
        data.setFluxErr         (static_cast<double>(j + 20));
        data.setPsfMag          (static_cast<double>(j + 21));
        data.setPsfMagErr       (static_cast<double>(j + 22));
        data.setApMag           (static_cast<double>(j + 23));
        data.setApMagErr        (static_cast<double>(j + 24));
        data.setModelMag        (static_cast<double>(j + 25));
        data.setModelMagErr     (static_cast<double>(j + 26));
        data.setColcErr         (static_cast<float> (j + 27));
        data.setRowcErr         (static_cast<float> (j + 28));
        data.setFwhmA           (static_cast<float> (j + 29));
        data.setFwhmB           (static_cast<float> (j + 30));
        data.setFwhmTheta       (static_cast<float> (j + 31));
        data.setApDia           (static_cast<float> (j + 32));
        data.setIxx             (static_cast<float> (j + 33));
        data.setIxxErr          (static_cast<float> (j + 34));
        data.setIyy             (static_cast<float> (j + 35));
        data.setIyyErr          (static_cast<float> (j + 36));
        data.setIxy             (static_cast<float> (j + 37));
        data.setIxyErr          (static_cast<float> (j + 38));
        data.setSnr             (static_cast<float> (j + 39));
        data.setChi2            (static_cast<float> (j + 40));
        data.setScId            (j + 41);
        data.setFlag4association(j + 42);
        data.setFlag4detection  (j + 43);
        data.setFlag4wcs        (j + 44);
        data.setFilterId        (-1);
        data.setDataSource      ('a' + (i & 15));
        if (i < Source::NUM_NULLABLE_FIELDS) {
            data.setNotNull();
            data.setNull(static_cast<Source::NullableField>(i));
        } else if ((i & 1) == 0) {
            data.setNotNull();
        } else {
            data.setNull();
        }
        v.push_back(data);
    }
}


// Make at least a token attempt at generating a unique visit id
// (in-db table name collisions could cause spurious testcase failures)
static int createVisitId() {
    struct timeval tv;
    ::gettimeofday(&tv, 0);
    return static_cast<int>(tv.tv_sec);
}


static void testBoost(void) {
    // Create a blank Policy and DataProperty.
    Policy::Ptr           policy(new Policy);
    DataProperty::PtrType props = DataProperty::createPropertyNode("root");
    int visitId = createVisitId();
    // Not really how ccdExposureId should be set, but good enough for now.
    props->addProperty(DataProperty("visitId",    boost::any(visitId)));
    props->addProperty(DataProperty("exposureId", boost::any(static_cast<int64_t>(visitId)*2)));
    props->addProperty(DataProperty("ccdId",      boost::any(std::string("0"))));
    props->addProperty(DataProperty("sliceId",    boost::any(static_cast<int>(0))));

    // Setup test location
    LogicalLocation loc(makeTempFile());

    // Intialize test data
    Source       ds(0, 0.0, 1.0, 2.0, 3.0);
    SourceVector dsv;
    initTestData(dsv);
    dsv.push_back(ds);


    Persistence::Ptr pers = Persistence::getPersistence(policy);

    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage("BoostStorage", loc));
        pers->persist(dsv, storageList, props);
        Assert(dsv[0].getId() == (static_cast<int64_t>(visitId) << 24) + 1LL, "Source id not changed to expected value");
    }

    // read in data
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage("BoostStorage", loc));
        Persistable::Ptr p = pers->retrieve("SourceVector", storageList, props);
        Assert(p.get() != 0, "Failed to retrieve Persistable");
        SourceVector::Ptr v =
            boost::dynamic_pointer_cast<SourceVector, Persistable>(p);
        Assert(v, "Couldn't cast to SourceVector");
        Assert(*v == dsv, "persist()/retrieve() resulted in SourceVector corruption");
    }
    ::unlink(loc.locString().c_str());
}


static DataProperty::PtrType createDbTestProps(
    int         const   sliceId,
    int         const   numSlices,
    std::string const & itemName
) {
    Assert(sliceId < numSlices && numSlices > 0, "invalid slice parameters");

    DataProperty::PtrType props = DataProperty::createPropertyNode("root");

    if (numSlices > 1) {
        DataProperty::PtrType dias = DataProperty::createPropertyNode("Source");
        dias->addProperty(DataProperty("isPerSliceTable", boost::any(true)));
        dias->addProperty(DataProperty("numSlices",       boost::any(numSlices)));
        props->addProperty(dias);
    }
    int visitId = createVisitId();
    props->addProperty(DataProperty("visitId", visitId));
    props->addProperty(DataProperty("exposureId", boost::any(static_cast<int64_t>(visitId)*2)));
    // Not really how ccdExposureId should be set, but good enough for now.
    props->addProperty(DataProperty("ccdId",    std::string("0")));
    props->addProperty(DataProperty("sliceId",  boost::any(sliceId)));
    props->addProperty(DataProperty("itemName", boost::any(itemName)));
    return props;
}


// comparison operator used to sort Source in id order
struct SourceLessThan {
    bool operator()(Source const & d1, Source const & d2) {
        return d1.getId() < d2.getId();
    }
};


static void testDb(std::string const & storageType) {
    // Create the required Policy and DataProperty
    Policy::Ptr           policy(new Policy);
    DataProperty::PtrType props(createDbTestProps(0, 1, "Source"));

    Persistence::Ptr pers = Persistence::getPersistence(policy);
    LogicalLocation loc("mysql://lsst10.ncsa.uiuc.edu:3306/test");

    // 1. Test on a single Source
    Source ds(0, 0.0, 1.0, 2.0, 3.0);
    SourceVector dsv;
    dsv.push_back(ds);
    int64_t visitId = static_cast<int64_t>(boost::any_cast<int>(props->findUnique("visitId")->getValue()));
    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(dsv, storageList, props);
        Assert(dsv[0].getId() == (visitId << 24) + 1LL,
            "Source id not changed to expected value");
    }
    // and read it back in (in a SourceVector)
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("SourceVector", storageList, props);
        Assert(p != 0, "Failed to retrieve Persistable");
        SourceVector::Ptr v = boost::dynamic_pointer_cast<SourceVector, Persistable>(p);
        Assert(v.get() != 0, "Couldn't cast to SourceVector");
        Assert(v->at(0) == dsv[0], "persist()/retrieve() resulted in SourceVector corruption");
    }
    afwFormatters::dropAllVisitSliceTables(loc, policy, props);

    // 2. Test on a SourceVector
    dsv.clear();
    initTestData(dsv);
    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(dsv, storageList, props);
        int i = 1;
        for (SourceVector::iterator it = dsv.begin();
             it != dsv.end(); ++it) {
            Assert(it->getId() == (visitId << 24) + i,
                "Source id in vector not changed to expected value");
            ++i;
        }
    }
    // and read it back in
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("SourceVector", storageList, props);
        Assert(p != 0, "Failed to retrieve Persistable");
        SourceVector::Ptr v = boost::dynamic_pointer_cast<SourceVector, Persistable>(p);
        Assert(v.get() != 0, "Couldn't cast to SourceVector");
        // sort in ascending id order (database does not give any ordering guarantees
        // in the absence of an ORDER BY clause)
        std::sort(v->begin(), v->end(), SourceLessThan());
        Assert(v.get() != &dsv && *v == dsv, "persist()/retrieve() resulted in SourceVector corruption");
    }
    afwFormatters::dropAllVisitSliceTables(loc, policy, props);
}


int main(int const argc, char const * const * const argv) {
    try {
        testBoost();
        if (lsst::daf::persistence::DbAuth::available()) {
            testDb("DbStorage");
            testDb("DbTsvStorage");
        }
        if (lsst::daf::base::Citizen::census(0) == 0) {
            std::clog << "No leaks detected" << std::endl;
        } else {
            Assert(false, "Detected memory leaks");
        }
        return EXIT_SUCCESS;
    } catch (lsst::pex::exceptions::ExceptionStack & exs) {
        std::clog << exs.what() << exs.getStack()->toString("...", true) << std::endl;
    } catch (std::exception & ex) {
        std::clog << ex.what() << std::endl;
    }

    if (lsst::daf::base::Citizen::census(0) != 0) {
        std::clog << "Leaked memory blocks:" << std::endl;
        lsst::daf::base::Citizen::census(std::clog);
    }

    return EXIT_FAILURE;
}
