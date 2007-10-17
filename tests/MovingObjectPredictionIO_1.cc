// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   MovingObjectPredictionIO_1.cc
//! \brief  Testing of IO via the persistence framework for
//!         MovingObjectPrediction and MovingObjectPredictionVector.
//
//##====----------------                                ----------------====##/

#include <sys/time.h>
#include <cstdlib>

#include <boost/cstdint.hpp>

#include <lsst/mwi/exceptions.h>
#include <lsst/mwi/data/DataProperty.h>
#include <lsst/mwi/data/SupportFactory.h>
#include <lsst/mwi/policy/Policy.h>
#include <lsst/mwi/persistence/Persistence.h>
#include <lsst/mwi/persistence/LogicalLocation.h>

#include "lsst/fw/MovingObjectPrediction.h"
#include "lsst/fw/formatters/Utils.h"

#include <stdexcept>

using boost::int64_t;

using lsst::mwi::data::DataProperty;
using lsst::mwi::data::SupportFactory;
using lsst::mwi::policy::Policy;
using lsst::mwi::persistence::LogicalLocation;
using lsst::mwi::persistence::Persistence;
using lsst::mwi::persistence::Persistable;
using lsst::mwi::persistence::Storage;

using namespace lsst::fw;


#define Assert(pred, msg) do { if (!(pred)) { doThrow((msg), __LINE__); } } while(false)

static void doThrow(std::string const & msg, int line) {
    std::ostringstream oss;
    oss << __FILE__ << ':' << line << ":\n" << msg << std::ends;
    throw std::runtime_error(oss.str());
}


static std::string const makeTempFile() {
    char name[64];
    std::strncpy(name, "MovingObjectPrediction_XXXXXX", 63);
    name[63] = 0;
    int const fd = ::mkstemp(name);
    Assert(fd != -1, "Failed to create temporary file");
    ::close(fd);
    return std::string(name);
}


static void initTestData(MovingObjectPredictionVector & v, int sliceId = 0) {
    v.reserve(8);
    for (int i = 0; i < 8; ++i) {
        MovingObjectPrediction data;
        // make sure each field has a different value
        // Note: MovingObjectPrediction ids are generated in ascending order
        int j = i*16;
        data.setId                 (j + sliceId*8*16);
        data.setRa                 (static_cast<double>(j + 1));
        data.setDec                (static_cast<double>(j + 2));
        data.setSemiMinorAxisLength(static_cast<double>(j + 3));
        data.setSemiMajorAxisLength(static_cast<double>(j + 4));
        data.setPositionAngle      (static_cast<double>(j + 5));
        data.setMjd                (static_cast<double>(j + 6));
        data.setMagnitude          (static_cast<double>(j + 7));        
        v.push_back(data);
    }
}


static void testBoost(void) {
    // Create a blank Policy and DataProperty.
    Policy::Ptr           policy(new Policy);
    DataProperty::PtrType props = SupportFactory::createPropertyNode("root");

    // Setup test location
    LogicalLocation loc(makeTempFile());

    // Intialize test data
    MovingObjectPrediction       mop;
    MovingObjectPredictionVector mopv;

    mop.setId(1003);
    mop.setRa(30.1);
    mop.setDec(-85.3305);
    mop.setPositionAngle(-5.0314);
    mop.setSemiMajorAxisLength(0.77);
    mop.setSemiMinorAxisLength(0.40);
    initTestData(mopv);
    mopv.push_back(mop);

    Persistence::Ptr pers = Persistence::getPersistence(policy);

    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage("BoostStorage", loc));
        pers->persist(mopv, storageList, props);
    }

    // read in data
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage("BoostStorage", loc));
        Persistable::Ptr p = pers->retrieve("MovingObjectPredictionVector", storageList, props);
        Assert(p.get() != 0, "Failed to retrieve Persistable");
        MovingObjectPredictionVector::Ptr v =
            boost::dynamic_pointer_cast<MovingObjectPredictionVector, Persistable>(p);
        Assert(v, "Couldn't cast to MovingObjectPredictionVector");
        Assert(*v == mopv, "persist()/retrieve() resulted in MovingObjectPredictionVector corruption");
    }
    ::unlink(loc.locString().c_str());
}


// Make at least a token attempt at generating a unique visit id
// (in-db table name collisions could cause spurious testcase failures)
static int64_t createVisitId() {
    struct timeval tv;
    ::gettimeofday(&tv, 0);
    return static_cast<int64_t>(tv.tv_sec)*1000000 + static_cast<int64_t>(tv.tv_usec);
}


static DataProperty::PtrType createDbTestProps(
    int         const   sliceId,
    int         const   numSlices,
    std::string const & itemName
) {
    Assert(sliceId < numSlices && numSlices > 0, "invalid slice parameters");

    DataProperty::PtrType props = SupportFactory::createPropertyNode("root");

    if (numSlices > 1) {
        DataProperty::PtrType dias = SupportFactory::createPropertyNode("MovingObjectPrediction");
        dias->addProperty(DataProperty("isPerSliceTable", boost::any(true)));
        dias->addProperty(DataProperty("numSlices",       boost::any(numSlices)));
        props->addProperty(dias);
    }
    props->addProperty(DataProperty("visitId", createVisitId()));
    props->addProperty(DataProperty("sliceId", boost::any(sliceId)));
    props->addProperty(DataProperty("itemName", boost::any(itemName)));
    return props;
}


// comparison operator used to sort MovingObjectPrediction in id order
struct MovingObjectPredictionLessThan {
    bool operator()(MovingObjectPrediction const & d1, MovingObjectPrediction const & d2) {
        return d1.getId() < d2.getId();
    }
};


static void testDb(std::string const & storageType) {
    // Create the required Policy and DataProperty
    Policy::Ptr           policy(new Policy);
    DataProperty::PtrType props(createDbTestProps(0, 1, "MovingObjectPrediction"));

    Persistence::Ptr pers = Persistence::getPersistence(policy);
    LogicalLocation loc("mysql://test:globular!test@lsst10.ncsa.uiuc.edu:3306/test");

    // 1. Test on a single MovingObjectPrediction
    MovingObjectPrediction mop;
    MovingObjectPredictionVector mopv;
    mop.setId(13);
    mop.setRa(360.0);
    mop.setDec(-85.0);
    mop.setPositionAngle(35.0);
    mop.setSemiMajorAxisLength(1.0);
    mop.setSemiMinorAxisLength(0.5); 
    mopv.push_back(mop);

    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(mopv, storageList, props);
    }
    // and read it back in
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("MovingObjectPredictionVector", storageList, props);
        Assert(p != 0, "Failed to retrieve Persistable");
        MovingObjectPredictionVector::Ptr d = 
            boost::dynamic_pointer_cast<MovingObjectPredictionVector, Persistable>(p);
        Assert(d.get() != 0, "Couldn't cast to MovingObjectPredictionVector");
        Assert(d->at(0) == mop, "persist()/retrieve() resulted in MovingObjectPredictionVector corruption");
    }
    formatters::dropAllVisitSliceTables(loc, policy, props);

    // 2. Test on multiple MovingObjectPredictions
    mopv.clear();
    initTestData(mopv);
    // write out data
    {
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(mopv, storageList, props);
    }
    // and read it back in
    {
        Storage::List storageList;
        storageList.push_back(pers->getRetrieveStorage(storageType, loc));
        Persistable::Ptr p = pers->retrieve("MovingObjectPredictionVector", storageList, props);
        Assert(p != 0, "Failed to retrieve Persistable");
        MovingObjectPredictionVector::Ptr d =
            boost::dynamic_pointer_cast<MovingObjectPredictionVector, Persistable>(p);
        Assert(d.get() != 0, "Couldn't cast to MovingObjectPredictionVector");
        // sort in ascending id order (database does not give any ordering guarantees
        // in the absence of an ORDER BY clause)
        std::sort(d->begin(), d->end(), MovingObjectPredictionLessThan());
        Assert(d.get() != &mopv && *d == mopv,
               "persist()/retrieve() resulted in MovingObjectPredictionVector corruption");
    }
    formatters::dropAllVisitSliceTables(loc, policy, props);
}


static void testDb2(std::string const & storageType) {
    // Create the required Policy and DataProperty
    Policy::Ptr policy(new Policy);
    std::string policyRoot(std::string("Formatter.") + "MovingObjectPredictionVector");
    // use custom table name patterns for this test
    policy->set(policyRoot + ".MovingObjectPredictions.templateTableName", "MovingObjectPredictionTemplate");
    policy->set(policyRoot + ".MovingObjectPredictions.perVisitTableNamePattern", "MopsPred_%1%");
    policy->set(policyRoot + ".MovingObjectPredictions.perSliceAndVisitTableNamePattern", "MopsPred_%1%_%2%");

    Policy::Ptr nested(policy->getPolicy(policyRoot));

    Persistence::Ptr pers = Persistence::getPersistence(policy);
    LogicalLocation loc("mysql://test:globular!test@lsst10.ncsa.uiuc.edu:3306/test");

    MovingObjectPredictionVector all;
    int const numSlices = 3; // and use multiple slice tables
    DataProperty::PtrType props(createDbTestProps(0, numSlices, "MovingObjectPredictions"));

    // 1. Write out each slice table seperately
    for (int sliceId = 0; sliceId < numSlices; ++sliceId) {
        DataProperty::PtrType dp = props->findUnique("sliceId");
        dp->setValue(boost::any(sliceId));
        MovingObjectPredictionVector mopv;
        initTestData(mopv, sliceId);
        all.insert(all.end(), mopv.begin(), mopv.end());
        Storage::List storageList;
        storageList.push_back(pers->getPersistStorage(storageType, loc));
        pers->persist(mopv, storageList, props);
    }

    // 2. Read in all slice tables - simulates association pipeline
    //    gathering the results of numSlices image processing pipeline slices
    Storage::List storageList;
    storageList.push_back(pers->getRetrieveStorage(storageType, loc));
    Persistable::Ptr p = pers->retrieve("MovingObjectPredictionVector", storageList, props);
    Assert(p != 0, "Failed to retrieve Persistable");
    MovingObjectPredictionVector::Ptr v =
        boost::dynamic_pointer_cast<MovingObjectPredictionVector, Persistable>(p);
    Assert(v, "Couldn't cast to MovingObjectPredictionVector");
    // sort in ascending id order (database does not give any ordering guarantees
    // in the absence of an ORDER BY clause)
    std::sort(v->begin(), v->end(), MovingObjectPredictionLessThan());
    Assert(v.get() != &all && *v == all,
           "persist()/retrieve() resulted in MovingObjectPredictionVector corruption");
    formatters::dropAllVisitSliceTables(loc, nested, props);
}


int main(int const argc, char const * const * const argv) {
    try {
        testBoost();
        testDb("DbStorage");
        testDb("DbTsvStorage");
        testDb2("DbStorage");
        testDb2("DbTsvStorage");
        if (lsst::mwi::data::Citizen::census(0) == 0) {
            std::clog << "No leaks detected" << std::endl;
        } else {
            Assert(false, "Had memory leaks");
        }
        return EXIT_SUCCESS;
    } catch (lsst::mwi::exceptions::ExceptionStack & exs) {
        std::clog << exs.what() << exs.getStack()->toString("...", true) << std::endl;
    } catch (std::exception & ex) {
        std::clog << ex.what() << std::endl;
    }

    if (lsst::mwi::data::Citizen::census(0) != 0) {
        std::clog << "Leaked memory blocks:" << std::endl;
        lsst::mwi::data::Citizen::census(std::clog);
    }

    return EXIT_FAILURE;
}
