#include <stdexcept>
#include <iostream>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SourceMatchIo

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#if 0

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/persistence.h"
#include "lsst/afw/detection/SourceMatch.h"

using namespace std;
using lsst::daf::base::PropertySet;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::LogicalLocation;
using lsst::daf::persistence::Persistence;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::Source;
using lsst::afw::detection::SourceMatch;
using lsst::afw::detection::SourceMatchVector;
using lsst::afw::detection::PersistableSourceMatchVector;

namespace afwForm = lsst::afw::formatters;

BOOST_AUTO_TEST_CASE(persistUnpersistMatchList) {
	SourceMatchVector smv;

	vector<boost::int64_t> srcids;
	vector<boost::int64_t> refids;

	for (int i=0; i<20; i++) {
		boost::int64_t theid;

		theid = 1000 + i;
		refids.push_back(theid);
		Source::Ptr s1(new Source());
		s1->setSourceId(theid);

		theid = 1000000 + 1000*i;
		srcids.push_back(theid);
		Source::Ptr s2(new Source());
		s2->setSourceId(theid);

		// By convention, "first" is the reference catalog object, and "second" is the source object.
		SourceMatch m;
		m.first = s1;
		m.second = s2;
		m.distance = 40.0;
		smv.push_back(m);
	}
	std::string fn = "tests/data/matchlist2.fits";

	LogicalLocation loc(fn);
    Policy::Ptr      policy(new Policy);
	PersistableSourceMatchVector::Ptr psmv(new PersistableSourceMatchVector(smv));
    Persistence::Ptr pers = Persistence::getPersistence(policy);
    PropertySet::Ptr metadata(new PropertySet());

	Storage::List storageList;
	storageList.push_back(pers->getPersistStorage("FitsStorage", loc));
	pers->persist(*psmv, storageList, metadata);

	// read
	Storage::List storageList2;
    PropertySet::Ptr props(new PropertySet());
	storageList2.push_back(pers->getRetrieveStorage("FitsStorage", loc));
	Persistable::Ptr p = pers->retrieve("PersistableSourceMatchVector",
										storageList2, props);
	BOOST_CHECK_MESSAGE(p.get() != 0, "Failed to retrieve Persistable");
	PersistableSourceMatchVector::Ptr persistVec =
		boost::dynamic_pointer_cast<PersistableSourceMatchVector, Persistable>(p);
	BOOST_CHECK_MESSAGE(persistVec.get() != 0, "Couldn't cast to PersistableSourceVector");
	//BOOST_CHECK_MESSAGE(*persistVec == dsv, "persist()/retrieve() resulted in PersistableSourceVector corruption");
	SourceMatchVector smv2 = persistVec->getSourceMatches();
	std::cout << "Persisted size: " << smv.size() << ", unpersisted size: " << smv2.size() << std::endl;
	BOOST_CHECK_MESSAGE(smv2.size() == smv.size(), "unpersisted SourceMatchVector has the wrong size.");

	for (unsigned int i = 0; i != smv.size(); i++) {
            BOOST_CHECK_MESSAGE(smv2[i].first->getSourceId() == refids[i], "Reference id is wrong");
            BOOST_CHECK_MESSAGE(smv2[i].second->getSourceId() == srcids[i], "Source id is wrong");
	}
}

#endif
