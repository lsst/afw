#include <stdexcept>

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

void test() {
	SourceMatchVector smv;

	vector<boost::int64_t> srcids;
	vector<boost::int64_t> refids;

	for (int i=0; i<20; i++) {
		Source::Ptr s1(new Source());
		boost::int64_t theid = 1000 + i;
		srcids.push_back(theid);
		s1->setSourceId(theid);
		theid = 1000000 + 1000*i;
		refids.push_back(theid);
		Source::Ptr s2(new Source());
		s2->setSourceId(theid);
		SourceMatch m;
		m.first = s1;
		m.second = s2;
		m.distance = 40.0;
		smv.push_back(m);
	}
	std::string fn = "test/data/matchlist2.fits";

	LogicalLocation loc(fn);
    Policy::Ptr      policy(new Policy);
	PersistableSourceMatchVector::Ptr psmv(new PersistableSourceMatchVector(smv));
    Persistence::Ptr pers = Persistence::getPersistence(policy);
    PropertySet::Ptr metadata(new PropertySet());

	Storage::List storageList;
	storageList.push_back(pers->getPersistStorage("FitsStorage", loc));
	pers->persist(*psmv, storageList, metadata);

	// read
	/*
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
	 */
	 

}

int main(int argc, char *argv[]) {
    try {
        test();
    } catch (std::exception const &e) {
        clog << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

