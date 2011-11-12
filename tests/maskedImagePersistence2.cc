#include "lsst/daf/persistence.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/image/MaskedImage.h"

int main()
{
    lsst::afw::image::MaskedImage<float> maskedImage(512, 512);
    //self.maskedImage.set((666, 0x1, 123))

    std::string miPath = "tests/data/Dest";
    lsst::daf::persistence::LogicalLocation logicalLocation(miPath);
    lsst::daf::persistence::Persistence::Ptr persistence =
        lsst::daf::persistence::Persistence::getPersistence(lsst::pex::policy::Policy::Ptr(new lsst::pex::policy::Policy()));
    lsst::daf::persistence::Storage::Ptr storage =
        persistence->getPersistStorage("BoostStorage", logicalLocation);
    lsst::daf::persistence::Storage::List storageList;
    storageList.push_back(storage);

    lsst::daf::base::PropertySet::Ptr additionalData(new lsst::daf::base::PropertySet());
    persistence->persist(maskedImage, storageList, additionalData);
}
