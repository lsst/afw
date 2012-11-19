#include "lsst/afw/image/Utils.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace image {

PTR(daf::base::PropertySet) readMetadata(std::string const& fileName, int hdu, bool strip) {
    PTR(daf::base::PropertySet) metadata(new lsst::daf::base::PropertyList);
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    fitsfile.readMetadata(*metadata, strip);
    return metadata;
}

}}} // namespace lsst::afw::image
