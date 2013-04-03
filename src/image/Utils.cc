#include "lsst/afw/image/Utils.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace image {

PTR(daf::base::PropertySet) readMetadata(std::string const& fileName, int hdu, bool strip) {
    PTR(daf::base::PropertySet) metadata(new lsst::daf::base::PropertyList);
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    fitsfile.readMetadata(*metadata, strip);
    // if INHERIT=T, we want to also include header entries from the primary HDU
    if (fitsfile.getHdu() != 1 && metadata->exists("INHERIT")) {
        bool inherit = false;
        if (metadata->typeOf("INHERIT") == typeid(std::string)) {
            inherit = (metadata->get<std::string>("INHERIT") == "T");
        } else {
            inherit = metadata->get<bool>("INHERIT");
        }
        if (strip) metadata->remove("INHERIT");
        if (inherit) {
            fitsfile.setHdu(1);
            // We don't want to just just call fitsfile.readMetadata to append the new keys,
            // because PropertySet::get will return the last value added when multiple values
            // are present and a scalar is requested; in that case, we want the non-inherited
            // value to be added last, so it's the one that takes precedence.
            PTR(daf::base::PropertySet) inherited(new daf::base::PropertyList);
            fitsfile.readMetadata(*inherited, strip);
            inherited->combine(metadata);
            inherited.swap(metadata);
        }
    }
    return metadata;
}

}}} // namespace lsst::afw::image
