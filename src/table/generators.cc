#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/table/generators.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table {

void RecordOutputGeneratorSet::writeFits(afw::fits::Fits & fitsfile, std::string const & kind) const {
    int idx = 0;
    PTR(daf::base::PropertyList) metadata(new daf::base::PropertyList());
    metadata->set("EXTTYPE", kind);
    metadata->set(kind + "_NAME", name, "Name of the " + kind + " class stored here");
    metadata->set(kind + "_NHDU", int(generators.size()), "Total number of HDUs for this " + kind);
    for (Vector::const_iterator iter = generators.begin(); iter != generators.end(); ++iter, ++idx) {
        RecordOutputGenerator & g = **iter;
        BaseCatalog cat(g.getSchema());
        metadata->set(kind + "_IDX", idx, "Index of this HDU out of all for this " + kind);
        cat.getTable()->setMetadata(metadata);
        cat.reserve(g.getRecordCount());
        for (int n = 0; n < g.getRecordCount(); ++n) {
            g.fill(*cat.addNew());
        }
        cat.writeFits(fitsfile);
    }
}

RecordInputGeneratorSet RecordInputGeneratorSet::readFits(fits::Fits & fitsfile) {
    BaseCatalog firstCat = BaseCatalog::readFits(fitsfile);
    PTR(daf::base::PropertyList) metadata = firstCat.getTable()->getMetadata();
    std::string kind = metadata->get<std::string>("EXTTYPE");
    metadata->remove("EXTTYPE");
    std::string nameKey = kind + "_NAME";
    std::string nHduKey = kind + "_NHDU";
    std::string idxKey = kind + "_IDX";
    RecordInputGeneratorSet result(metadata->get<std::string>(nameKey));
    metadata->remove(nameKey);
    int nHdu = metadata->get<int>(nHduKey);
    int idx = metadata->get<int>(idxKey);
    if (idx != 0) {
        throw LSST_FITS_EXCEPT(
            fits::FitsError,
            fitsfile,
            boost::format("Current HDU has %s=%d, not 0") % idxKey % idx
        );
    }
    metadata->remove(nHduKey);
    result.generators.reserve(nHdu);
    result.generators.push_back(RecordInputGenerator::make(firstCat));
    for (int n = 1; n < nHdu; ++n) {
        fitsfile.setHdu(1, true);
        BaseCatalog cat = BaseCatalog::readFits(fitsfile);
        metadata = cat.getTable()->getMetadata();
        std::string exttype = metadata->get<std::string>("EXTTYPE");
        if (kind != exttype) {
            throw LSST_FITS_EXCEPT(
                fits::FitsError,
                fitsfile,
                boost::format("Wrong EXTTYPE for HDU %d; expected '%s', got '%s'")
                % fitsfile.getHdu() % kind % exttype
            );
        }
        idx = metadata->get<int>(idxKey);
        if (idx != n) {
            throw LSST_FITS_EXCEPT(
                fits::FitsError,
                fitsfile,
                boost::format("HDU %d has %s=%d, not %d") % fitsfile.getHdu() % idxKey % idx % n
            );
        }
        result.generators.push_back(RecordInputGenerator::make(cat));
    }
    return result;
}

}}} // namespace lsst::afw::table
