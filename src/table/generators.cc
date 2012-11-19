#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/table/generators.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table {

void RecordOutputGeneratorSet::writeFits(
    afw::fits::Fits & fitsfile, std::string const & kind, CONST_PTR(daf::base::PropertySet) metadata_i
) const {
    bool firstHdu = true;
    for (Vector::const_iterator iter = generators.begin(); iter != generators.end(); ++iter) {
        RecordOutputGenerator & g = **iter;
        BaseCatalog cat(g.getSchema());
        PTR(daf::base::PropertyList) metadata(new daf::base::PropertyList());
        metadata->set("EXTTYPE", kind);
        if (firstHdu) {
            metadata->set(kind + "_NAME", name, "Name of the " + kind + " class stored here.");
            metadata->set(kind + "_NHDU", int(generators.size()), "Number of HDUs for this " + kind);
            if (metadata_i) metadata->combine(metadata_i);
            firstHdu = false;
        }
        cat.getTable()->setMetadata(metadata);
        cat.reserve(g.getRecordCount());
        for (int n = 0; n < g.getRecordCount(); ++n) {
            g.fill(*cat.addNew());
        }
        cat.writeFits(fitsfile);
    }
}

RecordInputGeneratorSet RecordInputGeneratorSet::readFits(
    fits::Fits & fitsfile,
    PTR(daf::base::PropertySet) metadata
) {
    BaseCatalog firstCat = BaseCatalog::readFits(fitsfile);
    if (metadata) {
        metadata->combine(firstCat.getTable()->getMetadata());
    } else {
        metadata = firstCat.getTable()->getMetadata();
    }
    std::string kind = metadata->get<std::string>("EXTTYPE");
    metadata->remove("EXTTYPE");
    std::string nameKey = kind + "_NAME";
    std::string nHduKey = kind + "_NHDU";
    RecordInputGeneratorSet result(metadata->get<std::string>(nameKey));
    metadata->remove(nameKey);
    int nHdu = metadata->get<int>(nHduKey);
    metadata->remove(nHduKey);
    result.generators.reserve(nHdu);
    result.generators.push_back(RecordInputGenerator::make(firstCat));
    for (int n = 1; n < nHdu; ++n) {
        fitsfile.setHdu(1, true);
        BaseCatalog cat = BaseCatalog::readFits(fitsfile);
        if (kind != cat.getTable()->getMetadata()->get<std::string>("EXTTYPE")) {
            throw LSST_FITS_EXCEPT(
                fits::FitsError,
                fitsfile,
                boost::format("Wrong EXTTYPE for HDU %d; expected '%s', got '%s'")
                % fitsfile.getHdu() % kind % 
                cat.getTable()->getMetadata()->get<std::string>("EXTTYPE")
            );
        }
        result.generators.push_back(RecordInputGenerator::make(cat));
    }
    return result;
}

}}} // namespace lsst::afw::table
