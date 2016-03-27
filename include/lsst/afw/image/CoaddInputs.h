// -*- LSST-C++ -*- // fixed format comment for emacs
/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_IMAGE_CoaddInputs_h_INCLUDED
#define LSST_AFW_IMAGE_CoaddInputs_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace image {

/**
 *  @brief A simple Persistable struct containing ExposureCatalogs that record the inputs to a coadd.
 *
 *  The visits catalog corresponds to what task code refers to as coaddTempExps, while the 
 *  ccds catalog corresponds to individual input CCD images (calexps), and has a "visitId"
 *  column that points back to the visits catalog.
 *
 *  The records in the visits catalog will all have the same Wcs as the coadd, as they
 *  represent images that have already been warped to the coadd frame.  Regardless of whether
 *  or not the coadd is PSF-matched, the visit record Psf will generally be CoaddPsfs (albeit
 *  single-depth ones, so they simply pick out the single non-coadd-Psf that is valid for each
 *  point).
 */
class CoaddInputs : public table::io::PersistableFacade<CoaddInputs>, public table::io::Persistable {
public:
    table::ExposureCatalog visits;
    table::ExposureCatalog ccds;
    
    /**
     *  @brief Default constructor.
     *
     *  This simply calls the Catalog default constructors, which means the catalogs have no associated
     *  Table and hence cannot be used for anything until a valid Catalog is assigned to them.
     */
    CoaddInputs();

    /// Construct new catalogs from the given schemas.
    CoaddInputs(table::Schema const & visitSchema, table::Schema const & ccdSchema);

    /// Construct from shallow copies of the given catalogs.
    CoaddInputs(table::ExposureCatalog const & visits_, table::ExposureCatalog const & ccds_);

    /**
     *  @brief Whether the object is in fact persistable - in this case, always true.
     *
     *  To avoid letting coadd provenance prevent coadd code from running, if a nested Wcs or
     *  Psf is not persistable, it will silently not be saved, instead of throwing an exception.
     */
    virtual bool isPersistable() const;

protected:
    virtual std::string getPersistenceName() const;
    virtual std::string getPythonModule() const;
    virtual void write(OutputArchiveHandle & handle) const;
};

}}} // lsst::afw::image

#endif // !LSST_AFW_IMAGE_CoaddInputs_h_INCLUDED
