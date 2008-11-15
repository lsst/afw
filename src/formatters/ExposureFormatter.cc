// -*- lsst-c++ -*-

/** @file
 * @brief Implementation of ExposureFormatter class
 *
 * @author $Author: ktlim $
 * @version $Revision: 2151 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * @ingroup afw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include "boost/serialization/shared_ptr.hpp"

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/persistence.h"
#include "lsst/daf/persistence/DataPropertyFormatter.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/formatters/ExposureFormatter.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/image/Exposure.h"


// #include "lsst/afw/image/LSSTFitsResource.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.ExposureFormatter", level, s);
}

using lsst::daf::base::Persistable;
using lsst::daf::persistence::Storage;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::base::Persistable;
using lsst::afw::image::Exposure;
using lsst::afw::image::MaskPixel;
using lsst::afw::image::VariancePixel;

namespace lsst {
namespace afw {
namespace formatters {

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
class ExposureFormatterTraits {
public:
    static std::string name;
};

template<> std::string ExposureFormatterTraits<boost::uint16_t, MaskPixel, VariancePixel>::name("ExposureU");
template<> std::string ExposureFormatterTraits<int, MaskPixel, VariancePixel>::name("ExposureI");
template<> std::string ExposureFormatterTraits<float, MaskPixel, VariancePixel>::name("ExposureF");
template<> std::string ExposureFormatterTraits<double, MaskPixel, VariancePixel>::name("ExposureD");


template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
lsst::daf::persistence::FormatterRegistration ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::registration(
    ExposureFormatterTraits<ImagePixelT, MaskPixelT, VariancePixelT>::name,
    typeid(Exposure<ImagePixelT, MaskPixelT, VariancePixelT>),
    createInstance);

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::ExposureFormatter(
    lsst::pex::policy::Policy::Ptr policy) :
    lsst::daf::persistence::Formatter(typeid(*this)), _policy(policy) {
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::~ExposureFormatter(void) {
}

/** Lookup a filter number in the database to find a filter name.
 */
static std::string lookupFilterName(
    DbStorage* db,  //!< Database to look in
    int filterId    //!< Number of filter to lookup
    ) {
    db->setTableForQuery("Filter");
    db->outColumn("filterName");
    db->condParam<int>("id", filterId);
    db->setQueryWhere("filterId = :id");
    db->query();
    if (!db->next() || db->columnIsNull(0)) {
        throw lsst::pex::exceptions::Runtime("Unable to get name for filter id: " + static_cast<int>(filterId));
    }
    std::string filterName = db->getColumnByPos<std::string>(0);
    if (db->next()) {
        throw lsst::pex::exceptions::Runtime("Multiple names for filter id: " + static_cast<int>(filterId));

    }
    db->finishQuery();
    return filterName;
}


/** Set an output column's value from a DataProperty, setting it to NULL if
 * the desired child property does not exist.
 */
template <typename T>
static void setColumn(
    DbStorage* db,                                  //!< Destination database
    std::string const& colName,                     //!< Output column name
    lsst::daf::base::DataProperty::PtrType source,  //!< Source DataProperty
    std::string const& dpName                       //!< Child name
    ) {
    lsst::daf::base::DataProperty::PtrType dp = source->findUnique(dpName);
    if (!dp) {
        db->setColumnToNull(colName);
    }
    else {
        db->setColumn<T>(colName, boost::any_cast<T>(dp->getValue()));
    }
}

/** Set an output column's value from a DataProperty, setting it to NULL if
 * the desired child property does not exist.  Casts from DataProperty type to
 * database field type.
 */
template <typename T1, typename T2>
static void setColumn(
    DbStorage* db,                                  //!< Destination database
    std::string const& colName,                     //!< Output column name
    lsst::daf::base::DataProperty::PtrType source,  //!< Source DataProperty
    std::string const& dpName                       //!< Child name
    ) {
    lsst::daf::base::DataProperty::PtrType dp = source->findUnique(dpName);
    if (!dp) {
        db->setColumnToNull(colName);
    }
    else {
        db->setColumn<T1>(colName, static_cast<T1>(boost::any_cast<T2>(
            source->findUnique(dpName)->getValue())));
    }
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::base::DataProperty::PtrType additionalData) {
    execTrace("ExposureFormatter write start");
    Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const* ip =
        dynamic_cast<Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const*>(persistable);
    if (ip == 0) {
        throw lsst::pex::exceptions::Runtime("Persisting non-Exposure");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ExposureFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("ExposureFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("ExposureFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());

        lsst::daf::base::DataProperty::PtrType wcsDP =
            lsst::afw::formatters::WcsFormatter::generateDataProperty(
                *(ip->_wcsPtr));

        Exposure<ImagePixelT, MaskPixelT, VariancePixelT>* vip = const_cast<Exposure<ImagePixelT, MaskPixelT, VariancePixelT>*>(ip);
        vip->getMetaData()->addChildren(wcsDP);
        ip->_maskedImage.writeFits(fits->getPath());
        execTrace("ExposureFormatter write end");
        return;
    } else if (typeid(*storage) == typeid(DbStorage)) {
        execTrace("ExposureFormatter write DbStorage");
        DbStorage* db = dynamic_cast<DbStorage*>(storage.get());

        // Get the Wcs headers.
        lsst::daf::base::DataProperty::PtrType wcsDP =
            lsst::afw::formatters::WcsFormatter::generateDataProperty(*(ip->_wcsPtr));

        // Get the image headers.
        lsst::daf::base::DataProperty::PtrType dp = ip->getMetaData();
        if (!dp) {
            throw lsst::pex::exceptions::Runtime("Unable to retrieve metadata from MaskedImage's Image");
        }

        // Select a table to insert into based on the itemName.
        std::string itemName = boost::any_cast<std::string>(
            additionalData->findUnique("itemName")->getValue());
        std::string tableName = itemName;
        if (_policy->exists(itemName)) {
            lsst::pex::policy::Policy::Ptr itemPolicy = _policy->getPolicy(itemName);
            if (itemPolicy->exists("TableName")) {
                tableName = itemPolicy->getString("TableName");
            }
        }
        if (tableName != "Raw_CCD_Exposure" &&
            tableName != "Science_CCD_Exposure") {
            throw lsst::pex::exceptions::Runtime(
                "Unknown table name for persisting Exposure to DbStorage: " +
                tableName + "for item " + itemName);
        }
        db->setTableForInsert(tableName);

        // Set the identifier columns.

        int ccdId = extractCcdId(additionalData);
        int64_t exposureId = extractExposureId(additionalData);
        int64_t ccdExposureId = extractCcdExposureId(additionalData);

        if (tableName == "Raw_CCD_Exposure") {
            db->setColumn<long long>("rawCCDExposureId", ccdExposureId);
            db->setColumn<long long>("rawFPAExposureId", exposureId);
        }
        else { // Science_CCD_Exposure
            db->setColumn<long long>("scienceCCDExposureId", ccdExposureId);
            db->setColumn<long long>("scienceFPAExposureId", exposureId);
            db->setColumn<long long>("rawCCDExposureId", ccdExposureId);
            /// \todo Check that rawCCDExposureId == scienceCCDExposureId --
            /// KTL -- 2008-01-25
        }

        db->setColumn<int>("ccdDetectorId", ccdId);

        // Set the URL column with the location of the FITS file.
        setColumn<std::string>(db, "url",
                              additionalData, "StorageLocation.FitsStorage");


        // Set the Wcs information columns.
        setColumn<std::string>(db, "ctype1", wcsDP, "CTYPE1");
        setColumn<std::string>(db, "ctype2", wcsDP, "CTYPE2");
        setColumn<float, double>(db, "crpix1", wcsDP, "CRPIX1");
        setColumn<float, double>(db, "crpix2", wcsDP, "CRPIX2");
        setColumn<double>(db, "crval1", wcsDP, "CRVAL1");
        setColumn<double>(db, "crval2", wcsDP, "CRVAL2");
        setColumn<double>(db, "cd11", wcsDP, "CD1_1");
        setColumn<double>(db, "cd21", wcsDP, "CD2_1");
        setColumn<double>(db, "cd12", wcsDP, "CD1_2");
        setColumn<double>(db, "cd22", wcsDP, "CD2_2");

        if (tableName == "Science_CCD_Exposure") {
            // Set calibration data columns.
            setColumn<float, double>(db, "photoFlam", dp, "PHOTFLAM");
            setColumn<float, double>(db, "photoZP", dp, "PHOTZP");
        }

        // Phew!  Insert the row now.
        db->insertRow();

        execTrace("ExposureFormatter write end");
        return;
    }
    throw lsst::pex::exceptions::Runtime("Unrecognized Storage for Exposure");
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
Persistable* ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::read(
    Storage::Ptr storage,
    lsst::daf::base::DataProperty::PtrType additionalData) {
    execTrace("ExposureFormatter read start");
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ExposureFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        Exposure<ImagePixelT, MaskPixelT, VariancePixelT>* ip = new Exposure<ImagePixelT, MaskPixelT, VariancePixelT>;
        boost->getIArchive() & *ip;
        execTrace("ExposureFormatter read end");
        return ip;
    } else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("ExposureFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        Exposure<ImagePixelT, MaskPixelT, VariancePixelT>* ip = new Exposure<ImagePixelT, MaskPixelT, VariancePixelT>(fits->getPath());
        execTrace("ExposureFormatter read end");
        return ip;
    } else if (typeid(*storage) == typeid(DbStorage)) {
        execTrace("ExposureFormatter read DbStorage");
        DbStorage* db = dynamic_cast<DbStorage*>(storage.get());

        // Select a table to retrieve from based on the itemName.
        std::string itemName = boost::any_cast<std::string>(
            additionalData->findUnique("itemName")->getValue());
        std::string tableName = itemName;
        if (_policy->exists(itemName)) {
            lsst::pex::policy::Policy::Ptr itemPolicy =
                _policy->getPolicy(itemName);
            if (itemPolicy->exists("TableName")) {
                tableName = itemPolicy->getString("TableName");
            }
        }
        if (tableName != "Raw_CCD_Exposure" &&
            tableName != "Science_CCD_Exposure") {
            throw lsst::pex::exceptions::Runtime(
                "Unknown table name for retrieving Exposure from DbStorage: " +
                tableName + " for item " + itemName);
        }
        db->setTableForQuery(tableName);


        // Set the identifier column tests.
        db->condParam<int64_t>("id", boost::any_cast<int64_t>(
                additionalData->findUnique("ccdExposureId")->getValue()));
        if (tableName == "Raw_CCD_Exposure") {
            db->setQueryWhere("rawCCDExposureId = :id");
        }
        else { // Science_CCD_Exposure
            db->setQueryWhere("scienceCCDExposureId = :id");
        }

        db->outColumn("url");

        if (tableName == "Science_CCD_Exposure") {
            // Set the Wcs information columns.
            db->outColumn("ctype1");
            db->outColumn("ctype2");
            db->outColumn("crpix1");
            db->outColumn("crpix2");
            db->outColumn("crval1");
            db->outColumn("crval2");
            db->outColumn("cd11");
            db->outColumn("cd21");
            db->outColumn("cd12");
            db->outColumn("cd22");

            // Set calibration data columns.
            db->outColumn("photoFlam");
            db->outColumn("photoZP");
        }

        // Phew!  Run the query.
        db->query();
        if (!db->next()) {
            throw lsst::pex::exceptions::Runtime("Unable to retrieve row");
        }
        // ...
        if (db->next()) {
            throw lsst::pex::exceptions::Runtime("Non-unique Exposure retrieved");
        }
        db->finishQuery();

        //! \todo Should really have FITS be a separate Storage.
        // - KTL - 2007-11-29

        // Restore image from FITS...
        Exposure<ImagePixelT, MaskPixelT, VariancePixelT>* ip =
            new Exposure<ImagePixelT, MaskPixelT, VariancePixelT>(db->getColumnByPos<std::string>(0));
        lsst::daf::base::DataProperty::PtrType dp = ip->getMetadata();

        // Look up the filter name given the ID.
        int filterId = db->getColumnByPos<int>(1);
        std::string filterName = lookupFilterName(db, filterId);
        dp->deleteAll("FILTER");
        dp->addProperty(
            lsst::daf::base::DataProperty::PtrType(
                new lsst::daf::base::DataProperty("FILTER", filterName)));

        // Set the image headers.
        // Set the Wcs headers in ip->_wcsPtr.

        //! \todo Need to implement overwriting of FITS metadata DataProperty
        // with values from database. - KTL - 2007-12-18

        execTrace("ExposureFormatter read end");
        return ip;
    }
    throw lsst::pex::exceptions::Runtime("Unrecognized Storage for Exposure");
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::daf::base::DataProperty::PtrType additionalData) {
    //! \todo Implement update from FitsStorage, keeping DB-provided headers.
    // - KTL - 2007-11-29
    throw lsst::pex::exceptions::Runtime("Unexpected call to update for Exposure");
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT> template <class Archive>
void ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::delegateSerialize(
    Archive& ar, unsigned int const version, Persistable* persistable) {
    execTrace("ExposureFormatter delegateSerialize start");
    Exposure<ImagePixelT, MaskPixelT, VariancePixelT>* ip =
        dynamic_cast<Exposure<ImagePixelT, MaskPixelT, VariancePixelT>*>(persistable);
    if (ip == 0) {
        throw lsst::pex::exceptions::Runtime("Serializing non-Exposure");
    }
    ar & ip->_metaData & ip->_maskedImage & ip->_wcsPtr;
    execTrace("ExposureFormatter delegateSerialize end");
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
lsst::daf::persistence::Formatter::Ptr ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new ExposureFormatter<ImagePixelT, MaskPixelT, VariancePixelT>(policy));
}

#define INSTANTIATE(I, M, V) \
    template class ExposureFormatter<I, M, V>; \
    template void ExposureFormatter<I, M, V>::delegateSerialize<boost::archive::text_oarchive>( \
        boost::archive::text_oarchive &, unsigned int const, Persistable *); \
    template void ExposureFormatter<I, M, V>::delegateSerialize<boost::archive::text_iarchive>( \
        boost::archive::text_iarchive &, unsigned int const, Persistable *);

INSTANTIATE(uint16_t, MaskPixel, VariancePixel)
INSTANTIATE(int, MaskPixel, VariancePixel)
INSTANTIATE(float, MaskPixel, VariancePixel)
INSTANTIATE(double, MaskPixel, VariancePixel)

}}} // namespace lsst::afw::formatters
