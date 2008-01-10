// -*- lsst-c++ -*-

/** \file
 * \brief Implementation of ExposureFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2151 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * \ingroup fw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include "lsst/fw/formatters/ExposureFormatter.h"
#include "lsst/fw/Exposure.h"

#include "lsst/mwi/persistence/FormatterImpl.h"

#include "lsst/fw/formatters/MaskedImageFormatter.h"
#include "lsst/fw/formatters/Utils.h"
#include "lsst/fw/formatters/WcsFormatter.h"

#include "lsst/mwi/exceptions.h"
#include "lsst/mwi/persistence/LogicalLocation.h"
#include "lsst/mwi/persistence/BoostStorage.h"
#include "lsst/mwi/persistence/DateTime.h"
#include "lsst/mwi/persistence/DbStorage.h"
#include "lsst/mwi/persistence/FitsStorage.h"
#include "lsst/mwi/utils/Trace.h"

#include <boost/serialization/shared_ptr.hpp>
// #include "lsst/fw/LSSTFitsResource.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::mwi::utils::Trace("fw.ExposureFormatter", level, s);
}

using namespace lsst::mwi::persistence;

namespace lsst {
namespace fw {
namespace formatters {

template <typename ImagePixelT, typename MaskPixelT>
class ExposureFormatterTraits {
public:
    static std::string name;
};

template<> std::string ExposureFormatterTraits<boost::uint16_t, maskPixelType>::name("ExposureU");
template<> std::string ExposureFormatterTraits<float, maskPixelType>::name("ExposureF");
template<> std::string ExposureFormatterTraits<double, maskPixelType>::name("ExposureD");


template <typename ImagePixelT, typename MaskPixelT>
FormatterRegistration ExposureFormatter<ImagePixelT, MaskPixelT>::registration(
    ExposureFormatterTraits<ImagePixelT, MaskPixelT>::name,
    typeid(Exposure<ImagePixelT, MaskPixelT>),
    createInstance);

template <typename ImagePixelT, typename MaskPixelT>
ExposureFormatter<ImagePixelT, MaskPixelT>::ExposureFormatter(
    lsst::mwi::policy::Policy::Ptr policy) :
    Formatter(typeid(*this)) {
}

template <typename ImagePixelT, typename MaskPixelT>
ExposureFormatter<ImagePixelT, MaskPixelT>::~ExposureFormatter(void) {
}

/** Lookup a filter name in the database to find a filter id number.
 */
static int lookupFilterId(
    DbStorage* db,                  //!< Database to look in
    std::string const& filterName   //!< Name of filter to lookup
    ) {
    db->setTableForQuery("Filter");
    db->outColumn("filterId");
    db->condParam<std::string>("name", filterName);
    db->setQueryWhere("filtName = :name");
    db->query();
    if (!db->next() || db->columnIsNull(0)) {
        throw lsst::mwi::exceptions::Runtime("Unable to get id for filter type: " + filterName);
    }
    int filterId = db->getColumnByPos<int>(0);
    if (db->next()) {
        throw lsst::mwi::exceptions::Runtime("Multiple ids for filter type: " + filterName);

    }
    db->finishQuery();
    return filterId;
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
        throw lsst::mwi::exceptions::Runtime("Unable to get name for filter id: " + static_cast<int>(filterId));
    }
    std::string filterName = db->getColumnByPos<std::string>(0);
    if (db->next()) {
        throw lsst::mwi::exceptions::Runtime("Multiple names for filter id: " + static_cast<int>(filterId));

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
    lsst::mwi::data::DataProperty::PtrType source,  //!< Source DataProperty
    std::string const& dpName                       //!< Child name
    ) {
    lsst::mwi::data::DataProperty::PtrType dp = source->findUnique(dpName);
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
    lsst::mwi::data::DataProperty::PtrType source,  //!< Source DataProperty
    std::string const& dpName                       //!< Child name
    ) {
    lsst::mwi::data::DataProperty::PtrType dp = source->findUnique(dpName);
    if (!dp) {
        db->setColumnToNull(colName);
    }
    else {
        db->setColumn<T1>(colName, static_cast<T1>(boost::any_cast<T2>(
            source->findUnique(dpName)->getValue())));
    }
}

template <typename ImagePixelT, typename MaskPixelT>
void ExposureFormatter<ImagePixelT, MaskPixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::mwi::data::DataProperty::PtrType additionalData) {
    execTrace("ExposureFormatter write start");
    Exposure<ImagePixelT, MaskPixelT> const* ip =
        dynamic_cast<Exposure<ImagePixelT, MaskPixelT> const*>(persistable);
    if (ip == 0) {
        throw lsst::mwi::exceptions::Runtime("Persisting non-Exposure");
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

        lsst::mwi::data::DataProperty::PtrType wcsDP =
            lsst::fw::formatters::WcsFormatter::generateDataProperty(
                *(ip->_wcsPtr));

        Exposure<ImagePixelT, MaskPixelT>* vip =
            const_cast<Exposure<ImagePixelT, MaskPixelT>*>(ip);
        vip->_maskedImage.getImage()->getMetaData()->addChildren(wcsDP);
        vip->_maskedImage.getVariance()->getMetaData()->addChildren(wcsDP);
        ip->_maskedImage.writeFits(fits->getPath());
        execTrace("ExposureFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(DbStorage)) {
        execTrace("ExposureFormatter write DbStorage");
        DbStorage* db = dynamic_cast<DbStorage*>(storage.get());

        // Get the WCS headers.
        lsst::mwi::policy::Policy::Ptr policy;
        boost::shared_ptr<WcsFormatter> wcsFormatter =
            boost::dynamic_pointer_cast<WcsFormatter, Formatter>(
                WcsFormatter::createInstance(policy));
        lsst::mwi::data::DataProperty::PtrType wcsDP =
            wcsFormatter->generateDataProperty(*(ip->_wcsPtr));

        // Get the image headers.
        lsst::mwi::data::DataProperty::PtrType dp =
            ip->_maskedImage.getImage()->getMetaData();
        if (!dp) {
            throw lsst::mwi::exceptions::Runtime("Unable to retrieve metadata from MaskedImage's Image");
        }

        // Look up the filter ID given the name.
        std::string filterName =
            boost::any_cast<std::string>(dp->findUnique("FILTER")->getValue());
        int filterId = lookupFilterId(db, filterName);


        // Select a table to insert into based on the itemName.
        std::string itemName = boost::any_cast<std::string>(
            additionalData->findUnique("itemName")->getValue());
        if (itemName != "Raw_CCD_Exposure" &&
            itemName != "Science_CCD_Exposure") {
            throw lsst::mwi::exceptions::Runtime(
                "Unknown table name for persisting Exposure to DbStorage: " +
                itemName);
        }
        db->setTableForInsert(itemName);


        // Set the identifier columns.
        if (itemName == "Raw_CCD_Exposure") {
            setColumn<long long>(db, "rawCCDExposureId",
                                 additionalData, "ccdExposureId");
            setColumn<int>(db, "rawFPAExposureId",
                           additionalData, "exposureId");
        }
        else { // Science_CCD_Exposure
            setColumn<long long>(db, "scienceCCDExposureId",
                                 additionalData, "ccdExposureId");
            setColumn<int>(db, "scienceFPAExposureId",
                           additionalData, "exposureId");
            setColumn<long long>(db, "rawCCDExposureId",
                                 additionalData, "ccdExposureId");
        }

        setColumn<int>(db, "visitId", additionalData, "visitId");
        setColumn<int>(db, "ccdDetectorId", additionalData, "ccdId");
        // Set the URL column with the location of the FITS file.
        setColumn<std::string>(db, "url",
                              additionalData, "StorageLocation.FitsStorage");


        // Set the filter ID column.
        db->setColumn<int>("filterId", filterId);

        // Set the RA and declination columns for raw images.
        if (itemName == "Raw_CCD_Exposure") {
            setColumn<std::string>(db, "radecSys", dp, "RADECSYS");
            setColumn<double>(db, "ra", dp, "RA");
            setColumn<double>(db, "decl", dp, "DECL");
        }

        // Set the WCS information columns.
        setColumn<float, double>(db, "equinox", dp, "EQUINOX");
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

        // Set the observation start time and exposure time columns.
        double mjdObs = boost::any_cast<double>(
            dp->findUnique("MJD-OBS")->getValue());
        if (itemName == "Raw_CCD_Exposure") {
            DateTime utc(mjdObs);
            db->setColumn<DateTime>("dateObs", utc);
            db->setColumn<DateTime>("taiObs", utc.utc2tai());
            db->setColumn<double>("mjdObs", mjdObs);
        }
        else { // Science_CCD_Exposure
            db->setColumn<DateTime>("dateObs", DateTime(mjdObs));
        }
        setColumn<float>(db, "expTime", dp, "EXPTIME");

        // Set calibration input/output data columns.
        if (itemName == "Raw_CCD_Exposure") {
            setColumn<float, double>(db, "darkTime", dp, "DARKTIME");
            setColumn<float, double>(db, "zd", dp, "ZD"); // Zenith distance
            setColumn<float, double>(db, "airmass", dp, "AIRMASS");
        }
        else { // Science_CCD_Exposure
            setColumn<float, double>(db, "photoFlam", dp, "PHOTFLAM");
            setColumn<float, double>(db, "photoZP", dp, "PHOTZP");
        }

        // Phew!  Insert the row now.
        db->insertRow();

        execTrace("ExposureFormatter write end");
        return;
    }
    throw lsst::mwi::exceptions::Runtime("Unrecognized Storage for Exposure");
}

template <typename ImagePixelT, typename MaskPixelT>
Persistable* ExposureFormatter<ImagePixelT, MaskPixelT>::read(
    Storage::Ptr storage,
    lsst::mwi::data::DataProperty::PtrType additionalData) {
    execTrace("ExposureFormatter read start");
    Exposure<ImagePixelT, MaskPixelT>* ip = new Exposure<ImagePixelT, MaskPixelT>;
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ExposureFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("ExposureFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("ExposureFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        ip->readFits(fits->getPath());
        execTrace("ExposureFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(DbStorage)) {
        execTrace("ExposureFormatter read DbStorage");
        DbStorage* db = dynamic_cast<DbStorage*>(storage.get());

        // Select a table to retrieve from based on the itemName.
        std::string itemName = boost::any_cast<std::string>(
            additionalData->findUnique("itemName")->getValue());
        if (itemName != "Raw_CCD_Exposure" &&
            itemName != "Science_CCD_Exposure") {
            throw lsst::mwi::exceptions::Runtime(
                "Unknown table name for retrieving Exposure from DbStorage: " +
                itemName);
        }
        db->setTableForQuery(itemName);


        // Set the identifier column tests.
        db->condParam<int64_t>("id", boost::any_cast<int64_t>(
                additionalData->findUnique("ccdExposureId")->getValue()));
        if (itemName == "Raw_CCD_Exposure") {
            db->setQueryWhere("rawCCDExposureId = :id");
        }
        else { // Science_CCD_Exposure
            db->setQueryWhere("scienceCCDExposureId = :id");
        }

        db->outColumn("url");
        db->outColumn("filterId");

        // Set the WCS information columns.
        db->outColumn("equinox");
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

        // Set the observation start time and exposure time columns.
        db->outColumn("dateObs");
        db->outColumn("expTime");

        // Set calibration input/output data columns.
        if (itemName == "Raw_CCD_Exposure") {
            db->outColumn("radecSys");
            db->outColumn("ra");
            db->outColumn("decl");

            db->outColumn("darkTime");
            db->outColumn("zd");
            db->outColumn("airmass");
        }
        else { // Science_CCD_Exposure
            db->outColumn("photoFlam");
            db->outColumn("photoZP");
        }

        // Phew!  Run the query.
        db->query();
        if (!db->next()) {
            throw lsst::mwi::exceptions::Runtime("Unable to retrieve row");
        }
        // ...
        if (db->next()) {
            throw lsst::mwi::exceptions::Runtime("Non-unique Exposure retrieved");
        }
        db->finishQuery();

        //! \todo Should really have FITS be a separate Storage.
        // - KTL - 2007-11-29

        // Restore image from FITS...
        ip->readFits(db->getColumnByPos<std::string>(0));
        DataProperty::PtrType dp = ip->getMetadata();

        // Look up the filter name given the ID.
        int filterId = db->getColumnByPos<int>(1);
        std::string filterName = lookupFilterName(db, filterId);
        dp->deleteAll("FILTER");
        dp->addProperty(lsst::mwi::data::SupportFactory::createLeafProperty(
                "FILTER", filterName));

        // Set the image headers.
        // Set the WCS headers in ip->_wcsPtr.

        //! \todo Need to implement overwriting of FITS metadata DataProperty
        // with values from database. - KTL - 2007-12-18

        execTrace("ExposureFormatter read end");
        return ip;
    }
    throw lsst::mwi::exceptions::Runtime("Unrecognized Storage for Exposure");
}

template <typename ImagePixelT, typename MaskPixelT>
void ExposureFormatter<ImagePixelT, MaskPixelT>::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::mwi::data::DataProperty::PtrType additionalData) {
    //! \todo Implement update from FitsStorage, keeping DB-provided headers.
    // - KTL - 2007-11-29
    throw lsst::mwi::exceptions::Runtime("Unexpected call to update for Exposure");
}

template <typename ImagePixelT, typename MaskPixelT> template <class Archive>
void ExposureFormatter<ImagePixelT, MaskPixelT>::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("ExposureFormatter delegateSerialize start");
    Exposure<ImagePixelT, MaskPixelT>* ip = dynamic_cast<Exposure<ImagePixelT, MaskPixelT>*>(persistable);
    if (ip == 0) {
        throw lsst::mwi::exceptions::Runtime("Serializing non-Exposure");
    }
    ar & ip->_maskedImage & ip->_wcsPtr;
    execTrace("ExposureFormatter delegateSerialize end");
}

template <typename ImagePixelT, typename MaskPixelT>
Formatter::Ptr ExposureFormatter<ImagePixelT, MaskPixelT>::createInstance(
    lsst::mwi::policy::Policy::Ptr policy) {
    return Formatter::Ptr(new ExposureFormatter<ImagePixelT, MaskPixelT>(policy));
}

template class ExposureFormatter<boost::uint16_t, lsst::fw::maskPixelType>;
template class ExposureFormatter<float, lsst::fw::maskPixelType>;
template class ExposureFormatter<double, lsst::fw::maskPixelType>;

}}} // namespace lsst::fw::formatters
