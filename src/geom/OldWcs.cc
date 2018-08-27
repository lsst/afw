/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include <algorithm>
#include <exception>
#include <memory>

#include "astshim.h"

#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/geom/wcsUtils.h"
#include "lsst/afw/geom/SkyWcs.h"

namespace lsst {
namespace afw {
namespace geom {
namespace {

/*
 * Does one string end with another?
 *
 * From https://stackoverflow.com/a/2072890
 */
inline bool endsWith(std::string const& value, std::string const& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

/*
 * @internal Get FITS WCS metadata from a record for an lsst::afw::image::Wcs
 * or the non-SIP terms of an lsst::afw::image::TanWcs
 *
 * @param[in] record  Record holding data for an old lsst::afw::image::Wcs
 * @return FITS WCS metadata
 */
std::shared_ptr<daf::base::PropertyList> getOldWcsMetadata(table::BaseRecord const& record);

/*
 * @internal Get FITS WCS metadata for the SIP terms of a TAN-SIP WCS
 * saved by lsst::afw::image::TanWcs
 *
 * @param[in] record  Record holding SIP terms for an old lsst::afw::image::TanWcs
 * @return FITS WCS metadata
 */
std::shared_ptr<daf::base::PropertyList> getOldSipMetadata(table::BaseRecord const& record);

// Unpersist a SkyWcs from table data for an afw::image::Wcs
class OldWcsFactory : public table::io::PersistableFactory {
public:
    explicit OldWcsFactory(std::string const& name) : table::io::PersistableFactory(name) {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() >= 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        auto const& record = catalogs.front().front();
        auto const metadata = getOldWcsMetadata(record);
        return std::make_shared<SkyWcs>(*metadata);
    }
};

OldWcsFactory registerWcs("Wcs");

// Unpersist a SkyWcs from table data for an afw::image::TanWcs, which may have a record for SIP terms
class OldTanWcsFactory : public table::io::PersistableFactory {
public:
    explicit OldTanWcsFactory(std::string const& name) : table::io::PersistableFactory(name) {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() >= 1u);
        auto const& record = catalogs.front().front();
        auto const metadata = getOldWcsMetadata(record);

        if (catalogs.size() > 1u) {
            LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
            LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
            LSST_ARCHIVE_ASSERT(catalogs.back().size() == 1u);
            std::shared_ptr<table::BaseRecord const> sipRecord = catalogs.back().begin();

            // if CTYPE1 or CTYPE2 does not end with -SIP (and likely it never does), append it
            std::string const ctypeSuffix = "-SIP";
            for (auto i = 1; i <= 2; ++i) {
                std::string const ctypeName = "CTYPE" + std::to_string(i);
                auto const ctypeValue = metadata->getAsString(ctypeName);
                if (!endsWith(ctypeValue, ctypeSuffix)) {
                    metadata->set(ctypeName, ctypeValue + ctypeSuffix);
                }
            }

            auto sipMetadata = getOldSipMetadata(*sipRecord);
            metadata->combine(sipMetadata);
        }
        return std::make_shared<SkyWcs>(*metadata);
    }
};

OldTanWcsFactory registerTanWcs("TanWcs");

// Read-only singleton struct containing the schema and keys that lsst::afw::image::Wcs
// and the non-SIP portion of lsst::afw::image::TanWcs were mapped to.
struct OldWcsPersistenceHelper {
    table::Schema schema;
    table::PointKey<double> crval;
    table::PointKey<double> crpix;
    table::Key<table::Array<double>> cd;
    table::Key<std::string> ctype1;
    table::Key<std::string> ctype2;
    table::Key<double> equinox;
    table::Key<std::string> radesys;
    table::Key<std::string> cunit1;
    table::Key<std::string> cunit2;

    static OldWcsPersistenceHelper const& get() {
        static OldWcsPersistenceHelper instance;
        return instance;
    };

    // No copying
    OldWcsPersistenceHelper(const OldWcsPersistenceHelper&) = delete;
    OldWcsPersistenceHelper& operator=(const OldWcsPersistenceHelper&) = delete;

    // No moving
    OldWcsPersistenceHelper(OldWcsPersistenceHelper&&) = delete;
    OldWcsPersistenceHelper& operator=(OldWcsPersistenceHelper&&) = delete;

private:
    OldWcsPersistenceHelper()
            : schema(),
              crval(table::PointKey<double>::addFields(schema, "crval", "celestial reference point", "deg")),
              crpix(table::PointKey<double>::addFields(schema, "crpix", "pixel reference point", "pixel")),
              cd(schema.addField<table::Array<double>>(
                      "cd", "linear transform matrix, ordered (1_1, 2_1, 1_2, 2_2)", 4)),
              ctype1(schema.addField<std::string>("ctype1", "coordinate type", 72)),
              ctype2(schema.addField<std::string>("ctype2", "coordinate type", 72)),
              equinox(schema.addField<double>("equinox", "equinox of coordinates")),
              radesys(schema.addField<std::string>("radesys", "coordinate system for equinox", 72)),
              cunit1(schema.addField<std::string>("cunit1", "coordinate units", 72)),
              cunit2(schema.addField<std::string>("cunit2", "coordinate units", 72)) {
        schema.getCitizen().markPersistent();
    }
};

std::shared_ptr<daf::base::PropertyList> getOldWcsMetadata(table::BaseRecord const& record) {
    auto const& keys = OldWcsPersistenceHelper::get();
    LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);

    auto metadata = std::make_shared<daf::base::PropertyList>();

    auto crvalDeg = record.get(keys.crval);
    auto crpix = record.get(keys.crpix);
    auto cd = record.get(keys.cd);
    metadata->set("CRVAL1", crvalDeg[0]);
    metadata->set("CRVAL2", crvalDeg[1]);
    // Add 1 to CRPIX because the field was saved using the LSST standard: 0,0 is lower left pixel
    metadata->set("CRPIX1", crpix[0] + 1);
    metadata->set("CRPIX2", crpix[1] + 1);
    metadata->set("CD1_1", cd[0]);
    metadata->set("CD2_1", cd[1]);
    metadata->set("CD1_2", cd[2]);
    metadata->set("CD2_2", cd[3]);
    metadata->set("CTYPE1", record.get(keys.ctype1));
    metadata->set("CTYPE2", record.get(keys.ctype2));
    metadata->set("EQUINOX", record.get(keys.equinox));
    metadata->set("RADESYS", record.get(keys.radesys));
    metadata->set("CUNIT1", record.get(keys.cunit1));
    metadata->set("CUNIT2", record.get(keys.cunit2));
    return metadata;
}

std::shared_ptr<daf::base::PropertyList> getOldSipMetadata(table::BaseRecord const& record) {
    // Cannot use a PersistenceHelper for the SIP terms because the length of each SIP array
    // must be read from the schema
    afw::table::Key<table::Array<double>> kA;
    afw::table::Key<table::Array<double>> kB;
    afw::table::Key<table::Array<double>> kAp;
    afw::table::Key<table::Array<double>> kBp;
    try {
        kA = record.getSchema()["A"];
        kB = record.getSchema()["B"];
        kAp = record.getSchema()["Ap"];
        kBp = record.getSchema()["Bp"];
    } catch (...) {
        throw LSST_EXCEPT(afw::table::io::MalformedArchiveError,
                          "Incorrect schema for TanWcs distortion terms");
    }

    // Adding 0.5 and truncating the result guarantees we'll get the right answer
    // for small ints even when round-off error is involved.
    int nA = static_cast<int>(std::sqrt(kA.getSize() + 0.5));
    int nB = static_cast<int>(std::sqrt(kB.getSize() + 0.5));
    int nAp = static_cast<int>(std::sqrt(kAp.getSize() + 0.5));
    int nBp = static_cast<int>(std::sqrt(kBp.getSize() + 0.5));
    if (nA * nA != kA.getSize()) {
        throw LSST_EXCEPT(table::io::MalformedArchiveError, "Forward X SIP matrix is not square.");
    }
    if (nB * nB != kB.getSize()) {
        throw LSST_EXCEPT(table::io::MalformedArchiveError, "Forward Y SIP matrix is not square.");
    }
    if (nAp * nAp != kAp.getSize()) {
        throw LSST_EXCEPT(table::io::MalformedArchiveError, "Reverse X SIP matrix is not square.");
    }
    if (nBp * nBp != kBp.getSize()) {
        throw LSST_EXCEPT(table::io::MalformedArchiveError, "Reverse Y SIP matrix is not square.");
    }
    Eigen::Map<Eigen::MatrixXd const> mapA((record)[kA].getData(), nA, nA);
    Eigen::Map<Eigen::MatrixXd const> mapB((record)[kB].getData(), nB, nB);
    Eigen::Map<Eigen::MatrixXd const> mapAp((record)[kAp].getData(), nAp, nAp);
    Eigen::Map<Eigen::MatrixXd const> mapBp((record)[kBp].getData(), nBp, nBp);

    auto metadata = makeSipMatrixMetadata(mapA, "A");
    metadata->combine(makeSipMatrixMetadata(mapB, "B"));
    metadata->combine(makeSipMatrixMetadata(mapAp, "AP"));
    metadata->combine(makeSipMatrixMetadata(mapBp, "BP"));
    return metadata;
}

}  // namespace
}  // namespace geom
}  // namespace afw
}  // namespace lsst
