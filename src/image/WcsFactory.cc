// -*- lsst-c++ -*-

#include "boost/format.hpp"

#include "wcslib/wcs.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/WcsFactory.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst { namespace afw { namespace image {

namespace {

// Read-only singleton struct containing the schema and keys that a simple Wcs is mapped
// to in record persistence.
struct WcsSchema : private boost::noncopyable {
    table::Schema schema;
    table::Key< table::Point<double> > crval;
    table::Key< table::Point<double> > crpix;
    table::Key< table::Array<double> > cd;
    table::Key<std::string> ctype1;
    table::Key<std::string> ctype2;
    table::Key<double> equinox;
    table::Key<std::string> radesys;
    table::Key<std::string> cunit1;
    table::Key<std::string> cunit2;

    static WcsSchema const & get() {
        static WcsSchema instance;
        return instance;
    };

private:
    WcsSchema() :
        schema(),
        crval(schema.addField< table::Point<double> >("crval", "celestial reference point")),
        crpix(schema.addField< table::Point<double> >("crpix", "pixel reference point")),
        cd(schema.addField< table::Array<double> >(
               "cd", "linear transform matrix, ordered (1_1, 2_1, 1_2, 2_2)", 4)),
        ctype1(schema.addField< std::string >("ctype1", "coordinate type", 72)),
        ctype2(schema.addField< std::string >("ctype2", "coordinate type", 72)),
        equinox(schema.addField< double >("equinox", "equinox of coordinates")),
        radesys(schema.addField< std::string >("radesys", "coordinate system for equinox", 72)),
        cunit1(schema.addField< std::string >("cunit1", "coordinate units", 72)),
        cunit2(schema.addField< std::string >("cunit2", "coordinate units", 72))
    {
        schema.getCitizen().markPersistent();
    }
};

WcsFactory registration("Wcs");

} // anonymous

std::string Wcs::getPersistenceName() const { return "Wcs"; }

void Wcs::write(OutputArchive::Handle & handle) const {
    WcsSchema const & keys = WcsSchema::get();
    afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(afw::table::BaseRecord) record = catalog.addNew();
    record->set(keys.crval, getSkyOrigin()->getPosition(afw::geom::degrees));
    record->set(keys.crpix, getPixelOrigin());
    Eigen::Matrix2d cdIn = getCDMatrix();
    Eigen::Map<Eigen::Matrix2d> cdOut((*record)[keys.cd].getData());
    cdOut = cdIn;
    record->set(keys.ctype1, std::string(_wcsInfo[0].ctype[0]));
    record->set(keys.ctype2, std::string(_wcsInfo[0].ctype[1]));
    record->set(keys.equinox, _wcsInfo[0].equinox);
    record->set(keys.radesys, std::string(_wcsInfo[0].radesys));
    record->set(keys.cunit1, std::string(_wcsInfo[0].cunit[0]));
    record->set(keys.cunit2, std::string(_wcsInfo[0].cunit[1]));
    handle.saveCatalog(catalog);
}

bool Wcs::isPersistable() const {
    if (_wcsInfo[0].naxis != 2) return false;
    if (std::strcmp(_wcsInfo[0].cunit[0], "deg") != 0) return false;
    if (std::strcmp(_wcsInfo[0].cunit[1], "deg") != 0) return false;
    return true;
}

Wcs::Wcs(afw::table::BaseRecord const & record) :
    daf::base::Citizen(typeid(this)),
    _wcsInfo(NULL),
    _nWcsInfo(0),
    _relax(0),
    _wcsfixCtrl(0),
    _wcshdrCtrl(0),
    _nReject(0),
    _coordSystem(static_cast<afw::coord::CoordSystem>(-1))
{
    WcsSchema const & keys = WcsSchema::get();
    if (!record.getSchema().contains(keys.schema)) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Incorrect schema for Wcs persistence"
        );
    }
    _setWcslibParams();
    Eigen::Matrix2d cd = Eigen::Map<Eigen::Matrix2d const>(record[keys.cd].getData());
    initWcsLib(
        record.get(keys.crval), record.get(keys.crpix), cd,
        record.get(keys.ctype1), record.get(keys.ctype2),
        record.get(keys.equinox), record.get(keys.radesys),
        record.get(keys.cunit1), record.get(keys.cunit2)
    );
    _initWcs();
}

PTR(table::io::Persistable)
WcsFactory::read(InputArchive const & inputs, CatalogVector const & catalogs) const {
    WcsSchema const & keys = WcsSchema::get();
    assert(catalogs.front().size() == 1u);
    assert(catalogs.front().getSchema() == keys.schema);
    PTR(Wcs) result(new Wcs(catalogs.front().front()));
    return result;
}

WcsFactory::WcsFactory(std::string const & name) : table::io::PersistableFactory(name) {}

}}} // namespace lsst::afw::image
