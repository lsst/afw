// -*- lsst-c++ -*-

#include "boost/format.hpp"

#include "wcslib/wcs.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/WcsRecordFactory.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst { namespace afw { namespace image {

namespace {

typedef std::map<std::string,WcsRecordFactory*> Registry;

Registry & getRegistry() {
    static Registry registry;
    return registry;
}

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

WcsRecordFactory registration("Base");

} // anonymous

class WcsRecordOutputGenerator : public table::RecordOutputGenerator {
public:

    WcsRecordOutputGenerator(Wcs const & wcs, table::Schema const & schema, int recordCount) :
        table::RecordOutputGenerator(schema, recordCount),
        _wcs(&wcs)
        {}
    
    virtual void fill(table::BaseRecord & record) {
        WcsSchema const & keys = WcsSchema::get();
        record.set(keys.crval.getX(), _wcs->_wcsInfo[0].crval[0]);
        record.set(keys.crval.getY(), _wcs->_wcsInfo[0].crval[1]);
        record.set(keys.crpix, _wcs->getPixelOrigin());
        Eigen::Matrix2d cdIn = _wcs->getCDMatrix();
        Eigen::Map<Eigen::Matrix2d> cdOut(record[keys.cd].getData());
        cdOut = cdIn;
        record.set(keys.ctype1, std::string(_wcs->_wcsInfo[0].ctype[0]));
        record.set(keys.ctype2, std::string(_wcs->_wcsInfo[0].ctype[1]));
        record.set(keys.equinox, _wcs->_wcsInfo[0].equinox);
        record.set(keys.radesys, std::string(_wcs->_wcsInfo[0].radesys));
        record.set(keys.cunit1, std::string(_wcs->_wcsInfo[0].cunit[0]));
        record.set(keys.cunit2, std::string(_wcs->_wcsInfo[0].cunit[1]));
    }

protected:
    Wcs const * _wcs;
};

table::Schema WcsRecordFactory::getSchema() {
    return WcsSchema::get().schema;
}

WcsRecordFactory::WcsRecordFactory(std::string const & name) {
    getRegistry()[name] = this;
}

PTR(Wcs) WcsRecordFactory::operator()(table::RecordInputGeneratorSet const & inputs) const {
    CONST_PTR(afw::table::BaseRecord) record = inputs.generators.front()->next();
    PTR(Wcs) result(new Wcs(*record));
    return result;
}

bool Wcs::hasRecordPersistence() const {
    if (_wcsInfo[0].naxis != 2) return false;
    return true;
}

table::RecordOutputGeneratorSet Wcs::writeToRecords() const {
    if (!hasRecordPersistence()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Record persistence is not implemented for this Wcs"
        );
    }
    afw::table::RecordOutputGeneratorSet result("Base");
    result.generators.push_back(
        boost::make_shared<WcsRecordOutputGenerator>(*this, WcsSchema::get().schema, 1)
    );
    return result;
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
    _setWcslibParams();
    WcsSchema const & keys = WcsSchema::get();
    if (!record.getSchema().contains(keys.schema)) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Incorrect schema for Wcs persistence"
        );
    }
    Eigen::Matrix2d cd = Eigen::Map<Eigen::Matrix2d const>(record[keys.cd].getData());
    initWcsLib(
        record.get(keys.crval), record.get(keys.crpix), cd,
        record.get(keys.ctype1), record.get(keys.ctype2),
        record.get(keys.equinox), record.get(keys.radesys),
        record.get(keys.cunit1), record.get(keys.cunit2)
    );
    _initWcs();
}

PTR(Wcs) Wcs::readFromRecords(afw::table::RecordInputGeneratorSet const & inputs) {
    Registry::iterator i = getRegistry().find(inputs.name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            boost::str(boost::format("No WcsRecordFactory with name '%s'") % inputs.name)
        );
    }
    return (*i->second)(inputs);
}

}}} // namespace lsst::afw::image
