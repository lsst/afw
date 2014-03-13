// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

%module testTableArchivesLib

%pythonnondynamic;
%naturalvar;  // use const reference typemaps

%include "lsst/p_lsstSwig.i"

%lsst_exceptions()

%{
// It's really horrible that I have to include all this just to get this simple
// test module to build, but that's the way it is right now.
#include "lsst/afw/detection.h"
#include "lsst/afw/image.h"
#include "lsst/pex/logging.h"
#include "lsst/afw/math.h"
#include "lsst/afw/cameraGeom.h"
%}

%import "lsst/afw/detection/detectionLib.i"

%shared_ptr(DummyPsf);

%inline {

// not really a Psf, just a Persistable we can stuff in an Exposure
class DummyPsf : public lsst::afw::detection::Psf {
public:

    virtual PTR(lsst::afw::detection::Psf) clone() const {
        return PTR(lsst::afw::detection::Psf)(new DummyPsf(_x));
    }

    virtual bool isPersistable() const { return true; }

    double getValue() const { return _x; }

    explicit DummyPsf(double x) : _x(x) {}

protected:

    virtual PTR(Image) doComputeKernelImage(
        lsst::afw::geom::Point2D const & ccdXY,
        lsst::afw::image::Color const & color
    ) const {
        return PTR(Image)();
    }

    virtual double doComputeApertureFlux(
        double radius,
        lsst::afw::geom::Point2D const & ccdXY,
        lsst::afw::image::Color const & color
    ) const {
        return 0.0;
    }

    virtual lsst::afw::geom::ellipses::Quadrupole doComputeShape(
        lsst::afw::geom::Point2D const & ccdXY,
        lsst::afw::image::Color const & color
    ) const {
        return lsst::afw::geom::ellipses::Quadrupole();
    }

    virtual std::string getPersistenceName() const { return "DummyPsf"; }

    virtual std::string getPythonModule() const { return "testTableArchivesLib"; }

    virtual void write(OutputArchiveHandle & handle) const;

    double _x;
};

}

%{
#include "boost/make_shared.hpp"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace {

struct DummyPsfPersistenceHelper : private boost::noncopyable {
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<double> x;

    static DummyPsfPersistenceHelper const & get() {
        static DummyPsfPersistenceHelper instance;
        return instance;
    }

private:
    DummyPsfPersistenceHelper() :
        schema(),
        x(schema.addField<double>("x", "dummy parameter"))
    {
        schema.getCitizen().markPersistent();
    }
};

class DummyPsfFactory : public lsst::afw::table::io::PersistableFactory {
public:

    virtual PTR(lsst::afw::table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        static DummyPsfPersistenceHelper const & keys = DummyPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const & record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        return boost::make_shared<DummyPsf>(
            record.get(keys.x)
        );
    }

    DummyPsfFactory(std::string const & name) : lsst::afw::table::io::PersistableFactory(name) {}

};

DummyPsfFactory registration("DummyPsf");

} // anonymous

void DummyPsf::write(OutputArchiveHandle & handle) const {
    static DummyPsfPersistenceHelper const & keys = DummyPsfPersistenceHelper::get();
    lsst::afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(lsst::afw::table::BaseRecord) record = catalog.addNew();
    (*record).set(keys.x, _x);
    handle.saveCatalog(catalog);
}

%}

%lsst_persistable(DummyPsf);
