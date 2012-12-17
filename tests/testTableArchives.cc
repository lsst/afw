#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-archives
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#define PRINT_CATALOGS 0

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "boost/filesystem.hpp"

#include "lsst/utils/ieee.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "ndarray.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

class Comparable : public Persistable {
public:

    virtual bool operator==(Comparable const & other) const = 0;

    bool operator!=(Comparable const & other) const {
        return !this->operator==(other);
    }

    virtual void stream(std::ostream & os) const = 0;

    friend std::ostream & operator<<(std::ostream & os, Comparable const & b) {
        b.stream(os);
        return os;
    }

};

class ExampleA : public public PersistableFacade<ExampleA>, Comparable {
public:

    int var1;
    double var2;
    ndarray::Array<float,1,1> var3;

    virtual bool operator==(Comparable const & other) const {
        ExampleA const * b = dynamic_cast<ExampleA const *>(&other);
        return b && var1 == b->var1 && var2 == b->var2 && ndarray::allclose(var3, b->var3);
    }

    virtual void stream(std::ostream & os) const {
        os << "ExampleA(var1=" << var1 << ", var2=" << var2 << ", var3=" << var3 << ")";
    }

    ExampleA(int v1, double v2, ndarray::Array<float,1,1> const & v3) : var1(v1), var2(v2), var3(v3) {}

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleA"; }

    virtual void write(OutputArchiveHandle & handle) const {
        Schema schema;
        Key<int> k1 = schema.addField<int>("var1", "doc for var1");
        Key<double> k2 = schema.addField<double>("var2", "doc for var2");
        Key< Array<float> > k3 = schema.addField< Array<float> >("var3", "doc for var3", var3.getSize<0>());
        BaseCatalog catalog = handle.makeCatalog(schema);
        PTR(BaseRecord) record = catalog.addNew();
        record->set(k1, var1);
        record->set(k2, var2);
        record->set(k3, var3);
        handle.saveCatalog(catalog);
    }

    class Factory : public PersistableFactory {
    public:
        explicit Factory(std::string const & name) : PersistableFactory(name) {}
        
        virtual PTR(Persistable) read(InputArchive const & archive, CatalogVector const & catalogs) const {
            BaseRecord const & record = catalogs.front().front();
            Schema const & schema = record.getSchema();
            Key<int> k1 = schema["var1"];
            Key<double> k2 = schema["var2"];
            Key< Array<float> > k3 = schema["var3"];
            PTR(Persistable) r(new ExampleA(record.get(k1), record.get(k2), ndarray::copy(record.get(k3))));
            return r;
        }
    };
};

class ExampleB : public PersistableFacade<ExampleB>, public Comparable {
public:

    int var1;
    std::vector<double> var2;

    virtual bool operator==(Comparable const & other) const {
        ExampleB const * b = dynamic_cast<ExampleB const *>(&other);
        return b && var1 == b->var1 && var2 == b->var2;
    }

    virtual void stream(std::ostream & os) const {
        os << "ExampleB(var1=" << var1 << ", var2=[";
        for (std::vector<double>::const_iterator i = var2.begin(); i != var2.end(); ++i) {
            os << (*i) << ", ";
        }
        os << "])";
    }

    ExampleB(int v1, std::vector<double> const & v2) : var1(v1), var2(v2) {}

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleB"; }

    virtual void write(OutputArchiveHandle & handle) const {
        Schema schema1;
        Key<int> k1 = schema1.addField<int>("var1", "doc for var1");
        Schema schema2;
        Key<double> k2 = schema2.addField<double>("var2", "doc for var2");
        BaseCatalog catalog1 = handle.makeCatalog(schema1);
        PTR(BaseRecord) record1 = catalog1.addNew();
        record1->set(k1, var1);
        BaseCatalog catalog2 = handle.makeCatalog(schema2);
        for (std::vector<double>::const_iterator i = var2.begin(); i != var2.end(); ++i) {
            catalog2.addNew()->set(k2, *i);
        }
        handle.saveCatalog(catalog1);
        handle.saveCatalog(catalog2);
    }

    class Factory : public PersistableFactory {
    public:
        explicit Factory(std::string const & name) : PersistableFactory(name) {}
        
        virtual PTR(Persistable) read(InputArchive const & archive, CatalogVector const & catalogs) const {
            BaseRecord const & record1 = catalogs.front().front();
            Schema const & schema1 = record1.getSchema();
            Key<int> k1 = schema1["var1"];
            BaseCatalog const & catalog2 = catalogs.back();
            Schema const & schema2 = catalog2.getSchema();
            std::vector<double> v2;
            Key<double> k2 = schema2["var2"];
            for (BaseCatalog::const_iterator i = catalog2.begin(); i != catalog2.end(); ++i) {
                v2.push_back(i->get(k2));
            }
            PTR(Persistable) r(new ExampleB(record1.get(k1), v2));
            return r;
        }
    };
};

class ExampleC : public PersistableFacade<ExampleC>, public Comparable {
public:

    int var1;
    PTR(Comparable) var2;
    PTR(Comparable) var3;

    virtual bool operator==(Comparable const & other) const {
        ExampleC const * c = dynamic_cast<ExampleC const *>(&other);
        if (!c) return false;
        if ((!var2 && c->var2) || (var2 && !c->var2)) return false;
        if ((!var3 && c->var3) || (var3 && !c->var3)) return false;
        return var1 == c->var1
            && (var2 == c->var2 || (*var2) == (*c->var2))
            && (var3 == c->var3 || (*var3) == (*c->var3));
    }

    virtual void stream(std::ostream & os) const {
        os << "ExampleC(var1=" << var1 << ", var2=";
        if (var2) {
            os << (*var2);
        } else {
            os << "0";
        }
        os << ", var3=";
        if (var3) {
            os << (*var3);
        } else {
            os << "0";
        }
        os << ")";
    }

    ExampleC(int v1, PTR(Comparable) v2 = PTR(Comparable)(), PTR(Comparable) v3 = PTR(Comparable)())
        : var1(v1), var2(v2), var3(v3) {}

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleC"; }

    virtual void write(OutputArchiveHandle & handle) const {
        Schema schema;
        Key<int> k1 = schema.addField<int>("var1", "doc for var1");
        Key<int> k2 = schema.addField<int>("var2", "doc for var2");
        Key<int> k3 = schema.addField<int>("var3", "doc for var3");
        int id2 = handle.put(var2.get());
        int id3 = handle.put(var3.get());
        BaseCatalog catalog = handle.makeCatalog(schema);
        PTR(BaseRecord) record = catalog.addNew();
        record->set(k1, var1);
        record->set(k2, id2);
        record->set(k3, id3);
        handle.saveCatalog(catalog);
    }

    class Factory : public PersistableFactory {
    public:
        explicit Factory(std::string const & name) : PersistableFactory(name) {}
        
        virtual PTR(Persistable) read(InputArchive const & archive, CatalogVector const & catalogs) const {
            BaseRecord const & record = catalogs.front().front();
            Schema const & schema = record.getSchema();
            Key<int> k1 = schema["var1"];
            Key<int> k2 = schema["var2"];
            Key<int> k3 = schema["var3"];
            PTR(Comparable) v2 = boost::dynamic_pointer_cast<Comparable>(archive.get(record.get(k2)));
            PTR(Comparable) v3 = boost::dynamic_pointer_cast<Comparable>(archive.get(record.get(k3)));
            PTR(Persistable) r(new ExampleC(record.get(k1), v2, v3));
            return r;
        }
    };
};

static ExampleA::Factory const registrationA("ExampleA");
static ExampleB::Factory const registrationB("ExampleB");
static ExampleC::Factory const registrationC("ExampleC");

template <int M, int N>
std::vector< ndarray::Vector<PTR(Comparable),M> >
roundtripAndCompare(
    ndarray::Vector<PTR(Comparable),M> const & inputs,
    ndarray::Vector<int,N> const & expectedSizes
) {
    std::vector< ndarray::Vector<PTR(Comparable),M> > outputs;
    ndarray::Vector<int,M> inputIds;
    OutputArchive outArchive;
    for (int i = 0; i < M; ++i) {
        inputIds[i] = outArchive.put(inputs[i].get());
    }

    BOOST_CHECK_EQUAL(outArchive.countCatalogs(), N + 1);
    CatalogVector catalogs;
    for (int j = 1; j <= N; ++j) {
        catalogs.push_back(outArchive.getCatalog(j));
        BOOST_CHECK_EQUAL(expectedSizes[j-1], int(catalogs.back().size()));
    }

#if PRINT_CATALOGS
    std::cerr << "Index Catalog:\n";
    ArchiveIndexSchema const & keys = ArchiveIndexSchema::get();
    BaseCatalog index = outArchive.getIndexCatalog();
    for (BaseCatalog::iterator iter = index.begin(); iter != index.end(); ++iter) {
        std::cerr << "id=" << iter->get(keys.id);
        std::cerr << " name='" << iter->get(keys.name);
        std::cerr << "' catArchive=" << iter->get(keys.catArchive);
        std::cerr << " catPersistable=" << iter->get(keys.catPersistable);
        std::cerr << " row0=" << iter->get(keys.row0);
        std::cerr << " nRows=" << iter->get(keys.nRows);
        std::cerr << std::endl;
    }
#endif

    // Round-trip and compare once, just transferring catalogs from output archive to input archive
    outputs.push_back(ndarray::Vector<PTR(Comparable),M>());
    InputArchive inArchive1(outArchive.getIndexCatalog(), catalogs);
    for (int i = 0; i < M; ++i) {
        PTR(Comparable) outObj = boost::dynamic_pointer_cast<Comparable>(inArchive1.get(inputIds[i]));
        BOOST_CHECK_EQUAL(*outObj, *inputs[i]);
        outputs.back()[i] = outObj;
    }

    // Round-trip and compare again, via an in-memory FITS file
    outputs.push_back(ndarray::Vector<PTR(Comparable),M>());
    fits::MemFileManager manager;
    fits::Fits outFits2(manager, "w", fits::Fits::AUTO_CHECK);
    outArchive.writeFits(outFits2);
    outFits2.closeFile();
    fits::Fits inFits2(manager, "r", fits::Fits::AUTO_CHECK);
    inFits2.setHdu(0);
    InputArchive inArchive2 = InputArchive::readFits(inFits2);
    inFits2.closeFile();
    for (int i = 0; i < M; ++i) {
        PTR(Comparable) outObj = boost::dynamic_pointer_cast<Comparable>(inArchive2.get(inputIds[i]));
        BOOST_CHECK_EQUAL(*outObj, *inputs[i]);
        outputs.back()[i] = outObj;
    }

    return outputs;
}

} // anonymous

}}}} // namespace lsst::afw::table::io

BOOST_AUTO_TEST_CASE(Simple) {
    using namespace lsst::afw::table::io;

    ndarray::Array<float,1,1> av1 = ndarray::allocate(2);
    av1[0] = 1.1;
    av1[1] = 1.2;
    PTR(Comparable) a1(new ExampleA(3, 2.5, av1));
    roundtripAndCompare(ndarray::makeVector(a1), ndarray::makeVector(1));

    std::vector<double> bv1;
    bv1.push_back(2.1);
    bv1.push_back(2.2);
    PTR(Comparable) b1(new ExampleB(4, bv1));

    roundtripAndCompare(ndarray::makeVector(b1), ndarray::makeVector(1, 2));

    roundtripAndCompare(ndarray::makeVector(a1, b1), ndarray::makeVector(1, 1, 2));

    roundtripAndCompare(ndarray::makeVector(b1, a1), ndarray::makeVector(1, 2, 1));
}

BOOST_AUTO_TEST_CASE(CompatibleSchemas) {
    // when we save two objects with the same name and schema, they should go in the same catalog
    using namespace lsst::afw::table::io;

    ndarray::Array<float,1,1> av1 = ndarray::allocate(2);
    av1[0] = 1.1;
    av1[1] = 1.2;
    PTR(Comparable) a1(new ExampleA(3, 2.5, av1));

    ndarray::Array<float,1,1> av2 = ndarray::allocate(2);
    av1[0] = 2.1;
    av1[1] = 2.2;
    PTR(Comparable) a2(new ExampleA(4, 3.5, av2));

    roundtripAndCompare(ndarray::makeVector(a1, a2), ndarray::makeVector(2));

    std::vector<double> bv1;
    bv1.push_back(2.1);
    bv1.push_back(2.2);
    bv1.push_back(2.3);
    PTR(Comparable) b1(new ExampleB(4, bv1));

    roundtripAndCompare(ndarray::makeVector(a1, a2, b1), ndarray::makeVector(2, 1, 3));

    std::vector<double> bv2;
    bv2.push_back(3.1);
    bv2.push_back(3.2);
    bv2.push_back(3.3);
    bv2.push_back(3.4);
    PTR(Comparable) b2(new ExampleB(5, bv2));

    roundtripAndCompare(ndarray::makeVector(a1, a2, b1, b2), ndarray::makeVector(2, 2, 7));
}

BOOST_AUTO_TEST_CASE(IncompatibleSchemas) {
    // when we save two objects with the same name and different schemas, they cannot go in the same catalog
    using namespace lsst::afw::table::io;

    ndarray::Array<float,1,1> av1 = ndarray::allocate(2);
    av1[0] = 1.1;
    av1[1] = 1.2;
    PTR(Comparable) a1(new ExampleA(3, 2.5, av1));

    ndarray::Array<float,1,1> av2 = ndarray::allocate(3);
    av1[0] = 2.1;
    av1[1] = 2.2;
    av1[1] = 2.3;
    PTR(Comparable) a2(new ExampleA(4, 3.5, av2));

    roundtripAndCompare(ndarray::makeVector(a1, a2), ndarray::makeVector(1, 1));

    std::vector<double> bv1;
    bv1.push_back(2.1);
    bv1.push_back(2.2);
    bv1.push_back(2.3);
    PTR(Comparable) b1(new ExampleB(4, bv1));

    roundtripAndCompare(ndarray::makeVector(a1, a2, b1), ndarray::makeVector(1, 1, 1, 3));
}

BOOST_AUTO_TEST_CASE(Nested) {
    using namespace lsst::afw::table::io;

    PTR(Comparable) c1(new ExampleC(1));
    roundtripAndCompare(ndarray::makeVector(c1), ndarray::makeVector(1));

    ndarray::Array<float,1,1> av2 = ndarray::allocate(2);
    av2[0] = 1.1;
    av2[1] = 1.2;
    PTR(Comparable) a2(new ExampleA(3, 2.5, av2));
    PTR(Comparable) c2(new ExampleC(2, a2, a2));

    std::vector< ndarray::Vector<PTR(Comparable),1> > r2
        = roundtripAndCompare(ndarray::makeVector(c2), ndarray::makeVector(1,1));
    for (std::size_t i = 0; i < r2.size(); ++i) {
        PTR(ExampleC) c3 = boost::dynamic_pointer_cast<ExampleC>(r2[i][0]);
        BOOST_REQUIRE(c3);
        BOOST_CHECK_EQUAL(c3->var2, c3->var3);
    }

    std::vector< ndarray::Vector<PTR(Comparable),2> > r3
        = roundtripAndCompare(ndarray::makeVector(a2,c2), ndarray::makeVector(1,1));
    for (std::size_t i = 0; i < r3.size(); ++i) {
        PTR(ExampleC) c3 = boost::dynamic_pointer_cast<ExampleC>(r3[i][1]);
        BOOST_CHECK_EQUAL(c3->var3, c3->var3);
        BOOST_CHECK_EQUAL(c3->var3, r3[i][0]);
    }

}
