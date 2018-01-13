#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table - archives
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#define PRINT_CATALOGS 0

#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <map>

#include "boost/filesystem.hpp"
#include "Eigen/Core"

#include "lsst/utils/Utils.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/HeavyFootprint.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "ndarray/eigen.h"

#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"

#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/fitsDefaults.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

namespace {

class Comparable : public Persistable {
public:
    virtual bool operator==(Comparable const &other) const = 0;

    bool operator!=(Comparable const &other) const { return !this->operator==(other); }

    virtual void stream(std::ostream &os) const = 0;

    friend std::ostream &operator<<(std::ostream &os, Comparable const &b) {
        b.stream(os);
        return os;
    }
};

class ExampleA : public PersistableFacade<ExampleA>, public Comparable {
public:
    int var1;
    double var2;
    ndarray::Array<float, 1, 1> var3;

    virtual bool operator==(Comparable const &other) const {
        ExampleA const *b = dynamic_cast<ExampleA const *>(&other);
        return b && var1 == b->var1 && var2 == b->var2 && ndarray::allclose(var3, b->var3);
    }

    virtual void stream(std::ostream &os) const {
        os << "ExampleA(var1=" << var1 << ", var2=" << var2 << ", var3=" << var3 << ")";
    }

    ExampleA(int v1, double v2, ndarray::Array<float, 1, 1> const &v3) : var1(v1), var2(v2), var3(v3) {}

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleA"; }

    virtual void write(OutputArchiveHandle &handle) const {
        Schema schema;
        Key<int> k1 = schema.addField<int>("var1", "doc for var1");
        Key<double> k2 = schema.addField<double>("var2", "doc for var2");
        Key<Array<float>> k3 = schema.addField<Array<float>>("var3", "doc for var3", var3.getSize<0>());
        BaseCatalog catalog = handle.makeCatalog(schema);
        std::shared_ptr<BaseRecord> record = catalog.addNew();
        record->set(k1, var1);
        record->set(k2, var2);
        record->set(k3, var3);
        handle.saveCatalog(catalog);
    }

    class Factory : public PersistableFactory {
    public:
        explicit Factory(std::string const &name) : PersistableFactory(name) {}

        virtual std::shared_ptr<Persistable> read(InputArchive const &archive,
                                                  CatalogVector const &catalogs) const {
            BaseRecord const &record = catalogs.front().front();
            Schema const &schema = record.getSchema();
            Key<int> k1 = schema["var1"];
            Key<double> k2 = schema["var2"];
            Key<Array<float>> k3 = schema["var3"];
            std::shared_ptr<Persistable> r(
                    new ExampleA(record.get(k1), record.get(k2), ndarray::copy(record.get(k3))));
            return r;
        }
    };
};

class ExampleB : public PersistableFacade<ExampleB>, public Comparable {
public:
    int var1;
    std::vector<double> var2;

    virtual bool operator==(Comparable const &other) const {
        ExampleB const *b = dynamic_cast<ExampleB const *>(&other);
        return b && var1 == b->var1 && var2 == b->var2;
    }

    virtual void stream(std::ostream &os) const {
        os << "ExampleB(var1=" << var1 << ", var2=[";
        for (std::vector<double>::const_iterator i = var2.begin(); i != var2.end(); ++i) {
            os << (*i) << ", ";
        }
        os << "])";
    }

    ExampleB(int v1, std::vector<double> const &v2) : var1(v1), var2(v2) {}

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleB"; }

    virtual void write(OutputArchiveHandle &handle) const {
        Schema schema1;
        Key<int> k1 = schema1.addField<int>("var1", "doc for var1");
        Schema schema2;
        Key<double> k2 = schema2.addField<double>("var2", "doc for var2");
        BaseCatalog catalog1 = handle.makeCatalog(schema1);
        std::shared_ptr<BaseRecord> record1 = catalog1.addNew();
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
        explicit Factory(std::string const &name) : PersistableFactory(name) {}

        virtual std::shared_ptr<Persistable> read(InputArchive const &archive,
                                                  CatalogVector const &catalogs) const {
            BaseRecord const &record1 = catalogs.front().front();
            Schema const &schema1 = record1.getSchema();
            Key<int> k1 = schema1["var1"];
            BaseCatalog const &catalog2 = catalogs.back();
            Schema const &schema2 = catalog2.getSchema();
            std::vector<double> v2;
            Key<double> k2 = schema2["var2"];
            for (BaseCatalog::const_iterator i = catalog2.begin(); i != catalog2.end(); ++i) {
                v2.push_back(i->get(k2));
            }
            std::shared_ptr<Persistable> r(new ExampleB(record1.get(k1), v2));
            return r;
        }
    };
};

class ExampleC : public PersistableFacade<ExampleC>, public Comparable {
public:
    int var1;
    std::shared_ptr<Comparable> var2;
    std::shared_ptr<Comparable> var3;

    virtual bool operator==(Comparable const &other) const {
        ExampleC const *c = dynamic_cast<ExampleC const *>(&other);
        if (!c) return false;
        if ((!var2 && c->var2) || (var2 && !c->var2)) return false;
        if ((!var3 && c->var3) || (var3 && !c->var3)) return false;
        return var1 == c->var1 && (var2 == c->var2 || (*var2) == (*c->var2)) &&
               (var3 == c->var3 || (*var3) == (*c->var3));
    }

    virtual void stream(std::ostream &os) const {
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

    ExampleC(int v1, std::shared_ptr<Comparable> v2 = std::shared_ptr<Comparable>(),
             std::shared_ptr<Comparable> v3 = std::shared_ptr<Comparable>())
            : var1(v1), var2(v2), var3(v3) {}

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleC"; }

    virtual void write(OutputArchiveHandle &handle) const {
        Schema schema;
        Key<int> k1 = schema.addField<int>("var1", "doc for var1");
        Key<int> k2 = schema.addField<int>("var2", "doc for var2");
        Key<int> k3 = schema.addField<int>("var3", "doc for var3");
        int id2 = handle.put(var2.get());
        int id3 = handle.put(var3.get());
        BaseCatalog catalog = handle.makeCatalog(schema);
        std::shared_ptr<BaseRecord> record = catalog.addNew();
        record->set(k1, var1);
        record->set(k2, id2);
        record->set(k3, id3);
        handle.saveCatalog(catalog);
    }

    class Factory : public PersistableFactory {
    public:
        explicit Factory(std::string const &name) : PersistableFactory(name) {}

        virtual std::shared_ptr<Persistable> read(InputArchive const &archive,
                                                  CatalogVector const &catalogs) const {
            BaseRecord const &record = catalogs.front().front();
            Schema const &schema = record.getSchema();
            Key<int> k1 = schema["var1"];
            Key<int> k2 = schema["var2"];
            Key<int> k3 = schema["var3"];
            std::shared_ptr<Comparable> v2 =
                    std::dynamic_pointer_cast<Comparable>(archive.get(record.get(k2)));
            std::shared_ptr<Comparable> v3 =
                    std::dynamic_pointer_cast<Comparable>(archive.get(record.get(k3)));
            std::shared_ptr<Persistable> r(new ExampleC(record.get(k1), v2, v3));
            return r;
        }
    };
};


class ExampleD : public PersistableFacade<ExampleD>, public Comparable {
public:

    static std::shared_ptr<ExampleD> get() {
        static std::shared_ptr<ExampleD> instance(new ExampleD());
        return instance;
    }

    ExampleD(ExampleD const &) = delete;
    ExampleD(ExampleD &&) = delete;
    ExampleD & operator=(ExampleD const &) = delete;
    ExampleD & operator=(ExampleD &&) = delete;

    virtual bool operator==(Comparable const &other) const {
        return static_cast<Comparable const *>(this) == &other;
    }

    virtual void stream(std::ostream &os) const {
        os << "ExampleD()";
    }

    virtual bool isPersistable() const { return true; }

    virtual std::string getPersistenceName() const { return "ExampleD"; }

    virtual void write(OutputArchiveHandle &handle) const {
        handle.saveEmpty();
    }

    class Factory : public PersistableFactory {
    public:
        explicit Factory(std::string const &name) : PersistableFactory(name) {}

        virtual std::shared_ptr<Persistable> read(InputArchive const &archive,
                                                  CatalogVector const &catalogs) const {
            LSST_ARCHIVE_ASSERT(catalogs.size() == 0u);
            return ExampleD::get();
        }
    };

private:
    ExampleD() {}
};


static ExampleA::Factory const registrationA("ExampleA");
static ExampleB::Factory const registrationB("ExampleB");
static ExampleC::Factory const registrationC("ExampleC");
static ExampleD::Factory const registrationD("ExampleD");


template <int M, int N>
std::vector<ndarray::Vector<std::shared_ptr<Comparable>, M>> roundtripAndCompare(
        ndarray::Vector<std::shared_ptr<Comparable>, M> const &inputs,
        ndarray::Vector<ndarray::Size, N> const &expectedSizes) {
    std::vector<ndarray::Vector<std::shared_ptr<Comparable>, M>> outputs;
    ndarray::Vector<int, M> inputIds;
    OutputArchive outArchive;
    for (int i = 0; i < M; ++i) {
        inputIds[i] = outArchive.put(inputs[i].get());
    }

    BOOST_CHECK_EQUAL(outArchive.countCatalogs(), N + 1);
    CatalogVector catalogs;
    for (int j = 1; j <= N; ++j) {
        catalogs.push_back(outArchive.getCatalog(j));
        BOOST_CHECK_EQUAL(expectedSizes[j - 1], ndarray::Size(catalogs.back().size()));
    }

#if PRINT_CATALOGS
    std::cerr << "Index Catalog:\n";
    ArchiveIndexSchema const &keys = ArchiveIndexSchema::get();
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
    outputs.push_back(ndarray::Vector<std::shared_ptr<Comparable>, M>());
    InputArchive inArchive1(outArchive.getIndexCatalog(), catalogs);
    for (int i = 0; i < M; ++i) {
        std::shared_ptr<Comparable> outObj =
                std::dynamic_pointer_cast<Comparable>(inArchive1.get(inputIds[i]));
        BOOST_CHECK_EQUAL(*outObj, *inputs[i]);
        outputs.back()[i] = outObj;
    }

    // Round-trip and compare again, via an in-memory FITS file
    outputs.push_back(ndarray::Vector<std::shared_ptr<Comparable>, M>());
    fits::MemFileManager manager;
    fits::Fits outFits2(manager, "w", fits::Fits::AUTO_CHECK);
    outArchive.writeFits(outFits2);
    outFits2.closeFile();
    fits::Fits inFits2(manager, "r", fits::Fits::AUTO_CHECK);
    inFits2.setHdu(fits::DEFAULT_HDU);
    InputArchive inArchive2 = InputArchive::readFits(inFits2);
    inFits2.closeFile();
    for (int i = 0; i < M; ++i) {
        std::shared_ptr<Comparable> outObj =
                std::dynamic_pointer_cast<Comparable>(inArchive2.get(inputIds[i]));
        BOOST_CHECK_EQUAL(*outObj, *inputs[i]);
        outputs.back()[i] = outObj;
    }

    return outputs;
}

}  // anonymous
}
}
}
}  // namespace lsst::afw::table::io

BOOST_AUTO_TEST_CASE(Simple) {
    using namespace lsst::afw::table::io;

    ndarray::Array<float, 1, 1> av1 = ndarray::allocate(2);
    av1[0] = 1.1;
    av1[1] = 1.2;
    std::shared_ptr<Comparable> a1(new ExampleA(3, 2.5, av1));
    roundtripAndCompare(ndarray::makeVector(a1), ndarray::makeVector<ndarray::Size>(1));

    std::vector<double> bv1;
    bv1.push_back(2.1);
    bv1.push_back(2.2);
    std::shared_ptr<Comparable> b1(new ExampleB(4, bv1));

    roundtripAndCompare(ndarray::makeVector(b1), ndarray::makeVector<ndarray::Size>(1, 2));

    roundtripAndCompare(ndarray::makeVector(a1, b1), ndarray::makeVector<ndarray::Size>(1, 1, 2));

    roundtripAndCompare(ndarray::makeVector(b1, a1), ndarray::makeVector<ndarray::Size>(1, 2, 1));
}

BOOST_AUTO_TEST_CASE(CompatibleSchemas) {
    // when we save two objects with the same name and schema, they should go in the same catalog
    using namespace lsst::afw::table::io;

    ndarray::Array<float, 1, 1> av1 = ndarray::allocate(2);
    av1[0] = 1.1;
    av1[1] = 1.2;
    std::shared_ptr<Comparable> a1(new ExampleA(3, 2.5, av1));

    ndarray::Array<float, 1, 1> av2 = ndarray::allocate(2);
    av2[0] = 2.1;
    av2[1] = 2.2;
    std::shared_ptr<Comparable> a2(new ExampleA(4, 3.5, av2));

    roundtripAndCompare(ndarray::makeVector(a1, a2), ndarray::makeVector<ndarray::Size>(2));

    std::vector<double> bv1;
    bv1.push_back(2.1);
    bv1.push_back(2.2);
    bv1.push_back(2.3);
    std::shared_ptr<Comparable> b1(new ExampleB(4, bv1));

    roundtripAndCompare(ndarray::makeVector(a1, a2, b1), ndarray::makeVector<ndarray::Size>(2, 1, 3));

    std::vector<double> bv2;
    bv2.push_back(3.1);
    bv2.push_back(3.2);
    bv2.push_back(3.3);
    bv2.push_back(3.4);
    std::shared_ptr<Comparable> b2(new ExampleB(5, bv2));

    roundtripAndCompare(ndarray::makeVector(a1, a2, b1, b2), ndarray::makeVector<ndarray::Size>(2, 2, 7));
}

BOOST_AUTO_TEST_CASE(IncompatibleSchemas) {
    // when we save two objects with the same name and different schemas, they cannot go in the same catalog
    using namespace lsst::afw::table::io;

    ndarray::Array<float, 1, 1> av1 = ndarray::allocate(2);
    av1[0] = 1.1;
    av1[1] = 1.2;
    std::shared_ptr<Comparable> a1(new ExampleA(3, 2.5, av1));

    ndarray::Array<float, 1, 1> av2 = ndarray::allocate(3);
    av1[0] = 2.1;
    av1[1] = 2.2;
    av1[1] = 2.3;
    std::shared_ptr<Comparable> a2(new ExampleA(4, 3.5, av2));

    roundtripAndCompare(ndarray::makeVector(a1, a2), ndarray::makeVector<ndarray::Size>(1, 1));

    std::vector<double> bv1;
    bv1.push_back(2.1);
    bv1.push_back(2.2);
    bv1.push_back(2.3);
    std::shared_ptr<Comparable> b1(new ExampleB(4, bv1));

    roundtripAndCompare(ndarray::makeVector(a1, a2, b1), ndarray::makeVector<ndarray::Size>(1, 1, 1, 3));
}

BOOST_AUTO_TEST_CASE(Nested) {
    using namespace lsst::afw::table::io;

    std::shared_ptr<Comparable> c1(new ExampleC(1));
    roundtripAndCompare(ndarray::makeVector(c1), ndarray::makeVector<ndarray::Size>(1));

    ndarray::Array<float, 1, 1> av2 = ndarray::allocate(2);
    av2[0] = 1.1;
    av2[1] = 1.2;
    std::shared_ptr<Comparable> a2(new ExampleA(3, 2.5, av2));
    std::shared_ptr<Comparable> c2(new ExampleC(2, a2, a2));

    std::vector<ndarray::Vector<std::shared_ptr<Comparable>, 1>> r2 =
            roundtripAndCompare(ndarray::makeVector(c2), ndarray::makeVector<ndarray::Size>(1, 1));
    for (std::size_t i = 0; i < r2.size(); ++i) {
        std::shared_ptr<ExampleC> c3 = std::dynamic_pointer_cast<ExampleC>(r2[i][0]);
        BOOST_REQUIRE(c3);
        BOOST_CHECK_EQUAL(c3->var2, c3->var3);
    }

    std::vector<ndarray::Vector<std::shared_ptr<Comparable>, 2>> r3 =
            roundtripAndCompare(ndarray::makeVector(a2, c2), ndarray::makeVector<ndarray::Size>(1, 1));
    for (std::size_t i = 0; i < r3.size(); ++i) {
        std::shared_ptr<ExampleC> c3 = std::dynamic_pointer_cast<ExampleC>(r3[i][1]);
        BOOST_CHECK_EQUAL(c3->var3, c3->var3);
        BOOST_CHECK_EQUAL(c3->var3, r3[i][0]);
    }
}

namespace {

std::vector<double> makeRandomVector(int size) {
    std::vector<double> v(size);
    Eigen::Map<Eigen::VectorXd>(&v.front(), size).setRandom();
    return v;
}

ndarray::Array<double, 2, 2> makeRandomArray(int width, int height) {
    ndarray::Array<double, 2, 2> array(ndarray::allocate(height, width));
    array.asEigen().setRandom();
    return array;
}

template <typename T>
std::shared_ptr<T> roundtrip(T const *input) {
    using namespace lsst::afw::table::io;
    using namespace lsst::afw::fits;
    OutputArchive outArchive;
    int id = outArchive.put(input);
    MemFileManager manager;
    Fits outFits(manager, "w", Fits::AUTO_CHECK);
    outArchive.writeFits(outFits);
    outFits.closeFile();
    Fits inFits(manager, "r", Fits::AUTO_CHECK);
    inFits.setHdu(DEFAULT_HDU);
    InputArchive inArchive = InputArchive::readFits(inFits);
    inFits.closeFile();
    return std::dynamic_pointer_cast<T>(inArchive.get(id));
}

template <typename T>
void compareFunctions(lsst::afw::math::Function<T> const &a, lsst::afw::math::Function<T> const &b) {
    BOOST_CHECK(typeid(a) == typeid(b));
    BOOST_REQUIRE_EQUAL(a.getNParameters(), b.getNParameters());
    for (unsigned int i = 0; i < a.getNParameters(); ++i) {
        BOOST_CHECK_EQUAL(a.getParameter(i), b.getParameter(i));
    }
}

}  // anonymous

BOOST_AUTO_TEST_CASE(GaussianFunction2) {
    namespace afwMath = lsst::afw::math;
    std::shared_ptr<afwMath::PolynomialFunction2<double>> p1(
            new afwMath::PolynomialFunction2<double>(makeRandomVector(15)));
    std::shared_ptr<afwMath::PolynomialFunction2<double>> p2 = roundtrip(p1.get());
    compareFunctions(*p1, *p2);
}

BOOST_AUTO_TEST_CASE(PolynomialFunction2) {
    namespace afwMath = lsst::afw::math;
    std::shared_ptr<afwMath::PolynomialFunction2<double>> p1(
            new afwMath::PolynomialFunction2<double>(makeRandomVector(15)));
    std::shared_ptr<afwMath::PolynomialFunction2<double>> p2 = roundtrip(p1.get());
    compareFunctions(*p1, *p2);
}

BOOST_AUTO_TEST_CASE(Chebyshev1Function2) {
    namespace afwMath = lsst::afw::math;
    std::shared_ptr<afwMath::Chebyshev1Function2<double>> p1(
            new afwMath::Chebyshev1Function2<double>(makeRandomVector(15)));
    std::shared_ptr<afwMath::Chebyshev1Function2<double>> p2 = roundtrip(p1.get());
    compareFunctions(*p1, *p2);
    BOOST_CHECK(p1->getXYRange() == p2->getXYRange());
}

BOOST_AUTO_TEST_CASE(FixedKernel) {
    namespace afwMath = lsst::afw::math;
    namespace afwImage = lsst::afw::image;
    afwImage::Image<double> image1(makeRandomArray(5, 7));
    std::shared_ptr<afwMath::FixedKernel> p1(new afwMath::FixedKernel(image1));
    std::shared_ptr<afwMath::FixedKernel> p2 = roundtrip(p1.get());
    BOOST_CHECK_EQUAL(p1->getWidth(), p2->getWidth());
    BOOST_CHECK_EQUAL(p1->getHeight(), p2->getHeight());
    BOOST_CHECK_EQUAL(p1->getCtrX(), p2->getCtrX());
    BOOST_CHECK_EQUAL(p1->getCtrY(), p2->getCtrY());
    BOOST_CHECK_EQUAL(p1->getNSpatialParameters(), p2->getNSpatialParameters());
    BOOST_CHECK_EQUAL(p1->getNKernelParameters(), p2->getNKernelParameters());
    afwImage::Image<double> image2(p2->getDimensions());
    double s1 = p1->computeImage(image1, false);
    double s2 = p2->computeImage(image2, false);
    BOOST_CHECK_EQUAL(s1, s2);
    BOOST_CHECK(ndarray::all(ndarray::equal(image1.getArray(), image2.getArray())));
}

BOOST_AUTO_TEST_CASE(AnalyticKernel1) {
    namespace afwMath = lsst::afw::math;
    namespace afwImage = lsst::afw::image;
    std::shared_ptr<afwMath::AnalyticKernel> p1(
            new afwMath::AnalyticKernel(5, 7, afwMath::DoubleGaussianFunction2<double>(1.0, 2.0, 0.1)));
    std::shared_ptr<afwMath::AnalyticKernel> p2 = roundtrip(p1.get());
    BOOST_CHECK_EQUAL(p1->getWidth(), p2->getWidth());
    BOOST_CHECK_EQUAL(p1->getHeight(), p2->getHeight());
    BOOST_CHECK_EQUAL(p1->getCtrX(), p2->getCtrX());
    BOOST_CHECK_EQUAL(p1->getCtrY(), p2->getCtrY());
    BOOST_CHECK_EQUAL(p1->getNSpatialParameters(), p2->getNSpatialParameters());
    BOOST_CHECK_EQUAL(p1->getNKernelParameters(), p2->getNKernelParameters());
    compareFunctions(*p1->getKernelFunction(), *p2->getKernelFunction());
    afwImage::Image<double> image1(p1->getDimensions());
    afwImage::Image<double> image2(p2->getDimensions());
    double s1 = p1->computeImage(image1, false);
    double s2 = p2->computeImage(image2, false);
    BOOST_CHECK_EQUAL(s1, s2);
    BOOST_CHECK(ndarray::all(ndarray::equal(image1.getArray(), image2.getArray())));
}

BOOST_AUTO_TEST_CASE(AnalyticKernel2) {
    namespace afwMath = lsst::afw::math;
    namespace afwImage = lsst::afw::image;
    std::vector<std::shared_ptr<afwMath::Kernel::SpatialFunction>> spatialFunctions(3);
    spatialFunctions[0].reset(new afwMath::PolynomialFunction2<double>(makeRandomVector(10)));
    spatialFunctions[1].reset(new afwMath::PolynomialFunction2<double>(makeRandomVector(6)));
    spatialFunctions[2].reset(new afwMath::PolynomialFunction2<double>(makeRandomVector(21)));
    std::shared_ptr<afwMath::AnalyticKernel> p1(new afwMath::AnalyticKernel(
            5, 7, afwMath::GaussianFunction2<double>(1.0, 1.0), spatialFunctions));
    std::shared_ptr<afwMath::AnalyticKernel> p2 = roundtrip(p1.get());
    BOOST_CHECK_EQUAL(p1->getWidth(), p2->getWidth());
    BOOST_CHECK_EQUAL(p1->getHeight(), p2->getHeight());
    BOOST_CHECK_EQUAL(p1->getCtrX(), p2->getCtrX());
    BOOST_CHECK_EQUAL(p1->getCtrY(), p2->getCtrY());
    BOOST_CHECK_EQUAL(p1->getNSpatialParameters(), p2->getNSpatialParameters());
    BOOST_CHECK_EQUAL(p1->getNKernelParameters(), p2->getNKernelParameters());
    compareFunctions(*p1->getKernelFunction(), *p2->getKernelFunction());
    BOOST_CHECK(p1->getSpatialParameters() == p2->getSpatialParameters());
    afwImage::Image<double> image1(p1->getDimensions());
    afwImage::Image<double> image2(p2->getDimensions());
    Eigen::VectorXd x = Eigen::VectorXd::Random(10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(10);
    for (int i = 0; i < 10; ++i) {
        double s1 = p1->computeImage(image1, false, x[i], y[i]);
        double s2 = p2->computeImage(image2, false, x[i], y[i]);
        BOOST_CHECK_EQUAL(s1, s2);
        BOOST_CHECK(ndarray::all(ndarray::equal(image1.getArray(), image2.getArray())));
    }
}

BOOST_AUTO_TEST_CASE(LinearCombinationKernel1) {
    namespace afwMath = lsst::afw::math;
    namespace afwImage = lsst::afw::image;
    namespace afwGeom = lsst::afw::geom;
    int const nComponents = 4;
    int const width = 5;
    int const height = 7;
    std::vector<std::shared_ptr<afwMath::Kernel>> kernelList(nComponents);
    std::vector<std::shared_ptr<afwMath::Kernel::SpatialFunction>> spatialFunctionList(nComponents);
    for (int i = 0; i < nComponents; ++i) {
        kernelList[i].reset(new afwMath::DeltaFunctionKernel(width, height, afwGeom::Point2I(i, i)));
        spatialFunctionList[i].reset(new afwMath::PolynomialFunction2<double>(makeRandomVector(10)));
    }
    std::shared_ptr<afwMath::LinearCombinationKernel> p1(
            new afwMath::LinearCombinationKernel(kernelList, spatialFunctionList));
    std::shared_ptr<afwMath::LinearCombinationKernel> p2 = roundtrip(p1.get());
    BOOST_CHECK_EQUAL(p1->getWidth(), p2->getWidth());
    BOOST_CHECK_EQUAL(p1->getHeight(), p2->getHeight());
    BOOST_CHECK_EQUAL(p1->getCtrX(), p2->getCtrX());
    BOOST_CHECK_EQUAL(p1->getCtrY(), p2->getCtrY());
    BOOST_CHECK_EQUAL(p1->getNSpatialParameters(), p2->getNSpatialParameters());
    BOOST_CHECK_EQUAL(p1->getNKernelParameters(), p2->getNKernelParameters());
    BOOST_CHECK(p1->getSpatialParameters() == p2->getSpatialParameters());
    afwImage::Image<double> image1(p1->getDimensions());
    afwImage::Image<double> image2(p2->getDimensions());
    Eigen::VectorXd x = Eigen::VectorXd::Random(10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(10);
    for (int i = 0; i < 10; ++i) {
        double s1 = p1->computeImage(image1, false, x[i], y[i]);
        double s2 = p2->computeImage(image2, false, x[i], y[i]);
        BOOST_CHECK_EQUAL(s1, s2);
        BOOST_CHECK(ndarray::all(ndarray::equal(image1.getArray(), image2.getArray())));
    }
}

BOOST_AUTO_TEST_CASE(LinearCombinationKernel2) {
    namespace afwMath = lsst::afw::math;
    namespace afwImage = lsst::afw::image;
    namespace afwGeom = lsst::afw::geom;
    int const nComponents = 4;
    int const width = 5;
    int const height = 7;
    std::vector<std::shared_ptr<afwMath::Kernel>> kernelList(nComponents);
    std::vector<double> kernelParams = makeRandomVector(nComponents);
    for (int i = 0; i < nComponents; ++i) {
        kernelList[i].reset(new afwMath::DeltaFunctionKernel(width, height, afwGeom::Point2I(i, i)));
        kernelParams[i] *= kernelParams[i];  // want positive amplitudes
    }
    std::shared_ptr<afwMath::LinearCombinationKernel> p1(
            new afwMath::LinearCombinationKernel(kernelList, kernelParams));
    std::shared_ptr<afwMath::LinearCombinationKernel> p2 = roundtrip(p1.get());
    BOOST_CHECK_EQUAL(p1->getWidth(), p2->getWidth());
    BOOST_CHECK_EQUAL(p1->getHeight(), p2->getHeight());
    BOOST_CHECK_EQUAL(p1->getCtrX(), p2->getCtrX());
    BOOST_CHECK_EQUAL(p1->getCtrY(), p2->getCtrY());
    BOOST_CHECK_EQUAL(p1->getNSpatialParameters(), p2->getNSpatialParameters());
    BOOST_CHECK_EQUAL(p1->getNKernelParameters(), p2->getNKernelParameters());
    BOOST_CHECK(p1->getSpatialParameters() == p2->getSpatialParameters());
    afwImage::Image<double> image1(p1->getDimensions());
    afwImage::Image<double> image2(p2->getDimensions());
    Eigen::VectorXd x = Eigen::VectorXd::Random(10);
    Eigen::VectorXd y = Eigen::VectorXd::Random(10);
    for (int i = 0; i < 10; ++i) {
        double s1 = p1->computeImage(image1, false, x[i], y[i]);
        double s2 = p2->computeImage(image2, false, x[i], y[i]);
        BOOST_CHECK_EQUAL(s1, s2);
        BOOST_CHECK(ndarray::all(ndarray::equal(image1.getArray(), image2.getArray())));
    }
}

BOOST_AUTO_TEST_CASE(ArchiveImporter) {
    // From pure-C++ code, we shouldn't be able to read the PSF from this file, because it's defined
    // in a Python module that hasn't been loaded.  But we should get a warning and the PSF
    // will be set to null.
    std::cerr << "--------------------------------------------------------------------------------------\n";
    std::cerr << "The following warning is expected, and is an indication the test has passed.\n";
    std::cerr << "--------------------------------------------------------------------------------------\n";

    boost::filesystem::path testFilePath(lsst::utils::getPackageDir("afw"));
    boost::filesystem::path testsDir("tests");
    boost::filesystem::path dataDir("data");
    boost::filesystem::path filename("archiveImportTest.fits");
    boost::filesystem::path testPath = testFilePath / testsDir / dataDir / filename;

    lsst::afw::image::Exposure<float> exposure(testPath.string());
    BOOST_CHECK(!exposure.getPsf());
}

BOOST_AUTO_TEST_CASE(ArchiveMetadata) {
    // Test that we only write one AR_NAME header key for each class saved in an HDU, not each instance
    // (see DM-7221).
    using namespace lsst::afw::table::io;
    using namespace lsst::afw::fits;
    using namespace lsst::afw::detection;
    using namespace lsst::afw::geom;
    OutputArchive outArchive;
    outArchive.put(
            std::make_shared<Footprint>(std::make_shared<SpanSet>(Box2I(Point2I(2, 3), Point2I(5, 4)))));
    outArchive.put(
            std::make_shared<Footprint>(std::make_shared<SpanSet>(Box2I(Point2I(1, 2), Point2I(7, 6)))));
    outArchive.put(std::make_shared<HeavyFootprint<float>>(
            Footprint(std::make_shared<SpanSet>(Box2I(Point2I(1, 2), Point2I(7, 6))))));
    MemFileManager manager;
    Fits outFits(manager, "w", Fits::AUTO_CHECK);
    outArchive.writeFits(outFits);
    outFits.closeFile();
    Fits inFits(manager, "r", Fits::AUTO_CHECK);
    inFits.setHdu(2);
    lsst::daf::base::PropertyList metadata;
    inFits.readMetadata(metadata);
    inFits.closeFile();
    // Get all values corresponding to the AR_NAME key as a list; there
    // should be exactly two.
    auto names = metadata.getArray<std::string>("AR_NAME");
    std::sort(names.begin(), names.end());
    BOOST_CHECK_EQUAL(names.size(), 2u);
    BOOST_CHECK_EQUAL(names[0], "Footprint");
    BOOST_CHECK_EQUAL(names[1], "HeavyFootprintF");
}

BOOST_AUTO_TEST_CASE(Singleton) {
    using namespace lsst::afw::table::io;

    auto d1 = ExampleD::get();
    auto d2 = roundtrip(d1.get());
    BOOST_CHECK_EQUAL(d1, d2);
}
