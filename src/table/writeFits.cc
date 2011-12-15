// -*- lsst-c++ -*-

#include <cstdio>

#include "boost/cstdint.hpp"

#include "lsst/afw/fits.h"
#include "lsst/afw/table/TableBase.h"

namespace lsst { namespace afw { namespace table {

namespace {

template <typename T> struct FitsFormat;
char getFormatCode(boost::uint8_t*) { return 'B'; }
char getFormatCode(boost::int8_t*) { return 'S'; }
char getFormatCode(boost::int16_t*) { return 'I'; }
char getFormatCode(boost::uint16_t*) { return 'U'; }
char getFormatCode(boost::int32_t*) { return 'J'; }
char getFormatCode(boost::uint32_t*) { return 'V'; }
char getFormatCode(boost::int64_t*) { return 'K'; }
char getFormatCode(float*) { return 'E'; }
char getFormatCode(double*) { return 'D'; }
char getFormatCode(std::complex<float>*) { return 'C'; }
char getFormatCode(std::complex<double>*) { return 'M'; }

// FITS doesn't deal with unsigned types, but FITSIO can fake it.  But it doesn't fake uint64; we
// need to do that ourselves (see code involving UINT64_ZERO), below.
char getFormatCode(boost::uint64_t*) { return 'K'; }

template <typename T>
std::string makeColumnFormat(int n = 1) {
    return (boost::format("%d%c") % n % getFormatCode((T*)0)).str();
}

struct ColumnKeyEditor {

    template <typename T>
    afw::fits::Fits & write(
        afw::fits::Fits & fits, char const * prefix, int n, T value, char const * comment=0
    ) const {
        std::sprintf(keyBuf, "%s%d", prefix, n + 1);
        return fits.writeKey(keyBuf, value, comment);
    }

    template <typename T>
    afw::fits::Fits & update(
        afw::fits::Fits & fits, char const * prefix, int n, T value, char const * comment=0
    ) const {
        std::sprintf(keyBuf, "%s%d", prefix, n + 1);
        return fits.updateKey(keyBuf, value, comment);
    }
    
    mutable char keyBuf[9];
};

struct MakeCreateTableArgs {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        ttype->push_back(item.field.getName());
        tform->push_back(makeColumnFormat<typename Field<T>::Element>(item.field.getElementCount()));
        docs->push_back(item.field.getDoc());
    }

    void operator()(SchemaItem<Flag> const & item) const {
        flagNames->push_back(item.field.getName());
        flagDocs->push_back(item.field.getDoc());
    }

    std::vector<std::string> * ttype;
    std::vector<std::string> * tform;
    std::vector<std::string> * docs;
    std::vector<std::string> * flagNames;
    std::vector<std::string> * flagDocs;
};

struct FinishTableHeader {

    template <typename T>
    void writeClass(SchemaItem<T> const & item) const {}

    template <typename T>
    void writeClass(SchemaItem< Point<T> > const & item) const {
        keyEditor->write(*fits, "TCCLS", n, "Point");
    }

    template <typename T>
    void writeClass(SchemaItem< Shape<T> > const & item) const {
        keyEditor->write(*fits, "TCCLS", n, "Shape");
    }

    template <typename T>
    void writeClass(SchemaItem< Covariance<T> > const & item) const {
        keyEditor->write(*fits, "TCCLS", n, "Covariance");
    }

    template <typename T>
    void writeClass(SchemaItem< Covariance< Point<T> > > const & item) const {
        keyEditor->write(*fits, "TCCLS", n, "Covariance(Point)");
    }

    template <typename T>
    void writeClass(SchemaItem< Covariance< Shape<T> > > const & item) const {
        keyEditor->write(*fits, "TCCLS", n, "Covariance(Shape)");
    }

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        if (!item.field.getUnits().empty()) {
            keyEditor->write(*fits, "TUNIT", n, item.field.getUnits().c_str());
        }
        writeClass(item);
        ++n;
    }

    void operator()(SchemaItem<Flag> const & item) const {}

    mutable std::size_t n;
    ColumnKeyEditor * keyEditor;
    afw::fits::Fits * fits;
};

void initializeTable(afw::fits::Fits & fits, Schema const & schema, int nRecords, bool sanitizeNames) {
    static char const * UINT64_ZERO = "9223372036854775808"; // used to fake uint64 fields in FITS.
    std::vector<std::string> ttype;
    std::vector<std::string> tform;
    std::vector<std::string> docs;
    std::vector<std::string> flagNames;
    std::vector<std::string> flagDocs;
    ttype.push_back("id");
    tform.push_back(makeColumnFormat<RecordId>());
    docs.push_back("unique ID for record");
    if (schema.hasTree()) {
        ttype.push_back("parent");
        tform.push_back(makeColumnFormat<RecordId>());
        docs.push_back("ID of parent record");
    }
    MakeCreateTableArgs makeCreateTableArgs = { &ttype, &tform, &docs, &flagNames, &flagDocs };
    schema.forEach(makeCreateTableArgs);
    if (!flagNames.empty()) {
        ttype.push_back("flags");
        tform.push_back((boost::format("%uX") % flagNames.size()).str());
        docs.push_back("see TFLAGn for bit assignments");
    }
    if (sanitizeNames) {
        for (std::vector<std::string>::iterator i = ttype.begin(); i != ttype.end(); ++i) {
            std::replace(i->begin(), i->end(), '.', '_');
        }
    }
    fits.createTable(nRecords, ttype, tform).checkStatus();

    ColumnKeyEditor keyEditor;

    if (boost::is_same<RecordId, boost::uint64_t>::value) {
        // Fake uint64 support by messing with TZEROn
        keyEditor.write(fits, "TSCAL", 1, 1); 
        keyEditor.write(fits, "TZERO", 1, UINT64_ZERO);
        if (schema.hasTree()) {
            keyEditor.write(fits, "TSCAL", 2, 1); 
            keyEditor.write(fits, "TZERO", 2, UINT64_ZERO);
        }
        fits.checkStatus();
    }

    for (std::size_t n = 0; n < ttype.size(); ++n) {
        keyEditor.update(fits, "TTYPE", n, ttype[n].c_str(), docs[n].c_str());
    }
    fits.checkStatus();

    for (std::size_t n = 0; n < flagNames.size(); ++n) {
        keyEditor.write(fits, "TFLAG", n, flagNames[n].c_str(), flagDocs[n].c_str());
    }
    fits.checkStatus();

    FinishTableHeader finishTableHeader = { 1 + schema.hasTree(), &keyEditor, &fits };
    schema.forEach(finishTableHeader);

}

} // anonymous

}}} // namespace lsst::afw::table
