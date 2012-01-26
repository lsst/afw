// -*- lsst-c++ -*-

#include <cstdio>
#include <complex>
#include <sstream>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/regex.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/cstdint.hpp"
#include "boost/format.hpp"
#include "boost/scoped_array.hpp"

#include "lsst/afw/fits.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace fits {

namespace {

char getFormatCode(bool*) { return 'X'; }
char getFormatCode(boost::uint8_t*) { return 'B'; }
char getFormatCode(boost::int16_t*) { return 'I'; }
char getFormatCode(boost::uint16_t*) { return 'U'; }
char getFormatCode(boost::int32_t*) { return 'J'; }
char getFormatCode(boost::uint32_t*) { return 'V'; }
char getFormatCode(boost::int64_t*) { return 'K'; }
char getFormatCode(float*) { return 'E'; }
char getFormatCode(double*) { return 'D'; }
char getFormatCode(std::complex<float>*) { return 'C'; }
char getFormatCode(std::complex<double>*) { return 'M'; }
char getFormatCode(lsst::afw::geom::Angle*) { return 'D'; }

template <typename T>
std::string makeColumnFormat(int size = 1) {
    if (size > 0) {
        return (boost::format("%d%c") % size % getFormatCode((T*)0)).str();
    } else if (size < 0) {
        // variable length, max size given as -size
        return (boost::format("1P%c(%d)") % getFormatCode((T*)0) % (-size)).str();
    } else {
        // variable length, max size unknown
        return (boost::format("1P%c") % getFormatCode((T*)0)).str();
    }
}

template <typename T> struct FitsType;

template <> struct FitsType<bool> { static int const CONSTANT = TBIT; };
template <> struct FitsType<unsigned char> { static int const CONSTANT = TBYTE; };
template <> struct FitsType<short> { static int const CONSTANT = TSHORT; };
template <> struct FitsType<unsigned short> { static int const CONSTANT = TUSHORT; };
template <> struct FitsType<int> { static int const CONSTANT = TINT; };
template <> struct FitsType<unsigned int> { static int const CONSTANT = TUINT; };
template <> struct FitsType<long> { static int const CONSTANT = TLONG; };
template <> struct FitsType<unsigned long> { static int const CONSTANT = TULONG; };
template <> struct FitsType<LONGLONG> { static int const CONSTANT = TLONGLONG; };
template <> struct FitsType<float> { static int const CONSTANT = TFLOAT; };
template <> struct FitsType<double> { static int const CONSTANT = TDOUBLE; };
template <> struct FitsType<lsst::afw::geom::Angle> { static int const CONSTANT = TDOUBLE; };
template <> struct FitsType< std::complex<float> > { static int const CONSTANT = TCOMPLEX; };
template <> struct FitsType< std::complex<double> > { static int const CONSTANT = TDBLCOMPLEX; };

std::string strip(std::string const & s) {
    std::size_t i1 = s.find_first_not_of(" '");
    std::size_t i2 = s.find_last_not_of(" '");
    return s.substr(i1, (i1 == std::string::npos) ? 0 : 1 + i2 - i1);
}

} // anonymous

std::string makeErrorMessage(std::string const & fileName, int status, std::string const & msg) {
    std::ostringstream os;
    os << "cfitsio error";
    if (fileName != "") {
        os << " (" << fileName << ")";
    }
    if (status != 0) {
        char fitsErrMsg[FLEN_ERRMSG];
        fits_get_errstatus(status, fitsErrMsg);
        os << ": " << fitsErrMsg << " (" << status << ")";
    }
    if (msg != "") {
        os << " : " << msg;
    }
    return os.str();
}

std::string makeErrorMessage(void * fptr, int status, std::string const & msg) {
    std::string fileName = "";
    fitsfile * fd = reinterpret_cast<fitsfile*>(fptr);
    if (fd != 0 && fd->Fptr != 0 && fd->Fptr->filename != 0) {
        fileName = fd->Fptr->filename;
    }
    return makeErrorMessage(fileName, status, msg);
}

void Fits::updateKey(char const * key, char const * value, char const * comment) {
    fits_update_key_str(
        reinterpret_cast<fitsfile*>(fptr),
        const_cast<char*>(key),
        const_cast<char*>(value),
        const_cast<char*>(comment),
        &status
    );
}

void Fits::writeKey(char const * key, char const * value, char const * comment) {
    fits_write_key_str(
        reinterpret_cast<fitsfile*>(fptr),
        const_cast<char*>(key),
        const_cast<char*>(value),
        const_cast<char*>(comment),
        &status
    );
}

template <typename T>
void Fits::updateKey(char const * key, T value, char const * comment) {
    fits_update_key(
        reinterpret_cast<fitsfile*>(fptr),
        FitsType<T>::CONSTANT,
        const_cast<char*>(key),
        &value,
        const_cast<char*>(comment),
        &status
    );
}

template <typename T>
void Fits::writeKey(char const * key, T value, char const * comment) {
    fits_write_key(
        reinterpret_cast<fitsfile*>(fptr),
        FitsType<T>::CONSTANT,
        const_cast<char*>(key),
        &value,
        const_cast<char*>(comment),
        &status
    );
}

template <typename T>
void Fits::updateColumnKey(char const * prefix, int n, T value, char const * comment) {
    char keyBuf[9] = { 0 };
    std::snprintf(keyBuf, 8, "%s%d", prefix, n + 1);
    updateKey(keyBuf, value, comment);
}

template <typename T>
void Fits::writeColumnKey(char const * prefix, int n, T value, char const * comment) {
    char keyBuf[9] = { 0 };
    std::snprintf(keyBuf, 8, "%s%d", prefix, n + 1);
    writeKey(keyBuf, value, comment);
}

template <typename T>
void Fits::readKey(char const * key, T & value) {
    fits_read_key(
        reinterpret_cast<fitsfile*>(fptr), 
        FitsType<T>::CONSTANT,
        const_cast<char*>(key),
        &value,
        0,
        &status
    );
}

void Fits::readKey(char const * key, std::string & value) {
    std::string buf(80, '\0');
    fits_read_keyword(
        reinterpret_cast<fitsfile*>(fptr), 
        const_cast<char*>(key),
        &buf[0],
        0,
        &status
    );
    value = strip(buf);
}

void Fits::forEachKey(HeaderIterationFunctor & functor) {
    char key[80];
    char value[80];
    char comment[80];
    int nKeys = 0;
    fits_get_hdrspace(reinterpret_cast<fitsfile*>(fptr), &nKeys, 0, &status);
    for (int i = 1; i <= nKeys; ++i) {
        fits_read_keyn(reinterpret_cast<fitsfile*>(fptr), i, key, value, comment, &status);
        functor(key, value, comment);
    }
}

template <typename T>
int Fits::addColumn(char const * ttype, int size, char const * comment) {
    int nCols = 0;
    fits_get_num_cols(
        reinterpret_cast<fitsfile*>(fptr),
        &nCols,
        &status
    );
    std::string tform = makeColumnFormat<T>(size);
    fits_insert_col(
        reinterpret_cast<fitsfile*>(fptr),
        nCols + 1,
        const_cast<char*>(ttype),
        const_cast<char*>(tform.c_str()),
        &status
    );
    if (comment)
        updateColumnKey("TTYPE", nCols, ttype, comment);
    return nCols;
}

std::size_t Fits::addRows(std::size_t nRows) {
    long first = 0;
    fits_get_num_rows(
        reinterpret_cast<fitsfile*>(fptr),
        &first,
        &status
    );
    fits_insert_rows(
        reinterpret_cast<fitsfile*>(fptr),
        first,
        nRows,
        &status
    );
    return first;
}

std::size_t Fits::countRows() {
    long r = 0;
    fits_get_num_rows(
        reinterpret_cast<fitsfile*>(fptr),
        &r,
        &status
    );
    return r;
}

template <typename T>
void Fits::writeTableArray(std::size_t row, int col, int nElements, T const * value) {
    fits_write_col(
        reinterpret_cast<fitsfile*>(fptr), 
        FitsType<T>::CONSTANT, 
        col + 1, row + 1, 
        1, nElements,
        const_cast<T*>(value),
        &status
    );
}

template <typename T>
void Fits::readTableArray(std::size_t row, int col, int nElements, T * value) {
    int anynul = false;
    fits_read_col(
        reinterpret_cast<fitsfile*>(fptr), 
        FitsType<T>::CONSTANT, 
        col + 1, row + 1, 
        1, nElements,
        0,
        value,
        &anynul,
        &status
    );
}

long Fits::getTableArraySize(int col) {
    int typecode = 0;
    long result = 0;
    long width = 0;
    fits_get_coltype(
        reinterpret_cast<fitsfile*>(fptr),
        col + 1,
        &typecode,
        &result,
        &width,
        &status
    );
    return result;
}

long Fits::getTableArraySize(std::size_t row, int col) {
    long result = 0;
    long offset = 0;
    fits_read_descript(
        reinterpret_cast<fitsfile*>(fptr),
        col + 1,
        row + 1,
        &result,
        &offset,
        &status
    );
    return result;
}

void Fits::createTable() {
    char * ttype = 0;
    char * tform = 0;
    fits_create_tbl(reinterpret_cast<fitsfile*>(fptr), BINARY_TBL, 0, 0, &ttype, &tform, 0, 0, &status);
}

Fits Fits::createFile(char const * filename) {
    Fits result;
    result.status = 0;
    fits_create_file(reinterpret_cast<fitsfile**>(&result.fptr), const_cast<char*>(filename), &result.status);
    return result;
}

Fits Fits::openFile(char const * filename, bool writeable) {
    Fits result;
    result.status = 0;
    fits_open_file(
        reinterpret_cast<fitsfile**>(&result.fptr),
        const_cast<char*>(filename), 
        writeable ? READWRITE : READONLY,
        &result.status
    );
    return result;
}

void Fits::closeFile() {
    fits_close_file(reinterpret_cast<fitsfile*>(fptr), &status);
}

#define INSTANTIATE_EDIT_KEY(r, data, T)                                \
    template void Fits::updateKey(char const * key, T value, char const * comment); \
    template void Fits::writeKey(char const * key, T value, char const * comment); \
    template void Fits::readKey(char const * key, T & value);
    
#define INSTANTIATE_EDIT_COLUMN_KEY(r, data, T)                         \
    template void Fits::updateColumnKey(char const * prefix, int n, T value, char const * comment); \
    template void Fits::writeColumnKey(char const * prefix, int n, T value, char const * comment);

#define INSTANTIATE_ADD_COLUMN(r, data, T)                              \
    template int Fits::addColumn<T>(char const * ttype, int size, char const * comment);

#define INSTANTIATE_EDIT_TABLE_ARRAY(r, data, T)    \
    template void Fits::writeTableArray(std::size_t row, int col, int nElements, T const * value); \
    template void Fits::readTableArray(std::size_t row, int col, int nElements, T * value);

#define KEY_TYPES                                                       \
    (unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)(LONGLONG) \
    (float)(double)(std::complex<float>)(std::complex<double>)

#define COLUMN_TYPES                            \
    (boost::uint8_t)(boost::int16_t)(boost::uint16_t)(boost::int32_t)(boost::uint32_t) \
    (boost::int64_t)(float)(double)(lsst::afw::geom::Angle)(std::complex<float>)(std::complex<double>)(bool)

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_EDIT_KEY, _, KEY_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_EDIT_COLUMN_KEY, _, KEY_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_ADD_COLUMN, _, COLUMN_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_EDIT_TABLE_ARRAY, _, COLUMN_TYPES)

INSTANTIATE_EDIT_COLUMN_KEY(_, _, char const *)

}}} // namespace lsst::afw::fits
