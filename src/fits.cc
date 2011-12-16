// -*- lsst-c++ -*-

#include <cstdio>
#include <complex>
#include <sstream>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/cstdint.hpp"
#include "boost/format.hpp"
#include "boost/scoped_array.hpp"

#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace fits {

namespace {

char getFormatCode(bool*) { return 'X'; }
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

template <typename T>
std::string makeColumnFormat(int size = 1) {
    return (boost::format("%d%c") % size % getFormatCode((T*)0)).str();
}

template <typename T>
int addColumnImpl(Fits & fits, char const * ttype, int size, char const * comment, T *) {
    int nCols = 0;
    fits_get_num_cols(
        reinterpret_cast<fitsfile*>(fits.fptr),
        &nCols,
        &fits.status
    );
    std::string tform = makeColumnFormat<T>(size);
    fits_insert_col(
        reinterpret_cast<fitsfile*>(fits.fptr),
        nCols + 1,
        const_cast<char*>(ttype),
        const_cast<char*>(tform.c_str()),
        &fits.status
    );
    if (comment)
        fits.updateColumnKey("TTYPE", nCols, ttype, comment);
    return nCols;
}

int addColumnImpl(Fits & fits, char const * ttype, int size, char const * comment, boost::uint64_t *) {
    static char const * UINT64_ZERO = "9223372036854775808"; // used to fake uint64 fields in FITS.
    int nCols = 0;
    fits_get_num_cols(
        reinterpret_cast<fitsfile*>(fits.fptr),
        &nCols,
        &fits.status
    );
    std::string tform = makeColumnFormat<boost::int64_t>(size);
    fits_insert_col(
        reinterpret_cast<fitsfile*>(fits.fptr),
        nCols + 1,
        const_cast<char*>(ttype),
        const_cast<char*>(tform.c_str()),
        &fits.status
    );
    fits.updateColumnKey("TSCAL", nCols, 1);
    fits.updateColumnKey("TZERO", nCols, UINT64_ZERO);
    if (comment) 
        fits.updateColumnKey("TTYPE", nCols, ttype, comment);
    return nCols;
}

template <typename T> struct FitsType;

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
template <> struct FitsType< std::complex<float> > { static int const CONSTANT = TCOMPLEX; };
template <> struct FitsType< std::complex<double> > { static int const CONSTANT = TDBLCOMPLEX; };

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
    char keyBuf[9];
    std::sprintf(keyBuf, "%s%d", prefix, n + 1);
    return updateKey(keyBuf, value, comment);
}

template <typename T>
void Fits::writeColumnKey(char const * prefix, int n, T value, char const * comment) {
    char keyBuf[9];
    std::sprintf(keyBuf, "%s%d", prefix, n + 1);
    return writeKey(keyBuf, value, comment);
}

template <typename T>
int Fits::addColumn(char const * ttype, int size, char const * comment) {
    return addColumnImpl(*this, ttype, size, comment, (T*)0);
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
    template void Fits::writeKey(char const * key, T value, char const * comment);
    
#define INSTANTIATE_EDIT_COLUMN_KEY(r, data, T)                         \
    template void Fits::updateColumnKey(char const * prefix, int n, T value, char const * comment); \
    template void Fits::writeColumnKey(char const * prefix, int n, T value, char const * comment);

#define INSTANTIATE_ADD_COLUMN(r, data, T)                              \
    template int Fits::addColumn<T>(char const * ttype, int size, char const * comment);

#define KEY_TYPES                                                       \
    (unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)(LONGLONG) \
    (float)(double)(std::complex<float>)(std::complex<double>)

#define COLUMN_TYPES                            \
    (boost::int8_t)(boost::uint8_t)(boost::int16_t)(boost::uint16_t)(boost::int32_t)(boost::uint32_t) \
    (boost::int64_t)(boost::uint64_t)(float)(double)(std::complex<float>)(std::complex<double>)(bool)

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_EDIT_KEY, _, KEY_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_EDIT_COLUMN_KEY, _, KEY_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_ADD_COLUMN, _, COLUMN_TYPES)

INSTANTIATE_EDIT_COLUMN_KEY(_, _, char const *)

}}} // namespace lsst::afw::fits
