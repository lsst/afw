// -*- lsst-c++ -*-

#include <cstdio>
#include <complex>
#include <sstream>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/cstdint.hpp"
#include "boost/format.hpp"
#include "boost/scoped_array.hpp"

#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace fits {

namespace {

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

void extractCStrings(std::vector<std::string> const & vector, boost::scoped_array<char*> & array) {
    array.reset(new char*[vector.size()]);
    char ** p = array.get();
    for (std::vector<std::string>::const_iterator i = vector.begin(); i != vector.end(); ++i, ++p) {
        *p = const_cast<char*>(i->c_str());
    }
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

Fits & Fits::updateKey(char const * key, char const * value, char const * comment) {
    fits_update_key_str(
        reinterpret_cast<fitsfile*>(fptr),
        const_cast<char*>(key),
        const_cast<char*>(value),
        const_cast<char*>(comment),
        &status
    );
    return *this;
}

Fits & Fits::writeKey(char const * key, char const * value, char const * comment) {
    fits_write_key_str(
        reinterpret_cast<fitsfile*>(fptr),
        const_cast<char*>(key),
        const_cast<char*>(value),
        const_cast<char*>(comment),
        &status
    );
    return *this;
}

template <typename T>
Fits & Fits::updateKey(char const * key, T value, char const * comment) {
    fits_update_key(
        reinterpret_cast<fitsfile*>(fptr),
        FitsType<T>::CONSTANT,
        const_cast<char*>(key),
        &value,
        const_cast<char*>(comment),
        &status
    );
    return *this;
}

template <typename T>
Fits & Fits::writeKey(char const * key, T value, char const * comment) {
    fits_write_key(
        reinterpret_cast<fitsfile*>(fptr),
        FitsType<T>::CONSTANT,
        const_cast<char*>(key),
        &value,
        const_cast<char*>(comment),
        &status
    );
    return *this;
}

Fits & Fits::createTable(
    long nRows,
    std::vector<std::string> const & ttype,
    std::vector<std::string> const & tform,
    char const * extname
) {
    assert(ttype.size() == tform.size());
    boost::scoped_array<char*> ttypeArray;
    boost::scoped_array<char*> tformArray; 
    extractCStrings(ttype, ttypeArray);
    extractCStrings(tform, tformArray);
    fits_create_tbl(
        reinterpret_cast<fitsfile*>(fptr),
        BINARY_TBL,
        nRows,
        ttype.size(),
        ttypeArray.get(),
        tformArray.get(),
        0,
        extname,
        &status
    );
    return *this;
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

Fits & Fits::closeFile() {
    fits_close_file(reinterpret_cast<fitsfile*>(fptr), &status);
    return *this;
}

#define INSTANTIATE(T)                                                  \
    template Fits & Fits::updateKey(char const * key, T value, char const * comment); \
    template Fits & Fits::writeKey(char const * key, T value, char const * comment)

INSTANTIATE(unsigned char);
INSTANTIATE(short);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(unsigned int);
INSTANTIATE(long);
INSTANTIATE(unsigned long);
INSTANTIATE(LONGLONG);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::complex<float>);
INSTANTIATE(std::complex<double>);

}}} // namespace lsst::afw::fits
