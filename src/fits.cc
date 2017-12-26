// -*- lsst-c++ -*-

#include <cstdint>
#include <cstdio>
#include <complex>
#include <cmath>
#include <sstream>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/regex.hpp"
#include "boost/filesystem.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/fitsCompression.h"

namespace lsst {
namespace afw {
namespace fits {

// ----------------------------------------------------------------------------------------------------------
// ---- Miscellaneous utilities -----------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

namespace {

// Strip leading and trailing single quotes and whitespace from a string.
std::string strip(std::string const &s) {
    if (s.empty()) return s;
    std::size_t i1 = s.find_first_not_of(" '");
    std::size_t i2 = s.find_last_not_of(" '");
    return s.substr(i1, (i1 == std::string::npos) ? 0 : 1 + i2 - i1);
}

// ---- FITS binary table format codes for various C++ types. -----------------------------------------------

char getFormatCode(bool *) { return 'X'; }
char getFormatCode(std::string *) { return 'A'; }
char getFormatCode(std::int8_t *) { return 'S'; }
char getFormatCode(std::uint8_t *) { return 'B'; }
char getFormatCode(std::int16_t *) { return 'I'; }
char getFormatCode(std::uint16_t *) { return 'U'; }
char getFormatCode(std::int32_t *) { return 'J'; }
char getFormatCode(std::uint32_t *) { return 'V'; }
char getFormatCode(std::int64_t *) { return 'K'; }
char getFormatCode(float *) { return 'E'; }
char getFormatCode(double *) { return 'D'; }
char getFormatCode(std::complex<float> *) { return 'C'; }
char getFormatCode(std::complex<double> *) { return 'M'; }
char getFormatCode(geom::Angle *) { return 'D'; }

// ---- Create a TFORM value for the given type and size ----------------------------------------------------

template <typename T>
std::string makeColumnFormat(int size = 1) {
    if (size > 0) {
        return (boost::format("%d%c") % size % getFormatCode((T *)0)).str();
    } else if (size < 0) {
        // variable length, max size given as -size
        return (boost::format("1Q%c(%d)") % getFormatCode((T *)0) % (-size)).str();
    } else {
        // variable length, max size unknown
        return (boost::format("1Q%c") % getFormatCode((T *)0)).str();
    }
}

// ---- Traits class to get cfitsio type constants from templates -------------------------------------------

template <typename T>
struct FitsType;

template <>
struct FitsType<bool> {
    static int const CONSTANT = TLOGICAL;
};
template <>
struct FitsType<char> {
    static int const CONSTANT = TSTRING;
};
template <>
struct FitsType<signed char> {
    static int const CONSTANT = TSBYTE;
};
template <>
struct FitsType<unsigned char> {
    static int const CONSTANT = TBYTE;
};
template <>
struct FitsType<short> {
    static int const CONSTANT = TSHORT;
};
template <>
struct FitsType<unsigned short> {
    static int const CONSTANT = TUSHORT;
};
template <>
struct FitsType<int> {
    static int const CONSTANT = TINT;
};
template <>
struct FitsType<unsigned int> {
    static int const CONSTANT = TUINT;
};
template <>
struct FitsType<long> {
    static int const CONSTANT = TLONG;
};
template <>
struct FitsType<unsigned long> {
    static int const CONSTANT = TULONG;
};
template <>
struct FitsType<long long> {
    static int const CONSTANT = TLONGLONG;
};
template <>
struct FitsType<unsigned long long> {
    static int const CONSTANT = TLONGLONG;
};
template <>
struct FitsType<float> {
    static int const CONSTANT = TFLOAT;
};
template <>
struct FitsType<double> {
    static int const CONSTANT = TDOUBLE;
};
template <>
struct FitsType<geom::Angle> {
    static int const CONSTANT = TDOUBLE;
};
template <>
struct FitsType<std::complex<float> > {
    static int const CONSTANT = TCOMPLEX;
};
template <>
struct FitsType<std::complex<double> > {
    static int const CONSTANT = TDBLCOMPLEX;
};

// We use TBIT when writing booleans to table cells, but TLOGICAL in headers.
template <typename T>
struct FitsTableType : public FitsType<T> {};
template <>
struct FitsTableType<bool> {
    static int const CONSTANT = TBIT;
};

template <typename T>
struct FitsBitPix;

template <>
struct FitsBitPix<unsigned char> {
    static int const CONSTANT = BYTE_IMG;
};
template <>
struct FitsBitPix<short> {
    static int const CONSTANT = SHORT_IMG;
};
template <>
struct FitsBitPix<unsigned short> {
    static int const CONSTANT = USHORT_IMG;
};
template <>
struct FitsBitPix<int> {
    static int const CONSTANT = LONG_IMG;
};  // not a typo!
template <>
struct FitsBitPix<unsigned int> {
    static int const CONSTANT = ULONG_IMG;
};
template <>
struct FitsBitPix<std::int64_t> {
    static int const CONSTANT = LONGLONG_IMG;
};
template <>
struct FitsBitPix<std::uint64_t> {
    static int const CONSTANT = LONGLONG_IMG;
};
template <>
struct FitsBitPix<float> {
    static int const CONSTANT = FLOAT_IMG;
};
template <>
struct FitsBitPix<double> {
    static int const CONSTANT = DOUBLE_IMG;
};

static bool allowImageCompression = true;

int fitsTypeForBitpix(int bitpix) {
    switch (bitpix) {
      case 8: return TBYTE;
      case 16: return TSHORT;
      case 32: return TINT;
      case 64: return TLONGLONG;
      case -32: return TFLOAT;
      case -64: return TDOUBLE;
      default:
        std::ostringstream os;
        os << "Invalid bitpix value: " << bitpix;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
}

/*
 * Information about one item of metadata: is a comment? is valid?
 *
 * See isCommentIsValid for more information.
 */
struct ItemInfo {
    ItemInfo(bool isComment, bool isValid) : isComment(isComment), isValid(isValid) {}
    bool isComment;
    bool isValid;
};

/*
 * Is an item a commnt (or history) and is it usable in a FITS header?
 *
 * For an item to be valid:
 * - If name is COMMENT or HISTORY then the item must be of type std::string
 * - All other items are always valid
 */
ItemInfo isCommentIsValid(daf::base::PropertyList const &pl, std::string const &name) {
    if (!pl.exists(name)) {
        return ItemInfo(false, false);
    }
    std::type_info const &type = pl.typeOf(name);
    if ((name == "COMMENT") || (name == "HISTORY")) {
        return ItemInfo(true, type == typeid(std::string));
    }
    return ItemInfo(false, true);
}

}  // anonymous

// ----------------------------------------------------------------------------------------------------------
// ---- Implementations for stuff in fits.h -----------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

std::string makeErrorMessage(std::string const &fileName, int status, std::string const &msg) {
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
    os << "\ncfitsio error stack:\n";
    char cfitsioMsg[FLEN_ERRMSG];
    while (fits_read_errmsg(cfitsioMsg) != 0) {
        os << "  " << cfitsioMsg << "\n";
    }
    return os.str();
}

std::string makeErrorMessage(void *fptr, int status, std::string const &msg) {
    std::string fileName = "";
    fitsfile *fd = reinterpret_cast<fitsfile *>(fptr);
    if (fd != 0 && fd->Fptr != 0 && fd->Fptr->filename != 0) {
        fileName = fd->Fptr->filename;
    }
    return makeErrorMessage(fileName, status, msg);
}

void MemFileManager::reset() {
    if (_managed) std::free(_ptr);
    _ptr = 0;
    _len = 0;
    _managed = true;
}

void MemFileManager::reset(std::size_t len) {
    reset();
    _ptr = std::malloc(len);
    _len = len;
    _managed = true;
}

template <typename T>
int getBitPix() {
    return FitsBitPix<T>::CONSTANT;
}

// ----------------------------------------------------------------------------------------------------------
// ---- Implementations for Fits class ----------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

std::string Fits::getFileName() const {
    std::string fileName = "<unknown>";
    fitsfile *fd = reinterpret_cast<fitsfile *>(fptr);
    if (fd != 0 && fd->Fptr != 0 && fd->Fptr->filename != 0) {
        fileName = fd->Fptr->filename;
    }
    return fileName;
}

int Fits::getHdu() {
    int n = 1;
    fits_get_hdu_num(reinterpret_cast<fitsfile *>(fptr), &n);
    return n - 1;
}

void Fits::setHdu(int hdu, bool relative) {
    if (relative) {
        fits_movrel_hdu(reinterpret_cast<fitsfile *>(fptr), hdu, 0, &status);
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, boost::format("Incrementing HDU by %d") % hdu);
        }
    } else {
        if (hdu != DEFAULT_HDU) {
            fits_movabs_hdu(reinterpret_cast<fitsfile *>(fptr), hdu + 1, 0, &status);
        }
        if (hdu == DEFAULT_HDU && getHdu() == 0 && getImageDim() == 0) {
            // want a silent failure here
            int tmpStatus = status;
            fits_movrel_hdu(reinterpret_cast<fitsfile *>(fptr), 1, 0, &tmpStatus);
        }
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, boost::format("Moving to HDU %d") % hdu);
        }
    }
}

int Fits::countHdus() {
    int n = 0;
    fits_get_num_hdus(reinterpret_cast<fitsfile *>(fptr), &n, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Getting number of HDUs in file.");
    }
    return n;
}

// ---- Writing and updating header keys --------------------------------------------------------------------

namespace {

// Impl functions in the anonymous namespace do special handling for strings, bools, and IEEE fp values.

/**
 * @internal Convert a double to a special string for writing FITS keyword values
 *
 * Non-finite values are written as special strings.  If the value is finite,
 * an empty string is returned.
 */
std::string nonFiniteDoubleToString(double value) {
    if (std::isfinite(value)) {
        return "";
    }
    if (std::isnan(value)) {
        return "NAN";
    }
    if (value < 0) {
        return "-INFINITY";
    }
    return "+INFINITY";
}

/** @internal Convert a special string to double when reading FITS keyword values
 *
 * Returns zero if the provided string is not one of the recognised special
 * strings for doubles; otherwise, returns the mapped value.
 */
double stringToNonFiniteDouble(std::string const &value) {
    if (value == "NAN") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (value == "+INFINITY") {
        return std::numeric_limits<double>::infinity();
    }
    if (value == "-INFINITY") {
        return -std::numeric_limits<double>::infinity();
    }
    return 0;
}

template <typename T>
void updateKeyImpl(Fits &fits, char const *key, T const &value, char const *comment) {
    fits_update_key(reinterpret_cast<fitsfile *>(fits.fptr), FitsType<T>::CONSTANT, const_cast<char *>(key),
                    const_cast<T *>(&value), const_cast<char *>(comment), &fits.status);
}

void updateKeyImpl(Fits &fits, char const *key, std::string const &value, char const *comment) {
    fits_update_key_longstr(reinterpret_cast<fitsfile *>(fits.fptr), const_cast<char *>(key),
                            const_cast<char *>(value.c_str()), const_cast<char *>(comment), &fits.status);
}

void updateKeyImpl(Fits &fits, char const *key, bool const &value, char const *comment) {
    int v = value;
    fits_update_key(reinterpret_cast<fitsfile *>(fits.fptr), TLOGICAL, const_cast<char *>(key), &v,
                    const_cast<char *>(comment), &fits.status);
}

void updateKeyImpl(Fits &fits, char const *key, double const &value, char const *comment) {
    std::string strValue = nonFiniteDoubleToString(value);
    if (!strValue.empty()) {
        updateKeyImpl(fits, key, strValue, comment);
    } else {
        fits_update_key(reinterpret_cast<fitsfile *>(fits.fptr), FitsType<double>::CONSTANT,
                        const_cast<char *>(key), const_cast<double *>(&value), const_cast<char *>(comment),
                        &fits.status);
    }
}

template <typename T>
void writeKeyImpl(Fits &fits, char const *key, T const &value, char const *comment) {
    fits_write_key(reinterpret_cast<fitsfile *>(fits.fptr), FitsType<T>::CONSTANT, const_cast<char *>(key),
                   const_cast<T *>(&value), const_cast<char *>(comment), &fits.status);
}

void writeKeyImpl(Fits &fits, char const *key, std::string const &value, char const *comment) {
    if (strncmp(key, "COMMENT", 7) == 0) {
        fits_write_comment(reinterpret_cast<fitsfile *>(fits.fptr), const_cast<char *>(value.c_str()),
                           &fits.status);
    } else if (strncmp(key, "HISTORY", 7) == 0) {
        fits_write_history(reinterpret_cast<fitsfile *>(fits.fptr), const_cast<char *>(value.c_str()),
                           &fits.status);
    } else {
        fits_write_key_longstr(reinterpret_cast<fitsfile *>(fits.fptr), const_cast<char *>(key),
                               const_cast<char *>(value.c_str()), const_cast<char *>(comment), &fits.status);
    }
}

void writeKeyImpl(Fits &fits, char const *key, bool const &value, char const *comment) {
    int v = value;
    fits_write_key(reinterpret_cast<fitsfile *>(fits.fptr), TLOGICAL, const_cast<char *>(key), &v,
                   const_cast<char *>(comment), &fits.status);
}

void writeKeyImpl(Fits &fits, char const *key, double const &value, char const *comment) {
    std::string strValue = nonFiniteDoubleToString(value);
    if (!strValue.empty()) {
        writeKeyImpl(fits, key, strValue, comment);
    } else {
        fits_write_key(reinterpret_cast<fitsfile *>(fits.fptr), FitsType<double>::CONSTANT,
                       const_cast<char *>(key), const_cast<double *>(&value), const_cast<char *>(comment),
                       &fits.status);
    }
}

}  // anonymous

template <typename T>
void Fits::updateKey(std::string const &key, T const &value, std::string const &comment) {
    updateKeyImpl(*this, key.c_str(), value, comment.c_str());
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Updating key '%s': '%s'") % key % value);
    }
}

template <typename T>
void Fits::writeKey(std::string const &key, T const &value, std::string const &comment) {
    writeKeyImpl(*this, key.c_str(), value, comment.c_str());
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Writing key '%s': '%s'") % key % value);
    }
}

template <typename T>
void Fits::updateKey(std::string const &key, T const &value) {
    updateKeyImpl(*this, key.c_str(), value, 0);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Updating key '%s': '%s'") % key % value);
    }
}

template <typename T>
void Fits::writeKey(std::string const &key, T const &value) {
    writeKeyImpl(*this, key.c_str(), value, 0);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Writing key '%s': '%s'") % key % value);
    }
}

template <typename T>
void Fits::updateColumnKey(std::string const &prefix, int n, T const &value, std::string const &comment) {
    updateKey((boost::format("%s%d") % prefix % (n + 1)).str(), value, comment);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Updating key '%s%d': '%s'") % prefix % (n + 1) % value);
    }
}

template <typename T>
void Fits::writeColumnKey(std::string const &prefix, int n, T const &value, std::string const &comment) {
    writeKey((boost::format("%s%d") % prefix % (n + 1)).str(), value, comment);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Writing key '%s%d': '%s'") % prefix % (n + 1) % value);
    }
}

template <typename T>
void Fits::updateColumnKey(std::string const &prefix, int n, T const &value) {
    updateKey((boost::format("%s%d") % prefix % (n + 1)).str(), value);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Updating key '%s%d': '%s'") % prefix % (n + 1) % value);
    }
}

template <typename T>
void Fits::writeColumnKey(std::string const &prefix, int n, T const &value) {
    writeKey((boost::format("%s%d") % prefix % (n + 1)).str(), value);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Writing key '%s%d': '%s'") % prefix % (n + 1) % value);
    }
}

// ---- Reading header keys ---------------------------------------------------------------------------------

namespace {

template <typename T>
void readKeyImpl(Fits &fits, char const *key, T &value) {
    fits_read_key(reinterpret_cast<fitsfile *>(fits.fptr), FitsType<T>::CONSTANT, const_cast<char *>(key),
                  &value, 0, &fits.status);
}

void readKeyImpl(Fits &fits, char const *key, std::string &value) {
    char *buf = 0;
    fits_read_key_longstr(reinterpret_cast<fitsfile *>(fits.fptr), const_cast<char *>(key), &buf, 0,
                          &fits.status);
    if (buf) {
        value = strip(buf);
        free(buf);
    }
}

void readKeyImpl(Fits &fits, char const *key, double &value) {
    // We need to check for the possibility that the value is a special string (for NAN, +/-Inf).
    // If a quote mark (') is present then it's a string.

    char buf[FLEN_VALUE];
    fits_read_keyword(reinterpret_cast<fitsfile *>(fits.fptr), const_cast<char *>(key), buf, 0, &fits.status);
    if (fits.status != 0) {
        return;
    }
    if (std::string(buf).find('\'') != std::string::npos) {
        std::string unquoted;
        readKeyImpl(fits, key, unquoted);  // Let someone else remove quotes and whitespace
        if (fits.status != 0) {
            return;
        }
        value = stringToNonFiniteDouble(unquoted);
        if (value == 0) {
            throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    (boost::format("Unrecognised string value for keyword '%s' when parsing as double: %s") %
                     key % unquoted)
                            .str());
        }
    } else {
        fits_read_key(reinterpret_cast<fitsfile *>(fits.fptr), FitsType<double>::CONSTANT,
                      const_cast<char *>(key), &value, 0, &fits.status);
    }
}

}  // anonymous

template <typename T>
void Fits::readKey(std::string const &key, T &value) {
    readKeyImpl(*this, key.c_str(), value);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Reading key '%s'") % key);
    }
}

void Fits::forEachKey(HeaderIterationFunctor &functor) {
    char key[81];  // allow for terminating NUL
    char value[81];
    char comment[81];
    int nKeys = 0;
    fits_get_hdrspace(reinterpret_cast<fitsfile *>(fptr), &nKeys, 0, &status);
    std::string keyStr;
    std::string valueStr;
    std::string commentStr;
    int i = 1;
    while (i <= nKeys) {
        fits_read_keyn(reinterpret_cast<fitsfile *>(fptr), i, key, value, comment, &status);
        keyStr = key;
        valueStr = value;
        commentStr = comment;
        ++i;
        while (valueStr.size() > 2 && valueStr[valueStr.size() - 2] == '&' && i <= nKeys) {
            // we're using key to hold the entire record here; the actual key is safe in keyStr
            fits_read_record(reinterpret_cast<fitsfile *>(fptr), i, key, &status);
            if (strncmp(key, "CONTINUE", 8) != 0) {
                // require both trailing '&' and CONTINUE to invoke long-string handling
                break;
            }
            std::string card = key;
            valueStr.erase(valueStr.size() - 2);
            std::size_t firstQuote = card.find('\'');
            if (firstQuote == std::string::npos) {
                throw LSST_EXCEPT(
                        FitsError,
                        makeErrorMessage(
                                fptr, status,
                                boost::format("Invalid CONTINUE at header key %d: \"%s\".") % i % card));
            }
            std::size_t lastQuote = card.find('\'', firstQuote + 1);
            if (lastQuote == std::string::npos) {
                throw LSST_EXCEPT(
                        FitsError,
                        makeErrorMessage(
                                fptr, status,
                                boost::format("Invalid CONTINUE at header key %d: \"%s\".") % i % card));
            }
            valueStr += card.substr(firstQuote + 1, lastQuote - firstQuote);
            std::size_t slash = card.find('/', lastQuote + 1);
            if (slash != std::string::npos) {
                commentStr += strip(card.substr(slash + 1));
            }
            ++i;
        }
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, boost::format("Reading key '%s'") % keyStr);
        }
        functor(keyStr, valueStr, commentStr);
    }
}

// ---- Reading and writing PropertySet/PropertyList --------------------------------------------------------

namespace {

bool isKeyIgnored(std::string const &key) {
    return key == "SIMPLE" || key == "BITPIX" || key == "EXTEND" || key == "GCOUNT" || key == "PCOUNT" ||
           key == "XTENSION" || key == "TFIELDS" || key == "BSCALE" || key == "BZERO" ||
           key.compare(0, 5, "NAXIS") == 0;
}

class MetadataIterationFunctor : public HeaderIterationFunctor {
public:
    virtual void operator()(std::string const &key, std::string const &value, std::string const &comment);

    template <typename T>
    void add(std::string const &key, T value, std::string const &comment) {
        if (list) {
            list->add(key, value, comment);
        } else {
            set->add(key, value);
        }
    }

    bool strip;
    daf::base::PropertySet *set;
    daf::base::PropertyList *list;
};

void MetadataIterationFunctor::operator()(std::string const &key, std::string const &value,
                                          std::string const &comment) {
    static boost::regex const boolRegex("[tTfF]");
    static boost::regex const intRegex("[+-]?[0-9]+");
    static boost::regex const doubleRegex("[+-]?([0-9]*\\.?[0-9]+|[0-9]+\\.?[0-9]*)([eE][+-]?[0-9]+)?");
    static boost::regex const fitsStringRegex("'(.*?) *'");
    boost::smatch matchStrings;

    if (strip && isKeyIgnored(key)) {
        return;
    }

    std::istringstream converter(value);
    if (boost::regex_match(value, boolRegex)) {
        // convert the string to an bool
        add(key, bool(value == "T" || value == "t"), comment);
    } else if (boost::regex_match(value, intRegex)) {
        // convert the string to an int
        std::int64_t val;
        converter >> val;
        if (val < (1LL << 31) && val > -(1LL << 31)) {
            add(key, static_cast<int>(val), comment);
        } else {
            add(key, val, comment);
        }
    } else if (boost::regex_match(value, doubleRegex)) {
        // convert the string to a double
        double val;
        converter >> val;
        add(key, val, comment);
    } else if (boost::regex_match(value, matchStrings, fitsStringRegex)) {
        std::string const str = matchStrings[1].str();  // strip off the enclosing single quotes
        double val = stringToNonFiniteDouble(str);
        if (val != 0.0) {
            add(key, val, comment);
        } else {
            add(key, str, comment);
        }
    } else if (key == "HISTORY" || key == "COMMENT") {
        add(key, comment, "");
    } else if (value.empty()) {
        // do nothing for empty values
    } else {
        throw LSST_EXCEPT(
                afw::fits::FitsError,
                (boost::format("Could not parse header value for key '%s': '%s'") % key % value).str());
    }
}

void writeKeyFromProperty(Fits &fits, daf::base::PropertySet const &metadata, std::string const &key,
                          char const *comment = 0) {
    std::type_info const &valueType = metadata.typeOf(key);
    if (valueType == typeid(bool)) {
        if (metadata.isArray(key)) {
            std::vector<bool> tmp = metadata.getArray<bool>(key);
            // work around unfortunate specialness of std::vector<bool>
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), static_cast<bool>(tmp[i]), comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<bool>(key), comment);
        }
    } else if (valueType == typeid(std::uint8_t)) {
        if (metadata.isArray(key)) {
            std::vector<std::uint8_t> tmp = metadata.getArray<std::uint8_t>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<std::uint8_t>(key), comment);
        }
    } else if (valueType == typeid(int)) {
        if (metadata.isArray(key)) {
            std::vector<int> tmp = metadata.getArray<int>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<int>(key), comment);
        }
    } else if (valueType == typeid(long)) {
        if (metadata.isArray(key)) {
            std::vector<long> tmp = metadata.getArray<long>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<long>(key), comment);
        }
    } else if (valueType == typeid(long long)) {
        if (metadata.isArray(key)) {
            std::vector<long long> tmp = metadata.getArray<long long>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<long long>(key), comment);
        }
    } else if (valueType == typeid(std::int64_t)) {
        if (metadata.isArray(key)) {
            std::vector<std::int64_t> tmp = metadata.getArray<std::int64_t>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<std::int64_t>(key), comment);
        }
    } else if (valueType == typeid(double)) {
        if (metadata.isArray(key)) {
            std::vector<double> tmp = metadata.getArray<double>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<double>(key), comment);
        }
    } else if (valueType == typeid(std::string)) {
        if (metadata.isArray(key)) {
            std::vector<std::string> tmp = metadata.getArray<std::string>(key);
            for (std::size_t i = 0; i != tmp.size(); ++i) {
                writeKeyImpl(fits, key.c_str(), tmp[i], comment);
            }
        } else {
            writeKeyImpl(fits, key.c_str(), metadata.get<std::string>(key), comment);
        }
    } else {
        // FIXME: inherited this error handling from fitsIo.cc; need a better option.
        LOGLS_WARN("afw.writeKeyFromProperty",
                   makeErrorMessage(fits.fptr, fits.status,
                                    boost::format("In %s, unknown type '%s' for key '%s'.") %
                                            BOOST_CURRENT_FUNCTION % valueType.name() % key));
    }
    if (fits.behavior & Fits::AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(fits, boost::format("Writing key '%s'") % key);
    }
}

}  // anonymous

void Fits::readMetadata(daf::base::PropertySet &metadata, bool strip) {
    MetadataIterationFunctor f;
    f.strip = strip;
    f.set = &metadata;
    f.list = dynamic_cast<daf::base::PropertyList *>(&metadata);
    forEachKey(f);
}

void Fits::writeMetadata(daf::base::PropertySet const &metadata) {
    typedef std::vector<std::string> NameList;
    daf::base::PropertyList const *pl = dynamic_cast<daf::base::PropertyList const *>(&metadata);
    NameList paramNames;
    if (pl) {
        paramNames = pl->getOrderedNames();
    } else {
        paramNames = metadata.paramNames(false);
    }
    for (NameList::const_iterator i = paramNames.begin(); i != paramNames.end(); ++i) {
        if (!isKeyIgnored(*i)) {
            if (pl) {
                writeKeyFromProperty(*this, metadata, *i, pl->getComment(*i).c_str());
            } else {
                writeKeyFromProperty(*this, metadata, *i);
            }
        }
    }
}

// ---- Manipulating tables ---------------------------------------------------------------------------------

void Fits::createTable() {
    char *ttype = 0;
    char *tform = 0;
    fits_create_tbl(reinterpret_cast<fitsfile *>(fptr), BINARY_TBL, 0, 0, &ttype, &tform, 0, 0, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Creating binary table");
    }
}

template <typename T>
int Fits::addColumn(std::string const &ttype, int size) {
    int nCols = 0;
    fits_get_num_cols(reinterpret_cast<fitsfile *>(fptr), &nCols, &status);
    std::string tform = makeColumnFormat<T>(size);
    fits_insert_col(reinterpret_cast<fitsfile *>(fptr), nCols + 1, const_cast<char *>(ttype.c_str()),
                    const_cast<char *>(tform.c_str()), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Adding column '%s' with size %d") % ttype % size);
    }
    return nCols;
}

template <typename T>
int Fits::addColumn(std::string const &ttype, int size, std::string const &comment) {
    int nCols = addColumn<T>(ttype, size);
    updateColumnKey("TTYPE", nCols, ttype, comment);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Adding column '%s' with size %d") % ttype % size);
    }
    return nCols;
}

std::size_t Fits::addRows(std::size_t nRows) {
    long first = 0;
    fits_get_num_rows(reinterpret_cast<fitsfile *>(fptr), &first, &status);
    fits_insert_rows(reinterpret_cast<fitsfile *>(fptr), first, nRows, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Adding %d rows to binary table") % nRows);
    }
    return first;
}

std::size_t Fits::countRows() {
    long r = 0;
    fits_get_num_rows(reinterpret_cast<fitsfile *>(fptr), &r, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Checking how many rows are in table");
    }
    return r;
}

template <typename T>
void Fits::writeTableArray(std::size_t row, int col, int nElements, T const *value) {
    fits_write_col(reinterpret_cast<fitsfile *>(fptr), FitsTableType<T>::CONSTANT, col + 1, row + 1, 1,
                   nElements, const_cast<T *>(value), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Writing %d-element array at table cell (%d, %d)") %
                                              nElements % row % col);
    }
}

void Fits::writeTableScalar(std::size_t row, int col, std::string const &value) {
    // cfitsio doesn't let us specify the size of a string, it just looks for null terminator.
    // Using std::string::c_str() guarantees that we have one.  But we can't store arbitrary
    // data in a string field because cfitsio will also chop off anything after the first null
    // terminator.
    char const *tmp = value.c_str();
    fits_write_col(reinterpret_cast<fitsfile *>(fptr), TSTRING, col + 1, row + 1, 1, 1,
                   const_cast<char const **>(&tmp), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Writing value at table cell (%d, %d)") % row % col);
    }
}

template <typename T>
void Fits::readTableArray(std::size_t row, int col, int nElements, T *value) {
    int anynul = false;
    fits_read_col(reinterpret_cast<fitsfile *>(fptr), FitsTableType<T>::CONSTANT, col + 1, row + 1, 1,
                  nElements, 0, value, &anynul, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Reading value at table cell (%d, %d)") % row % col);
    }
}

void Fits::readTableScalar(std::size_t row, int col, std::string &value, bool isVariableLength) {
    int anynul = false;
    long size = isVariableLength ? getTableArraySize(row, col) : getTableArraySize(col);
    // We can't directly write into a std::string until C++17.
    std::vector<char> buf(size + 1, 0);
    // cfitsio wants a char** because they imagine we might want an array of strings,
    // but we only want one element.
    char *tmp = &buf.front();
    fits_read_col(reinterpret_cast<fitsfile *>(fptr), TSTRING, col + 1, row + 1, 1, 1, 0, &tmp, &anynul,
                  &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Reading value at table cell (%d, %d)") % row % col);
    }
    value = std::string(tmp);
}

long Fits::getTableArraySize(int col) {
    int typecode = 0;
    long result = 0;
    long width = 0;
    fits_get_coltype(reinterpret_cast<fitsfile *>(fptr), col + 1, &typecode, &result, &width, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Looking up array size for column %d") % col);
    }
    return result;
}

long Fits::getTableArraySize(std::size_t row, int col) {
    long result = 0;
    long offset = 0;
    fits_read_descript(reinterpret_cast<fitsfile *>(fptr), col + 1, row + 1, &result, &offset, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Looking up array size for cell (%d, %d)") % row % col);
    }
    return result;
}

// ---- Manipulating images ---------------------------------------------------------------------------------

void Fits::createEmpty() {
    long naxes = 0;
    fits_create_img(reinterpret_cast<fitsfile *>(fptr), 8, 0, &naxes, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Creating empty image HDU");
    }
}

void Fits::createImageImpl(int bitpix, int naxis, long const *naxes) {
    fits_create_img(reinterpret_cast<fitsfile *>(fptr), bitpix, naxis,
                    const_cast<long*>(naxes), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Creating new image HDU");
    }
}

template <typename T>
void Fits::writeImageImpl(T const *data, int nElements) {
    fits_write_img(reinterpret_cast<fitsfile *>(fptr), FitsType<T>::CONSTANT, 1, nElements,
                   const_cast<T *>(data), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Writing image");
    }
}

namespace {

/// RAII for activating compression
///
/// Compression is a property of the file in cfitsio, so we need to set it,
/// do our stuff and then disable it.
struct ImageCompressionContext {
  public:
    ImageCompressionContext(Fits & fits_, ImageCompressionOptions const& useThis)
        : fits(fits_), old(fits.getImageCompression()) {
        fits.setImageCompression(useThis);
    }
    ~ImageCompressionContext() {
        int status = 0;
        std::swap(status, fits.status);
        try {
             fits.setImageCompression(old);
        } catch (...) {
            LOGLS_WARN("afw.fits",
                       makeErrorMessage(fits.fptr, fits.status, "Failed to restore compression settings"));
        }
        std::swap(status, fits.status);
    }
  private:
    Fits & fits;  // FITS file we're working on
    ImageCompressionOptions old;  // Former compression options, to be restored
};

} // anonymous namespace

template <typename T>
void Fits::writeImage(
    image::ImageBase<T> const& image,
    ImageWriteOptions const& options,
    std::shared_ptr<daf::base::PropertySet const> header,
    std::shared_ptr<image::Mask<image::MaskPixel> const> mask
) {
    auto fits = reinterpret_cast<fitsfile *>(fptr);
    ImageCompressionOptions const& compression = image.getBBox().getArea() > 0 ? options.compression :
        ImageCompressionOptions(ImageCompressionOptions::NONE);  // cfitsio can't compress empty images
    ImageCompressionContext comp(*this, compression);  // RAII
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Activating compression for write image");
    }

    ImageScale scale = options.scaling.determine(image, mask);

    // We need a place to put the image+header, and CFITSIO needs to know the dimenions.
    ndarray::Vector<long, 2> dims(image.getArray().getShape().reverse());
    createImageImpl(scale.bitpix == 0 ? detail::Bitpix<T>::value : scale.bitpix, 2, dims.elems);

    // Write the header
    std::shared_ptr<daf::base::PropertyList> wcsMetadata =
        image::detail::createTrivialWcsAsPropertySet(image::detail::wcsNameForXY0,
                                                     image.getX0(), image.getY0());
    if (header) {
        std::shared_ptr<daf::base::PropertySet> copy = header->deepCopy();
        copy->combine(wcsMetadata);
        header = copy;
    } else {
        header = wcsMetadata;
    }
    writeMetadata(*header);

    // Scale the image how we want it on disk
    ndarray::Array<T const, 2, 2> array = makeContiguousArray(image.getArray());
    auto pixels = scale.toFits(array, compression.quantizeLevel != 0, options.scaling.fuzz,
                               options.compression.tiles, options.scaling.seed);

    // We only want cfitsio to do the scale and zero for unsigned 64-bit integer types. For those,
    // "double bzero" has sufficient precision to represent the appropriate value. We'll let
    // cfitsio handle it itself.
    // In all other cases, we will convert the image to use the appropriate scale and zero
    // (because we want to fuzz the numbers in the quantisation), so we don't want cfitsio
    // rescaling.
    if (!std::is_same<T, std::uint64_t>::value) {
        fits_set_bscale(fits, 1.0, 0.0, &status);
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, "Setting bscale,bzero");
        }
    }

    // Write the pixels
    int const fitsType = scale.bitpix == 0 ? FitsType<T>::CONSTANT : fitsTypeForBitpix(scale.bitpix);
    fits_write_img(fits, fitsType, 1, pixels->getNumElements(),
                   const_cast<void*>(pixels->getData()), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Writing image");
    }

    // Now write the headers we didn't want cfitsio to know about when we were writing the pixels
    // (because we don't want it using them to modify the pixels, and we don't want it overwriting
    // these values).
    if (!std::is_same<T, std::int64_t>::value && !std::is_same<T, std::uint64_t>::value &&
        std::isfinite(scale.bzero) && std::isfinite(scale.bscale) && (scale.bscale != 0.0)) {
        if (std::numeric_limits<T>::is_integer) {
            if (scale.bzero != 0.0) {
                fits_write_key_lng(fits, "BZERO", static_cast<long>(scale.bzero),
                                   "Scaling: MEMORY = BZERO + BSCALE * DISK", &status);
            }
            if (scale.bscale != 1.0) {
                fits_write_key_lng(fits, "BSCALE", static_cast<long>(scale.bscale),
                                   "Scaling: MEMORY = BZERO + BSCALE * DISK", &status);
            }
        } else {
            fits_write_key_dbl(fits, "BZERO", scale.bzero, 12,
                               "Scaling: MEMORY = BZERO + BSCALE * DISK", &status);
            fits_write_key_dbl(fits, "BSCALE", scale.bscale, 12,
                               "Scaling: MEMORY = BZERO + BSCALE * DISK", &status);
        }
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, "Writing BSCALE,BZERO");
        }
    }

    if (scale.bitpix > 0) {
        fits_write_key_lng(fits, "BLANK", scale.blank, "Value for undefined pixels", &status);
        fits_write_key_lng(fits, "ZBLANK", scale.blank, "Value for undefined pixels", &status);
        if (!std::numeric_limits<T>::is_integer) {
            fits_write_key_lng(fits, "ZDITHER0", options.scaling.seed, "Dithering seed", &status);
            fits_write_key_str(fits, "ZQUANTIZ", "SUBTRACTIVE_DITHER_1", "Dithering algorithm", &status);
        }
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, "Writing [Z]BLANK");
        }
    }

    // cfitsio says this is deprecated, but Pan-STARRS found that it was sometimes necessary, writing:
    // "This forces a re-scan of the header to ensure everything's kosher.
    // Without this, compressed HDUs have been written out with PCOUNT=0 and TFORM1 not correctly set."
    fits_set_hdustruc(fits, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Finalizing header");
    }
}


namespace {

/// Value for BLANK pixels
template <typename T, class Enable=void>
struct NullValue {
    static T constexpr value = 0;
};

/// Floating-point values
template <typename T>
struct NullValue<T, typename std::enable_if<std::numeric_limits<T>::has_quiet_NaN>::type> {
    static T constexpr value = std::numeric_limits<T>::quiet_NaN();
};

}

template <typename T>
void Fits::readImageImpl(int nAxis, T *data, long *begin, long *end, long *increment) {
    T null = NullValue<T>::value;
    int anyNulls = 0;
    fits_read_subset(reinterpret_cast<fitsfile *>(fptr), FitsType<T>::CONSTANT, begin, end, increment,
                     reinterpret_cast<void*>(&null), data, &anyNulls, &status);
    if (behavior & AUTO_CHECK) LSST_FITS_CHECK_STATUS(*this, "Reading image");
}

int Fits::getImageDim() {
    int nAxis = 0;
    fits_get_img_dim(reinterpret_cast<fitsfile *>(fptr), &nAxis, &status);
    if (behavior & AUTO_CHECK) LSST_FITS_CHECK_STATUS(*this, "Getting NAXIS");
    return nAxis;
}

void Fits::getImageShapeImpl(int maxDim, long *nAxes) {
    fits_get_img_size(reinterpret_cast<fitsfile *>(fptr), maxDim, nAxes, &status);
    if (behavior & AUTO_CHECK) LSST_FITS_CHECK_STATUS(*this, "Getting NAXES");
}

template <typename T>
bool Fits::checkImageType() {
    int imageType = 0;
    fits_get_img_equivtype(reinterpret_cast<fitsfile *>(fptr), &imageType, &status);
    if (behavior & AUTO_CHECK) LSST_FITS_CHECK_STATUS(*this, "Getting image type");
    if (std::numeric_limits<T>::is_integer) {
        if (imageType < 0) {
            return false;  // can't represent floating-point with integer
        }
        return imageType <= FitsBitPix<T>::CONSTANT;  // can represent a small int by a larger int
    }
    if (std::is_same<T, double>::value) {
        // Everything can be represented by double
        return true;
    }
    assert((std::is_same<T, float>::value)); // Everything else has been dealt with
    switch (imageType) {
      case TLONGLONG:
      case TDOUBLE:
        // Not enough precision in 'float'
        return false;
      default:
        return true;
    }
}

ImageCompressionOptions Fits::getImageCompression()
{
    auto fits = reinterpret_cast<fitsfile *>(fptr);
    int compType = 0;  // cfitsio compression type
    fits_get_compression_type(fits, &compType, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Getting compression type");
    }

    ImageCompressionOptions::Tiles tiles = ndarray::allocate(MAX_COMPRESS_DIM);
    fits_get_tile_dim(fits, tiles.getNumElements(), tiles.getData(), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Getting tile dimensions");
    }

    float quantizeLevel;
    fits_get_quantize_level(fits, &quantizeLevel, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Getting quantizeLevel");
    }

    return ImageCompressionOptions(compressionAlgorithmFromCfitsio(compType), tiles, quantizeLevel);
}

void Fits::setImageCompression(ImageCompressionOptions const& comp)
{
    auto fits = reinterpret_cast<fitsfile *>(fptr);
    fits_unset_compression_request(fits, &status);  // wipe the slate and start over
    ImageCompressionOptions::CompressionAlgorithm const algorithm = allowImageCompression ? comp.algorithm :
        ImageCompressionOptions::NONE;
    fits_set_compression_type(fits, compressionAlgorithmToCfitsio(algorithm), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Setting compression type");
    }

    if (algorithm == ImageCompressionOptions::NONE) {
        // Nothing else worth doing
        return;
    }

    fits_set_tile_dim(fits, comp.tiles.getNumElements(), comp.tiles.getData(), &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Setting tile dimensions");
    }

    if (comp.algorithm != ImageCompressionOptions::PLIO && std::isfinite(comp.quantizeLevel)) {
        fits_set_quantize_level(fits, comp.quantizeLevel, &status);
        if (behavior & AUTO_CHECK) {
            LSST_FITS_CHECK_STATUS(*this, "Setting quantization level");
        }
    }
}

void setAllowImageCompression(bool allow) {
    allowImageCompression = allow;
}

bool getAllowImageCompression() {
    return allowImageCompression;
}

// ---- Manipulating files ----------------------------------------------------------------------------------

Fits::Fits(std::string const &filename, std::string const &mode, int behavior_)
        : fptr(0), status(0), behavior(behavior_) {
    if (mode == "r" || mode == "rb") {
        fits_open_file(reinterpret_cast<fitsfile **>(&fptr), const_cast<char *>(filename.c_str()), READONLY,
                       &status);
    } else if (mode == "w" || mode == "wb") {
        boost::filesystem::remove(filename);  // cfitsio doesn't like over-writing files
        fits_create_file(reinterpret_cast<fitsfile **>(&fptr), const_cast<char *>(filename.c_str()), &status);
    } else if (mode == "a" || mode == "ab") {
        fits_open_file(reinterpret_cast<fitsfile **>(&fptr), const_cast<char *>(filename.c_str()), READWRITE,
                       &status);
        int nHdu = 0;
        fits_get_num_hdus(reinterpret_cast<fitsfile *>(fptr), &nHdu, &status);
        fits_movabs_hdu(reinterpret_cast<fitsfile *>(fptr), nHdu, NULL, &status);
        if ((behavior & AUTO_CHECK) && (behavior & AUTO_CLOSE) && (status) && (fptr)) {
            // We're about to throw an exception, and the destructor won't get called
            // because we're in the constructor, so cleanup here first.
            int tmpStatus = 0;
            fits_close_file(reinterpret_cast<fitsfile *>(fptr), &tmpStatus);
        }
    } else {
        throw LSST_EXCEPT(
                FitsError,
                (boost::format("Invalid mode '%s' given when opening file '%s'") % mode % filename).str());
    }
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, boost::format("Opening file '%s' with mode '%s'") % filename % mode);
    }
}

Fits::Fits(MemFileManager &manager, std::string const &mode, int behavior_)
        : fptr(0), status(0), behavior(behavior_) {
    typedef void *(*Reallocator)(void *, std::size_t);
    // It's a shame this logic is essentially a duplicate of above, but the innards are different enough
    // we can't really reuse it.
    if (mode == "r" || mode == "rb") {
        fits_open_memfile(reinterpret_cast<fitsfile **>(&fptr), "unused", READONLY, &manager._ptr,
                          &manager._len, 0, 0,  // no reallocator or deltasize necessary for READONLY
                          &status);
    } else if (mode == "w" || mode == "wb") {
        Reallocator reallocator = 0;
        if (manager._managed) reallocator = &std::realloc;
        fits_create_memfile(reinterpret_cast<fitsfile **>(&fptr), &manager._ptr, &manager._len, 0,
                            reallocator,  // use default deltasize
                            &status);
    } else if (mode == "a" || mode == "ab") {
        Reallocator reallocator = 0;
        if (manager._managed) reallocator = &std::realloc;
        fits_open_memfile(reinterpret_cast<fitsfile **>(&fptr), "unused", READWRITE, &manager._ptr,
                          &manager._len, 0, reallocator, &status);
        int nHdu = 0;
        fits_get_num_hdus(reinterpret_cast<fitsfile *>(fptr), &nHdu, &status);
        fits_movabs_hdu(reinterpret_cast<fitsfile *>(fptr), nHdu, NULL, &status);
        if ((behavior & AUTO_CHECK) && (behavior & AUTO_CLOSE) && (status) && (fptr)) {
            // We're about to throw an exception, and the destructor won't get called
            // because we're in the constructor, so cleanup here first.
            int tmpStatus = 0;
            fits_close_file(reinterpret_cast<fitsfile *>(fptr), &tmpStatus);
        }
    } else {
        throw LSST_EXCEPT(FitsError,
                          (boost::format("Invalid mode '%s' given when opening memory file at '%s'") % mode %
                           manager._ptr)
                                  .str());
    }
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(
                *this, boost::format("Opening memory file at '%s' with mode '%s'") % manager._ptr % mode);
    }
}

void Fits::closeFile() {
    fits_close_file(reinterpret_cast<fitsfile *>(fptr), &status);
    fptr = nullptr;
}

std::shared_ptr<daf::base::PropertyList> combineMetadata(
        std::shared_ptr<const daf::base::PropertyList> first,
        std::shared_ptr<const daf::base::PropertyList> second) {
    auto combined = std::make_shared<daf::base::PropertyList>();
    bool const asScalar = true;
    for (auto const &name : first->getOrderedNames()) {
        auto const iscv = isCommentIsValid(*first, name);
        if (iscv.isComment) {
            if (iscv.isValid) {
                combined->add<std::string>(name, first->getArray<std::string>(name));
            }
        } else {
            combined->copy(name, first, name, asScalar);
        }
    }
    for (auto const &name : second->getOrderedNames()) {
        auto const iscv = isCommentIsValid(*second, name);
        if (iscv.isComment) {
            if (iscv.isValid) {
                combined->add<std::string>(name, second->getArray<std::string>(name));
            }
        } else {
            // `copy` will replace an item, even if has a different type, so no need to call `remove`
            combined->copy(name, second, name, asScalar);
        }
    }
    return combined;
}

std::shared_ptr<daf::base::PropertyList> readMetadata(std::string const &fileName, int hdu, bool strip) {
    fits::Fits fp(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fp.setHdu(hdu);
    return readMetadata(fp, strip);
}

std::shared_ptr<daf::base::PropertyList> readMetadata(fits::MemFileManager &manager, int hdu, bool strip) {
    fits::Fits fp(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fp.setHdu(hdu);
    return readMetadata(fp, strip);
}

std::shared_ptr<daf::base::PropertyList> readMetadata(fits::Fits &fitsfile, bool strip) {
    auto metadata = std::make_shared<lsst::daf::base::PropertyList>();
    fitsfile.readMetadata(*metadata, strip);
    // if INHERIT=T, we want to also include header entries from the primary HDU
    if (fitsfile.getHdu() != 0 && metadata->exists("INHERIT")) {
        bool inherit = false;
        if (metadata->typeOf("INHERIT") == typeid(std::string)) {
            inherit = (metadata->get<std::string>("INHERIT") == "T");
        } else {
            inherit = metadata->get<bool>("INHERIT");
        }
        if (strip) metadata->remove("INHERIT");
        if (inherit) {
            fitsfile.setHdu(0);
            // Combine the metadata from the primary HDU with the metadata from the specified HDU,
            // with non-comment values from the specified HDU superseding those in the primary HDU
            // and comments from the specified HDU appended to comments from the primary HDU
            auto primaryHduMetadata = std::make_shared<daf::base::PropertyList>();
            fitsfile.readMetadata(*primaryHduMetadata, strip);
            metadata = combineMetadata(primaryHduMetadata, metadata);
        }
    }
    return metadata;
}

namespace {

// RAII for moving the HDU
class HduMove {
  public:
    HduMove() = delete;
    HduMove(HduMove const&) = delete;

    HduMove(Fits & fits, int hdu) : _fits(fits), _enabled(true) {
        _fits.setHdu(hdu);
    }
    ~HduMove() {
        if (!_enabled) {
            return;
        }
        int status = 0;
        std::swap(status, _fits.status);
        try {
             _fits.setHdu(0);
        } catch (...) {
            LOGLS_WARN("afw.fits",
                       makeErrorMessage(_fits.fptr, _fits.status, "Failed to move back to PHU"));
        }
        std::swap(status, _fits.status);
    }
    void disable() { _enabled = false; }

  private:
    Fits & _fits;  // FITS file we're working on
    bool _enabled;  // Move back to PHU on destruction?
};

} // anonymous namespace

bool Fits::checkCompressedImagePhu() {
    auto fits = reinterpret_cast<fitsfile *>(fptr);
    if (getHdu() != 0 || countHdus() == 1) {
        return false;  // Can't possibly be the PHU leading a compressed image
    }
    // Check NAXIS = 0
    int naxis;
    fits_get_img_dim(fits, &naxis, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Checking NAXIS of PHU");
    }
    if (naxis != 0) {
        return false;
    }
    // Check first extension (and move back there when we're done if we're not compressed)
    HduMove move{*this, 1};
    bool isCompressed = fits_is_compressed_image(fits, &status);
    if (behavior & AUTO_CHECK) {
        LSST_FITS_CHECK_STATUS(*this, "Checking compression");
    }
    if (isCompressed) {
        move.disable();
    }
    return isCompressed;
}


ImageWriteOptions::ImageWriteOptions(daf::base::PropertySet const& config)
  : compression(
        fits::compressionAlgorithmFromString(config.get<std::string>("compression.algorithm")),
        std::vector<long>{config.getAsInt64("compression.columns"), config.getAsInt64("compression.rows")},
        config.getAsDouble("compression.quantizeLevel")
    ),
    scaling(
        fits::scalingAlgorithmFromString(config.get<std::string>("scaling.algorithm")),
        config.getAsInt("scaling.bitpix"),
        config.exists("scaling.maskPlanes") ? config.getArray<std::string>("scaling.maskPlanes") :
            std::vector<std::string>{},
        config.getAsInt("scaling.seed"),
        config.getAsDouble("scaling.quantizeLevel"),
        config.getAsDouble("scaling.quantizePad"),
        config.get<bool>("scaling.fuzz"),
        config.getAsDouble("scaling.bscale"),
        config.getAsDouble("scaling.bzero")
    )
{}


namespace {

template <typename T>
void validateEntry(
    daf::base::PropertySet & output,
    daf::base::PropertySet const& input,
    std::string const& name,
    T defaultValue
) {
    output.add(name, input.get<T>(name, defaultValue));
}


template <typename T>
void validateEntry(
    daf::base::PropertySet & output,
    daf::base::PropertySet const& input,
    std::string const& name,
    std::vector<T> defaultValue
) {
    output.add(name, input.exists(name) ? input.getArray<T>(name) : defaultValue);
}

} // anonymous namespace


std::shared_ptr<daf::base::PropertySet>
ImageWriteOptions::validate(daf::base::PropertySet const& config) {
    auto validated = std::make_shared<daf::base::PropertySet>();

    validateEntry(*validated, config, "compression.algorithm", std::string("NONE"));
    validateEntry(*validated, config, "compression.columns", 0);
    validateEntry(*validated, config, "compression.rows", 1);
    validateEntry(*validated, config, "compression.quantizeLevel", 0.0);

    validateEntry(*validated, config, "scaling.algorithm", std::string("NONE"));
    validateEntry(*validated, config, "scaling.bitpix", 0);
    validateEntry(*validated, config, "scaling.maskPlanes", std::vector<std::string>{"NO_DATA"});
    validateEntry(*validated, config, "scaling.seed", 1);
    validateEntry(*validated, config, "scaling.quantizeLevel", 5.0);
    validateEntry(*validated, config, "scaling.quantizePad", 10.0);
    validateEntry(*validated, config, "scaling.fuzz", true);
    validateEntry(*validated, config, "scaling.bscale", 1.0);
    validateEntry(*validated, config, "scaling.bzero", 0.0);

    // Check for additional entries that we don't support (e.g., from typos)
    for (auto const& name : config.names(false)) {
        if (!validated->exists(name)) {
            std::ostringstream os;
            os << "Invalid image write option: " << name;
            throw LSST_EXCEPT(pex::exceptions::RuntimeError, os.str());
        }
    }

    return validated;
}


#define INSTANTIATE_KEY_OPS(r, data, T)                                                            \
    template void Fits::updateKey(std::string const &, T const &, std::string const &);            \
    template void Fits::writeKey(std::string const &, T const &, std::string const &);             \
    template void Fits::updateKey(std::string const &, T const &);                                 \
    template void Fits::writeKey(std::string const &, T const &);                                  \
    template void Fits::updateColumnKey(std::string const &, int, T const &, std::string const &); \
    template void Fits::writeColumnKey(std::string const &, int, T const &, std::string const &);  \
    template void Fits::updateColumnKey(std::string const &, int, T const &);                      \
    template void Fits::writeColumnKey(std::string const &, int, T const &);                       \
    template void Fits::readKey(std::string const &, T &);

#define INSTANTIATE_IMAGE_OPS(r, data, T)                                \
    template void Fits::writeImageImpl(T const *, int);                  \
    template void Fits::writeImage(image::ImageBase<T> const&, \
                                   ImageWriteOptions const&, \
                                   std::shared_ptr<daf::base::PropertySet const>, \
                                   std::shared_ptr<image::Mask<image::MaskPixel> const>); \
    template void Fits::readImageImpl(int, T *, long *, long *, long *); \
    template bool Fits::checkImageType<T>();                             \
    template int getBitPix<T>();

#define INSTANTIATE_TABLE_OPS(r, data, T)                                \
    template int Fits::addColumn<T>(std::string const &ttype, int size); \
    template int Fits::addColumn<T>(std::string const &ttype, int size, std::string const &comment);
#define INSTANTIATE_TABLE_ARRAY_OPS(r, data, T)                                                   \
    template void Fits::writeTableArray(std::size_t row, int col, int nElements, T const *value); \
    template void Fits::readTableArray(std::size_t row, int col, int nElements, T *value);

// ----------------------------------------------------------------------------------------------------------
// ---- Explicit instantiation ------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

#define KEY_TYPES                                                                                   \
    (bool)(unsigned char)(short)(unsigned short)(int)(unsigned int)(long)(unsigned long)(LONGLONG)( \
            float)(double)(std::complex<float>)(std::complex<double>)(std::string)

#define COLUMN_TYPES                                                                             \
    (bool)(std::string)(std::uint8_t)(std::int16_t)(std::uint16_t)(std::int32_t)(std::uint32_t)( \
            std::int64_t)(float)(double)(geom::Angle)(std::complex<float>)(std::complex<double>)

#define COLUMN_ARRAY_TYPES                                                                              \
    (bool)(char)(std::uint8_t)(std::int16_t)(std::uint16_t)(std::int32_t)(std::uint32_t)(std::int64_t)( \
            float)(double)(geom::Angle)(std::complex<float>)(std::complex<double>)

#define IMAGE_TYPES \
    (unsigned char)(short)(unsigned short)(int)(unsigned int)(std::int64_t)(std::uint64_t)(float)(double)

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_KEY_OPS, _, KEY_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TABLE_OPS, _, COLUMN_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TABLE_ARRAY_OPS, _, COLUMN_ARRAY_TYPES)
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_IMAGE_OPS, _, IMAGE_TYPES)
}
}
}  // namespace lsst::afw::fits
