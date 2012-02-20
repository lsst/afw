// -*- lsst-c++ -*-
#ifndef LSST_AFW_fits_h_INCLUDED
#define LSST_AFW_fits_h_INCLUDED

/**
 *  @file lsst/afw/fits.h
 *
 *  Utilities for working with FITS files.  These are mostly thin wrappers around
 *  cfitsio calls, and their main purpose is to transform functions signatures from
 *  void pointers and cfitsio's preprocessor type enums to a more type-safe and
 *  convenient interface using overloads and templates.
 *
 *  This was written as part of implementing the afw/table library.  Someday
 *  the afw/image FITS I/O should be modified to use some of these with the goal
 *  of eliminating a lot of code between the two.
 */

#include <string>

#include <boost/format.hpp>

#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"

namespace lsst { namespace afw { namespace fits {

/**
 *  @brief Base class for polymorphic functors used to iterator over FITS key headers.
 *
 *  Subclass this, and then pass an instance to Fits::forEachKey to iterate over all the
 *  keys in a header.
 */
class HeaderIterationFunctor {
public:

    virtual void operator()(
        std::string const & key,
        std::string const & value,
        std::string const & comment
    ) = 0;

    virtual ~HeaderIterationFunctor() {}

};

/**
 * @brief An exception thrown when problems are found when reading or writing FITS files.
 */
LSST_EXCEPTION_TYPE(FitsError, lsst::pex::exceptions::Exception, lsst::afw::fits::FitsError)

/**
 * @brief An exception thrown when a FITS file has the wrong type.
 */
LSST_EXCEPTION_TYPE(FitsTypeError, lsst::afw::fits::FitsError, lsst::afw::fits::FitsTypeError)

/**
 *  @brief Return an error message reflecting FITS I/O errors.
 *
 *  @param[in] fileName   FITS filename to be included in the error message.
 *  @param[in] status     The last status value returned by the cfitsio library; if nonzero,
 *                        the error message will include a description from cfitsio.
 *  @param[in] msg        An additional custom message to include.
 */
std::string makeErrorMessage(std::string const & fileName="", int status=0, std::string const & msg="");
inline std::string makeErrorMessage(std::string const & fileName, int status, boost::format const & msg) {
    return makeErrorMessage(fileName, status, msg.str());
}

/**
 *  @brief Return an error message reflecting FITS I/O errors.
 *
 *  @param[in] fptr       A cfitsio fitsfile pointer to be inspected for a filename.
 *                        Passed as void* to avoid including fitsio.h in the header file.
 *  @param[in] status     The last status value returned by the cfitsio library; if nonzero,
 *                        the error message will include a description from cfitsio.
 *  @param[in] msg        An additional custom message to include.
 */
std::string makeErrorMessage(void * fptr, int status=0, std::string const & msg="");
inline std::string makeErrorMessage(void * fptr, int status, boost::format const & msg) {
    return makeErrorMessage(fptr, status, msg.str());
}

/**
 *  A FITS-related replacement for LSST_EXCEPT that takes an additional Fits object
 *  and uses makeErrorMessage(fitsObj.fptr, fitsObj.status, ...) to construct the message.
 */
#define LSST_FITS_EXCEPT(type, fitsObj, ...) \
    type(LSST_EXCEPT_HERE, lsst::afw::fits::makeErrorMessage((fitsObj).fptr, (fitsObj).status, __VA_ARGS__))

/**
 *  Throw a FitsError exception if the status of the given Fits object is nonzero.
 */
#define LSST_FITS_CHECK_STATUS(fitsObj, ...)                            \
    if ((fitsObj).status != 0) LSST_FITS_EXCEPT(lsst::afw::fits::FitsError, fitsObj, __VA_ARGS__)

/**
 *  @brief A simple struct that combines the two arguments that must be passed to most cfitsio routines
 *         and contains thin and/or templated wrappers around common cfitsio routines.
 *
 *  This is NOT intended to be an object-oriented C++ wrapper around cfitsio; it's simply a thin layer that
 *  saves a lot of repetition, const/reinterpret casts, and replaces void pointer args and type codes
 *  with templates and overloads.
 *
 *  @note All functions that take a row or column number befow are 0-indexed; the internal cfitsio
 *  calls are all 1-indexed.
 */
struct Fits {

    //@{
    /// @brief Set a FITS header key, editing if it already exists and appending it if not.
    template <typename T>
    void updateKey(std::string const & key, T const & value, std::string const & comment);
    void updateKey(std::string const & key, char const * value, std::string const & comment) {
        updateKey(key, std::string(value), comment);
    }
    template <typename T>
    void updateKey(std::string const & key, T const & value);
    void updateKey(std::string const & key, char const * value) {
        updateKey(key, std::string(value));
    }
    //@}

    //@{
    /**
     *  @brief Add a FITS header key to the bottom of the header.
     *
     *  If the key is HISTORY or COMMENT and the value is a std::string or C-string, 
     *  a special HISTORY or COMMENT key will be appended (and the comment argument 
     *  will be ignored if present).
     */
    template <typename T>
    void writeKey(std::string const & key, T const & value, std::string const & comment);
    void writeKey(std::string const & key, char const * value, std::string const & comment) {
        updateKey(key, std::string(value), comment);
    }
    template <typename T>
    void writeKey(std::string const & key, T const & value);
    void writeKey(std::string const & key, char const * value) {
        updateKey(key, std::string(value));
    }
    //@}

    //@{
    /// @brief Update a key of the form XXXXXnnn, where XXXXX is the prefix and nnn is a column number.
    template <typename T>
    void updateColumnKey(std::string const & prefix, int n, T const & value, std::string const & comment);
    void updateColumnKey(std::string const & prefix, int n, char const * value, std::string const & comment) {
        updateColumnKey(prefix, n, std::string(value), comment);
    }
    template <typename T>
    void updateColumnKey(std::string const & prefix, int n, T const & value);
    void updateColumnKey(std::string const & prefix, int n, char const * value) {
        updateColumnKey(prefix, n, std::string(value));
    }
    //@}

    //@{
    /// @brief Write a key of the form XXXXXnnn, where XXXXX is the prefix and nnn is a column number.
    template <typename T>
    void writeColumnKey(std::string const & prefix, int n, T const & value, std::string const & comment);
    void writeColumnKey(std::string const & prefix, int n, char const * value, std::string const & comment) {
        writeColumnKey(prefix, n, std::string(value), comment);
    }
    template <typename T>
    void writeColumnKey(std::string const & prefix, int n, T const & value);
    void writeColumnKey(std::string const & prefix, int n, char const * value) {
        writeColumnKey(prefix, n, std::string(value));
    }
    //@}

    /**
     *  @brief Read a FITS header into a PropertySet or PropertyList.
     *
     *  @param[in]     metadata  A PropertySet or PropertyList whose items will be appended
     *                           to the FITS header.
     *
     *  All keys will be appended to the FITS header rather than used to update existing keys.  Order of keys
     *  will be preserved if and only if the metadata object is actually a PropertyList.
     */
    void writeMetadata(daf::base::PropertySet const & metadata);

    /**
     *  @brief Read a FITS header into a PropertySet or PropertyList.
     *
     *  @param[in,out] metadata  A PropertySet or PropertyList that FITS header items will be added to.
     *  @param[in]     strip     If true, common FITS keys that usually have non-metadata intepretations
     *                           (e.g. NAXIS, BITPIX) will be ignored.
     *
     *  Order will preserved if and only if the metadata object is actually a PropertyList.
     */
    void readMetadata(daf::base::PropertySet & metadata, bool strip=false);

    /// @brief Read a FITS header key into the given reference.
    template <typename T>
    void readKey(std::string const & key, T & value);

    /**
     *  @brief Call a polymorphic functor for every key in the header.
     *
     *  Each value is passed in as a string, and the single quotes that mark an actual
     *  string value are not removed (neither are extra spaces).  However, long strings
     *  that make use of the CONTINUE keyword are concatenated to look as if they were
     *  on a single line.
     */
    void forEachKey(HeaderIterationFunctor & functor);

    /**
     *  @brief Add a column to a table
     *
     *  If size <= 0, the field will be a variable length array, with max set by (-size),
     *  or left unknown if size == 0.
     */
    template <typename T>
    int addColumn(std::string const & ttype, int size, std::string const & comment);

    /**
     *  @brief Add a column to a table
     *
     *  If size <= 0, the field will be a variable length array, with max set by (-size),
     *  or left unknown if size == 0.
     */
    template <typename T>
    int addColumn(std::string const & ttype, int size);

    /// @brief Append rows to a table, and return the index of the first new row.
    std::size_t addRows(std::size_t nRows);

    /// @brief Return the number of row in a table.
    std::size_t countRows();

    /// @brief Write an array value to a binary table.
    template <typename T>
    void writeTableArray(std::size_t row, int col, int nElements, T const * value);

    /// @brief Write an scalar value to a binary table.
    template <typename T>
    void writeTableScalar(std::size_t row, int col, T value) { writeTableArray(row, col, 1, &value); }

    /// @brief Read an array value from a binary table.
    template <typename T>
    void readTableArray(std::size_t row, int col, int nElements, T * value);

    /// @brief Read an array scalar from a binary table.
    template <typename T>
    void readTableScalar(std::size_t row, int col, T & value) { readTableArray(row, col, 1, &value); }
    
    /// @brief Return the size of an array column.
    long getTableArraySize(int col);

    /// @brief Return the size of an variable-length array field.
    long getTableArraySize(std::size_t row, int col);

    /// @brief Create a new FITS file.
    static Fits createFile(std::string const & filename);

    /// @brief Open a an existing FITS file.
    static Fits openFile(std::string const & filename, bool writeable);

    /// @brief Create a new binary table extension.
    void createTable();

    /// @brief Close a FITS file.
    void closeFile();

    Fits() : fptr(0), status(0), alwaysCheck(false) {}

    void * fptr;  // the actual cfitsio fitsfile pointer; void to avoid including fitsio.h here.
    int status;   // the cfitsio status indicator that gets passed to every cfitsio call.
    bool alwaysCheck; // if true, member functions will check status and throw exceptions on failure
}; 

}}} /// namespace lsst::afw::fits

#endif // !LSST_AFW_fits_h_INCLUDED
