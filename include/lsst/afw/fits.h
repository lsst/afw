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

namespace lsst { namespace afw { namespace fits {

/**
 *  @brief Base class for polymorphic functors used to iterator over FITS key headers.
 *
 *  Subclass this, and then pass an instance to Fits::forEachKey to iterate over all the
 *  keys in a header.
 */
class HeaderIterationFunctor {
public:

    virtual void operator()(char const * key, char const * value, char const * comment) = 0;

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

//@{
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
//@}

//@{
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
//@}

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

    /// @brief Set a FITS header key, editing if it already exists and appending it if not.
    void updateKey(char const * key, char const * value, char const * comment=0);

    /// @brief Add a FITS header key to the bottom of the header.
    void writeKey(char const * key, char const * value, char const * comment=0);

    /// @brief Set a FITS header key, editing if it already exists and appending it if not.
    template <typename T>
    void updateKey(char const * key, T value, char const * comment=0);

    /// @brief Add a FITS header key to the bottom of the header.
    template <typename T>
    void writeKey(char const * key, T value, char const * comment=0);

    /// @brief Update a key of the form XXXXnnn, where XXXX is the prefix and nnn is a column number.
    template <typename T>
    void updateColumnKey(char const * prefix, int n, T value, char const * comment=0);

    /// @brief Write a key of the form XXXXnnn, where XXXX is the prefix and nnn is a column number.
    template <typename T>
    void writeColumnKey(char const * prefix, int n, T value, char const * comment=0);

    /// @brief Read a FITS header key into the given reference.
    template <typename T>
    void readKey(char const * key, T & value);

    /// @brief Read a FITS header key into the given reference.
    void readKey(char const * key, std::string & value);

    /// @brief Call a polymorphic functor for every key in the header.
    void forEachKey(HeaderIterationFunctor & functor);

    /**
     *  @brief Add a column to a table
     *
     *  If size <= 0, the field will be a variable length array, with max set by (-size),
     *  or left unknown if size == 0.
     */
    template <typename T>
    int addColumn(char const * ttype, int size, char const * comment=0);

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
    static Fits createFile(char const * filename);

    /// @brief Open a an existing FITS file.
    static Fits openFile(char const * filename, bool writeable);

    /// @brief Create a new binary table extension.
    void createTable();

    /// @brief Close a FITS file.
    void closeFile();

    /// @brief Throw a reasonably informative exception if the status is nonzero.
    void checkStatus() const {
        if (status != 0) throw LSST_EXCEPT(FitsError, makeErrorMessage(fptr, status));
    }

    void * fptr;  // the actual cfitsio fitsfile pointer; void to avoid including fitsio.h here.
    int status;   // the cfitsio status indicator that gets passed to every cfitsio call.
}; 

}}} /// namespace lsst::afw::fits

#endif // !LSST_AFW_fits_h_INCLUDED
