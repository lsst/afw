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

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "ndarray.h"

namespace lsst { namespace afw { namespace fits {

/**
 * @brief An exception thrown when problems are found when reading or writing FITS files.
 */
LSST_EXCEPTION_TYPE(FitsError, lsst::pex::exceptions::IoError, lsst::afw::fits::FitsError)

/**
 * @brief An exception thrown when a FITS file has the wrong type.
 */
LSST_EXCEPTION_TYPE(FitsTypeError, lsst::afw::fits::FitsError, lsst::afw::fits::FitsTypeError)

#ifndef SWIG // only want SWIG to see the exceptions; everything else is too low-level for Python.

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
    if ((fitsObj).status != 0) throw LSST_FITS_EXCEPT(lsst::afw::fits::FitsError, fitsObj, __VA_ARGS__)

/// Return the cfitsio integer BITPIX code for the given data type.
template <typename T> int getBitPix();

/**
 *  @brief Lifetime-management for memory that goes into FITS memory files.
 */
class MemFileManager {
public:

    /**
     *  @brief Construct a MemFileManager with no initial memory buffer.
     *
     *  The manager will still free the memory when it goes out of scope, but all allocation
     *  and reallocation will be performed by cfitsio as needed.
     */
    MemFileManager() : _ptr(0), _len(0), _managed(true) {}

    /**
     *  @brief Construct a MemFileManager with (len) bytes of initial memory.
     *
     *  The manager will free the memory when it goes out of scope, and cfitsio will be allowed
     *  to reallocate the internal memory as needed.
     */
    explicit MemFileManager(std::size_t len) : _ptr(0), _len(0), _managed(true) { reset(len); }

    /**
     *  @brief Construct a MemFileManager that references and does not manage external memory.
     *
     *  The manager will not manage the given pointer, and it will not allow cfitsio to do so
     *  either.  The user must provide enough initial memory and is responsible for freeing
     *  it manually after the FITS file has been closed.
     */
    MemFileManager(void * ptr, std::size_t len) : _ptr(ptr), _len(len), _managed(false) {}

    /**
     *  @brief Return the manager to the same state it would be if default-constructed.
     *
     *  This must not be called while a FITS file that uses this memory is open.
     */
    void reset();

    /**
     *  @brief Set the size of the internal memory buffer, freeing the current buffer if necessary.
     *
     *  This must not be called while a FITS file that uses this memory is open.
     *
     *  Memory allocated with this overload of reset can be reallocated by cfitsio
     *  and will be freed when the manager goes out of scope or is reset.
     */
    void reset(std::size_t len);

    /**
     *  @brief Set the internal memory buffer to an manually-managed external block.
     *
     *  This must not be called while a FITS file that uses this memory is open.
     *
     *  Memory passed to this overload of reset cannot be reallocated by cfitsio
     *  and will not be freed when the manager goes out of scope or is reset.
     */
    void reset(void * ptr, std::size_t len) { reset(); _ptr = ptr; _len = len; _managed = false; }

    ~MemFileManager() { reset(); }

    // No copying
    MemFileManager (const MemFileManager&) = delete;
    MemFileManager& operator=(const MemFileManager&) = delete;

    // No moving
    MemFileManager (MemFileManager&&) = delete;
    MemFileManager& operator=(MemFileManager&&) = delete;

    /// @brief Return the buffer
    void* getData() const { return _ptr; }

    /// @brief Return the buffer length
    std::size_t getLength() const { return _len; }

private:

    friend class Fits;

    void * _ptr;
    std::size_t _len;
    bool _managed;
};

/**
 *  @brief A simple struct that combines the two arguments that must be passed to most cfitsio routines
 *         and contains thin and/or templated wrappers around common cfitsio routines.
 *
 *  This is NOT intended to be an object-oriented C++ wrapper around cfitsio; it's simply a thin layer that
 *  saves a lot of repetition, const/reinterpret casts, and replaces void pointer args and type codes
 *  with templates and overloads.
 *
 *  Like a cfitsio pointer, a Fits object always considers one HDU the "active" one, and most operations
 *  will be applied to that HDU.
 *
 *  All member functions are non-const because they may modify the 'status' data member.
 *
 *  @note All functions that take a row or column number below are 0-indexed; the internal cfitsio
 *  calls are all 1-indexed.
 */
class Fits {
    template <typename T> void createImageImpl(int nAxis, long * nAxes);
    template <typename T> void writeImageImpl(T const * data, int nElements);
    template <typename T> void readImageImpl(int nAxis, T * data, long * begin, long * end, long * increment);
    void getImageShapeImpl(int maxDim, long * nAxes);
public:

    enum BehaviorFlags {
        AUTO_CLOSE = 0x01, // Close files when the Fits object goes out of scope if fptr != NULL
        AUTO_CHECK = 0x02  // Call LSST_FITS_CHECK_STATUS after every cfitsio call
    };

    /// @brief Return the file name associated with the FITS object or "<unknown>" if there is none.
    std::string getFileName() const;

    /// @brief Return the current HDU (1-indexed; 1 is the Primary HDU).
    int getHdu();

    /**
     *  @brief Set the current HDU.
     *
     *  @param[in] hdu                 The HDU to move to (1-indexed; 1 is the Primary HDU).
     *                                 The special value of 0 moves to the first extension
     *                                 if the Primary HDU is empty (has NAXIS==0) and the
     *                                 the Primary HDU is the current one.
     *  @param[in] relative            If true, move relative to the current HDU.
     */
    void setHdu(int hdu, bool relative=false);

    /// @brief Return the number of HDUs in the file.
    int countHdus();

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
        writeKey(key, std::string(value), comment);
    }
    template <typename T>
    void writeKey(std::string const & key, T const & value);
    void writeKey(std::string const & key, char const * value) {
        writeKey(key, std::string(value));
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
     *  @brief Create an empty image HDU with NAXIS=0 at the end of the file.
     *
     *  This is primarily useful to force the first "real" HDU to be an extension HDU by creating
     *  an empty Primary HDU.  The new HDU is set as the active one.
     */
    void createEmpty();

    /**
     *  @brief Create an image with pixel type provided by the given explicit PixelT template parameter
     *         and shape defined by an ndarray index.
     *
     *  @note The shape parameter is ordered fastest-dimension last (i.e. [y, x]) as is conventional
     *        with ndarray.
     *
     *  The new image will be in a new HDU at the end of the file, which may be the Primary HDU
     *  if the FITS file is empty.
     */
    template <typename PixelT, int N>
    void createImage(ndarray::Vector<ndarray::Size,N> const & shape) {
        ndarray::Vector<long,N> nAxes(shape.reverse());
        createImageImpl<PixelT>(N, nAxes.elems);
    }

    /**
     *  @brief Create a 2-d image with pixel type provided by the given explicit PixelT template parameter.
     *
     *  The new image will be in a new HDU at the end of the file, which may be the Primary HDU
     *  if the FITS file is empty.
     */
    template <typename PixelT>
    void createImage(long x, long y) {
        long naxes[2] = { x, y };
        createImageImpl<PixelT>(2, naxes);
    }

    /**
     *  @brief Write an ndarray::Array to a FITS image HDU.
     *
     *  The HDU must already exist and have the correct bitpix.
     *
     *  An extra deep-copy may be necessary if the array is not fully contiguous.
     */
    template <typename T, int N, int C>
    void writeImage(ndarray::Array<T const,N,C> const & array) {
        ndarray::Array<T const,N,N> contiguous = ndarray::dynamic_dimension_cast<2>(array);
        if (contiguous.empty()) contiguous = ndarray::copy(array);
        writeImageImpl(contiguous.getData(), contiguous.getNumElements());
    }

    /// @brief Return the number of dimensions in the current HDU.
    int getImageDim();

    /**
     *  @brief Return the shape of the current (image) HDU.
     *
     *  The order of dimensions is reversed from the FITS ordering, reflecting the usual
     *  (y,x) ndarray convention.
     *
     *  The template parameter must match the actual number of dimension in the image.
     */
    template <int N>
    ndarray::Vector<ndarray::Size,N> getImageShape() {
        ndarray::Vector<long,N> nAxes(1);
        getImageShapeImpl(N, nAxes.elems);
        ndarray::Vector<ndarray::Size,N> shape;
        for (int i = 0; i < N; ++i) shape[i] = nAxes[N-i-1];
        return shape;
    }

    /**
     *  @brief Return true if the current HDU has the given pixel type..
     *
     *  This takes into account the BUNIT and BSCALE keywords, which can allow integer
     *  images to be interpreted as floating point.
     */
    template <typename T>
    bool checkImageType();

    /**
     *  @brief Read an array from a FITS image.
     *
     *  @param[out]  array    Array to be filled.  Must already be allocated to the desired shape.
     *  @param[in]   offset   Indices of the first pixel to be read from the image.
     */
    template <typename T, int N>
    void readImage(
        ndarray::Array<T,N,N> const & array,
        ndarray::Vector<int,N> const & offset
    ) {
        ndarray::Vector<long,N> begin(offset.reverse());
        ndarray::Vector<long,N> end(begin);
        end += array.getShape().reverse();
        ndarray::Vector<long,N> increment(1);
        begin += increment;  // first FITS pixel is 1, not 0
        readImageImpl(N, array.getData(), begin.elems, end.elems, increment.elems);
    }

    /// @brief Create a new binary table extension.
    void createTable();

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

    /// @brief Write a string to a binary table.
    void writeTableScalar(std::size_t row, int col, std::string const & value);

    /// @brief Read an array value from a binary table.
    template <typename T>
    void readTableArray(std::size_t row, int col, int nElements, T * value);

    /// @brief Read an array scalar from a binary table.
    template <typename T>
    void readTableScalar(std::size_t row, int col, T & value) { readTableArray(row, col, 1, &value); }

    /// @brief Read a string from a binary table.
    void readTableScalar(std::size_t row, int col, std::string & value);

    /// @brief Return the size of an array column.
    long getTableArraySize(int col);

    /// @brief Return the size of an variable-length array field.
    long getTableArraySize(std::size_t row, int col);

    /// @brief Default constructor; set all data members to 0.
    Fits() : fptr(0), status(0), behavior(0) {}

    /// @brief Open or create a FITS file from disk.
    Fits(std::string const & filename, std::string const & mode, int behavior);

    /// @brief Open or create a FITS file from an in-memory file.
    Fits(MemFileManager & manager, std::string const & mode, int behavior);

    /// @brief Close a FITS file.
    void closeFile();

    ~Fits() { if ((fptr) && (behavior & AUTO_CLOSE)) closeFile(); }

    // No copying
    Fits (const Fits&) = delete;
    Fits& operator=(const Fits&) = delete;

    // No moving
    Fits (Fits&&) = delete;
    Fits& operator=(Fits&&) = delete;

    void * fptr;  // the actual cfitsio fitsfile pointer; void to avoid including fitsio.h here.
    int status;   // the cfitsio status indicator that gets passed to every cfitsio call.
    int behavior; // bitwise OR of BehaviorFlags
};

#endif // !SWIG

//@{
/// @brief Read FITS header
///
/// If 'strip' is true, common FITS keys that usually have non-metadata intepretations
/// (e.g. NAXIS, BITPIX) will be ignored.
///
/// Includes support for the INHERIT convention: if 'INHERIT = T' is in the header, the
/// PHU will be read as well, and nominated HDU will override any duplicated values.
PTR(daf::base::PropertyList) readMetadata(std::string const & fileName, int hdu=0, bool strip=false);
PTR(daf::base::PropertyList) readMetadata(fits::MemFileManager & manager, int hdu=0, bool strip=false);
PTR(daf::base::PropertyList) readMetadata(fits::Fits & fitsfile, bool strip=false);
//@}

}}} /// namespace lsst::afw::fits

#endif // !LSST_AFW_fits_h_INCLUDED
