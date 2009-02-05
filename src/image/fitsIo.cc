/// \file
/// \brief  Utilities that use cfitsio
/// \author Robert Lupton (rhl@astro.princeton.edu)\n
///         Princeton University
/// \date   September 2008
#include <cstring>
#include "boost/format.hpp"
#include "boost/regex.hpp"

#include "lsst/pex/exceptions.h"

#include "lsst/afw/image/fits/fits_io_private.h"


namespace lsst { namespace afw { namespace image { namespace cfitsio {
                
std::string err_msg(std::string const& fileName, ///< (possibly empty) file name
                    int const status, ///< cfitsio error status (default 0 => no error)
                    std::string const & errMsg ///< optional error description
                   ) {
    std::ostringstream os;
    os << "cfitsio error";
    if (fileName != "") {
        os << " (" << fileName << ")";
    }
    if (status != 0) {
        char fitsErrMsg[FLEN_ERRMSG];
        (void)lsst::afw::image::cfitsio::fits_get_errstatus(status, fitsErrMsg);
        os << ": " << fitsErrMsg << " (" << status << ")";
    }
    if (errMsg != "") {
        os << " : " << errMsg;
    }
    return os.str();
}

std::string err_msg(lsst::afw::image::cfitsio::fitsfile const * fd, ///< (possibly invalid) file descriptor
                    int const status, ///< cfitsio error status (default 0 => no error)
                    std::string const & errMsg ///< optional error description
                   ) {
    std::string fileName = "";
    if (fd != 0 && fd->Fptr != 0 && fd->Fptr->filename != 0) {
        fileName = fd->Fptr->filename;
    }
    return err_msg(fileName, status, errMsg);
}

/************************************************************************************************************/
    
int ttypeFromBitpix(const int bitpix) {
    switch (bitpix) {
      case BYTE_IMG:
        return TBYTE;
      case SHORT_IMG:                   // int16
        return TSHORT;
      case USHORT_IMG:                  // uint16
        return TUSHORT;                 // n.b. cfitsio does magic things with bzero/bscale to make Uint16
      case LONG_IMG:                    // int32
        return TINT;
      case FLOAT_IMG:                   // float
        return TFLOAT;
      case DOUBLE_IMG:                  // double
        return TDOUBLE;
      default:
        throw LSST_EXCEPT(FitsErrorException, (boost::format("Unsupported value BITPIX==%d") % bitpix).str());
    }
}

/******************************************************************************/
//! \brief Move to the specified HDU
void move_to_hdu(lsst::afw::image::cfitsio::fitsfile *fd, //!< cfitsio file descriptor
                 int hdu,               //!< desired HDU
                 bool relative          //!< Is move relative to current HDU? (default: false)
                ) {
    int status = 0;     // cfitsio status
        
    if (relative) {
        if (fits_movrel_hdu(fd, hdu, NULL, &status) != 0) {
            throw LSST_EXCEPT(FitsErrorException,
                              err_msg(fd, status, boost::format("Attempted to select relative HDU %d") % hdu));
        }
    } else {
        if (hdu == 0) { // PDU; go there
            hdu = 1;
        } else {
            if (fits_movabs_hdu(fd, hdu, NULL, &status) != 0) {
                throw LSST_EXCEPT(FitsErrorException,
                                  err_msg(fd, status, boost::format("Attempted to select absolute HDU %d") % hdu));
            }
        }
    }
}

/************************************************************************************************************/
// append a record to the FITS header.   Note the specialization to string values

void appendKey(lsst::afw::image::cfitsio::fitsfile* fd, std::string const &keyWord,
               std::string const &keyComment, lsst::daf::base::PropertySet::Ptr metadata) {

    // NOTE:  the sizes of arrays are tied to FITS standard
    // These shenanigans are required only because fits_write_key does not take const args...
    
    char keyWordChars[80];
    char keyValueChars[80];
    char keyCommentChars[80];
    
    strncpy(keyWordChars, keyWord.c_str(), 80);
    strncpy(keyCommentChars, keyComment.c_str(), 80);
    
    int status = 0;
    std::type_info const & valueType = metadata->typeOf(keyWord); 
    if (valueType == typeid(int)) {
        int tmp = metadata->get<int>(keyWord);
        fits_write_key(fd, TINT, keyWordChars, &tmp, keyCommentChars, &status);

    } else if (valueType == typeid(double)) {
        double tmp = metadata->get<double>(keyWord);
        fits_write_key(fd, TDOUBLE, keyWordChars, &tmp, keyCommentChars, &status);

    } else if (valueType == typeid(std::string)) {
        std::string tmp = metadata->get<std::string>(keyWord);
        strncpy(keyValueChars, tmp.c_str(), 80);
        fits_write_key(fd, TSTRING, keyWordChars, keyValueChars, keyCommentChars, &status);
    }

    if (status) {
        throw LSST_EXCEPT(FitsErrorException, err_msg(fd, status));
    }
}

//! Get the number of keywords in the header
int getNumKeys(fitsfile* fd) {
     int keynum = 0;
     int numKeys = 0;
     int status = 0;
 
     if (fits_get_hdrpos(fd, &numKeys, &keynum, &status) != 0) {
          throw LSST_EXCEPT(FitsErrorException, err_msg(fd, status));
     }

     return numKeys;
}

void getKey(fitsfile* fd,
            int n, std::string & keyWord, std::string & keyValue, std::string & keyComment) {
     // NOTE:  the sizes of arrays are tied to FITS standard
     char keyWordChars[80];
     char keyValueChars[80];
     char keyCommentChars[80];

     int status = 0;
     if (fits_read_keyn(fd, n, keyWordChars, keyValueChars, keyCommentChars, &status) != 0) {
          throw LSST_EXCEPT(FitsErrorException, err_msg(fd, status));
     }
         
     keyWord = keyWordChars;
     keyValue = keyValueChars;
     keyComment = keyCommentChars;
}

void addKV(lsst::daf::base::PropertySet::Ptr metadata, std::string key, std::string value) {
    boost::regex const intRegex("(\\Q+\\E|\\Q-\\E){0,1}[0-9]+");
    boost::regex const doubleRegex("(\\Q+\\E|\\Q-\\E){0,1}([0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*)((e|E)(\\Q+\\E|\\Q-\\E){0,1}[0-9]+){0,1}");
    boost::regex const fitsStringRegex("'(.*)'");

    boost::smatch matchStrings;
    std::istringstream converter(value);

    if (boost::regex_match(value, intRegex)) {
        // convert the string to an int
        int val;
        converter >> val;
        metadata->add(key, val);
    } else if (boost::regex_match(value, doubleRegex)) {
        // convert the string to a double
        double val;
        converter >> val;
        metadata->add(key, val);
    } else if (boost::regex_match(value, matchStrings, fitsStringRegex)) {
        // strip off the enclosing single quotes and return the string
        metadata->add(key, matchStrings[1].str());
    }
}

// Private function to build a PropertySet that contains all the FITS kw-value pairs
void getMetadata(fitsfile* fd, lsst::daf::base::PropertySet::Ptr metadata) {
    // Get all the kw-value pairs from the FITS file, and add each to DataProperty
    if (metadata.get() == NULL) {
        return;
    }

    for (int i=1; i<=getNumKeys(fd); i++) {
        std::string keyName;
        std::string val;
        std::string comment;
        getKey(fd, i, keyName, val, comment);

        if (keyName != "SIMPLE" && keyName != "BITPIX" && keyName != "EXTEND" &&
            keyName != "NAXIS" && keyName != "NAXIS1" && keyName != "NAXIS2" &&
            keyName != "BSCALE" && keyName != "BZERO") {
            addKV(metadata, keyName, val);
        }
    }
}

}}}} // namespace lsst::afw::image::cfitsio
