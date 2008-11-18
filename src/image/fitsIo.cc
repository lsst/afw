/// \file
/// \brief  Utilities that use cfitsio
/// \author Robert Lupton (rhl@astro.princeton.edu)\n
///         Princeton University
/// \date   September 2008
#include <cstring>
#include "boost/format.hpp"
#include "lsst/afw/image/fits/fits_io_private.h"

#include "lsst/pex/exceptions.h"


namespace lsst { namespace afw { namespace image { namespace cfitsio {
                
void _throw_cfitsio_error(const char *file, const int line,   //!< line in file (from __FILE__, __LINE__)
                          lsst::afw::image::cfitsio::fitsfile *fd, //!< (possibly invalid) file descriptor
                          const int status,                        //!< cfitsio error status (default 0 => no error)
                          const std::string errStr //!< optional extra information
                         ) {
    if (status == 0) {
        if (errStr == "") {
            return;
        }

        throw lsst::pex::exceptions::FitsError(boost::format("%s:%d: %s") % file % line % errStr);
    } else {
        char fitsErr[FLEN_ERRMSG];
        (void)lsst::afw::image::cfitsio::fits_get_errstatus(status, fitsErr);
        boost::format msg = boost::format("%s:%d: cfitsio error: %s%s%s")
            % file % line
            % fitsErr
            % (errStr == "" ? "" : " : ")
            % (errStr == "" ? std::string("") : errStr);
            
        switch (status) {
          case FILE_NOT_OPENED:
            throw lsst::pex::exceptions::FitsError(msg);
            break;
          default:
            throw lsst::pex::exceptions::FitsError(msg) << lsst::daf::base::DataProperty("status", status);
            break;
        }
    }
}

void _throw_cfitsio_error(const char *file, const int line, //!< line in file (from __FILE__/__LINE__)
                          lsst::afw::image::cfitsio::fitsfile *fd, //!< (possibly invalid) file descriptor
                          const std::string errStr //!< optional extra information
                         ) {
    _throw_cfitsio_error(file, line, fd, 0, errStr);
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
        throw lsst::pex::exceptions::FitsError(boost::format("Unsupported value BITPIX==%d") % bitpix);
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
            throw_cfitsio_error(fd, status, str(boost::format("Attempted to select relative HDU %d") % hdu));
        }
    } else {
        if (hdu == 0) { // PDU; go there
            hdu = 1;
        } else {
            if (fits_movabs_hdu(fd, hdu, NULL, &status) != 0) {
                throw_cfitsio_error(fd, status, str(boost::format("Attempted to select absolute HDU %d") % hdu));
            }
        }
    }
}

/************************************************************************************************************/
// append a record to the FITS header.   Note the specialization to string values

void appendKey(lsst::afw::image::cfitsio::fitsfile* fd,
               const std::string & keyWord, const boost::any & keyValue, const std::string & keyComment) {

    // NOTE:  the sizes of arrays are tied to FITS standard
    // These shenanigans are required only because fits_write_key does not take const args...
    
    char keyWordChars[80];
    char keyValueChars[80];
    char keyCommentChars[80];
    
    strncpy(keyWordChars, keyWord.c_str(), 80);
    strncpy(keyCommentChars, keyComment.c_str(), 80);
    
    int status = 0;
    if (keyValue.type() == typeid(int)) {
        int tmp = boost::any_cast<const int>(keyValue);
        fits_write_key(fd, TINT, keyWordChars, &tmp, keyCommentChars, &status);

    } else if (keyValue.type() == typeid(double)) {
        double tmp = boost::any_cast<const double>(keyValue);
        fits_write_key(fd, TDOUBLE, keyWordChars, &tmp, keyCommentChars, &status);

    } else if (keyValue.type() == typeid(std::string)) {
        std::string tmp = boost::any_cast<const std::string>(keyValue);
        strncpy(keyValueChars, tmp.c_str(), 80);
        fits_write_key(fd, TSTRING, keyWordChars, keyValueChars, keyCommentChars, &status);
    }

    if (status) {
        throw_cfitsio_error(fd, status);
    }
}

//! Get the number of keywords in the header
int getNumKeys(fitsfile* fd) {
     int keynum = 0;
     int numKeys = 0;
     int status = 0;
 
     if (fits_get_hdrpos(fd, &numKeys, &keynum, &status) != 0) {
          throw_cfitsio_error(fd, status);
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
          throw_cfitsio_error(fd, status);
     }
         
     keyWord = keyWordChars;
     keyValue = keyValueChars;
     keyComment = keyCommentChars;
}

// Private function to build a lsst::daf::base::DataProperty that contains all the FITS kw-value pairs

void getMetadata(fitsfile* fd,
                 lsst::daf::base::DataProperty::PtrType metadata) {
    // Get all the kw-value pairs from the FITS file, and add each to DataProperty

    if (metadata.get() == NULL) {
        return;
    }

    if( metadata->isNode() != true ) {
        throw lsst::pex::exceptions::InvalidParameter( "Given metadata object is not a lsst::daf::base::DataProperty node" );
    }
    
    for (int i=1; i<=getNumKeys(fd); i++) {
        std::string keyName;
        std::string val;
        std::string comment;
        getKey(fd, i, keyName, val, comment);

        if (keyName != "SIMPLE" && keyName != "BITPIX" && keyName != "EXTEND" &&
            keyName != "NAXIS" && keyName != "NAXIS1" && keyName != "NAXIS2" &&
            keyName != "BSCALE" && keyName != "BZERO") {
            lsst::daf::base::DataProperty::PtrType dpItemPtr(
                                                             new lsst::daf::base::DataProperty(keyName,
                                                                                               lsst::utils::stringToAny(val)));
            metadata->addProperty(dpItemPtr);
        }
    }
}
                
}}}}                             // namespace lsst::afw::image::cfitsio
