// -*- lsst-c++ -*-
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


namespace lsst {
namespace afw {
namespace image {
namespace cfitsio {
                
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
        throw LSST_EXCEPT(FitsException, (boost::format("Unsupported value BITPIX==%d") % bitpix).str());
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
            throw LSST_EXCEPT(FitsException,
                          err_msg(fd, status, boost::format("Attempted to select relative HDU %d") % hdu));
        }
    } else {
        if (hdu == 0) { // PDU; go there
            hdu = 1;
        } else {
            if (fits_movabs_hdu(fd, hdu, NULL, &status) != 0) {
                throw LSST_EXCEPT(FitsException,
                            err_msg(fd, status, boost::format("Attempted to select absolute HDU %d") % hdu));
            }
        }
    }
}

/************************************************************************************************************/
// append a record to the FITS header.   Note the specialization to string values

void appendKey(lsst::afw::image::cfitsio::fitsfile* fd, std::string const &keyWord,
               std::string const &keyComment, boost::shared_ptr<const lsst::daf::base::PropertySet> metadata) {

    // NOTE:  the sizes of arrays are tied to FITS standard
    // These shenanigans are required only because fits_write_key does not take const args...
    
    char keyWordChars[80];
    char keyValueChars[80];
    char keyCommentChars[80];
    
    strncpy(keyWordChars, keyWord.c_str(), 80);
    strncpy(keyCommentChars, keyComment.c_str(), 80);
    
    int status = 0;
    std::type_info const & valueType = metadata->typeOf(keyWord); 
    if (valueType == typeid(bool)) {
        if (metadata->isArray(keyWord)) {
            std::vector<bool> tmp = metadata->getArray<bool>(keyWord);
            for (unsigned int i = 0; i != tmp.size(); ++i) {
                bool tmp_i = tmp[i];    // avoid icc warning; is vector<bool> special as it only needs 1 bit?
                fits_write_key(fd, TLOGICAL, keyWordChars, &tmp_i, keyCommentChars, &status);
            }
        } else {
            bool tmp = metadata->get<bool>(keyWord);

            fits_write_key(fd, TLOGICAL, keyWordChars, &tmp, keyCommentChars, &status);
        }
    } else if (valueType == typeid(int)) {
        if (metadata->isArray(keyWord)) {
            std::vector<int> tmp = metadata->getArray<int>(keyWord);
            for (unsigned int i = 0; i != tmp.size(); ++i) {
                fits_write_key(fd, TINT, keyWordChars, &tmp[i], keyCommentChars, &status);
            }
        } else {
            int tmp = metadata->get<int>(keyWord);

            fits_write_key(fd, TINT, keyWordChars, &tmp, keyCommentChars, &status);
        }
    } else if (valueType == typeid(long)) {
        if (metadata->isArray(keyWord)) {
            std::vector<long> tmp = metadata->getArray<long>(keyWord);
            for (unsigned long i = 0; i != tmp.size(); ++i) {
                fits_write_key(fd, TLONG, keyWordChars, &tmp[i], keyCommentChars, &status);
            }
        } else {
            long tmp = metadata->get<long>(keyWord);

            fits_write_key(fd, TLONG, keyWordChars, &tmp, keyCommentChars, &status);
        }
    } else if (valueType == typeid(double)) {
        if (metadata->isArray(keyWord)) {
            std::vector<double> tmp = metadata->getArray<double>(keyWord);
            for (unsigned int i = 0; i != tmp.size(); ++i) {
                fits_write_key(fd, TDOUBLE, keyWordChars, &tmp[i], keyCommentChars, &status);
            }
        } else {
            double tmp = metadata->get<double>(keyWord);
            fits_write_key(fd, TDOUBLE, keyWordChars, &tmp, keyCommentChars, &status);
        }
    } else if (valueType == typeid(std::string)) {
        if (metadata->isArray(keyWord)) {
            std::vector<std::string> tmp = metadata->getArray<std::string>(keyWord);

            for (unsigned int i = 0; i != tmp.size(); ++i) {
                strncpy(keyValueChars, tmp[i].c_str(), 80);
                fits_write_key(fd, TSTRING, keyWordChars, keyValueChars, keyCommentChars, &status);
            }
        } else {
            std::string tmp = metadata->get<std::string>(keyWord);
            strncpy(keyValueChars, tmp.c_str(), 80);
            fits_write_key(fd, TSTRING, keyWordChars, keyValueChars, keyCommentChars, &status);
        }
    } else {
        std::cerr << "In " << BOOST_CURRENT_FUNCTION << " Unknown type: " << valueType.name() <<
            " for keyword " << keyWord << std::endl;
    }

    if (status) {
        throw LSST_EXCEPT(FitsException, err_msg(fd, status));
    }
}

//! Get the number of keywords in the header
int getNumKeys(fitsfile* fd) {
     int keynum = 0;
     int numKeys = 0;
     int status = 0;
 
     if (fits_get_hdrpos(fd, &numKeys, &keynum, &status) != 0) {
          throw LSST_EXCEPT(FitsException, err_msg(fd, status));
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
          throw LSST_EXCEPT(FitsException, err_msg(fd, status));
     }
         
     keyWord = keyWordChars;
     keyValue = keyValueChars;
     keyComment = keyCommentChars;
}

void addKV(lsst::daf::base::PropertySet::Ptr metadata, std::string key, std::string value) {
    boost::regex const boolRegex("[tTfF]");
    boost::regex const intRegex("(\\Q+\\E|\\Q-\\E){0,1}[0-9]+");
    boost::regex const doubleRegex("(\\Q+\\E|\\Q-\\E){0,1}([0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*)((e|E)(\\Q+\\E|\\Q-\\E){0,1}[0-9]+){0,1}");
    boost::regex const fitsStringRegex("'(.*)'");

    boost::smatch matchStrings;
    std::istringstream converter(value);

    if (boost::regex_match(value, boolRegex)) {
        // convert the string to an bool
#if 0
        bool val;
        converter >> val;               // converter doesn't handle bool; T/F always return 255
#else
        bool val = (value == "T" || value == "t");
#endif
        metadata->add(key, val);
    } else if (boost::regex_match(value, intRegex)) {
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
    void getMetadata(fitsfile* fd, lsst::daf::base::PropertySet::Ptr metadata, bool strip) {
    // Get all the kw-value pairs from the FITS file, and add each to DataProperty
    if (metadata.get() == NULL) {
        return;
    }

    for (int i=1; i<=getNumKeys(fd); i++) {
        std::string keyName;
        std::string val;
        std::string comment;
        getKey(fd, i, keyName, val, comment);

        if (strip && (keyName == "SIMPLE" || keyName == "BITPIX" || keyName == "EXTEND" ||
                      keyName == "NAXIS" || keyName == "NAXIS1" || keyName == "NAXIS2" ||
                      keyName == "BSCALE" || keyName == "BZERO")) {
            ;
        } else {
            addKV(metadata, keyName, val);
        }
    }
}
} // namespace cfitsio

/************************************************************************************************************/

/**
 * \brief Return the metadata from a fits file
 */
lsst::daf::base::PropertySet::Ptr readMetadata(std::string const& fileName, ///< File to read
                                               const int hdu,               ///< HDU to read
                                               bool strip       ///< Should I strip e.g. NAXIS1 from header?
                                              ) {
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertySet);

    detail::fits_reader m(fileName, metadata, hdu);
    cfitsio::getMetadata(m.get(), metadata, strip);

    return metadata;
}
    
}}} // namespace lsst::afw::image
