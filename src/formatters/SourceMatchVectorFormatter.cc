// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for PersistableSourceMatchVector instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/SourceMatch.h"
#include "lsst/afw/formatters/SourceMatchVectorFormatter.h"

namespace lsst { namespace afw { namespace formatters {

LSST_EXCEPTION_TYPE(FitsException,
                    lsst::pex::exceptions::Exception, lsst::pex::exceptions::LogicErrorException)

namespace cfitsio {
extern "C" {
#include "fitsio.h"
}

    // stolen from afw/image/fits/fits_io_private.h:

    std::string err_msg(fitsfile const *fd, int const status = 0, std::string const &errMsg = "");
    std::string err_msg(std::string const &fileName, int const status = 0, std::string const &errMsg = "");
    inline std::string err_msg(fitsfile const *fd, int const status, boost::format const &fmt) { return err_msg(fd, status, fmt.str()); }
    void appendKey(lsst::afw::formatters::cfitsio::fitsfile* fd, std::string const &keyWord,
                   std::string const& keyComment, boost::shared_ptr<const lsst::daf::base::PropertySet> metadata);


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
        (void)lsst::afw::formatters::cfitsio::fits_get_errstatus(status, fitsErrMsg);
        os << ": " << fitsErrMsg << " (" << status << ")";
    }
    if (errMsg != "") {
        os << " : " << errMsg;
    }
    return os.str();
}

std::string err_msg(lsst::afw::formatters::cfitsio::fitsfile const * fd, ///< (possibly invalid) file descriptor
                    int const status, ///< cfitsio error status (default 0 => no error)
                    std::string const & errMsg ///< optional error description
                   ) {
    std::string fileName = "";
    if (fd != 0 && fd->Fptr != 0 && fd->Fptr->filename != 0) {
        fileName = fd->Fptr->filename;
    }
    return err_msg(fileName, status, errMsg);
}

void appendKey(lsst::afw::formatters::cfitsio::fitsfile* fd, std::string const &keyWord,
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


} // end of cfitsio namespace
}}}

namespace ex = lsst::pex::exceptions;
namespace det = lsst::afw::detection;

using lsst::daf::base::Persistable;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::SourceMatch;
using lsst::afw::detection::SourceMatchVector;
using lsst::afw::detection::PersistableSourceMatchVector;

namespace form = lsst::afw::formatters;

// -- SourceMatchVectorFormatter ----------------

form::SourceMatchVectorFormatter::SourceMatchVectorFormatter(Policy::Ptr const & policy) :
    lsst::daf::persistence::Formatter(typeid(this)),
    _policy(policy)
{}


form::SourceMatchVectorFormatter::~SourceMatchVectorFormatter() {}


lsst::daf::persistence::Formatter::Ptr form::SourceMatchVectorFormatter::createInstance(Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new SourceMatchVectorFormatter(policy));
}


lsst::daf::persistence::FormatterRegistration form::SourceMatchVectorFormatter::registration(
    "PersistableSourceMatchVector",
    typeid(PersistableSourceMatchVector),
    createInstance
);

/*
readFits():

            if (fits_open_file(&_fd_s, filename.c_str(), READONLY, &status) != 0) {
                throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
            }

 fits_open_table ?

 fits_moveabs_hdu
 fits_get_num_hdus
 fits_get_hdu_type

 fits_get_num_rows
 fits_get_num_cols
 fits_get_colnum
 fits_get_colname
 fits_get_coltype

 fits_read_col

 */


void form::SourceMatchVectorFormatter::writeFits(const PersistableSourceMatchVector* p,
                                                 FitsStorage* fs,
                                                 lsst::daf::base::PropertySet::Ptr metadata) {
    SourceMatchVector matches = p->getSourceMatches();

    printf("Persisting a list of %i sources to file \"%s\"\n",
           (int)matches.size(), fs->getPath().c_str());
    for (size_t i=0; i<matches.size(); i++) {
        printf("  %i: %li ==> %li\n", (int)i, matches[i].first->getSourceId(),
               matches[i].second->getSourceId());
    }

    // Much of this is stolen from afw/image/fits/fits_io_private.h --
    // should be refactored!!!

    cfitsio::fitsfile* fitsfile = NULL;
    int status;
    //char* fn = fs->getPath().c_str();
    std::string filename = fs->getPath();

    (void)unlink(filename.c_str()); // cfitsio doesn't like over-writing files

    status = 0;
    if (fits_create_file(&fitsfile, filename.c_str(), &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
    }

    /*
     fits_update_key(fitsfile, TLONG, "EXPOSURE", &exposure, 
     "Total Exposure Time", &status); 
     */

    status = 0;
    if (fits_create_img(fitsfile, 8, 0, NULL, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }

    if (metadata != NULL) {
        typedef std::vector<std::string> NameList;
        NameList paramNames = metadata->paramNames(false);
        for (NameList::const_iterator i = paramNames.begin(), e = paramNames.end(); i != e; ++i) {
            if (*i != "SIMPLE" && *i != "BITPIX" &&
                *i != "NAXIS" && *i != "NAXIS1" && *i != "NAXIS2" && *i != "EXTEND") {
                cfitsio::appendKey(fitsfile, *i, "", metadata);
            }
        }
    }

    status = 0;
    int ncols = 2;
    int result;
    std::string coltype = "K";
    std::string cname0 = "SOURCE_ID";
    std::string cname1 = "REF_ID";
    char* tform[] = { strdup(coltype.c_str()), strdup(coltype.c_str()) };
    char* ttype[] = { strdup(cname0.c_str()),  strdup(cname1.c_str()) };

    std::string extname = "MatchList";
    char* cextname = strdup(extname.c_str());

    result = fits_create_tbl(fitsfile, BINARY_TBL, 0, ncols, ttype, tform, NULL, cextname, &status);
    // free these before raising exception...
    free(cextname);
    for (int i=0; i<ncols; i++) {
        free(tform[i]);
        free(ttype[i]);
    }
    if (result) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }

    int nrows = matches.size();
    int64_t* values = new int64_t[nrows];
    for (int i=0; i<nrows; i++)
        values[i] = matches[i].first->getSourceId();
    // fitsfile, datatype, column, firstrow, firstelement, nelements, data, status
    result = fits_write_col(fitsfile, TLONGLONG, 0, 0, 0, nrows, values, &status);
    if (result) {
        free(values);
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }

    for (int i=0; i<nrows; i++)
        values[i] = matches[i].second->getSourceId();
    result = fits_write_col(fitsfile, TLONGLONG, 1, 0, 0, nrows, values, &status);
    free(values);
    if (result) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }

    status = 0;
    if (cfitsio::fits_close_file(fitsfile, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }

}

/** 
 * Persist a PersistableSourceMatchVector to FitsStorage
 */
void form::SourceMatchVectorFormatter::write(
    Persistable const * persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {
    if (!persistable)
        throw LSST_EXCEPT(ex::InvalidParameterException, "No Persistable provided");
    if (!storage)
        throw LSST_EXCEPT(ex::InvalidParameterException, "No Storage provided");

    PersistableSourceMatchVector const * p = dynamic_cast<PersistableSourceMatchVector const *>(persistable);
    if (!p)
        throw LSST_EXCEPT(ex::RuntimeErrorException,
                          "Persistable was not of concrete type SourceMatchVector");

    if (typeid(*storage) == typeid(FitsStorage)) {
        FitsStorage* bs = dynamic_cast<FitsStorage *>(storage.get());

        printf("SourceMatchVectorFormatter: persisting to path \"%s\"\n",
               bs->getPath().c_str());
        writeFits(p, bs, additionalData);

    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, 
                          "Storage type is not supported"); 
    }
}


/** 
 * Retrieve a PersistableSourceMatchVector from FitsStorage.
 */
Persistable* form::SourceMatchVectorFormatter::read(
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {   
    std::auto_ptr<PersistableSourceMatchVector> p(new PersistableSourceMatchVector);
    
    if (typeid(*storage) == typeid(FitsStorage)) {
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, 
                          "Storage type is not supported");
    }
    return p.release();
}

void form::SourceMatchVectorFormatter::update(Persistable*, 
                                              Storage::Ptr, lsst::daf::base::PropertySet::Ptr
                                              ) {
    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                      "SourceMatchVectorFormatter: updates not supported");
}

