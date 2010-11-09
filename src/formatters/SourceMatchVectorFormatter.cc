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
#include "lsst/afw/detection/Source.h"
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
using lsst::afw::detection::Source;
using lsst::afw::detection::SourceMatch;
using lsst::afw::detection::SourceMatchVector;
using lsst::afw::detection::PersistableSourceMatchVector;

namespace form = lsst::afw::formatters;

// -- SourceMatchVectorFormatter ----------------

// constants
const std::string form::SourceMatchVectorFormatter::REF_CAT_ID_COLUMN_NAME("REF_CAT_ID");
const std::string form::SourceMatchVectorFormatter::REF_ID_COLUMN_NAME("REF_ID");
const std::string form::SourceMatchVectorFormatter::SOURCE_ID_COLUMN_NAME("SOURCE_ID");

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


void form::SourceMatchVectorFormatter::readFits(PersistableSourceMatchVector* p,
                                                FitsStorage* fs,
                                                lsst::daf::base::PropertySet::Ptr metadata) {
    //printf("SourceMatchVectorFormatter: persisting to path \"%s\"\n", fs->getPath().c_str());

    cfitsio::fitsfile* fitsfile = NULL;
    int status;
    std::string filename = fs->getPath();

    status = 0;
    if (fits_open_file(&fitsfile, filename.c_str(), READONLY, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
    }

    int nhdu = 0;
    if (fits_get_num_hdus(fitsfile, &nhdu, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
    }
    //printf("%i HDUs\n", nhdu);

    if (nhdu != 2) {
        std::ostringstream os;
        os << "readFits: expected 2 HDUs, got " << nhdu;
        throw LSST_EXCEPT(FitsException, os.str());
    }

    if (fits_movabs_hdu(fitsfile, 2, NULL, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
    }

    //int hdu = -1;
    //printf("Current HDU: %i\n", fits_get_hdu_num(fitsfile, &hdu));

    int hdutype = -1;
    if (fits_get_hdu_type(fitsfile, &hdutype, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(filename, status));
    }
    //printf("Current HDU type: %i (want %i)\n", hdutype, BINARY_TBL);

    if (hdutype != BINARY_TBL) {
        throw LSST_EXCEPT(FitsException, "Expected FITS binary table extension; got type code " + hdutype);
    }

    long nrows;
    if (fits_get_num_rows(fitsfile, &nrows, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }
    //printf("Table has %i rows\n", (int)nrows);

    int ncols;
    if (fits_get_num_cols(fitsfile, &ncols, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }
    //printf("Table has %i columns\n", ncols);
    int NCOLS = 3;
    if (ncols != NCOLS) {
        throw LSST_EXCEPT(FitsException, (boost::format("Expected %i columns; got %i") % NCOLS % ncols).str());
    }

	SourceMatchVector smv;
    for (int i=0; i<nrows; i++) {
		Source::Ptr s1(new Source());
		Source::Ptr s2(new Source());
		SourceMatch m;
		m.first = s1;
		m.second = s2;
		m.distance = 0.0;
		smv.push_back(m);
    }

    for (int i=0; i<NCOLS; i++) {
        const std::string names[] = { REF_CAT_ID_COLUMN_NAME, REF_ID_COLUMN_NAME, SOURCE_ID_COLUMN_NAME };
        const std::string strcname = names[i];
        int col;
        char* cname;

        cname = strdup(strcname.c_str());
        if (fits_get_colnum(fitsfile, CASEINSEN, cname, &col, &status)) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
        }
        free(cname);

        int coltype;
        long repeat;
        if (fits_get_coltype(fitsfile, col, &coltype, &repeat, NULL, &status)) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
        }
        if (coltype != TLONGLONG) {
            throw LSST_EXCEPT(FitsException, (boost::format("Expected column \"%s\" to be type TLONGLONG, got code %i")
                                              % strcname % coltype).str());
        }
        if (repeat != 1) {
            throw LSST_EXCEPT(FitsException, (boost::format("Expected column \"%s\" to be a scalar, but got repeat count %i")
                                              % strcname % repeat).str());
        }

        int64_t* ids = new int64_t[nrows];
        int64_t nulval = -1;
        int anynul = 0;

        if (fits_read_col(fitsfile, TLONGLONG, col, 1, 1, nrows, &nulval, ids,
                          &anynul, &status)) {
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
        }

        for (int j=0; j<nrows; j++) {
            if (i == 0) {
                // pass
            } else if (i == 1) {
                smv[j].first->setSourceId(ids[j]);
            } else if (i == 2) {
                smv[j].second->setSourceId(ids[j]);
            }
        }
        //printf("  %li  =>  %li\n", (long)refids[i], (long)srcids[i]);

        delete[] ids;

    }
    p->setSourceMatches(smv);

    status = 0;
    if (cfitsio::fits_close_file(fitsfile, &status)) {
        throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
    }
}

void form::SourceMatchVectorFormatter::writeFits(const PersistableSourceMatchVector* p,
                                                 FitsStorage* fs,
                                                 lsst::daf::base::PropertySet::Ptr metadata) {
    SourceMatchVector matches = p->getSourceMatches();

    /*
     printf("Persisting a list of %i sources to file \"%s\"\n",
     (int)matches.size(), fs->getPath().c_str());
     for (size_t i=0; i<matches.size(); i++) {
     printf("  %i: %li ==> %li\n", (int)i, (long)matches[i].first->getSourceId(),
     (long)matches[i].second->getSourceId());
     }
     */

    // FIXME -- Much of this is stolen from afw/image/fits/fits_io_private.h --
    // should be refactored!!!

    cfitsio::fitsfile* fitsfile = NULL;
    int status;
    std::string filename = fs->getPath();

    // cfitsio doesn't like over-writing files
    (void)unlink(filename.c_str());

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
    int ncols = 3;
    int result;
    std::string coltype = "K";

    std::string strcname[] = { REF_CAT_ID_COLUMN_NAME, REF_ID_COLUMN_NAME, SOURCE_ID_COLUMN_NAME };

    char* tform[ncols];
    char* ttype[ncols];

    for (int i=0; i<ncols; i++) {
        tform[i] = strdup(coltype.c_str());
        ttype[i] = strdup(strcname[i].c_str());
    }

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

    for (int j=0; j<ncols; j++) {

        if (j == 0) {
            // HACK -- REF_CAT_ID = 0.
            for (int i=0; i<nrows; i++)
                values[i] = 0;
        } else if (j == 1) {
            // Grab the "first" object IDs -- by convention, "first" is the reference catalog object.
            for (int i=0; i<nrows; i++)
                values[i] = matches[i].first->getSourceId();
        } else if (j == 2) {
            // Grab the "second" object IDs -- by convention, "second" is the image source.
            for (int i=0; i<nrows; i++)
                values[i] = matches[i].second->getSourceId();
        }

        // In cfitsio's bizarro world, most things are 1-indexed (row, column, element)
        // fits_write_col(fitsfile, datatype, column, firstrow, firstelement, nelements, data, status)
        result = fits_write_col(fitsfile, TLONGLONG, j+1, 1, 1, nrows, values, &status);
        if (result) {
            delete[] values;
            throw LSST_EXCEPT(FitsException, cfitsio::err_msg(fitsfile, status));
        }
        
    }
    delete[] values;

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
        //printf("SourceMatchVectorFormatter: persisting to path \"%s\"\n", bs->getPath().c_str());
        writeFits(p, bs, additionalData);

    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Storage type is not supported"); 
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
        FitsStorage* bs = dynamic_cast<FitsStorage *>(storage.get());
        //printf("SourceMatchVectorFormatter: unpersisting from path \"%s\"\n", bs->getPath().c_str());
        readFits(p.get(), bs, additionalData);

    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Storage type is not supported");
    }
    return p.release();
}

void form::SourceMatchVectorFormatter::update(Persistable*, 
                                              Storage::Ptr, lsst::daf::base::PropertySet::Ptr
                                              ) {
    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                      "SourceMatchVectorFormatter: updates not supported");
}

