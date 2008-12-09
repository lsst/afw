// -*- lsst-c++ -*-
/**
 * @file
 * @brief Implementation of Wcs as a thin wrapper around wcslib
 */
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstring>

#include "wcslib/wcs.h"
#include "wcslib/wcsfix.h"
#include "wcslib/wcshdr.h"

#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/daf/data/FitsFormatter.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Wcs.h"

using lsst::daf::base::DataProperty;
using lsst::daf::data::LsstBase;
using lsst::daf::data::FitsFormatter;
using lsst::afw::image::PointD;

/**
 * @brief Construct an invalid Wcs given no arguments
 *
 * @throw lsst::pex::exceptions::Runtime on error
 */
lsst::afw::image::Wcs::Wcs() :
    LsstBase(typeid(this)),
    _fitsMetadata(),
    _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0) {
}


/**
 * @brief Construct a Wcs that performs a linear conversion between pixels and radec
 *
 * I'm mostly playing here, I have little idea whether any of this is correct
 *
 */
lsst::afw::image::Wcs::Wcs(PointD crval, ///< ra/dec of centre of image
                           PointD crpix, ///< pixel coordinates of centre of image
                           boost::numeric::ublas::matrix<double> CD ///< Conversion matrix with elements as defined
                                                                    ///< in wcs.h
                          ) : LsstBase(typeid(this)),
                              _fitsMetadata(),
                              _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0) {
    //Allocate memory for the wcs structure
    wcsini(true, 2, _wcsInfo);   //Is this remotely correct?

    _wcsInfo->crval[0] = crval.getX();
    _wcsInfo->crval[1] = crval.getY();

    _wcsInfo->crpix[0] = crpix.getX();
    _wcsInfo->crpix[1] = crpix.getY();

    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            _wcsInfo->cd[(2*i) + j] = CD(i,j);
        }
    }

    //Specify that we have a CD matrix, but no PC or CROTA
    _wcsInfo->altlin &= 2;
}

    
    
/**
 * @brief Construct a Wcs from a FITS header, represented as DataProperty::PtrType
 *
 * @throw lsst::pex::exceptions::Runtime on error
 */
lsst::afw::image::Wcs::Wcs(
    lsst::daf::base::DataProperty::PtrType fitsMetadata  ///< The contents of a valid FITS header
) :
    LsstBase(typeid(this)),
    _fitsMetadata(fitsMetadata),
    _wcsInfo(NULL),
    _nWcsInfo(0),
    _nReject(0)
{
    // these should be set via policy - but for the moment...

    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    std::string metadataStr = FitsFormatter::formatDataProperty( fitsMetadata, false );
    int nCards = FitsFormatter::countFITSHeaderCards( fitsMetadata, false );
    if (nCards <= 0) {
        throw lsst::pex::exceptions::Runtime("Could not parse FITS WCS: no header cards found");
    }
    
    // wcspih takes a non-const char* (because some versions of ctrl modify the string)
    // but we cannot afford to allow that to happen, so make a copy...
    int len = metadataStr.size();
    char *hdrString = new(char[len+1]);
    std::strcpy(hdrString, metadataStr.c_str());

    int pihStatus = wcspih(hdrString, nCards, _relax, _wcshdrCtrl, &_nReject, &_nWcsInfo, &_wcsInfo);
    delete hdrString;
    if (pihStatus != 0) {
        throw lsst::pex::exceptions::Runtime(
            boost::format("Could not parse FITS WCS: wcspih status = %d") % pihStatus);
    }

    /*
     * Fix any bad values in the Wcs
     * Should we throw an exception or continue if this fails?
     * For now be paranoid...
     */

    const int *naxes = NULL;            // should be {NAXIS1, NAXIS2, ...} to check cylindrical projections
    int stats[NWCSFIX];			// status returns from wcsfix
    int fixStatus = wcsfix(_wcsfixCtrl, naxes, _wcsInfo, stats);
    if (fixStatus != 0) {
        std::stringstream errStream;
        errStream << "Could not parse FITS WCS: wcsfix failed " << std::endl;
        for (int ii = 0; ii < NWCSFIX; ++ii) {
	  if (stats[ii] >= 0) {
	    errStream << "\t" << ii << ": " << stats[ii] << " " << wcsfix_errmsg[stats[ii]] << std::endl;
	  } else {
	    errStream << "\t" << ii << ": " << stats[ii] << std::endl;
	  }
	}
#if 0	  
         throw lsst::pex::exceptions::Runtime(errStream.str());
#endif
    }
}

/**
 * @brief Wcs copy constructor
 *
 * @throw lsst::pex::exceptions::Memory or lsst::pex::exceptions::Runtime on error
 */
lsst::afw::image::Wcs::Wcs(Wcs const & rhs):
    LsstBase(typeid(this)),
    _fitsMetadata(rhs._fitsMetadata),
    _wcsInfo(NULL),
    _nWcsInfo(0),
    _relax(rhs._relax),
    _wcsfixCtrl(rhs._wcsfixCtrl),
    _wcshdrCtrl(rhs._wcshdrCtrl),
    _nReject(rhs._nReject)
{
    if (rhs._nWcsInfo > 0) {
        _wcsInfo = static_cast<struct wcsprm *>(calloc(rhs._nWcsInfo, sizeof(struct wcsprm)));
        if (_wcsInfo == NULL) {
            throw lsst::pex::exceptions::Memory("Cannot allocate WCS info");
        }
        _nWcsInfo = rhs._nWcsInfo;
        for (int ii = 0; ii < rhs._nWcsInfo; ++ii) {
            // wcssub deep copies each _wcsInfo structure into newly allocated memory
            // this memory is managed by wcslib and so must be freed by wcsfree
            _wcsInfo[ii].flag = -1;
            int status = wcscopy(1, rhs._wcsInfo + ii, _wcsInfo + ii);
            if (status != 0) {
                wcsvfree(&_nWcsInfo, &_wcsInfo);
                throw lsst::pex::exceptions::Runtime(
                    boost::format("Could not copy WCS: wcscopy status = %d for wcs index %d") % status % ii);
            }
        }
    }
}

/**
 * @brief Wcs assignment operator
 *
 * @throw lsst::pex::exceptions::Memory or lsst::pex::exceptions::Runtime on error
 */
lsst::afw::image::Wcs & lsst::afw::image::Wcs::operator = (const lsst::afw::image::Wcs & rhs) {
    if (this != &rhs) {
        if (_nWcsInfo > 0) {
            wcsvfree(&_nWcsInfo, &_wcsInfo);
        }
        _fitsMetadata = rhs._fitsMetadata;
        _nWcsInfo = 0;
        _wcsInfo = NULL;
        _relax = rhs._relax;
        _wcsfixCtrl = rhs._wcsfixCtrl;
        _wcshdrCtrl = rhs._wcshdrCtrl;
        _nReject = rhs._nReject;
        if (rhs._nWcsInfo > 0) {
            // allocate wcs structs
            _wcsInfo = static_cast<struct wcsprm *>(calloc(rhs._nWcsInfo, sizeof(struct wcsprm)));
            if (_wcsInfo == NULL) {
                throw lsst::pex::exceptions::Memory("Cannot allocate WCS info");
            }
            _nWcsInfo = rhs._nWcsInfo;
            // deep-copy wcs data
            for (int ii = 0; ii < rhs._nWcsInfo; ++ii) {
                _wcsInfo[ii].flag = -1;
                int status = wcscopy(1, rhs._wcsInfo + ii, _wcsInfo + ii);
                if (status != 0) {
                    wcsvfree(&_nWcsInfo, &_wcsInfo);
                    throw lsst::pex::exceptions::Runtime(
                        boost::format("Failed to copy WCS info; wcscopy status = %d") % status);
                }
            }
        }
    }
    
    return *this;
}

/// Destructor for Wcs
lsst::afw::image::Wcs::~Wcs() {
    if (_wcsInfo != NULL) {
        wcsvfree(&_nWcsInfo, &_wcsInfo);
    }
}


/// Return (ra, dec) of the central pixel of the image
lsst::afw::image::PointD lsst::afw::image::Wcs::getRaDecCenter() const {
    return PointD(_wcsInfo->crval);
}


/// Return row column number of central pixel of the image
lsst::afw::image::PointD lsst::afw::image::Wcs::getColRowCenter() const {
    return PointD(_wcsInfo->crpix);
}


/// Convert from (ra, dec) to (column, row) coordinates
///
/// The conversion is of the form
/// / ra \ = / c11 c12 \   [ col ]
/// \dec /   \ c21 c22 /   [ row ] 
///
/// where (col,row) = (0,0) = (ra, dec) is the centre of the image, and the matrix C is return by this function.
boost::numeric::ublas::matrix<double> lsst::afw::image::Wcs::getLinearTransformMatrix() const {
    int const naxis = _wcsInfo->naxis;

    //If naxis != 2, I'm not sure if any of what follows is correct
    assert(naxis == 2);
    
    boost::numeric::ublas::matrix<double> C(naxis, naxis);

    for (int i=0; i< naxis; ++i){
        for (int j=0; j<naxis; ++j) {
            C.insert_element(i, j, _wcsInfo->cd[ (i*naxis) + j ]);
        }
    }

    return C;
}
                           




/// Convert from (ra, dec) to (column, row) coordinates
///
/// \return The desired (col, row) position
lsst::afw::image::PointD lsst::afw::image::Wcs::raDecToColRow(
    const double ra,   ///< Input right ascension
    const double dec   ///< Input declination
) const {
    double const skyTmp[2] = { ra, dec };
    double imgcrd[2];
    double phi, theta;
    double pixTmp[2];

    int status = 0;
    wcss2p(_wcsInfo, 1, 2, skyTmp, &phi, &theta, imgcrd, pixTmp, &status);

    return lsst::afw::image::PointD(pixTmp);
}


/// \return The desired (col, row) position
lsst::afw::image::PointD lsst::afw::image::Wcs::raDecToColRow(
    lsst::afw::image::PointD sky        ///< Input (ra, dec)
) const {
    return raDecToColRow(sky.getX(), sky.getY());
}

/// Convert from (column, row) to (ra, dec) coordinates
/// \return The desired (ra, dec) position
lsst::afw::image::PointD lsst::afw::image::Wcs::colRowToRaDec(
    double const x,                     ///< Input column position
    double const y                      ///< Input row position
) const {
    double const pixTmp[2] = { x, y };
    double imgcrd[2];
    double phi, theta;
    double skyTmp[2];

    int status = 0;
    wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, &phi, &theta, skyTmp, &status);

    return lsst::afw::image::PointD(skyTmp);
}

/// Convert from (column, row) to (ra, dec) coordinates
/// \return The desired (ra, dec) position
lsst::afw::image::PointD lsst::afw::image::Wcs::colRowToRaDec(
    lsst::afw::image::PointD pix        ///< Input (x, y)
) const {
    return colRowToRaDec(pix.getX(), pix.getY());
}

/// Return the pixel area in deg^2 at a given pixel coordinate
double lsst::afw::image::Wcs::pixArea(lsst::afw::image::PointD pix0) const
{
    lsst::afw::image::PointD sky0, sky1, deltaSky;
    lsst::afw::image::PointD pix1 = pix0 + lsst::afw::image::PointD(1,1);

    sky0 = colRowToRaDec(pix0);
    sky1 = colRowToRaDec(pix1);

    deltaSky = sky1 - sky0;

    double cosDec, area;

    cosDec = cos(sky0.getY() * M_PI/180.0);
    area = std::fabs(deltaSky.getX()*cosDec * deltaSky.getY());

    return area;
}
