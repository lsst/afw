/**
 * \file
 * \brief Implementation of WCS as a thin wrapper around wcslib
 */
#include <iostream>
#include <sstream>

#include <lsst/mwi/exceptions.h>
#include <lsst/mwi/data/FitsFormatter.h>

#include <lsst/fw/WCS.h>

#include <wcslib/wcs.h>
#include <wcslib/wcsfix.h>
#include <wcslib/wcshdr.h>

using namespace lsst::fw;
using namespace vw::math;

using lsst::mwi::data::LsstBase;
using lsst::mwi::data::DataProperty;
using lsst::mwi::data::FitsFormatter;

/**
 * \brief Construct an invalid WCS given no arguments
 *
 * \throw lsst::mwi::exceptions::Runtime on error
 */
WCS::WCS() : LsstBase(typeid(this)),
             _fitsMetaData(),
             _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0) {
}

/**
 * \brief Construct a WCS from a FITS header, represented as DataProperty::PtrType
 *
 * \throw lsst::mwi::exceptions::Runtime on error
 */
WCS::WCS(DataProperty::PtrType fitsMetaData  ///< The contents of a valid FITS header
        ) : LsstBase(typeid(this)),
            _fitsMetaData(fitsMetaData),
            _wcsInfo(NULL),
            _nWcsInfo(0),
            _nReject(0)
{
    // these should be set via policy - but for the moment...

    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    std::string metadataStr = FitsFormatter::formatDataProperty( fitsMetaData, false );
    int nCards = FitsFormatter::countFITSHeaderCards( fitsMetaData, false );
    if (nCards <= 0) {
        throw lsst::mwi::exceptions::Runtime("Could not parse FITS WCS: no header cards found");
    }
    
    // wcspih takes a non-const char* (because some versions of ctrl modify the string)
    // but we cannot afford to allow that to happen, so make a copy...
    int len = metadataStr.size();
    char *hdrString = new(char[len+1]);
    strcpy(hdrString, metadataStr.c_str());

    int pihStatus = wcspih(hdrString, nCards, _relax, _wcshdrCtrl, &_nReject, &_nWcsInfo, &_wcsInfo);
    delete hdrString;
    if (pihStatus != 0) {
        throw lsst::mwi::exceptions::Runtime(
            boost::format("Could not parse FITS WCS: wcspih status = %d") % pihStatus);
    }

    /*
     * Fix any bad values in the WCS
     * Should we throw an exception or continue if this fails?
     * For now be paranoid...
     */
    const int *naxes = NULL;            // should be {NAXIS1, NAXIS2, ...} to check cylindrical projections
    int stats[NWCSFIX];			// status returns from wcsfix
    int fixStatus = wcsfix(_wcsfixCtrl, naxes, _wcsInfo, stats);
    if (fixStatus != 0) {
        std::stringstream errStream;
        errStream << "Could not parse FITS WCS: wcsfix failed";
        for (int ii = 0; ii < NWCSFIX; ++ii) {
            if (stats[ii] != 0) {
                errStream << "; " << ii << ": " << wcsfix_errmsg[ii];
            }
        }
        errStream << std::endl;
        throw lsst::mwi::exceptions::Runtime(errStream.str());
    }
}

/**
 * \brief WCS copy constructor
 *
 * \throw lsst::mwi::exceptions::Memory or lsst::mwi::exceptions::Runtime on error
 */
WCS::WCS(WCS const & rhs):
        LsstBase(typeid(this)),
        _fitsMetaData(rhs._fitsMetaData),
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
            throw lsst::mwi::exceptions::Memory("Cannot allocate WCS info");
        }
        _nWcsInfo = rhs._nWcsInfo;
        for (int ii = 0; ii < rhs._nWcsInfo; ++ii) {
            // wcssub deep copies each _wcsInfo structure into newly allocated memory
            // this memory is managed by wcslib and so must be freed by wcsfree
            _wcsInfo[ii].flag = -1;
            int status = wcscopy(1, rhs._wcsInfo + ii, _wcsInfo + ii);
            if (status != 0) {
                wcsvfree(&_nWcsInfo, &_wcsInfo);
                throw lsst::mwi::exceptions::Runtime(
                    boost::format("Could not copy WCS: wcscopy status = %d for wcs index %d") % status % ii);
            }
        }
    }
}

/**
 * \brief WCS assignment operator
 *
 * \throw lsst::mwi::exceptions::Memory or lsst::mwi::exceptions::Runtime on error
 */
WCS & WCS::operator = (const WCS & rhs) {
    if (this != &rhs) {
        if (_nWcsInfo > 0) {
            wcsvfree(&_nWcsInfo, &_wcsInfo);
        }
        _fitsMetaData = rhs._fitsMetaData;
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
                throw lsst::mwi::exceptions::Memory("Cannot allocate WCS info");
            }
            _nWcsInfo = rhs._nWcsInfo;
            // deep-copy wcs data
            for (int ii = 0; ii < rhs._nWcsInfo; ++ii) {
                _wcsInfo[ii].flag = -1;
                int status = wcscopy(1, rhs._wcsInfo + ii, _wcsInfo + ii);
                if (status != 0) {
                    wcsvfree(&_nWcsInfo, &_wcsInfo);
                    throw lsst::mwi::exceptions::Runtime(
                        boost::format("Failed to copy wcs info; wcscopy status = %d") % status);
                }
            }
        }
    }
    
    return *this;
}

/// Destructor for WCS
WCS::~WCS() {
    if (_wcsInfo != NULL) {
        wcsvfree(&_nWcsInfo, &_wcsInfo);
    }
}

/// Convert from (ra, dec) to (column, row) coordinates
void WCS::raDecToColRow(Coord2D sky,    ///< Input (ra, dec)
                        Coord2D& pix    ///< Desired (col, row)
                       ) const {
    double skyTmp[2];
    skyTmp[0] = sky[0];
    skyTmp[1] = sky[1];

    double imgcrd[2];
    double phi[1];
    double theta[1];

    double pixTmp[2];

    int status[1];

    wcss2p(_wcsInfo, 1, 2, skyTmp, phi, theta, imgcrd, pixTmp, status);

    pix[0] = pixTmp[0];
    pix[1] = pixTmp[1];
}

/// Convert from (ra, dec) to (column, row) coordinates
///
/// \return The desired (col, row) position
Coord2D WCS::raDecToColRow(Coord2D sky  ///< Input (ra, dec)
                          ) const {
    Coord2D pix;
    raDecToColRow(sky, pix);

    return pix;
}

/// Convert from (ra, dec) to (column, row) coordinates
///
/// \return The desired (col, row) position
Coord2D WCS::raDecToColRow(double const ra,   ///< Input right ascension
                           double const dec   ///< Input declination
                          ) const {
    Coord2D sky(ra, dec);
    Coord2D pix;
    raDecToColRow(sky, pix);

    return pix;
}

/// Convert from (column, row) to (ra, dec) coordinates
void WCS::colRowToRaDec(Coord2D pix,    ///< Input (col, row)
                        Coord2D& sky    ///< Desired (ra, dec)
                       ) const {
    double pixTmp[2];
    pixTmp[0] = pix[0];
    pixTmp[1] = pix[1];

    double imgcrd[2];
    double phi[1];
    double theta[1];

    double skyTmp[2];

    int status[1];

    wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, phi, theta, skyTmp, status);

    sky[0] = skyTmp[0];
    sky[1] = skyTmp[1];
}

/// Convert from (column, row) to (ra, dec) coordinates
/// \return The desired (ra, dec) position
Coord2D WCS::colRowToRaDec(Coord2D pix  ///< Input (col, row)
                          ) const {
    Coord2D sky;
 
    colRowToRaDec(pix, sky);
   
    return sky;
}

/// Convert from (column, row) to (ra, dec) coordinates
/// \return The desired (ra, dec) position
Coord2D WCS::colRowToRaDec(double const col, ///< Input column position
                           double const row ///< Input row position
                          ) const {
    Coord2D pix(col, row);
    Coord2D sky;
 
    colRowToRaDec(pix, sky);
   
    return sky;
}
