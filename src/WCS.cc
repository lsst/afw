//! \file
//! \brief Implementation of WCS
#include "lsst/fw/WCS.h"

#include "wcslib/wcs.h"
#include "wcslib/wcshdr.h"

using namespace lsst::fw;
using namespace vw::math;

/// Constructor for WCS
WCS::WCS(DataPropertyPtrT fitsMetaData) : LsstBase(typeid(this)) {
    int nCards = 0;

    // these should be set via policy - but for the moment...

    _relax = 1;
    _ctrl = 2;

    std::ostringstream fitsMetaDataStr;
    fitsMetaData->reprCfitsio(fitsMetaDataStr, &nCards, false);
     
    // idiocy required because wcspih takes a non-const char*
    int len = fitsMetaDataStr.str().size();
    char *hdrString = new(char[len+1]);
    strcpy(hdrString, fitsMetaDataStr.str().c_str());

    _status = wcspih(hdrString, nCards, _relax, _ctrl, &_nReject, &_nWcsInfo, &_wcsInfo);

    // What do we do if _status != 0?

    delete hdrString;
}

/// Destructor for WCS
WCS::~WCS() {
    if (_wcsInfo != NULL) {
        _status = wcsvfree(&_nWcsInfo, &_wcsInfo);
        assert(_status == 0);               // true unless _wcsInfo is NULL
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
