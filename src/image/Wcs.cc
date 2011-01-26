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
 




#include <iostream>
#include <sstream>
#include <cmath>
#include <cstring>

#include "boost/format.hpp"

#include "wcslib/wcs.h"
#include "wcslib/wcsfix.h"
#include "wcslib/wcshdr.h"

#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/Wcs.h"

namespace except = lsst::pex::exceptions; 
namespace afwImg = lsst::afw::image;
namespace afwCoord = lsst::afw::coord;
namespace geom = lsst::afw::geom;


using namespace std;

typedef lsst::daf::base::PropertySet PropertySet;
typedef lsst::daf::base::PropertyList PropertyList;
typedef lsst::afw::image::Wcs Wcs;
typedef lsst::afw::geom::Point2D GeomPoint;
typedef lsst::afw::coord::Coord::Ptr CoordPtr;

//The amount of space allocated to strings in wcslib
const int STRLEN = 72;

const int lsstToFitsPixels = +1;
const int fitsToLsstPixels = -1;

//
// Constructors
//


///@brief Construct an invalid Wcs given no arguments
lsst::afw::image::Wcs::Wcs() :
    LsstBase(typeid(this)),
    _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
    _coordSystem(static_cast<afwCoord::CoordSystem>(-1)), _skyCoordsReversed(false) {
    _initWcs();    
}


///Create a Wcs from a fits header. Don't call this directly. Use makeWcs() instead, which will figure
///out which (if any) sub-class of Wcs is appropriate
Wcs::Wcs(PropertySet::Ptr const fitsMetadata):
                LsstBase(typeid(this)),
                _wcsInfo(NULL), 
                _nWcsInfo(0), 
                _relax(0), 
                _wcsfixCtrl(0), 
                _wcshdrCtrl(2),
                _nReject(0),
                _coordSystem(static_cast<afwCoord::CoordSystem>(-1)),
                _skyCoordsReversed(false)
{
    //Internal params for wcslib. These should be set via policy - but for the moment...
    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    initWcsLibFromFits(fitsMetadata);
    _initWcs();
}

/*
 * Set some internal variables that we need to refer to
 */
void Wcs::_initWcs()
{
    if (_wcsInfo) {
        _coordSystem = afwCoord::makeCoordEnum(_wcsInfo->radesys);
        //Check for strange images where the ctypes as swapped.
        char const *type = _wcsInfo->ctype[0];
        int const ncompare = 4;                       // we only care about type's first 4 chars
        _skyCoordsReversed = (strncmp(type, "DEC-", ncompare) == 0 ||
                              strncmp(type, "ELON", ncompare) == 0 ||
                              strncmp(type, "ELAT", ncompare) == 0 ||
                              strncmp(type, "GLON", ncompare) == 0 ||
                              strncmp(type, "GLAT", ncompare) == 0) ? true : false;
    }
}

///\brief Create a Wcs object with some known information.
///
///\param crval The sky position of the reference point
///\param crpix The pixel position corresponding to crval in LSST units
///\param CD    Matrix describing transformations from pixel to sky positions
///\param ctype1 Projection system used (see description of Wcs)
///\param ctype2 Projection system used (see description of Wcs)
///\param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
///\param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
///\param cunits1 Units of sky position. One of deg, arcmin or arcsec
///\param cunits2 Units of sky position. One of deg, arcmin or arcsec
///
///\note LSST units are zero indexed while FITs units are 1 indexed. So a value of crpix stored in a fits
///header of 127,127 corresponds to a pixel position in LSST units of 128, 128
Wcs::Wcs(const GeomPoint crval, const GeomPoint crpix, const Eigen::Matrix2d &CD, 
                 const std::string ctype1, const std::string ctype2,
                 double equinox, std::string raDecSys,
                 const std::string cunits1, const std::string cunits2
                ):
                 LsstBase(typeid(this)),
                 _wcsInfo(NULL), 
                 _nWcsInfo(0), 
                 _relax(1), 
                 _wcsfixCtrl(2), 
                 _wcshdrCtrl(2),
                 _nReject(0),
                 _coordSystem(static_cast<afwCoord::CoordSystem>(-1)),
                 _skyCoordsReversed(false)
{
    initWcsLib(crval, crpix, CD, 
               ctype1, ctype2,
               equinox, raDecSys,
               cunits1, cunits2);
    _initWcs();
}
               
    
///Parse a fits header, extract the relevant metadata and create a Wcs object
void Wcs::initWcsLibFromFits(PropertySet::Ptr const fitsMetadata){
    // Some headers (e.g. SDSS ones from FNAL) have EQUINOX as a string.  Fix this,
    // as wcslib 4.4.4 refuses to handle it
    {
        std::string const& key = "EQUINOX";
        if (fitsMetadata->exists(key) && fitsMetadata->typeOf(key) == typeid(std::string)) {
            double equinox = ::atof(fitsMetadata->getAsString(key).c_str());
            fitsMetadata->set(key, equinox);
        }
    }

    //Check header isn't empty
    std::string metadataStr = lsst::afw::formatters::formatFitsProperties(fitsMetadata);
    int nCards = lsst::afw::formatters::countFitsHeaderCards(fitsMetadata);
    if (nCards <= 0) {
        string msg = "Could not parse FITS WCS: no header cards found";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }

    //While the standard does not insist on CRVAL and CRPIX being present, it 
    //is almost certain their absence indicates a problem.   
    //Check for CRPIX
    if( !fitsMetadata->exists("CRPIX1") && !fitsMetadata->exists("CRPIX1a")) {
        string msg = "Neither CRPIX1 not CRPIX1a found";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }

    if( !fitsMetadata->exists("CRPIX2") && !fitsMetadata->exists("CRPIX2a")) {
        string msg = "Neither CRPIX2 not CRPIX2a found";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }

    //And the same for CRVAL
    if( !fitsMetadata->exists("CRVAL1") && !fitsMetadata->exists("CRVAL1a")) {
        string msg = "Neither CRVAL1 not CRVAL1a found";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }

    if( !fitsMetadata->exists("CRVAL2") && !fitsMetadata->exists("CRVAL2a")) {
        string msg = "Neither CRVAL2 not CRVAL2a found";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }


    //Pass the header into wcslib's formatter to extract setup the Wcs. First need
    //to convert to a C style string, so the compile doesn't complain about constness
    int len = metadataStr.size();
    char *hdrString = new char[len + 1];
    strncpy(hdrString, metadataStr.c_str(), len + 1);
    int pihStatus = wcspih(hdrString, nCards, _relax, _wcshdrCtrl, &_nReject, &_nWcsInfo, &_wcsInfo);
    delete[] hdrString;

    if (pihStatus != 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Could not parse FITS WCS: wcspih status = %d (%s)") %
                           pihStatus % wcs_errmsg[pihStatus]).str());
    }    

    //Run wcsfix on _wcsInfo to try and fix any problems it knows about.
    const int *naxes = NULL;            // should be {NAXIS1, NAXIS2, ...} to check cylindrical projections
    int stats[NWCSFIX];                 // status returns from wcsfix
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
    }
    
    //The Wcs standard requires a default value for RADESYS if the keyword
    //doesn't exist in header, but wcslib doesn't set it. So we do so here. This code 
    //conforms to Calbretta & Greisen 2002 \S 3.1
    if (!(fitsMetadata->exists("RADESYS") || fitsMetadata->exists("RADESYSa"))) {

        //If equinox exist and < 1984, use FK5. If >= 1984, use FK5
        if (fitsMetadata->exists("EQUINOX") || fitsMetadata->exists("EQUINOXa")) {
            std::string const EQUINOX = fitsMetadata->exists("EQUINOX") ? "EQUINOX" : "EQUINOXa";
            double const equinox = fitsMetadata->getAsDouble(EQUINOX);
            if(equinox < 1984) {
                snprintf(_wcsInfo->radesys, STRLEN, "FK4");
            } else {
                snprintf(_wcsInfo->radesys, STRLEN, "FK5");
            }
        } else {
            //If Equinox doesn't exist, default to ICRS
            snprintf(_wcsInfo->radesys, STRLEN, "ICRS");
        }
    }
}



///\brief Manually initialise a wcs struct using values passed by the constructor    
///\param crval The sky position of the reference point
///\param crpix The pixel position corresponding to crval in LSST units
///\param CD    Matrix describing transformations from pixel to sky positions
///\param ctype1 Projection system used (see description of Wcs)
///\param ctype2 Projection system used (see description of Wcs)
///\param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
///\param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
///\param cunits1 Units of sky position. One of deg, arcmin or arcsec
///\param cunits2 Units of sky position. One of deg, arcmin or arcsec
void Wcs::initWcsLib(GeomPoint const crval, GeomPoint const crpix, Eigen::Matrix2d const CD, 
                 const std::string ctype1, const std::string ctype2,
                 double equinox, std::string raDecSys,
                 const std::string cunits1, const std::string cunits2) {

    //Check CD is a valid size
    if( (CD.rows() != 2) || (CD.cols() != 2) ) {
        string msg = "CD is not a 2x2 matrix";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }


    //Check that cunits are legitimate values
    bool isValid = (cunits1 == "deg");
    isValid |= (cunits1 == "arcmin");
    isValid |= (cunits1 == "arcsec");
    isValid |= (cunits1 == "mas");
    
    if (!isValid) {
        string msg =  "CUNITS1 must be one of {deg|arcmin|arcsec|mas}";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }        

    isValid = (cunits2 == "deg");
    isValid |= (cunits2 == "arcmin");
    isValid |= (cunits2 == "arcsec");
    isValid |= (cunits2 == "mas");
    
    if (!isValid) {
        string msg =  "CUNITS2 must be one of {deg|arcmin|arcsec|mas}";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }        


    //Initialise the wcs struct
    _wcsInfo = static_cast<struct wcsprm *>(malloc(sizeof(struct wcsprm)));
    if (_wcsInfo == NULL) {
        throw LSST_EXCEPT(except::MemoryException, "Cannot allocate WCS info");
    }

    _wcsInfo->flag = -1;
    int status = wcsini(true, 2, _wcsInfo);   //2 indicates a naxis==2, a two dimensional image
    if(status != 0) {
        throw LSST_EXCEPT(except::MemoryException,
                          (boost::format("Failed to allocate memory with wcsini. Status %d: %s") %
                           status % wcs_errmsg[status] ).str());
    }
    
    
    //Set crval, crpix and CD. Internally to the class, we use fits units for consistency with
    //wcslib.
    _wcsInfo->crval[0] = crval.getX();
    _wcsInfo->crval[1] = crval.getY();
    _wcsInfo->crpix[0] = crpix.getX() + lsstToFitsPixels;
    _wcsInfo->crpix[1] = crpix.getY() + lsstToFitsPixels;

    //Set the CD matrix
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            _wcsInfo->cd[(2*i) + j] = CD(i,j);
        }
    }

    //Specify that we have a CD matrix, but no PC or CROTA
    _wcsInfo->altlin = 2;
    _wcsInfo->flag   = 0;   //values have been updated

    //This is a work around for what I think is a bug in wcslib. ->types is neither
    //initialised or set to NULL by default, so if I try to delete a Wcs object,
    //wcslib then attempts to free non-existent space, and the code can crash.
    _wcsInfo->types = NULL;

    //Set the coordinate system
    strncpy(_wcsInfo->ctype[0], ctype1.c_str(), STRLEN);
    strncpy(_wcsInfo->ctype[1], ctype2.c_str(), STRLEN);
    strncpy(_wcsInfo->radesys, raDecSys.c_str(), STRLEN);
    _wcsInfo->equinox = equinox;
    
    //Set the units
    strncpy(_wcsInfo->cunit[0], cunits1.c_str(), STRLEN);
    strncpy(_wcsInfo->cunit[1], cunits2.c_str(), STRLEN);
    
    _nWcsInfo = 1;   //Specify that we have only one coordinate representation

    //Tell wcslib that we are need to set up internal values
    status=wcsset(_wcsInfo);
    if(status != 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Failed to setup wcs structure with wcsset. Status %d: %s") %
                           status % wcs_errmsg[status] ).str());

    }
}


///Copy constructor
Wcs::Wcs(afwImg::Wcs const & rhs) : 
    LsstBase(typeid(this)),
    _wcsInfo(NULL), 
    _nWcsInfo(rhs._nWcsInfo), 
    _relax(rhs._relax), 
    _wcsfixCtrl(rhs._wcsfixCtrl), 
    _wcshdrCtrl(rhs._wcshdrCtrl),
    _nReject(rhs._nReject),
    _coordSystem(static_cast<afwCoord::CoordSystem>(-1)),
    _skyCoordsReversed(false)
{
    
    if (rhs._nWcsInfo > 0) {
        _wcsInfo = static_cast<struct wcsprm *>(calloc(rhs._nWcsInfo, sizeof(struct wcsprm)));
        if (_wcsInfo == NULL) {
            throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
        }

        _wcsInfo->flag = -1;
        int alloc = 1;                  //Unconditionally allocate memory when calling
        for (int i = 0; i != rhs._nWcsInfo; ++i) {
            int status = wcscopy(alloc, &rhs._wcsInfo[i], &_wcsInfo[i]);
            if (status != 0) {
                wcsvfree(&i, &_wcsInfo);
                throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException,
                                  (boost::format("Could not copy WCS: wcscopy status = %d : %s") %
                                   status % wcs_errmsg[status]).str());
            }
        }
    }
    _initWcs();
}
       

///Assignment operator    
Wcs::Wcs & Wcs::operator = (const Wcs & rhs){
    if (this != &rhs) {
        if (_nWcsInfo > 0) {
            wcsvfree(&_nWcsInfo, &_wcsInfo);
        }
        _nWcsInfo = 0;
        _wcsInfo = NULL;
        _relax = rhs._relax;
        _wcsfixCtrl = rhs._wcsfixCtrl;
        _wcshdrCtrl = rhs._wcshdrCtrl;
        _nReject = rhs._nReject;

        if (rhs._nWcsInfo > 0) {
            // allocate wcs structs
            _wcsInfo = static_cast<struct wcsprm *>(calloc(1, sizeof(struct wcsprm)));
            if (_wcsInfo == NULL) {
                throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
            }
            _wcsInfo->flag = -1;
            _nWcsInfo = 1;

            _wcsInfo[0].flag = -1;
            int status = wcscopy(1, rhs._wcsInfo, _wcsInfo);
            if (status != 0) {
                wcsvfree(&_nWcsInfo, &_wcsInfo);
                throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException,
                    (boost::format("Failed to copy WCS info; wcscopy status = %d. %s") %
                     status % wcs_errmsg[status]).str());
            }
        }
    }
    
    return *this;
}


Wcs::~Wcs() {
    if (_wcsInfo != NULL) {
        wcsvfree(&_nWcsInfo, &_wcsInfo);
    }
}
    

Wcs::Ptr Wcs::clone(void) const {
    return Wcs::Ptr(new Wcs(*this));
}


//
// Accessors
//

///Return crval. Note that this need not be the centre of the image
CoordPtr Wcs::getSkyOrigin() const {

    if(_wcsInfo != NULL) {
        return makeCorrectCoord(_wcsInfo->crval[0], _wcsInfo->crval[1]);
    } else {
        throw(LSST_EXCEPT(except::RuntimeErrorException, "Wcs structure is not initialised"));
    }
}

///Return crpix in the lsst convention. Note that this need not be the centre of the image
GeomPoint Wcs::getPixelOrigin() const {

    if(_wcsInfo != NULL) {
        //Convert from fits units back to lsst units
        double p1 = _wcsInfo->crpix[0] + fitsToLsstPixels;
        double p2 = _wcsInfo->crpix[1] + fitsToLsstPixels;
        return geom::Point2D(p1, p2);
    } else {
        throw(LSST_EXCEPT(except::RuntimeErrorException, "Wcs structure not initialised"));
    }
}


///Return the CD matrix
Eigen::Matrix2d Wcs::getCDMatrix() const {
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }
    
    int const naxis = _wcsInfo->naxis;

    //If naxis != 2, I'm not sure if any of what follows is correct
    assert(naxis == 2);
    
    Eigen::Matrix2d C;

    for (int i=0; i< naxis; ++i){
        for (int j=0; j<naxis; ++j) {
            C(i,j) = _wcsInfo->cd[ (i*naxis) + j ];
        }
    }

    return C;
}


///Return the Wcs as a fits header
PropertyList::Ptr Wcs::getFitsMetadata() const {
    return lsst::afw::formatters::WcsFormatter::generatePropertySet(*this);
}




///
/// Returns the orientation of the Wcs
///
/// The conventional sense for a Wcs image is to have North up and East to the left, or at least to be
/// able to rotate the image to that orientation. It is possible to create a "flipped" Wcs, where East
/// points right when the image is rotated such that North is up. Flipping a Wcs is akin to producing a mirror
/// image. This function tests whether the image is flipped or not.
///
/// It does so by calculating the determinant of the CD (i.e the rotation and scaling) matrix. If this
/// determinant is positive, then the image can be rotated to a position where increasing the right ascension
/// and declination increases the horizontal and vertical pixel position. In this case the image is flipped.
bool Wcs::isFlipped()  const{

    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(except::RuntimeErrorException, "Wcs structure is not initialised"));
    }

    double det = _wcsInfo->cd[0] * _wcsInfo->cd[3];
    det -= _wcsInfo->cd[1] * _wcsInfo->cd[2];
    
    if(det == 0){
        throw(LSST_EXCEPT(except::RuntimeErrorException, "Wcs scaling matrix is singular"));
    }

    if(det>0) {
        return true;
    }
    return false;
}


static double square(double x) {
    return x*x;
}

///Sky area covered by a pixel at position \c pix00. In units of square degrees.
double Wcs::pixArea(GeomPoint pix00     ///< The pixel point where the area is desired
                   ) const {
    //
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners of a square drawn in pixel
    // It'd be better to centre the square at sky00, but that would involve another conversion between sky and
    // pixel coordinates so I didn't bother
    //
    const double side = 1;             // length of the square's sides in pixels

    // Work in 3-space to avoid RA wrapping and pole issues.
    afwGeom::Point3D v0 = pixelToSky(pix00)->getVector();

    // Step by "side" in x and y pixel directions...
    GeomPoint px(pix00);
    GeomPoint py(pix00);
    px.shift(geom::Extent2D(side, 0));
    py.shift(geom::Extent2D(0, side));
    // Push the points through the WCS, and find difference in 3-space.
    afwGeom::Extent3D dx = pixelToSky(px)->getVector() - v0;
    afwGeom::Extent3D dy = pixelToSky(py)->getVector() - v0;

    // Compute |cross product| = area of parallelogram with sides dx,dy
    // FIXME -- this is slightly incorrect -- it's making the small-angle
    // approximation, taking the distance *through* the unit sphere
    // rather than over its surface.
    // This is in units of ~radians^2
    double area = sqrt(square(dx[1]*dy[2] - dx[2]*dy[1]) +
                       square(dx[2]*dy[0] - dx[0]*dy[2]) +
                       square(dx[0]*dy[1] - dx[1]*dy[0]));

    return area / square(side) * square(180./M_PI);
}

double Wcs::pixelScale() const {
    return 3600. * sqrt(pixArea(getPixelOrigin()));
}

/*
 * Worker routine for skyToPixel
 */
GeomPoint Wcs::skyToPixelImpl(double sky1, ///< Longitude coordinate; DEGREES
                              double sky2  ///< latitude  coordinate; DEGREES
                             ) const {
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    double const skyTmp[2] = { sky1, sky2 };
    double imgcrd[2];
    double phi, theta;
    double pixTmp[2];

    //Estimate pixel coordinates
    int stat[1];
    int status = 0;
    status = wcss2p(_wcsInfo, 1, 2, skyTmp, &phi, &theta, imgcrd, pixTmp, stat);
    if (status > 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of %d. %s") %
                           status % wcs_errmsg[status]).str());
    }

    // wcslib assumes 1-indexed coords
    return geom::Point2D(pixTmp[0] + lsst::afw::image::PixelZeroPos + fitsToLsstPixels,
                                    pixTmp[1] + lsst::afw::image::PixelZeroPos + fitsToLsstPixels); 
}

///\brief Convert from sky coordinates (e.g ra/dec) to pixel positions.
///
GeomPoint Wcs::skyToPixel(afwCoord::Coord::ConstPtr coord ///< The sky position
                         ) const {

    GeomPoint const sky = convertCoordToSky(coord);
    return skyToPixelImpl(sky[0], sky[1]);
}


///Given a Coord (as a shared pointer), return the sky position in the correct coordinate system
///for this Wcs. The first element of the pair is the coordinate value corresponding to ctype1
///and the second element corresponds to ctype2.
GeomPoint Wcs::convertCoordToSky(lsst::afw::coord::Coord::ConstPtr coord) const {
    //Construct a coord object of the correct type
    CONST_PTR(afwCoord::Coord) convertedCoord = coord->convert(_coordSystem);

    if (_skyCoordsReversed) {
        return geom::Point2D(convertedCoord->getLatitude(afwCoord::DEGREES),
                                convertedCoord->getLongitude(afwCoord::DEGREES));
    } else {    
        return geom::Point2D(convertedCoord->getLongitude(afwCoord::DEGREES),
                                convertedCoord->getLatitude(afwCoord::DEGREES));
    }
}

///\brief Convert from sky coordinates (e.g ra/dec) to pixel positions.
///
///Convert a sky position (e.g ra/dec) to a pixel position. The exact meaning of sky1, sky2 
///and the return value depend on the properties of the wcs (i.e the values of CTYPE1 and
///CTYPE2), but the inputs are usually ra/dec. The outputs are x and y pixel position.

GeomPoint Wcs::skyToPixel(double sky1, double sky2) const {
    return _skyCoordsReversed ?
        skyToPixelImpl(sky2, sky1) :
        skyToPixelImpl(sky1, sky2);
}

///\brief Convert from sky coordinates (e.g ra/dec) to intermediate world coordinates
///
GeomPoint Wcs::skyToIntermediateWorldCoord(lsst::afw::coord::Coord::ConstPtr coord) const {
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    GeomPoint const sky = convertCoordToSky(coord);

    double const skyTmp[2] = { sky[0], sky[1]};
    double imgcrd[2];
    double phi, theta;
    double pixTmp[2];

    //Estimate pixel coordinates
    int stat[1];
    int status = 0;
    status = wcss2p(_wcsInfo, 1, 2, skyTmp, &phi, &theta, imgcrd, pixTmp, stat);
    if (status > 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of %d. %s") %
                           status % wcs_errmsg[status]).str());
    }

    // wcslib assumes 1-indexed coords
    return geom::Point2D(imgcrd[0], imgcrd[1]); 
}

/*
 * Worker routine for pixelToSky
 */
void
Wcs::pixelToSkyImpl(double pixel1, double pixel2, double skyTmp[2]) const
{
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }
    
    // wcslib assumes 1-indexed coordinates
    double pixTmp[2] = { pixel1 - lsst::afw::image::PixelZeroPos + lsstToFitsPixels,
                         pixel2 - lsst::afw::image::PixelZeroPos + lsstToFitsPixels}; 
    double imgcrd[2];
    double phi, theta;
    
    int status = 0;
    status = wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, &phi, &theta, skyTmp, &status);
    if (status > 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of %d. %s") %
                           status % wcs_errmsg[status]).str());
    }
}

///\brief Convert from pixel position to sky coordinates (e.g ra/dec)
///
///Convert a pixel position (e.g x,y) to a celestial coordinate (e.g ra/dec). The output coordinate
///system depends on the values of CTYPE used to construct the object. For ra/dec, the CTYPES should
///be RA--TAN and DEC-TAN. 
CoordPtr Wcs::pixelToSky(const GeomPoint pixel) const {
    return pixelToSky(pixel.getX(), pixel.getY());
}

///\brief Convert from pixel position to sky coordinates (e.g ra/dec)
///
///Convert a pixel position (e.g x,y) to a celestial coordinate (e.g ra/dec). The output coordinate
///system depends on the values of CTYPE used to construct the object. For ra/dec, the CTYPES should
///be RA--TAN and DEC-TAN. 
CoordPtr Wcs::pixelToSky(double pixel1, double pixel2) const {
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    double skyTmp[2];
    pixelToSkyImpl(pixel1, pixel2, skyTmp);

    return makeCorrectCoord(skyTmp[0], skyTmp[1]);
}

///\brief Convert from pixel position to sky coordinates (e.g ra/dec)
///
///Convert a pixel position (e.g x,y) to a celestial coordinate (e.g ra/dec)
///
/// \note This routine is designed for the knowledgeable user in need of performance; it's safer to call
/// the version that returns a CoordPtr
///
afwGeom::Point2D Wcs::pixelToSky(double pixel1, double pixel2, bool) const {
    double skyTmp[2];
    pixelToSkyImpl(pixel1, pixel2, skyTmp);

    return afwGeom::Point2D(skyTmp[0], skyTmp[1]);
}

///\brief Given a sky position, use the values stored in ctype and radesys to return the correct
///sub-class of Coord
CoordPtr Wcs::makeCorrectCoord(double sky0, double sky1) const {

    //Construct a coord object of the correct type
    int const ncompare = 4;                       // we only care about type's first 4 chars
    char *type = _wcsInfo->ctype[0];
    char *radesys = _wcsInfo->radesys;
    double equinox = _wcsInfo->equinox;

    if (strncmp(type, "RA--", ncompare) == 0) { // Our default.  If it's often something else, consider
        ;                                       // using an tr1::unordered_map
        if(strcmp(radesys, "ICRS") == 0) {
            return afwCoord::makeCoord(afwCoord::ICRS, sky0, sky1);
        }
        if(strcmp(radesys, "FK5") == 0) {
            return afwCoord::makeCoord(afwCoord::FK5, sky0, sky1, equinox);
        } else {   
            throw LSST_EXCEPT(except::RuntimeErrorException,
                              (boost::format("Can't create Coord object: Unrecognised radesys %s") %
                               radesys).str());
        }

    } else if (strncmp(type, "GLON", ncompare) == 0) {
        return afwCoord::makeCoord(afwCoord::GALACTIC, sky0, sky1);   
    } else if (strncmp(type, "ELON", ncompare) == 0) {
        return afwCoord::makeCoord(afwCoord::ECLIPTIC, sky0, sky1, equinox);
    } else if (strncmp(type, "DEC-", ncompare) == 0) {
        //check for the case where the ctypes are swapped. Note how sky0 and sky1 are swapped as well

        //Our default
        if(strcmp(radesys, "ICRS") == 0) {
            return afwCoord::makeCoord(afwCoord::ICRS, sky1, sky0);
        }
        if(strcmp(radesys, "FK5") == 0) {
            return afwCoord::makeCoord(afwCoord::FK5, sky1, sky0, equinox);
        } else {   
            throw LSST_EXCEPT(except::RuntimeErrorException,
                              (boost::format("Can't create Coord object: Unrecognised radesys %s") %
                               radesys).str());
        }
    } else if (strncmp(type, "GLAT", ncompare) == 0) {
        return afwCoord::makeCoord(afwCoord::GALACTIC, sky1, sky0);   
    } else if (strncmp(type, "ELAT", ncompare) == 0) {
        return afwCoord::makeCoord(afwCoord::ECLIPTIC, sky1, sky0, equinox);
    } else {
    //Give up in disgust
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Can't create Coord object: Unrecognised sys %s") %
                           type).str());
    }
    
    //Can't get here
    assert(0);
}


/**
 * Return the local linear approximation to Wcs::pixelToSky at the given point (in sky coordinates).
 *
 * The local linear approximation is defined such the following is true (ignoring floating-point errors):
 * @code
 * wcs.linearizePixelToSky(sky, skyUnit)(wcs.skyToPixel(sky)) == sky.getPosition(skyUnit);
 * @endcode
 * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
 *
 * This is currently implemented as a numerical derivative, but we should specialise the Wcs class (or rather
 * its implementation) to handle "simple" cases such as TAN-SIP analytically
 *
 * @param (in) coord   Position in sky coordinates where transform is desired.
 * @param (in) skyUnit Units to use for sky coordinates; units of matrix elements will be skyUnits/pixel.
 */
lsst::afw::geom::AffineTransform Wcs::linearizePixelToSky(
    lsst::afw::coord::Coord::ConstPtr const & coord,
    lsst::afw::coord::CoordUnit skyUnit
) const {
    return linearizePixelToSkyInternal(skyToPixel(coord), coord, skyUnit);
}

/**
 * Return the local linear approximation to Wcs::pixelToSky at the given point (in pixel coordinates).
 *
 * The local linear approximation is defined such the following is true (ignoring floating-point errors):
 * @code
 * wcs.linearizePixelToSky(pix, skyUnit)(pix) == wcs.pixelToSky(pix).getPosition(skyUnit)
 * @endcode
 * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
 *
 * This is currently implemented as a numerical derivative, but we should specialise the Wcs class (or rather
 * its implementation) to handle "simple" cases such as TAN-SIP analytically
 *
 * @param (in) pix     Position in pixel coordinates where transform is desired.
 * @param (in) skyUnit Units to use for sky coordinates; units of matrix elements will be skyUnits/pixel.
 */
lsst::afw::geom::AffineTransform Wcs::linearizePixelToSky(
    lsst::afw::geom::Point2D const & pix,
    lsst::afw::coord::CoordUnit skyUnit
) const {
    return linearizePixelToSkyInternal(pix, pixelToSky(pix), skyUnit);
}

/**
 * Implementation for the overloaded public linearizePixelToSky methods, requiring both a pixel coordinate
 * and the corresponding sky coordinate.
 */
lsst::afw::geom::AffineTransform Wcs::linearizePixelToSkyInternal(
    lsst::afw::geom::Point2D const & pix00,
    lsst::afw::coord::Coord::ConstPtr const & coord,
    lsst::afw::coord::CoordUnit skyUnit
) const {
    //
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners of a square drawn in pixel
    // It'd be better to centre the square at sky00, but that would involve another conversion between sky and
    // pixel coordinates so I didn't bother
    //
    const double side = 10;             // length of the square's sides in pixels
    GeomPoint const sky00 = coord->getPosition(skyUnit);

    GeomPoint const dsky10 = pixelToSky(pix00 + geom::Extent2D(side, 0))->getPosition(skyUnit) -
        geom::Extent<double>(sky00);
    GeomPoint const dsky01 = pixelToSky(pix00 + geom::Extent2D(0, side))->getPosition(skyUnit) -
        geom::Extent<double>(sky00);
    
    Eigen::Matrix2d m;
    m(0, 0) = dsky10.getX()/side;
    m(0, 1) = dsky01.getX()/side;
    m(1, 0) = dsky10.getY()/side;
    m(1, 1) = dsky01.getY()/side;

    Eigen::Vector2d sky00v;
    sky00v << sky00.getX(), sky00.getY();
    Eigen::Vector2d pix00v;
    pix00v << pix00.getX(), pix00.getY();
    //return lsst::afw::geom::AffineTransform(m, lsst::afw::geom::Extent2D(sky00v - m * pix00v));
    return lsst::afw::geom::AffineTransform(m, (sky00v - m * pix00v));
}

/**
 * Return the local linear approximation to Wcs::skyToPixel at the given point (in sky coordinates).
 *
 *
 * The local linear approximation is defined such the following is true (ignoring floating-point errors):
 * @code
 * wcs.linearizeSkyToPixel(sky, skyUnit)(sky.getPosition(skyUnit)) == wcs.skyToPixel(sky)
 * @endcode
 * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
 *
 * This is currently implemented as a numerical derivative, but we should specialise the Wcs class (or rather
 * its implementation) to handle "simple" cases such as TAN-SIP analytically
 *
 * @param (in) coord   Position in sky coordinates where transform is desired.
 * @param (in) skyUnit Units to use for sky coordinates; units of matrix elements will be pixels/skyUnit.
 */
lsst::afw::geom::AffineTransform Wcs::linearizeSkyToPixel(
    lsst::afw::coord::Coord::ConstPtr const & coord,
    lsst::afw::coord::CoordUnit skyUnit
) const {
    return linearizeSkyToPixelInternal(skyToPixel(coord), coord, skyUnit);
}

/**
 * Return the local linear approximation to Wcs::skyToPixel at the given point (in pixel coordinates).
 *
 * The local linear approximation is defined such the following is true (ignoring floating-point errors):
 * @code
 * wcs.linearizeSkyToPixel(pix, skyUnit)(wcs.pixelToSky(pix).getPosition(skyUnit)) == pix
 * @endcode
 * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
 *
 * This is currently implemented as a numerical derivative, but we should specialise the Wcs class (or rather
 * its implementation) to handle "simple" cases such as TAN-SIP analytically
 *
 * @param (in) pix     Position in pixel coordinates where transform is desired.
 * @param (in) skyUnit Units to use for sky coordinates; units of matrix elements will be pixels/skyUnit.
 */
lsst::afw::geom::AffineTransform Wcs::linearizeSkyToPixel(
    lsst::afw::geom::Point2D const & pix,
    lsst::afw::coord::CoordUnit skyUnit
) const {
    return linearizeSkyToPixelInternal(pix, pixelToSky(pix), skyUnit);
}

/**
 * Implementation for the overloaded public linearizeSkyToPixel methods, requiring both a pixel coordinate
 * and the corresponding sky coordinate.
 */
lsst::afw::geom::AffineTransform Wcs::linearizeSkyToPixelInternal(
    lsst::afw::geom::Point2D const & pix00,
    lsst::afw::coord::Coord::ConstPtr const & coord,
    lsst::afw::coord::CoordUnit skyUnit
) const {
    lsst::afw::geom::AffineTransform inverse = linearizePixelToSkyInternal(pix00, coord, skyUnit);
    return inverse.invert();
}



/**
 * Return the linear part of the Wcs, the CD matrix in FITS speak, as an AffineTransform
 *
 * \sa 
 */
lsst::afw::geom::LinearTransform Wcs::getLinearTransform() const
{
    return lsst::afw::geom::LinearTransform(getCDMatrix());
}





//
//Mutators
//


/// \brief Move the pixel reference position by (dx, dy)
///Used when persisting and retrieving sub-images. The lsst convention is that Wcs returns pixel position
///(which is based on position in the parent image), but the fits convention is to return pixel index
///(which is bases on position in the sub-image). In order that the fits files we create make sense
///to other fits viewers, we change to the fits convention when writing out images.
void Wcs::shiftReferencePixel(double dx, double dy) {
    //If the _wcsInfo structure hasn't been initialised yet, then there's nothing to do
    if(_wcsInfo != NULL) {
        _wcsInfo->crpix[0] += dx;
        _wcsInfo->crpix[1] += dy;
    }
}

/************************************************************************************************************/
/*
 * Now WCSA, pixel coordinates, but allowing for X0 and Y0
 */
namespace lsst {
namespace afw {
namespace image {

/*
 * A Wcs object used to indicate a default argument
 */
Wcs NoWcs;

namespace {
    struct InitWcs {
        InitWcs() {
            NoWcs.markPersistent();
        }
    };

    InitWcs initWcs;                    // Call the ctor to label NoWcs as a persistent object
}
    
namespace detail {

/**
 * Define a trivial WCS that maps the lower left corner (LLC) pixel of an image to a given value
 */
lsst::daf::base::PropertyList::Ptr
createTrivialWcsAsPropertySet(std::string const& wcsName, ///< Name of desired WCS
                              int const x0,               ///< Column coordinate of LLC pixel
                              int const y0                ///< Row coordinate of LLC pixel
                             ) {
    lsst::daf::base::PropertyList::Ptr wcsMetaData(new lsst::daf::base::PropertyList);

    wcsMetaData->set("CRVAL1" + wcsName, x0, "Column pixel of Reference Pixel");
    wcsMetaData->set("CRVAL2" + wcsName, y0, "Row pixel of Reference Pixel");
    wcsMetaData->set("CRPIX1" + wcsName, 1, "Column Pixel Coordinate of Reference");
    wcsMetaData->set("CRPIX2" + wcsName, 1, "Row Pixel Coordinate of Reference");
    wcsMetaData->set("CTYPE1" + wcsName, "LINEAR", "Type of projection");
    wcsMetaData->set("CTYPE2" + wcsName, "LINEAR", "Type of projection");
    wcsMetaData->set("CUNIT1" + wcsName, "PIXEL", "Column unit");
    wcsMetaData->set("CUNIT2" + wcsName, "PIXEL", "Row unit");

    return wcsMetaData;
}
/**
 * Return a Point2I(x0, y0) given a PropertySet containing a suitable WCS (e.g. "A")
 *
 * The WCS must have CRPIX[12] == 1 and CRVAL[12] must be present.  If this is true, the WCS
 * cards are removed from the metadata
 */
image::Point2I getImageXY0FromMetadata(std::string const& wcsName,            ///< the WCS to search (E.g. "A")
                                      lsst::daf::base::PropertySet *metadata ///< the metadata, maybe containing the WCS
                                     ) {
        
    int x0 = 0;                         // Our value of X0
    int y0 = 0;                         // Our value of Y0
    
    try {
        //
        // Only use WCS if CRPIX[12] == 1 and CRVAL[12] is present
        //
        if (metadata->getAsDouble("CRPIX1" + wcsName) == 1 &&
            metadata->getAsDouble("CRPIX2" + wcsName) == 1) {
            
            x0 = metadata->getAsInt("CRVAL1" + wcsName);
            y0 = metadata->getAsInt("CRVAL2" + wcsName);
            //
            // OK, we've got it.  Remove it from the header
            //
            metadata->remove("CRVAL1" + wcsName);
            metadata->remove("CRVAL2" + wcsName);
            metadata->remove("CRPIX1" + wcsName);
            metadata->remove("CRPIX2" + wcsName);
            metadata->remove("CTYPE1" + wcsName);
            metadata->remove("CTYPE1" + wcsName);
            metadata->remove("CUNIT1" + wcsName);
            metadata->remove("CUNIT2" + wcsName);
        }
    } catch(lsst::pex::exceptions::NotFoundException &) {
        ;                               // OK, not present
    }

    return image::Point2I(x0, y0);
}

/**
 * Strip keywords from the input metadata that are related to the generated Wcs
 *
 * It isn't entirely obvious that this is enough --- e.g. if the input metadata has deprecated
 * WCS keywords such as CDELT[12] they won't be stripped.  Well, actually we catch CDELT[12],
 * but there may be others
 */
int stripWcsKeywords(PTR(lsst::daf::base::PropertySet) metadata, ///< Metadata to be stripped
                     CONST_PTR(Wcs) wcs                          ///< A Wcs with (implied) keywords
                    )
{
    PTR(lsst::daf::base::PropertySet) wcsMetadata = wcs->getFitsMetadata();
    std::vector<std::string> paramNames = wcsMetadata->paramNames();
    paramNames.push_back("CDELT1");
    paramNames.push_back("CDELT2");
    for (std::vector<std::string>::const_iterator ptr = paramNames.begin(); ptr != paramNames.end(); ++ptr) {
        metadata->remove(*ptr);
    }

    return 0;                           // would be ncard if remove returned a status
}

}}}}
