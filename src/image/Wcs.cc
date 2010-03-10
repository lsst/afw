// -*- lsst-c++ -*-




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
namespace geom = lsst::afw::geom;


using namespace std;

typedef lsst::daf::base::PropertySet PropertySet;
typedef lsst::afw::image::Wcs Wcs;
typedef lsst::afw::geom::PointD GeomPoint;


//The amount of space allocated to strings in wcslib
const int STRLEN = 72;

//
// Constructors
//


///@brief Construct an invalid Wcs given no arguments
lsst::afw::image::Wcs::Wcs() :
    LsstBase(typeid(this)),
    _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0) {
    
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
                _nReject(0) {

    //Internal params for wcslib. These should be set via policy - but for the moment...
    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    initWcsLibFromFits(fitsMetadata);
}


///Create a Wcs object with some known information.
///\param crval The sky position of the reference point
///\param crpix The pixel position corresponding to crval
///\param CD    Matrix describing transformations from pixel to sky positions
///\param ctype1 Projection system used (see description of Wcs)
///\param ctype2 Projection system used (see description of Wcs)
///\param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
///\param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
///\param cunits1 Units of sky position. One of deg, arcmin or arcsec
///\param cunits2 Units of sky position. One of deg, arcmin or arcsec
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
                 _nReject(0) {
    
    initWcsLib(crval, crpix, CD, 
               ctype1, ctype2,
               equinox, raDecSys,
               cunits1, cunits2);
}
               
    
///Parse a fits header, extract the relevant metadata and create a Wcs object
void Wcs::initWcsLibFromFits(PropertySet::Ptr const fitsMetadata){

    //Check header isn't empty
    std::string metadataStr = lsst::afw::formatters::formatFitsProperties(fitsMetadata);
    int nCards = lsst::afw::formatters::countFitsHeaderCards(fitsMetadata);
    if (nCards <= 0) {
        string msg = "Could not parse FITS WCS: no header cards found";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }

    //Check that only one WCS is in the header. I may want to generalise to multiple wcs'es
    //at a  later date, but that's more work
    if( fitsMetadata->exists("WCSAXES") && fitsMetadata->getAsInt("WCSAXES") != 1) {
        string msg = "Can't treat more than one WCSAXIS yet";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }
    
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
    char *hdrString = (char *) malloc((len + 1)*sizeof(char));
    strncpy(hdrString, metadataStr.c_str(), len + 1);
    int pihStatus = wcspih(hdrString, nCards, _relax, _wcshdrCtrl, &_nReject, &_nWcsInfo, &_wcsInfo);

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

}



///Manually initialise a wcs struct using values passed by the constructor    
///\param crval The sky position of the reference point
///\param crpix The pixel position corresponding to crval
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
    
    
    //Set crval, crpix and CD
    _wcsInfo->crval[0] = crval.getX();
    _wcsInfo->crval[1] = crval.getY();
    _wcsInfo->crpix[0] = crpix.getX();
    _wcsInfo->crpix[1] = crpix.getY();

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
    _nReject(rhs._nReject) {
    
    if( rhs._nWcsInfo > 0) {
        _wcsInfo = static_cast<struct wcsprm *>(malloc(sizeof(struct wcsprm)));
        if (_wcsInfo == NULL) {
            throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
        }

        _wcsInfo->flag = -1;
        int alloc=1;    //Unconditionally allocate memory when calling
        int status = wcscopy(alloc, rhs._wcsInfo, _wcsInfo);
        if(status != 0) {
            wcsvfree(&_nWcsInfo, &_wcsInfo);
            throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException,
                (boost::format("Could not copy WCS: wcscopy status = %d : %s") %
                 status % wcs_errmsg[status]).str());
        }
    }
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
    


//
// Accessors
//

///Return crval. Note that this need not be the centre of the image
GeomPoint Wcs::getSkyOrigin() const {

    if(_wcsInfo != NULL) {
        return geom::makePointD(_wcsInfo->crval[0], _wcsInfo->crval[1]);
    } else {
        throw(LSST_EXCEPT(except::RuntimeErrorException, "Wcs structure is not initialised"));
    }
}

///Return crpix. Note that this need not be the centre of the image
GeomPoint Wcs::getPixelOrigin() const {

    if(_wcsInfo != NULL) {
        return geom::makePointD(_wcsInfo->crpix[0], _wcsInfo->crpix[1]);
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
PropertySet::Ptr Wcs::getFitsMetadata() const {
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


///Sky area covered by a pixel at position \param pix00. In units of cunits.
double Wcs::pixArea(GeomPoint pix00) const {
    //
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners of a square drawn in pixel
    // It'd be better to centre the square at sky00, but that would involve another conversion between sky and
    // pixel coordinates so I didn't bother
    //
    const double side = 10;             // length of the square's sides in pixels

    GeomPoint const sky00 = pixelToSky(pix00);
    
    //This doesn't necessarily make sense if you're not familiar with Points and Extents. In the 
    //context of afw::geom, a Point is a position on the sky, and it doesn't make sense to add or subtract
    //them. Instead you can add a vector (or an "Extent") to a point. Extents can be constructed from 
    //points, or from a pair of numbers using the makeExtent syntax
    GeomPoint const dsky10 = pixelToSky(pix00 + geom::makeExtentD(side, 0)) - geom::Extent<double>(sky00);
    GeomPoint const dsky01 = pixelToSky(pix00 + geom::makeExtentD(0, side)) - geom::Extent<double>(sky00);

    double const cosDec = cos(sky00.getY()*M_PI/180.0);
    return cosDec*fabs(dsky01.getX()*dsky10.getY() - dsky01.getY()*dsky10.getX())/(side*side);
}



///Convert a sky position (e.g ra/dec) to a pixel position. The exact meaning of sky1, sky2 
///depend on the properties of the wcs (i.e the values of CTYPE1 and
///CTYPE2), but the inputs are usually ra/dec, and the outputs are x and y pixel position.
GeomPoint Wcs::skyToPixel(const GeomPoint sky) const {
    return skyToPixel(sky.getX(), sky.getY());
}

///Convert a pixel position (e.g x,y) to a celestial coordinate (e.g ra/dec). The output coordinate
///system depends on the values of CTYPE used to construct the object. For ra/dec, the CTYPES should
///be RA--TAN and DEC-TAN. 
GeomPoint Wcs::pixelToSky(const GeomPoint pixel) const {
    return pixelToSky(pixel.getX(), pixel.getY());
}


///Convert a sky position (e.g ra/dec) to a pixel position. The exact meaning of sky1, sky2 
///and the return value depend on the properties of the wcs (i.e the values of CTYPE1 and
///CTYPE2), but the inputs are usually ra/dec, and the outputs are x and y pixel position.
GeomPoint Wcs::skyToPixel(double sky1, double sky2) const {
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
    return geom::makePointD(pixTmp[0] + lsst::afw::image::PixelZeroPos - 1,
                                    pixTmp[1] + lsst::afw::image::PixelZeroPos - 1); 
}


///Convert a pixel position (e.g x,y) to a celestial coordinate (e.g ra/dec). The output coordinate
///system depends on the values of CTYPE used to construct the object. For ra/dec, the CTYPES should
///be RA--TAN and DEC-TAN. 
GeomPoint Wcs::pixelToSky(double pixel1, double pixel2) const {
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    // wcslib assumes 1-indexed coordinates
    double pixTmp[2] = { pixel1 - lsst::afw::image::PixelZeroPos + 1,
                               pixel2 - lsst::afw::image::PixelZeroPos + 1}; 
    double imgcrd[2];
    double phi, theta;
    double skyTmp[2];

    
    int stat[1];
    int status = 0;
    status = wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, &phi, &theta, skyTmp, stat);
    if (status > 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of %d. %s") %
                           status % wcs_errmsg[status]).str());
    }

    return geom::makePointD(skyTmp[0], skyTmp[1]);
}


///Stop-gap functions. Convert sky positions in radians to pixels.
///Once the Coord class is implemented this should go away
GeomPoint Wcs::skyRadiansToPixel(double sky1Radians, double sky2Radians) const {
    double sky1Deg = sky1Radians*180./M_PI;
    double sky2Deg = sky2Radians*180./M_PI;
    
    return skyToPixel(sky1Deg, sky2Deg);
}

///Stop-gap functions. Convert pixel positions to sky positions in radians (not degrees as normal)
///Once the Coord class is implemented this should go away
GeomPoint Wcs::pixelToSkyRadians(double pixel1, double pixel2) const {
    //In degrees
    GeomPoint sky = pixelToSky(pixel1, pixel2);
    
    //convert to radians
    sky[0] *= M_PI/180.0;
    sky[1] *= M_PI/180.0;
    return sky;
}


/**
 * Return the linear part of the Wcs, the CD matrix in FITS speak, as an AffineTransform
 *
 * \sa 
 */
lsst::afw::geom::AffineTransform Wcs::getAffineTransform() const
{
    return lsst::afw::geom::AffineTransform(getCDMatrix());
}


/**
 * Return the local linear approximation to Wcs::xyToRaDec at the point (ra, y) = sky
 *
 * This is currently implemented as a numerical derivative, but we should specialise the Wcs class (or rather
 * its implementation) to handle "simple" cases such as TAN-SIP analytically
 *
 * @param(in) sky Position in sky coordinates where transform is desired
 */
lsst::afw::geom::AffineTransform Wcs::linearizeAt(GeomPoint const & sky) const
{
    //
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners of a square drawn in pixel
    // It'd be better to centre the square at sky00, but that would involve another conversion between sky and
    // pixel coordinates so I didn't bother
    //
    const double side = 10;             // length of the square's sides in pixels
    GeomPoint const sky00 = sky;
    GeomPoint const pix00 = skyToPixel(sky00);

    //Adding vectors to points is tricky. See note in Wcs::pixArea()
    GeomPoint const dsky10 = pixelToSky(pix00 + geom::makeExtentD(side, 0)) - geom::Extent<double>(sky00);
    GeomPoint const dsky01 = pixelToSky(pix00 + geom::makeExtentD(0, side)) - geom::Extent<double>(sky00);
    
    Eigen::Matrix2d m;
    m(0, 0) = dsky10.getX()/side;
    m(0, 1) = dsky01.getX()/side;
    m(1, 0) = dsky10.getY()/side;
    m(1, 1) = dsky01.getY()/side;

    Eigen::Vector2d sky00v;
    sky00v << sky00.getX(), sky00.getY();
    Eigen::Vector2d pix00v;
    pix00v << pix00.getX(), pix00.getY();
    return lsst::afw::geom::AffineTransform(m, lsst::afw::geom::ExtentD(sky00v - m * pix00v));
}


//
//Mutators
//


///Used when creating sub images
/// Move the pixel reference position by (dx, dy)
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
namespace detail {

/**
 * Define a trivial WCS that maps the lower left corner (LLC) pixel of an image to a given value
 */
lsst::daf::base::PropertySet::Ptr
createTrivialWcsAsPropertySet(std::string const& wcsName, ///< Name of desired WCS
                              int const x0,               ///< Column coordinate of LLC pixel
                              int const y0                ///< Row coordinate of LLC pixel
                             ) {
    lsst::daf::base::PropertySet::Ptr wcsMetaData(new lsst::daf::base::PropertySet);

    wcsMetaData->set("CRVAL1" + wcsName, x0); // (output) Column pixel of Reference Pixel
    wcsMetaData->set("CRVAL2" + wcsName, y0); // (output) Row pixel of Reference Pixel
    wcsMetaData->set("CRPIX1" + wcsName, 1);  //  Column Pixel Coordinate of Reference
    wcsMetaData->set("CRPIX2" + wcsName, 1);  //  Row Pixel Coordinate of Reference
    wcsMetaData->set("CTYPE1" + wcsName, "LINEAR"); // Type of projection
    wcsMetaData->set("CTYPE1" + wcsName, "LINEAR"); // Type of projection
    wcsMetaData->set("CUNIT1" + wcsName, "PIXEL");  // Column unit
    wcsMetaData->set("CUNIT2" + wcsName, "PIXEL");  // Row unit

    return wcsMetaData;
}
/**
 * Return a PointI(x0, y0) given a PropertySet containing a suitable WCS (e.g. "A")
 *
 * The WCS must have CRPIX[12] == 1 and CRVAL[12] must be present.  If this is true, the WCS
 * cards are removed from the metadata
 */
image::PointI getImageXY0FromMetadata(std::string const& wcsName,            ///< the WCS to search (E.g. "A")
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

    return image::PointI(x0, y0);
}

}}}}
