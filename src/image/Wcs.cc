// -*- lsst-c++ -*-
/**
 * @file
 * @brief Implementation of Wcs, wrapping wcslib and treating the effect of distortion
 *
 * Image distortion is treated using the Simple Imaging Polynomial (SIP) convention.
 * This convention is described in Shupe et al. (2005) (Astronomical Data Analysis Software and Systems
 * XIV, Asp Conf. Series Vol XXX, Ed: Shopbell et al.), and descibed in some more detail in
 * http://web.ipac.caltech.edu/staff/fmasci/home/wise/codeVdist.html
 *
 * To convert from pixel coordintates to radec ("intermediate world coordinates"), first use the matrices
 * _sipA and _sipB to calculate undistorted coorinates (i.e where on the chip the image would lie if
 * the optics gave undistorted images), then pass these undistorted coorinates wcsp2s() to calculate radec.
 *
 * For the reverse, radec -> pixels, convert the radec to undistorted coords, and then use the _sipAp and
 * _sipBp matrices to add in the distortion terms.
 *
 *
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

using lsst::daf::base::PropertySet;
using lsst::daf::data::LsstBase;
using namespace std;

typedef lsst::afw::image::PointD PointD;

//The amount of space allocated to strings in wcslib
const int STRLEN = 72;

//Error codes
static std::string wcslibErrorMsg(int status)
{
    int numCodes = 13;
    
    if(status < 0)
    {   return std::string("Illegal status code in wcslib");
    }
    
    if(status > numCodes)
    {   return std::string("Unknown error code");
    }
    
    return wcs_errmsg[status];
}

/**
 * @brief Construct an invalid Wcs given no arguments
 *
 * @throw lsst::pex::exceptions::RuntimeErrorException on error
 */
lsst::afw::image::Wcs::Wcs() :
    LsstBase(typeid(this)),
    _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
    _sipA(1,1), _sipB(1,1), _sipAp(1,1), _sipBp(1,1){
}


///
/// Function to initialise the wcslib structure. Should only be called from Wcs constructors
void lsst::afw::image::Wcs::initWcslib(
                PointD crval,                   ///< Origin in sky coords
                PointD crpix,                   ///< Origin in pixel coords
                Eigen::Matrix2d CD,             ///< Transformation matrix
                double equinox,                 ///< Equinox of coordinate system, eg 2000
                std::string raDecSys,           ////< Definition of coordinate system, eg. FK5 or ICRS
                std::string ctype1,  ///< Type of 1st coordinate axis
                std::string ctype2   ///< Type of 2nd coordinate axis
                ) {

    _wcsInfo = static_cast<struct wcsprm *>(malloc(sizeof(struct wcsprm)));
    if (_wcsInfo == NULL) {
        throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
    }

    _wcsInfo->flag = -1;
    int status = wcsini(true, 2, _wcsInfo);   //2 indicates a naxis==2, a two dimensional image
    if(status != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException,
                          (boost::format("Failed to allocate memory with wcsini. Status %d: %s") %
                           status % wcslibErrorMsg(status) ).str());
    }
    
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
    
    _nWcsInfo = 1;   //Specify that we have only one coordinate representation

    //Tell wcslib that we are set up
    status=wcsset(_wcsInfo);
    if(status != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Failed to setup wcs structure with wcsset. Status %d: %s") %
                           status % wcslibErrorMsg(status) ).str());
    }
}

/**
 * @brief Construct a Wcs that performs a linear conversion between pixels and radec
 *
 *
 */
lsst::afw::image::Wcs::Wcs(PointD crval, ///< ra/dec of centre of image
                           PointD crpix, ///< pixel coordinates of centre of image
                           Eigen::Matrix2d CD, ///< Conversion matrix with elements as defined in wcs.h
                           double equinox,         ///< Equinox used to define coord sys, e.g J2000
                           std::string raDecSys   ///<  Astrometry System, e.g FK5 or ICRS
                          ) : LsstBase(typeid(this)),
                              _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
                              _sipA(1,1), _sipB(1,1), _sipAp(1,1), _sipBp(1,1) {

    initWcslib(crval, crpix, CD, equinox, raDecSys, "RA---TAN", "DEC--TAN");
}

lsst::afw::image::Wcs::Wcs(
    PointD crval, ///< (ra, dec)
    PointD crpix,  ///< (x,y) pixel coords corresponding to crval
    Eigen::Matrix2d CD, ///< Linear mapping from crpix to crval
    Eigen::MatrixXd sipA, ///< Forward distortion Matrix A
    Eigen::MatrixXd sipB, ///< Forward distortion Matrix B
    Eigen::MatrixXd sipAp, ///<Reverse distortion Matrix Ap
    Eigen::MatrixXd sipBp,  ///<Reverse distortion Matrix Bp
    double equinox,               ///< Equinox of coord system, e.g J2000
    std::string raDecSys          ///< Celestial reference frame used, e.g FK5 or ICRS
                          ): LsstBase(typeid(this)),
                             _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
                             _sipA(sipA), _sipB(sipB), _sipAp(sipAp), _sipBp(sipBp) {

    if (sipA.rows() != sipA.cols() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipA must be square");
    }

    if (sipB.rows() != sipB.cols() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipB must be square");
    }

    if (sipAp.rows() != sipAp.cols() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipAp must be square");
    }

    if (sipBp.rows() != sipBp.cols() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipBp must be square");
    }

    //FIXME: When wcslib4.4 is released, the coordinate system should specify SIP. Until
    //then we need to work around the bug in earlier versions
    //initWcslib(crval, crpix, CD, equinox, raDecSys, "RA---TAN-SIP", "DEC--TAN-SIP"); //Wcslib >= 4.4
    initWcslib(crval, crpix, CD, equinox, raDecSys, "RA---TAN", "DEC--TAN");    
    
    //Init the SIP matrices
    _sipA = sipA;
    _sipB = sipB;
    _sipAp = sipAp;
    _sipBp = sipBp;

}


/**
 * @brief Decode the SIP headers for a given matrix, if present.
 */
static void decodeSipHeader(lsst::daf::base::PropertySet::Ptr fitsMetadata,
                            std::string const& which,
                            Eigen::MatrixXd *m) {
    std::string header = which + "_ORDER";
    if (!fitsMetadata->exists(header)) return;
    int order = fitsMetadata->get<int>(header);
    m->resize(order + 1, order + 1);
    boost::format format("%1%_%2%_%3%");
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order; ++j) {
            header = (format % which % i % j).str();
            if (fitsMetadata->exists(header)) {
                (*m)(i,j) = fitsMetadata->get<double>(header);
            }
            else {
                (*m)(i, j) = 0.0;
            }
        }
    }
}



/**
 * @brief Construct a Wcs from a FITS header, represented as PropertySet::Ptr
 *
 * @throw lsst::pex::exceptions::RuntimeErrorException on error
 */
lsst::afw::image::Wcs::Wcs(
    lsst::daf::base::PropertySet::Ptr fitsMetadata  ///< The contents of a valid FITS header
) :
    LsstBase(typeid(this)),
    _wcsInfo(NULL),
    _nWcsInfo(0),
    _nReject(0),
    _sipA(1,1), _sipB(1,1),
    _sipAp(1,1), _sipBp(1,1)
{
    // these should be set via policy - but for the moment...

    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    int nCards = lsst::afw::formatters::countFitsHeaderCards(fitsMetadata);
    if (nCards <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Could not parse FITS WCS: no header cards found");
    }

#define OLD 0
#if OLD
    // wcspih takes a non-const char* (because some versions of ctrl modify the string)
    // but we cannot afford to allow that to happen, so make a copy...
    std::string metadataStr = lsst::afw::formatters::formatFitsProperties(fitsMetadata);
    int len = metadataStr.size();
    boost::shared_ptr<char> hdrString = boost::shared_ptr<char>(new char[len + 1]);
    std::strcpy(hdrString.get(), metadataStr.c_str());

    int pihStatus = wcspih(hdrString.get(), nCards, _relax, _wcshdrCtrl, &_nReject, &_nWcsInfo, &_wcsInfo);

    if (pihStatus != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Could not parse FITS WCS: wcspih status = %d (%s)") %
                           pihStatus % wcslibErrorMsg(pihStatus) ).str());
    }
#else
    
    //Read out the bits of Wcs info that wcslib needs from the header
    try {
        PointD crval(fitsMetadata->getAsDouble("CRVAL1"), fitsMetadata->getAsDouble("CRVAL2") );
        PointD crpix(fitsMetadata->getAsDouble("CRPIX1"), fitsMetadata->getAsDouble("CRPIX2") ); 

        Eigen::Matrix2d CD;
        CD(0,0) = fitsMetadata->getAsDouble("CD1_1");
        CD(0,1) = fitsMetadata->getAsDouble("CD1_2");
        CD(1,0) = fitsMetadata->getAsDouble("CD2_1");
        CD(1,1) = fitsMetadata->getAsDouble("CD2_2");

        
        double equinox = -1;
        if(fitsMetadata->exists("EQUINOX")) {
            equinox = fitsMetadata->getAsDouble("EQUINOX");
        }
        
        std::string raDecSys = "";
        if(fitsMetadata->exists("RADESYS")) {
            raDecSys = fitsMetadata->getAsString("RADESYS");
        } else if(fitsMetadata->exists("RADECSYS")) {
            raDecSys = fitsMetadata->getAsString("RADECSYS");
        }
        //
        // Try to deal with missing equinox/radecsys
        //
        if (equinox == -1) {
            if (raDecSys == "" || raDecSys == "FK5") {
                equinox = 2000.0;
                raDecSys = "FK5";
            } else if (raDecSys == "FK4") {
                equinox = 1950.0;
            }
        } else if(equinox == 1950.0 && raDecSys == "") {
            raDecSys = "FK4";
        } else if(equinox == 2000.0 && raDecSys == "") {
            raDecSys = "FK5";
        }
        
        std::string ctype1 = fitsMetadata->getAsString("CTYPE1");
        std::string ctype2 = fitsMetadata->getAsString("CTYPE2");
        
        initWcslib(crval, crpix, CD, equinox, raDecSys, ctype1, ctype2);
        
    } catch(lsst::pex::exceptions::NotFoundException &e) {
        std::string msg = "An error occurred in Wcs() while parsing a fits header";
        e.addMessage(__FILE__, __LINE__, "Wcs", msg);
        throw;
    }
#endif


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
#if 0     
         throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, errStream.str());
#endif
    }

    /*
     * Decipher SIP headers if present.
     */
    decodeSipHeader(fitsMetadata, "A", &_sipA);
    decodeSipHeader(fitsMetadata, "B", &_sipB);
    decodeSipHeader(fitsMetadata, "AP", &_sipAp);
    decodeSipHeader(fitsMetadata, "BP", &_sipBp);

#if OLD == 0
    //FIXME:
    //Hack. Workaround a bug in wcslib that gets thinks RA---TAN-* is different to RA---TAN
    //Horrible consequences surely follow from this. This line should be removed when wcslib 4.4
    //appears. Note that is the ctypes are flipped (e.g SDSS images), this workaround will fail to
    //clean out the sip polynomials.
    if(!strncmp(_wcsInfo->ctype[0], "RA---TAN-SIP", STRLEN)) {
        strncpy(_wcsInfo->ctype[0], "RA---TAN", STRLEN);
        cout << "Changed RA\n";
    }

    if(!strncmp(_wcsInfo->ctype[1], "DEC--TAN-SIP", STRLEN)) {
        strncpy(_wcsInfo->ctype[1], "DEC--TAN", STRLEN);
        cout << "Changed Dec\n";
    }

    int status=wcsset(_wcsInfo);
    if(status != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Failed to setup wcs structure with wcsset. Status %d: %s") %
                           status % wcslibErrorMsg(status) ).str());
    }
    
#endif
}





/**
 * @brief Wcs copy constructor
 *
 * @throw lsst::pex::exceptions::MemoryException 
 * @throw lsst::pex::exceptions::RuntimeErrorException
 */
lsst::afw::image::Wcs::Wcs(Wcs const & rhs):
    LsstBase(typeid(this)),
    _wcsInfo(NULL),
    _nWcsInfo(0),
    _relax(rhs._relax),
    _wcsfixCtrl(rhs._wcsfixCtrl),
    _wcshdrCtrl(rhs._wcshdrCtrl),
    _nReject(rhs._nReject),

    _sipA(rhs._sipA),
    _sipB(rhs._sipB),
    _sipAp(rhs._sipAp),
    _sipBp(rhs._sipBp)
{
    if (rhs._nWcsInfo > 0) {
        _wcsInfo = static_cast<struct wcsprm *>(calloc(rhs._nWcsInfo, sizeof(struct wcsprm)));
        if (_wcsInfo == NULL) {
            throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
        }
        _wcsInfo->flag = -1;
        _nWcsInfo = rhs._nWcsInfo;
        for (int ii = 0; ii < rhs._nWcsInfo; ++ii) {
            // wcssub deep copies each _wcsInfo structure into newly allocated memory
            // this memory is managed by wcslib and so must be freed by wcsfree
            _wcsInfo[ii].flag = -1;
            int alloc=1;    //Unconditionally allocate memory when calling wcsini
            int status = wcscopy(alloc, rhs._wcsInfo + ii, _wcsInfo + ii);
            if (status != 0) {
                wcsvfree(&_nWcsInfo, &_wcsInfo);
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                    (boost::format("Could not copy WCS: wcscopy status = %d for wcs index %d. %s") % status % ii % wcslibErrorMsg(status) ).str());

            }
        }
    }
}

/**
 * @brief Wcs assignment operator
 *
 * @throw lsst::pex::exceptions::MemoryException 
 * @throw lsst::pex::exceptions::RuntimeErrorException
 */
lsst::afw::image::Wcs & lsst::afw::image::Wcs::operator = (const lsst::afw::image::Wcs & rhs) {
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
        _sipA = rhs._sipA;
        _sipB = rhs._sipB;
        _sipAp = rhs._sipAp;
        _sipBp = rhs._sipBp;

        if (rhs._nWcsInfo > 0) {
            // allocate wcs structs
            _wcsInfo = static_cast<struct wcsprm *>(calloc(rhs._nWcsInfo, sizeof(struct wcsprm)));
            if (_wcsInfo == NULL) {
                throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
            }
            _wcsInfo->flag = -1;
            _nWcsInfo = rhs._nWcsInfo;
            // deep-copy wcs data
            for (int ii = 0; ii < rhs._nWcsInfo; ++ii) {
                _wcsInfo[ii].flag = -1;
                int status = wcscopy(1, rhs._wcsInfo + ii, _wcsInfo + ii);
                if (status != 0) {
                    wcsvfree(&_nWcsInfo, &_wcsInfo);
                    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                        (boost::format("Failed to copy WCS info; wcscopy status = %d. %s") % status % wcslibErrorMsg(status)).str());
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

/// Return this, converted to a set of FITS cards
lsst::daf::base::PropertySet::Ptr lsst::afw::image::Wcs::getFitsMetadata() const { 
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
bool lsst::afw::image::Wcs::isFlipped() {

    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure is not initialised"));
    }

    double det = _wcsInfo->cd[0] * _wcsInfo->cd[3];
    det -= _wcsInfo->cd[1] * _wcsInfo->cd[2];
    
    if(det == 0){
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs scaling matrix is singular"));
    }

    if(det>0) {
        return true;
    }
    return false;
}
        
        

///
/// Move the pixel reference position by (dx, dy)
///
void lsst::afw::image::Wcs::shiftReferencePixel(double const dx, ///< How many pixels to shift in the column direction
                                                double const dy  ///< How many pixels to shift in the row direction
                                               ) {
    //If the _wcsInfo structure hasn't been initialised yet, then there's nothing to do
    if(_wcsInfo != NULL) {
        _wcsInfo->crpix[0] += dx;
        _wcsInfo->crpix[1] += dy;
    }
}

/// Return (ra, dec) of the origin of the WCS solution. Note that this need not
/// be the center of the image
lsst::afw::image::PointD lsst::afw::image::Wcs::getOriginRaDec() const {

    if(_wcsInfo != NULL) {
        return PointD(_wcsInfo->crval);
    } else {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure is not initialised"));
    }
   
}


/// Return XY  of the origin of the WCS solution. Note that this need not be
///the centre of the image
lsst::afw::image::PointD lsst::afw::image::Wcs::getOriginXY() const {

    if(_wcsInfo != NULL) {
        return PointD(_wcsInfo->crpix);
    } else {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

}


/// Convert from (ra, dec) to (column, row) coordinates
///
/// The conversion is of the form
/// [ ra ] = [ c11 c12 ]   [ col ]
/// [dec ]   [ c21 c22 ]   [ row ] 
///
/// where (col,row) = (0,0) = (ra, dec) is the centre of the WCS colution, and the matrix C is return by this function.
Eigen::Matrix2d lsst::afw::image::Wcs::getLinearTransformMatrix() const {

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

/**
 * Return the linear part of the Wcs, the CD matrix in FITS speak
 *
 * \sa 
 */
lsst::afw::geom::AffineTransform lsst::afw::image::Wcs::getAffineTransform() const
{
    return lsst::afw::geom::AffineTransform(getLinearTransformMatrix());
}

/*
 * Return the local linear approximation to the Wcs at the point (ra, dec) = sky
 *
 * This is currently implemented as a numerical derivative, but we should specialise the Wcs class (or rather
 * its implementation) to handle "simple" cases such as TAN-SIP analytically
 */                           
lsst::afw::geom::AffineTransform lsst::afw::image::Wcs::getAffineTransform(
        lsst::afw::image::PointD sky00  ///< Position in sky coordinates where transform is desired
                                                                          ) const
{
    //
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners of a square drawn in pixel
    // It'd be better to centre the square at sky00, but that would involve another conversion between sky and
    // pixel coordinates so I didn't bother
    //
    const double side = 10;             // length of the square's sides in pixels

    lsst::afw::image::PointD const pix00 = raDecToXY(sky00);
    
    lsst::afw::image::PointD const dsky10 = xyToRaDec(pix00 + lsst::afw::image::PointD(side, 0)) - sky00;
    lsst::afw::image::PointD const dsky01 = xyToRaDec(pix00 + lsst::afw::image::PointD(0, side)) - sky00;

    Eigen::Matrix2d m;
    m(0, 0) = dsky10.getX()/side;
    m(0, 1) = dsky01.getX()/side;
    m(1, 0) = dsky10.getY()/side;
    m(1, 1) = dsky01.getY()/side;

    Eigen::Vector2d p00;
    p00 << sky00.getX(), sky00.getY();
    return lsst::afw::geom::AffineTransform(m, lsst::afw::geom::ExtentD(p00));
}


/// Convert from (ra, dec) to (column, row) coordinates
///
/// \return The desired (col, row) position
lsst::afw::image::PointD lsst::afw::image::Wcs::raDecToXY(
    const double ra,   ///< Input right ascension
    const double dec   ///< Input declination
) const {

    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    double const skyTmp[2] = { ra, dec };
    double imgcrd[2];
    double phi, theta;
    double pixTmp[2];

    //Estimate undistorted pixel coordinates
    int stat[1];
    int status = 0;
    status = wcss2p(_wcsInfo, 1, 2, skyTmp, &phi, &theta, imgcrd, pixTmp, stat);
    if (status > 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of %d. %s") % status % wcslibErrorMsg(status)).str());
    }

    
    //Correct for distortion. We follow the notation of Shupe et al. here, including
    //capitalisation
    if( _sipAp.rows() > 1){
        //If the following assertions aren't true then something has gone seriously wrong.
        assert(_sipBp.rows() > 0 );
        assert(_sipAp.rows() == _sipAp.cols());
        assert(_sipBp.rows() == _sipBp.cols());        
        
        double U = pixTmp[0] - _wcsInfo->crpix[0];  //Relative, undistorted pixel coords
        double V = pixTmp[1]-  _wcsInfo->crpix[1];
    
        double F = 0;
        for(int i=0; i< _sipAp.rows(); ++i) {
            for(int j=0; j< _sipAp.cols(); ++j) {
                if (i+j>1 && i+j < _sipAp.rows() ) {
                    F += _sipAp(i,j)* pow(U, (int) i) * pow(V, (int) j);
                }
            }
        }    

        double G = 0;
        for(int i=0; i< _sipBp.rows(); ++i) {
            for(int j=0; j< _sipBp.cols(); ++j) {
                if (i+j>1 && i+j < _sipBp.rows() ) {
                    G += _sipBp(i,j)* pow(U, (int) i) * pow(V, (int) j);
                }
            }
        }

        pixTmp[0] = U + F + _wcsInfo->crpix[0];
        pixTmp[1] = V + G + _wcsInfo->crpix[1];
    }


    return lsst::afw::image::PointD(pixTmp[0] + lsst::afw::image::PixelZeroPos - 1,
                                    pixTmp[1] + lsst::afw::image::PixelZeroPos - 1); // wcslib assumes 1-indexed coords
}




/// \return The desired (col, row) position
lsst::afw::image::PointD lsst::afw::image::Wcs::raDecToXY(
    lsst::afw::image::PointD sky        ///< Input (ra, dec)
) const {
    return raDecToXY(sky.getX(), sky.getY());
}

/// Convert from (column, row) to (ra, dec) coordinates
/// \return The desired (ra, dec) position
lsst::afw::image::PointD lsst::afw::image::Wcs::xyToRaDec(
    double const x,                     ///< Input column position
    double const y                      ///< Input row position
) const {

    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    double pixTmp[2] = { x - lsst::afw::image::PixelZeroPos + 1,
                               y - lsst::afw::image::PixelZeroPos + 1}; // wcslib assumes 1-indexed coordinates
    double imgcrd[2];
    double phi, theta;
    double skyTmp[2];

    
    //Correct pixel positions for distortion if necessary
    if( _sipA.rows() > 1) {
        //If the following assertions aren't true then something has gone seriously wrong.
        assert(_sipB.rows() > 0 );
        assert(_sipA.rows() == _sipA.cols());
        assert(_sipB.rows() == _sipB.cols());

        double u = x - _wcsInfo->crpix[0];  //Relative pixel coords
        double v = y -  _wcsInfo->crpix[1];
        
        double f = 0;
        for(int i=0; i< _sipA.rows(); ++i) {
            for(int j=0; j< _sipA.cols(); ++j) {
                if (i+j>1 && i+j < _sipA.rows() ) {
                    f += _sipA(i,j)* pow(u, (int) i) * pow(v, (int) j);
                }
            }
        }

        double g = 0;
        for(int i=0; i< _sipB.rows(); ++i) {
            for(int j=0; j< _sipB.cols(); ++j) {
                if (i+j>1 && i+j < _sipB.rows() ) {
                    g += _sipB(i,j)* pow(u, (int) i) * pow(v, (int) j);
                }
            }
        }
        pixTmp[0]+= f;
        pixTmp[1]+= g;
    }
 
    int stat[1];
    int status = 0;
    status = wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, &phi, &theta, skyTmp, stat);
    if (status > 0) {
        wcsprt(_wcsInfo);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of  %d. %s") % status % wcslibErrorMsg(status) ).str());
                          
    }

    return lsst::afw::image::PointD(skyTmp);
}

/// Convert from (column, row) to (ra, dec) coordinates
/// \return The desired (ra, dec) position
lsst::afw::image::PointD lsst::afw::image::Wcs::xyToRaDec(
    lsst::afw::image::PointD pix        ///< Input (x, y)
) const {
    return xyToRaDec(pix.getX(), pix.getY());
}

/// Return the pixel area in deg^2 at a given pixel coordinate
///
/// \note This is an expensive routine, and should NOT be called for every pixel.  If you need a distortion
/// map, it'd be better to evaluate the scale at a number of points and interpolate
///
double lsst::afw::image::Wcs::pixArea(lsst::afw::image::PointD pix00) const
{
    //
    // Figure out the (0, 0), (0, 1), and (1, 0) ra/dec coordinates of the corners of a square drawn in pixel
    // It'd be better to centre the square at sky00, but that would involve another conversion between sky and
    // pixel coordinates so I didn't bother
    //
    const double side = 10;             // length of the square's sides in pixels

    lsst::afw::image::PointD const sky00 = xyToRaDec(pix00);
    
    lsst::afw::image::PointD const dsky10 = xyToRaDec(pix00 + lsst::afw::image::PointD(side, 0)) - sky00;
    lsst::afw::image::PointD const dsky01 = xyToRaDec(pix00 + lsst::afw::image::PointD(0, side)) - sky00;

    double const cosDec = cos(sky00.getY()*M_PI/180.0);
    return cosDec*fabs(dsky01.getX()*dsky10.getY() - dsky01.getY()*dsky10.getX())/(side*side);
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
        if (metadata->getAsDouble("CRPIX1" + wcsName) == 1 && metadata->getAsDouble("CRPIX2" + wcsName) == 1) {
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
