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
namespace ublas = boost::numeric::ublas;


/**
 * @brief Construct an invalid Wcs given no arguments
 *
 * @throw lsst::pex::exceptions::RuntimeErrorException on error
 */
lsst::afw::image::Wcs::Wcs() :
    LsstBase(typeid(this)),
    _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
    _sipA(0,0), _sipB(0,0), _sipAp(0,0), _sipBp(0,0){
}


///
/// Function to initialise the wcslib structure. Should only be called from Wcs constructors
void lsst::afw::image::Wcs::initWcslib(PointD crval, PointD crpix, ublas::matrix<double> CD){

    _wcsInfo = static_cast<struct wcsprm *>(malloc(sizeof(struct wcsprm)));
    if (_wcsInfo == NULL) {
        throw LSST_EXCEPT(lsst::pex::exceptions::MemoryException, "Cannot allocate WCS info");
    }
    _wcsInfo->flag = -1;
    wcsini(true, 2, _wcsInfo);   //2 indicates a naxis==2, a two dimensional image

    _wcsInfo->crval[0] = crval.getX();
    _wcsInfo->crval[1] = crval.getY();

    _wcsInfo->crpix[0] = crpix.getX();
    _wcsInfo->crpix[1] = crpix.getY();

    //Setting this to TAN-SIP gives the wrong result causes the test
    //testWcs.cc::radec_to_xy to fail when I construct using
    //wcs(crval, crpix, CD);
    strncpy(_wcsInfo->ctype[0], "RA---TAN", 72);  //wcsini sets ctype[] to have length 72
    strncpy(_wcsInfo->ctype[1], "DEC--TAN", 72);


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
    
    _nWcsInfo = 1;   //Specify that we have only one coordinate representation   
}
    

/**
 * @brief Construct a Wcs that performs a linear conversion between pixels and radec
 *
 *
 */
lsst::afw::image::Wcs::Wcs(PointD crval, ///< ra/dec of centre of image
                           PointD crpix, ///< pixel coordinates of centre of image
                           ublas::matrix<double> CD ///< Conversion matrix with elements as defined
                                                    ///< in wcs.h
                          ) : LsstBase(typeid(this)),
                              _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
                              _sipA(0,0), _sipB(0,0), _sipAp(0,0), _sipBp(0,0) {

    initWcslib(crval, crpix, CD);
}

lsst::afw::image::Wcs::Wcs(
    PointD crval, ///< (ra, dec)
    PointD crpix,  ///< (x,y) pixel coords corresponding to crval
    ublas::matrix<double> CD, ///< Linear mapping from crpix to crval
    ublas::matrix<double> sipA, ///< Forward distortion Matrix A
    ublas::matrix<double> sipB, ///< Forward distortion Matrix B
    ublas::matrix<double> sipAp, ///<Reverse distortion Matrix Ap
    ublas::matrix<double> sipBp  ///<Reverse distortion Matrix Bp
                          ): LsstBase(typeid(this)),
                             _wcsInfo(NULL), _nWcsInfo(0), _relax(0), _wcsfixCtrl(0), _wcshdrCtrl(0), _nReject(0),
                             _sipA(0,0), _sipB(0,0), _sipAp(0,0), _sipBp(0,0) {

    if (sipA.size1() != sipA.size2() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipA must be square");
    }

    if (sipB.size1() != sipB.size2() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipB must be square");
    }

    if (sipAp.size1() != sipAp.size2() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipAp must be square");
    }

    if (sipBp.size1() != sipBp.size2() ){
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Error: Matrix sipBp must be square");
    }

    
    initWcslib(crval, crpix, CD);

    //Init the SIP matrices
    _sipA = sipA;
    _sipB = sipB;
    _sipAp = sipAp;
    _sipBp = sipBp;

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
    _sipA(0,0), _sipB(0,0),
    _sipAp(0,0), _sipBp(0,0)
{
    // these should be set via policy - but for the moment...

    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    std::string metadataStr = lsst::afw::formatters::formatFitsProperties(fitsMetadata);
    int nCards = lsst::afw::formatters::countFitsHeaderCards(fitsMetadata);
    if (nCards <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          "Could not parse FITS WCS: no header cards found");
    }
    
    // wcspih takes a non-const char* (because some versions of ctrl modify the string)
    // but we cannot afford to allow that to happen, so make a copy...
    int len = metadataStr.size();
    boost::shared_ptr<char> hdrString = boost::shared_ptr<char>(new char[len + 1]);
    std::strcpy(hdrString.get(), metadataStr.c_str());

    int pihStatus = wcspih(hdrString.get(), nCards, _relax, _wcshdrCtrl, &_nReject, &_nWcsInfo, &_wcsInfo);

    if (pihStatus != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Could not parse FITS WCS: wcspih status = %d (%s)") %
                           pihStatus % wcs_errmsg[pihStatus]).str());
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
         throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, errStream.str());
#endif
    }
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
            int status = wcscopy(1, rhs._wcsInfo + ii, _wcsInfo + ii);
            if (status != 0) {
                wcsvfree(&_nWcsInfo, &_wcsInfo);
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                    (boost::format("Could not copy WCS: wcscopy status = %d for wcs index %d") % status % ii).str());
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
                        (boost::format("Failed to copy WCS info; wcscopy status = %d") % status).str());
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
boost::numeric::ublas::matrix<double> lsst::afw::image::Wcs::getLinearTransformMatrix() const {

    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }
    
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
                          (boost::format("Error: wcslib returned a status code of  %d") % status).str());
    }

    //Correct for distortion. We follow the notation of Shupe et al. here, including
    //capitalisation
    if( _sipAp.size1() > 0){
        //If the following assertions aren't true then something has gone seriously wrong.
        assert(_sipBp.size1() > 0 );
        assert(_sipAp.size1() == _sipAp.size2());
        assert(_sipBp.size1() == _sipBp.size2());        
        
        double U = pixTmp[0] - _wcsInfo->crpix[0];  //Relative, undistorted pixel coords
        double V = pixTmp[1]-  _wcsInfo->crpix[1];
    
        double F = 0;
        for(unsigned int i=0; i< _sipAp.size1(); ++i) {
            for(unsigned int j=0; j< _sipAp.size2(); ++j) {
                if (i+j>1 && i+j < _sipAp.size1() ) {
                    F += _sipAp(i,j)* pow(U, (int) i) * pow(V, (int) j);
                }
            }
        }    

        double G = 0;
        for(unsigned int i=0; i< _sipBp.size1(); ++i) {
            for(unsigned int j=0; j< _sipBp.size2(); ++j) {
                if (i+j>1 && i+j < _sipBp.size1() ) {
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
    if( _sipA.size1() > 0) {
        //If the following assertions aren't true then something has gone seriously wrong.
        assert(_sipB.size1() > 0 );
        assert(_sipA.size1() == _sipA.size2());
        assert(_sipB.size1() == _sipB.size2());

        double u = x - _wcsInfo->crpix[0];  //Relative pixel coords
        double v = y -  _wcsInfo->crpix[1];
        
        double f = 0;
        for(unsigned int i=0; i< _sipA.size1(); ++i) {
            for(unsigned int j=0; j< _sipA.size2(); ++j) {
                if (i+j>1 && i+j < _sipA.size1() ) {
                    f += _sipA(i,j)* pow(u, (int) i) * pow(v, (int) j);
                }
            }
        }

        double g = 0;
        for(unsigned int i=0; i< _sipB.size1(); ++i) {
            for(unsigned int j=0; j< _sipB.size2(); ++j) {
                if (i+j>1 && i+j < _sipB.size1() ) {
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
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of  %d") % status).str());
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
    // Figure out the (0, 0), (0, 1), and (1, 0) pixel coordinates of the corners of a square drawn in ra/dec
    // It'd be better to centre the square at pix00, but that would involve another conversion from sky to
    // pixel coordinates so I didn't bother
    //
    const double side = 1e-3;           // length of the square's sides in degrees

    lsst::afw::image::PointD const sky00 = xyToRaDec(pix00);
    double const cosDec = cos((sky00.getY() + side/2)*M_PI/180.0);
    
    lsst::afw::image::PointD const dpix01 = raDecToXY(sky00 + lsst::afw::image::PointD(0,           side)) - pix00;
    lsst::afw::image::PointD const dpix10 = raDecToXY(sky00 + lsst::afw::image::PointD(side*cosDec, 0))    - pix00;

    return (side*side)/fabs(dpix01.getX()*dpix10.getY() - dpix01.getY()*dpix10.getX());
}
