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
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/formatters/TanWcsFormatter.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/TanWcs.h"    

namespace except = lsst::pex::exceptions; 
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
using namespace std;


typedef lsst::daf::base::PropertySet PropertySet;
typedef lsst::afw::image::TanWcs TanWcs;
typedef lsst::afw::geom::Point2D GeomPoint;
typedef lsst::afw::coord::Coord Coord;

const int lsstToFitsPixels = +1;
const int fitsToLsstPixels = -1;


static void decodeSipHeader(lsst::daf::base::PropertySet::Ptr fitsMetadata,
                            std::string const& which,
                            Eigen::MatrixXd *m);


TanWcs::TanWcs() : 
    Wcs(),
    _hasDistortion(false),
    _sipA(), _sipB(), _sipAp(), _sipBp() {
}

afwGeom::Angle TanWcs::pixelScale() const {
	// HACK -- assume "CD" elements are set (and are in degrees)
	double* cd = _wcsInfo->m_cd;
	assert(cd);
	return sqrt(fabs(cd[0]*cd[3] - cd[1]*cd[2])) * afwGeom::degrees;
}

///Create a Wcs from a fits header. Don't call this directly. Use makeWcs() instead, which will figure
///out which (if any) sub-class of Wcs is appropriate
TanWcs::TanWcs(lsst::daf::base::PropertySet::Ptr const fitsMetadata) : 
    Wcs(fitsMetadata),
    _hasDistortion(false),
    _sipA(), _sipB(), _sipAp(), _sipBp() {

    //Internal params for wcslib. These should be set via policy - but for the moment...
    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    //Check that the header isn't empty
    if(fitsMetadata->nameCount() == 0) {
        string msg = "Fits metadata contains no cards";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }
    
    //Check for tangent plane projection
    string ctype1 = fitsMetadata->getAsString("CTYPE1");
    string ctype2 = fitsMetadata->getAsString("CTYPE2");

    if((ctype1.substr(5, 3) != "TAN") || (ctype2.substr(5, 3) != "TAN") ) {
        string msg = "One or more axes isn't in TAN projection (ctype1 = \"" + ctype1 + "\", ctype2 = \"" + ctype2 + "\")";
        throw LSST_EXCEPT(except::InvalidParameterException, msg);
    }

    //Check for distorton terms. With two ctypes, there are 4 alternatives, only
    //two of which are valid.. Both have distortion terms or both don't. 
    int nSip = (((ctype1.substr(8, 4) == "-SIP") ? 1 : 0) +
		((ctype2.substr(8, 4) == "-SIP") ? 1 : 0));
    
    switch (nSip) {
        case 0:
            _hasDistortion = false;
            break;
        case 1:
            {//Invalid case. Throw an exception
                string msg = "Distortion key found for only one CTYPE";
                throw LSST_EXCEPT(except::InvalidParameterException, msg);
            }
            break;  //Not necessary, but looks naked without it.
        case 2:
            _hasDistortion = true;
            
            //Hide the distortion from wcslib
            fitsMetadata->set<string>("CTYPE1", ctype1.substr(0,8));
            fitsMetadata->set<string>("CTYPE2", ctype2.substr(0,8));
            
            //Save SIP information
            decodeSipHeader(fitsMetadata, "A", &_sipA);
            decodeSipHeader(fitsMetadata, "B", &_sipB);
            decodeSipHeader(fitsMetadata, "AP", &_sipAp);
            decodeSipHeader(fitsMetadata, "BP", &_sipBp);

            // this gets called in the Wcs (base class) constructor
            // We just changed fitsMetadata, so we have to re-init wcslib
            initWcsLibFromFits(fitsMetadata);
            
            break;
    }
    
    //Check that the existence of forward sip matrices <=> existence of reverse matrices
    if (_hasDistortion) {
        if (_sipA.rows() <= 1 || _sipB.rows() <= 1) {
                string msg = "Existence of forward distorton matrices suggested, but not found";
                throw LSST_EXCEPT(except::InvalidParameterException, msg);
        }

        if (_sipAp.rows() <= 1 || _sipBp.rows() <= 1) {
                string msg = "Forward distorton matrices present, but no reverse matrices";
                throw LSST_EXCEPT(except::InvalidParameterException, msg);
        }
    }
            
}




///@brief Decode the SIP headers for a given matrix, if present.
static void decodeSipHeader(lsst::daf::base::PropertySet::Ptr fitsMetadata,
                            std::string const& which,
                            Eigen::MatrixXd *m) {
    std::string header = which + "_ORDER";
    if (!fitsMetadata->exists(header)) return;
    int order = fitsMetadata->getAsInt(header);
    m->resize(order + 1, order + 1);
    boost::format format("%1%_%2%_%3%");
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order; ++j) {
            header = (format % which % i % j).str();
            if (fitsMetadata->exists(header)) {
                (*m)(i,j) = fitsMetadata->getAsDouble(header);
            }
            else {
                (*m)(i, j) = 0.0;
            }
        }
    }
}



/// \brief Construct a tangent plane wcs without distortion terms    
/// \param crval The sky position of the reference point
/// \param crpix The pixel position corresponding to crval in Lsst units
/// \param CD    Matrix describing transformations from pixel to sky positions
/// \param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
/// \param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
/// \param cunits1 Units of sky position. One of deg, arcmin or arcsec
/// \param cunits2 Units of sky position. One of deg, arcmin or arcsec
TanWcs::TanWcs(
        const lsst::afw::geom::Point2D crval,
        const lsst::afw::geom::Point2D crpix,
        const Eigen::Matrix2d &CD,
        double equinox,
        std::string raDecSys,
        const std::string cunits1,
        const std::string cunits2
       ) :
       Wcs(crval, crpix, CD, "RA---TAN", "DEC--TAN", equinox, raDecSys, cunits1, cunits2),
       _hasDistortion(false),
       _sipA(), _sipB(), _sipAp(), _sipBp() {
       
       //Nothing to do here
}


/// \brief Construct a tangent plane wcs with distortion terms    
/// \param crval The sky position of the reference point
/// \param crpix The pixel position corresponding to crval
/// \param CD    Matrix describing transformations from pixel to sky positions
/// \param sipA Forward distortion matrix for axis 1
/// \param sipB Forward distortion matrix for axis 2
/// \param sipAp Reverse distortion matrix for axis 1
/// \param sipBp Reverse distortion matrix for axis 2
/// \param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
/// \param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
/// \param cunits1 Units of sky position. One of deg, arcmin or arcsec
/// \param cunits2 Units of sky position. One of deg, arcmin or arcsec
TanWcs::TanWcs(
        const lsst::afw::geom::Point2D crval,
        const lsst::afw::geom::Point2D crpix,
        const Eigen::Matrix2d &CD, 
        Eigen::MatrixXd const & sipA, 
        Eigen::MatrixXd const & sipB, 
        Eigen::MatrixXd const & sipAp, 
        Eigen::MatrixXd const & sipBp,  
        double equinox, std::string raDecSys,
        const std::string cunits1, const std::string cunits2
           ) :
           Wcs(crval, crpix, CD, "RA---TAN", "DEC--TAN", equinox, raDecSys, cunits1, cunits2),
           _hasDistortion(true),
           //Sip's set by a dedicated method that does error checking
           _sipA(), _sipB(), _sipAp(), _sipBp() {

    //Input checking is done constructor of base class, so don't need to do 
    //any here.

    //Set the distortion terms
    setDistortionMatrices(sipA, sipB, sipAp, sipBp);
}


///Copy constructor
TanWcs::TanWcs(lsst::afw::image::TanWcs const & rhs) :
    Wcs(rhs),
    _hasDistortion(rhs._hasDistortion),
    _sipA(rhs._sipA), 
    _sipB(rhs._sipB),
    _sipAp(rhs._sipAp), 
    _sipBp(rhs._sipBp) {
    
}

bool TanWcs::operator==(const TanWcs &rhs) const {
    return Wcs::operator==(rhs) &&
        _hasDistortion == rhs._hasDistortion &&
        (!_hasDistortion || (_sipA == rhs._sipA &&
                             _sipB == rhs._sipB &&
                             _sipAp == rhs._sipAp &&
                             _sipBp == rhs._sipBp));
}

///Assignment operator    
TanWcs & TanWcs::operator = (const TanWcs & rhs){
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

        _hasDistortion = false;        
        if (rhs._hasDistortion) {
            _hasDistortion = true;
            _sipA = rhs._sipA;
            _sipB = rhs._sipB;
            _sipAp = rhs._sipAp;
            _sipBp = rhs._sipBp;
        }
    }
    
    return *this;
}

/// \brief Clone a TanWcs.
afwImage::Wcs::Ptr TanWcs::clone(void) const {
    return afwImage::Wcs::Ptr(new TanWcs(*this));
}

//
// Accessors
//
GeomPoint TanWcs::skyToPixelImpl(afwGeom::Angle sky1, // RA
                                 afwGeom::Angle sky2  // Dec
                                ) const {
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(except::RuntimeErrorException, "Wcs structure not initialised"));
    }

    double skyTmp[2];
    double imgcrd[2];
    double phi, theta;
    double pixTmp[2];

    //Estimate undistorted pixel coordinates
    int stat[1];
    int status = 0;

    skyTmp[_wcsInfo->lng] = sky1.asDegrees();
    skyTmp[_wcsInfo->lat] = sky2.asDegrees();

    status = wcss2p(_wcsInfo, 1, 2, skyTmp, &phi, &theta, imgcrd, pixTmp, stat);
    if (status > 0) {
        throw LSST_EXCEPT(except::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of %d. %s") %
                           status % wcs_errmsg[status]).str());
    }

    
    //Correct for distortion. We follow the notation of Shupe et al. here, including
    //capitalisation
    if( _hasDistortion) {
        GeomPoint pix = afwGeom::Point2D(pixTmp[0], pixTmp[1]);
        GeomPoint dpix = distortPixel(pix);
        pixTmp[0] = dpix[0];
        pixTmp[1] = dpix[1];
    }

    // wcslib assumes 1-indexed coords
    double offset = lsst::afw::image::PixelZeroPos + fitsToLsstPixels;
    return afwGeom::Point2D(pixTmp[0]+offset, pixTmp[1]+offset);

}

GeomPoint TanWcs::undistortPixel(const lsst::afw::geom::Point2D pix) const {
    if (!_hasDistortion) {
        return GeomPoint(pix);
    }
    //If the following assertions aren't true then something has gone seriously wrong.
    assert(_sipB.rows() > 0 );
    assert(_sipA.rows() == _sipA.cols());
    assert(_sipB.rows() == _sipB.cols());

    double u = pix[0] - _wcsInfo->crpix[0];  //Relative pixel coords
    double v = pix[1] - _wcsInfo->crpix[1];
        
    double f = 0;
    for(int i=0; i< _sipA.rows(); ++i) {
        for(int j=0; j< _sipA.cols(); ++j) {
            if (i+j>1 && i+j < _sipA.rows() ) {
                f += _sipA(i,j)* pow(u, i) * pow(v, j);
            }
        }
    }

    double g = 0;
    for(int i=0; i< _sipB.rows(); ++i) {
        for(int j=0; j< _sipB.cols(); ++j) {
            if (i+j>1 && i+j < _sipB.rows() ) {
                g += _sipB(i,j)* pow(u, i) * pow(v, j);
            }
        }
    }

    return afwGeom::Point2D(pix[0] + f, pix[1] + g);
}

GeomPoint TanWcs::distortPixel(const lsst::afw::geom::Point2D pix) const {
    if (!_hasDistortion) {
        return GeomPoint(pix);
    }
    //If the following assertions aren't true then something has gone seriously wrong.
    assert(_sipBp.rows() > 0 );
    assert(_sipAp.rows() == _sipAp.cols());
    assert(_sipBp.rows() == _sipBp.cols());        
        
    double U = pix[0] - _wcsInfo->crpix[0];  //Relative, undistorted pixel coords
    double V = pix[1] - _wcsInfo->crpix[1];
    
    double F = 0;
    for(int i=0; i< _sipAp.rows(); ++i) {
        for(int j=0; j< _sipAp.cols(); ++j) {
            F += _sipAp(i,j)* pow(U, i) * pow(V, j);
        }
    }    

    double G = 0;
    for(int i=0; i< _sipBp.rows(); ++i) {
        for(int j=0; j< _sipBp.cols(); ++j) {
            G += _sipBp(i,j)* pow(U, i) * pow(V, j);
        }
    }
    return afwGeom::Point2D(U + F + _wcsInfo->crpix[0],
                            V + G + _wcsInfo->crpix[1]);
}

/************************************************************************************************************/
/*
 * Worker routine for pixelToSky
 */
void
TanWcs::pixelToSkyImpl(double pixel1, double pixel2, afwGeom::Angle sky[2]) const
{
    if(_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Wcs structure not initialised"));
    }

    // wcslib assumes 1-indexed coordinates
    double pixTmp[2] = { pixel1 - lsst::afw::image::PixelZeroPos + lsstToFitsPixels,
                         pixel2 - lsst::afw::image::PixelZeroPos + lsstToFitsPixels}; 
    double imgcrd[2];
    double phi, theta;
    
    //Correct pixel positions for distortion if necessary
    if( _hasDistortion) {
        GeomPoint pix = afwGeom::Point2D(pixTmp[0], pixTmp[1]);
        GeomPoint dpix = undistortPixel(pix);
        pixTmp[0] = dpix[0];
        pixTmp[1] = dpix[1];
    }
 
    int status = 0;
	double skyTmp[2];
    if (wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, &phi, &theta, skyTmp, &status) > 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                          (boost::format("Error: wcslib returned a status code of  %d. %s") %
                           status % wcs_errmsg[status]).str());
    }
	sky[0] = skyTmp[0] * afwGeom::degrees;
	sky[1] = skyTmp[1] * afwGeom::degrees;
}

/************************************************************************************************************/

lsst::daf::base::PropertyList::Ptr TanWcs::getFitsMetadata() const {
    return lsst::afw::formatters::TanWcsFormatter::generatePropertySet(*this);       
}


#if 0
//Rely on base class implementation for now, because this implementation isn't working
lsst::afw::afwGeom::AffineTransform TanWcs::linearizeAt(lsst::afw::geom::Point2D const & sky) const {
    
    Eigen::Matrix2d CD(2,2);
    CD(0,0) = _wcsInfo->cd[0];
    CD(0,1) = _wcsInfo->cd[1];
    CD(1,0) = _wcsInfo->cd[2];
    CD(1,1) = _wcsInfo->cd[3];
    
    GeomPoint const pix00 = skyToPixel(sky);
    Eigen::Vector2d pix(pix00[0] - _wcsInfo->crpix[0], pix00[1] - _wcsInfo->crpix[1]);
    
    //iwc == Intermediate world coordinates (x,y) in Greisen & Calabretta
    Eigen::Vector2d iwc = CD*pix;
    double x2 = iwc[0]*iwc[0];
    double y2 = iwc[1]*iwc[1];
    
    Eigen::Matrix2d m;
    m(0,0) = CD(0,0)/(1+x2);
    m(0,1) = CD(0,1)/(1+x2);
    m(1,0) = CD(1,0)/(1+y2);
    m(1,1) = CD(1,1)/(1+y2);
    
    cout << "TanWcs" << endl << m << endl;
    Eigen::Vector2d sky00v;
    sky00v << sky.getX(), sky.getY();
    Eigen::Vector2d pix00v;
    pix00v << pix00.getX(), pix00.getY();
    return lsst::afw::afwGeom::AffineTransform(m, lsst::afw::afwGeom::Extent2D(sky00v - m * pix00v));
}
#endif

//
// Mutators
//    

/// \brief Set the distortion matrices
/// \param sipA Forward distortion matrix for 1st axis
/// \param sipB Forward distortion matrix for 2nd axis
/// \param sipAp Reverse distortion matrix for 1st axis
/// \param sipBp Reverse distortion matrix for 2nd axis
void TanWcs::setDistortionMatrices(Eigen::MatrixXd const & sipA, 
                                   Eigen::MatrixXd const & sipB,
                                   Eigen::MatrixXd const & sipAp,
                                   Eigen::MatrixXd const & sipBp) {

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

            
    //Set the SIP terms
    _hasDistortion = true;
    _sipA = sipA;
    _sipB = sipB;
    _sipAp = sipAp;
    _sipBp = sipBp;
}


