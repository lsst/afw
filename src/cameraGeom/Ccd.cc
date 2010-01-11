/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Ccd.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace cameraGeom = lsst::afw::cameraGeom;

/************************************************************************************************************/
/**
 * Add an Amp to the set known to be part of this Ccd
 *
 *  The \c iX and \c iY values are the 0-indexed position of the Amp on the CCD; e.g. (4, 1)
 * for the top right Amp on a CCD with serials across the top and bottom, and each serial split 5 ways
 */
void cameraGeom::Ccd::addAmp(int const iX, ///< x-index of this Amp
                             int const iY, ///< y-index of this Amp
                             cameraGeom::Amp const& amp_c ///< The amplifier to add to the Ccd's manifest
                         )
{
    cameraGeom::Amp::Ptr amp(new Amp(amp_c)); // the Amp with absolute coordinates
    //
    // Correct Amp's coordinate system to be absolute within CCD
    //
    {
        afwImage::BBox ampPixels = amp->getAllPixels();
        amp->shift(iX*ampPixels.getWidth(), iY*ampPixels.getHeight());
    }

    getAllPixels().grow(amp->getAllPixels().getLLC());
    getAllPixels().grow(amp->getAllPixels().getURC());
    //
    // Now deal with the geometry after we trim everything except the dataSec
    //
    {
        afwImage::BBox dataSec = amp_c.getDataSec();
        dataSec.shift(-dataSec.getX0(), -dataSec.getY0());
        dataSec.shift(iX*dataSec.getWidth(), iY*dataSec.getHeight());
        amp->getDataSec(true) = dataSec;
        amp->getAllPixels(true) = dataSec;
    }
    getAllTrimmedPixels().grow(amp->getDataSec(true).getLLC());
    getAllTrimmedPixels().grow(amp->getDataSec(true).getURC());
    
    _amps.push_back(amp);

    setCenterPixel(afwGeom::PointI::makeXY(getAllPixels(true).getWidth()/2, getAllPixels(true).getHeight()/2));
}

/************************************************************************************************************/
/**
 * Return the pixel position given an offset from the chip centre, in mm
 */
afwGeom::Point2I cameraGeom::Ccd::getIndexFromPosition(
        afwGeom::Point2D const& pos     ///< Offset from chip centre, mm
                                                      ) const
{
    if (isTrimmed()) {
        return cameraGeom::Detector::getIndexFromPosition(pos);
    }

    lsst::afw::geom::Point2I centerPixel = getCenterPixel();
    double pixelSize = getPixelSize();

    return afwGeom::Point2I::makeXY(centerPixel[0] + pos[0]/pixelSize, centerPixel[1] + pos[1]/pixelSize);
}

/**
 * Return the offset from the chip centre, in mm, given a pixel position wrt the chip centre
 * \sa getPositionFromPixel
 */
afwGeom::Point2D cameraGeom::Ccd::getPositionFromIndex(
        afwGeom::Point2I const& pix,    ///< Pixel coordinates wrt Ccd's centre
        bool const isTrimmed            ///< Is this detector trimmed?
                                                      ) const
{
    if (isTrimmed) {
        return cameraGeom::Detector::getPositionFromIndex(pix, isTrimmed);
    }

    double pixelSize = getPixelSize();

    afwGeom::Point2I const& centerPixel = getCenterPixel();
    cameraGeom::Amp::ConstPtr amp = findAmp(afwGeom::Point2I::makeXY(pix[0] + centerPixel[0],
                                                                pix[1] + centerPixel[1]));
    afwImage::PointI const off = amp->getDataSec(false).getLLC() - amp->getDataSec(true).getLLC();
    afwGeom::Point2I const offsetPix = pix - afwGeom::Extent2I(afwGeom::Point2I::makeXY(off[0], off[1]));

    return afwGeom::Point2D::makeXY(offsetPix[0]*pixelSize, offsetPix[1]*pixelSize);
}    

namespace {
    struct findById {
        findById(cameraGeom::Id id) : _id(id) {}
        bool operator()(cameraGeom::Amp::Ptr amp) const {
            return _id == amp->getId();
        }
    private:
        cameraGeom::Id _id;
    };

    struct findByPixel {
        findByPixel(
                  afwGeom::Point2I point,
                  bool isTrimmed
                 ) :
            _point(afwImage::PointI(point[0], point[1])),
            _isTrimmed(isTrimmed)
        { }

        bool operator()(cameraGeom::Amp::Ptr amp) const {
            return amp->getAllPixels(_isTrimmed).contains(_point);
        }
    private:
        afwImage::PointI _point;
        bool _isTrimmed;
    };
}

/// Set the trimmed status of this Ccd
void cameraGeom::Ccd::setTrimmed(bool isTrimmed ///< True iff the bias/overclock have been removed
                                ) {
    cameraGeom::Detector::setTrimmed(isTrimmed);
    // And the Amps too
    for (iterator ptr = begin(); ptr != end(); ++ptr) {
        (*ptr)->setTrimmed(isTrimmed);
    }
}

/**
 * Find an Amp given an Id
 */
cameraGeom::Amp::Ptr cameraGeom::Ccd::findAmp(cameraGeom::Id const id ///< The desired Id
                                             ) const {
    AmpSet::const_iterator result = std::find_if(_amps.begin(), _amps.end(), findById(id));
    if (result == _amps.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Amp with serial %||") % id).str());
    }
    return *result;
}

/**
 * Find an Amp given a position
 */
cameraGeom::Amp::Ptr cameraGeom::Ccd::findAmp(afwGeom::Point2I const& pixel ///< The desired pixel
                                             ) const {
    return findAmp(pixel, isTrimmed());
}

/**
 * Find an Amp given a position and a request for trimmed or untrimmed coordinates
 */
cameraGeom::Amp::Ptr cameraGeom::Ccd::findAmp(afwGeom::Point2I const& pixel, ///< The desired pixel 
                                        bool const isTrimmed                 ///< Is Ccd trimmed?
                                             ) const {
    AmpSet::const_iterator result = std::find_if(_amps.begin(), _amps.end(), findByPixel(pixel, isTrimmed));
    if (result == _amps.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Amp containing pixel (%d, %d)") %
                           pixel.getX() % pixel.getY()).str());
    }
    return *result;
}

#include "boost/bind.hpp"

/// Offset a Ccd by the specified amount
void cameraGeom::Ccd::shift(int dx,     ///< How much to offset in x (pixels)
                            int dy      ///< How much to offset in y (pixels)
                        ) {
    Detector::shift(dx, dy);
    
    std::for_each(_amps.begin(), _amps.end(), boost::bind(&Amp::shift, _1, boost::ref(dx), boost::ref(dx)));
}
