/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Ccd.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace camGeom = lsst::afw::cameraGeom;

/************************************************************************************************************/
/**
 * Add an Amp to the set known to be part of this Ccd
 *
 *  The \c iX and \c iY values are the 0-indexed position of the Amp on the CCD; e.g. (4, 1)
 * for the top right Amp on a CCD with serials across the top and bottom, and each serial split 5 ways
 */
void camGeom::Ccd::addAmp(int const iX, ///< x-index of this Amp
                          int const iY, ///< y-index of this Amp
                          camGeom::Amp const& amp_c ///< The amplifier to add to the Ccd's manifest
                         )
{
    camGeom::Amp amp = amp_c;           // the Amp with absolute coordinates
    //
    // Correct Amp's coordinate system to be absolute within CCD
    //
    {
        afwImage::BBox ampPixels = amp.getAllPixels();
        amp.shift(iX*ampPixels.getWidth(), iY*ampPixels.getHeight());
    }

    getAllPixels().grow(amp.getAllPixels().getLLC());
    getAllPixels().grow(amp.getAllPixels().getURC());
    //
    // Now deal with the geometry after we trim everything except the dataSec
    //
    {
        afwImage::BBox dataSec = amp_c.getDataSec();
        dataSec.shift(-dataSec.getX0(), -dataSec.getY0());
        dataSec.shift(iX*dataSec.getWidth(), iY*dataSec.getHeight());
        amp.getDataSec(true) = dataSec;
        amp.getAllPixels(true) = dataSec;
    }
    getAllTrimmedPixels().grow(amp.getDataSec(true).getLLC());
    getAllTrimmedPixels().grow(amp.getDataSec(true).getURC());
    
    _amps.push_back(amp);
}

/************************************************************************************************************/
/**
 * Return the pixel position given an offset from the chip centre, in mm
 */
afwGeom::Point2I camGeom::Ccd::getIndexFromPosition(
        afwGeom::Point2D pos            ///< Offset from chip centre, mm
                                                        ) const
{
    if (isTrimmed()) {
        return camGeom::Detector::getIndexFromPosition(pos);
    }

    lsst::afw::geom::Point2I centerPixel = getCenterPixel();
    double pixelSize = getPixelSize();

    Eigen::Vector2i pix;
    pix << centerPixel[0] + pos[0]/pixelSize, centerPixel[1] + pos[1]/pixelSize;
    return afwGeom::Point2I(pix);
}

/**
 * Return the offset from the chip centre, in mm, given a pixel position
 */
afwGeom::Point2D camGeom::Ccd::getPositionFromIndex(
        afwGeom::Point2I pix            ///< Pixel coordinates wrt bottom left of Ccd
                                                        ) const
{
    if (isTrimmed()) {
        return camGeom::Detector::getPositionFromIndex(pix);
    }

    lsst::afw::geom::Point2I centerPixel = getCenterPixel();
    double pixelSize = getPixelSize();

    camGeom::Amp amp = getAmp(afwGeom::Point2I::makeXY(pix[0], pix[1]));
    {
        afwImage::PointI off = amp.getDataSec(true).getLLC() - amp.getDataSec(false).getLLC();
        pix += afwGeom::Extent2I(afwGeom::Point2I::makeXY(off[0], off[1]));
    }

    Eigen::Vector2d pos;
    pos << (pix[0] - centerPixel[0])*pixelSize, (pix[1] - centerPixel[1])*pixelSize;
    return afwGeom::Point2D(pos);
}    

namespace {
    struct findById {
        findById(camGeom::Id id) : _id(id) {}
        bool operator()(camGeom::Amp const& amp) const {
            return _id == amp.getId();
        }
    private:
        camGeom::Id _id;
    };

    struct findByPos {
        findByPos(
                  afwGeom::Point2I point,
                  bool isTrimmed
                 ) :
            _point(afwImage::PointI(point[0], point[1])),
            _isTrimmed(isTrimmed)
        { }

        bool operator()(camGeom::Amp const& amp) const {
            return amp.getAllPixels(_isTrimmed).contains(_point);
        }
    private:
        afwImage::PointI _point;
        bool _isTrimmed;
    };
}

/// Set the trimmed status of this Ccd
void camGeom::Ccd::setTrimmed(bool isTrimmed ///< True iff the bias/overclock have been removed
                                     ) {
    camGeom::Detector::setTrimmed(isTrimmed);
    // And the Amps too
    for (iterator ptr = begin(); ptr != end(); ++ptr) {
        (*ptr).setTrimmed(isTrimmed);
    }
}

/**
 * Find an Amp given an Id
 */
camGeom::Amp camGeom::Ccd::getAmp(camGeom::Id const id) const {
    camGeom::Ccd::AmpSet::const_iterator result = std::find_if(_amps.begin(), _amps.end(), findById(id));
    if (result == _amps.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Amp with serial %||") % id).str());
    }
    return *result;
}

/**
 * Find an Amp given a position
 */
camGeom::Amp camGeom::Ccd::getAmp(afwGeom::Point2I const& pixel) const {
    return getAmp(pixel, isTrimmed());
}

/**
 * Find an Amp given a position and a request for trimmed or untrimmed coordinates
 */
camGeom::Amp camGeom::Ccd::getAmp(afwGeom::Point2I const& pixel,
                                  bool const isTrimmed) const {
    AmpSet::const_iterator result = std::find_if(_amps.begin(), _amps.end(), findByPos(pixel, isTrimmed));
    if (result == _amps.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Amp containing pixel (%d, %d)") %
                           pixel.getX() % pixel.getY()).str());
    }
    return *result;
}
