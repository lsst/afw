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
 
#if !defined(LSST_AFW_CAMERAGEOM_AMP_H)
#define LSST_AFW_CAMERAGEOM_AMP_H

#include <string>
#include <limits>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Defect.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/Wcs.h"

/**
 * @file
 *
 * Describe an amplifier
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

struct Linearity {
    /// How to correct for non-linearity
    enum LinearityType {
        PROPORTIONAL                    ///< Correction is proportional to flux
    };
    /**
     * An object to describe an amp's non-linear behaviour
     *
     * \note
     * This is default constructible solely so that we don't have to change
     * all camera .paf files -- for now, we'll set the coefficients in a camera's mapper.
     * This is not a good solution, and should be changed when we move away from using
     * .paf files to initialise Cameras
     */
    explicit Linearity(LinearityType type_ = PROPORTIONAL, ///< Type of correction to apply
                       float threshold_=0,    ///< DN where non-linear response commences
                       /// Maximum DN where non-linearity correction is believable
                       float maxCorrectable_=std::numeric_limits<float>::max(),
                       float coefficient_ = 0.0 ///< Coefficient for linearity correction
                      ) : type(type_), threshold(threshold_), maxCorrectable(maxCorrectable_),
                          coefficient(coefficient_) {}

    LinearityType type;                 // Type of correction to apply
    float threshold;                    // DN where non-linear response commences
    float maxCorrectable;               // Maximum DN where non-linearity correction is believable
    float coefficient;                  // Coefficient for linearity correction
};

/**
 * The electronic behaviour of an Amp
 */
class ElectronicParams {
public:
    typedef boost::shared_ptr<ElectronicParams> Ptr;
    typedef boost::shared_ptr<const ElectronicParams> ConstPtr;

    explicit ElectronicParams(float gain, float readNoise, float saturationLevel);
    virtual ~ElectronicParams() {}
#if 0
    /// Set the bad pixels in the provided Mask
    template<typename MaskPixelT>
    lsst::afw::image::Mask::Ptr setBadPixelMask(
        typename lsst::afw::image::Mask<MaskPixelT>::Ptr mask ///< Mask to set
                                                ) const;
#endif

    /// Set amplifier's gain
    void setGain(float const gain       ///< Amplifier's gain
                ) {
        _gain = gain;
    }
    /// Return amplifier's gain
    float getGain() const {
        return _gain;
    }

    /// Set amplifier's read noise
    void setReadNoise(float const readNoise ///< Amplifier's read noise
                     ) {
        _readNoise = readNoise;
    }
    /// Return amplifier's read noise
    float getReadNoise() const {
        return _readNoise;
    }

    /// Set amplifier's saturation level
    void setSaturationLevel(float const saturationLevel ///< Amplifier's saturation level
                           ) {
        _saturationLevel = saturationLevel;
    }
    /// Return amplifier's saturation level
    float getSaturationLevel() const {
        return _saturationLevel;
    }
    /// Set the description of the Amp's non-linearity
    void setLinearity(Linearity const &linearity ///< How to correct for non-linearity
                     ) { _linearity = linearity; }
    /// Return the description of the Amp's non-linearity
    Linearity getLinearity() const { return _linearity; }

private:
    float _gain;                        // Amplifier's gain
    float _readNoise;                   // Amplifier's read noise (units? ADU)
    float _saturationLevel;             // Amplifier's saturation level. N.b. float in case we scale data
    Linearity _linearity;               // Amplifier's non-linearity
};
    
/**
 * An amplifier; a set of pixels read out through a particular amplifier
 */
class Amp : public Detector {
public:
    typedef boost::shared_ptr<Amp> Ptr;
    typedef boost::shared_ptr<const Amp> ConstPtr;

    /// location of first pixel read
    // N.b. must be in the order that a rotation by 90 shift right by one;  LLC -> ULC 
    enum ReadoutCorner { LLC, LRC, URC, ULC };
    enum DiskCoordSys { AMP, SENSOR, CAMERA };

    explicit Amp(Id id, lsst::afw::geom::Box2I const& allPixels,
                 lsst::afw::geom::Box2I const& biasSec, lsst::afw::geom::Box2I const& dataSec,
                 ElectronicParams::Ptr eparams);

    ~Amp() {}

    void shift(int dx, int dy);
    void rotateBy90(lsst::afw::geom::Extent2I const& dimensions, int n90);
    void prepareWcsData(lsst::afw::image::Wcs::Ptr wcs);

    /// Return Amp's electronic properties
    ElectronicParams::Ptr getElectronicParams() const { return _eParams; }

    /// Return amplifier's bias section
    lsst::afw::geom::Box2I getBiasSec() const {
        return getBiasSec(isTrimmed());
    }

    /// Return amplifier's bias section
    lsst::afw::geom::Box2I getBiasSec(bool isTrimmed // has the bias/overclock been removed?
                                     ) const {
        return isTrimmed ? lsst::afw::geom::Box2I() : _biasSec;
    }

    /// Return amplifier's data section
    lsst::afw::geom::Box2I getDataSec() const {
        return getDataSec(isTrimmed());
    }

    lsst::afw::geom::Box2I getDataSec(bool getTrimmed) const {
        return getTrimmed ? _trimmedDataSec : _dataSec;
    }

    /// Return the corner that's read first
    ReadoutCorner getReadoutCorner() const { return _readoutCorner; }

    /// Return the first pixel read
    lsst::afw::geom::Point2I getFirstPixelRead() const {
        switch (_readoutCorner) {
          case LLC:
            return lsst::afw::geom::Point2I(0,                             0);
          case LRC:
            return lsst::afw::geom::Point2I(getAllPixels().getWidth() - 1, 0);
          case URC:
            return lsst::afw::geom::Point2I(getAllPixels().getWidth() - 1, getAllPixels().getHeight() - 1);
          case ULC:
            return lsst::afw::geom::Point2I(0,                             getAllPixels().getHeight() - 1);
        }
        abort();                        // NOTREACHED
    }

    void setTrimmedGeom();
    /// Define the position and orientation of an Amp in a chip (in Detector coordinates)
    void setElectronicToChipLayout(lsst::afw::geom::Point2I, int, bool, DiskCoordSys);

    /// Return the biasSec in electronic coordinates
    lsst::afw::geom::Box2I getElectronicBiasSec() const {
        return _mapToElectronic(getBiasSec(false));
    }

    /// Return the dataSec in electronic coordinates
    lsst::afw::geom::Box2I getElectronicDataSec() const {
        return _mapToElectronic(getDataSec(false));
    }

    /// Return all pixels in electronic coordinates
    lsst::afw::geom::Box2I getElectronicAllPixels() const {
        return _mapToElectronic(getAllPixels(false));
    }

    /// Return the biasSec in as stored coordinates
    lsst::afw::geom::Box2I getDiskBiasSec() const {
        return _mapToDisk(getBiasSec(false));
    }

    /// Return the dataSec in as stored coordinates
    lsst::afw::geom::Box2I getDiskDataSec() const {
        return _mapToDisk(getDataSec(false));
    }

    /// Return all pixels in as stored coordinates
    lsst::afw::geom::Box2I getDiskAllPixels() const {
        return _mapToDisk(getAllPixels(false));
    }

    /// Return the pixel coordinate system on disk
    DiskCoordSys getDiskCoordSys() const { return _diskCoordSys; }

    template<typename ImageT>
    const ImageT prepareAmpData(ImageT const im);
    
private:
    lsst::afw::geom::Box2I _biasSec;    // Bounding box of amplifier's bias section
    lsst::afw::geom::Box2I _dataSec;    // Bounding box of amplifier's data section
    ReadoutCorner _readoutCorner;       // location of first pixel read
    ElectronicParams::Ptr _eParams;     // electronic properties of Amp
    lsst::afw::geom::Box2I _trimmedDataSec; // Bounding box of all the Detector's pixels after bias trimming
    //
    // These values refer to the way that the Amplifier data is laid out in the sensor. 
    // The parent CCD object caries the information about how the sensor was actually installed in the camera.
    //
    lsst::afw::geom::Point2I _originInDetector;     // Origin of Amplifier data on disk (in Detector coordinates)
    int _nQuarter;                      // number of quarter-turns to apply in +ve direction
    bool _flipLR;                       // flip the data left <--> right before rotation
    // The way the pixel data is stored on disk varies from one project to the next.  The three obvious ones are:
    // amp: All images are a single amp in electronic coordinates (all images from identical devices look the same). -- ImSim images
    // sensor: Pixels are assembled into a grid relative to the (0,0) index sensor pixel.  CFHTLS images
    // camera: Pixels are assembled into a sensor grid and rotated to reflect rotation as installed in the camera.  SuprimeCam images
    DiskCoordSys _diskCoordSys;

    lsst::afw::geom::Box2I _mapToElectronic(lsst::afw::geom::Box2I bbox) const;
    lsst::afw::geom::Box2I _mapFromElectronic(lsst::afw::geom::Box2I bbox) const;
    lsst::afw::geom::Box2I _mapToDisk(lsst::afw::geom::Box2I bbox) const;
};
    
}}}

#endif
