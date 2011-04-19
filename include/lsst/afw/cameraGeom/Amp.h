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
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Defect.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"

/**
 * @file
 *
 * Describe an amplifier
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

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
private:
    float _gain;                        // Amplifier's gain
    float _readNoise;                   // Amplifier's read noise (units? ADU)
    float _saturationLevel;             // Amplifier's saturation level. N.b. float in case we scale data
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

    explicit Amp(Id id, lsst::afw::geom::Box2I const& allPixels,
                 lsst::afw::geom::Box2I const& biasSec, lsst::afw::geom::Box2I const& dataSec,
                 ReadoutCorner readoutCorner, ElectronicParams::Ptr eparams);

    ~Amp() {}

    void shift(int dx, int dy);
    void rotateBy90(lsst::afw::geom::Extent2I const& dimensions, int n90);

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

    /// Return amplifier's data section
    lsst::afw::geom::Box2I& getDataSec(bool getTrimmed=false) {
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

    /// Set the origin of Amplifier data when on disk (in Detector coordinates)
    void setDiskLayout(
            lsst::afw::geom::Point2I const& originOnDisk, // Origin of Amp data on disk (in Detector coords)
            int nQuarter,                         // number of quarter-turns in +ve direction
            bool flipLR,                          // Flip the Amp data left <--> right before rotation
            bool flipTB                           // Flip the Amp data top <--> bottom before rotation
                      ) {
        _originOnDisk = originOnDisk;
        _nQuarter = nQuarter;
        _flipLR = flipLR;
        _flipTB = flipTB;
    }

    /// Return the biasSec as read from disk
    lsst::afw::geom::Box2I getDiskBiasSec() const {
        return _mapToDisk(getBiasSec(false));
    }

    /// Return the dataSec as read from disk
    lsst::afw::geom::Box2I getDiskDataSec() const {
        return _mapToDisk(getDataSec(false));
    }

    /// Return the biasSec as read from disk
    lsst::afw::geom::Box2I getDiskAllPixels() const {
        return _mapToDisk(getAllPixels(false));
    }

    template<typename ImageT>
    typename ImageT::Ptr prepareAmpData(ImageT const& im);
    
private:
    lsst::afw::geom::Box2I _biasSec;    // Bounding box of amplifier's bias section
    lsst::afw::geom::Box2I _dataSec;    // Bounding box of amplifier's data section
    ReadoutCorner _readoutCorner;       // location of first pixel read
    ElectronicParams::Ptr _eParams;     // electronic properties of Amp
    lsst::afw::geom::Box2I _trimmedDataSec; // Bounding box of all the Detector's pixels after bias trimming
    //
    // These values refer to the way that the Amplifier data is laid out on disk.  If the Amps have
    // been assembled into a single Ccd image _originOnDisk == (0, 0) and _nQuarter == 0
    //
    lsst::afw::geom::Point2I _originOnDisk;     // Origin of Amplifier data on disk (in Detector coordinates)
    int _nQuarter;                      // number of quarter-turns to apply in +ve direction
    bool _flipLR;                       // flip the data left <--> right before rotation
    bool _flipTB;                       // Flip the data top <--> bottom before rotation

    lsst::afw::geom::Box2I _mapToDisk(lsst::afw::geom::Box2I bbox) const;
};
    
}}}

#endif
