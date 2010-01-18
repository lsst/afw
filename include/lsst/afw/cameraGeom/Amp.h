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

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
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
    lsst::afw::image::Mask::Ptr setBadPixelMask(typename lsst::afw::image::Mask<MaskPixelT>::Ptr mask ///< Mask to set
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

    explicit Amp(Id id, lsst::afw::image::BBox const& allPixels,
                 lsst::afw::image::BBox const& biasSec, lsst::afw::image::BBox const& dataSec,
                 ReadoutCorner readoutCorner, ElectronicParams::Ptr eparams);

    ~Amp() {}
    /// Are two Amps identical?
    bool operator==(Amp const& rhs      ///< Amp to compare too
                   ) const {
        return getId() == rhs.getId();
    }

    void shift(int dx, int dy);
    void rotateBy90(afwGeom::Extent2I const& dimensions, int n90);

    /// Return Amp's electronic properties
    ElectronicParams::Ptr getElectronicParams() const { return _eParams; }

    /// Return amplifier's bias section
    lsst::afw::image::BBox getBiasSec() const {
        return getBiasSec(isTrimmed());
    }

    /// Return amplifier's bias section
    lsst::afw::image::BBox getBiasSec(bool isTrimmed // has the bias/overclock been removed?
                                     ) const {
        return isTrimmed ? lsst::afw::image::BBox() : _biasSec;
    }

    /// Return amplifier's data section
    lsst::afw::image::BBox getDataSec() const {
        return getDataSec(isTrimmed());
    }

    lsst::afw::image::BBox getDataSec(bool getTrimmed) const {
        return getTrimmed ? _trimmedDataSec : _dataSec;
    }

    /// Return amplifier's data section
    lsst::afw::image::BBox& getDataSec(bool getTrimmed=false) {
        return getTrimmed ? _trimmedDataSec : _dataSec;
    }

    /// Return the corner that's read first
    ReadoutCorner getReadoutCorner() const { return _readoutCorner; }

    /// Return the first pixel read
    afwGeom::Point2I getFirstPixelRead() const {
        switch (_readoutCorner) {
          case LLC:
            return afwGeom::makePointI(0,                             0);
          case LRC:
            return afwGeom::makePointI(getAllPixels().getWidth() - 1, 0);
          case URC:
            return afwGeom::makePointI(getAllPixels().getWidth() - 1, getAllPixels().getHeight() - 1);
          case ULC:
            return afwGeom::makePointI(0,                             getAllPixels().getHeight() - 1);
        }
        abort();                        // NOTREACHED
    }

    void setTrimmedGeom();
private:
    lsst::afw::image::BBox _biasSec;    // Bounding box of amplifier's bias section
    lsst::afw::image::BBox _dataSec;    // Bounding box of amplifier's data section
    ReadoutCorner _readoutCorner;       // location of first pixel read
    ElectronicParams::Ptr _eParams;     // electronic properties of Amp
    lsst::afw::image::BBox _trimmedDataSec; // Bounding box of all the Detector's pixels after bias trimming
};
    
}}}

#endif
