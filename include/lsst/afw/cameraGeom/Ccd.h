#if !defined(LSST_AFW_CAMERAGEOM_CCD_H)
#define LSST_AFW_CAMERAGEOM_CCD_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/Amp.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a CCD, containing a number of Amp%s
 */
class Ccd : public Detector {
public:
    typedef boost::shared_ptr<Ccd> Ptr;
    typedef boost::shared_ptr<const Ccd> ConstPtr;

    typedef std::vector<Amp::Ptr> AmpSet;
#if 0                                   // N.b. don't say "AmpSet::iterator" for swig's sake
    typedef ampSet::iterator iterator;
#else
    typedef std::vector<boost::shared_ptr<Amp> >::iterator iterator;
#endif
    typedef std::vector<Amp::Ptr>::const_iterator const_iterator;

    Ccd(Id id, double pixelSize=0.0) : Detector(id, true, pixelSize) {}
    virtual ~Ccd() {}
    //
    // Provide iterators for all the Ccd's Amps
    //
    iterator begin() { return _amps.begin(); }
    const_iterator begin() const { return _amps.begin(); }
    iterator end() { return _amps.end(); }
    const_iterator end() const { return _amps.end(); }
    //
    // Add an Amp to the Ccd
    //
    void addAmp(int const iX, int const iY, Amp const& amp);

    virtual void setTrimmed(bool isTrimmed);
    //
    // Find an Amp given an Id or pixel position
    //
    Amp::Ptr findAmp(Id const id) const;
    Amp::Ptr findAmp(lsst::afw::geom::Point2I const& pixel) const;
    Amp::Ptr findAmp(lsst::afw::geom::Point2I const& pixel, bool const isTrimmed) const;
    //
    // Translate between physical positions in mm to pixels
    //
    virtual lsst::afw::geom::Point2I getIndexFromPosition(lsst::afw::geom::Point2D pos) const;
    virtual lsst::afw::geom::Point2D getPositionFromIndex(lsst::afw::geom::Point2I pos) const;

    virtual void shift(int dx, int dy);
private:
    AmpSet _amps;                       // the Amps that make up this Ccd
};
    
}}}

#endif
