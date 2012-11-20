// -*- LSST-C++ -*- // fixed format comment for emacs
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

#ifndef LSST_AFW_IMAGE_ExposureInfo_h_INCLUDED
#define LSST_AFW_IMAGE_ExposureInfo_h_INCLUDED

#include "lsst/base.h"
#include "lsst/daf/base.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image/Filter.h"

namespace lsst { namespace afw {

namespace cameraGeom {
class Detector;
}

namespace detection {
class Psf;
}

namespace fits {
class Fits;
}

namespace image {

class Calib;
class Wcs;

/**
 *  @brief A collection of all the things that make an Exposure different from a MaskedImage
 *
 *  The constness semantics of the things held by ExposureInfo are admittedly a bit of a mess,
 *  but they're that way to preserve backwards compatibility for now.  Eventually I'd like to make
 *  a lot of these things immutable, but in the meantime, here's the summary:
 *   - Wcs, Calib, and Psf are held by non-const pointer, and you can get a non-const pointer via a
 *     non-const member function accessor and a const pointer via a const member function accessor.
 *   - Detector is held by const pointer and only returned by const pointer (but if you're
 *     in Python, SWIG will have casted all that constness away).
 *   - Filter is held and returned by value.
 *   - Metadata is held by non-const pointer, and you can get a non-const pointer via a const
 *     member function accessor (i.e. constness is not propagated).
 *
 *  The setters for Wcs, Calib, and Psf all clone their input arguments (this is a departure
 *  from the previous behavior for Calib and Wcs, but not Psf, but it's safer w.r.t. aliasing
 *  and it matches the old (and current) behavior of the Exposure and ExposureInfo constructors,
 *  which clone their arguments.  The setter for Detector does *not* clone its input argument,
 *  because while it technically isn't, we can safely consider a Detector to be immutable once
 *  it's attached to an ExposureInfo.
 */
class ExposureInfo {
public:

    /// Does this exposure have a Wcs?
    bool hasWcs() const { return static_cast<bool>(_wcs); }
    
    /// Return the coordinate system of the exposure
    PTR(Wcs) getWcs() { return _wcs; }
    
    /// Return the coordinate system of the exposure
    CONST_PTR(Wcs) getWcs() const { return _wcs; }

    /// Set the coordinate system of the exposure
    void setWcs(CONST_PTR(Wcs) wcs) { _wcs = _cloneWcs(wcs); }

    /// Does this exposure have Detector information?
    bool hasDetector() const { return static_cast<bool>(_detector); }

    /// Return the exposure's Detector information
    CONST_PTR(cameraGeom::Detector) getDetector() const { return _detector; }

    /// Set the exposure's Detector information
    void setDetector(CONST_PTR(cameraGeom::Detector) detector) { _detector = detector; }

    /// Return the exposure's filter
    Filter getFilter() const { return _filter; }

    /// Set the exposure's filter
    void setFilter(Filter const& filter) { _filter = filter; }

    /// Does this exposure have a Calib?
    bool hasCalib() const { return static_cast<bool>(_calib); }

    /// Return the exposure's photometric calibration
    PTR(Calib) getCalib() { return _calib; }

    /// Return the exposure's photometric calibration
    CONST_PTR(Calib) getCalib() const { return _calib; }

    /// Set the Exposure's Calib object
    void setCalib(CONST_PTR(Calib) calib) { _calib = _cloneCalib(calib); }

    /// Return flexible metadata
    PTR(daf::base::PropertySet) getMetadata() const { return _metadata; }

    /// Set the flexible metadata
    void setMetadata(PTR(daf::base::PropertySet) metadata) { _metadata = metadata; }
    
    /// Does this exposure have a Psf?
    bool hasPsf() const { return static_cast<bool>(_psf); }

    /// Return the exposure's point-spread function
    PTR(detection::Psf) getPsf() { return _psf; }

    /// Return the exposure's point-spread function
    CONST_PTR(detection::Psf) getPsf() const { return _psf; }

    /// Set the exposure's point-spread function
    void setPsf(CONST_PTR(detection::Psf) psf) { _psf = _clonePsf(psf); }

    /**
     *  @brief Construct an ExposureInfo from its various components.
     *
     *  If a null Calib and/or PropertySet pointer is passed (the default),
     *  a new Calib and/or PropertyList will be created.  To set these pointers
     *  to null, you must explicitly call setCalib or setMetadata after construction.
     */
    explicit ExposureInfo(
        CONST_PTR(Wcs) const & wcs = CONST_PTR(Wcs)(),
        CONST_PTR(detection::Psf) const & psf = CONST_PTR(detection::Psf)(),
        CONST_PTR(Calib) const & calib = CONST_PTR(Calib)(),
        CONST_PTR(cameraGeom::Detector) const & detector = CONST_PTR(cameraGeom::Detector)(),
        Filter const & filter = Filter(),
        PTR(daf::base::PropertySet) const & metadata = PTR(daf::base::PropertySet)()
    );

    /// Copy constructor; deep-copies all components except the metadata.
    ExposureInfo(ExposureInfo const & other);

    /// Copy constructor; deep-copies everything, possibly including the metadata.
    ExposureInfo(ExposureInfo const & other, bool copyMetadata);

    /// Assignment; deep-copies all components except the metadata.
    ExposureInfo & operator=(ExposureInfo const & other);

    // Destructor defined in source file because we need access to destructors of forward-declared components
    ~ExposureInfo();

    /**
     *  @brief Generate the metadata that saves some components of the ExposureInfo to a FITS header.
     *
     *  This returns a pair of PropertySets; the first contains the metadata that is generally written
     *  to the main image HDU and is read to reconstruct the ExposureInfo, while the second contains
     *  additional metadata intended for the mask and variance headers.
     *
     *  FITS persistence is separated into getFitsMetadata() and writeFitsHdus() so that
     *  the Primary FITS header can be at least mostly written before the main image HDUs
     *  are written, while the additional ExposureInfo HDUs are written afterwards. This
     *  is desirable in order to reduce the chance that we'll have to shift the images on
     *  disk in order to make space for addition header entries.
     *
     *  @param[in]  hdu   The number of the HDU that will hold the main metadata.  Used
     *                    to compute which HDUs will be used for non-MaskedImage components
     *                    (such as the Psf), assuming three MaskedImage planes.
     *  @param[in]  xy0   The origin of the exposure associated with this object, used to
     *                    install a linear offset-only WCS in the FITS header.
     */
    std::pair<PTR(daf::base::PropertyList),PTR(daf::base::PropertyList)>
    getFitsMetadata(int hdu, geom::Point2I const & xy0=geom::Point2I()) const;

    /**
     *  @brief Write any additional non-image HDUs to a FITS file.
     *
     *  @param[in]  fitsfile   Open FITS object to write to.  Does not need to be positioned to any
     *                         particular HDU.
     *
     *  The additional HDUs will be appended to the FITS file, and should line up with the HDU index
     *  keys included in the result of getFitsMetadata() if this is called after writing the
     *  MaskedImage HDUs.
     */
    void writeFitsHdus(fits::Fits & fitsfile) const;

    /**
     *  @brief Read from a FITS file and metadata.
     *
     *  This operates in-place on this instead of returning a new object, because it will usually
     *  only be called by the exposure constructor, which starts by default-constructing the
     *  ExposureInfo.
     */
    void readFits(fits::Fits & fitsfile, PTR(daf::base::PropertySet) metadata);

private:

    static PTR(detection::Psf) _clonePsf(CONST_PTR(detection::Psf) psf);
    static PTR(Calib) _cloneCalib(CONST_PTR(Calib) calib);
    static PTR(Wcs) _cloneWcs(CONST_PTR(Wcs) wcs);

    PTR(Wcs) _wcs;
    PTR(detection::Psf) _psf;
    PTR(Calib) _calib;
    CONST_PTR(cameraGeom::Detector) _detector;
    Filter _filter;
    PTR(daf::base::PropertySet) _metadata;
};

}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_EXPOSURE_H
