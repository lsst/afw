// -*- LSST-C++ -*- // fixed format comment for emacs
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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
#include "lsst/geom/Point.h"
#include "lsst/afw/image/Filter.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/image/CoaddInputs.h"
#include "lsst/afw/image/VisitInfo.h"
#include "lsst/afw/typehandling/GenericMap.h"

namespace lsst {
namespace afw {

namespace cameraGeom {
class Detector;
}

namespace detection {
class Psf;
}

namespace geom {
namespace polygon {
class Polygon;
class SkyWcs;
}  // namespace polygon
}  // namespace geom

namespace fits {
class Fits;
}

namespace image {

class PhotoCalib;
class ApCorrMap;
class VisitInfo;
class TransmissionCurve;

/**
 *  A collection of all the things that make an Exposure different from a MaskedImage
 *
 *  The constness semantics of the things held by ExposureInfo are admittedly a bit of a mess,
 *  but they're that way to preserve backwards compatibility for now.  Eventually I'd like to make
 *  a lot of these things immutable, but in the meantime, here's the summary:
 *   - Filter is held and returned by value.
 *   - VisitInfo is immutable and is held by a const ptr and has a setter and getter.
 *   - Metadata is held by non-const pointer, and you can get a non-const pointer via a const
 *     member function accessor (i.e. constness is not propagated).
 *   - all other types are only accessible through non-const pointers
 *
 *  The setter for Wcs clones its input arguments (this is a departure from the
 *  previous behavior for Wcs but it's safer w.r.t. aliasing and it matches the old
 *  (and current) behavior of the Exposure and ExposureInfo constructors, which clone their
 *  arguments.  The setter for Psf and constructors do not clone the Psf, as Psfs are immutable
 *  and hence we don't need to ensure strict ownership.  The setter for Detector does *not*
 *  clone its input argument, because while it technically isn't, we can safely consider a
 *  Detector to be immutable once it's attached to an ExposureInfo.
 */
class ExposureInfo final {
public:
    /// Standard key for looking up the Wcs.
    static typehandling::Key<std::string, std::shared_ptr<geom::SkyWcs const>> const KEY_WCS;
    /// Standard key for looking up the point-spread function.
    static typehandling::Key<std::string, std::shared_ptr<detection::Psf const>> const KEY_PSF;
    /// Standard key for looking up the photometric calibration.
    static typehandling::Key<std::string, std::shared_ptr<PhotoCalib const>> const KEY_PHOTO_CALIB;
    /// Standard key for looking up the detector information.
    static typehandling::Key<std::string, std::shared_ptr<cameraGeom::Detector const>> const KEY_DETECTOR;

    /// Does this exposure have a Wcs?
    bool hasWcs() const;

    /// Return the WCS of the exposure
    std::shared_ptr<geom::SkyWcs const> getWcs() const;

    /// Set the WCS of the exposure
    void setWcs(std::shared_ptr<geom::SkyWcs const> wcs);

    /// Does this exposure have Detector information?
    bool hasDetector() const;

    /// Return the exposure's Detector information
    std::shared_ptr<cameraGeom::Detector const> getDetector() const;

    /// Set the exposure's Detector information
    void setDetector(std::shared_ptr<cameraGeom::Detector const> detector);

    /// Return the exposure's filter
    Filter getFilter() const { return _filter; }

    /// Set the exposure's filter
    void setFilter(Filter const& filter) { _filter = filter; }

    /// Does this exposure have a photometric calibration?
    bool hasPhotoCalib() const;

    /// Return the exposure's photometric calibration
    std::shared_ptr<PhotoCalib const> getPhotoCalib() const;

    /// Set the Exposure's PhotoCalib object
    void setPhotoCalib(std::shared_ptr<PhotoCalib const> photoCalib);

    /// Does this exposure have a photometric calibration?
    [[deprecated("Replaced with hasPhotoCalib (will be removed in 18.0)")]] bool hasCalib() const {
        return hasPhotoCalib();
    }

    /// Return the exposure's photometric calibration
    [[deprecated("Replaced with getPhotoCalib (will be removed in 18.0)")]] std::shared_ptr<PhotoCalib const>
    getCalib() const {
        return getPhotoCalib();
    }

    /// Set the Exposure's PhotoCalib object
    [[deprecated("Replaced with setPhotoCalib (will be removed in 18.0)")]] void setCalib(
            std::shared_ptr<PhotoCalib const> photoCalib) {
        setPhotoCalib(photoCalib);
    }

    /// Return flexible metadata
    std::shared_ptr<daf::base::PropertySet> getMetadata() const { return _metadata; }

    /// Set the flexible metadata
    void setMetadata(std::shared_ptr<daf::base::PropertySet> metadata) { _metadata = metadata; }

    /// Does this exposure have a Psf?
    bool hasPsf() const;

    /// Return the exposure's point-spread function
    std::shared_ptr<detection::Psf const> getPsf() const;

    /// Set the exposure's point-spread function
    void setPsf(std::shared_ptr<detection::Psf const> psf);

    /// Does this exposure have a valid Polygon
    bool hasValidPolygon() const { return static_cast<bool>(_validPolygon); }

    /// Return the valid Polygon
    std::shared_ptr<geom::polygon::Polygon const> getValidPolygon() const { return _validPolygon; }

    /// Set the exposure's valid Polygon
    void setValidPolygon(std::shared_ptr<geom::polygon::Polygon const> polygon) { _validPolygon = polygon; }

    /// Return true if the exposure has an aperture correction map
    bool hasApCorrMap() const { return static_cast<bool>(_apCorrMap); }

    /// Return the exposure's aperture correction map (null pointer if !hasApCorrMap())
    std::shared_ptr<ApCorrMap> getApCorrMap() { return _apCorrMap; }

    /// Return the exposure's aperture correction map (null pointer if !hasApCorrMap())
    std::shared_ptr<ApCorrMap const> getApCorrMap() const { return _apCorrMap; }

    /// Set the exposure's aperture correction map (null pointer if !hasApCorrMap())
    void setApCorrMap(std::shared_ptr<ApCorrMap> apCorrMap) { _apCorrMap = apCorrMap; }

    /**
     *  Set the exposure's aperture correction map to a new, empty map
     *
     *  Note that the ExposureInfo constructors do not create an empty aperture correction map,
     *  so this method provide a convenient way to initialize one before filling it.
     */
    void initApCorrMap();

    /// Does this exposure have coadd provenance catalogs?
    bool hasCoaddInputs() const { return static_cast<bool>(_coaddInputs); }

    /// Set the exposure's coadd provenance catalogs.
    void setCoaddInputs(std::shared_ptr<CoaddInputs> coaddInputs) { _coaddInputs = coaddInputs; }

    /// Return a pair of catalogs that record the inputs, if this Exposure is a coadd (otherwise null).
    std::shared_ptr<CoaddInputs> getCoaddInputs() const { return _coaddInputs; }

    /// Return the exposure's visit info
    std::shared_ptr<image::VisitInfo const> getVisitInfo() const { return _visitInfo; }

    /// Does this exposure have visit info?
    bool hasVisitInfo() const { return static_cast<bool>(_visitInfo); }

    /// Set the exposure's visit info
    void setVisitInfo(std::shared_ptr<image::VisitInfo const> const visitInfo) { _visitInfo = visitInfo; }

    /// Does this exposure have a transmission curve?
    bool hasTransmissionCurve() const { return static_cast<bool>(_transmissionCurve); }

    /// Return the exposure's transmission curve.
    std::shared_ptr<TransmissionCurve const> getTransmissionCurve() const { return _transmissionCurve; }

    /// Set the exposure's transmission curve.
    void setTransmissionCurve(std::shared_ptr<TransmissionCurve const> tc) { _transmissionCurve = tc; }

    /**
     * Add a generic component to the ExposureInfo.
     *
     * If another component is already present under the key, it is
     * overwritten. If a component of a different type is present under the
     * same name, this method raises an exception.
     *
     * @tparam T a subclass of typehandling::Storable
     *
     * @param key a strongly typed identifier for the component
     * @param object the object to add.
     *
     * @throws pex::exceptions::TypeError Thrown if a component of a
     *      different type is present under the requested name.
     * @throws pex::exceptions::RuntimeError Thrown if the insertion failed for
     *      implementation-dependent reasons.
     * @exceptsafe Provides basic exception safety (a pre-existing component
     *             may be removed).
     *
     * @note if `object` is a null pointer, then `hasComponent(key)` shall
     *       return `false` after this method returns. This is for
     *       compatibility with old ExposureInfo idioms, which often use
     *       assignment of null to indicate no data.
     */
    // non-shared_ptr components are incompatible with table::io,
    // they may be supported later
    template <class T>
    void setComponent(typehandling::Key<std::string, std::shared_ptr<T>> const& key,
                      std::shared_ptr<T> const& object) {
        static_assert(std::is_base_of<typehandling::Storable, T>::value, "T must be a Storable");
        // "No data" always represented internally by absent key-value pair, not by mapping to null
        if (object != nullptr) {
            _setStorableComponent(key, object);
        } else {
            removeComponent(key);
        }
    }

    /**
     * Test whether a generic component is defined.
     *
     * @param key a strongly typed identifier for the component
     * @return `true` if there is a component with `key`, `false` otherwise
     *
     * @exceptsafe Provides strong exception safety.
     */
    template <class T>
    bool hasComponent(typehandling::Key<std::string, T> const& key) const {
        return _components->contains(key);
    }

    /**
     * Retrieve a generic component from the ExposureInfo.
     *
     * @param key a strongly typed identifier for the component
     * @return the component identified by that key, or a null pointer if no
     *         such component exists.
     *
     * @exceptsafe Provides strong exception safety.
     */
    template <class T>
    std::shared_ptr<T> getComponent(typehandling::Key<std::string, std::shared_ptr<T>> const& key) const {
        try {
            return _components->at(key);
        } catch (pex::exceptions::OutOfRangeError const& e) {
            return nullptr;
        }
    }

    /**
     * Clear a generic component from the ExposureInfo.
     *
     * @param key a strongly typed identifier for the component. Only
     *            components of a compatible type are removed.
     * @returns `true` if a component was removed, `false` otherwise.
     *
     * @exceptsafe Provides strong exception safety.
     */
    template <class T>
    bool removeComponent(typehandling::Key<std::string, T> const& key) {
        return _components->erase(key);
    }

    /// Get the version of FITS serialization that this ExposureInfo understands.
    static int getFitsSerializationVersion();

    /// Get the version of FITS serialization version info name
    static std::string const& getFitsSerializationVersionName();

    /**
     *  Construct an ExposureInfo from its various components.
     *
     *  If a null PhotoCalib and/or PropertySet pointer is passed (the default),
     *  a new PhotoCalib and/or PropertyList will be created.  To set these pointers
     *  to null, you must explicitly call setPhotoCalib or setMetadata after construction.
     */
    explicit ExposureInfo(
            std::shared_ptr<geom::SkyWcs const> const& wcs = std::shared_ptr<geom::SkyWcs const>(),
            std::shared_ptr<detection::Psf const> const& psf = std::shared_ptr<detection::Psf const>(),
            std::shared_ptr<PhotoCalib const> const& photoCalib = std::shared_ptr<PhotoCalib const>(),
            std::shared_ptr<cameraGeom::Detector const> const& detector =
                    std::shared_ptr<cameraGeom::Detector const>(),
            std::shared_ptr<geom::polygon::Polygon const> const& polygon =
                    std::shared_ptr<geom::polygon::Polygon const>(),
            Filter const& filter = Filter(),
            std::shared_ptr<daf::base::PropertySet> const& metadata =
                    std::shared_ptr<daf::base::PropertySet>(),
            std::shared_ptr<CoaddInputs> const& coaddInputs = std::shared_ptr<CoaddInputs>(),
            std::shared_ptr<ApCorrMap> const& apCorrMap = std::shared_ptr<ApCorrMap>(),
            std::shared_ptr<image::VisitInfo const> const& visitInfo =
                    std::shared_ptr<image::VisitInfo const>(),
            std::shared_ptr<TransmissionCurve const> const& transmissionCurve =
                    std::shared_ptr<TransmissionCurve>());

    /// Copy constructor; shares all components except the filter.
    ExposureInfo(ExposureInfo const& other);
    ExposureInfo(ExposureInfo&& other);

    /// Copy constructor; shares everything but the filter and possibly the metadata.
    ExposureInfo(ExposureInfo const& other, bool copyMetadata);

    /// Assignment; shares all components except the filter.
    ExposureInfo& operator=(ExposureInfo const& other);
    ExposureInfo& operator=(ExposureInfo&& other);

    // Destructor defined in source file because we need access to destructors of forward-declared components
    ~ExposureInfo();

private:
    template <typename ImageT, typename MaskT, typename VarianceT>
    friend class Exposure;

    /**
     *  A struct passed back and forth between Exposure and ExposureInfo when writing FITS files.
     *
     *  An ExposureInfo is generally held by an Exposure, and we implement much of Exposure persistence
     *  here in ExposureInfo.  FITS writing needs to take place in three steps:
     *   1. Exposure calls ExposureInfo::_startWriteFits to generate the image headers in the form of
     *      PropertyLists.  The headers  include archive IDs for the components of ExposureInfo, so we
     *      have to put those in the archive at this time, and transfer the PropertyLists and archive
     *      to the Exposure for the next step.
     *   2. Exposure calls MaskedImage::writeFits to save the Image, Mask, and Variance HDUs along
     *      with the headers.
     *   3. Exposure calls ExposureInfo::_finishWriteFits to save the archive to additional table HDUs.
     */
    struct FitsWriteData {
        std::shared_ptr<daf::base::PropertyList> metadata;
        std::shared_ptr<daf::base::PropertyList> imageMetadata;
        std::shared_ptr<daf::base::PropertyList> maskMetadata;
        std::shared_ptr<daf::base::PropertyList> varianceMetadata;
        table::io::OutputArchive archive;
    };

    /**
     * Add a Persistable object to FITS data.
     *
     * @param[out] data the FITS output data to update
     * @param[in] object the object to store
     * @param[in] key the FITS header keyword to contain a unique ID for the object
     * @param[in] comment the comment for ``key`` in the FITS header
     *
     * @return the unique ID for the object, as stored with ``key``
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if ``key`` contains the "." character.
     * @exceptsafe Does not provide exception safety; ``data`` may be corrupted in
     *             the event of an exception. No other side effects.
     *
     * @{
     */
    static int _addToArchive(FitsWriteData& data, table::io::Persistable const& object, std::string key,
                             std::string comment);

    static int _addToArchive(FitsWriteData& data, std::shared_ptr<table::io::Persistable const> const& object,
                             std::string key, std::string comment);

    /** @} */

    /**
     *  Start the process of writing an exposure to FITS.
     *
     *  @param[in]  xy0   The origin of the exposure associated with this object, used to
     *                    install a linear offset-only WCS in the FITS header.
     *
     *  @see FitsWriteData
     */
    FitsWriteData _startWriteFits(lsst::geom::Point2I const& xy0 = lsst::geom::Point2I()) const;

    /**
     *  Write any additional non-image HDUs to a FITS file.
     *
     *  @param[in]  fitsfile   Open FITS object to write to.  Does not need to be positioned to any
     *                         particular HDU.
     *  @param[in,out] data    The data returned by this object's _startWriteFits method.
     *
     *  The additional HDUs will be appended to the FITS file, and should line up with the HDU index
     *  keys included in the result of wcs.getFitsMetadata() if this is called after writing the
     *  MaskedImage HDUs.
     *
     *  @see FitsWriteData
     */
    void _finishWriteFits(fits::Fits& fitsfile, FitsWriteData const& data) const;

    /**
     * GenericMap visitor for saving Storable objects
     *
     * Consistent with ExposureInfo's original behavior, any Storables that are
     * not persistable are silently ignored.
     */
    class StorablePersister;

    static std::shared_ptr<ApCorrMap> _cloneApCorrMap(std::shared_ptr<ApCorrMap const> apCorrMap);

    // Implementation of setComponent, assumes T extends Storable or T = shared_ptr<? extends Storable>
    template <class T>
    void _setStorableComponent(typehandling::Key<std::string, T> const& key, T const& object) {
        if (_components->contains(key)) {
            _components->erase(key);
        } else if (_components->contains(key.getId())) {
            std::stringstream buffer;
            buffer << "Map has a key that conflicts with " << key;
            throw LSST_EXCEPT(pex::exceptions::TypeError, buffer.str());
        }
        try {
            bool success = _components->insert(key, object);
            if (!success) {
                throw LSST_EXCEPT(
                        pex::exceptions::RuntimeError,
                        "Insertion failed for unknown reasons. There may be something in the logs.");
            }
        } catch (std::exception const& e) {
            std::throw_with_nested(
                    LSST_EXCEPT(pex::exceptions::RuntimeError, "Insertion raised an exception."));
        }
    }

    std::shared_ptr<geom::polygon::Polygon const> _validPolygon;
    Filter _filter;
    std::shared_ptr<daf::base::PropertySet> _metadata;
    std::shared_ptr<CoaddInputs> _coaddInputs;
    std::shared_ptr<ApCorrMap> _apCorrMap;
    std::shared_ptr<image::VisitInfo const> _visitInfo;
    std::shared_ptr<TransmissionCurve const> _transmissionCurve;

    // Class invariant: all values in _components are shared_ptr<Storable>
    // This is required for table::io persistence to work correctly;
    //     other persistence frameworks may let us support other types
    // Class invariant: all pointers in _components are not null
    std::unique_ptr<typehandling::MutableGenericMap<std::string>> _components;
};
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_IMAGE_ExposureInfo_h_INCLUDED
