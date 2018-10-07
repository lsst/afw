/*
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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

#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include "lsst/base.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/table/AmpInfo.h"
#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/afw/cameraGeom/Orientation.h"
#include "lsst/afw/cameraGeom/TransformMap.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Type of imaging detector
 */
enum DetectorType {
    SCIENCE,
    FOCUS,
    GUIDER,
    WAVEFRONT,
};

/**
 * Information about a CCD or other imaging detector
 *
 * Supports transformation of points between FOCAL_PLANE and pixel-based coordinate systems.
 * Also an iterator over amplifiers (in C++ use begin(), end(), in Python use "for amplifier in detector").
 *
 * @todo: this would probably be a bit more robust if it used a ConstAmpInfoCatalog
 * (a catalog with const records) but I don't think const catalogs really work yet;
 * for instance it is not possible to construct one from a non-const catalog,
 * so I don't know how to construct one.
 */
class Detector final : public table::io::PersistableFacade<Detector>, public table::io::Persistable {
public:
    typedef ndarray::Array<float const, 2> CrosstalkMatrix;

    /**
     * Make a Detector
     *
     * @param name name of detector's location in the camera
     * @param id detector integer ID; used as keys in some tables
     * @param type type of detector
     * @param serial serial "number" that identifies the physical detector
     * @param bbox bounding box
     * @param ampInfoCatalog catalog of amplifier information
     * @param orientation detector position and orientation in focal plane
     * @param pixelSize pixel size (mm)
     * @param transforms map of CameraSys: afw::geom::Transform, where each
     *                   Transform's forward transform transforms from PIXELS
     *                   to the specified camera system
     * @param crosstalk matrix of crosstalk coefficients
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if any amplifier
     *     names are not unique
     *
     * @warning
     * * The keys for the detector-specific coordinate systems in the transform registry
     *   must include the detector name (even though this is redundant).
     */
    Detector(std::string const &name, int id, DetectorType type, std::string const &serial,
             lsst::geom::Box2I const &bbox, lsst::afw::table::AmpInfoCatalog const &ampInfoCatalog,
             Orientation const &orientation, lsst::geom::Extent2D const &pixelSize,
             TransformMap::Transforms const &transforms,
             CrosstalkMatrix const &crosstalk = CrosstalkMatrix());

    /**
     * Make a Detector
     *
     * @param name name of detector's location in the camera
     * @param id detector integer ID; used as keys in some tables
     * @param type type of detector
     * @param serial serial "number" that identifies the physical detector
     * @param bbox bounding box
     * @param ampInfoCatalog catalog of amplifier information
     * @param orientation detector position and orientation in focal plane
     * @param pixelSize pixel size (mm)
     * @param transformMap coordinate systems and transforms between them
     * @param crosstalk matrix of crosstalk coefficients
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if: any amplifier
     *     names are not unique
     */
    Detector(std::string const &name, int id, DetectorType type, std::string const &serial,
             lsst::geom::Box2I const &bbox, lsst::afw::table::AmpInfoCatalog const &ampInfoCatalog,
             Orientation const &orientation, lsst::geom::Extent2D const &pixelSize,
             std::shared_ptr<TransformMap const> transformMap,
             CrosstalkMatrix const &crosstalk = CrosstalkMatrix());

    ~Detector() = default;

    Detector(Detector const &) = delete;
    Detector(Detector &&) = delete;
    Detector &operator=(Detector const &) = delete;
    Detector &operator=(Detector &&) = delete;

    /** Get the detector name */
    std::string getName() const { return _name; }

    /** Get the detector ID */
    int getId() const { return _id; }

    DetectorType getType() const { return _type; }

    /** Get the detector serial "number" */
    std::string getSerial() const { return _serial; }

    /** Get the bounding box */
    lsst::geom::Box2I getBBox() const { return _bbox; }

    /** Get the corners of the detector in the specified camera coordinate system */
    std::vector<lsst::geom::Point2D> getCorners(CameraSys const &cameraSys) const;

    /** Get the corners of the detector in the specified camera coordinate system prefix */
    std::vector<lsst::geom::Point2D> getCorners(CameraSysPrefix const &cameraSysPrefix) const;

    /** Get the center of the detector in the specified camera coordinate system */
    lsst::geom::Point2D getCenter(CameraSys const &cameraSys) const;

    /** Get the center of the detector in the specified camera coordinate system prefix */
    lsst::geom::Point2D getCenter(CameraSysPrefix const &cameraSysPrefix) const;

    /** Get the amplifier information catalog */
    table::AmpInfoCatalog const getAmpInfoCatalog() const { return _ampInfoCatalog; }

    /** Get detector's orientation in the focal plane */
    Orientation const getOrientation() const { return _orientation; }

    /** Get size of pixel along (mm) */
    lsst::geom::Extent2D getPixelSize() const { return _pixelSize; }

    /** Get the transform registry */
    std::shared_ptr<TransformMap const> getTransformMap() const { return _transformMap; }

    /** Have we got crosstalk coefficients? */
    bool hasCrosstalk() const {
        return !(_crosstalk.isEmpty() || _crosstalk.getShape() == ndarray::makeVector(0, 0));
    }

    /** Get the crosstalk coefficients */
    CrosstalkMatrix const getCrosstalk() const { return _crosstalk; }

    /** Get iterator to beginning of amplifier list */
    table::AmpInfoCatalog::const_iterator begin() const { return _ampInfoCatalog.begin(); }

    /** Get iterator to end of amplifier list */
    table::AmpInfoCatalog::const_iterator end() const { return _ampInfoCatalog.end(); }

    /**
     * Get the amplifier specified by index
     *
     * @throws std::out_of_range if index is out of range
     */
    table::AmpInfoRecord const &operator[](size_t i) const { return _ampInfoCatalog.at(i); }

    /**
     * Get the amplifier specified by name
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if no such amplifier
     */
    table::AmpInfoRecord const &operator[](std::string const &name) const;

    /**
     * Get the amplifier specified by index, returning a shared pointer to an AmpInfo record
     *
     * @warning Intended only for use by pybind11. This exists because
     * Operator[] returns a data type that is difficult for pybind11 to use.
     * Since we have it, we also take advantage of the fact that it is only for pybind11
     * to support negative indices in the python style.
     *
     * @param[in] i  Ampifier index; if < 0 then treat as an offset from the end (the Python convention)
     *
     * @throws std::out_of_range if index is out of range
     */
    std::shared_ptr<table::AmpInfoRecord const> _get(int i) const;

    /**
     * Get the amplifier specified by name, returning a shared pointer to an AmpInfo record
     *
     * @warning Intended only for internal and pybind11 use. This exists because
     * Operator[] returns a data type that is difficult for pybind11 to use.
     *
     * @param[in] name  Amplifier name
     *
     * @throws std::out_of_range if index is out of range
     */
    std::shared_ptr<table::AmpInfoRecord const> _get(std::string const &name) const;

    /**
     * Get the number of amplifiers. Renamed to `__len__` in Python.
     */
    size_t size() const { return _ampInfoCatalog.size(); }

    /** Can this object convert between PIXELS and the specified camera coordinate system? */
    bool hasTransform(CameraSys const &cameraSys) const;

    /** Can this object convert between PIXELS and the specified camera coordinate system prefix? */
    bool hasTransform(CameraSysPrefix const &cameraSysPrefix) const;

    /**
     * Get a Transform from one camera coordinate system, or camera coordinate system prefix, to another.
     *
     * @tparam FromSysT, ToSysT  Type of `fromSys`, `toSys`: one of `CameraSys` or `CameraSysPrefix`
     *
     * @param fromSys, toSys camera coordinate systems or prefixes between which to transform
     * @returns a Transform that converts from `fromSys` to `toSys` in the forward direction.
     *      The Transform will be invertible.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toSys` is not supported.
     */
    template <typename FromSysT, typename ToSysT>
    std::shared_ptr<afw::geom::TransformPoint2ToPoint2> getTransform(FromSysT const &fromSys,
                                                                     ToSysT const &toSys) const;

    /**
     * Get a coordinate system from a coordinate system (return input unchanged and untested)
     *
     * @param[in] cameraSys  Camera coordinate system
     * @return `cameraSys` unchanged
     *
     * @note the CameraSysPrefix version needs the detector name, which is why this is not static.
     */
    CameraSys const makeCameraSys(CameraSys const &cameraSys) const { return cameraSys; }

    /**
     * Get a coordinate system from a detector system prefix (add detector name)
     *
     * @param[in] cameraSysPrefix  Camera coordinate system prefix
     * @return `cameraSysPrefix` with the detector name added
     */
    CameraSys const makeCameraSys(CameraSysPrefix const &cameraSysPrefix) const {
        return CameraSys(cameraSysPrefix, _name);
    }

    /**
     * Transform a point from one camera system to another
     *
     * @tparam FromSysT  Class of fromSys: one of CameraSys or CameraSysPrefix
     * @tparam ToSysT  Class of toSys: one of CameraSys or CameraSysPrefix
     * @param[in] point  Camera point to transform
     * @param[in] fromSys  Camera coordinate system of `point`
     * @param[in] toSys  Camera coordinate system of returned point
     * @return The transformed point
     *
     * @throws pex::exceptions::InvalidParameterError if fromSys or toSys is unknown
     */
    template <typename FromSysT, typename ToSysT>
    lsst::geom::Point2D transform(lsst::geom::Point2D const &point, FromSysT const &fromSys,
                                  ToSysT const &toSys) const;

    /**
     * Transform a vector of points from one camera system to another
     *
     * @tparam FromSysT  Class of fromSys: one of CameraSys or CameraSysPrefix
     * @tparam ToSysT  Class of toSys: one of CameraSys or CameraSysPrefix
     * @param[in] points  Camera points to transform
     * @param[in] fromSys  Camera coordinate system of `points`
     * @param[in] toSys  Camera coordinate system of returned points
     * @return The transformed points
     *
     * @throws pex::exceptions::InvalidParameterError if fromSys or toSys is unknown
     */
    template <typename FromSysT, typename ToSysT>
    std::vector<lsst::geom::Point2D> transform(std::vector<lsst::geom::Point2D> const &points,
                                               FromSysT const &fromSys, ToSysT const &toSys) const;

    /// The "native" coordinate system of this detector.
    CameraSys getNativeCoordSys() const { return _nativeSys; }

    /// Detectors are always peristable
    bool isPersistable() const noexcept override { return true; }

private:
    typedef std::unordered_map<std::string, table::AmpInfoCatalog::const_iterator> _AmpInfoMap;

    std::string getPersistenceName() const override;

    std::string getPythonModule() const override;

    void write(OutputArchiveHandle& handle) const override;

    std::string _name;                      ///< name of detector's location in the camera
    int _id;                                ///< detector numeric ID
    DetectorType _type;                     ///< type of detectorsize_t
    std::string _serial;                    ///< serial "number" that identifies the physical detector
    lsst::geom::Box2I _bbox;                ///< bounding box
    table::AmpInfoCatalog _ampInfoCatalog;  ///< list of amplifier data
    _AmpInfoMap _ampNameIterMap;            ///< map of amplifier name: catalog iterator
    Orientation _orientation;               ///< position and orientation of detector in focal plane
    lsst::geom::Extent2D _pixelSize;        ///< pixel size (mm)
    CameraSys _nativeSys;                   ///< native coordinate system of this detector
    std::shared_ptr<TransformMap const> _transformMap;   ///< registry of coordinate transforms
    CrosstalkMatrix _crosstalk;             ///< crosstalk coefficients
};
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst

#endif
