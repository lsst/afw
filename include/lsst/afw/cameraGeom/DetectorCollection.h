/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_AFW_CAMERAGEOM_DETECTORCOLLECTION_H
#define LSST_AFW_CAMERAGEOM_DETECTORCOLLECTION_H

#include <map>
#include <string>
#include <memory>

#include "lsst/geom/Box.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * An immutable collection of Detectors that can be accessed by name or ID
 */
class DetectorCollection : public table::io::PersistableFacade<DetectorCollection>,
                           public table::io::Persistable {
public:
    using NameMap = std::unordered_map<std::string, std::shared_ptr<Detector>>;
    using IdMap = std::unordered_map<int, std::shared_ptr<Detector>>;
    using List = std::vector<std::shared_ptr<Detector>>;

    explicit DetectorCollection(List const & detectorList);

    // DetectorCollection is immutable, so it cannot be moveable.  It is also
    // always held by shared_ptr, so there is no good reason to copy it.
    DetectorCollection(DetectorCollection const &) = delete;
    DetectorCollection(DetectorCollection &&) = delete;

    // DetectorCollection is immutable, so it cannot be assignable.
    DetectorCollection & operator=(DetectorCollection const &) = delete;
    DetectorCollection & operator=(DetectorCollection &&) = delete;

    virtual ~DetectorCollection() noexcept;

    /// Get an unordered map over detector names
    NameMap const & getNameMap() const noexcept { return _nameDict; }

    /// Get an unordered map over detector IDs
    IdMap const & getIdMap() const noexcept { return _idDict; }

    /// Return a focal plane bounding box that encompasses all detectors
    lsst::geom::Box2D const & getFpBBox() const noexcept { return _fpBBox; }

    /**
     * Get the number of detectors.  Renamed to `__len__` in Python.
     */
    std::size_t size() const noexcept { return _idDict.size(); }

    /**
     * Determine if the DetectorCollection contains any Detectors.
     */
    bool empty() const noexcept { return _idDict.empty(); }

    /**
     * Implement the [name] operator
     *
     * @param[in] name  detector name
     * @return pointer to detector entry
     */
    std::shared_ptr<Detector> operator[](std::string const & name) const;

    /**
     * Implement the [id] operator
     *
     * @param[in] id  detector name
     * @return pointer to detector entry
     */
    std::shared_ptr<Detector> operator[](int id) const;

    /**
     * Support the "in" operator
     *
     * @param[in] name  detector name
     * @param[in] def  default detector to return.  This defaults to the NULL pointer
     * @return pointer to detector entry if the entry exists, else return the default value
     */
    std::shared_ptr<Detector> get(std::string const & name, std::shared_ptr<Detector> def=nullptr) const;

    /**
     * Support the "in" operator
     *
     * @param[in] id  detector id
     * @param[in] def  default detector to return.  This defaults to the NULL pointer
     * @return pointer to detector entry if the entry exists, else return the default value
     */
    std::shared_ptr<Detector> get(int id, std::shared_ptr<Detector> def=nullptr) const;

    /// DetectorCollections are always persistable.
    bool isPersistable() const noexcept override {
        return true;
    }

protected:

    DetectorCollection(table::io::InputArchive const & archive, table::io::CatalogVector const & catalogs);

    std::string getPersistenceName() const override;

    std::string getPythonModule() const override;

    void write(OutputArchiveHandle& handle) const override;

private:

    class Factory;

    NameMap _nameDict;                //< map of detector names
    IdMap _idDict;                    //< map of detector ids
    lsst::geom::Box2D _fpBBox;        //< bounding box of collection
};

} // namespace cameraGeom
} // namespace afw
} // namespace lsst


#endif // LSST_AFW_CAMERAGEOM_DETECTORCOLLECTION_H
