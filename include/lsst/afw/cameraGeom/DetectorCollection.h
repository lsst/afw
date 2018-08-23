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
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

class DetectorCollection {

public:
    using NameMap = std::unordered_map<std::string, std::shared_ptr<Detector>>;
    using IdMap = std::unordered_map<int, std::shared_ptr<Detector>>;
    using List = std::vector<std::shared_ptr<Detector>>;

    explicit DetectorCollection(List const & detectorList);

    DetectorCollection(DetectorCollection const &);
    DetectorCollection(DetectorCollection &&) noexcept;

    DetectorCollection & operator=(DetectorCollection const &);
    DetectorCollection & operator=(DetectorCollection &&) noexcept;

    virtual ~DetectorCollection() noexcept;

    NameMap const & getNameMap() const noexcept { return _nameDict; }
    IdMap const & getIdMap() const noexcept { return _idDict; }
    lsst::geom::Box2D const & getFpBBox() const noexcept { return _fpBBox; }

    std::size_t size() const noexcept { return _idDict.size(); }
    bool empty() const noexcept { return _idDict.empty(); }
    std::shared_ptr<Detector> operator[](std::string const & name) const;
    std::shared_ptr<Detector> operator[](int id) const;

    std::shared_ptr<Detector> get(std::string const & name, std::shared_ptr<Detector> def=nullptr) const;
    std::shared_ptr<Detector> get(int id, std::shared_ptr<Detector> def=nullptr) const;

private:
    NameMap _nameDict;
    IdMap _idDict;
    lsst::geom::Box2D _fpBBox;
};

} // namespace cameraGeom
} // namespace afw
} // namespace lsst


#endif // LSST_AFW_CAMERAGEOM_DETECTORCOLLECTION_H
