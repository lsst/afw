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

#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/cameraGeom/DetectorCollection.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

template <typename T>
std::shared_ptr<T> DetectorCollectionBase<T>::operator[](std::string const & name) const {
    auto det = get(name);
    if (det == nullptr) {
        throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                          (boost::format("Detector with name %s not found") % name).str());
    }
    return det;
}

template <typename T>
std::shared_ptr<T> DetectorCollectionBase<T>::operator[](int id) const {
    auto det = get(id);
    if (det == nullptr) {
        throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                          (boost::format("Detector with ID %s not found") % id).str());
    }
    return det;
}

template <typename T>
std::shared_ptr<T> DetectorCollectionBase<T>::get(std::string const & name, std::shared_ptr<T> def) const {
    auto i = _nameDict.find(name);
    if (i == _nameDict.end()) {
        return def;
    }
    return i->second;
}

template <typename T>
std::shared_ptr<T> DetectorCollectionBase<T>::get(int id, std::shared_ptr<T> def) const {
    auto i = _idDict.find(id);
    if (i == _idDict.end()) {
        return def;
    }
    return i->second;
}

template <typename T>
DetectorCollectionBase<T>::DetectorCollectionBase(List const & detectorList) {
    for (auto const & detector : detectorList) {
        _nameDict[detector->getName()] = detector;
        _idDict[detector->getId()] = detector;
    }

    if (_idDict.size() < detectorList.size()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Detector IDs are not unique");
    }
    if (_nameDict.size() < detectorList.size()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Detector names are not unique");
    }
}

template <typename T>
void DetectorCollectionBase<T>::add(std::shared_ptr<T> detector) {
    auto idIter = _idDict.find(detector->getId());
    auto nameIter = _nameDict.find(detector->getName());
    if (idIter == _idDict.end()) {
        if (nameIter == _nameDict.end()) {
            _idDict.emplace(detector->getId(), detector);
            _nameDict.emplace(detector->getName(), detector);
        } else {
            throw LSST_EXCEPT(
                pex::exceptions::RuntimeError,
                (boost::format("Detector name %s is not unique.") % detector->getName()).str()
            );
        }
    } else {
        if (nameIter == _nameDict.end()) {
            throw LSST_EXCEPT(
                pex::exceptions::RuntimeError,
                (boost::format("Detector ID %s is not unique.") % detector->getId()).str()
            );
        } else {
            if (nameIter->second != detector) {
                assert(idIter->second != detector);
                throw LSST_EXCEPT(
                    pex::exceptions::RuntimeError,
                    (boost::format("Detector name %s and ID %s are not unique.") % detector->getName()
                     % detector->getId()).str()
                );
            }
            // detector is already present; do nothing
        }
    }
}

template <typename T>
void DetectorCollectionBase<T>::remove(std::string const & name) {
    auto nameIter = _nameDict.find(name);
    if (nameIter == _nameDict.end()) {
        throw LSST_EXCEPT(
            pex::exceptions::NotFoundError,
            (boost::format("Detector with name %s not found.") % name).str()
        );
    }
    auto idIter = _idDict.find(nameIter->second->getId());
    assert(idIter != _idDict.end());
    _nameDict.erase(nameIter);
    _idDict.erase(idIter);
}

template <typename T>
void DetectorCollectionBase<T>::remove(int id) {
    auto idIter = _idDict.find(id);
    if (idIter == _idDict.end()) {
        throw LSST_EXCEPT(
            pex::exceptions::NotFoundError,
            (boost::format("Detector with ID %s not found.") % id).str()
        );
    }
    auto nameIter = _nameDict.find(idIter->second->getName());
    assert(nameIter != _nameDict.end());
    _nameDict.erase(nameIter);
    _idDict.erase(idIter);
}

template class DetectorCollectionBase<Detector const>;


namespace {

class PersistenceHelper {
public:

    static PersistenceHelper const & get() {
        static PersistenceHelper const instance;
        return instance;
    }

    table::Schema schema;
    table::Key<int> detector;

    DetectorCollection::List makeDetectorList(
        table::io::InputArchive const & archive,
        table::io::CatalogVector const & catalogs
    ) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() >= 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == schema);
        DetectorCollection::List result;
        result.reserve(catalogs.front().size());
        for (auto const & record : catalogs.front()) {
            int archiveId = record.get(detector);
            result.push_back(archive.get<Detector>(archiveId));
        }
        return result;
    }

private:

    PersistenceHelper() :
        schema(),
        detector(schema.addField<int>("detector", "archive ID of Detector in a DetectorCollection"))
    {}

    PersistenceHelper(PersistenceHelper const &) = delete;
    PersistenceHelper(PersistenceHelper &&) = delete;

    PersistenceHelper & operator=(PersistenceHelper const &) = delete;
    PersistenceHelper & operator=(PersistenceHelper &&) = delete;

};

} // anonymous


DetectorCollection::DetectorCollection(List const & detectorList) :
    DetectorCollectionBase<Detector const>(detectorList)
{
    for (auto const & detector : detectorList) {
        for (auto const & corner : detector->getCorners(FOCAL_PLANE)) {
            _fpBBox.include(corner);
        }
    }
}

class DetectorCollection::Factory : public table::io::PersistableFactory {
public:

    Factory() : table::io::PersistableFactory("DetectorCollection") {}

    std::shared_ptr<Persistable> read(InputArchive const& archive,
                                      CatalogVector const& catalogs) const override {
        // can't use make_shared because ctor is protected
        return std::shared_ptr<DetectorCollection>(new DetectorCollection(archive, catalogs));
    }

    static Factory const registration;
};

DetectorCollection::Factory const DetectorCollection::Factory::registration;


DetectorCollection::DetectorCollection(
    table::io::InputArchive const & archive,
    table::io::CatalogVector const & catalogs
) : DetectorCollection(PersistenceHelper::get().makeDetectorList(archive, catalogs))
{}

std::string DetectorCollection::getPersistenceName() const {
    return "DetectorCollection";
}

std::string DetectorCollection::getPythonModule() const {
    return "lsst.afw.cameraGeom";
}

void DetectorCollection::write(OutputArchiveHandle& handle) const {
    auto const & keys = PersistenceHelper::get();
    auto cat = handle.makeCatalog(keys.schema);
    for (auto const & pair : getIdMap()) {
        auto record = cat.addNew();
        record->set(keys.detector, handle.put(pair.second));
    }
    handle.saveCatalog(cat);
}

} // namespace cameraGeom

namespace table {
namespace io {

template class PersistableFacade<cameraGeom::DetectorCollection>;

} // namespace io
} // namespace table

} // namespace afw
} // namespace lsst
