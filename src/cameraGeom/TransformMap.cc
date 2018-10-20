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

#include <exception>
#include <memory>
#include <sstream>
#include <type_traits>
#include <set>

#include "boost/optional.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/cameraGeom/TransformMap.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

std::shared_ptr<TransformMap const> TransformMap::make(
    CameraSys const & reference,
    Transforms const & transforms
) {
    return Builder(reference).connect(transforms).build();
}

lsst::afw::geom::Point2Endpoint TransformMap::_pointConverter;

// All resources owned by value or by smart pointer
TransformMap::~TransformMap() noexcept = default;

lsst::geom::Point2D TransformMap::transform(lsst::geom::Point2D const &point, CameraSys const &fromSys,
                                            CameraSys const &toSys) const {
    auto mapping = _getMapping(fromSys, toSys);
    return _pointConverter.pointFromData(mapping->applyForward(_pointConverter.dataFromPoint(point)));
}

std::vector<lsst::geom::Point2D> TransformMap::transform(std::vector<lsst::geom::Point2D> const &pointList,
                                                         CameraSys const &fromSys,
                                                         CameraSys const &toSys) const {
    auto mapping = _getMapping(fromSys, toSys);
    return _pointConverter.arrayFromData(mapping->applyForward(_pointConverter.dataFromArray(pointList)));
}

bool TransformMap::contains(CameraSys const &system) const noexcept { return _frameIds.count(system) > 0; }

std::shared_ptr<geom::TransformPoint2ToPoint2> TransformMap::getTransform(CameraSys const &fromSys,
                                                                          CameraSys const &toSys) const {
    return std::make_shared<geom::TransformPoint2ToPoint2>(*_getMapping(fromSys, toSys));
}

int TransformMap::_getFrame(CameraSys const &system) const {
    try {
        return _frameIds.at(system);
    } catch (std::out_of_range const &e) {
        std::ostringstream buffer;
        buffer << "Unsupported coordinate system: " << system;
        std::throw_with_nested(LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str()));
    }
}

std::shared_ptr<ast::Mapping const> TransformMap::_getMapping(CameraSys const &fromSys,
                                                              CameraSys const &toSys) const {
    return _transforms->getMapping(_getFrame(fromSys), _getFrame(toSys));
}

size_t TransformMap::size() const noexcept { return _frameIds.size(); }

TransformMap::TransformMap(std::unique_ptr<ast::FrameSet> && transforms,
                           CameraSysFrameIdMap && frameIds,
                           std::vector<std::pair<int, int>> && canonicalConnections) :
    _transforms(std::move(transforms)),
    _frameIds(std::move(frameIds)),
    _canonicalConnections(std::move(canonicalConnections))
{}


namespace {

struct PersistenceHelper {

    static PersistenceHelper const & get() {
        static PersistenceHelper const instance;
        return instance;
    }

    // Schema and keys for the catalog that stores TransformMap._frameIds.
    // Considered as a graph, this is a list of all of the vertices with the
    // integers that identify them in the list of edges below.
    table::Schema sysSchema;
    table::Key<std::string> sysName;
    table::Key<std::string> detectorName;
    table::Key<int> id;

    // Schema and keys for the catalog that stores
    // TransformMap._canonicalConnections entries and the associated Transform
    // extracted from TransformMap._transforms.
    // Considered as a graph, 'from' and 'to' identify vertices, and
    // 'transform' identifies an edge.
    table::Schema connectionSchema;
    table::Key<int> from;
    table::Key<int> to;
    table::Key<int> transform;

    CameraSys makeCameraSys(table::BaseRecord const & record) const {
        return CameraSys(record.get(sysName), record.get(detectorName));
    }

private:

    PersistenceHelper() :
        sysSchema(),
        sysName(sysSchema.addField<std::string>("sysName", "Camera coordinate system name", "", 0)),
        detectorName(sysSchema.addField<std::string>("detectorName",
                                                     "Camera coordinate system detector name", "", 0)),
        id(sysSchema.addField<int>("id", "AST ID of the Frame for the CameraSys", "")),
        connectionSchema(),
        from(connectionSchema.addField<int>("from", "AST ID of the Frame this transform maps from.", "")),
        to(connectionSchema.addField<int>("to", "AST ID of the Frame this transform maps to.", "")),
        transform(connectionSchema.addField<int>("transform", "Archive ID of the transform.", ""))
    {
        sysSchema.getCitizen().markPersistent();
        connectionSchema.getCitizen().markPersistent();
    }

    PersistenceHelper(PersistenceHelper const &) = delete;
    PersistenceHelper(PersistenceHelper &&) = delete;

    PersistenceHelper & operator=(PersistenceHelper const &) = delete;
    PersistenceHelper & operator=(PersistenceHelper &&) = delete;

};

std::string makeFrameName(CameraSys const & sys) {
    std::string r = "Ident=" + sys.getSysName();
    if (sys.hasDetectorName()) {
        r += "_";
        r += sys.getDetectorName();
    }
    return r;
}

}  // namespace


std::string TransformMap::getPersistenceName() const {
    return "TransformMap";
}

std::string TransformMap::getPythonModule() const {
    return "lsst.afw.cameraGeom";
}

void TransformMap::write(OutputArchiveHandle& handle) const {
    auto const & keys = PersistenceHelper::get();

    auto sysCat = handle.makeCatalog(keys.sysSchema);
    for (auto const & sysPair : _frameIds) {
        auto sysRecord = sysCat.addNew();
        sysRecord->set(keys.sysName, sysPair.first.getSysName());
        sysRecord->set(keys.detectorName, sysPair.first.getDetectorName());
        sysRecord->set(keys.id, sysPair.second);
    }
    sysCat.sort(keys.id);
    handle.saveCatalog(sysCat);

    auto connectionCat = handle.makeCatalog(keys.connectionSchema);
    for (auto const & connectionPair : _canonicalConnections) {
        auto connectionRecord = connectionCat.addNew();
        connectionRecord->set(keys.from, connectionPair.first);
        connectionRecord->set(keys.to, connectionPair.second);
        auto transform = geom::TransformPoint2ToPoint2(
            *_transforms->getMapping(connectionPair.first, connectionPair.second));
        connectionRecord->set(keys.transform, handle.put(transform));
    }
    handle.saveCatalog(connectionCat);
}

class TransformMap::Factory : public table::io::PersistableFactory {
public:

    Factory() : PersistableFactory("TransformMap") {}

    std::shared_ptr<Persistable> read(InputArchive const& archive,
                                      CatalogVector const& catalogs) const override {
        auto const & keys = PersistenceHelper::get();

        LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
        auto const & sysCat = catalogs[0];
        auto const & connectionCat = catalogs[1];
        LSST_ARCHIVE_ASSERT(sysCat.getSchema() == keys.sysSchema);
        LSST_ARCHIVE_ASSERT(connectionCat.getSchema() == keys.connectionSchema);
        LSST_ARCHIVE_ASSERT(sysCat.size() == connectionCat.size() + 1);
        LSST_ARCHIVE_ASSERT(sysCat.isSorted(keys.id));

        CameraSysFrameIdMap frameIdsBySys;
        std::unordered_map<int, ast::Frame> framesById;
        for (auto const & sysRecord : sysCat) {
            auto sys = keys.makeCameraSys(sysRecord);
            frameIdsBySys.emplace(sys, sysRecord.get(keys.id));
            framesById.emplace(sysRecord.get(keys.id), ast::Frame(2, makeFrameName(sys)));
        }

        auto baseFrameIter = framesById.find(1);
        LSST_ARCHIVE_ASSERT(baseFrameIter != framesById.end());
        auto frameSet = std::make_unique<ast::FrameSet>(baseFrameIter->second);
        std::vector<std::pair<int, int>> canonicalConnections;
        for (auto const & connectionRecord : connectionCat) {
            int const fromId = connectionRecord.get(keys.from);
            int const toId = connectionRecord.get(keys.to);
            auto const transform = archive.get<geom::TransformPoint2ToPoint2>(
                connectionRecord.get(keys.transform)
            );
            canonicalConnections.emplace_back(fromId, toId);
            auto toFrameIter = framesById.find(toId);
            LSST_ARCHIVE_ASSERT(toFrameIter != framesById.end());
            frameSet->addFrame(fromId, *transform->getMapping(), toFrameIter->second);
        }

        return std::shared_ptr<TransformMap>(new TransformMap(std::move(frameSet),
                                                              std::move(frameIdsBySys),
                                                              std::move(canonicalConnections)));
    }

};

TransformMap::Factory const TransformMap::registration;


TransformMap::Builder::Builder(CameraSys const & reference) : _reference(reference) {}

TransformMap::Builder::Builder(Builder const &) = default;
TransformMap::Builder::Builder(Builder &&) = default;

TransformMap::Builder & TransformMap::Builder::operator=(Builder const &) = default;
TransformMap::Builder & TransformMap::Builder::operator=(Builder &&) = default;

TransformMap::Builder::~Builder() noexcept = default;

TransformMap::Builder & TransformMap::Builder::connect(
    CameraSys const & fromSys,
    CameraSys const & toSys,
    std::shared_ptr<geom::TransformPoint2ToPoint2 const> transform
) {
    if (fromSys == toSys) {
        std::ostringstream buffer;
        buffer << "Identity connection detected for " << fromSys << ".";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    if (!transform->hasForward()) {
        std::ostringstream buffer;
        buffer << "Connection from " << fromSys << " to "
               << toSys << " has no forward transform.";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    if (!transform->hasInverse()) {
        std::ostringstream buffer;
        buffer << "Connection from " << fromSys << " to "
               << toSys << " has no inverse transform.";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    _connections.push_back(Connection{false, std::move(transform), fromSys, toSys});
    return *this;
}


TransformMap::Builder & TransformMap::Builder::connect(
    CameraSys const & fromSys,
    Transforms const & transforms
) {
    Builder other(_reference);  // temporary for strong exception safety
    for (auto const & item : transforms) {
        other.connect(fromSys, item.first, item.second);
    }
    _connections.insert(_connections.end(), other._connections.begin(), other._connections.end());
    return *this;
}


namespace {

/*
 * RAII object that just executes a functor in its destructor.
 */
template <typename Callable>
class OnScopeExit {
public:

    explicit OnScopeExit(Callable callable) : _callable(std::move(callable)) {}

    ~OnScopeExit() noexcept(noexcept(std::declval<Callable>())) {  // noexcept iff "_callable()"" is
        _callable();
    }

private:
    Callable _callable;
};

// Factory function for OnScopeExit.
template <typename Callable>
OnScopeExit<Callable> onScopeExit(Callable callable) {
    return OnScopeExit<Callable>(std::move(callable));
}

}  // namespace


std::shared_ptr<TransformMap const> TransformMap::Builder::build() const {

    int nFrames = 0;  // tracks frameSet->getNFrame() to avoid those (expensive) calls
    CameraSysFrameIdMap frameIds;  // mapping from CameraSys to Frame ID (int)
    std::vector<std::pair<int, int>> canonicalConnections;  // remembers the frame IDs we've connected

    // Local helper function that looks up the Frame ID for a CameraSys, with
    // results returned via boost::optional.
    auto findFrameIdForSys = [&frameIds](CameraSys const & sys) -> boost::optional<int> {
        auto iter = frameIds.find(sys);
        if (iter != frameIds.end()) {
            return boost::optional<int>(iter->second);
        } else {
            return boost::none;
        }
    };

    // Local helper function that creates a Frame, updates the nFrames counter,
    // and adds an entry to the frameIds map.  Returns the new Frame.
    // Should always be called in concert with an update to frameSet.
    auto addFrameForSys = [&frameIds, &nFrames](CameraSys const & sys) mutable -> ast::Frame {
        frameIds.emplace(sys, ++nFrames);
        return ast::Frame(2, makeFrameName(sys));
    };

    // FrameSet that manages all transforms; should always be updated in
    // concert with a call to addFrameForSys.
    auto frameSet = std::make_unique<ast::FrameSet>(addFrameForSys(_reference));

    // RAII: make sure all 'processed' fields are reset, no matter how we exit
    auto guard = onScopeExit(
        [this]() noexcept {
            for (auto const & c : _connections) { c.processed = false; }
        }
    );

    std::size_t nProcessedTotal = 0;
    while (nProcessedTotal != _connections.size()) {

        // Loop over all connections, only inserting those that are connected
        // to already-inserted connections.
        std::size_t nProcessedThisPass = 0;
        for (auto const & c : _connections) {
            if (c.processed) continue;
            auto fromId = findFrameIdForSys(c.fromSys);
            auto toId = findFrameIdForSys(c.toSys);
            if (fromId && toId) {  // We've already inserted both fromSys and toSys. That's a problem.
                std::ostringstream buffer;
                buffer << "Duplicate connection from " << c.fromSys << " to " << c.toSys << ".";
                throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
            } else if (fromId) {  // We've already inserted fromSys (only)
                frameSet->addFrame(*fromId, *c.transform->getMapping(), addFrameForSys(c.toSys));
                canonicalConnections.emplace_back(*fromId, nFrames);
                c.processed = true;
                ++nProcessedThisPass;
            } else if (toId) {  // We've already inserted toSys (only)
                frameSet->addFrame(*toId, *c.transform->inverted()->getMapping(), addFrameForSys(c.fromSys));
                canonicalConnections.emplace_back(*toId, nFrames);
                c.processed = true;
                ++nProcessedThisPass;
            }
            // If we haven't inserted either yet, just continue; hopefully
            // we'll have inserted one in a future pass.
        }

        if (nProcessedThisPass == 0u) {  // We're not making progress, so we must have orphans
            // Use std::set to compile the list of orphaned coordinate systems
            // for a friendlier (unique, predictably-ordered) error message.
            std::set<CameraSys> orphaned;
            for (auto const & c : _connections) {
                if (!c.processed) {
                    orphaned.insert(c.fromSys);
                    orphaned.insert(c.toSys);
                }
            }
            std::ostringstream buffer;
            auto o = orphaned.begin();
            buffer << "Orphaned coordinate system(s) found: " << *o;
            for (++o; o != orphaned.end(); ++o) {
                buffer << ", " << *o;
            }
            buffer << ".";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
        }

        nProcessedTotal += nProcessedThisPass;
    }

    // We've maintained our own counter for frame IDs for performance and
    // convenience reasons, but it had better match AST's internal counter.
    assert(frameSet->getNFrame() == nFrames);

    // Return the new TransformMap.
    // We can't use make_shared because TransformMap ctor is private.
    return std::shared_ptr<TransformMap>(new TransformMap(std::move(frameSet),
                                                          std::move(frameIds),
                                                          std::move(canonicalConnections)));
}

}  // namespace cameraGeom

namespace table {
namespace io {

template class PersistableFacade<cameraGeom::TransformMap>;

}  // namespace io
}  // namespace table

}  // namespace afw
}  // namespace lsst
