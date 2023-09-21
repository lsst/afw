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

#include <mutex>
#include <set>
#include "boost/format.hpp"
#include "boost/functional/hash.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/detail/MaskDict.h"

namespace lsst {
namespace afw {
namespace image {
namespace detail {

// A thread-safe singleton that manages MaskDict's global canonical state.
class MaskDict::GlobalState final {
public:
    static GlobalState &get() {
        static GlobalState instance;
        return instance;
    }

    // TODO: I don't think we want to deprecate this, for parseing MaskDicts from metadata
    std::shared_ptr<MaskDict> newMaskDictFromMaps(MaskPlaneDict const &mpd, MaskPlaneDocDict const &docs) {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        if (!mpd.empty()) {
            std::shared_ptr<MaskDict> dict(new MaskDict(mpd, docs));
            return dict;
        }
        return copy(*_defaultMaskDict);
    }

    std::shared_ptr<MaskDict> copy(MaskDict const &dict) {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        std::shared_ptr<MaskDict> result(new MaskDict(dict));
        return result;
    }

    std::shared_ptr<MaskDict> getDefault() const noexcept { return _defaultMaskDict; }

    void setDefault(std::shared_ptr<MaskDict> dict) { _defaultMaskDict = std::move(dict); }

    // std::shared_ptr<MaskDict> detachDefault() {
    //     std::lock_guard<std::recursive_mutex> lock(_mutex);
    //     _defaultMaskDict = copy(*_defaultMaskDict);
    //     return _defaultMaskDict;
    // }

    // template <typename Functor>
    // void forEachMaskDict(Functor functor) {
    //     std::lock_guard<std::recursive_mutex> lock(_mutex);
    //     _prune();  // guarantees dereference below is safe
    //     for (auto const &ptr : _allMaskDicts) {
    //         functor(*ptr.lock());
    //     }
    // }

    // Return the bit id of an existing plane, or the next available bit, if the name is not in the list of
    // canonical planes.
    int getBitIdForNewPlane(std::string name) {
        auto iter = std::find(_canonicalPlanes.begin(), _canonicalPlanes.end(), name);
        if (iter == _canonicalPlanes.end()) {
            _canonicalPlanes.push_back(name);
            return _canonicalPlanes.size() - 1;
        } else {
            return iter - _canonicalPlanes.begin();
        }
    }

    void clearCanonicalPlanes() { _canonicalPlanes.clear(); }

    // Set the canonical planes to the current contents of the default MaskDict.
    void setCanonicalPlanesFromDefault() {
        _canonicalPlanes.clear();
        for (auto const &pair : _defaultMaskDict->getMaskPlaneDict()) {
            _canonicalPlanes.push_back(pair.first);
        }
    }

private:
    GlobalState() : _defaultMaskDict(new MaskDict()) { _defaultMaskDict->_addInitialMaskPlanes(); }

    GlobalState(GlobalState const &) = delete;
    GlobalState(GlobalState &&) = delete;

    GlobalState &operator=(GlobalState const &) = delete;
    GlobalState &operator=(GlobalState &&) = delete;

    ~GlobalState() = default;

    std::recursive_mutex _mutex;  // guards _allMaskDicts and synchronizes updates to it and _defaultMaskDict
    std::shared_ptr<MaskDict> _defaultMaskDict;
    // Maintains the list of all plane names; bit ids are the vector index.
    // New planes always are added to the end (thus always receiving a larger bit id),
    // even if there are gaps in the list that could be used.
    std::vector<std::string> _canonicalPlanes;
};

std::shared_ptr<MaskDict> MaskDict::newMaskDictFromMaps(MaskPlaneDict const &mpd,
                                                        MaskPlaneDocDict const &docs) {
    return GlobalState::get().newMaskDictFromMaps(mpd, docs);
}

std::shared_ptr<MaskDict> MaskDict::getDefaultIfEmpty(std::shared_ptr<MaskDict> const &dict) {
    if (dict != nullptr) {
        return dict;
    }
    return getDefault();
}

std::tuple<int, std::shared_ptr<MaskDict>> MaskDict::withNewMaskPlane(std::string name, std::string doc,
                                                                      int maxPlanes, bool ignoreCanonical) {
    auto iter = _dict.find(name);
    if (iter == _dict.end()) {
        int id = maxPlanes;
        std::set<int> existingIds;  // to check for empty bits we can re-use
        for (auto const &item : _dict) {
            existingIds.insert(item.second);
            std::cout << item.second << std::endl;
        }
        if (ignoreCanonical) {
            for (int i; i < maxPlanes; ++i) {
                if (existingIds.find(i) == existingIds.end()) {
                    id = i;
                    break;
                }
            }
        } else {
            id = GlobalState::get().getBitIdForNewPlane(name);
            if (existingIds.find(id) != existingIds.end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                                  (boost::format("Plane %1% cannot be added because it will duplicate "
                                                 "canonical plane id %2%.") %
                                   name % id)
                                          .str());
            }
        }
        std::cout << name << " " << id << std::endl;
        if (id >= maxPlanes) {
            // TODO?? iterate over _dict looking for gaps.
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::RuntimeError,
                    (boost::format("Max number of planes (%1%) already used when trying to add '%2%'") %
                     maxPlanes % name)
                            .str());
        }
        if (existingIds.find(id) != existingIds.end()) {
            auto newMaskDict = GlobalState::get().copy(*this);
            newMaskDict->_dict[name] = id;
            newMaskDict->_docs[name] = doc;
            return std::make_tuple(id, newMaskDict);
        } else {
            _dict[name] = id;
            _docs[name] = doc;
            return std::make_tuple(id, shared_from_this());
        }
    }

    int id = iter->second;
    if (_docs.at(name) == doc || doc.empty()) {
        return std::make_tuple(id, shared_from_this());
    } else if (_docs.at(name).empty()) {
        _docs[name] = doc;
        return std::make_tuple(id, shared_from_this());
    } else {
        auto newMaskDict = GlobalState::get().copy(*this);
        newMaskDict->_dict[name] = id;
        newMaskDict->_docs[name] = doc;
        return std::make_tuple(id, newMaskDict);
    }
}

std::shared_ptr<MaskDict> MaskDict::withRemovedMaskPlane(std::string name) {
    auto newMaskDict = GlobalState::get().copy(*this);
    newMaskDict->_dict.erase(name);
    newMaskDict->_docs.erase(name);
    return newMaskDict;
}

std::shared_ptr<MaskDict> MaskDict::getDefault() { return GlobalState::get().getDefault(); }

void MaskDict::setDefault(std::shared_ptr<MaskDict> dict, bool resetCanonicalPlanes) {
    GlobalState::get().setDefault(std::move(dict));
    if (resetCanonicalPlanes) {
        GlobalState::get().setCanonicalPlanesFromDefault();
    }
}

// NOTE: static
int MaskDict::getCanonicalPlaneId(std::string name) {
    int id = GlobalState::get().getBitIdForNewPlane(name);
    auto iter = GlobalState::get().getDefault()->_dict.find(name);
    if (iter->second != id) {
        throw LSST_EXCEPT(pexExcept::LogicError,
                          (boost::format("Canonical plane id (%1%) does not match global plane id (%2%)") %
                           id % iter->second)
                                  .str());
    }
    return id;
}

// NOTE: static
void MaskDict::clearDefaultPlanes(bool clearCanonical) {
    GlobalState::get().getDefault()->_dict.clear();
    GlobalState::get().getDefault()->_docs.clear();
    if (clearCanonical) {
        GlobalState::get().clearCanonicalPlanes();
    }
}

// NOTE: static
void MaskDict::restoreDefaultMaskDict() {
    clearDefaultPlanes(true);
    GlobalState::get().getDefault()->_addInitialMaskPlanes();
    GlobalState::get().setCanonicalPlanesFromDefault();
}

// std::shared_ptr<MaskDict> MaskDict::detachDefault() { return GlobalState::get().detachDefault(); }

// void MaskDict::addMaskPlane(std::string const &name, int bitId, std::string const &doc) {
//     // GlobalState::get().forEachMaskDict([&name, bitId, doc](MaskDict &dict) {
//     auto const found = std::find_if(dict.begin(), dict.end(),
//                                     [bitId](auto const &item) { return item.second == bitId; });
//     if (found == dict.end()) {
//         // is name already in use?
//         if (dict.find(name) == dict.end()) {
//             dict.add(name, bitId, doc);
//         }
//         // TODO: need to do something here when the docstrings don't match?
//     }
//     // });
// }

MaskDict::~MaskDict() noexcept = default;

// std::shared_ptr<MaskDict> MaskDict::clone() const { return GlobalState::get().copy(*this); }

// int MaskDict::getUnusedPlane() const {
//     if (empty()) {
//         return 0;
//     }

//     auto const maxIter = std::max_element(begin(), end(),
//                                           [](auto const &a, auto const &b) { return a.second < b.second;
//                                           });
//     assert(maxIter != end());
//     int id = maxIter->second + 1;  // The maskPlane to use if there are no gaps

//     for (int i = 0; i < id; ++i) {
//         // is i already used in this Mask?
//         auto const sameIter =
//                 std::find_if(begin(), end(), [i](auto const &item) { return item.second == i; });
//         if (sameIter == end()) {  // Not used; so we'll use it
//             return i;
//         }
//     }

//     return id;
// }

int MaskDict::getPlaneId(std::string const &name) const {
    auto iter = _dict.find(name);
    return (iter == _dict.end()) ? -1 : iter->second;
}

std::string MaskDict::getPlaneDoc(std::string const &name) const {
    auto iter = _docs.find(name);
    return (iter == _docs.end()) ? "" : iter->second;
}

std::string MaskDict::print() const {
    std::ostringstream out;
    auto it_dict = _dict.begin();
    auto it_doc = _docs.begin();
    std::map<int, std::string> lines;
    while (it_dict != _dict.end()) {
        std::stringstream line;
        line << "Plane " << it_dict->second << " -> " << it_dict->first << " : " << it_doc->second;
        lines[it_dict->second] = line.str();
        ++it_dict;
        ++it_doc;
    }
    // Ensure the newlines are correct, with the output ordered on bit number.
    auto i_line = lines.begin();
    while (i_line != lines.end()) {
        // newline on every line but the last
        if (i_line != lines.begin()) {
            out << std::endl;
        }
        out << i_line->second;
        i_line++;
    }
    return out.str();
}

bool MaskDict::operator==(MaskDict const &rhs) const {
    return this == &rhs || (_dict == rhs._dict && _docs == rhs._docs);
}

void MaskDict::_addInitialMaskPlanes() {
    int i = -1;
    _dict["BAD"] = ++i;
    _docs["BAD"] = "This pixel is known to be bad (e.g. the amplifier is not working).";
    _dict["SAT"] = ++i;
    _docs["SAT"] = "This pixel is saturated and has bloomed.";
    _dict["INTRP"] = ++i;
    _docs["INTRP"] = "This pixel has been interpolated over. Check other mask planes for the reason.";
    _dict["CR"] = ++i;
    _docs["CR"] = "This pixel is contaminated by a cosmic ray.";
    _dict["EDGE"] = ++i;
    _docs["EDGE"] = "This pixel is too close to the edge to be processed properly.";
    _dict["DETECTED"] = ++i;
    _docs["DETECTED"] = "This pixel lies within an object's Footprint.";
    _dict["DETECTED_NEGATIVE"] = ++i;
    _docs["DETECTED_NEGATIVE"] =
            "This pixel lies within an object's Footprint, and the detection was looking for pixels *below* "
            "a specified level.";
    _dict["SUSPECT"] = ++i;
    _docs["SUSPECT"] =
            "This pixel is untrustworthy (e.g. contains an instrumental flux in ADU above the correctable "
            "non-linear regime).";
    _dict["NO_DATA"] = ++i;
    _docs["NO_DATA"] =
            "There was no data at this pixel location (e.g. no input images at this location in a coadd, or "
            "extremely high vignetting, such that there is no incoming signal).";
}

MaskDict::MaskDict() : _dict(), _docs() {}

MaskDict::MaskDict(MaskPlaneDict const &dict, MaskPlaneDocDict const &docs) : _dict(dict), _docs(docs) {}

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst
