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

#include "boost/functional/hash.hpp"
#include "lsst/afw/image/detail/MaskDict.h"

namespace lsst {
namespace afw {
namespace image {
namespace detail {

// A thread-safe singleton that manages MaskDict's global state.
class MaskDict::GlobalState final {
public:
    static GlobalState &get() {
        static GlobalState instance;
        return instance;
    }

    std::shared_ptr<MaskDict> copyOrGetDefault(MaskPlaneDict const &mpd, MaskPlaneDocDict const &docs) {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        if (!mpd.empty()) {
            std::shared_ptr<MaskDict> dict(new MaskDict(mpd, docs));
            _allMaskDicts.insert(dict);
            return dict;
        }
        return _defaultMaskDict;
    }

    std::shared_ptr<MaskDict> copy(MaskDict const &dict) {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        std::shared_ptr<MaskDict> result(new MaskDict(dict));
        _allMaskDicts.insert(result);
        return result;
    }

    std::shared_ptr<MaskDict> getDefault() const noexcept { return _defaultMaskDict; }

    void setDefault(std::shared_ptr<MaskDict> dict) { _defaultMaskDict = std::move(dict); }

    std::shared_ptr<MaskDict> detachDefault() {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        _defaultMaskDict = copy(*_defaultMaskDict);
        return _defaultMaskDict;
    }

    template <typename Functor>
    void forEachMaskDict(Functor functor) {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        _prune();  // guarantees dereference below is safe
        for (auto const &ptr : _allMaskDicts) {
            functor(*ptr.lock());
        }
    }

private:
    GlobalState() : _defaultMaskDict(new MaskDict()) {
        _allMaskDicts.insert(_defaultMaskDict);
        _defaultMaskDict->_addInitialMaskPlanes();
    }

    GlobalState(GlobalState const &) = delete;
    GlobalState(GlobalState &&) = delete;

    GlobalState &operator=(GlobalState const &) = delete;
    GlobalState &operator=(GlobalState &&) = delete;

    ~GlobalState() = default;

    // Prune expired weak_ptrs from _allMaskDicts.  Not thread safe; should
    // only be called by routines that have already locked the mutex.
    void _prune() {
        for (auto iter = _allMaskDicts.begin(); iter != _allMaskDicts.end();) {
            if (iter->expired()) {
                iter = _allMaskDicts.erase(iter);
            } else {
                ++iter;
            }
        }
    }

    std::recursive_mutex _mutex;  // guards _allMaskDicts and synchronizes updates to it and _defaultMaskDict
    std::set<std::weak_ptr<MaskDict>, std::owner_less<std::weak_ptr<MaskDict>>> _allMaskDicts;
    std::shared_ptr<MaskDict> _defaultMaskDict;
};

std::shared_ptr<MaskDict> MaskDict::copyOrGetDefault(MaskPlaneDict const &mpd, MaskPlaneDocDict const &docs) {
    return GlobalState::get().copyOrGetDefault(mpd, docs);
}

std::shared_ptr<MaskDict> MaskDict::getDefault() { return GlobalState::get().getDefault(); }

void MaskDict::setDefault(std::shared_ptr<MaskDict> dict) { GlobalState::get().setDefault(std::move(dict)); }

std::shared_ptr<MaskDict> MaskDict::detachDefault() { return GlobalState::get().detachDefault(); }

void MaskDict::addAllMasksPlane(std::string const &name, int bitId, std::string const &doc) {
    GlobalState::get().forEachMaskDict([&name, bitId, doc](MaskDict &dict) {
        auto const found = std::find_if(dict.begin(), dict.end(),
                                        [bitId](auto const &item) { return item.second == bitId; });
        if (found == dict.end()) {
            // is name already in use?
            if (dict.find(name) == dict.end()) {
                dict.add(name, bitId, doc);
            }
        }
    });
}

MaskDict::~MaskDict() noexcept = default;

std::shared_ptr<MaskDict> MaskDict::clone() const { return GlobalState::get().copy(*this); }

int MaskDict::getUnusedPlane() const {
    if (empty()) {
        return 0;
    }

    auto const maxIter = std::max_element(begin(), end(),
                                          [](auto const &a, auto const &b) { return a.second < b.second; });
    assert(maxIter != end());
    int id = maxIter->second + 1;  // The maskPlane to use if there are no gaps

    for (int i = 0; i < id; ++i) {
        // is i already used in this Mask?
        auto const sameIter =
                std::find_if(begin(), end(), [i](auto const &item) { return item.second == i; });
        if (sameIter == end()) {  // Not used; so we'll use it
            return i;
        }
    }

    return id;
}

int MaskDict::getMaskPlane(std::string const &name) const {
    auto iter = find(name);
    return (iter == end()) ? -1 : iter->second;
}

void MaskDict::print(std::ostream &out) const {
    auto it_dict = this->begin();
    auto it_doc = _docs.begin();
    std::map<int, std::string> lines;
    while (it_dict != this->end()) {
        std::stringstream line;
        line << "Plane " << it_dict->second << " -> " << it_dict->first << " : " << it_doc->second;
        lines[it_dict->second] = line.str();
        ++it_dict;
        ++it_doc;
    }
    for (auto const &line : lines) {
        out << line.second;
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
}

bool MaskDict::operator==(MaskDict const &rhs) const {
    return this == &rhs || (_hash == rhs._hash && _dict == rhs._dict);
}

void MaskDict::add(std::string const &name, int bitId, std::string const &doc) {
    _dict[name] = bitId;
    _docs[name] = doc;
    _hash = boost::hash<MaskPlaneDict>()(_dict);
}

void MaskDict::erase(std::string const &name) {
    _dict.erase(name);
    _docs.erase(name);
    _hash = boost::hash<MaskPlaneDict>()(_dict);
}

void MaskDict::clear() {
    _dict.clear();
    _docs.clear();
    _hash = 0x0;
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
    _hash = boost::hash<MaskPlaneDict>()(_dict);
}

MaskDict::MaskDict() : _dict(), _docs(), _hash(0x0) {}

MaskDict::MaskDict(MaskPlaneDict const &dict, MaskPlaneDocDict const &docs)
        : _dict(dict), _docs(docs), _hash(boost::hash<MaskPlaneDict>()(_dict)) {}

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst
