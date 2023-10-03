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

MaskDict::MaskDict(int maxPlanes, bool _default)
        : _maxPlanes(maxPlanes), _dict(std::make_shared<MaskDictImpl>(_default)) {}

MaskDict::MaskDict(int maxPlanes, MaskPlaneDict const &dict, MaskPlaneDocDict const &docs)
        : _maxPlanes(maxPlanes), _dict(std::make_shared<MaskDictImpl>(dict, docs)) {}

int MaskDict::add(std::string name, std::string doc) {
    auto iter = _dict->_dict.find(name);
    // New name does not exist in the map.
    if (iter == _dict->_dict.end()) {
        int id = _maxPlanes;
        // check for empty bits we can re-use
        std::set<int> existingIds;
        for (auto const &item : _dict->_dict) {
            existingIds.insert(item.second);
        }
        // Use the first empty id.
        for (int i = 0; i < _maxPlanes; ++i) {
            if (existingIds.find(i) == existingIds.end()) {
                id = i;
                break;
            }
        }
        if (id >= _maxPlanes) {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::RuntimeError,
                    (boost::format("Max number of planes (%1%) already used when trying to add '%2%'") %
                     _maxPlanes % name)
                            .str());
        }
        _dict->_dict[name] = id;
        _dict->_docs[name] = doc;
        return id;
    }

    // New name already exists in the map.
    int id = iter->second;
    if (_dict->_docs.at(name) == doc || doc.empty()) {
        // Matching docs require no change.
        return id;
    } else if (_dict->_docs.at(name).empty()) {
        // Overwrite an existing empty docstring.
        _dict->_docs[name] = doc;
        return id;
    } else {
        // Don't allow changing an existing docstring.
        throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                (boost::format(
                         "Not changing existing docstring for plane '%1%'; remove and re-add to modify it.") %
                 name)
                        .str());
    }
}

void MaskDict::remove(std::string name) {
    if (_dict.use_count() != 1) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                (boost::format("Cannot remove plane '%1%'; there are '%2%' entities sharing this MaskDict.") %
                 name % _dict.use_count())
                        .str());
    }
    _dict->_dict.erase(name);
    _dict->_docs.erase(name);
}

void MaskDict::conformTo(MaskDict const &other) {
    // TODO
}

MaskDict MaskDict::clone() const { return MaskDict(_maxPlanes, _dict->_dict, _dict->_docs); }

int MaskDict::getPlaneId(std::string const &name) const {
    auto iter = _dict->_dict.find(name);
    return (iter == _dict->_dict.end()) ? -1 : iter->second;
}

std::string MaskDict::getPlaneDoc(std::string const &name) const {
    auto iter = _dict->_docs.find(name);
    return (iter == _dict->_docs.end()) ? "" : iter->second;
}

std::string MaskDict::print() const {
    std::ostringstream out;
    auto it_dict = _dict->_dict.begin();
    auto it_doc = _dict->_docs.begin();
    std::map<int, std::string> lines;
    while (it_dict != _dict->_dict.end()) {
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
    return this == &rhs || _dict == rhs._dict ||
           (_dict->_dict == rhs._dict->_dict && _dict->_docs == rhs._dict->_docs);
}

MaskDict::MaskDictImpl::MaskDictImpl(bool _default) {
    if (_default) {
        _addInitialMaskPlanes();
    }
}

MaskDict::MaskDictImpl::MaskDictImpl(MaskPlaneDict const &dict, MaskPlaneDocDict const &docs)
        : _dict(dict), _docs(docs) {}

void MaskDict::MaskDictImpl::_addInitialMaskPlanes() {
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

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst
