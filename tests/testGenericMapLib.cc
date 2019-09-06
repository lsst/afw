/*
 * This file is part of afw.
 *
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

#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>

#include "lsst/pex/exceptions.h"

#include "lsst/afw/typehandling/GenericMap.h"
#include "lsst/afw/typehandling/Storable.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace typehandling {

namespace {

/**
 * Test whether a map contains a key-value pair.
 *
 * @param map The map to test
 * @param key, value The key-value pair to test.
 *
 * @throws pex::exceptions::NotFoundError Thrown if the key is not present in
 *      the map, or maps to a different value.
 */
template <typename T>
void assertKeyValue(GenericMap<std::string> const& map, std::string const& key, T const& value) {
    using lsst::pex::exceptions::NotFoundError;

    if (!map.contains(key)) {
        throw LSST_EXCEPT(NotFoundError, "Map does not contain key " + key);
    }

    auto typedKey = makeKey<T>(key);
    if (!map.contains(typedKey)) {
        std::stringstream buffer;
        buffer << "Map maps " << key << " to a different type than " << typedKey;
        throw LSST_EXCEPT(NotFoundError, buffer.str());
    }

    T const& mapValue = map.at(typedKey);
    if (mapValue != value) {
        std::stringstream buffer;
        buffer << "Map maps " << typedKey << " to " << mapValue << ", expected " << value;
        throw LSST_EXCEPT(NotFoundError, buffer.str());
    }
}

}  // namespace

namespace {

// Functions for working with values of arbitrary type
template <typename T>
void declareAnyTypeFunctions(py::module& mod) {
    mod.def("assertKeyValue",
            static_cast<void (*)(GenericMap<std::string> const&, std::string const&, T const&)>(
                    &assertKeyValue),
            "map"_a, "key"_a, "value"_a);
}

}  // namespace

PYBIND11_MODULE(testGenericMapLib, mod) {
    py::module::import("lsst.afw.typehandling");

    declareAnyTypeFunctions<long>(mod);
    declareAnyTypeFunctions<double>(mod);
    declareAnyTypeFunctions<std::string>(mod);
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst
