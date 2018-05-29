// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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

#ifndef LSST_AFW_GEOM_TRANSFORMUTILS_H
#define LSST_AFW_GEOM_TRANSFORMUTILS_H

#include <memory>
#include <vector>

#include "astshim.h"
#include "ndarray.h"

#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/daf/base/PropertyList.h"

namespace lsst {
namespace afw {
namespace geom {
namespace detail {
/// version of serialization used when writing (older versions may also be supported when reading)
constexpr int serializationVersion = 1;

/**
 * Deserialize a Transform from an input stream
 *
 * @tparam Transform  the Transform class; can be Transform<FromEndpoint, ToEndpoint>, SkyWcs,
 * or any other compatible class, i.e. it must support the following (see Transform.h for details):
 * - a constructor that takes an ast::FrameSet
 * - static method getShortClassName
 * - method getMapping
 *
 * @param[in] is  input stream from which to deserialize this Transform
 */
template<class Transform> 
std::shared_ptr<Transform> readStream(std::istream & is);

/**
 * Serialize a Transform to an output stream
 *
 * Version 1 format is as follows:
 * - The version number (an integer)
 * - A space
 * - The short class name, as obtained from getShortClassName
 * - A space
 * - The contained ast::FrameSet written using FrameSet.show(os, false)
 *
 * @param[out] os  output stream to which to serialize this Transform
 * @param[in] transform  Transform to serialize
 */
template<class Transform>
void writeStream(Transform const & transform, std::ostream & os);

/*
 * Provide definitions here in the header file to avoid the need for explicit instantiations
 */

template<class Transform> 
std::shared_ptr<Transform> readStream(std::istream & is) {
    int version;
    is >> version;
    if (version != 1) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "Unsupported version " + std::to_string(version));
    }
    std::string shortClassName;
    is >> shortClassName;
    if (shortClassName != Transform::getShortClassName()) {
        std::ostringstream os;
        os << "Class name in stream " << shortClassName << " != " << Transform::getShortClassName();
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }
    auto astStream = ast::Stream(&is, nullptr);
    auto astObjectPtr = ast::Channel(astStream).read();
    auto mapping = std::dynamic_pointer_cast<ast::Mapping>(astObjectPtr);
    if (!mapping) {
        std::ostringstream os;
        os << "The AST serialization was read as a " << astObjectPtr->getClassName()
           << " instead of a Mapping";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, os.str());
    }

    return std::make_shared<Transform>(*mapping);
}

template<class Transform>
void writeStream(Transform const & transform, std::ostream & os) {
    os << serializationVersion << " " << Transform::getShortClassName();
    transform.getMapping()->show(os, false);  // false = do not write comments
}

}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif
