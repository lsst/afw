// This file is part of afw.
//
// Developed for the LSST Data Management System.
// This product includes software developed by the LSST Project
// (http://www.lsst.org).
// See the COPYRIGHT file at the top-level directory of this distribution
// for details of code ownership.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef LSST_AFW_IMAGE_PYTHON_INDEXING_H
#define LSST_AFW_IMAGE_PYTHON_INDEXING_H

#include "pybind11/pybind11.h"
#include "lsst/afw/geom/Box.h"

namespace lsst { namespace afw { namespace image { namespace python {

inline void checkBounds(geom::Point2I const & index, geom::Box2I const & bbox) {
    if (!bbox.contains(index)) {
        std::string msg = (boost::format("Index (%d, %d) outside image bounds (%d, %d) to (%d, %d).")
                           % index.getX() % index.getY() % bbox.getMinX() % bbox.getMinY()
                           % bbox.getMaxX() % bbox.getMaxY()).str();
        if (index.getX() < 0 || index.getY() < 0) {
            msg += "  Note that negative indices are only interpreted as relative to the upper bound "
                   "when LOCAL coordinates are used.";
        }
        PyErr_SetString(PyExc_IndexError, msg.c_str());
        throw pybind11::error_already_set();
    }
}

}}}}  // namespace lsst::afw::image::python

#endif //! LSST_AFW_IMAGE_PYTHON_INDEXING_H

