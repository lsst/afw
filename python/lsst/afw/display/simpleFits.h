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

/*
 * Definitions to write a FITS image
 */
#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/geom/SkyWcs.h"

namespace lsst {
namespace afw {
namespace display {

template <typename ImageT>
void writeBasicFits(int fd, ImageT const& data, lsst::afw::geom::SkyWcs const* Wcs = nullptr,
                    char const* title = nullptr, std::shared_ptr<daf::base::PropertySet> fits_metadata = nullptr);

template <typename ImageT>
void writeBasicFits(std::string const& filename, ImageT const& data, lsst::afw::geom::SkyWcs const* Wcs = nullptr,
                    const char* title = nullptr, std::shared_ptr<daf::base::PropertySet> fits_metadata = nullptr);
}
}
}  // namespace lsst::afw::display
#endif
