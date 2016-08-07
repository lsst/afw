/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"

namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

int main() {
    image::MaskedImage<int> mi(geom::Extent2I(10,10));
    image::Image<int>       im(mi.getDimensions());

    image::MaskedImage<int>::xy_locator mi_loc = mi.xy_at(5,5);
    image::Image<int>::xy_locator       im_loc = im.xy_at(5,5);

    std::pair<int, int> const step = std::make_pair(1,1);

    mi_loc += step;
    im_loc += step;

    return 0;
}
