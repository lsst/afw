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

#include <cstdio>
#include <string>
#include <algorithm>

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

template <typename PixelT>
void print(afwImage::Image<PixelT>& src, const std::string& title = "") {
    if (title.size() > 0) {
        printf("%s:\n", title.c_str());
    }

    printf("%3s ", "");
    for (int x = 0; x != src.getWidth(); ++x) {
        printf("%4d ", x);
    }
    printf("\n");

    for (int y = src.getHeight() - 1; y >= 0; --y) {
        printf("%3d ", y);
        for (typename afwImage::Image<PixelT>::c_iterator src_it = src.row_begin(y); src_it != src.row_end(y);
            ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    afwImage::DecoratedImage<float> dimg(afwGeom::Extent2I(10, 6));
    afwImage::Image<float> img(*dimg.getImage());

    std::string file_u16;
    if (argc == 2) {
        file_u16 = std::string(argv[1]);
    } else {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            file_u16 = dataDir + "/data/small.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Error: provide fits file path as argument or setup afwdata.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Running with: " <<  file_u16 << std::endl;
    afwImage::DecoratedImage<float> dimg2(file_u16);

    return 0;
}
