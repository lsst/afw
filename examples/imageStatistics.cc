// -*- LSST-C++ -*-

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

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

#include "lsst/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

#include "lsst/afw/math/MaskedVector.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageF;
typedef image::MaskedImage<float> MaskedImageF;
typedef math::Statistics ImgStat;
typedef math::MaskedVector<float> MaskedVectorF;

/*
 * An example of how to use the Statistics class
 */

template <typename Image>
void printStats(Image &img, math::StatisticsControl const &sctrl) {
    // initialize a Statistics object with any stats we might want
    ImgStat stats = math::makeStatistics(
            img,
            math::NPOINT | math::STDEV | math::MEAN | math::VARIANCE | math::ERRORS | math::MIN | math::MAX |
                    math::VARIANCECLIP | math::MEANCLIP | math::MEDIAN | math::IQRANGE | math::STDEVCLIP,
            sctrl);

    // get various stats with getValue() and their errors with getError()
    double const npoint = stats.getValue(math::NPOINT);
    double const mean = stats.getValue(math::MEAN);
    double const var = stats.getValue(math::VARIANCE);
    double const dmean = stats.getError(math::MEAN);
    double const sd = stats.getValue(math::STDEV);
    double const min = stats.getValue(math::MIN);
    double const max = stats.getValue(math::MAX);
    double const meanclip = stats.getValue(math::MEANCLIP);
    double const varclip = stats.getValue(math::VARIANCECLIP);
    double const stdevclip = stats.getValue(math::STDEVCLIP);
    double const median = stats.getValue(math::MEDIAN);
    double const iqrange = stats.getValue(math::IQRANGE);

    // output
    std::cout << "N          " << npoint << std::endl;
    std::cout << "dmean      " << dmean << std::endl;

    std::cout << "mean:      " << mean << std::endl;
    std::cout << "meanclip:  " << meanclip << std::endl;

    std::cout << "var:       " << var << std::endl;
    std::cout << "varclip:   " << varclip << std::endl;

    std::cout << "stdev:     " << sd << std::endl;
    std::cout << "stdevclip: " << stdevclip << std::endl;

    std::cout << "min:       " << min << std::endl;
    std::cout << "max:       " << max << std::endl;
    std::cout << "median:    " << median << std::endl;
    std::cout << "iqrange:   " << iqrange << std::endl;
    std::cout << std::endl;
}

int main() {
    // declare an image and a masked image
    int const wid = 1024;
    ImageF img(lsst::geom::Extent2I(wid, wid));
    MaskedImageF mimg(img.getDimensions());
    std::vector<double> v(0);
    MaskedVectorF mv(wid * wid);

    // fill it with some noise (Cauchy noise in this case)
    for (int j = 0; j != img.getHeight(); ++j) {
        int k = 0;
        MaskedImageF::x_iterator mip = mimg.row_begin(j);
        for (ImageF::x_iterator ip = img.row_begin(j); ip != img.row_end(j); ++ip) {
            double const xUniform = M_PI * static_cast<ImageF::Pixel>(std::rand()) / RAND_MAX;
            double xLorentz = xUniform;  // tan(xUniform - M_PI/2.0);

            // throw in the occassional nan ... 1% of the time
            if (static_cast<double>(std::rand()) / RAND_MAX < 0.01) {
                xLorentz = NAN;
            }

            *ip = xLorentz;

            // mask the odd rows
            // variance actually diverges for Cauchy noise ... but stats doesn't access this.
            *mip = MaskedImageF::Pixel(xLorentz, (k % 2) ? 0x1 : 0x0, (k % 2) ? 1.0e99 : 1.0);

            v.push_back(xLorentz);
            ++k;
            ++mip;
        }
    }

    int j = 0;
    for (MaskedVectorF::iterator mvp = mv.begin(); mvp != mv.end(); ++mvp) {
        *mvp = MaskedVectorF::Pixel(v[j], (j % 2) ? 0x1 : 0x0, 10.0);
        ++j;
    }

    std::shared_ptr<std::vector<float> > vF = mv.getVector();

    // make a statistics control object and override some of the default properties
    math::StatisticsControl sctrl;
    sctrl.setNumIter(3);
    sctrl.setNumSigmaClip(5.0);
    sctrl.setAndMask(0x1);  // pixels with this mask bit set will be ignored.
    sctrl.setNanSafe(true);

    // ==================================================================
    // Get stats for the Image, MaskedImage, and vector
    std::cout << "image::Image" << std::endl;
    printStats(img, sctrl);
    std::cout << "image::MaskedImage" << std::endl;
    printStats(mimg, sctrl);
    std::cout << "std::vector" << std::endl;
    printStats(v, sctrl);
    std::cout << "image::MaskedVector" << std::endl;
    printStats(mv, sctrl);
    std::cout << "image::MaskedVector::getVector()" << std::endl;
    printStats(*vF, sctrl);

    // Now try the weighted statistics
    sctrl.setWeighted(true);
    sctrl.setAndMask(0x0);
    std::cout << "image::MaskedImage (weighted)" << std::endl;
    printStats(mimg, sctrl);

    // Now try the specialization to get NPOINT and SUM (bitwise OR) for an image::Mask
    math::Statistics mskstat = makeStatistics(*mimg.getMask(), (math::NPOINT | math::SUM), sctrl);
    std::cout << "image::Mask" << std::endl;
    std::cout << mskstat.getValue(math::NPOINT) << " " << mskstat.getValue(math::SUM) << std::endl;

    return 0;
}
