/*
 * Calculate some scalings.  Legacy code from NAOJ, which could probably be rewritten in python
 * if the need arose.  It could certainly use more LSST primitives (e.g. line fitting)
 */
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <stdexcept>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"

namespace lsst {
namespace afw {
namespace display {

template <class T>
static void getSample(image::Image<T> const& image, std::size_t const nSamples, std::vector<T>& vSample) {
    int const width = image.getWidth();
    int const height = image.getHeight();

    // extract, from image, about nSample samples
    // such that they form a grid.
    vSample.reserve(nSamples);

    int const initialStride =
            std::max(1, static_cast<int>(std::sqrt(width * height / static_cast<double>(nSamples))));

    for (int stride = initialStride; stride >= 1; --stride) {
        vSample.clear();

        for (std::size_t y = 0; y < height; y += stride) {
            for (std::size_t x = 0; x < width; x += stride) {
                T const elem = image(x, y);
                if (std::isfinite(elem)) {
                    vSample.push_back(elem);
                }
            }
        }

        // if more than 80% of nSamples were sampled, OK.
        if (5 * vSample.size() > 4 * nSamples) {
            break;
        }
    }
}

static inline double computeSigma(std::vector<double> const& vFlat, std::vector<int> const& vBadPix,
                                  std::size_t const nGoodPix) {
    if (nGoodPix <= 1) {
        return 0;
    }

    double sumz = 0, sumsq = 0;
    for (unsigned i = 0; i < vFlat.size(); ++i) {
        if (!vBadPix[i]) {
            double const z = vFlat[i];

            sumz += z;
            sumsq += z * z;
        }
    }

    double const goodPix = nGoodPix;
    double const tmp = sumsq / (goodPix - 1) - sumz * sumz / (goodPix * (goodPix - 1));

    return (tmp > 0) ? std::sqrt(tmp) : 0;
}

template <class T>
static std::pair<double, double> fitLine(
        int* nGoodPixOut,  // returned; it'd be nice to use std::tuple from C++11
        std::vector<T> const& vSample, double const nSigmaClip, int const nGrow, int const minpix,
        int const nIter) {
    // map the indices of vSample to [-1.0, 1.0]
    double const xscale = 2.0 / (vSample.size() - 1);
    std::vector<double> xnorm;
    xnorm.reserve(vSample.size());
    for (std::size_t i = 0; i < vSample.size(); ++i) {
        xnorm.push_back(i * xscale - 1.0);
    }

    // Mask that is used in k-sigma clipping
    std::vector<int> vBadPix(vSample.size(), 0);

    std::size_t nGoodPix = vSample.size();
    std::size_t nGoodPixOld = nGoodPix + 1;

    // values to be obtained
    double intercept = 0;
    double slope = 0;

    for (int iteration = 0; iteration < nIter; ++iteration) {
        if (nGoodPix < minpix || nGoodPix >= nGoodPixOld) {
            break;
        }

        double sum = nGoodPix;
        double sumx = 0, sumy = 0, sumxx = 0, sumxy = 0;
        for (std::size_t i = 0; i < vSample.size(); ++i) {
            if (!vBadPix[i]) {
                double const x = xnorm[i];
                double const y = vSample[i];

                sumx += x;
                sumy += y;
                sumxx += x * x;
                sumxy += x * y;
            }
        }

        double delta = sum * sumxx - sumx * sumx;

        // slope and intercept
        intercept = (sumxx * sumy - sumx * sumxy) / delta;
        slope = (sum * sumxy - sumx * sumy) / delta;

        // residue
        std::vector<double> vFlat;
        vFlat.reserve(vSample.size());
        for (unsigned i = 0; i < vSample.size(); ++i) {
            vFlat.push_back(vSample[i] - (xnorm[i] * slope + intercept));
        }

        // Threshold of k-sigma clipping
        double const sigma = computeSigma(vFlat, vBadPix, nGoodPix);
        double const hcut = sigma * nSigmaClip;
        double const lcut = -hcut;

        // revise vBadPix
        nGoodPixOld = nGoodPix;
        nGoodPix = 0;

        for (unsigned i = 0; i < vSample.size(); ++i) {
            double val = vFlat[i];
            if (val < lcut || hcut < val) {
                vBadPix[i] = 1;
            }
        }

        // blurr vBadPix
        std::vector<int> vBadPix_new;
        vBadPix_new.reserve(vSample.size());
        for (unsigned x = 0; x < vSample.size(); ++x) {
            int imin = (static_cast<int>(x) > nGrow) ? x - nGrow : -1;
            int val = 0;
            for (int i = x; i > imin; --i) {
                val += vBadPix[i];
            }
            vBadPix_new.push_back(val ? 1 : 0);
            if (!val) {
                ++nGoodPix;
            }
        }
        vBadPix = vBadPix_new;
    }

    // return the scale of x-axis
    *nGoodPixOut = nGoodPix;

    return std::make_pair(intercept - slope, slope * xscale);
}

template <class T>
std::pair<double, double> getZScale(image::Image<T> const& image, int const nSamples, double const contrast) {
    // extract samples
    std::vector<T> vSample;
    getSample(image, nSamples, vSample);
    int nPix = vSample.size();

    if (vSample.empty()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "ZScale: No pixel in image is finite");
    }

    std::sort(vSample.begin(), vSample.end());

    // max, min, median
    // N.b. you can get a median in linear time, but we need the sorted array for fitLine()
    // If we wanted to speed this up, the best option would be to quantize
    // the pixel values and build a histogram
    double const zmin = vSample.front();
    double const zmax = vSample.back();
    int const iCenter = nPix / 2;
    T median = (nPix & 1) ? vSample[iCenter] : (vSample[iCenter] + vSample[iCenter + 1]) / 2;

    // fit a line to the sorted sample
    const int maxRejectionRatio = 2;
    const int npixelsMin = 5;

    int minpix = std::max(npixelsMin, nPix / maxRejectionRatio);
    int nGrow = std::max(1, nPix / 100);

    const double nSigmaClip = 2.5;
    const int nIterations = 5;

    int nGoodPix = 0;
    std::pair<double, double> ret = fitLine(&nGoodPix, vSample, nSigmaClip, nGrow, minpix, nIterations);
#if 0  // unused, but calculated and potentially useful
    double const zstart = ret.first;
#endif
    double const zslope = ret.second;

    double z1, z2;
    if (nGoodPix < minpix) {
        z1 = zmin;
        z2 = zmax;
    } else {
        double const slope = zslope / contrast;

        z1 = std::max(zmin, median - iCenter * slope);
        z2 = std::min(zmax, median + (nPix - iCenter - 1) * slope);
    }

    return std::make_pair(z1, z2);
}
//
// Explicit instantiations
#define INSTANTIATE_GETZSCALE(T)                                                                   \
    template std::pair<double, double> getZScale(image::Image<T> const& image, int const nSamples, \
                                                 double const contrast)

INSTANTIATE_GETZSCALE(std::uint16_t);
INSTANTIATE_GETZSCALE(float);
}
}
}
