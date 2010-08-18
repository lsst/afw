// -*- lsst-c++ -*-

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
 
/**
 * @file
 *
 * @brief Utilities to support PCA analysis of a set of images
 */
#include "Eigen/Core"
#include "Eigen/QR"

#include <algorithm>
#include "lsst/afw/image/ImagePca.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace afw {
namespace image {

/// ctor
template <typename ImageT>
ImagePca<ImageT>::ImagePca(bool constantWeight ///< Should all stars be weighted equally?
                          ) :
    _imageList(),
    _fluxList(),
    _width(0), _height(0),
    _constantWeight(constantWeight),
    _eigenValues(std::vector<double>()),
    _eigenImages(ImageList()) {
}

/**
 * Add an image to the set to be analyzed
 *
 * @throw lsst::pex::exceptions::LengthErrorException if all the images aren't the same size
 */
template <typename ImageT>
void ImagePca<ImageT>::addImage(typename ImageT::Ptr img, ///< Image to add to set
                                double flux               ///< Image's flux
                               ) {
    if (_imageList.empty()) {
        _width = img->getWidth();
        _height = img->getHeight();
    } else {
        if (getDimensions() != img->getDimensions()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              (boost::format("Dimension mismatch: saw %dx%d; expected %dx%d") %
                               img->getWidth() % img->getHeight() % _width % _height).str());
        }
    }

    if (flux == 0.0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException, "Flux may not be zero");
    }

    _imageList.push_back(img);
    _fluxList.push_back(flux);
}

/// Return the list of images being analyzed
template <typename ImageT>
typename ImagePca<ImageT>::ImageList ImagePca<ImageT>::getImageList() const {
    return _imageList;
}

/************************************************************************************************************/
/**
 * Return the mean of the images in ImagePca's list
 */
template <typename ImageT>
typename ImageT::Ptr ImagePca<ImageT>::getMean() const {
    if (_imageList.empty()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException, "You haven't provided any images");
    }

    typename ImageT::Ptr mean(new ImageT(getDimensions()));
    *mean = 0;

    for (typename ImageList::const_iterator ptr = _imageList.begin(), end = _imageList.end();
         ptr != end; ++ptr) {
        *mean += **ptr;
    }
    *mean /= _imageList.size();

    return mean;
}

/************************************************************************************************************/
/*
 * Analyze the images in an ImagePca, calculating the PCA decomposition (== Karhunen-Lo\`eve basis)
 *
 * The notation is that in chapter 7 of Gyula Szokoly's thesis at JHU
 */
namespace {
    template<typename T>
    struct SortEvalueDecreasing : public std::binary_function<std::pair<T, int> const&,
                                                              std::pair<T, int> const&, bool> {
        bool operator()(std::pair<T, int> const& a, std::pair<T, int> const& b) {
            return a.first > b.first;   // N.b. sort on greater
        }
    };
}

template <typename ImageT>
void ImagePca<ImageT>::analyze()
{
    int const nImage = _imageList.size();

    if (nImage == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          "Please provide at least one Image for me to analyze");
    }
    /*
     * Eigen doesn't like 1x1 matrices, but we don't really need it to handle a single matrix...
     */
    if (nImage == 1) {
        _eigenImages.clear();
        _eigenImages.push_back(typename ImageT::Ptr(new ImageT(*_imageList[0], true)));

        _eigenValues.clear();
        _eigenValues.push_back(1.0);

        return;
    }
    /*
     * Find the eigenvectors/values of the scalar product matrix, R' (Eq. 7.4)
     */
    Eigen::MatrixXd R(nImage, nImage);  // residuals' inner products

    double flux_bar = 0;              // mean of flux for all regions
    for (int i = 0; i != nImage; ++i) {
        ImageT const& im_i = *_imageList[i];
        double const flux_i = getFlux(i);
        flux_bar += flux_i;

        for (int j = i; j != nImage; ++j) {
            ImageT const& im_j = *_imageList[j];
            double const flux_j = getFlux(j);

            double dot = innerProduct(im_i, im_j);
            if (_constantWeight) {
                dot /= flux_i*flux_j;
            }
            R(i, j) = R(j, i) = dot/nImage;
        }
    }
    flux_bar /= nImage;
   
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eVecValues(R);
    Eigen::MatrixXd const& Q = eVecValues.eigenvectors();
    Eigen::VectorXd const& lambda = eVecValues.eigenvalues();
    //
    // We need to sort the eigenValues, and remember the permutation we applied to the eigenImages
    // We'll use the vector lambdaAndIndex to achieve this
    //
    std::vector<std::pair<double, int> > lambdaAndIndex; // pairs (eValue, index)
    lambdaAndIndex.reserve(nImage);

    for (int i = 0; i != nImage; ++i) {
        lambdaAndIndex.push_back(std::make_pair(lambda(i), i));
    }
    std::sort(lambdaAndIndex.begin(), lambdaAndIndex.end(), SortEvalueDecreasing<double>());
    //
    // Save the (sorted) eigen values
    //
    _eigenValues.clear();
    _eigenValues.reserve(nImage);
    for (int i = 0; i != nImage; ++i) {
        _eigenValues.push_back(lambdaAndIndex[i].first);
    }    
    //
    // Contruct the first ncomp eigenimages in basis
    //
    int ncomp = 100;                    // number of components to keep

    _eigenImages.clear();
    _eigenImages.reserve(ncomp < nImage ? ncomp : nImage);

    for(int i = 0; i < ncomp; ++i) {
        if(i >= nImage) {
            continue;
        }

        int const i = lambdaAndIndex[i].second; // the index after sorting (backwards) by eigenvalue

        typename ImageT::Ptr eImage(new ImageT(_width, _height));
        *eImage = 0;

        for (int j = 0; j != nImage; ++j) {
            int const jj = lambdaAndIndex[j].second; // the index after sorting (backwards) by eigenvalue
            double const weight = Q(jj, i)*(_constantWeight ? flux_bar/getFlux(jj) : 1);
            eImage->scaledPlus(weight, *_imageList[jj]);
        }

#define FIX_BKGD_LEVEL 0
#if FIX_BKGD_LEVEL
/*
 * Estimate and subtract the mean background level from the i > 0
 * eigen images; if we don't do that then PSF variation can get mixed
 * with subtle variations in the background and potentially amplify
 * them disasterously.
 *
 * It is not at all clear that doing this is a good idea; it'd be
 * better to get the sky level right in the first place.
 *
 * N.b. this is unconverted SDSS code, so it won't compile for LSST
 */
        if(i > 0) {                 /* not the zeroth KL component */
            float sky = 0;          /* estimate of sky level */
            REGION *sreg;           /* reg_i minus the border */

            reg_i = basis->regs[i-1][0][0]->reg;
            shAssert(reg_i->type == TYPE_FL32);

            for(j = border + 1; j < nrow - border - 1; j++) {
                sky += reg_i->rows_fl32[j][border] +
                    reg_i->rows_fl32[j][ncol - border - 1];
            }
            for(k = border; k < ncol - border; k++) {
                sky += reg_i->rows_fl32[border][k] +
                    reg_i->rows_fl32[nrow - border - 1][k];
            }
            sky /= 2*((nrow - 2*border) + (ncol - 2*border)) - 4;
 
            sreg = shSubRegNew("", basis->regs[i-1][0][0]->reg,
                               nrow - 2*border, ncol - 2*border,
                               border, border, NO_FLAGS);
            shAssert(sreg != NULL);
 
            shRegIntConstAdd(sreg, -sky, 0);
            shRegDel(sreg);
        }
#endif
        /*
         * Normalise eigenImages to have a maximum of 1.0.  For n > 0 they
         * (should) have mean == 0, so we can't use that to normalize
         */
        lsst::afw::math::Statistics stats =
            lsst::afw::math::makeStatistics(*eImage, (lsst::afw::math::MIN | lsst::afw::math::MAX));
        double const min = stats.getValue(lsst::afw::math::MIN);
        double const max = stats.getValue(lsst::afw::math::MAX);

        double const extreme = (fabs(min) > max) ? min :max;
        if (extreme != 0.0) {
            *eImage /= extreme;
        }

        _eigenImages.push_back(eImage);
    }
}

/*******************************************************************************************************/    
namespace {
    template<typename T, typename U>
    struct IsSame {
        IsSame(T const&, U const&) {}
        bool operator()() { return false; }
    };

    template<typename T>
    struct IsSame<T, T> {
        IsSame(T const& im1, T const& im2) : _same(im1.row_begin(0) == im2.row_begin(0)) {}
        bool operator()() { return _same; }
    private:
        bool _same;
    };

    // Test if two Images are identical; they need not be of the same type
    template<typename Image1T, typename Image2T>
    bool imagesAreIdentical(Image1T const& im1, Image2T const& im2) {
        return IsSame<Image1T, Image2T>(im1, im2)();
    }
}
/**
 * Calculate the inner product of two %images
 * @return The inner product
 * @throw lsst::pex::exceptions::LengthErrorException if all the images aren't the same size
 */
template <typename Image1T, typename Image2T>
double innerProduct(Image1T const& lhs, ///< first image
                    Image2T const& rhs, ///< Other image to dot with first
                    int border          ///< number of pixels to ignore around the edge
                   ) {
    if (lhs.getWidth() <= 2*border || lhs.getHeight() <= 2*border) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("All image pixels are in the border of width %d: %dx%d") %
                           border % lhs.getWidth() % lhs.getHeight()).str());
    }

    double sum = 0.0;
    //
    // Handle I.I specially for efficiency, and to avoid advancing the iterator twice
    //
    if (imagesAreIdentical(lhs, rhs)) {
        for (int y = border; y != lhs.getHeight() - border; ++y) {
            for (typename Image1T::const_x_iterator lptr = lhs.row_begin(y) + border,
                     lend = lhs.row_end(y) - border; lptr != lend; ++lptr) {
                typename Image1T::Pixel val = *lptr;
                sum += val*val;
            }
        }
    } else {
        if (lhs.getDimensions() != rhs.getDimensions()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              (boost::format("Dimension mismatch: %dx%d v. %dx%d") %
                               lhs.getWidth() % lhs.getHeight() % rhs.getWidth() % rhs.getHeight()).str());
        }

        for (int y = border; y != lhs.getHeight() - border; ++y) {
            typename Image2T::const_x_iterator rptr = rhs.row_begin(y) + border;
            for (typename Image1T::const_x_iterator lptr = lhs.row_begin(y) + border,
                     lend = lhs.row_end(y) - border; lptr != lend; ++lptr, ++rptr) {
                sum += (*lptr)*(*rptr);
            }
        }
    }

    return sum;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(T) \
    template class ImagePca<Image<T> >; \
    template double innerProduct(Image<T> const&, Image<T> const&, int);

#define INSTANTIATE2(T, U)                \
    template double innerProduct(Image<T> const&, Image<U> const&, int);    \
    template double innerProduct(Image<U> const&, Image<T> const&, int);

INSTANTIATE(boost::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)

INSTANTIATE2(float, double)             // the two types must be different
    
}}}
