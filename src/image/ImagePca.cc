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
#include <algorithm>
#include "boost/make_shared.hpp"
#include "lsst/utils/ieee.h"

#include "Eigen/Core"
#include "Eigen/QR"
#include "Eigen/SVD"

#include "lsst/afw/image/ImagePca.h"
#include "lsst/afw/math/Statistics.h"

namespace afwMath = lsst::afw::math;

namespace lsst {
namespace afw {
namespace image {

/// ctor
template <typename ImageT>
ImagePca<ImageT>::ImagePca(bool constantWeight ///< Should all stars be weighted equally?
                          ) :
    _imageList(),
    _fluxList(),
    _dimensions(0,0),
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
        _dimensions = img->getDimensions();
    } else {
        if (getDimensions() != img->getDimensions()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                (boost::format("Dimension mismatch: saw %dx%d; expected %dx%d") %
                    img->getWidth() % img->getHeight() %
                    _dimensions.getX() % _dimensions.getY()
                ).str()
            );
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
    *mean = static_cast<typename ImageT::Pixel>(0);

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
/*
 * Some metafunctions to extract an Image::Ptr from a MaskedImage::Ptr (or return the original Image::Ptr)
 *
 * GetImage is the public interface (it forwards the tag --- just for the sake of the UI); the real work
 * is in GetImage_ which defines a typedef for the Image and a static function, getImage
 *
 * E.g.
 * In the function
 *
 * template<typename ImageT>
 * void func(typename ImageT::Ptr image) {
 *    typename GetImage<ImageT>::type::Ptr im = GetImage<ImageT>::getImage(image);
 * }
 *
 * "im" is an Image::Ptr irrespective of whether ImageT is Masked or not.
 */
    template<typename ImageT, typename TagT>
    struct GetImage_ {
        typedef ImageT type;
        static typename type::Ptr getImage(typename ImageT::Ptr image) {
            return image;
        }
    };

    template<typename ImageT>
    struct GetImage_<ImageT, typename image::detail::MaskedImage_tag> {
        typedef typename ImageT::Image type;
        static typename type::Ptr getImage(typename ImageT::Ptr image) {
            return image->getImage();
        }
    };

    template<typename ImageT>
    struct GetImage : public GetImage_<ImageT, typename ImageT::image_category> {
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
        typename GetImage<ImageT>::type const& im_i = *GetImage<ImageT>::getImage(_imageList[i]);
        double const flux_i = getFlux(i);
        flux_bar += flux_i;

        for (int j = i; j != nImage; ++j) {
            typename GetImage<ImageT>::type const& im_j = *GetImage<ImageT>::getImage(_imageList[j]);
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

        int const ii = lambdaAndIndex[i].second; // the index after sorting (backwards) by eigenvalue

        typename ImageT::Ptr eImage(new ImageT(_dimensions));
        *eImage = static_cast<typename ImageT::Pixel>(0);

        for (int j = 0; j != nImage; ++j) {
            int const jj = lambdaAndIndex[j].second; // the index after sorting (backwards) by eigenvalue
            double const weight = Q(jj, ii)*(_constantWeight ? flux_bar/getFlux(jj) : 1);
            eImage->scaledPlus(weight, *_imageList[jj]);
        }
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
#define FIX_BKGD_LEVEL 1
#if FIX_BKGD_LEVEL
/*
 * Estimate and subtract the mean background level from the i > 0
 * eigen images; if we don't do that then PSF variation can get mixed
 * with subtle variations in the background and potentially amplify
 * them disasterously.
 *
 * It is not at all clear that doing this is a good idea; it'd be
 * better to get the sky level right in the first place.
 */
        if(i > 0) {                 /* not the zeroth KL component */
#if 0                               // use the median of non-detected pixels
            afw::math::StatisticsControl sctrl;
            sctrl.setAndMask(Mask<>::getPlaneBitMask("DETECTED"));

            double const med = afwMath::makeStatistics(*eImage.getImage(), afwMath::MEDIAN, sctrl).getValue();
            //std::cout << "Eigen image " << i << "  median " << med << std::endl;
#else                               // use the median of the edge pixels, in a region of with border
            // If ImageT is a MaskedImage, unpack the Image
            typename GetImage<ImageT>::type::Ptr eImageIm = GetImage<ImageT>::getImage(eImage);

            int border = 3;
            int const height = eImage->getHeight();
            int const width = eImage->getWidth();
            if (width < 2*border || height < 2*border) {
                continue;               // not enough pixels
            }
            int const nEdge = width*height - (width - 2*border)*(height - 2*border);
            std::vector<double> edgePixels(nEdge);

            std::vector<double>::iterator bi = edgePixels.begin();

            typedef typename GetImage<ImageT>::type::x_iterator imIter;
            int y = 0;
            for(; y != border; ++y) {   // Bottom border of eImage
                for (imIter ptr = eImageIm->row_begin(y), end = eImageIm->row_end(y); ptr != end; ++ptr, ++bi) {
                    *bi = *ptr;
                }
            }
            for(; y != height - border; ++y) {   // Left and right borders of eImage
                for (imIter ptr = eImageIm->row_begin(y),
                         end = eImageIm->x_at(border, y); ptr != end; ++ptr, ++bi) {
                    *bi = *ptr;
                }
                for (imIter ptr = eImageIm->x_at(width - border, y),
                         end = eImageIm->row_end(y); ptr != end; ++ptr, ++bi) {
                    *bi = *ptr;
                }
            }
            for(; y != height; ++y) {   // Top border of eImage
                for (imIter ptr = eImageIm->row_begin(y), end = eImageIm->row_end(y); ptr != end; ++ptr, ++bi) {
                    *bi = *ptr;
                }
            }
            assert(distance(edgePixels.begin(), bi) == nEdge);
            
            double const med = afwMath::makeStatistics(edgePixels, afwMath::MEDIAN).getValue();
            //std::cout << "vector " << i << "  median " << med << std::endl;
#endif 
            *eImageIm -= med;
        }
#endif

        _eigenImages.push_back(eImage);
    }
}

/************************************************************************************************************/
/*
 * 
 */
namespace {
/*
 * Fit a LinearCombinationKernel to an Image, allowing the coefficients of the components to vary
 *
 * return std::pair(best-fit kernel, std::pair(amp, chi^2))
 */
template<typename MaskedImageT>
typename MaskedImageT::Image::Ptr fitEigenImagesToImage(
        typename ImagePca<MaskedImageT>::ImageList const& eigenImages, // Eigen images
        int nEigen,                                                    // Number of eigen images to use
        MaskedImageT const& image                                      // The image to be fit
                                                 )
{
    typedef typename MaskedImageT::Image ImageT;

    if (nEigen == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          "You must have at least one eigen image");
    } else if (nEigen > static_cast<int>(eigenImages.size())) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("You only have %d eigen images (you asked for %d)")
                           % eigenImages.size() % nEigen).str());
    }
    /*
     * Solve the linear problem  image = sum x_i K_i + epsilon; we solve this for x_i by constructing the
     * normal equations, A x = b
     */
    Eigen::MatrixXd A(nEigen, nEigen);
    Eigen::VectorXd b(nEigen);

    for (int i = 0; i != nEigen; ++i) {
        b(i) = innerProduct(*eigenImages[i]->getImage(), *image.getImage());

        for (int j = i; j != nEigen; ++j) {
            A(i, j) = A(j, i) = innerProduct(*eigenImages[i]->getImage(), *eigenImages[j]->getImage());
        }
    }
    Eigen::VectorXd x(nEigen);

    A.svd().solve(b, &x);
    //
    // Accumulate the best-fit-image in bestFitImage
    //
    typename ImageT::Ptr bestFitImage = boost::make_shared<ImageT>(eigenImages[0]->getDimensions());

    for (int i = 0; i != nEigen; ++i) {
        bestFitImage->scaledPlus(x[i], *eigenImages[i]->getImage());
    }
    
    return bestFitImage;
}

/************************************************************************************************************/

template <typename ImageT>
double do_updateBadPixels(detail::basic_tag const&,
                        typename ImagePca<ImageT>::ImageList const&,
                        std::vector<double> const&,
                        typename ImagePca<ImageT>::ImageList const&,
                        unsigned long, int const)
{
    return 0.0;
}

template <typename ImageT>
double do_updateBadPixels(
        detail::MaskedImage_tag const&,
        typename ImagePca<ImageT>::ImageList const& imageList,
        std::vector<double> const& fluxes,   // fluxes of images
        typename ImagePca<ImageT>::ImageList const& eigenImages, // Eigen images
        unsigned long mask, ///< Mask defining bad pixels
        int const ncomp     ///< Number of components to use in estimate
                                                                )
{
    int const nImage = imageList.size();

    if (nImage == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          "Please provide at least one Image for me to update");
    }
    geom::ExtentI dimensions = imageList[0]->getDimensions();
    int const height = dimensions.getY();
        
    double maxChange = 0.0;             // maximum change to the input images

    if (ncomp == 0) {                   // use mean of good pixels
        typename ImageT::Image mean(dimensions); // desired mean image
        image::Image<float> weight(mean.getDimensions()); // weight of each pixel

        for (int i = 0; i != nImage; ++i) {
            double const flux_i = fluxes[i];

            for (int y = 0; y != height; ++y) {
                typename ImageT::const_x_iterator iptr = imageList[i]->row_begin(y);
                image::Image<float>::x_iterator wptr = weight.row_begin(y);
                for (typename ImageT::Image::x_iterator mptr = mean.row_begin(y), end = mean.row_end(y);
                     mptr != end; ++mptr, ++iptr, ++wptr) {
                    if (!(iptr.mask() & mask)) {
                        typename ImageT::Image::Pixel value = iptr.image()/flux_i;
                        float const var = iptr.image()/(flux_i*flux_i);
                        float const ivar = 1.0/var;
                        if (!lsst::utils::isfinite(ivar)) {
                            continue;
                        }

                        *mptr += value*ivar;
                        *wptr += ivar;
                    }
                }
            }
        }
        //
        // Calculate mean
        //
        for (int y = 0; y != height; ++y) {
            image::Image<float>::x_iterator wptr = weight.row_begin(y);
            for (typename ImageT::Image::x_iterator mptr = mean.row_begin(y), end = mean.row_end(y);
                 mptr != end; ++mptr, ++wptr) {
                *mptr /= *wptr;
            }
        }
        //
        // Replace bad values by mean
        //
        for (int i = 0; i != nImage; ++i) {
            double const flux_i = fluxes[i];

            for (int y = 0; y != height; ++y) {
                typename ImageT::x_iterator iptr = imageList[i]->row_begin(y);
                for (typename ImageT::Image::x_iterator mptr = mean.row_begin(y), end = mean.row_end(y);
                     mptr != end; ++mptr, ++iptr) {
                    if ((iptr.mask() & mask)) {
                        double const delta = ::fabs(flux_i*(*mptr) - iptr.image());
                        if (delta > maxChange) {
                            maxChange = delta;
                        }
                        iptr.image() = flux_i*(*mptr);
                    }
                }
            }
        }
    } else {
        if (ncomp > static_cast<int>(eigenImages.size())) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              (boost::format("You only have %d eigen images (you asked for %d)")
                               % eigenImages.size() % ncomp).str());
        }

        for (int i = 0; i != nImage; ++i) {
            typename ImageT::Image::Ptr fitted = fitEigenImagesToImage(eigenImages, ncomp, *imageList[i]);

            for (int y = 0; y != height; ++y) {
                typename ImageT::x_iterator iptr = imageList[i]->row_begin(y);
                for (typename ImageT::Image::const_x_iterator fptr = fitted->row_begin(y),
                         end = fitted->row_end(y); fptr != end; ++fptr, ++iptr) {
                    if (iptr.mask() & mask) {
                        double const delta = ::fabs(*fptr - iptr.image());
                        if (delta > maxChange) {
                            maxChange = delta;
                        }

                        iptr.image() = *fptr;
                    }
                }
            }
        }
    }

    return maxChange;
}
}
/**
 * Update the bad pixels (i.e. those for which (value & mask) != 0) based on the current PCA decomposition;
 * if none is available, use the mean of the good pixels
 *
 * \return the maximum change made to any pixel
 *
 * N.b. the work is actually done in do_updateBadPixels as the code only makes sense and compiles when we are
 * doing a PCA on a set of MaskedImages
 */
template <typename ImageT>
double ImagePca<ImageT>::updateBadPixels(
        unsigned long mask, ///< Mask defining bad pixels
        int const ncomp     ///< Number of components to use in estimate
                                      )
{
    return do_updateBadPixels<ImageT>(typename ImageT::image_category(),
                                      _imageList, _fluxList, _eigenImages, mask, ncomp);
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
template class ImagePca<MaskedImage<float> >;
INSTANTIATE(double)

INSTANTIATE2(float, double)             // the two types must be different
    
}}}
