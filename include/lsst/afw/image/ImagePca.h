// -*- lsst-c++ -*-
/**
 * \file
 * \brief Support for PCA analysis of 2-D images
 */
#ifndef LSST_AFW_IMAGE_IMAGEPCA_H
#define LSST_AFW_IMAGE_IMAGEPCA_H

#include <vector>
#include <string>
#include <utility>

#include "boost/mpl/bool.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"

namespace lsst {
namespace afw {
namespace image {

    template <typename ImageT>
    class ImagePca {
    public:
        typedef typename boost::shared_ptr<ImageT> Ptr;
        typedef typename boost::shared_ptr<const ImageT> ConstPtr;

        typedef std::vector<typename ImageT::Ptr> ImageList;

        ImagePca();

        void addImage(typename ImageT::Ptr img, double flux=0.0);
        ImageList getImageList() const;

        /// Return the dimension of the images being analyzed
        const std::pair<int, int> getDimensions() const { return std::pair<int, int>(_width, _height); }

        typename ImageT::Ptr getMean() const;
        void analyze();
        /// Return Eigen values
        std::vector<double> const& getEigenValues() const { return _eigenValues; }
        /// Return Eigen images
        ImageList const& getEigenImages() const { return _eigenImages; }

    private:
        double getFlux(int i) const { return _fluxList[i]; }

        ImageList _imageList;           // image to analyze
        std::vector<double> _fluxList;  // fluxes of images
        int _width;                     // width of images on _imageList
        int _height;                    // height of images on _imageList

        std::vector<double> _eigenValues; // Eigen values
        ImageList _eigenImages;           // Eigen images
    };

template <typename ImageT>
double innerProduct(ImageT const& lhs, ImageT const& rhs);
    
}}}

#endif
