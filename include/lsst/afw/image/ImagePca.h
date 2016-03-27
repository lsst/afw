// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
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

        explicit ImagePca(bool constantWeight=true);
        virtual ~ImagePca() {}

        void addImage(typename ImageT::Ptr img, double flux=0.0);
        ImageList getImageList() const;

        /// Return the dimension of the images being analyzed
        geom::Extent2I const getDimensions() const { return _dimensions; }

        typename ImageT::Ptr getMean() const;
        virtual void analyze();
        virtual double updateBadPixels(unsigned long mask, int const ncomp);
        
        /// Return Eigen values
        std::vector<double> const& getEigenValues() const { return _eigenValues; }
        /// Return Eigen images
        ImageList const& getEigenImages() const { return _eigenImages; }

    private:
        double getFlux(int i) const { return _fluxList[i]; }

        ImageList _imageList;           // image to analyze
        std::vector<double> _fluxList;  // fluxes of images
        geom::Extent2I _dimensions;      // width/height of images on _imageList

        bool _constantWeight;           // should all stars have the same weight?
        
        std::vector<double> _eigenValues; // Eigen values
        ImageList _eigenImages;           // Eigen images
    };

template <typename Image1T, typename Image2T>
double innerProduct(Image1T const& lhs, Image2T const& rhs, int const border=0);
    
}}}

#endif
