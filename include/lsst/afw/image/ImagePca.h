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
 * \file
 * \brief Support for PCA analysis of 2-D images
 */
#ifndef LSST_AFW_IMAGE_IMAGEPCA_H
#define LSST_AFW_IMAGE_IMAGEPCA_H

#include <vector>
#include <string>
#include <utility>

#include "boost/mpl/bool.hpp"
#include <memory>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"

namespace lsst {
namespace afw {
namespace image {

    template <typename ImageT>
    class ImagePca {
    public:
        typedef typename std::shared_ptr<ImageT> Ptr;
        typedef typename std::shared_ptr<const ImageT> ConstPtr;

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
