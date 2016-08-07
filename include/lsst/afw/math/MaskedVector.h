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

#if !defined(LSST_AFW_MATH_MASKEDVECTOR_H)
#define LSST_AFW_MATH_MASKEDVECTOR_H

#include <memory>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"

namespace lsst {
namespace afw {
namespace math {

template<typename EntryT>
class MaskedVector : private lsst::afw::image::MaskedImage<EntryT> {
public:
    //typedef typename lsst::afw::image::Mask<typename lsst::afw::image::MaskPixel>::MaskPlaneDict MaskPlaneDict;
    typedef typename lsst::afw::image::MaskedImage<EntryT>::Pixel Pixel;

    explicit MaskedVector(int width=0) : //, MaskPlaneDict const& planeDict=MaskPlaneDict()) :
        lsst::afw::image::MaskedImage<EntryT>(geom::Extent2I(width, 1)) {} //, planeDict) {}

    // Getters
    /// Return a (Ptr to) the MaskedImage's %image
    std::shared_ptr<std::vector<EntryT> > getVector(bool const noThrow=false) const {
        if (!this->getImage() && !noThrow) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "MaskedVector's Image is NULL");
        }

        std::shared_ptr<std::vector<EntryT> > imgcp(new std::vector<EntryT>(0));
        for (int i_y = 0; i_y < this->getImage()->getHeight(); ++i_y) {
            for (typename lsst::afw::image::Image<EntryT>::x_iterator ptr = this->getImage()->row_begin(i_y);
                 ptr != this->getImage()->row_end(i_y); ++ptr) {
                imgcp->push_back(*ptr);
            }
        }
        return imgcp;
    }

    // if we're asked for a single value return the image pixel
    //EntryT &operator[](int const i) { return (*lsst::afw::image::MaskedImage<EntryT>::getImage())(i, 0); }
    Pixel &operator[](int const i) {
        return Pixel(lsst::afw::image::MaskedImage<EntryT>::getImage()(i, 0),
                     lsst::afw::image::MaskedImage<EntryT>::getMask()(i, 0),
                     lsst::afw::image::MaskedImage<EntryT>::getVariance()(i, 0)
                    );
    }


    typename lsst::afw::image::MaskedImage<EntryT>::Image::Pixel &value(int const i) {
        return (*lsst::afw::image::MaskedImage<EntryT>::getImage())(i, 0);
    }
    typename lsst::afw::image::MaskedImage<EntryT>::Mask::Pixel &mask(int const i) {
        return (*lsst::afw::image::MaskedImage<EntryT>::getMask())(i, 0);
    }
    typename lsst::afw::image::MaskedImage<EntryT>::Variance::Pixel &variance(int const i) {
        return (*lsst::afw::image::MaskedImage<EntryT>::getVariance())(i, 0);
    }


    typename lsst::afw::image::MaskedImage<EntryT>::ImagePtr getImage() const {
        return lsst::afw::image::MaskedImage<EntryT>::getImage();
    }
    typename lsst::afw::image::MaskedImage<EntryT>::MaskPtr getMask() const {
        return lsst::afw::image::MaskedImage<EntryT>::getMask();
    }
    typename lsst::afw::image::MaskedImage<EntryT>::VariancePtr getVariance() const {
        return lsst::afw::image::MaskedImage<EntryT>::getVariance();
    }

    //MaskedVector& operator=(Pixel const& rhs) { return lsst::afw::image::MaskedImage<EntryT>::operator=(rhs); }
    //MaskedVector& operator=(SinglePixel const& rhs) { return this->operator=(rhs); }

    // Make some std::vector methods
    int size() { return this->getWidth(0); }
    bool empty() { return this->getWidth(0) == 0; }

    class iterator : public lsst::afw::image::MaskedImage<EntryT>::x_iterator {
    public:
#if 0
        using typename lsst::afw::image::MaskedImage<EntryT>::x_iterator::mask;
        using typename lsst::afw::image::MaskedImage<EntryT>::x_iterator::variance;
#endif

        iterator(typename lsst::afw::image::MaskedImage<EntryT>::Image::x_iterator im,
                 typename lsst::afw::image::MaskedImage<EntryT>::Mask::x_iterator msk,
                 typename lsst::afw::image::MaskedImage<EntryT>::Variance::x_iterator var) :
            lsst::afw::image::MaskedImage<EntryT>::x_iterator(im, msk, var) {}
        iterator(typename lsst::afw::image::MaskedImage<EntryT>::x_iterator ptr) :
            lsst::afw::image::MaskedImage<EntryT>::x_iterator(ptr) {}

        typename lsst::afw::image::MaskedImage<EntryT>::Image::Pixel &value() { return this->image(); }
    };

    iterator begin() { return this->row_begin(0); }
    iterator end()   { return this->row_end(0); }
private:

};

}}}

#endif
