// -*- lsst-c++ -*-
#if !defined(LSST_AFW_MATH_MASKEDVECTOR_H)
#define LSST_AFW_MATH_MASKEDVECTOR_H

#include "boost/shared_ptr.hpp"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"

namespace image = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {

template<typename EntryT>    
class MaskedVector : private image::MaskedImage<EntryT> {    
public:
    //typedef typename image::Mask<typename image::MaskPixel>::MaskPlaneDict MaskPlaneDict;
    typedef typename image::MaskedImage<EntryT>::Pixel Pixel;
    
    explicit MaskedVector(int width=0) : //, MaskPlaneDict const& planeDict=MaskPlaneDict()) :
        image::MaskedImage<EntryT>(width, 1) {} //, planeDict) {}

    // Getters
    /// Return a (Ptr to) the MaskedImage's %image
    boost::shared_ptr<std::vector<EntryT> > getVector(bool const noThrow=false) const {
        if (!this->getImage() && !noThrow) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "MaskedVector's Image is NULL");
        }

        boost::shared_ptr<std::vector<EntryT> > imgcp(new std::vector<EntryT>(0));
        for (int i_y = 0; i_y < this->getImage()->getHeight(); ++i_y) {
            for (typename image::Image<EntryT>::x_iterator ptr = this->getImage()->row_begin(i_y);
                 ptr != this->getImage()->row_end(i_y); ++ptr) {
                imgcp->push_back(*ptr);
            }
        }
        return imgcp;
    }

    // if we're asked for a single value return the image pixel
    //EntryT &operator[](int const i) { return (*image::MaskedImage<EntryT>::getImage())(i, 0); }
    Pixel &operator[](int const i) {
        return Pixel(image::MaskedImage<EntryT>::getImage()(i, 0),
                     image::MaskedImage<EntryT>::getMask()(i, 0),
                     image::MaskedImage<EntryT>::getVariance()(i, 0)
                    );
    }

    
    typename image::MaskedImage<EntryT>::Image::Pixel &value(int const i) {
        return (*image::MaskedImage<EntryT>::getImage())(i, 0);
    }
    typename image::MaskedImage<EntryT>::Mask::Pixel &mask(int const i) {
        return (*image::MaskedImage<EntryT>::getMask())(i, 0);
    }
    typename image::MaskedImage<EntryT>::Variance::Pixel &variance(int const i) {
        return (*image::MaskedImage<EntryT>::getVariance())(i, 0);
    }

    
    typename image::MaskedImage<EntryT>::ImagePtr getImage() const {
        return image::MaskedImage<EntryT>::getImage();
    }
    typename image::MaskedImage<EntryT>::MaskPtr getMask() const {
        return image::MaskedImage<EntryT>::getMask();
    }
    typename image::MaskedImage<EntryT>::VariancePtr getVariance() const {
        return image::MaskedImage<EntryT>::getVariance();
    }

    //MaskedVector& operator=(Pixel const& rhs) { return image::MaskedImage<EntryT>::operator=(rhs); }
    //MaskedVector& operator=(SinglePixel const& rhs) { return this->operator=(rhs); }

    // Make some std::vector methods
    int size() { return this->getWidth(0); }
    bool empty() { return this->getWidth(0) > 0; }
    
    class iterator : public image::MaskedImage<EntryT>::x_iterator {
    public:
#if 0
        using typename image::MaskedImage<EntryT>::x_iterator::mask;
        using typename image::MaskedImage<EntryT>::x_iterator::variance;
#endif

        iterator(typename image::MaskedImage<EntryT>::Image::x_iterator im,
                 typename image::MaskedImage<EntryT>::Mask::x_iterator msk,
                 typename image::MaskedImage<EntryT>::Variance::x_iterator var) :
            image::MaskedImage<EntryT>::x_iterator(im, msk, var) {}
        iterator(typename image::MaskedImage<EntryT>::x_iterator ptr) :
            image::MaskedImage<EntryT>::x_iterator(ptr) {}

        typename image::MaskedImage<EntryT>::Image::Pixel &value() { return this->image(); }
    };

    iterator begin() { return this->row_begin(0); }
    iterator end()   { return this->row_end(0); }
private:
    
};

}}}

#endif
