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
 * \brief Implementation of the Class MaskedImage
 */

#ifndef LSST_IMAGE_MASKEDIMAGE_H
#define LSST_IMAGE_MASKEDIMAGE_H

#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include "boost/mpl/at.hpp"
#include "boost/iterator/zip_iterator.hpp"
#include "boost/tuple/tuple.hpp" // cannot convert to std::tuple (yet) because of use with boost::gil

#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/formatters/MaskedImageFormatter.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"

namespace lsst {
namespace afw {
namespace image {
    namespace detail {
        /// A traits class for MaskedImage
        struct MaskedImage_tag : public basic_tag { };
        /// A class used to identify classes that represent MaskedImage pixels
        struct MaskedImagePixel_tag { };

        std::string const fitsFile_RE = "\\.fits(\\.[fg]z)?$"; /// regexp to identify when MaskedImages should
                                                           /// be written as MEFs
        std::string const compressedFileNoMEF_RE = "(\\.gz)$";   /// regexp to identify compressed files that we can't write MEFs to
    }
}}}

#include "lsst/afw/image/Pixel.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace lsst {
namespace afw {
    namespace formatters {
        template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT> class MaskedImageFormatter;
    }

namespace image {

/// A class to manipulate images, masks, and variance as a single object
template<typename ImagePixelT, typename MaskPixelT=lsst::afw::image::MaskPixel,
         typename VariancePixelT=lsst::afw::image::VariancePixel>
class MaskedImage : public lsst::daf::base::Persistable,
                    public lsst::daf::base::Citizen {
public:
    /// shared pointer to the Image
    typedef typename Image<ImagePixelT>::Ptr ImagePtr;
    /// shared pointer to the Mask
    typedef typename Mask<MaskPixelT>::Ptr MaskPtr;
    /// shared pointer to the variance Image
    typedef typename Image<VariancePixelT>::Ptr VariancePtr;
    /// shared pointer to a MaskedImage
    typedef std::shared_ptr<MaskedImage> Ptr;
    typedef std::shared_ptr<const MaskedImage> ConstPtr;
    /// The Mask's MaskPlaneDict
    typedef typename Mask<MaskPixelT>::MaskPlaneDict MaskPlaneDict;

    typedef lsst::afw::image::Image<VariancePixelT> Variance; // These need to be here, and in this order, as
    typedef lsst::afw::image::Image<ImagePixelT> Image;       // "typedef Image::Ptr ImagePtr;" confuses swig (it can't
    typedef lsst::afw::image::Mask<MaskPixelT> Mask;          // find ImagePtr) and we can't use Image<> after these typedefs

    typedef detail::MaskedImage_tag image_category;

#if !defined(SWIG)
    /// A templated class to return this classes' type (present in Image/Mask/MaskedImage)
    template<typename ImagePT=ImagePixelT, typename MaskPT=MaskPixelT, typename VarPT=VariancePixelT>
    struct ImageTypeFactory {
        /// Return the desired type
        typedef MaskedImage<ImagePT, MaskPT, VarPT> type;
    };
#endif

    /************************************************************************************************************/
    /// An iterator to the MaskedImage
    template<typename, typename, typename> class MaskedImageIterator;
    /// An const iterator to the MaskedImage
    template<typename, typename, typename> class const_MaskedImageIterator;
    /// A locator for the MaskedImage
    template<typename, typename, typename> class MaskedImageLocator;
    /// A const locator for the MaskedImage
    template<typename, typename, typename> class const_MaskedImageLocator;

    /************************************************************************************************************/

#if !defined(SWIG)
    /// A Pixel in the MaskedImage
    typedef lsst::afw::image::pixel::Pixel<ImagePixelT, MaskPixelT, VariancePixelT> Pixel;
    /// A single Pixel of the same type as those in the MaskedImage
    typedef lsst::afw::image::pixel::SinglePixel<ImagePixelT, MaskPixelT, VariancePixelT> SinglePixel;

    /************************************************************************************************************/
    /// The base class for MaskedImageIterators (const and non-const)
    template<typename ImageIterator, typename MaskIterator, typename VarianceIterator,
             template<typename> class Ref=Reference>
    class MaskedImageIteratorBase {
        typedef boost::tuple<ImageIterator, MaskIterator, VarianceIterator> IMV_iterator_tuple;

    public:
        /// The underlying iterator tuple
        /// \note not really for public consumption;  could be made protected
        typedef typename boost::zip_iterator<IMV_iterator_tuple>::reference IMV_tuple;
        /// The underlying const iterator tuple
        /// \note not really for public consumption;  could be made protected
        template<typename, typename, typename> friend class const_MaskedImageIterator;
        /// Type pointed to by the iterator
        typedef Pixel type;

        /// Construct a MaskedImageIteratorBase from the image/mask/variance iterators
        MaskedImageIteratorBase(ImageIterator const& img, MaskIterator const& msk, VarianceIterator const &var) :
            _iter(boost::make_zip_iterator(boost::make_tuple(img, msk, var))) {
        }
        /// Return (a reference to) the image part of the Pixel pointed at by the iterator
        typename Ref<ImagePixelT>::type image() {
            return _iter->template get<0>()[0];
        }

        /// Return (a reference to) the mask part of the Pixel pointed at by the iterator
        typename Ref<MaskPixelT>::type mask() {
            return _iter->template get<1>()[0];
        }

        /// Return (a reference to) the variance part of the Pixel pointed at by the iterator
        typename Ref<VariancePixelT>::type variance() {
            return _iter->template get<2>()[0];
        }

        /// Return the underlying iterator tuple
        /// \note not really for public consumption;  could be made protected
        const IMV_iterator_tuple get_iterator_tuple() const {
            return _iter.get_iterator_tuple();
        }

        /// Increment the iterator by \c delta
        MaskedImageIteratorBase& operator+=(std::ptrdiff_t delta ///< how far to move the iterator
                       ) {
            _iter += delta;
            return *this;
        }
        /// Decrement the iterator by \c delta
        MaskedImageIteratorBase& operator-=(std::ptrdiff_t delta ///< how far to move the iterator
                       ) {
            _iter -= delta;
            return *this;
        }
        /// Increment the iterator (prefix)
        MaskedImageIteratorBase& operator++() {         // prefix
            ++_iter;
            return *this;
        }
        /// Increment the iterator (postfix)
        MaskedImageIteratorBase operator++(int) {      // postfix
            MaskedImageIteratorBase tmp(*this);
            _iter++;
            return tmp;
        }
        /// Return the distance between two iterators
        std::ptrdiff_t operator-(MaskedImageIteratorBase const& rhs) {
            return &this->_iter->template get<0>() - &rhs._iter->template get<0>();
        }
        /// Return true if the lhs equals the rhs
        bool operator==(MaskedImageIteratorBase const& rhs) {
            return &this->_iter->template get<0>() == &rhs._iter->template get<0>();
        }
        /// Return true if the lhs doesn't equal the rhs
        bool operator!=(MaskedImageIteratorBase const& rhs) {
            return &this->_iter->template get<0>() != &rhs._iter->template get<0>();
        }
        /// Return true if the lhs is less than the rhs
        bool operator<(MaskedImageIteratorBase const& rhs) {
            return &this->_iter->template get<0>() < &rhs._iter->template get<0>();
        }
        /// Convert an iterator to a Pixel
        operator Pixel() const {
            return Pixel(_iter->template get<0>()[0],
                         _iter->template get<1>()[0],
                         _iter->template get<2>()[0]);
        }

        /// Dereference the iterator, returning a Pixel
        Pixel operator*() {
            return Pixel(image(), mask(), variance());
        }
        /// Dereference the iterator, returning a const Pixel
        const Pixel operator*() const {
            return Pixel(image(), mask(), variance());
        }

    protected:
        typename boost::zip_iterator<IMV_iterator_tuple> _iter;
    };

    /// An iterator for a MaskedImage
    template<typename ImageIterator, typename MaskIterator, typename VarianceIterator>
    class MaskedImageIterator :
        public  MaskedImageIteratorBase<ImageIterator, MaskIterator, VarianceIterator> {
        typedef MaskedImageIteratorBase<ImageIterator, MaskIterator, VarianceIterator> MaskedImageIteratorBase_t;
    public:
        MaskedImageIterator(ImageIterator& img, MaskIterator& msk, VarianceIterator &var) :
            MaskedImageIteratorBase_t(img, msk, var) {
        }
        /// Return a MaskedImageIterator that's delta beyond this
        MaskedImageIterator operator+(std::ptrdiff_t delta) {
            MaskedImageIterator lhs = *this;
            lhs += delta;

            return lhs;
        }
    };

    /// An const iterator for a MaskedImage
    template<typename ImageIterator, typename MaskIterator, typename VarianceIterator>
    class const_MaskedImageIterator :
        public  MaskedImageIteratorBase<typename detail::const_iterator_type<ImageIterator>::type,
                                        typename detail::const_iterator_type<MaskIterator>::type,
                                        typename detail::const_iterator_type<VarianceIterator>::type,
                                        ConstReference> {

        typedef typename detail::const_iterator_type<ImageIterator>::type const_ImageIterator;
        typedef typename detail::const_iterator_type<MaskIterator>::type const_MaskIterator;
        typedef typename detail::const_iterator_type<VarianceIterator>::type const_VarianceIterator;

        typedef MaskedImageIteratorBase<const_ImageIterator, const_MaskIterator,
                                        const_VarianceIterator, ConstReference> MaskedImageIteratorBase_t;
    public:
        const_MaskedImageIterator(MaskedImageIterator<ImageIterator, MaskIterator, VarianceIterator> const& iter) :
            MaskedImageIteratorBase_t(const_ImageIterator(iter.get_iterator_tuple().template get<0>()),
                                      const_MaskIterator(iter.get_iterator_tuple().template get<1>()),
                                      const_VarianceIterator(iter.get_iterator_tuple().template get<2>())
                                     ) {
            ;
        }
        /// Return a const_MaskedImageIterator that's delta beyond this
        const_MaskedImageIterator& operator+(std::ptrdiff_t delta) {
            const_MaskedImageIterator lhs = *this;
            lhs += delta;

            return lhs;
        }
    };

    /************************************************************************************************************/
    /// The base class for MaskedImageLocator%s (const and non-const)
    template<typename ImageLocator, typename MaskLocator, typename VarianceLocator,
             template<typename> class Ref=Reference>
    class MaskedImageLocatorBase {
        typedef typename boost::tuple<ImageLocator, MaskLocator, VarianceLocator> IMVLocator;
        //
        // A class to provide _[xy]_iterator for MaskedImageLocator.  We can't just use
        // a zip_iterator as moving this iterator must be the same as moving the locator
        // itself, for consistency with {Image,Mask}::xy_locator
        //
        template<template<typename> class X_OR_Y >
        class _x_or_y_iterator {
        public:
            _x_or_y_iterator(MaskedImageLocatorBase* mil) : _mil(mil) {}

            _x_or_y_iterator& operator+=(const int di) {
                // Equivalent to "_mil->_loc.template get<0>().x() += di;"
                X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())() += di;
                X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())() += di;
                X_OR_Y<VarianceLocator>(_mil->_loc.template get<2>())() += di;
                return *this;
            }

            _x_or_y_iterator& operator++() {     // prefix
                // Equivalent to "++_mil->_loc.template get<0>().x();"
                ++X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())();
                ++X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())();
                ++X_OR_Y<VarianceLocator>(_mil->_loc.template get<2>())();
                return *this;
            }

            bool operator==(_x_or_y_iterator const& rhs) {
                return
                    X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())() ==
                    X_OR_Y<ImageLocator>(rhs._mil->_loc.template get<0>())();

            }
            bool operator!=(_x_or_y_iterator const& rhs) {
                return
                    X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())() !=
                    X_OR_Y<ImageLocator>(rhs._mil->_loc.template get<0>())();

            }
            bool operator<(_x_or_y_iterator const& rhs) {
                return
                    X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())() <
                    X_OR_Y<ImageLocator>(rhs._mil->_loc.template get<0>())();

            }

            Pixel operator*() {
                return Pixel((*(X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())()))[0],
                             (*(X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())()))[0],
                             (*(X_OR_Y<VarianceLocator>(_mil->_loc.template get<2>())()))[0]);
            }

            typename Ref<ImagePixelT>::type image() {
                // Equivalent to "return (*_mil->_loc.template get<0>().x())[0];"

                return (*(X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())()))[0];
            }
            typename Ref<MaskPixelT>::type mask() {
                return (*(X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())()))[0];
            }
            typename Ref<VariancePixelT>::type variance() {
                return (*(X_OR_Y<VarianceLocator>(_mil->_loc.template get<2>())()))[0];
            }
        protected:
            MaskedImageLocatorBase *_mil;
        };

        // Two classes to provide .x() and .y() in _x_or_y_iterator
        template<typename LocT>
        class apply_x {
            typedef typename LocT::x_iterator IterT;
        public:
            apply_x(LocT &loc) : _loc(loc) { }
            IterT& operator()() { return _loc.x(); }
        private:
            LocT& _loc;
        };

        template<typename LocT>
        class apply_y {
            typedef typename LocT::y_iterator IterT;
        public:
            apply_y(LocT &loc) : _loc(loc) { }
            IterT& operator()() { return _loc.y(); }
        private:
            LocT& _loc;
        };

    public:
        typedef typename boost::tuple<typename ImageLocator::cached_location_t,
                                      typename MaskLocator::cached_location_t,
                                      typename VarianceLocator::cached_location_t> IMVCachedLocation;
        /// An x_iterator that provides a view of the xy_locator (i.e. advancing one advances the other)
        typedef _x_or_y_iterator<apply_x> x_iterator;
        /// A y_iterator that provides a view of the xy_locator (i.e. advancing one advances the other)
        typedef _x_or_y_iterator<apply_y> y_iterator;
        /// A saved relative position, providing efficient access to neighbouring pixels
        class cached_location_t {
        public:
            //template<typename, typename, typename, template<typename> class> friend class MaskedImageLocatorBase;
            template<typename, typename, typename> friend class const_MaskedImageLocator;

            /// Create a cached_location_t that can be used to access pixels <tt>(x, y)</tt> away from \c loc
            cached_location_t(IMVLocator const& loc, int x, int y) :
                _imv(loc.template get<0>().cache_location(x, y),
                     loc.template get<1>().cache_location(x, y),
                     loc.template get<2>().cache_location(x, y)) {
                ;
            }
            //protected:
            IMVCachedLocation _imv;
        };
        /// Construct a MaskedImageLocator from %image/mask/variance locators
        MaskedImageLocatorBase(ImageLocator const& img, MaskLocator const& msk, VarianceLocator const& var) :
            _loc(img, msk, var) {
            ;
        }

        /// Dereference a locator, returning a Pixel
        Pixel operator*() {
            return Pixel(_loc.template get<0>().x()[0][0],
                         _loc.template get<1>().x()[0][0],
                         _loc.template get<2>().x()[0][0]);
        }

        /// Dereference a locator, returning a Pixel offset by <tt>(x, y)</tt> from the locator
        Pixel operator()(int x, int y) {
            return Pixel(_loc.template get<0>()(x, y)[0],
                         _loc.template get<1>()(x, y)[0],
                         _loc.template get<2>()(x, y)[0]);
        }

        /// Dereference a locator, returning a Pixel offset by the amount set when we created the \c cached_location_t
        Pixel operator[](cached_location_t const& cached_loc) {
            return Pixel(_loc.template get<0>()[cached_loc._imv.template get<0>()][0],
                         _loc.template get<1>()[cached_loc._imv.template get<1>()][0],
                         _loc.template get<2>()[cached_loc._imv.template get<2>()][0]);
        }
        /// Return an iterator that can be used to move (or dereference) a locator
        ///
        /// \note this x_locator is xy_locator::x_locator, not MaskedImage::x_locator
        x_iterator x() {
            return x_iterator(this);
        }

        /// Return an iterator that can be used to move (or dereference) a locator
        ///
        /// \note this y_locator is xy_locator::y_locator, not MaskedImage::y_locator
        y_iterator y() {
            return y_iterator(this);
        }

        /// Create a cached_location_t offset by <tt>(x, y)</tt> from locator
        cached_location_t cache_location(int x, int y) const {
            return cached_location_t(_loc, x, y);
        }
        //
        // We don't want to duplicate code for image/mask/variance -- but the boost::mpl stuff isn't pretty
        // as we can't say int_<N> within a template<int N>.  So define a set of functions apply_IMV
        // to do the dirty work
        //
        typedef typename boost::mpl::vector<ImagePixelT, MaskPixelT, VariancePixelT> PixelTVec;

        template<typename N>
        typename Ref<typename boost::mpl::at<PixelTVec, N>::type>::type apply_IMV(cached_location_t const& cached_loc) {
            return _loc.template get<N::value>()[cached_loc._imv.template get<N::value>()][0];
        }

        template<typename N>
        typename Ref<typename boost::mpl::at<PixelTVec, N>::type>::type apply_IMV() {
            return _loc.template get<N::value>()[0][0];
        }

        template<typename N>
        typename Ref<typename boost::mpl::at<PixelTVec, N>::type>::type apply_IMV(int x, int y) {
            return _loc.template get<N::value>()(x, y)[0];
        }
        //
        // Use those templated classes to implement image/mask/variance
        //
        /// Return a reference to the %image at the offset set when we created the \c cached_location_t
        typename Ref<ImagePixelT>::type image(cached_location_t const& cached_loc) {
            return apply_IMV<boost::mpl::int_<0> >(cached_loc);
        }
        /// Return a reference to the %image at the current position of the locator
        typename Ref<ImagePixelT>::type image() {
            return apply_IMV<boost::mpl::int_<0> >();
        }
        /// Return a reference to the %image offset by <tt>(x, y)</tt> from the current position of the locator
        typename Ref<ImagePixelT>::type image(int x, int y) {
            return apply_IMV<boost::mpl::int_<0> >(x, y);
        }

        /// Return a reference to the mask at the offset set when we created the \c cached_location_t
        typename Ref<MaskPixelT>::type mask(cached_location_t const& cached_loc) {
            return apply_IMV<boost::mpl::int_<1> >(cached_loc);
        }
        /// Return a reference to the mask at the current position of the locator
        typename Ref<MaskPixelT>::type mask() {
            return apply_IMV<boost::mpl::int_<1> >();
        }
        /// Return a reference to the mask offset by <tt>(x, y)</tt> from the current position of the locator
        typename Ref<MaskPixelT>::type mask(int x, int y) {
            return apply_IMV<boost::mpl::int_<1> >(x, y);
        }

        /// Return a reference to the variance at the offset set when we created the \c cached_location_t
        typename Ref<VariancePixelT>::type variance(cached_location_t const& cached_loc) {
            return apply_IMV<boost::mpl::int_<2> >(cached_loc);
        }
        /// Return a reference to the variance at the current position of the locator
        typename Ref<VariancePixelT>::type variance() {
            return apply_IMV<boost::mpl::int_<2> >();
        }
        /// Return a reference to the variance offset by <tt>(x, y)</tt> from the current position of the locator
        typename Ref<VariancePixelT>::type variance(int x, int y) {
            return apply_IMV<boost::mpl::int_<2> >(x, y);
        }

        /// Return true iff two locators are equal
        bool operator==(MaskedImageLocatorBase const& rhs) {
            return _loc.template get<0>() == rhs._loc.template get<0>();
        }
        /// Return true iff two locators are not equal
        bool operator!=(MaskedImageLocatorBase const& rhs) {
            return !(*this == rhs);
        }
        /// Return true iff lhs is less than rhs
        bool operator<(MaskedImageLocatorBase const& rhs) {
            return _loc.template get<0>() < rhs._loc.template get<0>();
        }

        /// Increment the locator's \c x and \c y positions by \c p
        MaskedImageLocatorBase& operator+=(pair2I const& p) {
            return operator+=(detail::difference_type(p.first, p.second));
        }

        /// Increment the locator's \c x and \c y positions by \c p
        MaskedImageLocatorBase& operator+=(detail::difference_type p) {
            _loc.template get<0>() += p;
            _loc.template get<1>() += p;
            _loc.template get<2>() += p;

            return *this;
        }

        // Workaround for DM-5590: clang-3.8 cannot access _loc from
        // friend class const_MaskedImageLocator.
        IMVLocator const & getLoc() const { return _loc; }

    protected:
        IMVLocator _loc;
    };

    /// A locator for a MaskedImage
    template<typename ImageLocator, typename MaskLocator, typename VarianceLocator>
    class MaskedImageLocator :
        public  MaskedImageLocatorBase<ImageLocator, MaskLocator, VarianceLocator> {
        typedef MaskedImageLocatorBase<ImageLocator, MaskLocator, VarianceLocator> MaskedImageLocatorBase_t;
    public:
        MaskedImageLocator(ImageLocator& img, MaskLocator& msk, VarianceLocator &var) :
            MaskedImageLocatorBase_t(img, msk, var) {
        }
    };

    /// A const locator for a MaskedImage
    template<typename ImageLocator, typename MaskLocator, typename VarianceLocator>
    class const_MaskedImageLocator :
        public  MaskedImageLocatorBase<typename detail::const_locator_type<ImageLocator>::type,
                                       typename detail::const_locator_type<MaskLocator>::type,
                                       typename detail::const_locator_type<VarianceLocator>::type,
                                       ConstReference> {

        typedef typename detail::const_locator_type<ImageLocator>::type const_ImageLocator;
        typedef typename detail::const_locator_type<MaskLocator>::type const_MaskLocator;
        typedef typename detail::const_locator_type<VarianceLocator>::type const_VarianceLocator;

        typedef MaskedImageLocatorBase<const_ImageLocator, const_MaskLocator, const_VarianceLocator,
                                       ConstReference> MaskedImageLocatorBase_t;
    public:
        const_MaskedImageLocator(MaskedImageLocator<ImageLocator, MaskLocator, VarianceLocator> const& iter) :
            MaskedImageLocatorBase_t(const_ImageLocator(iter.getLoc().template get<0>()),
                                     const_MaskLocator(iter.getLoc().template get<1>()),
                                     const_VarianceLocator(iter.getLoc().template get<2>())
                                    ) {
            ;
        }
    };

#endif  // !defined(SWIG)

/************************************************************************************************************/
    // An iterator to a MaskedImage
    typedef MaskedImageIterator<typename Image::iterator,
                                typename Mask::iterator, typename Variance::iterator> iterator;
    // A const_iterator to a MaskedImage
    typedef const_MaskedImageIterator<typename Image::iterator,
                                typename Mask::iterator, typename Variance::iterator> const_iterator;
    // A reverse_iterator to a MaskedImage
    typedef MaskedImageIterator<typename Image::reverse_iterator,
                                typename Mask::reverse_iterator, typename Variance::reverse_iterator> reverse_iterator;
#if 0                                   // doesn't compile.  I should fix this, but it's low priority. RHL
    /// a const_reverse_iterator
    typedef const_MaskedImageIterator<typename Image::reverse_iterator,
                                typename Mask::reverse_iterator, typename Variance::reverse_iterator> const_reverse_iterator;
#endif
    /// An iterator to a row of a MaskedImage
    typedef MaskedImageIterator<typename Image::x_iterator,
                                typename Mask::x_iterator, typename Variance::x_iterator> x_iterator;
    /// A const_iterator to a row of a MaskedImage
    typedef const_MaskedImageIterator<typename Image::x_iterator,
                                typename Mask::x_iterator, typename Variance::x_iterator> const_x_iterator;
    /// A fast STL compliant iterator for contiguous images
    /// N.b. The order of pixel access is undefined
    typedef x_iterator fast_iterator;
    /// An iterator to a column of a MaskedImage
    typedef MaskedImageIterator<typename Image::y_iterator,
                                typename Mask::y_iterator, typename Variance::y_iterator> y_iterator;
    /// A const_iterator to a column of a MaskedImage
    typedef const_MaskedImageIterator<typename Image::y_iterator,
                                typename Mask::y_iterator, typename Variance::y_iterator> const_y_iterator;

    /// A locator for a MaskedImage
    typedef MaskedImageLocator<typename Image::xy_locator,
                                typename Mask::xy_locator, typename Variance::xy_locator> xy_locator;
    /// A const_locator for a MaskedImage
    typedef const_MaskedImageLocator<typename Image::xy_locator,
                                typename Mask::xy_locator, typename Variance::xy_locator> const_xy_locator;

    /// an x_iterator associated with an xy_locator
    typedef typename MaskedImageLocator<typename Image::xy_locator,
                               typename Mask::xy_locator, typename Variance::xy_locator>::x_iterator xy_x_iterator;
    /// an y_iterator associated with an xy_locator
    typedef typename MaskedImageLocator<typename Image::xy_locator,
                               typename Mask::xy_locator, typename Variance::xy_locator>::y_iterator xy_y_iterator;

    /************************************************************************************************************/

    // Constructors
    explicit MaskedImage(
        unsigned int width, unsigned int height,
        MaskPlaneDict const& planeDict=MaskPlaneDict()
    );
    explicit MaskedImage(
        geom::Extent2I const & dimensions=geom::Extent2I(),
        MaskPlaneDict const& planeDict=MaskPlaneDict()
    );
    explicit MaskedImage(
        ImagePtr image,
        MaskPtr mask = MaskPtr(),
        VariancePtr variance = VariancePtr()
    );
    explicit MaskedImage(
        geom::Box2I const & bbox,
        MaskPlaneDict const& planeDict=MaskPlaneDict()
    );

    /**
     *  @brief Construct a MaskedImage by reading a regular FITS file.
     *
     *  @param[in]      fileName      File to read.
     *  @param[in,out]  metadata      Metadata read from the primary HDU header.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *  @param[in]      needAllHdus   If true, throw fits::FitsError if the mask and/or variance plane is
     *                                missing.  If false, silently initialize them to zero.
     *  @param[in,out]  imageMetadata      Metadata read from the image HDU header.
     *  @param[in,out]  maskMetadata       Metadata read from the mask HDU header.
     *  @param[in,out]  varianceMetadata   Metadata read from the variance HDU header.
     */
    explicit MaskedImage(
        std::string const & fileName,
        PTR(daf::base::PropertySet) metadata=PTR(daf::base::PropertySet)(),
        geom::Box2I const & bbox=geom::Box2I(), ImageOrigin origin=PARENT,
        bool conformMasks=false, bool needAllHdus=false,
        PTR(daf::base::PropertySet) imageMetadata=PTR(daf::base::PropertySet)(),
        PTR(daf::base::PropertySet) maskMetadata=PTR(daf::base::PropertySet)(),
        PTR(daf::base::PropertySet) varianceMetadata=PTR(daf::base::PropertySet)()
    );

    /**
     *  @brief Construct a MaskedImage by reading a FITS image in memory.
     *
     *  @param[in]      manager       An object that manages the memory buffer to read.
     *  @param[in,out]  metadata      Metadata read from the primary HDU header.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *  @param[in]      needAllHdus   If true, throw fits::FitsError if the mask and/or variance plane is
     *                                missing.  If false, silently initialize them to zero.
     *  @param[in,out]  imageMetadata      Metadata read from the image HDU header.
     *  @param[in,out]  maskMetadata       Metadata read from the mask HDU header.
     *  @param[in,out]  varianceMetadata   Metadata read from the variance HDU header.
     */
    explicit MaskedImage(
        fits::MemFileManager & manager,
        PTR(daf::base::PropertySet) metadata=PTR(daf::base::PropertySet)(),
        geom::Box2I const & bbox=geom::Box2I(), ImageOrigin origin=PARENT,
        bool conformMasks=false, bool needAllHdus=false,
        PTR(daf::base::PropertySet) imageMetadata=PTR(daf::base::PropertySet)(),
        PTR(daf::base::PropertySet) maskMetadata=PTR(daf::base::PropertySet)(),
        PTR(daf::base::PropertySet) varianceMetadata=PTR(daf::base::PropertySet)()
    );

    /**
     *  @brief Construct a MaskedImage from an already-open FITS object.
     *
     *  @param[in]      fitsfile      A FITS object to read from.  Current HDU is ignored.
     *  @param[in,out]  metadata      Metadata read from the primary HDU header.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *  @param[in]      needAllHdus   If true, throw fits::FitsError if the mask and/or variance plane is
     *                                missing.  If false, silently initialize them to zero.
     *  @param[in,out]  imageMetadata      Metadata read from the image HDU header.
     *  @param[in,out]  maskMetadata       Metadata read from the mask HDU header.
     *  @param[in,out]  varianceMetadata   Metadata read from the variance HDU header.
     */
    explicit MaskedImage(
        fits::Fits & fitsfile,
        PTR(daf::base::PropertySet) metadata=PTR(daf::base::PropertySet)(),
        geom::Box2I const & bbox=geom::Box2I(), ImageOrigin origin=PARENT,
        bool conformMasks=false, bool needAllHdus=false,
        PTR(daf::base::PropertySet) imageMetadata=PTR(daf::base::PropertySet)(),
        PTR(daf::base::PropertySet) maskMetadata=PTR(daf::base::PropertySet)(),
        PTR(daf::base::PropertySet) varianceMetadata=PTR(daf::base::PropertySet)()
    );

    MaskedImage(
        MaskedImage const& rhs,
        bool const deep=false
    );
    MaskedImage(
        MaskedImage const & rhs,
        geom::Box2I const & bbox,
        ImageOrigin const origin=PARENT,
        bool const deep=false
    );
    /// generalised copy constructor; defined here in the header so that the compiler can instantiate
    /// N(N-1)/2 conversions between N ImageBase types.
    ///
    /// We only support converting the Image part
    template<typename OtherPixelT>
    MaskedImage(
        MaskedImage<OtherPixelT, MaskPixelT, VariancePixelT> const& rhs, //!< Input image

        const bool deep     //!< Must be true; needed to disambiguate
    ) :
        lsst::daf::base::Citizen(typeid(this)), _image(), _mask(), _variance() {
        if (!deep) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                "Only deep copies are permitted for MaskedImages with different pixel types");
        }

        _image =    typename Image::Ptr(new Image(*rhs.getImage(), deep));
        _mask =     typename Mask::Ptr(new Mask(*rhs.getMask(), deep));
        _variance = typename Variance::Ptr(new Variance(*rhs.getVariance(), deep));
    }

#if defined(DOXYGEN)
    MaskedImage& operator=(MaskedImage const& rhs);
#endif

    virtual ~MaskedImage() {}

    void swap(MaskedImage &rhs);

    // Operators
    MaskedImage& operator=(Pixel const& rhs);
    MaskedImage& operator=(SinglePixel const& rhs);

    MaskedImage& operator<<=(MaskedImage const& rhs);

    void assign(MaskedImage const &rsh, geom::Box2I const &bbox = geom::Box2I(), ImageOrigin origin=PARENT);

    MaskedImage& operator+=(ImagePixelT const rhs);
    MaskedImage& operator+=(MaskedImage const& rhs);
    MaskedImage& operator+=(lsst::afw::image::Image<ImagePixelT> const& rhs) {
        *_image += rhs;
        return *this;
    }
    MaskedImage& operator+=(lsst::afw::math::Function2<double> const& function) {
        *_image += function;
        return *this;
    }
    void scaledPlus(double const c, MaskedImage const& rhs);

    MaskedImage& operator-=(ImagePixelT const rhs);
    MaskedImage& operator-=(MaskedImage const& rhs);
    MaskedImage& operator-=(lsst::afw::image::Image<ImagePixelT> const& rhs) {
        *_image -= rhs;
        return *this;
    }
    MaskedImage& operator-=(lsst::afw::math::Function2<double> const& function) {
        *_image -= function;
        return *this;
    }
    void scaledMinus(double const c, MaskedImage const& rhs);

    MaskedImage& operator*=(ImagePixelT const rhs);
    MaskedImage& operator*=(MaskedImage const& rhs);
    MaskedImage& operator*=(lsst::afw::image::Image<ImagePixelT> const& rhs) {
        *_image *= rhs;
        *_variance *= rhs;           // yes, multiply twice
        *_variance *= rhs;
        return *this;
    }
    void scaledMultiplies(double const c, MaskedImage const& rhs);

    MaskedImage& operator/=(ImagePixelT const rhs);
    MaskedImage& operator/=(MaskedImage const& rhs);
    MaskedImage& operator/=(lsst::afw::image::Image<ImagePixelT> const& rhs) {
        *_image /= rhs;
        *_variance /= rhs; // yes, divide twice
        *_variance /= rhs;
        return *this;
    }
    void scaledDivides(double const c, MaskedImage const& rhs);

    /**
     *  @brief Write a MaskedImage to a regular FITS file.
     *
     *  @param[in] fileName      Name of the file to write.  When writing separate files, this is
     *                           the "base" of the filename (e.g. foo reads foo_{img.msk.var}.fits).
     *  @param[in] metadata      Additional values to write to the primary HDU header (may be null).
     *  @param[in] imageMetadata      Metadata to be written to the image header.
     *  @param[in] maskMetadata       Metadata to be written to the mask header.
     *  @param[in] varianceMetadata   Metadata to be written to the variance header.
     *
     *  The FITS file will have four HDUs; the primary HDU will contain only metadata,
     *  while the image, mask, and variance HDU headers will use the "INHERIT='T'" convention
     *  to indicate that the primary metadata applies to those HDUs as well.
     */
    void writeFits(
        std::string const & fileName,
        CONST_PTR(daf::base::PropertySet) metadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) imageMetadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) maskMetadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) varianceMetadata = CONST_PTR(daf::base::PropertySet)()
    ) const;

    /**
     *  @brief Write a MaskedImage to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] metadata      Additional values to write to the primary HDU header (may be null).
     *  @param[in] imageMetadata      Metadata to be written to the image header.
     *  @param[in] maskMetadata       Metadata to be written to the mask header.
     *  @param[in] varianceMetadata   Metadata to be written to the variance header.
     *
     *  The FITS file will have four HDUs; the primary HDU will contain only metadata,
     *  while the image, mask, and variance HDU headers will use the "INHERIT='T'" convention
     *  to indicate that the primary metadata applies to those HDUs as well.
     */
    void writeFits(
        fits::MemFileManager & manager,
        CONST_PTR(daf::base::PropertySet) metadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) imageMetadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) maskMetadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) varianceMetadata = CONST_PTR(daf::base::PropertySet)()
    ) const;

    /**
     *  @brief Write a MaskedImage to a FITS RAM file.
     *
     *  @param[in] fitsfile           An empty FITS file object.
     *  @param[in] metadata           Additional values to write to the primary HDU header (may be null).
     *  @param[in] imageMetadata      Metadata to be written to the image header.
     *  @param[in] maskMetadata       Metadata to be written to the mask header.
     *  @param[in] varianceMetadata   Metadata to be written to the variance header.
     *
     *  The FITS file will have four HDUs; the primary HDU will contain only metadata,
     *  while the image, mask, and variance HDU headers will use the "INHERIT='T'" convention
     *  to indicate that the primary metadata applies to those HDUs as well.
     */
    void writeFits(
        fits::Fits & fitsfile,
        CONST_PTR(daf::base::PropertySet) metadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) imageMetadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) maskMetadata = CONST_PTR(daf::base::PropertySet)(),
        CONST_PTR(daf::base::PropertySet) varianceMetadata = CONST_PTR(daf::base::PropertySet)()
    ) const;

    /**
     *  @brief Read a MaskedImage from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     */
    static MaskedImage readFits(std::string const & filename) {
        return MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>(filename);
    }

    /**
     *  @brief Read a MaskedImage from a FITS RAM file.
     *
     *  @param[in] manager     Object that manages the memory to be read.
     */
    static MaskedImage readFits(fits::MemFileManager & manager) {
        return MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>(manager);
    }

    // Getters
    /// Return a (Ptr to) the MaskedImage's %image
    ImagePtr getImage(bool const noThrow=false) const {
        if (!_image && !noThrow) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                "MaskedImage's Image is NULL"
            );
        }
        return _image;
    }
    /// Return a (Ptr to) the MaskedImage's %mask
    MaskPtr getMask(bool const noThrow=false) const {
        if (!_mask && !noThrow) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                "MaskedImage's Mask is NULL"
            );
        }

        return _mask;
    }
    /// Return a (Ptr to) the MaskedImage's variance
    VariancePtr getVariance(bool const noThrow=false) const {
        if (!_variance && !noThrow) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                "MaskedImage's Variance is NULL"
            );
        }

        return _variance;
    }
    /// Return the number of columns in the %image
    int getWidth() const { return _image->getWidth(); }
    /// Return the number of rows in the %image
    int getHeight() const { return _image->getHeight(); }
    geom::Extent2I getDimensions() const {return _image->getDimensions();}
    geom::Box2I getBBox(ImageOrigin const origin=PARENT) const {return _image->getBBox(origin);}
    /**
     * Return the %image's column-origin
     *
     * This will usually be 0 except for images created using the
     * <tt>MaskedImage(fileName, hdu, BBox, mode)</tt> ctor or <tt>MaskedImage(ImageBase, BBox)</tt> cctor
     * The origin can be reset with setXY0()
     */
    int getX0() const { return _image->getX0(); }
    /**
     * Return the %image's row-origin
     *
     * This will usually be 0 except for images created using the
     * <tt>MaskedImage(fileName, hdu, BBox, mode)</tt> ctor or <tt>MaskedImage(ImageBase, BBox)</tt> cctor
     * The origin can be reset with setXY0()
     */
    int getY0() const { return _image->getY0(); }
    /**
     * Return the %image's origin
     *
     * This will usually be (0, 0) except for images created using the
     * <tt>MaskedImage(fileName, hdu, BBox, mode)</tt> ctor or <tt>MaskedImage(ImageBase, BBox)</tt> cctor
     * The origin can be reset with \c setXY0
     */
    geom::Point2I getXY0() const { return _image->getXY0(); }

    /**
     * Set the MaskedImage's origin
     *
     * The origin is usually set by the constructor, so you shouldn't need this function
     *
     * \note There are use cases (e.g. memory overlays) that may want to set these values, but
     * don't do so unless you are an Expert.
     */
    void setXY0(int const x0, int const y0) {
        setXY0(geom::Point2I(x0,y0));
    }

    /**
     * Set the MaskedImage's origin
     *
     * The origin is usually set by the constructor, so you shouldn't need this function
     *
     * \note There are use cases (e.g. memory overlays) that may want to set these values, but
     * don't do so unless you are an Expert.
     */
    void setXY0(geom::Point2I const origin) {
        if (_image) {
            _image->setXY0(origin);
        }

        if (_mask) {
            _mask->setXY0(origin);
        }

        if (_variance) {
            _variance->setXY0(origin);
        }
    }


    /**
     * @brief Convert image index to image position (see Image::indexToPosition)
     *
     * @return image position
     */
    inline double indexToPosition(
            double ind, ///< image index
            lsst::afw::image::xOrY const xy ///< Is this a column or row coordinate?
    ) const {
        return getImage()->indexToPosition(ind, xy);
    }

    /**
     * @brief Convert image position to index  (see Image::positionToIndex)
     *
     * @return std::pair(nearest integer index, fractional part)
     */
    std::pair<int, double> positionToIndex(
            double const pos, ///< image position
            lsst::afw::image::xOrY const xy ///< Is this a column or row coordinate?
    ) const {
        return getImage()->positionToIndex(pos, xy);
    }

    //
    // Iterators and Locators
    //
    iterator begin() const;
    iterator end() const;
    iterator at(int const x, int const y) const;
    reverse_iterator rbegin() const;
    reverse_iterator rend() const;

    fast_iterator begin(bool) const;
    fast_iterator end(bool) const;

    x_iterator row_begin(int y) const;
    x_iterator row_end(int y) const;

    /// Return an \c x_iterator at the point <tt>(x, y)</tt>
    x_iterator x_at(int x, int y) const {
#if 0
        typename Image::x_iterator imageEnd = getImage()->x_at(x, y);
        typename Mask::x_iterator maskEnd = getMask()->x_at(x, y);
        typename Variance::x_iterator varianceEnd = getVariance()->x_at(x, y);
#else  // bypass checks for non-NULL pointers
        typename Image::x_iterator imageEnd = _image->x_at(x, y);
        typename Mask::x_iterator maskEnd = _mask->x_at(x, y);
        typename Variance::x_iterator varianceEnd = _variance->x_at(x, y);
#endif

        return x_iterator(imageEnd, maskEnd, varianceEnd);
    }

    y_iterator col_begin(int x) const;
    y_iterator col_end(int x) const;

    /// Return an \c y_iterator at the point <tt>(x, y)</tt>
    y_iterator y_at(int x, int y) const {
#if 0
        typename Image::y_iterator imageEnd = getImage()->y_at(x, y);
        typename Mask::y_iterator maskEnd = getMask()->y_at(x, y);
        typename Variance::y_iterator varianceEnd = getVariance()->y_at(x, y);
#else  // bypass checks for non-NULL pointers
        typename Image::y_iterator imageEnd = _image->y_at(x, y);
        typename Mask::y_iterator maskEnd = _mask->y_at(x, y);
        typename Variance::y_iterator varianceEnd = _variance->y_at(x, y);
#endif
        return y_iterator(imageEnd, maskEnd, varianceEnd);
    }


    /// Return an \c xy_locator at the point <tt>(x, y)</tt>
    xy_locator xy_at(int x, int y) const {
#if 0
        typename Image::xy_locator imageEnd = getImage()->xy_at(x, y);
        typename Mask::xy_locator maskEnd = getMask()->xy_at(x, y);
        typename Variance::xy_locator varianceEnd = getVariance()->xy_at(x, y);
#else  // bypass checks for non-NULL pointers
        typename Image::xy_locator imageEnd = _image->xy_at(x, y);
        typename Mask::xy_locator maskEnd = _mask->xy_at(x, y);
        typename Variance::xy_locator varianceEnd = _variance->xy_at(x, y);
#endif

        return xy_locator(imageEnd, maskEnd, varianceEnd);
    }

private:

    LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskedImageFormatter<ImagePixelT, MaskPixelT, VariancePixelT>)
    void conformSizes();

    ImagePtr _image;
    MaskPtr _mask;
    VariancePtr _variance;
};

/**
 * A function to return a MaskedImage of the correct type (cf. std::make_pair)
 */
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>* makeMaskedImage(
    typename Image<ImagePixelT>::Ptr image, ///< %image
    typename Mask<MaskPixelT>::Ptr mask = typename Mask<MaskPixelT>::Ptr(),    ///< mask
    typename Image<VariancePixelT>::Ptr variance = typename Image<VariancePixelT>::Ptr() ///< variance
                                                                     ) {
    return new MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>(image, mask, variance);
}

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
namespace {
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
} // anonymous namespace

template<typename ImageT>
struct GetImage : public GetImage_<ImageT, typename ImageT::image_category> {
};

}}}  // lsst::afw::image

#endif //  LSST_IMAGE_MASKEDIMAGE_H
