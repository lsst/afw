// -*- lsst-c++ -*-
/**
 * \file
 * \brief Implementation of the Class MaskedImage
 */

#ifndef LSST_IMAGE_MASKEDIMAGE_H
#define LSST_IMAGE_MASKEDIMAGE_H

#include <ostream>
#include <list>
#include <map>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/mpl/at.hpp"
#include "boost/iterator/zip_iterator.hpp"

#include "lsst/daf/data/LsstBase.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/formatters/MaskedImageFormatter.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"

namespace lsst {
namespace afw {
namespace image {
    namespace detail {
        /// A traits class for MaskedImage
        struct MaskedImage_tag : basic_tag { };
        /// A class used to identify classes that represent MaskedImage pixels
        struct MaskedImagePixel_tag { };
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
    namespace mpl = boost::mpl;

    /// default type for variance images
    typedef float VariancePixel;
    /// A class to manipulate images, masks, and variance as a single object
    template<typename ImagePixelT, typename MaskPixelT=lsst::afw::image::MaskPixel,
             typename VariancePixelT=VariancePixel>
    class MaskedImage : public lsst::daf::base::Persistable,
                        public lsst::daf::data::LsstBase {
    public:
        /// shared pointer to the Image
        typedef typename Image<ImagePixelT>::Ptr ImagePtr;
        /// shared pointer to the Mask
        typedef typename Mask<MaskPixelT>::Ptr MaskPtr;
        /// shared pointer to the variance Image
        typedef typename Image<VariancePixelT>::Ptr VariancePtr;
        /// shared pointer to a MaskedImage
        typedef boost::shared_ptr<MaskedImage> Ptr;
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
            typename Ref<typename Image::Pixel>::type image() {
                return _iter->template get<0>()[0];
            }
        
            /// Return (a reference to) the mask part of the Pixel pointed at by the iterator
            typename Ref<typename Mask::Pixel>::type mask() {
                return _iter->template get<1>()[0];
            }

            /// Return (a reference to) the variance part of the Pixel pointed at by the iterator
            typename Ref<typename Variance::Pixel>::type variance() {
                return _iter->template get<2>()[0];
            }
            
            /// Return the underlying iterator tuple
            /// \note not really for public consumption;  could be made protected
            const IMV_iterator_tuple get_iterator_tuple() const {
                return _iter.get_iterator_tuple();
            }
    
            /// Increment the iterator by \c delta
            void operator+=(std::ptrdiff_t delta ///< how far to move the iterator
                           ) {
                _iter += delta;
            }
            /// Decrement the iterator by \c delta
            void operator-=(std::ptrdiff_t delta ///< how far to move the iterator
                           ) {
                _iter -= delta;
            }
            /// Increment the iterator (prefix)
            void operator++() {         // prefix
                ++_iter;
            }
            /// Increment the iterator (postfix)
            void operator++(int) {      // postfix
                _iter++;
            }
            /// Return the distance between two iterators
            std::ptrdiff_t operator-(MaskedImageIteratorBase const& rhs) {
                return this->_iter->template get<0>() - rhs._iter->template get<0>();
            }
            /// Return true if the lhs equals the rhs
            bool operator==(MaskedImageIteratorBase const& rhs) {
                return _iter == rhs._iter;
            }
            /// Return true if the lhs doesn't equal the rhs
            bool operator!=(MaskedImageIteratorBase const& rhs) {
                return _iter != rhs._iter;
            }
            /// Return true if the lhs is less than the rhs
            bool operator<(MaskedImageIteratorBase const& rhs) {
                return _iter < rhs._iter;
            }
            /// Convert an iterator to a Pixel
            operator Pixel() const {
                return Pixel(_image(this->template get<0>()[0]),
                             _mask(this->template get<1>()[0]),
                             _variance(this->template get<2>()[0]));
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
                MaskedImageIteratorBase_t::operator+=(delta);
                
                return *this;
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

                void operator+=(const int di) {
                    // Equivalent to "_mil->_loc.template get<0>().x() += di;"
                    X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())() += di;
                    X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())() += di;
                    X_OR_Y<VarianceLocator>(_mil->_loc.template get<2>())() += di;
                }

                void operator++() {     // prefix
                    // Equivalent to "++_mil->_loc.template get<0>().x();"
                    ++X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())();
                    ++X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())();
                    ++X_OR_Y<VarianceLocator>(_mil->_loc.template get<2>())();
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

                typename Ref<typename Image::Pixel>::type image() {
                    // Equivalent to "return (*_mil->_loc.template get<0>().x())[0];"

                    return (*(X_OR_Y<ImageLocator>(_mil->_loc.template get<0>())()))[0];
                }
                typename Ref<typename Mask::Pixel>::type mask() {
                    return (*(X_OR_Y<MaskLocator>(_mil->_loc.template get<1>())()))[0];
                }
                typename Ref<typename Variance::Pixel>::type variance() {
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
            template<typename, typename, typename> friend class const_MaskedImageLocator;

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
                template<typename, typename, typename, template<typename> class> friend class MaskedImageLocatorBase;
                template<typename, typename, typename> friend class const_MaskedImageLocator;

                /// Create a cached_location_t that can be used to access pixels <tt>(x, y)</tt> away from \c loc
                cached_location_t(IMVLocator const& loc, int x, int y) :
                    _imv(loc.template get<0>().cache_location(x, y),
                         loc.template get<1>().cache_location(x, y),
                         loc.template get<2>().cache_location(x, y)) {
                    ;
                }
            protected:
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
            typedef typename mpl::vector<ImagePixelT, MaskPixelT, VariancePixelT> PixelTVec;

            template<typename N>
            typename Ref<typename mpl::at<PixelTVec, N>::type>::type apply_IMV(cached_location_t const& cached_loc) {
                return _loc.template get<N::value>()[cached_loc._imv.template get<N::value>()][0];
            }
            
            template<typename N>
            typename Ref<typename mpl::at<PixelTVec, N>::type>::type apply_IMV() {
                return _loc.template get<N::value>()[0][0];
            }
            
            template<typename N>
            typename Ref<typename mpl::at<PixelTVec, N>::type>::type apply_IMV(int x, int y) {
                return _loc.template get<N::value>()(x, y)[0];
            }
            //
            // Use those templated classes to implement image/mask/variance
            //
            /// Return a reference to the %image at the offset set when we created the \c cached_location_t
            typename Ref<typename Image::Pixel>::type image(cached_location_t const& cached_loc) {
                return apply_IMV<mpl::int_<0> >(cached_loc);
            }            
            /// Return a reference to the %image at the current position of the locator
            typename Ref<typename Image::Pixel>::type image() {
                return apply_IMV<mpl::int_<0> >();
            }           
            /// Return a reference to the %image offset by <tt>(x, y)</tt> from the current position of the locator
            typename Ref<typename Image::Pixel>::type image(int x, int y) {
                return apply_IMV<mpl::int_<0> >(x, y);
            }
            
            /// Return a reference to the mask at the offset set when we created the \c cached_location_t
            typename Ref<typename Mask::Pixel>::type mask(cached_location_t const& cached_loc) {
                return apply_IMV<mpl::int_<1> >(cached_loc);
            }
            /// Return a reference to the mask at the current position of the locator
            typename Ref<typename Mask::Pixel>::type mask() {
                return apply_IMV<mpl::int_<1> >();
            }
            /// Return a reference to the mask offset by <tt>(x, y)</tt> from the current position of the locator
            typename Ref<typename Mask::Pixel>::type mask(int x, int y) {
                return apply_IMV<mpl::int_<1> >(x, y);
            }
        
            /// Return a reference to the variance at the offset set when we created the \c cached_location_t
            typename Ref<typename Variance::Pixel>::type variance(cached_location_t const& cached_loc) {
                return apply_IMV<mpl::int_<2> >(cached_loc);
            }
            /// Return a reference to the variance at the current position of the locator
            typename Ref<typename Variance::Pixel>::type variance() {
                return apply_IMV<mpl::int_<2> >();
            }
            /// Return a reference to the variance offset by <tt>(x, y)</tt> from the current position of the locator
            typename Ref<typename Variance::Pixel>::type variance(int x, int y) {
                return apply_IMV<mpl::int_<2> >(x, y);
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
                MaskedImageLocatorBase_t(const_ImageLocator(iter._loc.template get<0>()),
                                         const_MaskLocator(iter._loc.template get<1>()),
                                         const_VarianceLocator(iter._loc.template get<2>())
                                        ) {
                ;
            }
        };

#if 0                                   // this is no longer needed with Pixel.h
    private:
        //
        // Implementations of PixelCast that work for MaskedImage pixels (if true_type) or
        // other, presumably scalar, pixels if false_type
        //
        // The choice is made on the basis of inheritance from detail::MaskedImagePixel_tag
        //
        template<typename SinglePixelT>
        static SinglePixel doPixelCast(SinglePixelT rhs, boost::false_type) {
            return SinglePixel(rhs);
        }

        static SinglePixel doPixelCast(SinglePixel rhs, boost::true_type) {
            return rhs;
        }

        template<typename SinglePixelT>
        static SinglePixel doPixelCast(SinglePixelT rhs, boost::true_type) {
            return SinglePixel(rhs.image(), rhs.mask(), rhs.variance());
        }
    public:

        template<typename SinglePixelT>
        static SinglePixel PixelCast(SinglePixelT rhs) {
            return doPixelCast(rhs, typename boost::is_base_of<detail::MaskedImagePixel_tag, SinglePixelT>::type());
        }
#endif
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
        explicit MaskedImage(int width=0, int height=0, MaskPlaneDict const& planeDict=MaskPlaneDict());
        explicit MaskedImage(ImagePtr image,
                             MaskPtr mask = MaskPtr(static_cast<Mask *>(0)),
                             VariancePtr variance = VariancePtr(static_cast<Variance *>(0)));
        explicit MaskedImage(const std::pair<int, int> dimensions, MaskPlaneDict const& planeDict=MaskPlaneDict());
        explicit MaskedImage(std::string const& baseName, int const hdu=0,
#if 1                                   // Old name for boost::shared_ptrs
                             typename lsst::daf::base::DataProperty::PtrType
		metadata=lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(NULL)),
#else
                             typename lsst::daf::base::DataProperty::Ptr
		metadata=lsst::daf::base::DataProperty::Ptr(static_cast<lsst::daf::base::DataProperty *>(NULL)),
#endif
                             bool const conformMasks=false);
        
        MaskedImage(MaskedImage const& rhs, bool const deep=false);
        MaskedImage(const MaskedImage& rhs, const BBox& bbox, const bool deep=false);
        /// generalised copy constructor; defined here in the header so that the compiler can instantiate
        /// N(N-1)/2 conversions between N ImageBase types.
        ///
        /// We only support converting the Image part
        template<typename OtherPixelT>
        MaskedImage(MaskedImage<OtherPixelT, MaskPixelT, VariancePixelT> const& rhs, //!< Input image
                    const bool deep) :                                         //!< Must be true; needed to disambiguate
            lsst::daf::data::LsstBase(typeid(this)) {
            if (!deep) {
                throw lsst::pex::exceptions::InvalidParameter("Only deep copies are permitted for MaskedImages "
                                                              "with different pixel types");
            }

            Image tmp(*rhs.getImage(), true);
            _image->swap(tmp);          // See Meyers, Effective C++, Items 11 and 43
        }

        /**
         * \fn MaskedImage& operator=(MaskedImage const& rhs)
         * \brief Make the lhs use the rhs's pixels
         * \param MaskedImage const& rhs
         *
         * If you are copying a scalar value, a simple <tt>lhs = scalar;</tt> is OK, but
         * this is probably not the function that you want to use with an %image. To copy pixel values
         * from the rhs use \link operator<<\endlink.
         */
        //MaskedImage& operator=(MaskedImage const& rhs);
        
        virtual ~MaskedImage() {}
        
        /// Return the %image's size;  useful for passing to constructors
        std::pair<int, int> getDimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

        void swap(MaskedImage &rhs);

        // Variance functions
        void setVarianceFromGain();     // was setDefaultVariance();
        
        // Operators
        void operator<<=(MaskedImage const& rhs);

        void operator+=(ImagePixelT const rhs);
        void operator+=(MaskedImage& rhs);
        void operator-=(ImagePixelT const rhs);
        void operator-=(MaskedImage& rhs);
        void operator*=(ImagePixelT const rhs);
        void operator*=(MaskedImage& rhs);
        void operator/=(ImagePixelT const rhs);
        void operator/=(MaskedImage& rhs);
        
        // IO functions
        static std::string imageFileName(std::string const& baseName) { return baseName + "_img.fits"; }
        static std::string maskFileName(std::string const& baseName) { return baseName + "_msk.fits"; }
        static std::string varianceFileName(std::string const& baseName) { return baseName + "_var.fits"; }

        void writeFits(std::string const& baseName,
#if 1                                   // Old name for boost::shared_ptrs
              typename lsst::daf::base::DataProperty::PtrType
              metadata=lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0))
#else
              typename lsst::daf::base::DataProperty::ConstPtr
              metadata=lsst::daf::base::DataProperty::ConstPtr(static_cast<lsst::daf::base::DataProperty *>(0))
#endif
                      ) const;
        
        // Getters
        /// Return a (Ptr to) the MaskedImage's %image
        ImagePtr getImage() const { return _image; }
        /// Return a (Ptr to) the MaskedImage's %mask
        MaskPtr getMask() const { return _mask; }
        /// Return a (Ptr to) the MaskedImage's variance
        VariancePtr getVariance() const { return _variance; }
        /// Return the number of columns in the %image
        int getWidth() const { return _image->getWidth(); }
        /// Return the number of rows in the %image
        int getHeight() const { return _image->getHeight(); }
        /**
         * Return the %image's row-origin
         *
         * This will usually be 0 except for images created using the <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with setXY0()
         */
        unsigned int getX0() const { return _image->getX0(); }
        /**
         * Return the %image's column-origin
         *
         * This will usually be 0 except for images created using the <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with setXY0()
         */
        unsigned int getY0() const { return _image->getY0(); }
        //
        // Iterators and Locators
        //
        iterator begin() const;
        iterator end() const;
        iterator at(int const x, int const y) const;
        reverse_iterator rbegin() const;
        reverse_iterator rend() const;

        x_iterator begin(bool) const;
        x_iterator end(bool) const;

        x_iterator row_begin(int y) const;
        x_iterator row_end(int y) const;
        x_iterator x_at(int x, int y) const;

        y_iterator col_begin(int x) const;
        y_iterator col_end(int x) const;
        y_iterator y_at(int x, int y) const;

        xy_locator xy_at(int x, int y) const;
        /**
         * Set the MaskedImage's origin
         *
         * The origin is usually set by the constructor, so you shouldn't need this function
         *
         * \note There are use cases (e.g. memory overlays) that may want to set these values, but
         * don't do so unless you are an Expert.
         */
        void setXY0(PointI const origin) {
            _image->setXY0(origin);
            _mask->setXY0(origin);
            _variance->setXY0(origin);
        }
    private:

        LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskedImageFormatter<ImagePixelT, MaskPixelT, VariancePixelT>);
        void conformSizes();
        
        ImagePtr _image;
        MaskPtr _mask;
        VariancePtr _variance;
    };
}}}  // lsst::afw::image
        
#endif //  LSST_IMAGE_MASKEDIMAGE_H
