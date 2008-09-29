// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  MaskedImage.h
//  Implementation of the Class MaskedImage
///////////////////////////////////////////////////////////

#ifndef LSST_IMAGE_MASKEDIMAGE_H
#define LSST_IMAGE_MASKEDIMAGE_H

#include <list>
#include <map>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/mpl/at.hpp"
#include "boost/fusion/tuple.hpp"
#include "boost/iterator/zip_iterator.hpp"

#include "lsst/daf/data/LsstBase.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/formatters/MaskedImageFormatter.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"

namespace lsst {
namespace afw {
    namespace formatters {
        template<typename ImagePixelT, typename MaskPixelT> class MaskedImageFormatter;
    }

namespace image {
    namespace mpl = boost::mpl;
    
    template<typename ImagePixelT, typename MaskPixelT=lsst::afw::image::MaskPixel, typename VariancePixelT=ImagePixelT>
    class MaskedImage : public lsst::daf::base::Persistable,
                        public lsst::daf::data::LsstBase {
    public:
        typedef Image<VariancePixelT> Variance;
        typedef typename Image<ImagePixelT>::Ptr ImagePtr;
        typedef typename Mask<MaskPixelT>::Ptr MaskPtr;
        typedef typename Variance::Ptr VariancePtr;
        typedef boost::shared_ptr<MaskedImage> Ptr;
        typedef typename Mask<MaskPixelT>::MaskPlaneDict MaskPlaneDict;

        typedef Image<ImagePixelT> Image; // need to be here, as "typedef Image::Ptr ImagePtr;" confuses swig
        typedef Mask<MaskPixelT> Mask;    // and we can't use Image<> after these typedefs
        
        /************************************************************************************************************/

        template<typename, typename, typename> class maskedImageIterator;
        template<typename, typename, typename> class const_maskedImageIterator;
        template<typename, typename, typename> class maskedImageLocator;
        template<typename, typename, typename> class const_maskedImageLocator;
        
#if !defined(SWIG)
        template<typename ImageIterator, typename MaskIterator, typename VarianceIterator,
                 template<typename> class Ref=Reference>
        class maskedImageIteratorBase {
            typedef boost::tuple<ImageIterator, MaskIterator, VarianceIterator> IMV_iterator_tuple;
            typedef typename boost::zip_iterator<IMV_iterator_tuple>::reference IMV_tuple;

        public:
            template<typename, typename, typename> friend class const_maskedImageIterator;
            
            maskedImageIteratorBase(ImageIterator const& img, MaskIterator const& msk, VarianceIterator const &var) :
                _iter(boost::make_zip_iterator(boost::make_tuple(img, msk, var))) {
            }
    
            typename Ref<typename Image::Pixel>::type image() {
                return _iter->template get<0>()[0];
            }
        
            typename Ref<typename Mask::Pixel>::type mask() {
                return _iter->template get<1>()[0];
            }

            typename Ref<typename Variance::Pixel>::type variance() {
                return _iter->template get<2>()[0];
            }

            const IMV_iterator_tuple get_iterator_tuple() const {
                return _iter.get_iterator_tuple();
            }
    
            void operator+=(std::ptrdiff_t delta) {
                _iter += delta;
            }
            void operator++() {         // prefix
                ++_iter;
            }
            void operator++(int) {      // postfix
                _iter++;
            }
            bool operator!=(maskedImageIteratorBase const& rhs) {
                return _iter != rhs._iter;
            }
            bool operator<(maskedImageIteratorBase const& rhs) {
                return _iter < rhs._iter;
            }
            IMV_tuple operator*() {
                return *_iter;
            }
            const IMV_tuple operator*() const {
                return *_iter;
            }

        protected:
            typename boost::zip_iterator<IMV_iterator_tuple> _iter;
        };
        
        template<typename ImageIterator, typename MaskIterator, typename VarianceIterator>
        class maskedImageIterator :
            public  maskedImageIteratorBase<ImageIterator, MaskIterator, VarianceIterator> {
            typedef maskedImageIteratorBase<ImageIterator, MaskIterator, VarianceIterator> maskedImageIteratorBase;
        public:
            maskedImageIterator(ImageIterator& img, MaskIterator& msk, VarianceIterator &var) :
                maskedImageIteratorBase(img, msk, var) {
            }
        };

        template<typename ImageIterator, typename MaskIterator, typename VarianceIterator>
        class const_maskedImageIterator :
            public  maskedImageIteratorBase<typename details::const_iterator_type<ImageIterator>::type,
                                            typename details::const_iterator_type<MaskIterator>::type,
                                            typename details::const_iterator_type<VarianceIterator>::type,
                                            ConstReference> {

            typedef typename details::const_iterator_type<ImageIterator>::type const_ImageIterator;
            typedef typename details::const_iterator_type<MaskIterator>::type const_MaskIterator;
            typedef typename details::const_iterator_type<VarianceIterator>::type const_VarianceIterator;

            typedef maskedImageIteratorBase<const_ImageIterator, const_MaskIterator, const_VarianceIterator,
                                            ConstReference> maskedImageIteratorBase;
        public:
            const_maskedImageIterator(maskedImageIterator<ImageIterator, MaskIterator, VarianceIterator> const& iter) :
                maskedImageIteratorBase(const_ImageIterator(iter.get_iterator_tuple().template get<0>()),
                                        const_MaskIterator(iter.get_iterator_tuple().template get<1>()),
                                        const_VarianceIterator(iter.get_iterator_tuple().template get<2>())
                                       ) {
                ;
            }
        };

        template<typename ImageLocator, typename MaskLocator, typename VarianceLocator,
                 template<typename> class Ref=Reference>
        class maskedImageLocatorBase {
            typedef typename boost::tuple<ImageLocator, MaskLocator, VarianceLocator> IMVLocator;

            class _x_iterator {
            public:
                _x_iterator(maskedImageLocatorBase* mil) : _mil(mil) {}

                void operator+=(const int di) {
                    _mil->_loc.template get<0>().x() += di;
                    _mil->_loc.template get<1>().x() += di;
                    _mil->_loc.template get<2>().x() += di;
                }

                void operator++() {     // prefix
                    ++_mil->_loc.template get<0>().x();
                    ++_mil->_loc.template get<1>().x();
                    ++_mil->_loc.template get<2>().x();
                }
            protected:
                maskedImageLocatorBase *_mil;
            };

            class _y_iterator {
            public:
                _y_iterator(maskedImageLocatorBase* mil) : _mil(mil) {}

                void operator+=(const int di) {
                    _mil->_loc.template get<0>().y() += di;
                    _mil->_loc.template get<1>().y() += di;
                    _mil->_loc.template get<2>().y() += di;
                }

                void operator++() {     // prefix
                    ++_mil->_loc.template get<0>().y();
                    ++_mil->_loc.template get<1>().y();
                    ++_mil->_loc.template get<2>().y();
                }
            protected:
                maskedImageLocatorBase *_mil;
            };
        public:
            template<typename, typename, typename> friend class const_maskedImageLocator;

            typedef typename boost::tuple<typename ImageLocator::cached_location_t,
                                          typename MaskLocator::cached_location_t,
                                          typename VarianceLocator::cached_location_t> IMVCachedLocation;

            class cached_location_t {
            public:
                template<typename, typename, typename, template<typename> class> friend class maskedImageLocatorBase;
                template<typename, typename, typename> friend class const_maskedImageLocator;

                cached_location_t(IMVLocator const& loc, int x, int y) :
                    _imv(loc.template get<0>().cache_location(x, y),
                         loc.template get<1>().cache_location(x, y),
                         loc.template get<2>().cache_location(x, y)) {
                    ;
                }
            protected:
                IMVCachedLocation _imv;
            };
            
            maskedImageLocatorBase(ImageLocator const& img, MaskLocator const& msk, VarianceLocator const& var) :
                _loc(img, msk, var) {
                ;
            }

	    _x_iterator x() {
		return _x_iterator(this);
            }
        
            _y_iterator y() {
		return _y_iterator(this);
	    }

            cached_location_t cache_location(int x, int y) const {
                return cached_location_t(_loc, x, y);
            }
            //
            // We don't want to duplicate code for image/mask/variance -- but the boost::mpl stuff isn't pretty
            // as we can't say int_<N> within a template<int N>
            //
            typedef typename mpl::vector<ImagePixelT, MaskPixelT, VariancePixelT> PixelTVec;

            template<typename N>
            typename Ref<typename mpl::at<PixelTVec, N>::type>::type IMV(cached_location_t const& cached_loc) {
                return _loc.template get<N::value>()[cached_loc._imv.template get<N::value>()][0];
            }
            
            template<typename N>
            typename Ref<typename mpl::at<PixelTVec, N>::type>::type IMV() {
                return _loc.template get<N::value>()[0][0];
            }
            
            template<typename N>
            typename Ref<typename mpl::at<PixelTVec, N>::type>::type IMV(int x, int y) {
                return _loc.template get<N::value>()(x, y)[0];
            }
            //
            // Use those templated classes to implement image/mask/variance
            //
            typename Ref<typename Image::Pixel>::type image(cached_location_t const& cached_loc){
                return IMV<mpl::int_<0> >(cached_loc);
            }            
            typename Ref<typename Image::Pixel>::type image() {
                return IMV<mpl::int_<0> >();
            }           
            typename Ref<typename Image::Pixel>::type image(int x, int y) {
                return IMV<mpl::int_<0> >(x, y);
            }
            
            typename Ref<typename Mask::Pixel>::type mask(cached_location_t const& cached_loc) {
                return IMV<mpl::int_<1> >(cached_loc);
            }
            typename Ref<typename Mask::Pixel>::type mask() {
                return IMV<mpl::int_<1> >();
            }
            typename Ref<typename Mask::Pixel>::type mask(int x, int y) {
                return IMV<mpl::int_<1> >(x, y);
            }
        
            typename Ref<typename Variance::Pixel>::type variance(cached_location_t const& cached_loc) {
                return IMV<mpl::int_<2> >(cached_loc);
            }
            typename Ref<typename Variance::Pixel>::type variance() {
                return IMV<mpl::int_<2> >();
            }
            typename Ref<typename Variance::Pixel>::type variance(int x, int y) {
                return IMV<mpl::int_<2> >(x, y);
            }

            maskedImageLocatorBase& operator+=(std::pair<int, int> p) {
                return operator+=(details::difference_type(p.first, p.second));
            }

            maskedImageLocatorBase& operator+=(details::difference_type p) {
                _loc.template get<0>() += p;
                _loc.template get<1>() += p;
                _loc.template get<2>() += p;

                return *this;
            }
        protected:
            IMVLocator _loc;
        };

        template<typename ImageLocator, typename MaskLocator, typename VarianceLocator>
        class maskedImageLocator :
            public  maskedImageLocatorBase<ImageLocator, MaskLocator, VarianceLocator> {
            typedef maskedImageLocatorBase<ImageLocator, MaskLocator, VarianceLocator> maskedImageLocatorBase;
        public:
            maskedImageLocator(ImageLocator& img, MaskLocator& msk, VarianceLocator &var) :
                maskedImageLocatorBase(img, msk, var) {
            }
        };

        template<typename ImageLocator, typename MaskLocator, typename VarianceLocator>
        class const_maskedImageLocator :
            public  maskedImageLocatorBase<typename details::const_locator_type<ImageLocator>::type,
                                           typename details::const_locator_type<MaskLocator>::type,
                                           typename details::const_locator_type<VarianceLocator>::type,
                                           ConstReference> {

            typedef typename details::const_locator_type<ImageLocator>::type const_ImageLocator;
            typedef typename details::const_locator_type<MaskLocator>::type const_MaskLocator;
            typedef typename details::const_locator_type<VarianceLocator>::type const_VarianceLocator;

            typedef maskedImageLocatorBase<const_ImageLocator, const_MaskLocator, const_VarianceLocator,
                                           ConstReference>
											       maskedImageLocatorBase;
        public:
            const_maskedImageLocator(maskedImageLocator<ImageLocator, MaskLocator, VarianceLocator> const& iter) :
                maskedImageLocatorBase(const_ImageLocator(iter._loc.template get<0>()),
                                       const_MaskLocator(iter._loc.template get<1>()),
                                       const_VarianceLocator(iter._loc.template get<2>())
                                      ) {
                ;
            }
        };
#endif  // !defined(SWIG)

    /************************************************************************************************************/
    
        typedef maskedImageIterator<typename Image::iterator,
                                    typename Mask::iterator, typename Variance::iterator> iterator;
        typedef const_maskedImageIterator<typename Image::iterator,
                                    typename Mask::iterator, typename Variance::iterator> const_iterator;
        typedef maskedImageIterator<typename Image::reverse_iterator,
                                    typename Mask::reverse_iterator, typename Variance::reverse_iterator> reverse_iterator;
#if 0                                   // doesn't compile.  I should fix this, but it's low priority. RHL
        typedef const_maskedImageIterator<typename Image::reverse_iterator,
                                    typename Mask::reverse_iterator, typename Variance::reverse_iterator> const_reverse_iterator;
#endif
        typedef maskedImageIterator<typename Image::x_iterator,
                                    typename Mask::x_iterator, typename Variance::x_iterator> x_iterator;
        typedef const_maskedImageIterator<typename Image::x_iterator,
                                    typename Mask::x_iterator, typename Variance::x_iterator> const_x_iterator;
        typedef maskedImageIterator<typename Image::y_iterator,
                                    typename Mask::y_iterator, typename Variance::y_iterator> y_iterator;
        typedef const_maskedImageIterator<typename Image::y_iterator,
                                    typename Mask::y_iterator, typename Variance::y_iterator> const_y_iterator;

        typedef maskedImageLocator<typename Image::xy_locator,
                                    typename Mask::xy_locator, typename Variance::xy_locator> xy_locator;
        typedef const_maskedImageLocator<typename Image::xy_locator,
                                    typename Mask::xy_locator, typename Variance::xy_locator> const_xy_locator;
        
        /************************************************************************************************************/

        // Constructors
        explicit MaskedImage(int width=0, int height=0, MaskPlaneDict planeDefs=MaskPlaneDict());
        explicit MaskedImage(ImagePtr image,
                             MaskPtr mask = MaskPtr(static_cast<Mask *>(0)),
                             VariancePtr variance = VariancePtr(static_cast<Variance *>(0)));
        explicit MaskedImage(const std::pair<int, int> dimensions, MaskPlaneDict planeDefs=MaskPlaneDict());
        explicit MaskedImage(std::string const& baseName, int const hdu=0,
#if 1                                   // Old name for boost::shared_ptrs
                             typename lsst::daf::base::DataProperty::PtrType
		metadata=lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0)),
#else
                             typename lsst::daf::base::DataProperty::Ptr
		metadata=lsst::daf::base::DataProperty::Ptr(static_cast<lsst::daf::base::DataProperty *>(0)),
#endif
                             bool conformMasks=false
                            );                             

        MaskedImage(MaskedImage const& rhs, bool const deep=false);
        MaskedImage(const MaskedImage& src, const Bbox& bbox, const bool deep=false);

        // Use compiler generated functions
        //MaskedImage& operator=(MaskedImage const& rhs);
        
        virtual ~MaskedImage() {}
        
        std::pair<int, int> dimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

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
        ImagePtr getImage() const { return _image; }
        MaskPtr getMask() const { return _mask; }
        ImagePtr getVariance() const { return _variance; }
        int getWidth() const { return _image->getWidth(); }
        int getHeight() const { return _image->getHeight(); }
        unsigned int getX0() const { return _image->getX0(); }
        unsigned int getY0() const { return _image->getY0(); }
        //
        // Iterators and Locators
        //
        iterator at(int const x, int const y) const;
        iterator begin() const;
        iterator end() const;
        reverse_iterator rbegin() const;
        reverse_iterator rend() const;

        x_iterator x_at(int x, int y) const;
        x_iterator row_begin(int y) const;
        x_iterator row_end(int y) const;

        y_iterator y_at(int x, int y) const;
        y_iterator col_begin(int x) const;
        y_iterator col_end(int x) const;

        xy_locator xy_at(int x, int y) const;
        //
        // There are use cases (e.g. memory overlays) that may want to set these values, but
        // don't do so unless you are an Expert.
        //
        void setXY0(PointI const origin) {
            _image->setXY0(origin);
            _mask->setXY0(origin);
            _variance->setXY0(origin);
        }
    private:

        //LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskedImageFormatter<ImagePixelT, MaskPixelT>);
        void conformSizes();
        
        ImagePtr _image;
        MaskPtr _mask;
        ImagePtr _variance;
    };

}}}  // lsst::afw::image
        
#endif //  LSST_IMAGE_MASKEDIMAGE_H
