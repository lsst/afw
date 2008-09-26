#if !defined(GIL_LSST_H)
#define GIL_LSST_H 1
/*
 * Extend the gil types to provide non-scaling float/int32 images, type bits32[fs]_noscale
 */
#include <boost/mpl/assert.hpp>
#include <boost/mpl/bool.hpp>

//#define BOOST_GIL_USE_CONCEPT_CHECK 1

#include <boost/gil/gil_all.hpp>

namespace boost { namespace gil {
/*
 * Define a type that's a pure float, without scaling into [0, 1]
 */
typedef float bits32f_noscale;

GIL_DEFINE_BASE_TYPEDEFS(32f_noscale, gray)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(32f_noscale, dev2n, devicen_t<2>, devicen_layout_t<2>)

/////////////////////////////////////////////////////
///  bits32f_noscale "conversions" that don't do anything
/////////////////////////////////////////////////////

template <typename DstChannelV> struct channel_converter_unsigned<bits32f_noscale,DstChannelV> : public std::unary_function<bits32f_noscale,DstChannelV> {
    DstChannelV   operator()(bits32f_noscale x) const { return DstChannelV(x + 0.5f); }
};

template <typename SrcChannelV> struct channel_converter_unsigned<SrcChannelV,bits32f_noscale> : public std::unary_function<SrcChannelV,bits32f_noscale> {
    bits32f_noscale operator()(SrcChannelV   x) const { return bits32f_noscale(x); }
};

template <> struct channel_converter_unsigned<bits32f_noscale,bits32f_noscale> : public std::unary_function<bits32f_noscale,bits32f_noscale> {
    bits32f_noscale operator()(bits32f_noscale   x) const { return x; }
};

/// \brief Specialization of channel_multiply for 32f_noscale channels
template<> struct channel_multiplier_unsigned<bits32f_noscale> : public std::binary_function<bits32f_noscale,bits32f_noscale,bits32f_noscale> {
    bits32f_noscale operator()(bits32f_noscale a, bits32f_noscale b) const { return a*b; }
};

/************************************************************************************************************/
/*
 * Define a type that's a pure double, without scaling into [0, 1]
 */
typedef double bits64f_noscale;

GIL_DEFINE_BASE_TYPEDEFS(64f_noscale, gray)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(64f_noscale, dev2n, devicen_t<2>, devicen_layout_t<2>)

/////////////////////////////////////////////////////
///  bits64f_noscale "conversions" that don't do anything
/////////////////////////////////////////////////////

template <typename DstChannelV> struct channel_converter_unsigned<bits64f_noscale,DstChannelV> : public std::unary_function<bits64f_noscale,DstChannelV> {
    DstChannelV   operator()(bits64f_noscale x) const { return DstChannelV(x + 0.5f); }
};

template <typename SrcChannelV> struct channel_converter_unsigned<SrcChannelV,bits64f_noscale> : public std::unary_function<SrcChannelV,bits64f_noscale> {
    bits64f_noscale operator()(SrcChannelV   x) const { return bits64f_noscale(x); }
};

template <> struct channel_converter_unsigned<bits64f_noscale,bits64f_noscale> : public std::unary_function<bits64f_noscale,bits64f_noscale> {
    bits64f_noscale operator()(bits64f_noscale   x) const { return x; }
};

/// \brief Specialization of channel_multiply for 64f_noscale channels
template<> struct channel_multiplier_unsigned<bits64f_noscale> : public std::binary_function<bits64f_noscale,bits64f_noscale,bits64f_noscale> {
    bits64f_noscale operator()(bits64f_noscale a, bits64f_noscale b) const { return a*b; }
};
        
/************************************************************************************************************/
/*
 * Define a type that's a pure char, without scaling into [0, 255]
 */
typedef int8_t bits8_noscale;

GIL_DEFINE_BASE_TYPEDEFS(8_noscale, gray)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(8_noscale, dev2n, devicen_t<2>, devicen_layout_t<2>)

/////////////////////////////////////////////////////
///  bits8_noscale "conversions" that don't do anything
/////////////////////////////////////////////////////

template <typename DstChannelV> struct channel_converter<bits8_noscale,DstChannelV> : public std::unary_function<bits8_noscale,DstChannelV> {
    DstChannelV   operator()(bits8_noscale x) const { return DstChannelV(x); }
};

template <typename SrcChannelV> struct channel_converter<SrcChannelV,bits8_noscale> : public std::unary_function<SrcChannelV,bits8_noscale> {
    bits8_noscale operator()(SrcChannelV   x) const { return bits8_noscale(x); }
};

template <> struct channel_converter<bits8_noscale,bits8_noscale> : public std::unary_function<bits8_noscale,bits8_noscale> {
    bits8_noscale operator()(bits8_noscale   x) const { return x; }
};

/// \brief Specialization of channel_multiply for 8_noscale channels
template<> struct channel_multiplier<bits8_noscale> : public std::binary_function<bits8_noscale,bits8_noscale,bits8_noscale> {
    bits8_noscale operator()(bits8_noscale a, bits8_noscale b) const { return a*b; }
};
        
/************************************************************************************************************/
/*
 * Define a type that's a pure int, without scaling into [-2.1478e9, 2.1478e9]
 */
typedef int32_t bits32s_noscale;

GIL_DEFINE_BASE_TYPEDEFS(32s_noscale, gray)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(32s_noscale, dev2n, devicen_t<2>, devicen_layout_t<2>)

/////////////////////////////////////////////////////
///  bits32s_noscale "conversions" that don't do anything
/////////////////////////////////////////////////////

template <typename DstChannelV> struct channel_converter<bits32s_noscale,DstChannelV> : public std::unary_function<bits32s_noscale,DstChannelV> {
    DstChannelV   operator()(bits32s_noscale x) const { return DstChannelV(x); }
};

template <typename SrcChannelV> struct channel_converter<SrcChannelV,bits32s_noscale> : public std::unary_function<SrcChannelV,bits32s_noscale> {
    bits32s_noscale operator()(SrcChannelV   x) const { return bits32s_noscale(x); }
};

template <> struct channel_converter<bits32s_noscale,bits32s_noscale> : public std::unary_function<bits32s_noscale,bits32s_noscale> {
    bits32s_noscale operator()(bits32s_noscale   x) const { return x; }
};

/// \brief Specialization of channel_multiply for 32s_noscale channels
template<> struct channel_multiplier<bits32s_noscale> : public std::binary_function<bits32s_noscale,bits32s_noscale,bits32s_noscale> {
    bits32s_noscale operator()(bits32s_noscale a, bits32s_noscale b) const { return a*b; }
};
} }  // namespace boost::gil


/************************************************************************************************************/

namespace lsst { namespace afw { namespace image { namespace details {
    //
    // Map typenames to gil's types
    //
    template<typename T, bool rescale=false> struct types_traits {
        BOOST_MPL_ASSERT_MSG(boost::mpl::bool_<false>::value,
                             I_DO_NOT_KNOW_HOW_TO_MAP_THIS_TYPE_TO_A_GIL_TYPE,
                             ()
                            );
    };
                
    template<> struct types_traits<unsigned char, false> {
        typedef boost::gil::gray8_noscale_image_t image_t;
        typedef boost::gil::gray8_noscale_view_t view_t;
        typedef boost::gil::gray8_noscalec_view_t const_view_t;
        typedef boost::gil::channel_traits<char>::reference reference;
        typedef boost::gil::channel_traits<char>::const_reference const_reference;
    };

    template<> struct types_traits<short, false> {
        typedef boost::gil::gray16s_image_t image_t;
        typedef boost::gil::gray16s_view_t view_t;
        typedef boost::gil::gray16sc_view_t const_view_t;
        typedef boost::gil::channel_traits<short>::reference reference;
        typedef boost::gil::channel_traits<short>::const_reference const_reference;
    };

    template<> struct types_traits<unsigned short, false> {
        typedef boost::gil::gray16_image_t image_t;
        typedef boost::gil::gray16_view_t view_t;
        typedef boost::gil::gray16c_view_t const_view_t;
        typedef boost::gil::channel_traits<unsigned short>::reference reference;
        typedef boost::gil::channel_traits<unsigned short>::const_reference const_reference;
    };

    template<> struct types_traits<int, false> {
        typedef boost::gil::gray32s_noscale_image_t image_t;
        typedef boost::gil::gray32s_noscale_view_t view_t;
        typedef boost::gil::gray32s_noscalec_view_t const_view_t;
        typedef boost::gil::channel_traits<int>::reference reference;
        typedef boost::gil::channel_traits<int>::const_reference const_reference;
    };

    template<> struct types_traits<int, true> {
        typedef boost::gil::gray32s_image_t image_t;
        typedef boost::gil::gray32s_view_t view_t;
        typedef boost::gil::gray32sc_view_t const_view_t;
        typedef boost::gil::channel_traits<int>::reference reference;
        typedef boost::gil::channel_traits<int>::const_reference const_reference;
    };

    template<> struct types_traits<float, false> {
        typedef boost::gil::gray32f_noscale_image_t image_t;
        typedef boost::gil::gray32f_noscale_view_t view_t;
        typedef boost::gil::gray32f_noscalec_view_t const_view_t;
        typedef boost::gil::channel_traits<float>::reference reference;
        typedef boost::gil::channel_traits<float>::const_reference const_reference;
    };

    template<> struct types_traits<float, true> {
        typedef boost::gil::gray32f_image_t image_t;
        typedef boost::gil::gray32f_view_t view_t;
        typedef boost::gil::gray32fc_view_t const_view_t;
        typedef boost::gil::channel_traits<float>::reference reference;
        typedef boost::gil::channel_traits<float>::const_reference const_reference;
    };

    template<> struct types_traits<double, false> {
        typedef boost::gil::gray64f_noscale_image_t image_t;
        typedef boost::gil::gray64f_noscale_view_t view_t;
        typedef boost::gil::gray64f_noscalec_view_t const_view_t;
        typedef boost::gil::channel_traits<double>::reference reference;
        typedef boost::gil::channel_traits<double>::const_reference const_reference;
    };
#if 0                                   // Not provided by boost::gil
    template<> struct types_traits<double, true> {
        typedef boost::gil::gray64f_image_t image_t;
        typedef boost::gil::gray64f_view_t view_t;
        typedef boost::gil::gray64fc_view_t const_view_t;
        typedef boost::gil::channel_traits<double>::reference reference;
        typedef boost::gil::channel_traits<double>::const_reference const_reference;
    };
#endif

    template<typename T>
    struct const_iterator_type {
        typedef typename boost::gil::const_iterator_type<T>::type type;
    };

    template<typename T>
    struct const_locator_type {         // should assert that T is a locator
        typedef typename T::const_t type;
    };

    typedef boost::gil::point2<int> difference_type; // type used to advance locators
}}}}    // namespace lsst::afw::image::details

namespace boost { namespace gil {

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief transform_pixels with three sources
template <typename View1, typename View2, typename View3, typename ViewDest, typename F> GIL_FORCEINLINE 
F transform_pixels(const View1& src1, const View2& src2,const View3& src3,const ViewDest& dst, F fun) {
    for (std::ptrdiff_t y=0; y<dst.height(); ++y) {
        typename View1::x_iterator srcIt1=src1.row_begin(y);
        typename View2::x_iterator srcIt2=src2.row_begin(y);
        typename View2::x_iterator srcIt3=src3.row_begin(y);
        typename View3::x_iterator dstIt=dst.row_begin(y);
        for (std::ptrdiff_t x=0; x<dst.width(); ++x)
            dstIt[x]=fun(srcIt1[x],srcIt2[x],srcIt3[x]);
    }
    return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief transform_pixels with four sources
template <typename View1, typename View2, typename View3, typename View4, typename ViewDest, typename F> GIL_FORCEINLINE 
F transform_pixels(const View1& src1, const View2& src2,const View3& src3,const View4& src4,const ViewDest& dst, F fun) {
    for (std::ptrdiff_t y=0; y<dst.height(); ++y) {
        typename View1::x_iterator srcIt1=src1.row_begin(y);
        typename View2::x_iterator srcIt2=src2.row_begin(y);
        typename View2::x_iterator srcIt3=src3.row_begin(y);
        typename View2::x_iterator srcIt4=src4.row_begin(y);
        typename View3::x_iterator dstIt=dst.row_begin(y);
        for (std::ptrdiff_t x=0; x<dst.width(); ++x)
            dstIt[x]=fun(srcIt1[x],srcIt2[x],srcIt3[x],srcIt4[x]);
    }
    return fun;
}
}}                                  // namespace boost::gil
#endif
