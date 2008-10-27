// -*- lsst-c++ -*-
%define imageLib_DOCSTRING
"
Basic routines to talk to lsst::afw::image classes
and some underlying VisionWorkbench classes.
"
%enddef

%feature("autodoc", "1");
%module(docstring=imageLib_DOCSTRING, package="lsst.afw.image") imageLib

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=317                 // specialization of non-template
#pragma SWIG nowarn=362                 // operator=  ignored
#pragma SWIG nowarn=389                 // operator[] ignored
#pragma SWIG nowarn=503                 // Can't wrap 'operator unspecified_bool_type'

%{
#   include <fstream>
#   include <exception>
#   include <map>
#   include <boost/cstdint.hpp>
#   include <boost/static_assert.hpp>
#   include <boost/shared_ptr.hpp>
#   include <boost/any.hpp>
#   include <boost/array.hpp>
#   include <lsst/utils/Utils.h>
#   include <lsst/daf/base.h>
#   include <lsst/daf/data.h>
#   include <lsst/daf/persistence.h>
#   include <lsst/pex/exceptions.h>
#   include <lsst/pex/logging/Trace.h>
#   include <lsst/pex/policy/Policy.h>
#   include <lsst/afw/image.h>
#   include "lsst/afw/image/DiskImageResourceFITS.h"
%}

%inline %{
namespace lsst { namespace afw { namespace image { } } }
namespace lsst { namespace daf { namespace data { } } }
namespace vw {}
namespace boost {
    namespace filesystem {}
    class bad_any_cast;                 // for lsst/pex/policy/Policy.h
}
    
//using namespace lsst;
using namespace lsst::afw::image;
using namespace lsst::daf::data;
using namespace vw; // for BBox but I'm not sure why it's needed
%}

%init %{
%}

%include "lsst/p_lsstSwig.i"
%include "lsstImageTypes.i"     // vw and Image/Mask types and typedefs

%pythoncode %{
import lsst.daf.data
import lsst.utils

def version(HeadURL = r"$HeadURL$"):
    """Return a version given a HeadURL string. If a different version is setup, return that too"""

    version_svn = lsst.utils.guessSvnVersion(HeadURL)

    try:
        import eups
    except ImportError:
        return version_svn
    else:
        try:
            version_eups = eups.setup("afw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

/******************************************************************************/

%ignore vw::ImageView::origin;
%ignore operator vw::ImageView::unspecified_bool_type;

%import <vw/Core/FundamentalTypes.h>
%import <vw/Core/CompoundTypes.h>

%include <vw/Image/ImageViewBase.h>
%include <vw/Image/ImageView.h>
%include <vw/Image/PixelTypeInfo.h>
%include <vw/Image/PixelTypes.h>
%include <vw/Image/ImageResource.h>
%include <vw/Image/Statistics.h>
%include <vw/Math/BBox.h>
#if 0
%   include <vw/Math/Vector.h>             // swig doesn't like "const static int value = 0;"
#else
    template <class ElemT, int SizeN = 0>
    class Vector {
        boost::array<ElemT,SizeN> core_;
    public:
        Vector() {
            for( unsigned i=0; i<SizeN; ++i ) (*this)[i] = ElemT();
        }
        
        Vector( ElemT e1, ElemT e2 ) {
            BOOST_STATIC_ASSERT( SizeN >= 2 );
            (*this)[0] = e1; (*this)[1] = e2;
            for( unsigned i=2; i<SizeN; ++i ) (*this)[i] = ElemT();
        }

        ElemT x() {
            BOOST_STATIC_ASSERT( SizeN >= 1 );
            return core_[0];
        }
        
        ElemT y() {
            BOOST_STATIC_ASSERT( SizeN >= 2 );
            return core_[1];
        }
    };
#endif
%import <vw/FileIO/DiskImageResource.h>

%include "lsst/daf/base/Citizen.h"
%import "lsst/daf/base/Persistable.h"
%import "lsst/daf/base/DataProperty.h"
%include "../python/lsst/daf/data/dataLib.i"
%import "lsst/daf/persistence/Persistence.h"
%import "lsst/pex/exceptions.h"
%import "lsst/pex/logging/Trace.h"
%import "lsst/pex/policy/Policy.h"

/******************************************************************************/
// Masks and MaskedImages
%template(pairIntInt)    std::pair<int,int>;
%template(pairIntString) std::pair<int,std::string>;
%template(mapIntString)  std::map<std::string, int>;

%ignore lsst::afw::image::Image::origin;        // no need to swig origin (and the _wrap.cc file is invalid)
%ignore lsst::afw::image::Mask::origin;         // no need to swig origin (and the _wrap.cc file is invalid)

%ignore lsst::afw::image::Filter::operator int;
%include "lsst/afw/image/Filter.h"
%include "lsst/afw/image/Image.h"
%include "lsst/afw/image/Mask.h"
%include "lsst/afw/image/MaskedImage.h"

%include "lsst/afw/image/DiskImageResourceFITS.h"

%extend lsst::afw::image::Image<boost::uint16_t> {
    %rename(getVal) get;
    /**
     * Set an image to the value val
     */
    void set(boost::uint16_t val) {
        Image<boost::uint16_t>::ImageIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, boost::uint16_t val) {
        Image<boost::uint16_t>::ImageIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    boost::uint16_t get(int x, int y) {
        return self->operator()(x, y);
    }
%pythoncode %{
    getPtr = getVal	# Keep old (confusing) name
%}
}

%extend lsst::afw::image::Image<int> {
    %rename(getVal) get;
    /**
     * Set an image to the value val
     */
    void set(int val) {
        Image<int>::ImageIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, int val) {
        Image<int>::ImageIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    int get(int x, int y) {
        return self->operator()(x, y);
    }
%pythoncode %{
    getPtr = getVal	# Keep old (confusing) name
%}
}

%extend lsst::afw::image::Image<float> {
    %rename(getVal) get;
    /**
     * Set an image to the value val
     */
    void set(float val) {
        Image<float>::ImageIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, float val) {
        Image<float>::ImageIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    /**
     * Return the value of pixel (x,y)
     */
    float get(int x, int y) {
        return self->operator()(x, y);
    }
%pythoncode %{
    getPtr = getVal	# Keep old (confusing) name
%}
}

%extend lsst::afw::image::Image<double> {
    %rename(getVal) get;
    /**
     * Set an image to the value val
     */
    void set(double val) {
        Image<double>::ImageIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, double val) {
        Image<double>::ImageIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    /**
     * Return the value of pixel (x,y)
     */
    double get(int x, int y) {
        return self->operator()(x, y);
    }
%pythoncode %{
    getPtr = getVal	# Keep old (confusing) name
%}
}


%extend lsst::afw::image::Mask<lsst::afw::image::maskPixelType> {
    %rename(getVal) get;
    /**
     * Set entire mask to val
     */
    void set(lsst::afw::image::maskPixelType val) {
        Mask<lsst::afw::image::maskPixelType>::MaskIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, lsst::afw::image::maskPixelType val) {
        Mask<lsst::afw::image::maskPixelType>::MaskIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    /**
     * return the value of pixel (x,y).  Would be called get, except that that's taken
     */
    lsst::afw::image::maskPixelType get(int x, int y) {
        return self->operator()(x, y);
    }
%pythoncode %{
    getPtr = getVal	# Keep old (confusing) name
%}
}

%include "lsst/daf/base/persistenceMacros.i"
%template(ImageBaseU)           vw::ImageViewBase<vw::ImageView<boost::uint16_t> >;
%template(ImageViewU)           vw::ImageView<boost::uint16_t>;
%template(ImageU)               lsst::afw::image::Image<boost::uint16_t>;
%lsst_persistable_shared_ptr(ImageUPtr, lsst::afw::image::Image<boost::uint16_t>)

%template(ImageBaseI)           vw::ImageViewBase<vw::ImageView<int> >;
%template(ImageViewI)           vw::ImageView<int>;
%template(ImageI)               lsst::afw::image::Image<int>;
%lsst_persistable_shared_ptr(ImageIPtr, lsst::afw::image::Image<int>)

%template(ImageBaseF)           vw::ImageViewBase<vw::ImageView<float> >;
%template(ImageViewF)           vw::ImageView<float>;
%template(CompoundChannelTypeF) vw::CompoundChannelType<float>;
%template(PixelChannelTypeF)    vw::PixelChannelType<float>;
%template(ImageF)               lsst::afw::image::Image<float>;
%lsst_persistable_shared_ptr(ImageFPtr, lsst::afw::image::Image<float>)

%template(ImageBaseD)           vw::ImageViewBase<vw::ImageView<double> >;
%template(ImageViewD)           vw::ImageView<double>;
%template(CompoundChannelTypeD) vw::CompoundChannelType<double>;
%template(PixelChannelTypeD)    vw::PixelChannelType<double>;
%template(ImageD)               lsst::afw::image::Image<double>;
%lsst_persistable_shared_ptr(ImageDPtr, lsst::afw::image::Image<double>)

%template(listMaskPixelPtr)     std::list<lsst::afw::image::maskPixelType *>;
%template(CompoundChannelMaskTypeD) vw::CompoundChannelType<lsst::afw::image::maskPixelType>;
%template(PixelChannelMaskTypeD)    vw::PixelChannelType<lsst::afw::image::maskPixelType>;
%template(MaskU)                lsst::afw::image::Mask<lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(MaskUPtr,      lsst::afw::image::Mask<lsst::afw::image::maskPixelType>);

%boost_shared_ptr(MaskIVwPtrT,   vw::ImageView<lsst::afw::image::maskPixelType>);
//
// MaskedImage
//
%template(MaskedImageU)         lsst::afw::image::MaskedImage<boost::uint16_t, lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageUPtr, lsst::afw::image::MaskedImage<boost::uint16_t, lsst::afw::image::maskPixelType>);
%template(MaskedImageI)         lsst::afw::image::MaskedImage<int, lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageIPtr, lsst::afw::image::MaskedImage<int, lsst::afw::image::maskPixelType>);
%template(MaskedImageF)         lsst::afw::image::MaskedImage<float, lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageFPtr, lsst::afw::image::MaskedImage<float, lsst::afw::image::maskPixelType>);
%template(MaskedImageD)         lsst::afw::image::MaskedImage<double, lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageDPtr, lsst::afw::image::MaskedImage<double, lsst::afw::image::maskPixelType>);

// vw Statistics on Images
  /// Compute the mean value stored in all of the channels of all of the planes of the image.
  // template <class ViewT>
  // typename CompoundChannelType<typename ViewT::pixel_type>::type
  // mean_channel_value( const ImageViewBase<ViewT> &view_ ) {

%template(mean_channel_valueD)  vw::mean_channel_value<vw::ImageView<double> >;
%template(mean_channel_valueF)  vw::mean_channel_value<vw::ImageView<float> >;
%template(mean_channel_valueU)  vw::mean_channel_value<vw::ImageView<boost::uint16_t> >;
%template(mean_channel_valueI)  vw::mean_channel_value<vw::ImageView<int> >;

%pythoncode %{
    def mean_channel_value(img):
        iv = img.getIVw()
        if type(img) == type(ImageD()) or type(img) == type(ImageDPtr(ImageD())) :
            return mean_channel_valueD(iv)
        elif type(img) == type(ImageF()) or type(img) == type(ImageFPtr(ImageF())) :
            return mean_channel_valueF(iv)
        elif type(img) == type(ImageU()) or type(img) == type(ImageUPtr(ImageU())) :
            return mean_channel_valueU(iv)
        elif type(img) == type(ImageI()) or type(img) == type(ImageIPtr(ImageI())) :
            return mean_channel_valueI(iv)
        else:
            return None
%}

%template(BBox2i)               BBox<int32, 2>;
%template(BBox2f)               BBox<float, 2>;

%boost_shared_ptr(BBox2iPtr, BBox<int32, 2>);
%boost_shared_ptr(BBox2fPtr, BBox<float, 2>);

%template(Vector2i)             Vector<int32, 2>;
%template(Vector2f)             Vector<float, 2>;

%template(listPixelCoord)  std::list<lsst::afw::image::PixelCoord>;

%apply double &OUTPUT { double & };
%rename(positionToIndexAndResidual) lsst::afw::image::positionToIndex(double &, double);
%clear double &OUTPUT;

%include "lsst/afw/image/ImageUtils.h"

/************************************************************************************************************/

%{
#include "lsst/afw/image/Exposure.h"
%}

%include "lsst/afw/image/Exposure.h"

%template(ExposureU)    lsst::afw::image::Exposure<boost::uint16_t, lsst::afw::image::maskPixelType>;
%template(ExposureI)    lsst::afw::image::Exposure<int, lsst::afw::image::maskPixelType>;
%template(ExposureF)    lsst::afw::image::Exposure<float, lsst::afw::image::maskPixelType>;
%template(ExposureD)    lsst::afw::image::Exposure<double, lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(ExposureUPtr, lsst::afw::image::Exposure<boost::uint16_t, lsst::afw::image::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureIPtr, lsst::afw::image::Exposure<int, lsst::afw::image::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureFPtr, lsst::afw::image::Exposure<float, lsst::afw::image::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureDPtr, lsst::afw::image::Exposure<double, lsst::afw::image::maskPixelType>)

/************************************************************************************************************/

%{
    #include "lsst/afw/image/Wcs.h"
%}

%template(Coord2D)                  Vector<double, 2>;
%rename(isValid) operator bool;
%include "lsst/afw/image/Wcs.h"

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
