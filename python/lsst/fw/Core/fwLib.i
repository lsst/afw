// -*- lsst-c++ -*-
%define fwLib_DOCSTRING
"
Basic routines to talk to FW's classes (including visionWorkbench) and ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwLib_DOCSTRING) fwLib

// Suppress swig complaints from vw
// 317: Specialization of non-template
// 389: operator[] ignored
// 362: operator=  ignored
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=317
#pragma SWIG nowarn=362
#pragma SWIG nowarn=389

// define basic vectors
// these are used by Kernel and Function (and possibly other code)
%include "std_vector.i"
%template(vectorF) std::vector<float>;
%template(vectorD) std::vector<double>;
%template(vectorVectorF) std::vector<std::vector<float> >;
%template(vectorVectorD) std::vector<std::vector<double> >;

%{
#   include <fstream>
#   include <exception>
#   include <map>
#   include <boost/cstdint.hpp>
#   include <boost/static_assert.hpp>
#   include <boost/shared_ptr.hpp>
#   include <boost/any.hpp>
#   include <boost/array.hpp>
#   include "lsst/mwi/data/Citizen.h"
#   include "lsst/mwi/utils/Demangle.h"
#   include "lsst/mwi/utils/Trace.h"
#   include "lsst/mwi/utils/Utils.h"
#   include "lsst/mwi/logging/Log.h"
#   include "lsst/fw/DiskImageResourceFITS.h"
#   include "lsst/fw/Mask.h"
#   include "lsst/fw/MaskedImage.h"
#   include "lsst/fw/ImageUtils.h"
%}

%inline %{
namespace lsst { namespace fw { } }
namespace vw {}
    
using namespace lsst;
using namespace lsst::fw;
using namespace vw;
%}

%init %{
%}

%include "lsst/mwi/p_lsstSwig.i"
%include "lsstImageTypes.i"     // vw and Image/Mask types and typedefs

%pythoncode %{
import lsst.mwi.data
import lsst.mwi.utils

def version(HeadURL = r"$HeadURL$"):
    """Return a version given a HeadURL string.  If a different version's setup, return that too"""

    version_svn = guessSvnVersion(HeadURL)

    try:
        import eups
    except ImportError:
        return version_svn
    else:
        try:
            version_eups = eups.setup("fw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

/******************************************************************************/

%ignore vw::ImageView<boost::uint16_t>::origin;
%ignore vw::ImageView<float>::origin;
%ignore vw::ImageView<double>::origin;
%ignore vw::ImageView<lsst::fw::maskPixelType>::origin;
%ignore operator vw::ImageView::unspecified_bool_type;

%import <vw/Core/FundamentalTypes.h>
%import <vw/Core/CompoundTypes.h>

%include <vw/Image/ImageViewBase.h>
%include <vw/Image/ImageView.h>
%include <vw/Image/PixelTypeInfo.h>
%include <vw/Image/PixelTypes.h>
%include <vw/Image/ImageResource.h>
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

%include "lsst/mwi/data/Citizen.h"
%import "lsst/mwi/utils/Utils.h"
%import "lsst/mwi/policy/Policy.h"
%include "lsst/mwi/persistence/Persistable.h"
%include "lsst/mwi/data/LsstData.h"
%import "lsst/mwi/DataProperty.i"
%import "lsst/mwi/exceptions.h"

/******************************************************************************/
// Masks and MaskedImages
%clear int &;                           // no longer needed as of mwi 2.0
%template(pairIntString) std::pair<int,std::string>;
%template(mapIntString)  std::map<int,std::string>;
%apply int &OUTPUT { int & };

%ignore lsst::fw::Image::origin;        // no need to swig origin (and the _wrap.cc file is invalid)
%ignore lsst::fw::Mask::origin;         // no need to swig origin (and the _wrap.cc file is invalid)

%import "lsst/mwi/utils/Utils.h"
%include "lsst/mwi/data/Citizen.h"
%include "lsst/mwi/data/LsstImpl_DC2.h"
%include "lsst/mwi/data/LsstBase.h"
%include "lsst/fw/Image.h"
%include "lsst/fw/Mask.h"
%include "lsst/fw/MaskedImage.h"

%include "lsst/fw/DiskImageResourceFITS.h"

%extend lsst::fw::Image<boost::uint16_t> {
    %rename(getPtr) get;
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
}

%extend lsst::fw::Image<float> {
    %rename(getPtr) get;
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
}

%extend lsst::fw::Image<double> {
    %rename(getPtr) get;
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
}


%extend lsst::fw::Mask<lsst::fw::maskPixelType> {
    %rename(getPtr) get;
    /**
     * Set entire mask to val
     */
    void set(lsst::fw::maskPixelType val) {
        Mask<lsst::fw::maskPixelType>::MaskIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, lsst::fw::maskPixelType val) {
        Mask<lsst::fw::maskPixelType>::MaskIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    /**
     * return the value of pixel (x,y).  Would be called get, except that that's taken
     */
    lsst::fw::maskPixelType get(int x, int y) {
        return self->operator()(x, y);
    }
}

%include "lsst/mwi/persistenceMacros.i"
%template(ImageBaseU)           vw::ImageViewBase<vw::ImageView<boost::uint16_t> >;
%template(ImageViewU)           vw::ImageView<boost::uint16_t>;
%template(ImageU)               lsst::fw::Image<boost::uint16_t>;
%lsst_persistable_shared_ptr(ImageUPtr, lsst::fw::Image<boost::uint16_t>)

%template(ImageBaseF)           vw::ImageViewBase<vw::ImageView<float> >;
%template(ImageViewF)           vw::ImageView<float>;
%template(CompoundChannelTypeF) vw::CompoundChannelType<float>;
%template(PixelChannelTypeF)    vw::PixelChannelType<float>;
%template(ImageF)               lsst::fw::Image<float>;
%lsst_persistable_shared_ptr(ImageFPtr, lsst::fw::Image<float>)

%template(ImageBaseD)           vw::ImageViewBase<vw::ImageView<double> >;
%template(ImageViewD)           vw::ImageView<double>;
%template(CompoundChannelTypeD) vw::CompoundChannelType<double>;
%template(PixelChannelTypeD)    vw::PixelChannelType<double>;
%template(ImageD)               lsst::fw::Image<double>;
%lsst_persistable_shared_ptr(ImageDPtr, lsst::fw::Image<double>)

%template(listMaskPixelPtr)     std::list<lsst::fw::maskPixelType *>;
%template(CompoundChannelMaskTypeD) vw::CompoundChannelType<lsst::fw::maskPixelType>;
%template(PixelChannelMaskTypeD)    vw::PixelChannelType<lsst::fw::maskPixelType>;
%template(MaskU)                lsst::fw::Mask<lsst::fw::maskPixelType>;
%lsst_persistable_shared_ptr(MaskUPtr,      lsst::fw::Mask<lsst::fw::maskPixelType>);

%boost_shared_ptr(MaskIVwPtrT,   vw::ImageView<lsst::fw::maskPixelType>);
//
// MaskedImage
//
%template(MaskedImageF)         lsst::fw::MaskedImage<float, lsst::fw::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageFPtr, lsst::fw::MaskedImage<float, lsst::fw::maskPixelType>);
%template(MaskedImageD)         lsst::fw::MaskedImage<double, lsst::fw::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageDPtr, lsst::fw::MaskedImage<double, lsst::fw::maskPixelType>);
%template(MaskedImageU)         lsst::fw::MaskedImage<boost::uint16_t, lsst::fw::maskPixelType>;
%lsst_persistable_shared_ptr(MaskedImageUPtr, lsst::fw::MaskedImage<boost::uint16_t, lsst::fw::maskPixelType>);

%template(BBox2i)               BBox<int32, 2>;
%template(Vector2i)             Vector<int32, 2>;


%template(listPixelCoord)  std::list<lsst::fw::PixelCoord>;

%include "lsst/fw/ImageUtils.h"

/************************************************************************************************************/

%include "exposure.i"
%include "function.i"
%include "minimize.i"
%include "kernel.i"

/************************************************************************************************************/

%{
    #include "lsst/fw/WCS.h"
%}

%template(Coord2D)                  Vector<double, 2>;
%rename(isValid) operator bool;
%include "lsst/fw/WCS.h"

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
