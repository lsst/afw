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

%{
#   include <fstream>
#   include <exception>
#   include <map>
#   include <boost/cstdint.hpp>
#   include <boost/static_assert.hpp>
#   include <boost/shared_ptr.hpp>
#   include <boost/any.hpp>
#   include <boost/array.hpp>
#   include "lsst/fw/Citizen.h"
#   include "lsst/fw/Demangle.h"
#   include "lsst/fw/DiskImageResourceFITS.h"
#   include "lsst/fw/Log.h"
#   include "lsst/fw/Mask.h"
#   include "lsst/fw/MaskedImage.h"
#   include "lsst/fw/Trace.h"
#   include "lsst/fw/Utils.h"
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

%include "../Core/p_lsstSwig.i"
%include "lsst/fw/Utils.h"

%pythoncode %{

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

%ignore vw::ImageView<int>::origin;
%ignore vw::ImageView<ImagePixelType>::origin;
%ignore vw::ImageView<MaskPixelType>::origin;
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
%include "lsst/fw/DiskImageResourceFITS.h"

/******************************************************************************/
// Citizens, Trace, etc.
%include "lsst/fw/Citizen.h"
%include "lsst/fw/Log.h"
%include "lsst/fw/Trace.h"
%include "lsst/fw/DataProperty.h"

%extend lsst::fw::Log {
    static void setDestination(const std::string& fileName) {
        std::ofstream &fp = *new std::ofstream(fileName.c_str(), std::ios_base::out); // Log now owns this stream
        lsst::fw::Log::setDestination(fp);
    }
}

#if 0                                   // doesn't work (yet)
typedef boost::shared_ptr<DataProperty> DataPropertyPtr;

%contract DataPropertyPtr::DataPropertyPtr {
ensure:
    DataPropertyPtr_ptr.get() > 0;
}
#endif
    
%template(DataPropertyPtrT) boost::shared_ptr<DataProperty>;
%template(DataPropertyContainerT) std::list<DataPropertyPtrT>;

%extend lsst::fw::DataProperty {
    %exception {
        try {
            $action;
        } catch(boost::bad_any_cast &e) {
            SWIG_exception(SWIG_RuntimeError, e.what());
        }
    }
    
    DataProperty(std::string name, int val) {
        return new DataProperty(name, val);
    }
    DataProperty(std::string name, std::string val) {
        return new DataProperty(name, val);
    }

    int getValueInt() {
        return boost::any_cast<const int>(self->getValue());
    }
    std::string getValueString() {
        return boost::any_cast<const std::string>(self->getValue());
    }

    DataPropertyPtrT match(const std::string &pattern, bool reset=true) {
        boost::regex re(pattern);
        return self->find(re, reset);
    }
}

/******************************************************************************/
// Masks and MaskedImages
%newobject getMaskPlaneMetaData;
%clear int &;
%template(pairIntString) std::pair<int,std::string>;
%template(mapIntString)  std::map<int,std::string>;
%apply int &OUTPUT { int & };

%ignore lsst::fw::Image::origin;        // no need to swig origin (and the _wrap.cc file is invalid)
%ignore lsst::fw::Mask::origin;         // no need to swig origin (and the _wrap.cc file is invalid)

%include "lsst/fw/Image.h"
%include "lsst/fw/Mask.h"
%include "lsst/fw/MaskedImage.h"

%extend lsst::fw::Image<int> {
    %rename(getPtr) get;
    /**
     * Set an image to the value val
     */
    void set(double val) {
        Image<int>::ImageIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    ImagePixelType get(int x, int y) {
        return self->operator()(x, y);
    }
}

%extend lsst::fw::Image<ImagePixelType> {
    %rename(getPtr) get;
    /**
     * Set an image to the value val
     */
    void set(double val) {
        Image<ImagePixelType>::ImageIVwT& ivw = self->getIVw();
#if 1                                   // Using whole-image iterator
        std::fill(ivw.begin(), ivw.end(), val);
#elif 0                                 // Using whole-image iterator
        typedef PixelIterator<Image<ImagePixelType>::ImageIVwT> PixelIterator;
        
        const PixelIterator end = ivw.end();
        for (PixelIterator ptr = ivw.begin(); ptr < end; ptr++) {
            *ptr = val;
        }
#else  // Using per-row iterator
        typedef Image<ImagePixelType>::ImageIVwT::pixel_accessor pixAccessT;
        pixAccessT srow = ivw.origin();

        for (unsigned int y = 0; y < ivw.rows(); y++) {
            pixAccessT scol = srow;
            for (unsigned int x = 0; x < ivw.cols(); x++) {
                *scol = val;
                scol.next_col();
            }
            srow.next_row();
        }
#endif
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, double val) {
        Image<ImagePixelType>::ImageIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    /**
     * Return the value of pixel (x,y)
     */
    ImagePixelType get(int x, int y) {
        return self->operator()(x, y);
    }
}

//%ignore operator lsst::Mask::operator()(int, int); // RHL can't get this to work
%extend lsst::fw::Mask<MaskPixelType> {
    %rename(getPtr) get;
    /**
     * Set entire mask to val
     */
    void set(double val) {
        Mask<MaskPixelType>::MaskIVwT& ivw = self->getIVw();
        std::fill(ivw.begin(), ivw.end(), val);
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, double val) {
        Mask<MaskPixelType>::MaskIVwT& ivw = self->getIVw();
        *ivw.origin().advance(x, y) = val;
    }
    /**
     * return the value of pixel (x,y).  Would be called get, except that that's taken
     */
    MaskPixelType get(int x, int y) {
        return self->operator()(x, y);
    }
}

%template(CompoundChannelTypeD)		vw::CompoundChannelType<ImagePixelType>;
%template(PixelChannelTypeD)		vw::PixelChannelType<ImagePixelType>;
%template(ImageBaseD)			vw::ImageViewBase<vw::ImageView<ImagePixelType> >;
%template(ImageViewD)			vw::ImageView<ImagePixelType>;
%template(ImageD) 	                lsst::fw::Image<ImagePixelType>;
%template(ImagePtrD) 	                boost::shared_ptr<lsst::fw::Image<ImagePixelType> >;

%template(ImageBaseInt)			vw::ImageViewBase<vw::ImageView<int> >;
%template(ImageViewInt)                 vw::ImageView<int>;
%template(ImageInt)			lsst::fw::Image<int>;
%template(ImagePtrInt) 	                boost::shared_ptr<lsst::fw::Image<int> >;

%template(listMaskPixelPtr)		std::list<MaskPixelType *>;
%template(ImageBaseMask)                vw::ImageViewBase<vw::ImageView<MaskPixelType> >;
%template(ImageViewMask)                vw::ImageView<MaskPixelType>;
%template(CompoundChannelMaskTypeD)     vw::CompoundChannelType<MaskPixelType>;
%template(PixelChannelMaskTypeD)        vw::PixelChannelType<MaskPixelType>;
%template(MaskD)                        lsst::fw::Mask<MaskPixelType>;
%template(MaskDPtr)                     boost::shared_ptr<lsst::fw::Mask<MaskPixelType> >;

%template(MaskedImageD)                 lsst::fw::MaskedImage<ImagePixelType, MaskPixelType>;
%template(MaskedImageDPtr)              boost::shared_ptr<lsst::fw::MaskedImage<ImagePixelType, MaskPixelType> >;

%template(BBox2i)	                BBox<int32, 2>;
%template(Vector2i)	                Vector<int32, 2>;

//%delobject boost::shared_ptr<vw::ImageView<MaskPixelType> >::shared_ptr;
//%apply SWIGTYPE *DISOWN {Foo *foo};
%extend_smart_pointer(boost::shared_ptr<vw::ImageView<MaskPixelType> >);
%template(MaskIVwPtrT)                  boost::shared_ptr<vw::ImageView<MaskPixelType> >;

%pythoncode %{
def ImageViewMaskPtr(*args):
    """Return an MaskIVwPtrT that owns its ImageMask"""

    Trace("fw.memory", 5, "creating ImageViewMaskPtr")

    im = ImageViewMask(*args)
    im.this.disown()
    ivmPtr = MaskIVwPtrT(im)

    Trace("fw.memory", 5, "returning ImageViewMaskPtr")
        
    return ivmPtr

def DataPropertyPtr(*args):
    """Return an DataPropertyPtrT that owns its DataProperty"""

    Trace("fw.memory", 5, "creating DataPropertyPtrT")

    md = DataProperty(*args)
    md.this.disown()
    DataPropertyPtr = DataPropertyPtrT(md)

    Trace("fw.memory", 5, "returning DataPropertyPtr")
        
    return DataPropertyPtr
%}

%template(listPixelCoord)  std::list<lsst::fw::PixelCoord>;

/************************************************************************************************************/

%{
    #include "lsst/fw/WCS.h"
%}

%template(Coord2D)	                Vector<double, 2>;
%rename(isValid) operator bool;
%include "lsst/fw/WCS.h"

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
