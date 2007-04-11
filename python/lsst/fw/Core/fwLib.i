// -*- lsst-c++ -*-
%define fwLib_DOCSTRING
"
Basic routines to talk to FW's classes (including visionWorkbench) and ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwLib_DOCSTRING) fwLib

#if 0
   %rename("%(command:perl -pe 's/^act(.)/\l$1/' <<< )s") "";
#else
//   %include "fwRename.i"
#endif


// Suppress swig complaints from vw
// 317: Specialization of non-template
// 389: operator[] ignored
// 362: operator=  ignored
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=317
#pragma SWIG nowarn=362
#pragma SWIG nowarn=389

%{
#   include <exception>
#   include <map>
#   include <boost/cstdint.hpp>
#   include <boost/shared_ptr.hpp>
#   include <boost/any.hpp>
#   include "lsst/fw/Citizen.h"
#   include "lsst/fw/Demangle.h"
#   include "lsst/fw/DiskImageResourceFITS.h"
#   include "lsst/fw/Mask.h"
#   include "lsst/fw/MaskedImage.h"
#   include "lsst/fw/Trace.h"
#   include "lsst/fw/Utils.h"

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
    """Return a version given a HeadURL string; default: fw's version"""
    return guessSvnVersion(HeadURL)

%}

/******************************************************************************/

%ignore vw::ImageView<ImagePixelType>::origin;
%ignore vw::ImageView<MaskPixelType>::origin;
%ignore operator vw::ImageView::unspecified_bool_type;
%ignore operator lsst::Mask::operator()(int, int); // RHL can't get this to work

%include <vw/Image/ImageViewBase.h>
%include <vw/Image/ImageView.h>
%include <vw/Image/PixelTypeInfo.h>
%include <vw/Image/PixelTypes.h>
%include <vw/Image/ImageResource.h>
%include <vw/Math/BBox.h>

%import <vw/FileIO/DiskImageResource.h>
%include "lsst/fw/DiskImageResourceFITS.h"

/******************************************************************************/
// Citizens, Trace, etc.
%include "lsst/fw/Citizen.h"
%include "lsst/fw/Trace.h"
%include "lsst/fw/DataProperty.h"

#if 0                                   // doesn't work (yet)
typedef boost::shared_ptr<DataProperty> DataPropertyPtr;

%contract DataPropertyPtr::DataPropertyPtr {
ensure:
    DataPropertyPtr_ptr.get() > 0;
}
#endif
    
%template(DataPropertyPtrT) boost::shared_ptr<DataProperty>;

%extend lsst::DataProperty {
    DataProperty(std::string name, int val) {
        return new DataProperty(name, val);
    }
    DataProperty(std::string name, std::string val) {
        return new DataProperty(name, val);
    }
}

/******************************************************************************/
// Masks and MaskedImages
%newobject getMaskPlaneMetaData;
%clear int &;
%template(pairIntString) std::pair<int,std::string>;
%template(mapIntString)  std::map<int,std::string>;
%apply int &OUTPUT { int & };

%include "lsst/fw/Mask.h"
%include "lsst/fw/MaskedImage.h"

using namespace lsst;

%template(ImageBaseFloat) vw::ImageViewBase<vw::ImageView<ImagePixelType> >;
%template(ImageFloat)     vw::ImageView<ImagePixelType>;

%template(ImageBaseMask)  vw::ImageViewBase<vw::ImageView<MaskPixelType> >;
%template(ImageMask)      vw::ImageView<MaskPixelType>;

%template(MaskD)          lsst::Mask<MaskPixelType>;
%template(MaskedImageD)   lsst::MaskedImage<ImagePixelType, MaskPixelType>;
%template(MaskDPtr)       boost::shared_ptr<lsst::Mask<MaskPixelType> >;
%template(BBox2i)         BBox<int32, 2>;

%extend_smart_pointer(boost::shared_ptr<vw::ImageView<MaskPixelType> >);
//%delobject boost::shared_ptr<vw::ImageView<MaskPixelType> >::shared_ptr;
//%apply SWIGTYPE *DISOWN {Foo *foo};
%template(MaskIVwPtrT) boost::shared_ptr<vw::ImageView<MaskPixelType> >;

%pythoncode %{
def ImageMaskPtr(*args):
    """Return an MaskIVwPtrT that owns its ImageMask"""

    trace("fw.memory", 5, "creating maskImagePtr")

    im = ImageMask(*args)
    im.this.disown()
    maskImagePtr = MaskIVwPtrT(im)

    trace("fw.memory", 5, "returning maskImagePtr")
        
    return maskImagePtr
%}

%template(listPixelCoord)  std::list<lsst::PixelCoord>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
