// -*- lsst-c++ -*-
%define fwLib_DOCSTRING
"
Basic routines to talk to FW's classes (including visionWorkbench) and ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwLib_DOCSTRING, naturalvar=1) fwLib

#if 0
   %rename("%(command:perl -pe 's/^act(.)/\l$1/' <<< )s") "";
#else
//   %include "fwRename.i"
#endif


// Suppress swig complaints from vw
// 317: Specialization of non-template
// 389: operator[] ignored
// 362: operator=  ignored
// 503: Can't wrap 'operator unspecified_bool_type()'
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=317
#pragma SWIG nowarn=362
#pragma SWIG nowarn=389
%warnfilter(503) vw;

%{
#   include <boost/cstdint.hpp>
#   include "lsst/Citizen.h"
#   include "lsst/DiskImageResourceFITS.h"
#   include "lsst/Mask.h"
#   include "lsst/MaskedImage.h"
#   include "lsst/Trace.h"

using namespace lsst;
using namespace lsst::fw;
using namespace vw;
%}

%init %{
%}

%include "../Core/p_lsstSwig.i"
%import "lsst/Utils.h"

/******************************************************************************/

%{
typedef vw::PixelGray<float> ImagePixelType;
typedef vw::PixelGray<uint8> MaskPixelType;
%}

%ignore vw::ImageView<ImagePixelType>::origin;
%ignore vw::ImageView<MaskPixelType>::origin;

%include <vw/Image/ImageViewBase.h>
%include <vw/Image/ImageView.h>
%include <vw/Image/PixelTypeInfo.h>
%include <vw/Image/PixelTypes.h>
%include <vw/Image/ImageResource.h>
%include <vw/Math/BBox.h>

//%include <vw/Math/Vector.h>

%apply int {int32};
%apply int {vw::int32};

%import "vw/FileIO/DiskImageResource.h"
%include "lsst/DiskImageResourceFITS.h"


/******************************************************************************/
// Citizens, Trace, etc.
%include "lsst/Citizen.h"
%include "lsst/Trace.h"

/******************************************************************************/
// Masks and MaskedImages

%include "lsst/Mask.h"
//%include "lsst/MaskedImage.h"

using namespace lsst;

%template(ImageBaseFloat) vw::ImageViewBase<vw::ImageView<ImagePixelType> >;
%template(ImageFloat) vw::ImageView<ImagePixelType>;

%template(ImageBaseMask)  vw::ImageViewBase<vw::ImageView<MaskPixelType> >;
%template(ImageMask)  vw::ImageView<MaskPixelType>;

%template(MaskD)      lsst::Mask<MaskPixelType>;
//%template(MaskedImageD) lsst::MaskedImage<ImagePixelType, MaskPixelType>;
%template(BBox2i)     BBox<int32, 2>;

%template(listPixelCoord)  std::list<lsst::PixelCoord>;

//%template(pairInt)  vw::Vector<int, 2>;

/******************************************************************************/
//
// Define a class to (in this case) count pixels with CR set
//
%inline %{
template <typename MaskPixelT>
class testCrFunc : public MaskPixelBooleanFunc<MaskPixelT> {
public:
    typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
    testCrFunc(Mask<MaskPixelT>& m) : MaskPixelBooleanFunc<MaskPixelT>(m) {}
    void init() {
        MaskPixelBooleanFunc<MaskPixelT>::_mask.getPlaneBitMask("CR", _bitsCR);
    }        
    bool operator ()(MaskPixelT pixel) { 
        return ((pixel.v() & _bitsCR) !=0 ); 
    }
private:
    MaskChannelT _bitsCR;
};
%}

%template(MaskPixelBooleanFuncD) lsst::MaskPixelBooleanFunc<MaskPixelType>;
%template(testCrFuncD) testCrFunc<MaskPixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
