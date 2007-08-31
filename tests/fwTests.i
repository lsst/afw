// -*- lsst-c++ -*-
%define fwTests_DOCSTRING
"
Image processing code
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwTests_DOCSTRING) fwTests

%{
#   include "lsst/fw/Mask.h"
#   include "lsst/fw/MaskedImage.h"
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

%import  <vw/Core/FundamentalTypes.h>

%include "lsst/mwi/p_lsstSwig.i"
%include "lsst/fw/Core/lsstImageTypes.i"

%pythoncode %{
import lsst.mwi.data
import lsst.mwi.utils
%}

/******************************************************************************/

%import "lsst/mwi/utils/Utils.h"
%import "lsst/mwi/data/LsstData.h"
%import "lsst/mwi/data/DataProperty.h"
%import "lsst/mwi/exceptions/Exception.h"
%import "lsst/fw/Mask.h"
%import "lsst/fw/MaskedImage.h"
/******************************************************************************/
//
// Define a class to do very little with a PixelProcessingFunc
//
%inline %{
template <typename ImagePixelT, typename MaskPixelT>
class testPixProcFunc :
    public lsst::fw::PixelProcessingFunc<ImagePixelT, MaskPixelT> {
public:
    typedef typename PixelChannelType<ImagePixelT>::type ImageChannelT;
    typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
    typedef PixelLocator<ImagePixelT> ImageIteratorT;
    typedef PixelLocator<MaskPixelT> MaskIteratorT;
     
    testPixProcFunc(MaskedImage<ImagePixelT, MaskPixelT>& m) : PixelProcessingFunc<ImagePixelT, MaskPixelT>(m), initCount(0) {}
    
    void init() {
        PixelProcessingFunc<ImagePixelT, MaskPixelT>::_maskPtr->getPlaneBitMask("CR", bitsCR);
        testCount = 0;
        initCount++;
    }
        
    void operator ()(ImageIteratorT &i,MaskIteratorT &m ) { 
        //  In general, do something to the pixel values
        ImageIteratorT j = i;
        if (++testCount < 10) {
            //std::cout << *i << " " << *m << std::endl;
            *j = 1;
            int dx = 1;
            int dy = 0;
            if (initCount <2) *(j.advance(dx,dy)) = 2*testCount;
            //std::cout << "modified: " << *j << std::endl;
         }
     }

private:
    MaskChannelT bitsCR;
    int testCount;
    int initCount;
};
%}

namespace std {
    template<typename T, typename U>
    struct unary_function {};
}

%template(unary_function_tuple) std::unary_function<boost::tuple<vw::PixelGray<float > &,vw::PixelGray<uint8 > & > &,void >;
%template(unary_function_tuple2) std::unary_function<boost::tuple<ImagePixelType &,MaskPixelType & > &,void >;
%template(PixelProcessingFuncD) lsst::fw::PixelProcessingFunc<ImagePixelType, MaskPixelType>;
%template(testPixProcFuncD) testPixProcFunc<ImagePixelType, MaskPixelType>;

/******************************************************************************/
//
// Define a class to (in this case) count pixels with CR set
//
%inline %{
template <typename MaskPixelT>
class testCrFunc : public lsst::mwi::data::Citizen,
                   public MaskPixelBooleanFunc<MaskPixelT> {
public:
    typedef typename Mask<MaskPixelT>::MaskChannelT MaskChannelT;
    testCrFunc(Mask<MaskPixelT>& m) :
        lsst::mwi::data::Citizen(typeid(this)),
        MaskPixelBooleanFunc<MaskPixelT>(m) {}
    void init() {
        MaskPixelBooleanFunc<MaskPixelT>::_mask.getPlaneBitMask("CR", _bitsCR);
    }        
    bool operator ()(MaskPixelT pixel) const { 
        return ((pixel & _bitsCR) !=0 ); 
    }
private:
    MaskChannelT _bitsCR;
};
%}

%template(MaskPixelBooleanFuncD) lsst::fw::MaskPixelBooleanFunc<MaskPixelType>;
%template(testCrFuncD) testCrFunc<MaskPixelType>;

/******************************************************************************/
// Give python access to C++ copy constructors and/or operator=
// This are used to test ticket 144

// I'm not sure these are needed (they had no effect on ticket 144) but just in case...
%newobject copyMask;
%newobject copyImage;
%newobject copyMaskedImage;

%inline %{
    template <typename ImageT>
    lsst::fw::Image<ImageT> copyImage(lsst::fw::Image<ImageT> &src) {
        return src;
    }

    template <typename MaskT>
    lsst::fw::Mask<MaskT> copyMask(lsst::fw::Mask<MaskT> &src) {
        return src;
    }

    template <typename ImageT, typename MaskT>
    lsst::fw::MaskedImage<ImageT, MaskT> copyMaskedImage(lsst::fw::MaskedImage<ImageT, MaskT> &src) {
        return src;
    }
%}

%template(copyImageF) copyImage<ImagePixelType>;
%template(copyMaskF) copyMask<MaskPixelType>;
%template(copyMaskedImageF) copyMaskedImage<ImagePixelType, MaskPixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
