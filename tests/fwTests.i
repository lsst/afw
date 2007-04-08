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

using namespace lsst;
using namespace lsst::fw;
using namespace vw;
%}

%init %{
%}

%import  <vw/Core/FundamentalTypes.h>
%include "lsst/fw/Core/p_lsstSwig.i"

/******************************************************************************/

%inline %{
typedef vw::PixelGray<float> ImagePixelType;
typedef vw::PixelGray<uint8> MaskPixelType;
%}

%import "lsst/fw/Mask.h"
%import "lsst/fw/MaskedImage.h"

using namespace lsst;
using namespace vw;

/******************************************************************************/
//
// Define a class to do very little with a PixelProcessingFunc
//

%inline %{
template <typename ImagePixelT, typename MaskPixelT>
class testPixProcFunc :
    public lsst::PixelProcessingFunc<ImagePixelT, MaskPixelT> {
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
            std::cout << *i << " " << *m << std::endl;
            *j = 1;
            int dx = 1;
            int dy = 0;
            if (initCount <2) *(j.advance(dx,dy)) = 2*testCount;
            std::cout << "modified: " << *j << std::endl;
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
%template(PixelProcessingFuncD) lsst::PixelProcessingFunc<ImagePixelType, MaskPixelType>;
%template(testPixProcFuncD) testPixProcFunc<ImagePixelType, MaskPixelType>;

/******************************************************************************/
//
// Define a class to (in this case) count pixels with CR set
//
%import "lsst/fw/Utils.h"
%import "lsst/fw/Citizen.h"

%inline %{
template <typename MaskPixelT>
class testCrFunc : public lsst::fw::Citizen,
                   public MaskPixelBooleanFunc<MaskPixelT> {
public:
    typedef typename Mask<MaskPixelT>::MaskChannelT MaskChannelT;
    testCrFunc(Mask<MaskPixelT>& m) :
        Citizen(typeid(this)),
        MaskPixelBooleanFunc<MaskPixelT>(m) {}
    void init() {
        MaskPixelBooleanFunc<MaskPixelT>::_mask.getPlaneBitMask("CR", _bitsCR);
    }        
    bool operator ()(MaskPixelT pixel) const { 
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
