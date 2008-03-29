// -*- lsst-c++ -*-
%define fwTests_DOCSTRING
"
Image processing code
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwTests_DOCSTRING) fwTests

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored

%{
#   include "lsst/afw/image/Mask.h"
#   include "lsst/afw/image/MaskedImage.h"
%}

%inline %{
namespace lsst { namespace afw { } }
namespace vw {}
namespace boost { namespace filesystem {} }
    
using namespace lsst;
using namespace lsst::afw;
using namespace vw;
%}

%init %{
%}

%import  <vw/Core/FundamentalTypes.h>

%include "lsst/mwi/p_lsstSwig.i"
%include "lsst/afw/image/lsstImageTypes.i"

%pythoncode %{
import lsst.mwi.data
import lsst.mwi.utils
import lsst.fw.exceptions
%}

/******************************************************************************/

%import "lsst/mwi/utils/Utils.h"
%import "lsst/daf/base/Citizen.h"
%import "lsst/pex/policy/Policy.h"
%import "lsst/daf/base/Persistable.h"
%import "lsst/daf/data/LsstData.h"
%import "lsst/daf/base/DataProperty.h"
%import "lsst/pex/exceptions.h"
%import "lsst/afw/image/Mask.h"
%import "lsst/afw/image/MaskedImage.h"
/******************************************************************************/
//
// Define a class to do very little with a PixelProcessingFunc
//
%inline %{
template <typename ImagePixelT, typename MaskPixelT>
class testPixProcFunc :
    public lsst::afw::image::PixelProcessingFunc<ImagePixelT, MaskPixelT> {
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
%template(unary_function_tupleF) std::unary_function<boost::tuple<float &,lsst::afw::image::maskPixelType & > &,void >;
%template(unary_function_tupleD) std::unary_function<boost::tuple<double &,lsst::afw::image::maskPixelType & > &,void >;
%template(PixelProcessingFuncF) lsst::afw::image::PixelProcessingFunc<float, lsst::afw::image::maskPixelType>;
%template(PixelProcessingFuncD) lsst::afw::image::PixelProcessingFunc<double, lsst::afw::image::maskPixelType>;
%template(testPixProcFuncF) testPixProcFunc<float, lsst::afw::image::maskPixelType>;
%template(testPixProcFuncD) testPixProcFunc<double, lsst::afw::image::maskPixelType>;

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

%template(MaskPixelBooleanFuncD) lsst::afw::image::MaskPixelBooleanFunc<lsst::afw::image::maskPixelType>;
%template(testCrFuncD) testCrFunc<lsst::afw::image::maskPixelType>;

/******************************************************************************/
// Give python access to C++ copy constructors and/or operator=
// This are used to test ticket 144

// I'm not sure these are needed (they had no effect on ticket 144) but just in case...
%newobject copyMask;
%newobject copyImage;
%newobject copyMaskedImage;

%inline %{
    template <typename ImageT>
    lsst::afw::image::Image<ImageT> copyImage(lsst::afw::image::Image<ImageT> &src) {
        return src;
    }

    template <typename MaskT>
    lsst::afw::image::Mask<MaskT> copyMask(lsst::afw::image::Mask<MaskT> &src) {
        return src;
    }

    template <typename ImageT, typename MaskT>
    lsst::afw::image::MaskedImage<ImageT, MaskT> copyMaskedImage(lsst::afw::image::MaskedImage<ImageT, MaskT> &src) {
        return src;
    }
%}

%template(copyImageF) copyImage<float>;
%template(copyImageD) copyImage<double>;
%template(copyMaskU)  copyMask<lsst::afw::image::maskPixelType>;
%template(copyMaskedImageF) copyMaskedImage<float, lsst::afw::image::maskPixelType>;
%template(copyMaskedImageD) copyMaskedImage<double, lsst::afw::image::maskPixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
