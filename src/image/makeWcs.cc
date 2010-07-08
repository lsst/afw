// -*- lsst-c++ -*-
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"

namespace except = lsst::pex::exceptions; 
namespace afwImg = lsst::afw::image;

using namespace std;

typedef lsst::daf::base::PropertySet PropertySet;
typedef lsst::afw::image::Wcs Wcs;
typedef lsst::afw::image::PointD PointD;


/// Use this function to create a Wcs object using a fits header. It examines the header and determines the 
/// most suitable object to return, either a general Wcs object, or a more specific object specialised to a 
/// given coordinate system (e.g TanWcs)
/// 
afwImg::Wcs::Ptr afwImg::makeWcs(PropertySet::Ptr fitsMetadata){
    std::string ctype1;
    if( fitsMetadata->exists("CTYPE1")) {
        ctype1 = fitsMetadata->getAsString("CTYPE1");
    } else {
        return afwImg::Wcs::Ptr(new afwImg::Wcs());
    }
    
    if( ctype1.substr(5, 3) == "TAN") {
        return afwImg::Wcs::Ptr(new afwImg::TanWcs(fitsMetadata));
    }
    
    //Default Wcs class
    return afwImg::Wcs::Ptr(new afwImg::Wcs(fitsMetadata));
}
    
