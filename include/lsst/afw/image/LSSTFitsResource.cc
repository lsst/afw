// -*- lsst-c++ -*-
// This file can NOT be separately compiled!   It is included by LSSTFitsResource.h
#include <lsst/pex/exceptions.h>

template <typename PixelT>
lsst::afw::image::LSSTFitsResource<PixelT>::LSSTFitsResource() : DiskImageResourceFITS()
{
}

template <typename PixelT>
void lsst::afw::image::LSSTFitsResource<PixelT>::readFits(
    const std::string& filename,
    vw::ImageView<PixelT>& image,
    lsst::daf::base::DataProperty::PtrType metaData,
    int hdu)
{
    open(filename);
    setHdu(hdu);
    read_image(image, *this);
    getMetaData(metaData);
}

template <typename PixelT>
void lsst::afw::image::LSSTFitsResource<PixelT>::writeFits(
    vw::ImageView<PixelT>& image,
    lsst::daf::base::DataProperty::PtrType metaData,
    const std::string& filename,
    int hdu)
{
#if 0
    std::cout << metaData->toString("",true) << std::endl;
#endif

    create(filename, image.format(), metaData.get());
    write_image(*this, image);
}

// Private function to build a lsst::daf::base::DataProperty that contains all the FITS kw-value pairs

template <typename PixelT>
void lsst::afw::image::LSSTFitsResource<PixelT>::getMetaData(
    lsst::daf::base::DataProperty::PtrType dpPtr)
{
     // Get all the kw-value pairs from the FITS file, and add each to lsst::daf::base::DataProperty

     if( dpPtr->isNode() != true ) {
        throw lsst::pex::exceptions::InvalidParameter( "Given metadata object is not a lsst::daf::base::DataProperty node" );
     }
     
     for (int i=1; i<=getNumKeys(); i++) {
	  std::string kw;
	  std::string val;
	  std::string comment;
	  getKey(i, kw, val, comment);
	  lsst::daf::base::DataProperty::PtrType dpItemPtr(new lsst::daf::base::DataProperty(kw, stringToAny(val)));
	  dpPtr->addProperty(dpItemPtr);
     }
}
