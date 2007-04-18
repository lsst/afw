// -*- lsst-c++ -*-
// This file can NOT be separately compiled!   It is included by LSSTFitsResource.h

template <typename PixelT>
LSSTFitsResource<PixelT>::LSSTFitsResource(std::string const& filename) : DiskImageResourceFITS(filename)
{
}

template <typename PixelT>
void LSSTFitsResource<PixelT>::readFits(ImageView<PixelT>& image, DataProperty::DataPropertyPtrT metaData, int hdu)
{
    setHdu(hdu);
    read_image(image, *this);
    getMetaData(metaData);
}

template <typename PixelT>
void LSSTFitsResource<PixelT>::writeFits(ImageView<PixelT>&, DataProperty::DataPropertyPtrT metaData, int hdu)
{
}

// Private function to build a DataProperty that contains all the FITS kw-value pairs

template <typename PixelT>
void LSSTFitsResource<PixelT>::getMetaData(DataProperty::DataPropertyPtrT dpPtr)
{
     // Get all the kw-value pairs from the FITS file, and add each to DataProperty

     for (int i=1; i<=getNumKeys(); i++) {
	  std::string kw;
	  std::string val;
	  std::string comment;
	  getKey(i, kw, val, comment);
	  DataProperty::DataPropertyPtrT dpItemPtr(new DataProperty(kw, val));
	  dpPtr->addProperty(dpItemPtr);
     }

}

// Private function

template <typename PixelT>
void  LSSTFitsResource<PixelT>::setMetaData(DataProperty::DataPropertyPtrT metaData)
{
}
