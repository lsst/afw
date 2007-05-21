// -*- lsst-c++ -*-
// This file can NOT be separately compiled!   It is included by LSSTFitsResource.h

template <typename PixelT>
LSSTFitsResource<PixelT>::LSSTFitsResource() : DiskImageResourceFITS()
{
}

template <typename PixelT>
void LSSTFitsResource<PixelT>::readFits(const std::string& filename, ImageView<PixelT>& image, DataPropertyPtrT metaData, int hdu)
{
    open(filename);
    setHdu(hdu);
    read_image(image, *this);
    getMetaData(metaData);
}

template <typename PixelT>
void LSSTFitsResource<PixelT>::writeFits(ImageView<PixelT>& image, DataPropertyPtrT metaData, const std::string& filename, int hdu )
{
#if 0
    std::cout << metaData->repr();
#endif

    create(filename, image.format(), metaData.get());
    write_image(*this, image);
}

// Private function to build a DataProperty that contains all the FITS kw-value pairs

template <typename PixelT>
void LSSTFitsResource<PixelT>::getMetaData(DataPropertyPtrT dpPtr)
{
     // Get all the kw-value pairs from the FITS file, and add each to DataProperty

     for (int i=1; i<=getNumKeys(); i++) {
	  std::string kw;
	  std::string val;
	  std::string comment;
	  getKey(i, kw, val, comment);
	  DataPropertyPtrT dpItemPtr(new DataProperty(kw, stringToAny(val)));
	  dpPtr->addProperty(dpItemPtr);
     }

}
