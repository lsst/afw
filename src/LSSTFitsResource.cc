// -*- lsst-c++ -*-
// This file can NOT be separately compiled!   It is included by LSSTFitsResource.h

template <typename PixelT>
LSSTFitsResource<PixelT>::LSSTFitsResource() : DiskImageResourceFITS()
{
}

template <typename PixelT>
void LSSTFitsResource<PixelT>::readFits(const std::string& filename, ImageView<PixelT>& image, DataProperty::DataPropertyPtrT metaData, int hdu)
{
    open(filename);
    setHdu(hdu);
    read_image(image, *this);
    getMetaData(metaData);
}

template <typename PixelT>
void LSSTFitsResource<PixelT>::writeFits(ImageView<PixelT>& image, DataProperty::DataPropertyPtrT metaData, const std::string& filename, int hdu )
{
    create(filename, image.format());
//     setMetaData(metaData);
//     std::cout << "Numkeys: " << getNumKeys() << std::endl;
//     metaData->print();
    write_image(*this, image);
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
    DataProperty::DataPropertyContainerT dpC = metaData->getContents();
    DataProperty::DataPropertyContainerT::const_iterator pos;
    for (pos = dpC.begin(); pos != dpC.end(); pos++) {
        DataProperty::DataPropertyPtrT dpItemPtr = *pos;
        std::string tmp = boost::any_cast<const std::string>(dpItemPtr->getValue());
        appendKey(dpItemPtr->getName(), tmp, "");
    }
}
