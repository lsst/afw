#include "lsst/fw/LSSTFitsResource.h"

using namespace lsst;

LSSTFitsResource::LSSTFitsResource(std::string const& filename) : DiskImageResourceFITS(filename)
{
}

// Build a DataProperty that contains all the FITS kw-value pairs
// Any "indexed keywords", of the form name0, name1, etc, will be collected together
// into a single DataProperty with name "name"

DataProperty::DataPropertyPtrT LSSTFitsResource::getMetaData()
{
     DataProperty::DataPropertyPtrT dpPtr(new DataProperty("FitsMetaData", 0));

     // Get all the kw-value pairs from the FITS file, and add each to DataProperty

     for (int i=1; i<=getNumKeys(); i++) {
	  std::string kw;
	  std::string val;
	  std::string comment;
	  // do we need to reset the strings first?
	  getKey(i, kw, val, comment);
     }

     return dpPtr;
}

void  LSSTFitsResource::setMetaData(DataProperty::DataPropertyPtrT metaData)
{
}
