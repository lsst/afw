#ifndef LSST_LSSTFITSRESOURCE_H
#define LSST_LSSTFITSRESOURCE_H

#include "lsst/fw/DataProperty.h"
#include "lsst/fw/DiskImageResourceFITS.h"

namespace lsst {

    class LSSTFitsResource : public lsst::fw::DiskImageResourceFITS {
    public:
        LSSTFitsResource(std::string const& filename);
        DataProperty::DataPropertyPtrT getMetaData();
        void setMetaData(DataProperty::DataPropertyPtrT metaData);
    };
}

#endif // LSST_LSSTFITSRESOURCE_H
