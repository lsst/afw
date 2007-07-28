#ifndef LSST_LSSTFITSRESOURCE_H
#define LSST_LSSTFITSRESOURCE_H

#include <vw/Image.h>
#include "lsst/mwi/data/DataProperty.h"
#include "lsst/fw/DiskImageResourceFITS.h"

using namespace vw;

namespace lsst {

    namespace fw {

        using lsst::mwi::data::DataPropertyPtrT;

        template <typename PixelT> class LSSTFitsResource : public lsst::fw::DiskImageResourceFITS {
        public:
            LSSTFitsResource();
            void readFits(const std::string& filename, ImageView<PixelT>& image, DataPropertyPtrT metaData, int hdu=0);
            void writeFits(ImageView<PixelT>& image, DataPropertyPtrT metaData, const std::string& filename, int hdu=0);
        private:
            void getMetaData(DataPropertyPtrT metaData);
            void setMetaData(DataPropertyPtrT metaData);
        };
        
#include "LSSTFitsResource.cc"  

    } // namespace fw

} // namespace lsst

#endif // LSST_LSSTFITSRESOURCE_H
