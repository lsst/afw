#if !defined(LSST_FITS_IO_MPL_H)
#define LSST_FITS_IO_MPL_H 1

#include <exception>
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/vector.hpp"

#include "boost/gil/gil_all.hpp"

#include "lsst/afw/image/lsstGil.h"
#include "fits_io.h"

namespace {
struct found_type : std::exception { }; // type to throw when we've read our data

template<typename ImageT, typename ExceptionT>
class try_fits_read_image {
public:
    try_fits_read_image(const std::string& file, ImageT& img,
#if 1                                   // Old name for boost::shared_ptrs
                        typename lsst::daf::base::DataProperty::PtrType metaData
#else
                        typename lsst::daf::base::DataProperty::ConstPtr metaData
#endif
                       ) : _file(file), _img(img), _metaData(metaData) { }

    void operator()(ImageT x) {         // read directly into the desired type if the file's the same type
        try {
            lsst::afw::image::fits_read_image(_file, _img, _metaData);
            throw ExceptionT();         // signal that we've succeeded
        } catch(lsst::pex::exceptions::FitsError const& e) {
            // ah well.  We'll try another image type
        }
    }

    template<typename U> void operator()(U x) { // read and convert into the desired type
        try {
            U img;
            lsst::afw::image::fits_read_image(_file, img, _metaData);

            _img.recreate(img.dimensions());
            boost::gil::copy_and_convert_pixels(const_view(img), view(_img));

            throw found_type();
        } catch(lsst::pex::exceptions::FitsError const& e) {
            // pass
        }
    }
private:
    std::string _file;
    ImageT& _img;
#if 1                                   // Old name for boost::shared_ptrs
    lsst::daf::base::DataProperty::PtrType _metaData;
#else
    lsst::daf::base::DataProperty::ConstPtr _metaData;
#endif
    
};

}

namespace lsst { namespace afw { namespace image {
template<typename fits_img_types, typename ImageT>
bool fits_read_image(std::string const& file, ImageT& img,
#if 1                                   // Old name for boost::shared_ptrs
                     lsst::daf::base::DataProperty::PtrType
                     metaData = lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0))
#else
                     lsst::daf::base::DataProperty::ConstPtr
                     metaData = lsst::daf::base::DataProperty::ConstPtr(static_cast<lsst::daf::base::DataProperty *>(0))
#endif
                    ) {
    try {
        boost::mpl::for_each<fits_img_types>(try_fits_read_image<ImageT, found_type>(file, img, metaData));
    } catch (found_type &e) {
        return true;                    // success
    }

    return false;
}
}}}                                     // lsst::afw::image

#endif
