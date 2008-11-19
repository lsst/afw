/**
 * \file
 * \brief  Internal support for reading and writing FITS files
 *
 * Tell doxygen to (usually) ignore this file \cond GIL_IMAGE_INTERNALS
 * \author Robert Lupton (rhl@astro.princeton.edu)
 *         Princeton University
 * \date   September 2008
 */
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
                        typename lsst::daf::base::DataProperty::PtrType metadata
#else
                        typename lsst::daf::base::DataProperty::ConstPtr metadata
#endif
                       ) : _file(file), _img(img), _metadata(metadata) { }

    void operator()(ImageT x) {         // read directly into the desired type if the file's the same type
        try {
            lsst::afw::image::fits_read_image(_file, _img, _metadata);
            throw ExceptionT();         // signal that we've succeeded
        } catch(lsst::pex::exceptions::FitsError const& e) {
            // ah well.  We'll try another image type
        }
    }

    template<typename U> void operator()(U x) { // read and convert into the desired type
        try {
            U img;
            lsst::afw::image::fits_read_image(_file, img, _metadata);

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
    lsst::daf::base::DataProperty::PtrType _metadata;
#else
    lsst::daf::base::DataProperty::ConstPtr _metadata;
#endif
    
};

}

namespace lsst { namespace afw { namespace image {
template<typename fits_img_types, typename ImageT>
bool fits_read_image(std::string const& file, ImageT& img,
#if 1                                   // Old name for boost::shared_ptrs
                     lsst::daf::base::DataProperty::PtrType
                     metadata = lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0))
#else
                     lsst::daf::base::DataProperty::ConstPtr
                     metadata = lsst::daf::base::DataProperty::ConstPtr(static_cast<lsst::daf::base::DataProperty *>(0))
#endif
                    ) {
    try {
        boost::mpl::for_each<fits_img_types>(try_fits_read_image<ImageT, found_type>(file, img, metadata));
    } catch (found_type &e) {
        return true;                    // success
    }

    return false;
}
}}}                                     // lsst::afw::image
/// \endcond
#endif
