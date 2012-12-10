// -*- lsst-c++ -*-
#ifndef AFW_IMAGE_WcsFactory_h_INCLUDED
#define AFW_IMAGE_WcsFactory_h_INCLUDED

#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace image {


class WcsFactory : public table::io::PersistableFactory {
public:

    explicit WcsFactory(std::string const & name);

    virtual PTR(table::io::Persistable) read(
        InputArchive const & archive,
        CatalogVector const & catalogs
    ) const;

    virtual ~WcsFactory() {}

};

}}} // namespace lsst::afw::image

#endif // !AFW_IMAGE_WcsFactory_h_INCLUDED
