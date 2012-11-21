// -*- lsst-c++ -*-
#ifndef AFW_IMAGE_RecordGeneratorWcsFactory_h_INCLUDED
#define AFW_IMAGE_RecordGeneratorWcsFactory_h_INCLUDED

#include "lsst/afw/table/generators.h"

namespace lsst { namespace afw { namespace image {

class Wcs;

/**
 *  @brief Factory base class used to implement Wcs::readFromRecords.
 *
 *  Subclasses should be instantiated in a file scope variable exactly once; the base class constructor
 *  will then add that factory to the singleton registry.
 */
class RecordGeneratorWcsFactory : private boost::noncopyable {
public:

    explicit RecordGeneratorWcsFactory(std::string const & name);

    virtual PTR(Wcs) operator()(table::RecordInputGeneratorSet const & inputs) const = 0;

    virtual ~RecordGeneratorWcsFactory() {}

};

}}} // namespace lsst::afw::image

#endif // !AFW_IMAGE_RecordGeneratorWcsFactory_h_INCLUDED
