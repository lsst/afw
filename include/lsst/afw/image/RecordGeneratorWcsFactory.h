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

    /// Return the (immutable) schema used by the base class implementation.
    static table::Schema getSchema();

    /// Construct a factory and register it in the singleton registry.
    explicit RecordGeneratorWcsFactory(std::string const & name);

    /// Use the factory to create a Wcs object.
    virtual PTR(Wcs) operator()(table::RecordInputGeneratorSet const & inputs) const;

    virtual ~RecordGeneratorWcsFactory() {}

};

}}} // namespace lsst::afw::image

#endif // !AFW_IMAGE_RecordGeneratorWcsFactory_h_INCLUDED
