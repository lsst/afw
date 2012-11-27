// -*- lsst-c++ -*-
#ifndef AFW_DETECTION_PsfRecordGeneratorFactory_h_INCLUDED
#define AFW_DETECTION_PsfRecordGeneratorFactory_h_INCLUDED

#include "lsst/afw/table/generators.h"

namespace lsst { namespace afw { namespace detection {

class Psf;

/**
 *  @brief Factory base class used to implement Psf::readFromRecords.
 *
 *  Subclasses should be instantiated in a file scope variable exactly once; the base class constructor
 *  will then add that factory to the singleton registry.
 */
class PsfRecordGeneratorFactory : private boost::noncopyable {
public:

    explicit PsfRecordGeneratorFactory(std::string const & name);

    virtual PTR(Psf) operator()(table::RecordInputGeneratorSet const & inputs) const = 0;

    virtual ~PsfRecordGeneratorFactory() {}

};

}}} // namespace lsst::afw::detection

#endif // !AFW_DETECTION_PsfRecordGeneratorFactory_h_INCLUDED
