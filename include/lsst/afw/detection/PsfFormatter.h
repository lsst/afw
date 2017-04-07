// -*- LSST-C++ -*-
#if !defined(LSST_AFW_DETECTION_PSFFORMATTER_H)
#define LSST_AFW_DETECTION_PSFFORMATTER_H

/*
 * Interface for PsfFormatter class
 */

#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/persistence/Formatter.h"
#include "lsst/daf/persistence/Storage.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace detection {

/**
 * Formatter for persistence of Psf instances.
 */
class PsfFormatter : public lsst::daf::persistence::Formatter {
public:
    /** Minimal destructor.
     */
    virtual ~PsfFormatter(void);

    virtual void write(lsst::daf::base::Persistable const* persistable,
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData);

    virtual lsst::daf::base::Persistable* read(lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData);

    virtual void update(lsst::daf::base::Persistable* persistable,
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData);

    /** Serialize a Psf to a Boost archive.  Handles text or XML
     * archives, input or output.
     *
     * @param[in, out] ar Boost archive
     * @param[in] version Version of the Psf class
     * @param[in] persistable persistable Pointer to the Psf as a Persistable
     */
    template <class Archive>
    static void delegateSerialize(Archive& ar, unsigned int const version,
        lsst::daf::base::Persistable* persistable);

private:
    /** Constructor.
     * @param[in] policy Policy for configuring this Formatter
     */
    explicit PsfFormatter(lsst::pex::policy::Policy::Ptr policy);

    lsst::pex::policy::Policy::Ptr _policy;

    /** Factory method for PsfFormatter.
     * @param[in] policy Policy for configuring the PsfFormatter
     * @returns Shared pointer to a new instance
     */
    static lsst::daf::persistence::Formatter::Ptr
        createInstance(lsst::pex::policy::Policy::Ptr policy);

    /** Register this Formatter subclass through a static instance of
     * FormatterRegistration.
     */
    static lsst::daf::persistence::FormatterRegistration registration;
    static lsst::daf::persistence::FormatterRegistration doubleGaussianPsfRegistration;
    static lsst::daf::persistence::FormatterRegistration pcaPsfRegistration;
};

}}}

#endif
