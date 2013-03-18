// -*- lsst-c++ -*-

/**
 *  @file lsst/afw/detection/KernelPsfFactory.h
 *
 *  Utilities for persisting KernelPsf and subclasses thereof.  Should only be included
 *  directly in source files and never swigged.
 */

#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst { namespace afw { namespace detection {

/**
 *  @brief A read-only singleton struct containing the schema and key used in persistence for KernelPsf.
 *
 *  Implementations are in src/detection/Psf.cc.
 */
struct KernelPsfPersistenceHelper : private boost::noncopyable {
    afw::table::Schema schema;
    afw::table::Key<int> kernel;
    afw::table::Key<afw::table::Point<double> > averagePosition;

    static KernelPsfPersistenceHelper const & get();

private:
    KernelPsfPersistenceHelper();
};

/**
 *  @brief A PersistableFactory for KernelPsf and its subclasses.
 *
 *  If a KernelPsf subclass has no data members other than its kernel, table persistence for
 *  it can be implemented simply by reimplementing getPersistenceName() and registering
 *  a specialization of KernelPsfFactory.
 */
template <typename T=KernelPsf, typename K=afw::math::Kernel>
class KernelPsfFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(table::io::InputArchive const & archive, table::io::CatalogVector const & catalogs) const {
        static KernelPsfPersistenceHelper const & keys = KernelPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        table::BaseRecord const & record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        return PTR(T)(
            new T(
                archive.get<K>(record.get(keys.kernel)),
                record.get(keys.averagePosition)
            )
        );
    }

    KernelPsfFactory(std::string const & name) : table::io::PersistableFactory(name) {}

};

}}} // namespace lsst::afw::detection
