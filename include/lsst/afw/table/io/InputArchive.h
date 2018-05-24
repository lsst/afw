// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_InputArchive_h_INCLUDED
#define AFW_TABLE_IO_InputArchive_h_INCLUDED

#include <map>

#include "lsst/base.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace table {

class BaseRecord;
template <typename RecordT>
class CatalogT;
typedef CatalogT<BaseRecord> BaseCatalog;

namespace io {

class CatalogVector;

/**
 *  A multi-catalog archive object used to load table::io::Persistable objects.
 *
 *  An InputArchive can be constructed directly from the catalogs produced by OutputArchive,
 *  or more usefully, read from a multi-extension FITS file.
 *
 *  @see OutputArchive
 */
class InputArchive {
public:
    typedef std::map<int, std::shared_ptr<Persistable>> Map;

    /// Construct an empty InputArchive that contains no objects.
    InputArchive();

    /// Construct an archive from catalogs.
    InputArchive(BaseCatalog const& index, CatalogVector const& dataCatalogs);

    /// Copy-constructor.  Does not deep-copy loaded Persistables.
    InputArchive(InputArchive const& other);
    InputArchive(InputArchive&& other);

    /// Assignment.  Does not deep-copy loaded Persistables.
    InputArchive& operator=(InputArchive const& other);
    InputArchive& operator=(InputArchive&& other);

    ~InputArchive();

    /**
     *  Load the Persistable with the given ID and return it.
     *
     *  If the object has already been loaded once, the same instance will be returned again.
     */
    std::shared_ptr<Persistable> get(int id) const;

    /// Load an object of the given type and ID with error checking.
    template <typename T>
    std::shared_ptr<T> get(int id) const {
        std::shared_ptr<T> p = std::dynamic_pointer_cast<T>(get(id));
        LSST_ARCHIVE_ASSERT(p || id == 0);
        return p;
    }

    /// Load and return all objects in the archive.
    Map const& getAll() const;

    /**
     *  Read an object from an already open FITS object.
     *
     *  @param[in]  fitsfile     FITS object to read from, already positioned at the desired HDU.
     */
    static InputArchive readFits(fits::Fits& fitsfile);

private:
    class Impl;

    InputArchive(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;
};
}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_IO_InputArchive_h_INCLUDED
