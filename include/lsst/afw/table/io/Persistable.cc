#include <memory>

#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

/**
 * Dynamically cast a shared pointer and raise on failure.
 *
 * param[in] ptr  The pointer to be cast.
 * @returns The cast pointer.
 * @throws lsst::pex::exceptions::TypeError If the dynamic cast fails.
 */
template <typename T>
std::shared_ptr<T> PersistableFacade<T>::dynamicCast(std::shared_ptr<Persistable> const &ptr) {
    auto result = std::dynamic_pointer_cast<T>(ptr);
    if (!result) {
        throw LSST_EXCEPT(pex::exceptions::TypeError, "Dynamic pointer cast failed");
    }
    return result;
}

}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst
