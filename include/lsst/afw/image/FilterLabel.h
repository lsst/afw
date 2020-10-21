/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_AFW_IMAGE_FILTERLABEL_H
#define LSST_AFW_IMAGE_FILTERLABEL_H

#include <memory>
#include <string>

#include "lsst/afw/table/io/Persistable.h"  // Needed for PersistableFacade
#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace image {

/**
 * A group of labels for a filter in an exposure or coadd.
 *
 * This class provides only identifiers for filters. Other filter information
 * can be retrieved from an Exposure object or a Butler repository.
 *
 * FilterLabel does not expose a public constructor in C++, except for copy
 * and move constructors. You can create a FilterLabel by calling one of the
 * `from*` factory methods, or (in Python) through a keyword-only constructor.
 */
class FilterLabel final : public table::io::PersistableFacade<FilterLabel>, public typehandling::Storable {
public:
    /**
     * Construct a FilterLabel from specific inputs.
     *
     * @{
     */
    static FilterLabel fromBandPhysical(std::string const &band, std::string const &physical);
    static FilterLabel fromBand(std::string const &band);
    static FilterLabel fromPhysical(std::string const &physical);
    /** @} */

    /// Return whether the filter label names a band.
    bool hasBandLabel() const noexcept;

    /**
     * Return the band label.
     *
     * @returns The band label.
     * @throws lsst::pex::exceptions::LogicError Thrown if hasBandLabel() is `false`.
     */
    std::string getBandLabel() const;

    /// Return whether the filter label names a physical filter.
    bool hasPhysicalLabel() const noexcept;

    /**
     * Return the physical filter label.
     *
     * @returns The physical filter label.
     * @throws lsst::pex::exceptions::LogicError Thrown if hasPhysicalLabel() is `false`.
     */
    std::string getPhysicalLabel() const;

    /**
     * Filter labels compare equal if their components are equal.
     *
     * @note This operation does not test whether two *filters* are the same.
     *       Two FilterLabels corresponding to identically-named filters on
     *       different instruments will compare equal.
     *
     * @{
     */
    bool operator==(FilterLabel const &rhs) const noexcept;
    bool operator!=(FilterLabel const &rhs) const noexcept { return !(*this == rhs); }
    /** @} */

private:
    FilterLabel(bool hasBand, std::string const &band, bool hasPhysical, std::string const &physical);

    // A separate boolean leads to easier implementations (at the cost of more
    // memory) than a unique_ptr<string>.
    // _band and _physical are part of the object state iff _hasBand and _hasPhysical, respectively
    bool _hasBand, _hasPhysical;
    std::string _band, _physical;
};

}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif
