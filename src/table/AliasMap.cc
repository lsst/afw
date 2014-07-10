// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
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
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/AliasMap.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table {

namespace {

// equivalent to a.startswith(b) in Python
bool startsWith(std::string const & a, std::string const & b) {
    return a.size() >= b.size() && a.compare(0, b.size(), b) == 0;
}

} // anonymous

void AliasMap::_apply(std::string & name) const {
    Iterator i = _internal.lower_bound(name);
    if (i != _internal.end() && i->first.size() == name.size()) {
        // We have a complete match: alias matches the full name we were given
        name = i->second;
    } else if (i != _internal.begin()) {
        // Might still have a partial match: in this case, the iterator
        // lower_bound returns will be one past the best match, since
        // smaller strings are considered "less than" longer strings
        // that they share the same initial charaters with.
        --i;
        if (i->first.size() < name.size() && name.compare(0, i->first.size(), i->first) == 0) {
            name.replace(0, i->first.size(), i->second);
        }
    }
}

std::string AliasMap::apply(std::string name) const {
    _apply(name);
    return name;
}

std::string AliasMap::get(std::string const & name) const {
    Iterator i = _internal.lower_bound(name);
    if (i == _internal.end()) {
        throw LSST_EXCEPT(
            pex::exceptions::NotFoundError,
            (boost::format("Alias '%s' not found") % name).str()
        );
    }
    return i->second;
}

void AliasMap::set(std::string const & alias, std::string const & target) {
    _internal.insert(std::make_pair(alias, target));
    if (_table) {
        _table->handleAliasChange(alias);
    }
}

void AliasMap::remove(std::string const & alias) {
    _internal.erase(alias);
    if (_table) {
        _table->handleAliasChange(alias);
    }
}

}}} // namespace lsst::afw::table
