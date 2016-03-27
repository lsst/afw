// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/AliasMap.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table {

void AliasMap::_apply(std::string & name) const {
    // Loop in order to keep replacing as long as we keep finding matches,
    // but we count how many replacements we've made to avoid an infinite loop
    // due to a cycle between aliases.  That's not the most efficient way to
    // find cycles, but since a cycle represents a bug in some other code that
    // should be rare, we don't really care.
    for (std::size_t count = 0; count <= _internal.size(); ++count) {
        Iterator i = _internal.lower_bound(name);
        if (i != _internal.end() && i->first.size() == name.size() && i->first == name) {
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
            } else {
                return;  // no match; exit
            }
        } else {
            return;  // no match; exit
        }
    }
    throw LSST_EXCEPT(
        pex::exceptions::RuntimeError,
        (boost::format("Cycle detected in schema aliases involving name '%s'") % name).str()
    );
}

std::string AliasMap::apply(std::string const & name) const {
    std::string result(name);
    _apply(result);
    return result;
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
    _internal[alias] = target;
    if (_table) {
        _table->handleAliasChange(alias);
    }
}

bool AliasMap::erase(std::string const & alias) {
    bool result = _internal.erase(alias);
    if (_table) {
        _table->handleAliasChange(alias);
    }
    return result;
}

bool AliasMap::operator==(AliasMap const & other) const {
    return _internal == other._internal;
}

bool AliasMap::contains(AliasMap const & other) const {
    return std::includes(begin(), end(), other.begin(), other.end());
}

}}} // namespace lsst::afw::table
