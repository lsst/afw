#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/AliasMap.h"

namespace lsst { namespace afw { namespace table {

void AliasMap::_apply(std::string & name) const {
    Iterator i = _internal.lower_bound(name);
    if (i != _internal.end()) {
        // equivalent to "if name.startswith(alias)" in Python
        if (name.size() >= i->first.size() && name.compare(0, i->first.size(), i->first) == 0) {
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
}

void AliasMap::remove(std::string const & alias) {
    _internal.erase(alias);
}

void AliasMap::clear() {
    _internal.clear();
}

}}} // namespace lsst::afw::table
