// -*- lsst-c++ -*-
#ifndef AFW_TABLE_AliasMap_h_INCLUDED
#define AFW_TABLE_AliasMap_h_INCLUDED

#include <map>
#include <string>

#include "boost/shared_ptr.hpp"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Mapping class that holds aliases for a Schema
 *
 *  Aliases need not be complete, but they must map to the beginning of a field name to be useful.
 *  For example, if "a.b.c" is a true field name, "x.y->a.b" is a valid alias that will cause
 *  "x.y.c" to map to "a.b.c", but "y.z->b.c" will not cause "a.y.z" to be matched.
 *
 *  Aliases are not checked to see if they match any existing fields, and if an alias has the same
 *  name as a field name, it will take precedence and hide the true field.
 *
 *  Unlike the other components of a Schema, aliases can be modified and removed, even after a Table
 *  has been constructed from the Schema.
 */
class AliasMap {
public:

    // A map from aliases to field names (only public to appease Swig)
    typedef std::map<std::string,std::string> Internal;

    /// An iterator over alias->target pairs.
    typedef Internal::const_iterator Iterator;

    /// Return a iterator to the beginning of the map
    Iterator begin() const { return _internal.begin(); }

    /// Return a iterator to one past the end of the map
    Iterator end() const { return _internal.end(); }

    /// Apply any aliases that match the given field name and return a de-aliased name.
    std::string apply(std::string name) const;

    /**
     *  @brief Return the target of the given alias
     *
     *  Unlike apply(), this will not return partial matches.
     */
    std::string get(std::string const & name) const;

    /// Add an alias to the schema or replace an existing one.
    void set(std::string const & alias, std::string const & target);

    /// Remove an alias from the schema if it is present.
    void remove(std::string const & alias);

    /// Remove all aliases from the schema.
    void clear();

private:

    friend class Schema;
    friend class SubSchema;

    // Internal in-place implementation of apply()
    void _apply(std::string & name) const;

    Internal _internal;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_AliasMap_h_INCLUDED
