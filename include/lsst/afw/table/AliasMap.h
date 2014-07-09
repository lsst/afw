// -*- lsst-c++ -*-
#ifndef AFW_TABLE_AliasMap_h_INCLUDED
#define AFW_TABLE_AliasMap_h_INCLUDED

#include <map>
#include <string>

namespace lsst { namespace afw { namespace table {

class BaseTable;

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
 *
 *  AliasMaps are shared when Schemas are copy-constructed, but can be separated manually
 *  by calling Schema::disconnectAliases() or Schema::setAliases().  In addition, the AliasMap
 *  is deep-copied when used to construct a Table (or Catalog).
 *
 *  In order to allow Tables to react to changes in aliases (which may be used to define cached Keys
 *  held by the table, as in SourceTable's "slots" mechanism), an AliasMap that is part of a Schema held
 *  by a Table will hold a pointer to that Table, and call BaseTable::handleAliasChanges() when its
 *  aliases are set or removed.
 */
class AliasMap {
public:

    // Create an empty AliasMap
    AliasMap() : _internal(), _table(0) {}

    /**
     *  Deep-copy an AliasMap
     *
     *  The new AliasMap will not be linked to any tables, even if other is.
     */
    AliasMap(AliasMap const & other) : _internal(other._internal), _table(0) {}

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

private:

    friend class Schema;
    friend class SubSchema;
    friend class BaseTable;

    // Internal in-place implementation of apply()
    void _apply(std::string & name) const;

    Internal _internal;

    // Table to notify of any changes.  We can't use a shared_ptr here because the Table needs to set
    // this in its own constructor, but the Table does guarantee that this pointer is either valid or
    // null.
    BaseTable * _table;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_AliasMap_h_INCLUDED
