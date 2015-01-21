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
 *  Aliases need not be complete, but they must match to the beginning of a field name to be useful.
 *  For example, if "a_b_c" is a true field name, "x_->a_b" is a valid alias that will cause
 *  "x_y_c" to map to "a_b_c", but "y_z->b_c" will not cause "a_y_z" to be matched.
 *
 *  Aliases are not checked to see if they match any existing fields, and if an alias has the same
 *  name as a field name, it will take precedence and hide the true field.
 *
 *  Unlike the other components of a Schema, aliases can be modified and removed, even after a Table
 *  has been constructed from the Schema.
 *
 *  AliasMaps are shared when Schemas are copy-constructed, but can be separated manually
 *  by calling Schema::disconnectAliases() or Schema::setAliasMap().  In addition, the AliasMap
 *  is deep-copied when used to construct a Table (or Catalog).
 *
 *  In order to allow Tables to react to changes in aliases (which may be used to define cached Keys
 *  held by the table, as in SourceTable's "slots" mechanism), an AliasMap that is part of a Schema held
 *  by a Table will hold a pointer to that Table, and call BaseTable::handleAliasChanges() when its
 *  aliases are set or removed.
 */
class AliasMap {
    typedef std::map<std::string,std::string> Internal;
public:

    // Create an empty AliasMap
    AliasMap() : _internal(), _table(0) {}

    /**
     *  Deep-copy an AliasMap
     *
     *  The new AliasMap will not be linked to any tables, even if other is.
     */
    AliasMap(AliasMap const & other) : _internal(other._internal), _table(0) {}

    /// An iterator over alias->target pairs.
    typedef std::map<std::string,std::string>::const_iterator Iterator;

    /// Return a iterator to the beginning of the map
    Iterator begin() const { return _internal.begin(); }

    /// Return a iterator to one past the end of the map
    Iterator end() const { return _internal.end(); }

    /// Return the number of aliases
    std::size_t size() const { return _internal.size(); }

    /// Return the true if there are no aliases
    bool empty() const { return _internal.empty(); }

    /**
     *  Apply any aliases that match the given field name and return a de-aliased name.
     *
     *  Given a string that starts with any alias in the map, this returns a string
     *  with the part of the string that matches the alias replaced by that alias's
     *  target.  The longest such alias is used.
     *
     *  For example:
     *  @code
     *  m = AliasMap();
     *  m.set("q", "a");
     *  m.set("q1", "b");
     *  assert(m.apply("q3") == "a3");
     *  assert(m.apply("q12") == "b2");
     *  @endcode
     */
    std::string apply(std::string const & name) const;

    /**
     *  @brief Return the target of the given alias
     *
     *  Unlike apply(), this will not return partial matches.
     *
     *  @throw pex::exceptions::NotFoundError if no alias with the given name exists
     */
    std::string get(std::string const & alias) const;

    /// Add an alias to the schema or replace an existing one.
    void set(std::string const & alias, std::string const & target);

    /**
     *  @brief Remove an alias from the schema if it is present.
     *
     *  @return True if an alias was erased, and false if no such alias was found.
     */
    bool erase(std::string const & alias);

    //@{
    /**
     *  Equality comparison
     */
    bool operator==(AliasMap const & other) const;
    bool operator!=(AliasMap const & other) const { return !(other == *this); }
    //@}

    /// Return true if all aliases in this are also in other (with the same targets).
    bool contains(AliasMap const & other) const;

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
