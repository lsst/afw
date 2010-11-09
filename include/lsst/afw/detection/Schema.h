// -*- lsst-c++ -*-
#if !defined(LSST_AFW_DETECTION_SCHEMA_H)
#define LSST_AFW_DETECTION_SCHEMA_H 1
#include <iostream>                     // XXXX

#include <numeric>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#include "boost/any.hpp"
#include "boost/shared_ptr.hpp"

namespace lsst { namespace afw { namespace detection {
/**
 * Describe the schema of what we're measuring
 *
 * This class uses a simplified Go4 Composite to handle both individual items (e.g. flux:DOUBLE:0 -- the
 * SchemaEntry subclass) and collections of Schema or SchemaEntry objects.  In other words, it's general
 * enough to handle
 *   [psf:  [flux:DOUBLE, fluxErr:FLOAT],
 *    aper: [flux:DOUBLE, fluxErr:FLOAT, radius:INT], ...]
 * and also more complex situations where the elements of the list are themselves lists.  However, the
 * iterator interface does not (yet?) support iterating over all nodes recursively.
 *
 * The names (psf, aper, ...) are known as "component" names
 */
class Schema {
public:
    typedef boost::shared_ptr<Schema> Ptr;
    typedef boost::shared_ptr<const Schema> ConstPtr;
    typedef std::vector<boost::shared_ptr<Schema> >::iterator iterator;
    typedef std::vector<boost::shared_ptr<Schema> >::const_iterator const_iterator;
    /// Supported types
    typedef enum { UNKNOWN, CHAR, SHORT, INT, LONG, FLOAT, DOUBLE } Type;
    /// Write a Type to a stream
    ///
    /// You can use std::ostringstream to convert to a string if needs be
    ///
    /// \note implemented as a friend to keep the definition close to the enum
    friend std::ostream &operator<<(std::ostream &os, Type t) {
        switch (t) {
          case UNKNOWN: return os << "UNKNOWN";
          case CHAR:    return os << "CHAR";
          case SHORT:   return os << "SHORT";
          case INT:     return os << "INT";
          case LONG:    return os << "LONG";
          case FLOAT:   return os << "FLOAT";
          case DOUBLE:  return os << "DOUBLE";
        }

        std::ostringstream msg;
        msg << "Unknown Schema::Type " << int(t);
        throw std::runtime_error(msg.str());
    }

    Schema(std::string const& name="", int index=0,
           Schema::Type const& type=UNKNOWN, int dimen=1, std::string const& units="") :
        _name(name), _index(index), _type(type), _dimen(dimen), _units(units), _component(), _entries() {}
    virtual ~Schema() { }
    /// Clone a Schema
    Ptr clone() const { return Ptr(_clone()); }

    /// Return an iterator to the start of the set of Schema
    iterator begin() {
        return _entries.begin();
    }
    /// Return a const iterator to the start of the set of Schema
    const_iterator begin() const {
        return _entries.begin();
    }
    /// Return an iterator to the end of the set of Schema
    iterator end() {
        return _entries.end();
    }
    /// Return a const iterator to the end of the set of Schema
    const_iterator end() const {
        return _entries.end();
    }
    /// Set the name of a component
    void setComponent(std::string const& component) { _component = component; }
    /// Retrieve a component's name
    std::string const& getComponent() const { return _component; }
    /// Add a Schema to the list of components
    void add(Schema const& val) {
        add(val.clone());
    }
    /// Reset the schema to be empty
    void clear() {
        _entries.clear();
    }
    /// Add a Schema::Ptr to the list of components
    void add(Ptr val) {
        _entries.push_back(val);
    }

    inline virtual int size() const;
    /// Is this an array?
    int isArray() const { return (_dimen > 1); }
    /// Return the name if a leaf node
    std::string const& getName() const { return _name; }
    /// Return the index if a leaf node (\sa SchemaEntry)
    unsigned int getIndex() const { return _index; }
    /// Return the Type if a leaf node (\sa SchemaEntry)
    Schema::Type const& getType() const { return _type; }
    /// Return the dimension if a leaf node
    int getDimen() const { return _dimen; }
    /// Return the units if a leaf node
    std::string const& getUnits() const { return _units; }
    /// Are there any components?
    virtual operator bool() const {
        return !_entries.empty();
    }

    static Schema const& unknown();

    virtual Schema const& find(std::string const& name, std::string const& component="") const;

    virtual std::ostream &output(std::ostream &os) const;
private:
    // used if this is a leaf node
    std::string const _name;
    unsigned int const _index;
    Schema::Type const _type;
    int const _dimen;
    std::string const _units;
    // used if this is a composite
    std::string _component;
    std::vector<Schema::Ptr> _entries;

    virtual Schema *_clone() const { return new Schema(*this); }
};

namespace {
    int findSize(int n, Schema::ConstPtr s) {
        return n + s->size();
    }
}

/// Return the number of slots needed to hold our data
int Schema::size() const {
    return std::accumulate(_entries.begin(), _entries.end(), 0, findSize);
}

/**
 * An entry in a Schema (e.g. the flux)
 *
 * There are three things known about each entry; they're in the base class Schema to avoid the need
 * to either dynamic_cast to SchemaEntry& or make the getter functions (getName etc.) virtual.
 *
 * Three things:
 *   1/ The name
 *   2/ The type
 *   3/ The index into a data structure that actually holds the data.
 * If we wanted to make this more flexible we'd have to reconsider the design, but this is good enough for now
 */
class SchemaEntry : public Schema {
public:
    typedef boost::shared_ptr<SchemaEntry> Ptr;

    SchemaEntry(std::string const& name, int index, Schema::Type const& type,
                int dimen=1, std::string const& units="") :
        Schema(name, index, type, dimen, units) {}
    /// Is this a real entry?
    virtual operator bool() const {
        return getType() != UNKNOWN;
    }
    /// Does this SchemaEntry match? If so, return it.
    ///
    /// Note that this is overloaded;  Schema::find does more of a search, calling this on leaf nodes
    virtual Schema const& find(std::string const& name, std::string const&) const {
        return (name == getName()) ? *this : Schema::unknown();
    }

    /// Return the number of slots needed
    virtual int size() const { return getDimen(); }
    
    /// Virtual function called by operator<< to dynamically dispatch the type to a stream
    virtual std::ostream &output(std::ostream &os) const {
        os << getName() << ":" << getType();
        if (getUnits() != "") {
            os << " : " << getUnits();
        }
        return os;
    }
private:
    virtual SchemaEntry *_clone() const { return new SchemaEntry(*this); }
};

}}}
#endif
