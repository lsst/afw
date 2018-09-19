// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_SchemaImpl_h_INCLUDED
#define AFW_TABLE_DETAIL_SchemaImpl_h_INCLUDED

#include <vector>
#include <algorithm>
#include <map>
#include <set>

#include "boost/variant.hpp"
#include "boost/mpl/transform.hpp"

#include "lsst/daf/base/Citizen.h"

namespace lsst {
namespace afw {
namespace table {

class Schema;
class SubSchema;

/**
 *  @brief A simple pair-like struct for mapping a Field (name and description) with a Key
 *         (used for actual data access).
 */
template <typename T>
struct SchemaItem final {
    Key<T> key;
    Field<T> field;

    SchemaItem(Key<T> const& key_, Field<T> const& field_) : key(key_), field(field_) {}
};

namespace detail {

class Access;

/**
 *  A private implementation class to hide the messy details of Schema.
 *
 *  This can't be a real pimpl class, because some of the most important functionality
 *  is in the forEach function, a templated function we can't explicitly instantiate
 *  in a source file.  But putting all the details here draws a clear line between what
 *  users should look at (Schema) and what they shouldn't (this).
 *
 *  Because Schema holds SchemaImpl by shared pointer, one SchemaImpl can be shared between
 *  multiple Schemas (and SubSchemas), which implement copy-on-write by creating a new SchemaImpl
 *  if the pointer they have isn't unique when they are modified.
 *
 *  SchemaImpl inherits from Citizen; this allows both Schemas and SubSchemas to be tracked in
 *  a more meaningful way than if we derived either of those from Citizen.
 */
class SchemaImpl : public daf::base::Citizen {
private:
    /// Boost.MPL metafunction that returns a SchemaItem<T> given a T.
    struct MakeItem {
        template <typename T>
        struct apply {
            typedef SchemaItem<T> type;
        };
    };

public:
    static int const VERSION = 3;

    /// An MPL sequence of all the allowed SchemaItem templates.
    typedef boost::mpl::transform<FieldTypes, MakeItem>::type ItemTypes;
    /// A Boost.Variant type that can hold any one of the allowed SchemaItem types.
    typedef boost::make_variant_over<ItemTypes>::type ItemVariant;
    /// A std::vector whose elements can be any of the allowed SchemaItem types.
    typedef std::vector<ItemVariant> ItemContainer;
    /// A map from field names to position in the vector, so we can do name lookups.
    typedef std::map<std::string, int> NameMap;
    /// A map from standard field offsets to position in the vector, so we can do field lookups.
    typedef std::map<int, int> OffsetMap;
    /// A map from Flag field offset/bit pairs to position in the vector, so we can do Flag field lookups.
    typedef std::map<std::pair<int, int>, int> FlagMap;

    /// The size of a record in bytes.
    int getRecordSize() const { return _recordSize; }

    /// The total number of fields.
    int getFieldCount() const { return _names.size(); }

    /// The number of Flag fields.
    int getFlagFieldCount() const { return _flags.size(); }

    /// The number of non-Flag fields.
    int getNonFlagFieldCount() const { return _offsets.size(); }

    /// Find an item by name (used to implement Schema::find).
    template <typename T>
    SchemaItem<T> find(std::string const& name) const;

    /// Find an item by key (used to implement Schema::find).
    template <typename T>
    SchemaItem<T> find(Key<T> const& key) const;

    /// Find an item by key (used to implement Schema::find).
    SchemaItem<Flag> find(Key<Flag> const& key) const;

    /// Find an item by name and run the given functor on it.
    template <typename F>
    void findAndApply(std::string const& name, F&& func) const {
        auto iter = _names.find(name);
        if (iter == _names.end()) {
            throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                              (boost::format("Field with name '%s' not found") % name).str());
        }
        VisitorWrapper<F> visitor(std::forward<F>(func));
        visitor(_items[iter->second]);
    }

    /// Return a set of field names (used to implement Schema::getNames).
    std::set<std::string> getNames(bool topOnly) const;

    /// Return a set of field names (used to implement SubSchema::getNames).
    std::set<std::string> getNames(bool topOnly, std::string const& prefix) const;

    template <typename T>
    int contains(SchemaItem<T> const& item, int flags) const;

    /// Add a field to the schema (used to implement Schema::addField).
    template <typename T>
    Key<T> addField(Field<T> const& field, bool doReplace = false);

    /// Add a field to the schema (used to implement Schema::addField).
    Key<Flag> addField(Field<Flag> const& field, bool doReplace = false);

    /// Add a field to the schema (used to implement Schema::addField).
    template <typename T>
    Key<Array<T> > addField(Field<Array<T> > const& field, bool doReplace = false);

    /// Add a field to the schema (used to implement Schema::addField).
    Key<std::string> addField(Field<std::string> const& field, bool doReplace = false);

    /// Replace the Field in an existing SchemaItem without changing the Key.
    template <typename T>
    void replaceField(Key<T> const& key, Field<T> const& field);

    /**
     *  Return the vector of SchemaItem variants.
     *
     *  Fields are in the order they are added.  That means they're also ordered with increasing
     *  Key offsets, except for Flag fields, which are in increasing order of (offset, bit) relative
     *  to each other, but not relative to all the other fields.
     */
    ItemContainer const& getItems() const { return _items; }

    /// Default constructor.
    explicit SchemaImpl()
            : daf::base::Citizen(typeid(this)),
              _recordSize(0),
              _lastFlagField(-1),
              _lastFlagBit(-1),
              _items() {}

    /**
     *  A functor-wrapper used in the implementation of Schema::forEach.
     *
     *  Visitor functors used with Boost.Variant (see the Boost.Variant docs)
     *  must inherit from boost::static_visitor<> to declare their return type
     *  (void, in this case).  By wrapping user-supplied functors with this class,
     *  we can hide the fact that we've implemented SchemaImpl using Boost.Variant
     *  (because they won't need to inherit from static_visitor themselves.
     */
    template <typename F>
    struct VisitorWrapper : public boost::static_visitor<> {
        /// Call the wrapped function.
        template <typename T>
        void operator()(SchemaItem<T> const& x) const {
            _func(x);
        };

        /**
         *  Invoke the visitation.
         *
         *  The call to boost::apply_visitor will call the appropriate template of operator().
         *
         *  This overload allows a VisitorWrapper to be applied directly on a variant object
         *  with function-call syntax, allowing us to use it on our vector of variants with
         *  std::for_each and other STL algorithms.
         */
        void operator()(ItemVariant const& v) const { boost::apply_visitor(*this, v); }

        /// Construct the wrapper.
        template <typename T>
        explicit VisitorWrapper(T&& func) : _func(std::forward<T>(func)) {}

    private:
        F _func;
    };

private:
    friend class detail::Access;

    template <typename T>
    Key<T> addFieldImpl(int elementSize, int elementCount, Field<T> const& field, bool doReplace);

    int _recordSize;       // Size of a record in bytes.
    int _lastFlagField;    // Offset of the last flag field in bytes.
    int _lastFlagBit;      // Bit of the last flag field.
    ItemContainer _items;  // Vector of variants of SchemaItem<T>.
    NameMap _names;        // Field name to vector-index map.
    OffsetMap _offsets;    // Offset to vector-index map for regular fields.
    FlagMap _flags;        // Offset to vector-index map for flags.
};
}  // namespace detail
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_DETAIL_SchemaImpl_h_INCLUDED
