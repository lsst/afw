// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_SchemaImpl_h_INCLUDED
#define AFW_TABLE_DETAIL_SchemaImpl_h_INCLUDED

#include <vector>
#include <algorithm>
#include <map>
#include <variant>

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
 */
class SchemaImpl {
private:
    /// Type metafunction that returns a std::variant of SchemaItem given a
    /// list of the types to parameterize SchemaItem with.
    template <typename ...E>
    static std::variant<SchemaItem<E>...> makeItemVariantType(TypeList<E...>);
public:
    static int const VERSION = 3;

    /// A Boost.Variant type that can hold any one of the allowed SchemaItem types.
    using ItemVariant = decltype(makeItemVariantType(FieldTypes{}));
    /// A std::vector whose elements can be any of the allowed SchemaItem types.
    using ItemContainer = std::vector<ItemVariant>;
    /// A map from field names to position in the vector, so we can do name lookups.
    using NameMap = std::map<std::string, std::size_t>;
    /// A map from standard field offsets to position in the vector, so we can do field lookups.
    using OffsetMap = std::map<std::size_t, std::size_t>;
    /// A map from Flag field offset/bit pairs to position in the vector, so we can do Flag field lookups.
    using FlagMap = std::map<std::pair<std::size_t, std::size_t>, std::size_t>;

    /// The size of a record in bytes.
    std::size_t getRecordSize() const { return _recordSize; }

    /// The total number of fields.
    std::size_t getFieldCount() const { return _names.size(); }

    /// The number of Flag fields.
    std::size_t getFlagFieldCount() const { return _flags.size(); }

    /// The number of non-Flag fields.
    std::size_t getNonFlagFieldCount() const { return _offsets.size(); }

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
    decltype(auto) findAndApply(std::string const& name, F&& func) const {
        auto iter = _names.find(name);
        if (iter == _names.end()) {
            throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                              (boost::format("Field with name '%s' not found") % name).str());
        }
        return std::visit(std::forward<F>(func), _items[iter->second]);
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
    SchemaImpl() : _recordSize(0), _lastFlagField(0), _lastFlagBit(0), _items(), _initFlag(false) {}

private:
    friend class detail::Access;

    template <typename T>
    Key<T> addFieldImpl(std::size_t elementSize, std::size_t elementCount, Field<T> const& field, bool doReplace);

    std::size_t _recordSize;       // Size of a record in bytes.
    std::size_t _lastFlagField;    // Offset of the last flag field in bytes.
    std::size_t _lastFlagBit;      // Bit of the last flag field.
    ItemContainer _items;  // Vector of variants of SchemaItem<T>.
    NameMap _names;        // Field name to vector-index map.
    OffsetMap _offsets;    // Offset to vector-index map for regular fields.
    FlagMap _flags;        // Offset to vector-index map for flags.
    bool _initFlag;        // Indicates if record is valid
};
}  // namespace detail
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_DETAIL_SchemaImpl_h_INCLUDED
