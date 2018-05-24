// -*- lsst-c++ -*-
#ifndef AFW_TABLE_SchemaMapper_h_INCLUDED
#define AFW_TABLE_SchemaMapper_h_INCLUDED

#include <memory>

#include "lsst/afw/table/detail/SchemaMapperImpl.h"

namespace lsst {
namespace afw {
namespace table {

class BaseRecord;

/**
 *  A mapping between the keys of two Schemas, used to copy data between them.
 *
 *  SchemaMapper is initialized with its input Schema, and contains member functions
 *  to add mapped or unmapped fields to the output Schema.
 */
class SchemaMapper {
public:
    /// Return the input schema (copy-on-write).
    Schema const getInputSchema() const { return _impl->_input; }

    /// Return the output schema (copy-on-write).
    Schema const getOutputSchema() const { return _impl->_output; }

    /// Return a reference to the output schema that allows it to be modified in place.
    Schema& editOutputSchema() { return _impl->_output; }

    /// Add a new field to the output Schema that is not connected to the input Schema.
    template <typename T>
    Key<T> addOutputField(Field<T> const& newField, bool doReplace = false) {
        return _impl->_output.addField(newField, doReplace);
    }

    /**
     *  Add a new field to the output Schema that is a copy of a field in the input Schema.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field in the output Schema will be reset to a copy of the input Field.
     *
     *  If doReplace=True and a field with same name already exists in the output schema, that
     *  field will be mapped instead of adding a new field to the output schema.  If doReplace=false
     *  and a name conflict occurs, an exception will be thrown.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const& inputKey, bool doReplace = false);

    /**
     *  Add a new mapped field to the output Schema with new descriptions.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field will be replaced with the given one.
     *
     *  If doReplace=True and a field with same name already exists in the output schema, that
     *  field will be mapped instead of adding a new field to the output schema.  If doReplace=false
     *  and a name conflict occurs, an exception will be thrown.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const& inputKey, Field<T> const& outputField, bool doReplace = false);

    /**
     *  Add a new mapped field to the output Schema with a new name.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field will be replaced with one with the given name.
     *
     *  If doReplace=True and a field with same name already exists in the output schema, that
     *  field will be mapped instead of adding a new field to the output schema.  If doReplace=false
     *  and a name conflict occurs, an exception will be thrown.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const& inputKey, std::string const& outputName, bool doReplace = true);

    /**
     * This deliberately deleted overload ensures we don't accidentally cast string literals to bool.
     *
     * See DM-13787 for more information.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const& inputKey, char const* outputName, bool doReplace = true) = delete;

    /**
     *  Add mappings for all fields that match criteria defined by a predicate.
     *
     *  A mapping in the output Schema will be created for each SchemaItem 'i' in the input Schema
     *  such that 'predicate(i)' is true.  Note that the predicate must have a templated
     *  and/or sufficiently overloaded operator() to match all supported field types,
     *  not just those present in the input Schema.
     *
     *  If doReplace=True and a field with same name already exists in the output schema, that
     *  field will be mapped instead of adding a new field to the output schema.  If doReplace=false
     *  and a name conflict occurs, an exception will be thrown.
     */
    template <typename Predicate>
    void addMappingsWhere(Predicate predicate, bool doReplace = true);

    /**
     *  Add the given minimal schema to the output schema.
     *
     *  This is intended to be used to ensure the output schema starts with some minimal schema.
     *  It must be called before any other fields are added to the output schema.
     *
     *  @param[in] minimal     Minimal schema to be added to the beginning of the output schema.
     *  @param[in] doMap       Whether to map minimal schema fields that are also present
     *                         in the input schema.
     */
    void addMinimalSchema(Schema const& minimal, bool doMap = true);

    /**
     *  Create a mapper by removing fields from the front of a schema.
     *
     *  The returned mapper maps all fields in the input schema to all fields that are not
     *  in the minimal schema (compared by keys, so the overlap must appear at the beginning
     *  of the input schema).
     *
     *  @param[in] input       Input schema for the mapper.
     *  @param[in] minimal     The minimal schema the input schema starts with.
     */
    static SchemaMapper removeMinimalSchema(Schema const& input, Schema const& minimal);

    /// Swap the input and output schemas in-place.
    void invert();

    /// Return true if the given input Key is mapped to an output Key.
    template <typename T>
    bool isMapped(Key<T> const& inputKey) const;

    /// Return the output Key corresponding to the given input Key, or raise NotFoundError.
    template <typename T>
    Key<T> getMapping(Key<T> const& inputKey) const;

    /**
     *  Call the given functor for each key pair in the mapper.
     *
     *  Function objects should have a template and/or overloaded operator()
     *  that takes two Key objects with the same type:
     *
     *      struct Functor {
     *          template <typename T>
     *          void operator()(Key<T> const & input, Key<T> const & output) const;
     *      };
     *
     *  The order of iteration is the same as the order in which mappings were added.
     */
    template <typename F>
    void forEach(F&& func) const {
        Impl::VisitorWrapper<F> visitor(std::forward<F>(func));
        std::for_each(_impl->_map.begin(), _impl->_map.end(), visitor);
    }

    /// Construct an empty mapper; useless unless you assign a fully-constructed one to it.
    explicit SchemaMapper();

    /**
     *  Construct a mapper from the given input Schema and initial output Schema
     *
     *  @param[in] input    The Schema that fields will be mapped from.
     *  @param[in] output   The starting point for the Schema that fields will be mapped to (no
     *                      mappings will be created automaticaly).  Use addMapping() with
     *                      doReplace=true to connect input fields to preexisting fields in
     *                      the output schema.
     *
     *  Note that the addMapping() methods will not connect input schema fields to existing
     *  output schema fields unless doReplace=true; instead, these will by default append
     *  new fields to the output schema.  So most often you'll want to start with an empty
     *  output schema and construct it as fields are mapped from the input schema, or be sure
     *  to always pass doReplace=true to addMapping.
     */
    explicit SchemaMapper(Schema const& input, Schema const& output);

    /**
     *  Construct a mapper from the given input Schema
     *
     *  @param[in] input         The Schema that fields will be mapped from.
     *  @param[in] shareAliasMap If true, install the input Schema's AliasMap in the
     *                           output Schema.
     *
     *  Note that the addMapping() methods will not connect input schema fields to existing
     *  output schema fields unless doReplace=true; instead, these will by default append
     *  new fields to the output schema.  So most often you'll want to start with an empty
     *  output schema and construct it as fields are mapped from the input schema, or be sure
     *  to always pass doReplace=true to addMapping.
     *
     *  The initial (empty) output schema will have the same version as the input schema, and
     *  they will share the same AliasMap (use editOutputSchema().disconnectAliases() to
     *  use a copy of the AliasMap).
     */
    explicit SchemaMapper(Schema const& input, bool shareAliasMap = false);

    /// Copy construct (copy-on-write).
    SchemaMapper(SchemaMapper const& other);
    SchemaMapper(SchemaMapper&& other);

    /// Assignment (copy-on-write).
    SchemaMapper& operator=(SchemaMapper const& other);
    SchemaMapper& operator=(SchemaMapper&& other);

    ~SchemaMapper();

    /**
     *  Combine a sequence of schemas into one, creating a SchemaMapper for each.
     *
     *  @param[in]  inputs    A vector of input schemas to merge.
     *  @param[in]  prefixes  An optional vector of prefixes for the output field names,
     *                        either empty or of the same size as the inputs vector.
     *
     *  Each of the returned SchemaMappers has the same output schema.
     */
    static std::vector<SchemaMapper> join(
            std::vector<Schema> const& inputs,
            std::vector<std::string> const& prefixes = std::vector<std::string>());

private:
    template <typename Predicate>
    struct AddMappingsWhere {
        template <typename T>
        void operator()(SchemaItem<T> const& item) const {
            if (predicate(item)) mapper->addMapping(item.key, doReplace);
        }

        AddMappingsWhere(SchemaMapper* mapper_, Predicate predicate_, bool doReplace_)
                : mapper(mapper_), predicate(predicate_), doReplace(doReplace_) {}

        SchemaMapper* mapper;
        Predicate predicate;
        bool doReplace;
    };

    typedef detail::SchemaMapperImpl Impl;

    std::unique_ptr<Impl> _impl;
};

template <typename Predicate>
void SchemaMapper::addMappingsWhere(Predicate predicate, bool doReplace) {
    _impl->_input.forEach(AddMappingsWhere<Predicate>(this, predicate, doReplace));
}
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_SchemaMapper_h_INCLUDED
