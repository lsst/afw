// -*- lsst-c++ -*-
#ifndef AFW_TABLE_SchemaMapper_h_INCLUDED
#define AFW_TABLE_SchemaMapper_h_INCLUDED

#include "lsst/afw/table/detail/SchemaMapperImpl.h"

namespace lsst { namespace afw { namespace table {

class BaseRecord;

/**
 *  @brief A mapping between the keys of two Schemas, used to copy data between them
 *         or read/write only certain fields during serialization.
 *
 *  SchemaMapper is initialized with its input Schema, and contains member functions
 *  to add mapped or unmapped fields to the output Schema.
 */
class SchemaMapper {
public:
    
    /// @brief Return the input schema (copy-on-write).
    Schema const getInputSchema() const { return _impl->_input; }

    /// @brief Return the output schema (copy-on-write).
    Schema const getOutputSchema() const { return _impl->_output; }

    /// @brief Add a new field to the output Schema that is not connected to the input Schema.
    template <typename T>
    Key<T> addOutputField(Field<T> const & newField) {
        _edit();
        return _impl->_output.addField(newField);
    }

    /**
     *  @brief Add a new field to the output Schema that is a copy of a field in the input Schema.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field in the output Schema will be reset to a copy of the input Field.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const & inputKey);

    /**
     *  @brief Add a new mapped field to the output Schema with a new name and/or description.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field will be replaced with the given one.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const & inputKey, Field<T> const & outputField);

    /**
     *  @brief Add mappings for all fields that match criteria defined by a predicate.
     *
     *  A mapping in the output Schema will be created for each SchemaItem 'i' in the input Schema
     *  such that 'predicate(i)' is true.  Note that the predicate must have a templated
     *  and/or sufficiently overloaded operator() to match all supported field types,
     *  not just those present in the input Schema.
     */
    template <typename Predicate>
    void addMappingsWhere(Predicate predicate);

    /// @brief Swap the input and output schemas in-place.
    void invert();

    /// @brief Return true if the given input Key is mapped to an output Key.
    template <typename T>
    bool isMapped(Key<T> const & inputKey) const;

    /// @brief Return the output Key corresponding to the given input Key, or raise NotFoundException.
    template <typename T>
    Key<T> getMapping(Key<T> const & inputKey) const;

    /**
     *  @brief Call the given functor for each key pair in the mapper.
     *
     *  Function objects should have a template and/or overloaded operator()
     *  that takes two Key objects with the same type:
     *  @code
     *  struct Functor {
     *      template <typename T>
     *      void operator()(Key<T> const & input, Key<T> const & output) const;
     *  };
     *  @endcode
     *
     *  The order of iteration is the same as the order in which mappings were added.
     */
    template <typename F>
    void forEach(F func) const {
        Impl::VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(_impl->_map.begin(), _impl->_map.end(), visitor);
    }

    /// @brief Construct a mapper from the given input Schema.  
    explicit SchemaMapper(Schema const & input);

private:

    /// @brief Copy on write; should be called by all mutators.
    void _edit();

    template <typename Predicate>
    struct AddMappingsWhere {

        template <typename T>
        void operator()(SchemaItem<T> const & item) const {
            if (predicate(item)) mapper->addMapping(item.key);
        }

        AddMappingsWhere(SchemaMapper * mapper_, Predicate predicate_) :
            mapper(mapper_), predicate(predicate_) {}

        SchemaMapper * mapper;
        Predicate predicate;
    };

    typedef detail::SchemaMapperImpl Impl;

    boost::shared_ptr<Impl> _impl;
};

template <typename Predicate>
void SchemaMapper::addMappingsWhere(Predicate predicate) {
    _impl->_input.forEach(AddMappingsWhere<Predicate>(this, predicate));
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SchemaMapper_h_INCLUDED
