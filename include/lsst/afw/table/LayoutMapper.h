// -*- lsst-c++ -*-
#ifndef AFW_TABLE_LayoutMapper_h_INCLUDED
#define AFW_TABLE_LayoutMapper_h_INCLUDED

#include "lsst/afw/table/detail/LayoutMapperData.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class RecordBase;

} // namespace detail

class LayoutMapper {
public:

    Layout const getInputLayout() const { return _data->_input; }

    Layout const getOutputLayout() const { return _data->_output; }

    /// @brief Add a new field to the output Layout that is not connected to the input Layout.
    template <typename T>
    Key<T> addOutputField(Field<T> const & newField) {
        _edit();
        return _data->_output.addField(newField);
    }

    /**
     *  @brief Add a new field to the output Layout that is a copy of a field in the input Layout.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field in the output Layout will be reset to a copy of the input Field.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const & inputKey);

    /**
     *  @brief Add a new mapped field to the output Layout with a new name and/or description.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field will be replaced with the given one.
     */
    template <typename T>
    Key<T> addMapping(Key<T> const & inputKey, Field<T> const & outputField);

    /**
     *  @brief Add mappnigs for all fields that match criteria defined by a predicate.
     *
     *  A mapping in the output Layout will be created for each LayoutItem 'i' in the input Layout
     *  such that 'predicate(i)' is true.  Note that the predicate must have a templated
     *  and/or sufficiently overloaded operator() to match all supported field types,
     *  not just those present in the input Layout.
     */
    template <typename Predicate>
    void addMappingsWhere(Predicate predicate);

    /// @brief Swap the input and output layouts in-place.
    void invert();

    /// @brief Return true if the given input Key is mapped to an output Key.
    template <typename T>
    bool isMapped(Key<T> const & inputKey) const;

    /// @brief Return the output Key corresponding to the given input Key, or raise NotFoundException.
    template <typename T>
    Key<T> getMapping(Key<T> const & inputKey) const;

    /**
     *  @brief Copy values from one record according to the mapping.
     *
     *  IDs, parent/child relationships and auxiliary data are not copied.
     *
     *  @Note the fact that the output record is passed by const reference is weird but intentional;
     *  see the documentation for RecordBase for more information.
     */
    void copyRecord(detail::RecordBase const & input, detail::RecordBase const & output) const;

    template <typename F>
    void forEach(F func) const {
        Data::VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(_data->_map.begin(), _data->_map.end(), visitor);
    }

    /// @brief Construct a mapper from the given input Layout.  
    explicit LayoutMapper(Layout const & input);

private:

    /// @brief Copy on write; should be called by all mutators.
    void _edit();

    template <typename Predicate>
    struct AddMappingsWhere {

        template <typename T>
        void operator()(LayoutItem<T> const & item) const {
            if (predicate(item)) mapper->addMapping(item.key);
        }

        AddMappingsWhere(LayoutMapper * mapper_, Predicate predicate_) :
            mapper(mapper_), predicate(predicate_) {}

        LayoutMapper * mapper;
        Predicate predicate;
    };

    typedef detail::LayoutMapperData Data;

    boost::shared_ptr<Data> _data;
};

template <typename Predicate>
void LayoutMapper::addMappingsWhere(Predicate predicate) {
    _data->_input.forEach(AddMappingsWhere<Predicate>(this, predicate));
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_LayoutMapper_h_INCLUDED
