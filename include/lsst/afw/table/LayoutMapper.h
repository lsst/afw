// -*- lsst-c++ -*-
#ifndef AFW_TABLE_LayoutMapper_h_INCLUDED
#define AFW_TABLE_LayoutMapper_h_INCLUDED

#include "lsst/afw/table/config.h"

#include <map>

#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"


#include "lsst/afw/table/Layout.h"

namespace lsst { namespace afw { namespace table {

class LayoutMapper {
public:

    Layout const getInputLayout() const { return _input; }

    Layout const getOutputLayout() const { return _output; }

    /// @brief Add a new field to the output Layout that is not connected to the input Layout.
    template <typename T>
    Key<T> add(Field<T> const & newField) { return _output.add(newField); }

    /**
     *  @brief Add a new field to the output Layout that is a copy of a field in the input Layout.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field in the output Layout will be reset to a copy of the input Field.
     */
    template <typename T>
    Key<T> copy(Key<T> const & inputKey);

    /**
     *  @brief Add a new field to the output Layout with a new name and/or description.
     *
     *  If the input Key has already been mapped, the existing output Key will be reused
     *  but the associated Field will be replaced with the given one.
     */
    template <typename T>
    Key<T> copy(Key<T> const & inputKey, Field<T> const & outputField);

    /**
     *  @brief Copy all fields that match criteria defined by a predicate.
     *
     *  A mapping in the output Layout will be created for each LayoutItem i in the input Layout
     *  such that predicate(i) is true.  Note that the predicate must have a templated
     *  and/or sufficiently overloaded operator() to match all supported field types,
     *  not just those present in the input Layout.
     */
    template <typename Predicate>
    void copyIf(Predicate predicate);

    /// @brief Swap the input and output layouts in-place.
    void invert();

    /// @brief Return true if the given input Key is mapped to an output Key.
    template <typename T>
    bool isMapped(Key<T> const & inputKey) const;

    /// @brief Return the output Key corresponding to the given input Key, or raise NotFoundException.
    template <typename T>
    Key<T> getMapping(Key<T> const & inputKey) const;

    /// @brief Construct a mapper from the given input Layout.  
    explicit LayoutMapper(Layout const & input) : _input(input) {}

private:

    struct MakeMapperPair {
        template <typename T> struct apply {
            typedef boost::fusion::pair< T, std::map< Key<T>, Key<T> > > type;
        };
    };

    typedef boost::fusion::result_of::as_map<
        boost::mpl::transform< detail::FieldTypes, MakeMapperPair >::type
        >::type MapContainer;

    template <typename Predicate>
    struct CopyIf {

        template <typename T>
        void operator()(LayoutItem<T> const & item) const {
            if (predicate(item)) mapper->copy(item.key);
        }

        CopyIf(LayoutMapper * mapper_, Predicate predicate_) :
            mapper(mapper_), predicate(predicate_) {}

        LayoutMapper * mapper;
        Predicate predicate;
    };

    Layout _input;
    Layout _output;
    MapContainer _maps;
};

template <typename Predicate>
void LayoutMapper::copyIf(Predicate predicate) {
    _input.forEach(CopyIf<Predicate>(this, predicate));
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_LayoutMapper_h_INCLUDED
