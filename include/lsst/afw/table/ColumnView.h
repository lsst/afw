// -*- lsst-c++ -*-
#ifndef AFW_TABLE_ColumnView_h_INCLUDED
#define AFW_TABLE_ColumnView_h_INCLUDED

#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

/// Functor to compute a flag bit, used to create an ndarray expression template.
struct FlagBitExtractor {
    typedef Field<Flag>::Element argument_type;
    typedef bool result_type;

    result_type operator()(argument_type element) const { return element & _mask; }

    explicit FlagBitExtractor(Key<Flag> const & key) : _mask(argument_type(1) << key.getBit()) {}

private:
    argument_type _mask;
};

} // namespace detail

class TableBase;

/**
 *  @brief Column-wise view into a consolidated table.
 *
 *  A ColumnView can be constructed from a table using TableBase::getColumnView().
 *
 *  Geometric (point and shape) fields cannot be accessed through a ColumnView, but their
 *  scalar components can be.
 */
class ColumnView {
public:

    /// @brief Return the schema that defines the fields.
    Schema const getSchema() const;

    /// @brief Return a 1-d array corresponding to a scalar field (or subfield).
    template <typename T>
    ndarray::Array<T const,1> operator[](Key<T> const & key) const;

    /// @brief Return a 2-d array corresponding to an array field.
    template <typename T>
    ndarray::Array<T const,2,1> operator[](Key< Array<T> > const & key) const;

    /**
     *  @brief Return a 1-d array expression corresponding to a flag bit.
     *
     *  In C++, the return value is a lazy ndarray expression template that performs the bitwise
     *  AND on every element when that element is requested.  In Python,
     *  the result will be copied into a bool NumPy array.
     */
    ndarray::result_of::vectorize< detail::FlagBitExtractor,
                                   ndarray::Array< Field<Flag>::Element const,1> >::type
    operator[](Key<Flag> const & key) const;

    template <typename InputIterator>
    static ColumnView make(InputIterator first, InputIterator last);

    ~ColumnView();

private:

    friend class TableBase;

    struct Impl;

    ColumnView(Schema const & schema, int recordCount, void * buf, ndarray::Manager::Ptr const & manager);

    boost::shared_ptr<Impl> _impl;
};

template <typename InputIterator>
ColumnView ColumnView::make(InputIterator first, InputIterator last) {
    if (first == last) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Cannot create zero-length ColumnView."
        );
    }
    Schema schema = first->getSchema();
    std::size_t recordSize = schema.getRecordSize();
    std::size_t recordCount = 1;
    void * buf = first->_data;
    ndarray::Manager::Ptr manager = first->_manager;
    char * expected = reinterpret_cast<char*>(buf) + recordSize;
    for (++first; first != last; ++first, ++recordCount, expected += recordSize) {
        if (first->_data != expected || first->_manager != manager || first->getSchema() != schema) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Record data is not contiguous in memory."
            );
        }
    }
    return ColumnView(schema, recordCount, buf, manager);
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_ColumnView_h_INCLUDED
