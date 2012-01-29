%{
#include "lsst/afw/table/Vector.h"
%}

namespace lsst { namespace afw { namespace table {

template <typename RecordT, typename TableT>
class VectorT {
public:

    PTR(TableT) getTable() const;

    Schema getSchema() const;

    explicit VectorT(PTR(TableT) const & table = PTR(TableT)());

    explicit VectorT(Schema const & table);

    VectorT(VectorT const & other);

    void writeFits(std::string const & filename) const;

    static VectorT readFits(std::string const & filename);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();

};

%extend VectorT {
    std::size_t __len__() const { return self->size(); }
    PTR(RecordT) __getitem__(std::ptrdiff_t i) const {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Vector index %d out of range.") % i).str()
            );
        }
        return self->get(i);
    }
    void __setitem__(std::ptrdiff_t i, PTR(RecordT) const & p) {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Vector index %d out of range.") % i).str()
            );
        }
        self->set(i, p);
    }
    void __delitem__(std::ptrdiff_t i) {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Vector index %d out of range.") % i).str()
            );
        }
        self->erase(self->begin() + i);
    }
    void append(PTR(RecordT) const & p) { self->push_back(p); }
    void insert(std::ptrdiff_t i, PTR(RecordT) const & p) {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) > self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Vector index %d out of range.") % i).str()
            );
        }
        self->insert(self->begin() + i, p);
    }
    %pythoncode %{
    def extend(self, iterable):
        for e in iterable:
            self.append(e)
    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]
    table = property(getTable)
    schema = property(getSchema)
    %}
}

%template (BaseVector) VectorT<BaseRecord,BaseTable>;
%template (SourceVector) VectorT<SourceRecord,SourceTable>;

}}} // namespace lsst::afw::table
