%{
#include "lsst/afw/table/Vector.h"
#include "lsst/afw/table/Source.h"
%}

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class VectorT {
public:

    typedef typename RecordT::Table Table;

    PTR(Table) getTable() const;

    Schema getSchema() const;

    explicit VectorT(PTR(Table) const & table = PTR(Table)());

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

template <typename RecordT>
class SourceVectorT<RecordT> : public VectorT<RecordT> {
public:

    typedef typename RecordT::Table Table;

    explicit SourceVectorT(PTR(Table) const & table = PTR(Table)());

    explicit SourceVectorT(Schema const & table);

    SourceVectorT(SourceVectorT const & other);

    bool isSorted() const;
    void sort();
};

// For some reason, SWIG won't extend the template; it's only happy if
// we extend the instantiation (but compare to %extend VectorT, above)
// ...mystifying.  Good thing we only have one instantiation.
%extend SourceVectorT<SourceRecord> {
    PTR(SourceRecord) find(RecordId id) {
        return self->find(id);
    }
}

%template (BaseVector) VectorT<BaseRecord>;
%template (SourceVectorBase) VectorT<SourceRecord>;
%template (SourceVector) SourceVectorT<SourceRecord>;

typedef VectorT<BaseRecord> BaseVector;
typedef SourceVectorT<SourceRecord> SourceVector;

}}} // namespace lsst::afw::table
