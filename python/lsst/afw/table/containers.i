%{
#include "lsst/afw/table/Vector.h"
%}

%inline {

namespace lsst { namespace afw { namespace table {

// Custom iterator for Sets.  It holds a PyObject reference to its container to ensure its iterators
// don't get invalidated while it's alive.
// Of course, you can still invalidate iterators by erasing the elements that they point to, but
// that's something that the user would expect to cause problems, even in Python.
template <typename ContainerT>
class PythonIterator {
public:
    
    typedef typename ContainerT::Record Record;

    PTR(Record) _next() {
        if (_current == _end) return PTR(Record)();
        PTR(Record) r = _current;
        ++_current;
        return r;
    }

    PythonIterator(PyObject * py, ContainerT & c) : _py(py), _current(c.begin()), _end(c.end()) {
        Py_INCREF(_py);
    }

    ~PythonIterator() { Py_DECREF(_py); }

private:

    typedef typename ContainerT::iterator Iter;

    PyObject * _py;
    Iter _current;
    Iter _end;
};

 }}} // namespace lsst::afw::table

} // %inline

namespace lsst { namespace afw { namespace table {

%extend PythonIterator {

    %pythoncode %{
    def __iter__(self):
        return self
    def next(self):
        r = self._next()
        if r is None: raise StopIteration()
        return r
    %}

}

template <typename RecordT, typename TableT>
class VectorT {
public:
    
    PTR(TableT) getTable() const;

    Schema const getSchema() const;

    explicit VectorT(PTR(TableT) const & table = PTR(TableT)());

    explicit VectorT(Schema const & table);

    VectorT(VectorT const & other);

    void writeFits(std::string const & filename) const;

    static VectorT readFits(std::string const & filename);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();

};

template <typename RecordT=SourceRecord, typename TableT=typename RecordT::Table>
class SourceSetT {
public:

    PTR(TableT) getTable() const;

    Schema const getSchema() const;

    explicit SourceSetT(PTR(TableT) const & table = PTR(TableT)());

    explicit SourceSetT(Schema const & table);

    void writeFits(std::string const & filename) const;

    static SourceSetT readFits(std::string const & filename);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();
 
};

%extend VectorT {
    std::size_t __len__() const {
        return self->size();
    }
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
    void append(PTR(RecordT) const & p) {
        self->push_back(p);
    }
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

%extend SourceSetT {
    std::size_t __len__() const {
        return self->size();
    }
    PTR(RecordT) __getitem__(RecordId id) const {
        PTR(RecordT) r = self->get(id);
        if (!r) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                (boost::format("Source with ID %ld not found in set.") % id).str()
            );
        }
        return r;
    }
    void __delitem__(RecordId id) {
        lsst::afw::table::SourceSetT< RecordT, TableT >::iterator iter = self->find(id);
        if (iter == self->end()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                (boost::format("Source with ID %ld not found in set.") % id).str()
            );
        }
        self->erase(iter);
    }
    void add(PTR(RecordT) const & p) {
        self->insert(p);
    }
    %pythoncode %{
    table = property(getTable)
    schema = property(getSchema)
    def __iter__(self):
        return SourceSetIterator(self, self)
    %}
}

%template (BaseVector) VectorT<BaseRecord,BaseTable>;
%template (SourceVector) VectorT<SourceRecord,SourceTable>;
%template (SourceSet) SourceSetT<SourceRecord,SourceTable>;

%template (SourceSetIterator) PythonIterator<SourceSetT<SourceRecord,SourceTable> >;

}}}

