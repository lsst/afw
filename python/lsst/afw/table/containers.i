%{
#include "lsst/afw/table/Vector.h"
%}

namespace lsst { namespace afw { namespace table {

template <typename RecordT, typename TableT>
class Vector {
public:
    
    PTR(TableT) getTable() const;

    Schema const getSchema() const;

    explicit Vector(PTR(TableT) const & table = PTR(TableT)());

    explicit Vector(Schema const & table);

    Vector(Vector const & other);

    void writeFits(std::string const & filename) const;

    static Vector readFits(std::string const & filename);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();

};

template <typename RecordT=SourceRecord, typename TableT=typename RecordT::Table>
class SourceSet {
public:

    PTR(TableT) getTable() const;

    Schema const getSchema() const;

    explicit SourceSet(PTR(TableT) const & table = PTR(TableT)());

    explicit SourceSet(Schema const & table);

    void writeFits(std::string const & filename) const;

    static SourceSet readFits(std::string const & filename);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();
 
};

%extend Vector {
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

%extend SourceSet {
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
        const_iterator iter = self->find(id);
        if (i1 == self->end()) {
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
    %}
}

}}}

%template (BaseVector) lsst::afw::table::Vector<lsst::afw::table::BaseRecord,lsst::afw::table::BaseTable>;
%template (SourceVector) lsst::afw::table::Vector<lsst::afw::table::SourceRecord,lsst::afw::table::SourceTable>;
%template (SourceSet) lsst::afw::table::SourceSet<lsst::afw::table::SourceRecord,lsst::afw::table::SourceTable>;
