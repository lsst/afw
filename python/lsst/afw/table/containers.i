%{
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/Source.h"
%}

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class CatalogT {
public:

    typedef typename RecordT::Table Table;

    PTR(Table) getTable() const;

    Schema getSchema() const;

    explicit CatalogT(PTR(Table) const & table = PTR(Table)());

    explicit CatalogT(Schema const & table);

    CatalogT(CatalogT const & other);

    void writeFits(std::string const & filename) const;

    static CatalogT readFits(std::string const & filename);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();

};

%extend CatalogT {
    std::size_t __len__() const { return self->size(); }
    PTR(RecordT) __getitem__(std::ptrdiff_t i) const {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Catalog index %d out of range.") % i).str()
            );
        }
        return self->get(i);
    }
    void __setitem__(std::ptrdiff_t i, PTR(RecordT) const & p) {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Catalog index %d out of range.") % i).str()
            );
        }
        self->set(i, p);
    }
    void __delitem__(std::ptrdiff_t i) {
        if (i < 0) i = self->size() - i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Catalog index %d out of range.") % i).str()
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
                (boost::format("Catalog index %d out of range.") % i).str()
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
class SourceCatalogT<RecordT> : public CatalogT<RecordT> {
public:

    typedef typename RecordT::Table Table;

    explicit SourceCatalogT(PTR(Table) const & table = PTR(Table)());

    explicit SourceCatalogT(Schema const & table);

    SourceCatalogT(SourceCatalogT const & other);

    bool isSorted() const;
    void sort();
};

// For some reason, SWIG won't extend the template; it's only happy if
// we extend the instantiation (but compare to %extend CatalogT, above)
// ...mystifying.  Good thing we only have one instantiation.
%extend SourceCatalogT<SourceRecord> {
    PTR(SourceRecord) find(RecordId id) {
        return self->find(id);
    }
}

%template (BaseCatalog) CatalogT<BaseRecord>;
%template (SourceCatalogBase) CatalogT<SourceRecord>;
%template (SourceCatalog) SourceCatalogT<SourceRecord>;

typedef CatalogT<BaseRecord> BaseCatalog;
typedef SourceCatalogT<SourceRecord> SourceCatalog;

}}} // namespace lsst::afw::table
