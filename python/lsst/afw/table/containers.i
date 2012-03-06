%{
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
%}

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class CatalogT {
public:

    typedef typename RecordT::Table Table;

    PTR(Table) getTable() const;

    %feature("autodoc", "Return the table shared by all the catalog's records.") getTable;

    Schema getSchema() const;

    %feature("autodoc", "Shortcut for self.getTable().getSchema()") getSchema;

    explicit CatalogT(PTR(Table) const & table);

    explicit CatalogT(Schema const & table);

    CatalogT(CatalogT const & other);

    %feature(
        "autodoc", 
        "Constructors:  __init__(self, table) -> empty catalog with the given table\n"
        "               __init__(self, schema) -> empty catalog with a new table with the given schema\n"
        "               __init__(self, catalog) -> shallow copy of the given catalog\n"
    ) CatalogT;

    void writeFits(std::string const & filename, std::string const & mode="w") const;

    static CatalogT readFits(std::string const & filename, int hdu=2);

    ColumnView getColumnView() const;

    PTR(RecordT) addNew();

    CatalogT copy() const;

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
    def cast(self, type, deep=False):
        newTable = self.table.clone() if deep else self.table
        copy = type(newTable)
        for record in self:
            newRecord = newTable.copyRecord(record) if deep else record
            copy.append(newRecord)
        return copy
    table = property(getTable)
    schema = property(getSchema)
    %}
}

template <typename RecordT>
class SimpleCatalogT<RecordT> : public CatalogT<RecordT> {
public:

    typedef typename RecordT::Table Table;

    explicit SimpleCatalogT(PTR(Table) const & table);

    explicit SimpleCatalogT(Schema const & table);

    SimpleCatalogT(SimpleCatalogT const & other);

    static SimpleCatalogT readFits(std::string const & filename, int hdu=2);

    SimpleCatalogT copy() const;

    bool isSorted() const;
    void sort();
};

// For some reason, SWIG won't extend the template; it's only happy if
// we extend the instantiation (but compare to %extend CatalogT, above)
// ...mystifying.  Good thing we only have two instantiations.
%extend SimpleCatalogT<SimpleRecord> {
    PTR(SimpleRecord) find(RecordId id) {
        return self->find(id);
    }
}
%extend SimpleCatalogT<SourceRecord> {
    PTR(SourceRecord) find(RecordId id) {
        return self->find(id);
    }
}

%template (BaseCatalog) CatalogT<BaseRecord>;
%template (SimpleCatalogBase) CatalogT<SimpleRecord>;
%template (SimpleCatalog) SimpleCatalogT<SimpleRecord>;
%template (SourceCatalogBase) CatalogT<SourceRecord>;
%template (SourceCatalog) SimpleCatalogT<SourceRecord>;

typedef CatalogT<BaseRecord> BaseCatalog;
typedef SimpleCatalogT<SimpleRecord> SimpleCatalog;
typedef SimpleCatalogT<SourceRecord> SourceCatalog;

}}} // namespace lsst::afw::table
