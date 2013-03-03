/*
 * Wrappers for CatalogT, instantiation of BaseCatalog, and macros used in wrapping other catalogs.
 */

%{
#include "lsst/afw/table/Catalog.h"
%}

%include "cdata.i"

namespace lsst { namespace afw {

namespace fits {

struct MemFileManager {
     MemFileManager();
     MemFileManager(std::size_t len);
     void* getData() const;
     std::size_t getLength() const;
};

} namespace table {

template <typename RecordT>
class CatalogT {
public:

    typedef typename RecordT::Table Table;
    typedef typename RecordT::ColumnView ColumnView;

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
    void writeFits(fits::MemFileManager & manager, std::string const & mode="w") const;

    static CatalogT readFits(std::string const & filename, int hdu=2);
    static CatalogT readFits(fits::MemFileManager & manager, int hdu=2);

    ColumnView getColumnView() const;

    bool isContiguous() const;

    std::size_t capacity() const;

    void reserve(std::size_t n);

    %pythonappend getColumnView %{
        self._columns = val
    %}

    PTR(RecordT) addNew();

    CatalogT<RecordT> subset(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step) const;
};

%extend CatalogT {
    std::size_t __len__() const { return self->size(); }
    PTR(RecordT) __getitem__(std::ptrdiff_t i) const {
        if (i < 0) i = self->size() + i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Catalog index %d out of range.") % i).str()
            );
        }
        return self->get(i);
    }
    void extend(CatalogT const & other, bool deep) {
        self->insert(self->end(), other.begin(), other.end(), deep);
    }
    void extend(CatalogT const & other, SchemaMapper const & mapper) {
        self->insert(mapper, self->end(), other.begin(), other.end());
    }
    %feature("shadow") extend %{
    def extend(self, iterable, deep=None, mapper=None):
        """Append all records in the given iterable to the catalog.

        Arguments:
          iterable ------ any Python iterable containing records
          deep ---------- if True, the records will be deep-copied; ignored
                          if mapper is not None (that always implies True).
          mapper -------- a SchemaMapper object used to translate records
        """
        self._columns = None
        if isinstance(iterable, type(self)):
            if mapper is not None:
                $action(self, iterable, mapper)
            else:
                $action(self, iterable, deep)
        else:
            for record in iterable:
                if mapper is not None:
                    self.append(self.table.copyRecord(record, mapper))
                elif deep:
                    self.append(self.table.copyRecord(record))
                else:
                    self.append(record.cast(self.Record))
    %}
    %feature("shadow") __getitem__ %{
    def __getitem__(self, k):
        """Return the record at index k if k is an integer,
        or return a column if k is a string field name or Key.
        """
        if type(k) is slice:
            (start, stop, step) = (k.start, k.stop, k.step)
            if step is None:
                step = 1
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            return self.subset(start, stop, step)
        try:
            return $action(self, k)
        except TypeError:
            return self.columns[k]
    %}
    void __setitem__(std::ptrdiff_t i, PTR(RecordT) const & p) {
        if (i < 0) i = self->size() + i;
        if (std::size_t(i) >= self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Catalog index %d out of range.") % i).str()
            );
        }
        self->set(i, p);
    }
    void __delitem__(std::ptrdiff_t i) {
        if (i < 0) i = self->size() + i;
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
        if (i < 0) i = self->size() + i;
        if (std::size_t(i) > self->size()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Catalog index %d out of range.") % i).str()
            );
        }
        self->insert(self->begin() + i, p);
    }
    %feature("pythonprepend") __setitem__ %{
        self._columns = None
    %}
    %feature("pythonprepend") __delitem__ %{
        self._columns = None
    %}
    %feature("pythonprepend") append %{
        self._columns = None
    %}
    %feature("pythonprepend") insert %{
        self._columns = None
    %}
    %pythoncode %{
    def __getColumns(self):
        if not hasattr(self, "_columns") or self._columns is None:
            self._columns = self.getColumnView()
        return self._columns
    columns = property(__getColumns, doc="a column view of the catalog")
    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]
    def get(self, k):
        """Synonym for self[k]; provided for consistency with C++ interface."""
        return self[k]
    def cast(self, type_, deep=False):
        """Return a copy of the catalog with the given type, optionally
        cloning the table and deep-copying all records if deep==True.
        """
        if deep:
            table = self.table.clone()
            table.preallocate(len(self))
        else:
            table = self.table
        newTable = table.cast(type_.Table)
        copy = type_(newTable)
        copy.extend(self, deep=deep)
        return copy
    def copy(self, deep=False):
        return self.cast(type(self), deep)
    def __getattribute__(self, name):
        # Catalog forwards unknown method calls to its table and column view
        # for convenience.  (Feature requested by RHL; complaints about magic
        # should be directed to him.)
        # We have to use __getattribute__ because SWIG overrides __getattr__.
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name == "_columns":
                raise
        try:
            return getattr(self.table, name)
        except AttributeError:
            return getattr(self.columns, name)
    table = property(getTable)
    schema = property(getSchema)
    def __reduce__(self):
        manager = MemFileManager()
        self.writeFits(manager)
        size = manager.getLength()
        data = cdata(manager.getData(), size);
        return (lsst.afw.table.unpickleCatalog, (self.__class__, data, size))
    %}
}

%pythoncode %{
def unpickleCatalog(cls, data, size):
    """Unpickle a catalog with data produced by its __reduce__ method"""
    manager = MemFileManager(size)
    memmove(manager.getData(), data)
    return cls.readFits(manager)
%}

}}} // namespace lsst::afw::table

// Macro that should be used to instantiate a Catalog type.
%define %declareCatalog(TMPL, PREFIX)
%pythondynamic TMPL< PREFIX ## Record >;
%template (PREFIX ## Catalog) TMPL< PREFIX ## Record >;
typedef TMPL< PREFIX ## Record > PREFIX ## Catalog;
%extend TMPL< PREFIX ## Record > {
%pythoncode %{
    Table = PREFIX ## Table
    Record = PREFIX ## Record
    ColumnView = PREFIX ## ColumnView
%}
}
// Can't put this in class %extend blocks because they need to come after all class blocks in Python.
%pythoncode %{ 
PREFIX ## Record.Table = PREFIX ## Table
PREFIX ## Record.Catalog = PREFIX ## Catalog
PREFIX ## Record.ColumnView = PREFIX ## ColumnView
PREFIX ## Table.Record = PREFIX ## Record
PREFIX ## Table.Catalog = PREFIX ## Catalog
PREFIX ## Table.ColumnView = PREFIX ## ColumnView
PREFIX ## ColumnView.Record = PREFIX ## Record
PREFIX ## ColumnView.Table = PREFIX ## Table
PREFIX ## ColumnView.Catalog = PREFIX ## Catalog
%}
%enddef
