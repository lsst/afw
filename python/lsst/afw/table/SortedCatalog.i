%include "lsst/afw/table/Catalog.i"

%{
#include "lsst/afw/table/SortedCatalog.h"
%}

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class SortedCatalogT : public CatalogT<RecordT> {
public:

    typedef typename RecordT::Table Table;

    explicit SortedCatalogT(PTR(Table) const & table);

    explicit SortedCatalogT(Schema const & table);

    SortedCatalogT(SortedCatalogT const & other);

    %feature(
        "autodoc",
        "Constructors:  __init__(self, table) -> empty catalog with the given table\n"
        "               __init__(self, schema) -> empty catalog with a new table with the given schema\n"
        "               __init__(self, catalog) -> shallow copy of the given catalog\n"
    ) SortedCatalogT;

    static SortedCatalogT readFits(std::string const & filename, int hdu=0, int flags=0);
    static SortedCatalogT readFits(fits::MemFileManager & manager, int hdu=0, int flags=0);

    using CatalogT<RecordT>::isSorted;
    using CatalogT<RecordT>::sort;

    bool isSorted() const;
    void sort();

    SortedCatalogT<RecordT> subset(ndarray::Array<bool const,1> const & mask) const;

    SortedCatalogT<RecordT> subset(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step) const;
};

%extend SortedCatalogT {
    PTR(RecordT) find(RecordId id) {
        lsst::afw::table::SortedCatalogT< RecordT >::iterator i = self->find(id);
        if (i == self->end()) {
            return PTR(RecordT)();
        }
        return i;
    }

    %feature("shadow") find %{
    def find(self, value, key=None):
        """If key is None, return the Record with the given ID.
        Otherwise, return the Record for which record.get(key) == value.
        """
        if key is None:
            return $action(self, value)
        else:
            return type(self).__base__.find(self, value, key)
    %}

    %feature("shadow") sort %{
    def sort(self, key=None):
        """Sort the catalog (stable) by the given Key, or by ID if key is None.
        """
        if key is None:
            $action(self)
        else:
            type(self).__base__.sort(self, key)
    %}

    %feature("shadow") isSorted %{
    def isSorted(self, key=None):
        """Return True if self.sort(key) would be a no-op.
        """
        if key is None:
            return $action(self)
        else:
            return type(self).__base__.isSorted(self, key)
    %}
}

}}} // namespace lsst::afw::table

%define %declareSortedCatalog(TMPL, PREFIX)
%pythondynamic CatalogT< PREFIX ## Record >;
%template (_ ## PREFIX ## CatalogBase) CatalogT< PREFIX ## Record >;
%declareCatalog(TMPL, PREFIX)
%enddef
