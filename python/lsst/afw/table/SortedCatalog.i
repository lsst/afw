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

    static SortedCatalogT readFits(std::string const & filename, int hdu=2);
    static SortedCatalogT readFits(fits::MemFileManager & manager, int hdu=2);

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
}

}}} // namespace lsst::afw::table

%define %declareSortedCatalog(TMPL, PREFIX)
%pythondynamic CatalogT< PREFIX ## Record >;
%template (_ ## PREFIX ## CatalogBase) CatalogT< PREFIX ## Record >;
%declareCatalog(TMPL, PREFIX)
%enddef
