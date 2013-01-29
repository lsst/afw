/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

%include "lsst/afw/table/table_fwd.i"
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
%pythondynamic;
%template (_ ## PREFIX ## CatalogBase) CatalogT< PREFIX ## Record >;
%pythonnondynamic;
%declareCatalog(TMPL, PREFIX)
%enddef
