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
 
%define tableLib_DOCSTRING
"
Classes for interacting with tabular data, including:
 - Schema (column definition)
 - <X>Record (rows)
 - <X>Catalog (container of rows)
 - <X>Table (factory/memory manager for rows)
 - <X>ColumnView (view of a catalog as column arrays)
where <X> is one of [Base, Simple, Source, Exposure].
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.table", docstring=tableLib_DOCSTRING) tableLib

%include "lsst/p_lsstSwig.i"

%include "lsst/afw/table/misc.h"

namespace lsst { namespace afw { namespace table {

class Schema;
class SchemaMapper;

template <typename RecordT> class CatalogT;
template <typename RecordT> class SortedCatalogT;
template <typename RecordT> class ExposureCatalogT;
template <typename RecordT> class ColumnViewT;

class BaseRecord;
class BaseTable;
typedef CatalogT<BaseRecord> BaseCatalog;
class BitsColumn;
class BaseColumnView;

class IdFactory;
class SimpleRecord;
class SimpleTable;
typedef SortedCatalogT<SimpleRecord> SimpleCatalog;
typedef ColumnViewT<SimpleRecord> SimpleColumnView;

class SourceRecord;
class SourceTable;
typedef SortedCatalogT<SourceRecord> SourceCatalog;
typedef SourceColumnViewT<SourceRecord> SourceColumnView;

class ExposureRecord;
class ExposureTable;
typedef ExposureCatalogT<ExposureRecord> ExposureCatalog;
typedef ColumnViewT<ExposureRecord> ExposureColumnView;

template <typename Record1, typename Record2> struct Match;
typedef Match<SimpleRecord,SimpleRecord> SimpleMatch;
typedef Match<SimpleRecord,SourceRecord> ReferenceMatch;
typedef Match<SourceRecord,SourceRecord> SourceMatch;
typedef std::vector<SimpleMatch> SimpleMatchVector;
typedef std::vector<ReferenceMatch> ReferenceMatchVector;
typedef std::vector<SourceMatch> SourceMatchVector;

}}} // namespace lsst::afw::table
