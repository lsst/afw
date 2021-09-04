// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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

#ifndef LSST_AFW_TABLE_fwd_h_INCLUDED
#define LSST_AFW_TABLE_fwd_h_INCLUDED

/*
 *  Forward declarations and typedefs for afw::table
 *
 *  Because many of the types in afw::table are actually typedefs of template classes,
 *  manual forward declarations are verbose and fragile.  This file provides forward
 *  declarations and typedefs (of forward declarations) for all public classes in
 *  the package.
 */

#include <vector>

#include "lsst/afw/table/misc.h"

namespace lsst {
namespace afw {
namespace table {

template <typename T>
class Key;
template <typename T>
struct Field;
template <typename T>
struct SchemaItem;
class Schema;
class SchemaMapper;

template <typename T>
class OutputFunctorKey;
template <typename T>
class InputFunctorKey;

template <typename T>
class ColumnViewT;
template <typename RecordT>
class CatalogT;
template <typename RecordT>
class SortedCatalogT;
template <typename RecordT>
class SourceColumnViewT;
template <typename RecordT>
class ExposureCatalogT;

class BaseRecord;
class BaseTable;
class BaseColumnView;
using BaseCatalog = CatalogT<BaseRecord>;
using ConstBaseCatalog = CatalogT<const BaseRecord>;

class IdFactory;
class SimpleRecord;
class SimpleTable;
using SimpleColumnView = ColumnViewT<SimpleRecord>;
using SimpleCatalog = SortedCatalogT<SimpleRecord>;
using ConstSimpleCatalog = SortedCatalogT<const SimpleRecord>;

class SourceRecord;
class SourceTable;
using SourceColumnView = SourceColumnViewT<SourceRecord>;
using SourceCatalog = SortedCatalogT<SourceRecord>;
using ConstSourceCatalog = SortedCatalogT<const SourceRecord>;

class ExposureRecord;
class ExposureTable;
using ExposureColumnView = ColumnViewT<ExposureRecord>;
using ExposureCatalog = ExposureCatalogT<ExposureRecord>;
using ConstExposureCatalog = ExposureCatalogT<const ExposureRecord>;

class AmpInfoRecord;
class AmpInfoTable;
using AmpInfoColumnView = ColumnViewT<AmpInfoRecord>;
using AmpInfoCatalog = CatalogT<AmpInfoRecord>;
using ConstAmpInfoCatalog = CatalogT<const AmpInfoRecord>;

template <typename Record1, typename Record2>
struct Match;

using SimpleMatch = Match<SimpleRecord, SimpleRecord>;
using ReferenceMatch = Match<SimpleRecord, SourceRecord>;
using SourceMatch = Match<SourceRecord, SourceRecord>;

using SimpleMatchVector = std::vector<SimpleMatch>;
using ReferenceMatchVector = std::vector<ReferenceMatch>;
using SourceMatchVector = std::vector<SourceMatch>;

namespace io {

class Writer;
class Reader;
class FitsWriter;
class FitsReader;
class Persistable;
class InputArchive;
class OutputArchive;
class CatalogVector;

}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_TABLE_fwd_h_INCLUDED
