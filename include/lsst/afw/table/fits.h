// -*- lsst-c++ -*-
#ifndef AFW_TABLE_fits_h_INCLUDED
#define AFW_TABLE_fits_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/TableBase.h"

namespace lsst { namespace afw { namespace table { namespace fits {

typedef afw::fits::Fits Fits;

/**
 *  @brief Create a FITS binary table with a header that corresponds to the given schema.
 *
 *  This is a low-level routine to aid in implementing higher-level FITS writers.  It
 *  should generally not be called directly by users.
 *
 *  @param[in,out]  fits           An afw cfitsio wrapper object corresponding to a new FITS file
 *                                 or a new HDU of an existing FITS file.
 *  @param[in]      schema         Table schema to convert to FITS header form.
 *  @param[in]      sanitizeNames  If true, periods in names will be converted to underscores.
 *
 *  Most Schema fields are converted directly to corresponding FITS column types.  Compound
 *  fields will additionally have a special TCCLSn key set, denoting what compound field 
 *  template class they should be loaded into.  Without these special keys, FITS writers
 *  will load all compound fields into arrays.
 *
 *  Flag fields are also handled differently; all Flag fields in the Schema are combined
 *  into a single "flag" bit array column, and a TFLAGn key is set for each bit.  The order
 *  the TFLAGn keys appear in the header relative to the TTYPEn keys will determines the order
 *  of fields in a loaded Schema.
 */
void writeFitsHeader(Fits & fits, Schema const & schema, bool sanitizeNames);

/**
 *  @brief Write the records of a table to a FITS binary table.
 *
 *  The binary table must have already been created with writeFitsHeader, using the table's
 *  Schema.  Additional FITS columns may be present in addition to those created by
 *  writeFitsHeader (these will be ignored).  The table and record auxiliary data is not
 *  written.
 
 *  @param[in,out]  fits           An afw cfitsio wrapper object already processed by writeFitsHeader.
 *  @param[in]      table          Table to write to disk.
 */
void writeFitsRecords(Fits & fits, TableBase const & table);

/**
 *  @brief Read the header of FITS binary table, returning a Schema.
 *
 *  Fully general FITS tables are not supported; some FITS column types (mostly
 *  various-sized integers) have no afw::table counterpart, and only float and
 *  double arrays.  In addition, only one bit array column may be present, the FLAG_COL
 *  key must contain the number of this column, and all bits must be labeled using
 *  TFLAGn keys.  The TCCLSn keys will be read to determine the type of multi-element
 *  fields; if these are not set, multi-element columns will be loaded into array fields
 *  and single element columns will be loaded into scalar fields.
 *
 *  @param[in,out]  fits             An afw cfitsio wrapper object corresponding to a FITS
 *                                   binary table.
 *  @param[in]      unsanitizeNames  If True, underscores in names will be converted to periods.
 *  @param[in]      nCols            Number of FITS columns to read; any columns after
 *                                   this will be ignored (allowing them to have unsupported
 *                                   types and/or store auxiliary data to be loaded separately).
 *                                   Ignored if < 0.
 */
Schema readFitsHeader(Fits & fits, bool unsanitizeNames, int nCols=-1);

/**
 *  @brief Read the rows of a FITS binary table into a table.
 *
 *  The table's Schema must be equal to the Schema produced by calling readFitsHeader
 *  on the FITS table; this should almost always be done to initialize the table.
 *
 *  @param[in,out]  fits             An afw cfitsio wrapper object corresponding to a FITS
 *                                   binary table.
 *  @param[in,out]  table            Table to append records to.
 */
void readFitsRecords(Fits & fits, TableBase const & table);

}}}} // namespace lsst::afw::table::fits

#endif // !AFW_TABLE_fits_h_INCLUDED
