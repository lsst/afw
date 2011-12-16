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
void createFitsHeader(Fits & fits, Schema const & schema, bool sanitizeNames);

/**
 *  @brief Write the records of a table to a FITS binary table.
 *
 *  The binary table must have already been created with createFitsHeader, using the table's
 *  Schema.  Additional FITS columns may be present in addition to those created by
 *  createFitsHeader (these will be ignored).  The table and record auxiliary data is not
 *  written.
 */
void writeFitsRecords(Fits & fits, TableBase const & table);

/**
 *  @brief Write the records of a table to a FITS binary table, using a mapper.
 *
 *  The binary table must have already been created with createFitsHeader, using the output
 *  Schema of the SchemaMapper, and the table's Schema must be the same as the mapper's input
 *  Schema.  Additional FITS columns may be present in addition to those created by
 *  createFitsHeader (these will be ignored).  The table and record auxiliary data is not
 *  written.
 */
void writeFitsRecords(Fits & fits, TableBase const & table, SchemaMapper const & mapper);

}}}} // namespace lsst::afw::table::fits

#endif // !AFW_TABLE_fits_h_INCLUDED
