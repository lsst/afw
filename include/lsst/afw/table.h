// -*- lsst-c++ -*-
#ifndef LSST_AFW_table_h_INCLUDED
#define LSST_AFW_table_h_INCLUDED

#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @page afwTable Tables
 *
 *  @section afwTableBasicUsage Basic Usage
 *
 *  In C++:
 *  @code
 *  #include "lsst/afw/table.h"
 *  using namespace lsst::afw::table;
 *  Schema schema(false); // Or true to add parent/child tree to table.
 *  Key<int> k1 = schema.addField<int>("f1", "doc for f1");
 *  Key<float> k2 = schema.addField<float>("f2", "doc for f2", "units for f2);
 *  Key< Array<double> > k3 = schema.addField< Array<double> >("f3", "doc for f3", "units for f2", 5);
 *  SimpleTable table(schema, 15); // initial capacity for 15 rows.
 *  SimpleRecord record = table.addRecord();
 *  record.set(k1, 2);
 *  record.set(k2, 3.14);
 *  record[k3].setRandom(); // operator[] for arrays returns an Eigen::Map for in-place editing.
 *  std::cout << record.get(k1) << ", " << record.get(k2) << ", " << record.get(k3);
 *  @endcode
 *
 *  In Python:
 *  @code
 *  from lsst.afw.table import *
 *  schema = Schema(False) # Or True to add parent/child tree to table.
 *  k1 = schema.addField("f1", type=int, doc="doc for f1")
 *  k2 = schema.addField("f2", type=numpy.float32, "doc for f2", units="units for f2")
 *  k3 = schema.addField("f3", type="Array<F8>", doc="doc for f3", units="units for f2", size=5)
 *  table = SimpleTable(schema, 15) # initial capacity for 15 rows.
 *  record = table.addRecord()
 *  record.set(k1, 2)
 *  record.set(k2, 3.14)
 *  record.set(k3, numpy.random.randn(5)) # no reference array access in Python.
 *  print "%d, %f, %s" % (record.get(k1), record.get(k2), record.get(k3))
 *  @endcode
 *
 *  @section afwTableOverview Overview
 *  The primary objects users will interact with in the table library are Schemas, Keys, Tables,
 *  and Records.  Schema is a concrete class that defines the columns of a table; it behaves
 *  like a heterogeneous container of SchemaItem<T> objects, which are in turn composed of Field
 *  and Key objects.  A Field contains name, documentation, and units, while the Key object is
 *  a lightweight opaque object used to actually access elements of the table.  Using keys for
 *  access allows reads and writes to compile down to little (if any more) than a pointer
 *  offset and dereference.
 *  
 *  Record and table classes are defined in pairs; each record class has a 1-to-1
 *  correspondence with a table class.  A final table class inherits from the TableInterface
 *  template class, which inherits from the TableBase class (records have a parallel inheritance
 *  structure).  This inheritance is purely for implementation purposes; tables and records are
 *  not polymorphic and should always be passed by value.  Much of the public interface for
 *  tables and records is provided by these base classes.  SimpleRecord and SimpleTable are
 *  simple, general-purpose classes.  SourceRecord and SourceTable are designed to represent
 *  astronomical sources detected on an individual exposure, and contain a per-record Footprint
 *  in addition to the tabular data.  Additional record/table class pairs may be added in the
 *  future.
 *
 *  @section afwTableMemory Memory and Copy Semantics
 *  Tables and records share data through an internal shared_ptr.  Multiple records and tables
 *  will often refer to the same underlying data, and records cannot be constructed apart
 *  from the tables they belong to.
 *
 *  All copy constructors and assignment operators for records and tables are shallow - they
 *  affect what memory blocks the objects refer to, but do not modify the values of those memory
 *  blocks.  As in afw::image, RecordBase::operator<<= is overloaded to perform deep assignment of
 *  records.  Because this operator is not available in Python, so a "copyFrom" method is provided
 *  instead.
 *
 *  The memory in a table is allocated in blocks, as in most implementations of the C++ STL's deque.
 *  When the capacity of the most recent block is exhausted, a new block will be allocated for future
 *  records.  This means most - but not all - records will be close to their neighbors in memory.
 *  Unlike std::vector, the whole table is never silently reallocated.  This can be done explicitly,
 *  however, using TableBase::consolidate().  Columns of a table may be accessed as strided ndarray
 *  objects (and hency NumPy arrays in Python) using ColumnView, but a ColumnView can only be
 *  constructed from a consolidated table.
 *
 *  Because records and tables thus behave somewhat like "smart reference" objects, the usual
 *  constness semantics don't work for them.  Instead, only shallow mutators (like the regular
 *  assignment operators or TableBase::consolidate()) that change the pointer to the underlying
 *  memory are marked as non-const; accessors that modify the data itself are NOT marked as const.
 *  That means there is no convential way to prevent a user from modifying a table or record
 *  when passing it as an argument to another field.  To address this problem, tables and records
 *  (and iterators to records) carry ModificationFlags, which provide a runtime/assertion-based
 *  way of preventing code from modifications from happening in unexpected places.
 *
 *  @section afwTableFieldTypes Field Types
 *  In C++, field types are defined by the template arguments to Key and Field (among others).  Empty
 *  tag templates (Array, Point, Shape, Covariance) are used for compound fields.  In Python, strings
 *  are used to set field types.  The Key and Field classes for each type can be accessed through
 *  dictionaries (e.g. Key["F4"]), but usually these type strings are only explicitly written
 *  when passed as the 'type' argument of Schema.addField.  Aliases can also be used
 *  in place of type strings for scalar fields.
 *
 *  Some field types require a size argument to be passed to the Field constructor or Schema::addField;
 *  while this size can be set at compile time, all records must have the same size.
 *
 *  Not all field types support all types of data access.  All field types support atomic access to
 *  the entire field at once through RecordBase::get and RecordBase::set.  Some field types support
 *  square bracket access to mutable references as well.  Only scalar and array fields support column
 *  access through ColumnView.
 *
 *  A Key for an individual element of a compound fields can also be obtained from the compound Key
 *  object or (for 'named subfields') from the Schema directly (see Schema and the KeyBase specializations).
 *  Element keys can be used just like any other scalar Key, and hence provide access to column views.
 *
 *  <table border="1">
 *  <tr>
 *  <th>C++ Type</th>
 *  <th>Python Type String</th>
 *  <th>Python Aliases</th>
 *  <th>C++ Value (get/set) Type</th>
 *  <th>operator[]?</th>
 *  <th>Columns?</th>
 *  <th>Requires Size?</th>
 *  <th>Named Subfields</th>
 *  <th>Notes</th>
 *  </tr>
 *  <tr>
 *  <td>Flag</td> <td>"Flag"</td> <td></td> <td>bool</td>
 *  <td>No</td> <td>Yes</td> <td>No</td> <td></td> <td>Stored internally as a single bit</td>
 *  </tr>
 *  <tr>
 *  <td>boost::int32_t</td> <td>"I4"</td> <td>int, numpy.int32</td> <td>boost::int32_t</td>
 *  <td>Yes</td> <td>Yes</td> <td>No</td> <td></td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>boost::int64_t</td> <td>"I8"</td> <td>long, numpy.int64</td> <td>boost::int64_t</td>
 *  <td>Yes</td> <td>Yes</td> <td>No</td> <td></td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>float</td> <td>"F4"</td> <td>numpy.float32</td> <td>float</td>
 *  <td>Yes</td> <td>Yes</td> <td>No</td> <td></td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>double</td> <td>"F8"</td> <td>float, numpy.float64</td> <td>double</td>
 *  <td>Yes</td> <td>Yes</td> <td>No</td> <td></td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>Point<int></td> <td>"Point<I4>"</td> <td></td> <td>afw::geom::Point2i</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td>x,y</td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>Point<float></td> <td>"Point<F4>"</td> <td></td> <td>afw::geom::Point2D</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td>x,y</td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>Point<double></td> <td>"Point<F8>"</td> <td></td> <td>afw::geom::Point2D</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td>x,y</td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>Shape<float></td> <td>"Shape<F4>"</td> <td></td> <td>afw::geom::ellipses::Quadrupole</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td>xx,yy,xy</td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>Shape<double></td> <td>"Shape<F8>"</td> <td></td> <td>afw::geom::ellipses::Quadrupole</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td>xx,yy,xy</td> <td></td>
 *  </tr>
 *  <tr>
 *  <td>Array<float></td> <td>"Array<F4>"</td> <td></td> <td>Eigen::ArrayXf</td>
 *  <td>C++ only</td> <td>Yes</td> <td>Yes</td> <td></td> <td>operator[] returns an Eigen::Map</td>
 *  </tr>
 *  <tr>
 *  <td>Array<double></td> <td>"Array<F8>"</td> <td></td> <td>Eigen::ArrayXd</td>
 *  <td>C++ only</td> <td>Yes</td> <td>Yes</td> <td></td> <td>operator[] returns an Eigen::Map</td>
 *  </tr>
 *  <tr>
 *  <td>Covariance<float></td> <td>"Cov<F4>"</td> <td></td> <td>Eigen::MatrixXf</td>
 *  <td>No</td> <td>No</td> <td>Yes</td> <td></td>
 *  <td>symmetric matrix is stored packed (size*(size+1)/2 elements)</td>
 *  </tr>
 *  <tr>
 *  <td>Covariance<double></td> <td>"Cov<F8>"</td> <td></td> <td>Eigen::MatrixXd</td>
 *  <td>No</td> <td>No</td> <td>Yes</td> <td></td>
 *  <td>symmetric matrix is stored packed (size*(size+1)/2 elements)</td>
 *  </tr>
 *  <tr>
 *  <td>Covariance< Point<float> ></td> <td>"Cov<Point<F4>>"</td> <td></td> <td>Eigen::Matrix2f</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td></td> <td>symmetric matrix is stored packed (3 elements)</td>
 *  </tr>
 *  <tr>
 *  <td>Covariance< Point<double> ></td> <td>"Cov<Point<F8>>"</td> <td></td> <td>Eigen::Matrix2d</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td></td> <td>symmetric matrix is stored packed (3 elements)</td>
 *  </tr>
 *  <tr>
 *  <td>Covariance< Shape<float> ></td> <td>"Cov<Shape<F4>>"</td> <td></td> <td>Eigen::Matrix3f</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td></td> <td>symmetric matrix is stored packed (6 elements)</td>
 *  </tr>
 *  <tr>
 *  <td>Covariance< Shape<double> ></td> <td>"Cov<Shape<F8>>"</td> <td></td> <td>Eigen::Matrix3d</td>
 *  <td>No</td> <td>No</td> <td>No</td> <td></td> <td>symmetric matrix is stored packed (6 elements)</td>
 *  </tr>
 *  </table>
 */

}}} // namespace lsst::afw::table

#endif // !LSST_AFW_table_h_INCLUDED
