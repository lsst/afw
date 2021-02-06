// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_Flag_h_INCLUDED
#define LSST_AFW_TABLE_Flag_h_INCLUDED

// This is a backwards-compatibility header; the template specializations for
// flag have now been included in the same headers as the default definitions
// of those templates (included below).  This guards against implicit
// instantiation of the default templates for Flag, which would be a violation
// of the One Definition Rule.

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/KeyBase.h"
#include "lsst/afw/table/Key.h"

#endif  // !LSST_AFW_TABLE_Flag_h_INCLUDED
