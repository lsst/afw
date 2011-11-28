// -*- c++ -*-
#ifndef AFW_TABLE_config_h_INCLUDED
#define AFW_TABLE_config_h_INCLUDED

#define FUSION_MAX_VECTOR_SIZE 20
#define FUSION_MAX_MAP_SIZE 20

#include "boost/cstdint.hpp"

namespace lsst { namespace afw { namespace table {

typedef boost::uint64_t RecordId;

enum TreeMode { NO_NESTING, DEPTH_FIRST };

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_config_h_INCLUDED
