Parallel reading test
=====================

C++ minimal multi thread executable reading fits files
from validation_data_hsc by `lsst::afw::table::SourceCatalog::readFits`
in 2 threads. As of DM-18695, DM-19212, this file (usually) crashes, see
example output `ptest_example_run_2019-04-10.txt`.

Build notes
===========

* This example program by default is built with `-pthread` and `-Og`
  appended to CCFLAGS. `-O3` is removed from CCFLAGS.
   
* You may want to build the whole `afw` package or the whole lsstsw
  stack with `-pthread -Og` compilation flags. `scons` does not
  use shell environment variables for consistency.
   
* To change `-O3`, sconsUtils recognizes the `opt` command line option
  `opt=g` translates to `-Og`.
   
* sconsUtils reads the `buildOpts.py` file for command line options
  at the package top level (only). Copy `examples/threading/buildOpts.py`
  to the `afw` top level to set `-pthread -Og` compilation for a standalone
  `scons` afw package build. Alternatively, build with
  `scons opt=g archflags=-pthread`.
   
* `eupspkg build` however picks up shell environment variables and passes
  on to `scons` as cmd-line arguments. If no variable is set,
  defaults are passed on. Therefore  `opt=g`  in `buildOpts.py` is
  always disrespected if using `rebuild` for lsstsw rebuilding.
   
* Use ``EUPSPKG_SCONSFLAGS="opt=g archflags=-pthread"`` for
  `rebuild`. Note, as of DM-18695, only scons configured C++ builds
  are affected by this.

* To build a complete lsstsw stack with `-pthread -Og` from scratch and
  to debug thread safety by this example program, use::

     export EUPSPKG_SCONSFLAGS="opt=g archflags=-pthread"
     # -r optional tickets branch to use
     rebuild -r tickets/DM-18695 afw
     setup -t bNNNN afw

* Then in the `examples/threading` directory, make test data available
  by linking `validation_data_hsc`::

     cd afw/examples/threading
     ln -s path_validation_data_hsc validation_data_hsc
     ./ptest
     # CRASH ususally, but not always...
     
