dnl
dnl Macros to support shareable libraries, python, and swig

dnl
dnl RHL does not believe that the added complexity of libtool, e.g.
dnl   libtool --mode=execute gdb foo
dnl is warranted, especially since we're using ups to set e.g. LD_LIBRARY_PATH,
dnl so we'll set the variables by hand
dnl
AC_DEFUN([LSST_DYNAMIC_LIBS], [
   rhl_uname=$(uname)

   AC_MSG_NOTICE([Setting up shareable libraries for] $rhl_uname)
   if [[ $rhl_uname = "Darwin" ]]; then
      AC_SUBST(SO_LDFLAGS, ["-bundle -undefined suppress -flat_namespace"])
      AC_SUBST(SO, [so])
      AC_SUBST(DYLIB_LDFLAGS, ["-undefined suppress -flat_namespace -dynamiclib"])
      AC_SUBST(DYLIB, [dylib])
      CFLAGS="$CFLAGS -fPIC"
   elif [[ $rhl_uname = "Linux" ]]; then
      AC_SUBST(SO_LDFLAGS, ["-shared"])
      AC_SUBST(SO, [so])
      AC_SUBST(DYLIB_LDFLAGS, ["-shared"])
      AC_SUBST(DYLIB, [so])
     CFLAGS="$CFLAGS -fPIC"
   else
      AC_MSG_ERROR(Unknown O/S for setting up dynamic libraries: rhl_uname)
   fi
])

dnl
dnl Detect python and add appropriate flags to PYTHON_CFLAGS/PYTHON_LIBS
dnl
AC_DEFUN([LSST_FIND_PYTHON], [
   AC_ARG_WITH(python,
     [AS_HELP_STRING(--with-python=file,Specify name of python executable.)],
     [PYTHON="$withval"
     if [[ ! -x $PYTHON ]]; then
        PYTHON=""
     fi],
     AC_CHECK_PROG(PYTHON, python, python, ""))

   if [[ "$PYTHON" = "" ]]; then
      AC_MSG_FAILURE([You'll need python; try using --with-python=file.])
   fi

   PYTHON_INCDIR=$($PYTHON -c 'import distutils.sysconfig as ds; print ds.get_python_inc()')
   if [[ X"$PYTHON_DIR" != X"" ]]; then
      PYTHON_INCDIR=$(echo $PYTHON_INCDIR | sed -e "s|$PYTHON_DIR|\\\${PYTHON_DIR}|g")
   fi

   AC_SUBST(PYTHON_CFLAGS, [-I$PYTHON_INCDIR])
   AC_SUBST(PYTHON_LIBS, [])
])

dnl
dnl Detect numarray and add appropriate flags to PYTHON_CFLAGS/PYTHON_LIBS
dnl If $1 is defined, add it to PYTHON_CFLAGS -- e.g. LSST_FIND_NUMARRAY([-DUSE_NUMARRAY=1])
dnl
AC_DEFUN([LSST_FIND_NUMARRAY], [
   AC_ARG_ENABLE(numarray,
       [AS_HELP_STRING(--enable-numarray, Generate numarray code)])

   if [[ "$enable_numarray" = "" -o "$enable_numarray" = "yes" ]]; then
       AC_MSG_CHECKING([for numarray])
       NUMARRAY_INCDIR=[$($PYTHON -c 'import numarray, re; print re.sub(r"/numarray$", "", numarray.numinclude.include_dir)' 2> /dev/null)]
       if [[ $? != 0 ]]; then
          AC_MSG_RESULT([no])
          AC_MSG_WARN([Failed to find numarray; ignoring --enable-numarray])
       else
          AC_MSG_RESULT([ok ($NUMARRAY_INCDIR)])
          PYTHON_CFLAGS="$PYTHON_CFLAGS -I$NUMARRAY_INCDIR"
	  if [[ X"$NUMARRAY_DIR" != X"" ]]; then
	     PYTHON_CFLAGS=$(echo $PYTHON_CFLAGS | sed -e "s|$NUMARRAY_DIR|\\\${NUMARRAY_DIR}|g")
	  fi
	  ifelse($1, , ,
             [PYTHON_CFLAGS="$PYTHON_CFLAGS $1"])
       fi
   fi
])

dnl
dnl Detect numpy and add appropriate flags to PYTHON_CFLAGS/PYTHON_LIBS
dnl If $1 is defined, add it to PYTHON_CFLAGS -- e.g. LSST_FIND_NUMPY([-DUSE_NUMPY=1])
dnl
AC_DEFUN([LSST_FIND_NUMPY], [
   AC_ARG_ENABLE(numpy,
       [AS_HELP_STRING(--enable-numpy, Generate numpy code)])

   if [[ "$enable_numpy" = "" -o "$enable_numpy" = "yes" ]]; then
       AC_MSG_CHECKING([for numpy])
       NUMPY_INCDIR=$($PYTHON -c 'import numpy; print numpy.get_numpy_include()' 2> /dev/null)
       if [[ $? != 0 ]]; then
          AC_MSG_RESULT([no])
          AC_MSG_WARN([Failed to find numpy; ignoring --enable-numpy])
       else
          AC_MSG_RESULT([ok ($NUMPY_INCDIR)])
          PYTHON_CFLAGS="$PYTHON_CFLAGS -I$NUMPY_INCDIR"
	  if [[ X"$PYCORE_DIR" != X"" ]]; then
	     PYTHON_CFLAGS=$(echo $PYTHON_CFLAGS | sed -e "s|$PYCORE_DIR|\\\${PYCORE_DIR}|g")
	  fi
	  ifelse($1, , ,
             [PYTHON_CFLAGS="$PYTHON_CFLAGS $1"])
       fi
   fi
])

dnl ------------------- swig ---------------------
dnl
dnl Detect swig, possibly via --with-swig
dnl If you provide an argument such as 1.3.27, you'll be warned if the
dnl version found is older than the specified version.  If $2 is defined,
dnl an error is generated
dnl
AC_DEFUN([LSST_SWIG], [
   AC_ARG_WITH(swig,
     [AS_HELP_STRING(--with-swig=DIR,Specify location of SWIG executable.)],
     [SWIG="$withval/swig"
     if [[ ! -x $SWIG ]]; then
        SWIG=""
     fi],
     AC_CHECK_PROG(SWIG, swig, swig, ""))

   if [[ "$SWIG" != "" ]]; then
      [SWIG="$SWIG -w301,451 -python -Drestrict= -Dinline="]
   else
      AC_MSG_FAILURE([You'll need swig; try using --with-swig=DIR to specify its location.])
   fi

   ifelse($1, , , [
   swig_version=$($SWIG -version 2>&1 | perl -ne 'if(/^SWIG Version (\d)\.(\d)\.(\d+)/) { print 100000*[$]1 + 1000*[$]2 + [$]3; }')
   desired_swig_version=$(echo $1 | perl -ne 'if(/(\d)\.(\d)\.(\d+)/) { print 100000*[$]1 + 1000*[$]2 + [$]3; }')

   if [[ "$swig_version" = "" -o $swig_version -lt $desired_swig_version ]]; then
      ifelse($2, ,
	      AC_MSG_NOTICE([You would be better off with a swig version >= $1]),
	      AC_MSG_ERROR([Please provide a swig version >= $1]))
   fi
   unset swig_version; unset desired_swig_version])
])

dnl ------------------- wcstools ---------------------
dnl
dnl Detect WCSTools
dnl This package has an unusual organization, so EUPS_WITH_CONFIGURE won't work
dnl
AC_DEFUN([LSST_FIND_WCSTOOLS], [
   AC_ARG_WITH(wcstools,
     [AS_HELP_STRING(--with-wcstools=DIR, Specify the directory where WCSTools is installed.)],
     [WCSTOOLS_DIR="$withval"],
     [])
   echo -n checking for WCSTools...
   if [[ -z "$WCSTOOLS_DIR" ]]; then
      AC_MSG_FAILURE([You need WCSTools; use either setup or --with-wcstools=dir])
   fi

   if [[ ! -d "$WCSTOOLS_DIR" ]]; then
      AC_MSG_FAILURE(["$WCSTOOLS_DIR" is not a directory containing the package])
   fi

   if [[ ! -f $WCSTOOLS_DIR/libwcs/wcs.h ]]; then
      AC_MSG_FAILURE(["$WCSTOOLS_DIR" is missing include file: libwcs/wcs.h; is this really WCSTools?])
   fi
   if [[ ! -f $WCSTOOLS_DIR/libwcs/libwcs.a ]]; then
      AC_MSG_FAILURE(["$WCSTOOLS_DIR" is missing library: libwcs/libwcs.a; is WCSTools built?])
   fi
   echo " ok ($WCSTOOLS_DIR)" ])

AC_SUBST(WCSTOOLS_DIR)

dnl
dnl Aliases used in old RHL projects.
dnl
AC_DEFUN([RHL_DYNAMIC_LIBS], [LSST_DYNAMIC_LIBS])
AC_DEFUN([RHL_FIND_PYTHON],  [LSST_FIND_PYTHON])
AC_DEFUN([RHL_FIND_NUMPY],   [LSST_FIND_NUMPY])
AC_DEFUN([RHL_SWIG],         [LSST_SWIG])
