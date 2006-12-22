dnl 
dnl LSST configure.ac template
dnl  
dnl While all parts of this file can be updated to suit your needs, you 
dnl should at least consider updating the bits marked "UPDATE:" below.  
dnl
dnl Note: [ ] serve as quote characters to arguments to macros.  They are 
dnl       not always required, but never hurts to include them.  If you add
dnl       shell programming code to this file, you should be sure to use 
dnl       double brackets, [[ ]], in your if statments.  
dnl
dnl
AC_INIT([fw], [X.Y], [rallsman@lsst.org])

dnl ------------- Include files from ./build ---------------------------------
dnl Note:  We don't typically use aclocal.  Doing so may require updates to 
dnl dependent products too if when macros are added to ./build

m4_include(build/ac_build.m4)
m4_include(build/ac_eups.m4)

dnl ------------- EUPS directories and how to install -------------------------

EUPS_DEFINE_ROOT($HeadURL$)
EUPS_INSTALL_DIRS   

dnl C-compiler issues

AC_ARG_ENABLE(optimize,
    [AS_HELP_STRING(--enable-optimize, Turn on the optimizer)],
    [CFLAGS="-g -O2"], [CFLAGS="-g"])

dnl We'd like to use AC_PROG_CC_C99, but it isn't available until autoconf 2.60

ifdef([AC_PROG_CC_C99],
      [AC_PROG_CC_C99(cc gcc)],
      [AC_PROG_CC(cc gcc)
      AC_MSG_NOTICE([Adding -std=c99 to CC])
      CC="$CC -std=c99"])

AC_SUBST(LD, [$(echo $CC | awk '{print $1'})])

AC_PROG_RANLIB

AC_C_BIGENDIAN(,[CFLAGS="$CFLAGS -DPS_LITTLE_ENDIAN"])

if [[ "$(echo $CFLAGS | sed -ne '/-Wall/p')" = "" ]]; then
   CFLAGS="$CFLAGS -Wall"
fi

dnl ---------------- UPDATE: Require System libraries ------------------------
dnl
dnl Add AC_SEARCH_LIBS lines to check for functions provided by system 
dnl libraries (i.e. libraries not provided by the LSST software stack). 
dnl The required -lLIB will be added to the link options
dnl 
dnl            FUNC   LIB                  ERROR-MSG
AC_SEARCH_LIBS([pow], [m], , AC_MSG_ERROR([Failed to find pow (!)]))
dnl
dnl where the arguments are:
dnl    FUNC     a needed function
dnl    LIB      the library name as it would be appended to the "-l" 
dnl                linking option, i.e. -lLIB.
dnl

dnl ------------ UPDATE: Special LSST Checks ----------------------------------
dnl Uncomment one or more of the following lines if your package requires
dnl the package it refers to.  These will check to ensure that they have been
dnl installed and setup correctly.
dnl
LSST_FIND_PYTHON()                       dnl  if you require Python
LSST_FIND_NUMARRAY([-DUSE_NUMARRAY=1])   dnl  if you require Numarray
dnl LSST_FIND_NUMPY([-DUSE_NUMPY=1])         dnl  if you require Numarray
LSST_SWIG($SWIG_DIR/bin)                 dnl  if you require Swig


dnl --------- UPDATE: Products from UPS ----------------------------------------
dnl With the help of EUPS, this will check the availability of dependent 
dnl packages.  For each dependent package, add a EUPS_WITHOUT_CONFIGURE line 
dnl that looks like the following:
dnl 
dnl                        PKG-NAME  INCLUDE-FILE       LIBFLG  LIB,FUNCTION
dnl EUPS_WITHOUT_CONFIGURE([wcslib], [wcslib/wcslib.h], -lwcs,  [wcs,wcsini])
dnl
dnl where the arguments are:
dnl    PKG-NAME       the name of the package required (as known by EUPS)
dnl    INCLUDE-FILE   a space-delimited list of include files you would like 
dnl                     to check for; each file should be given with the 
dnl                     same path as would appear in your source code.
dnl    LIBFLG         a space-delimited list of "-l" options needed to link 
dnl                     the required libraries from this package.  (You do 
dnl                     *not* normally need to include any "-L" options.)
dnl    LIB,FUNCTION   a function you require from one of the libraries.  
dnl                     Provide this as 2 values separated by a comma (,).  
dnl                     The first value, LIB, is the name of the library as 
dnl                     you would pass it to -l; i.e. -lLIB.  The second value,
dnl                     FUNCTION, is the name of a function that you require.  
dnl                     Currently only one function can be checked for with 
dnl                     this macro; hopefully if one representative function 
dnl                     from this package is available, then it's likely that 
dnl                     they all are.  (Note that the [ ] are required on this
dnl                     pair of values.)

dnl Alternatively, if the dependent package, you can use the 
dnl EUPS_WITH_CONFIGURE macro.  This is currently not widely supported.
dnl Where applicable it is only necessary to provide the package name:
dnl
dnl EUPS_WITH_CONFIGURE([fancypackage])


dnl ------------ Creation of dynamic libraries ---------------------------------
	
LSST_DYNAMIC_LIBS()

AC_SUBST(EXTRA_LIBS, $LIBS)
AC_CONFIG_FILES([build/makedefs build/Makefile.global build/installdefs])
AC_OUTPUT
