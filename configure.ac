AC_INIT([fw], [X.Y], [rhl@astro.princeton.edu])

dnl ------------- Include files from m4 --------------------------------------
dnl Don't use aclocal, as you'll have to update dependent products too if
dnl you add macros to ./m4

dnl ------------- Include files from ./m4 ------------------------------------

define([SUPPORT_DIR], esyscmd(echo $SUPPORT_DIR | perl -pe 'chomp'))
m4_include(SUPPORT_DIR/m4/ac_build.m4)
m4_include(SUPPORT_DIR/m4/ac_eups.m4)
undefine([SUPPORT_DIR])

dnl ------------- EUPS directories and how to install -------------------------

EUPS_DEFINE_ROOT($HeadURL: file:///u/rhl/SVN/support/trunk/configure.ac $)
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

AC_C_BIGENDIAN(,[CFLAGS="$CFLAGS -DLSST_LITTLE_ENDIAN"])

dnl ---------------- Check for extra things that we need/want ------------------

if [[ "$(echo $CFLAGS | sed -ne '/-Wall/p')" = "" ]]; then
   CFLAGS="$CFLAGS -Wall"
fi

AC_SEARCH_LIBS([pow], [m], , AC_MSG_ERROR([Failed to find pow (!)]))

dnl ------------ Python/Swig ---------------------------------------------------

LSST_FIND_PYTHON()
LSST_FIND_NUMARRAY([-DUSE_NUMARRAY=1])
LSST_SWIG([1.3.27])

dnl --------- Products from UPS ------------------------------------------------
dnl (remember to update etc/Makefile.global if you add a new product)

EUPS_WITHOUT_CONFIGURE([wcstools],  [wcslib.h], -lwcs,     [wcs,wcscat])

dnl ------------ Creation of dynamic libraries ---------------------------------
	
LSST_DYNAMIC_LIBS()

AC_SUBST(EXTRA_LIBS, $LIBS)
AC_CONFIG_FILES([etc/makedefs etc/Makefile.global etc/installdefs])
AC_OUTPUT
