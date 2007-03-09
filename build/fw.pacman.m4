#
#   package version
#
#
m4_changequote([, ])m4_dnl
#
m4_dnl
m4_dnl  For a simple external package that follows the configure-make pattern,
m4_dnl  it may only be necessary to update the values of the following macros.
m4_dnl  Only m4_PACKAGE and m4_VERSION are required.  
m4_dnl
m4_define([m4_PACKAGE], [fw])m4_dnl
m4_define([m4_VERSION], [0.3.1])m4_dnl
m4_define([m4_TARBALL], [m4_PACKAGE-m4_VERSION.tar.gz])m4_dnl
# 
# set up the initial pacman definitions and environment variables.
#
m4_include([PacmanLsst-pre.m4])m4_dnl
m4_dnl
m4_dnl  uncomment and adjust freeMegsMinimum() if you know a good value
m4_dnl  for this package.
m4_dnl
# freeMegsMinimum(11)       # requires at least 11 Megs to build and install

#
# denote dependencies
#
# package('m4_CACHE:otherpkg-2.2')
setenvShellTemp('PYFITS_DIR', 'export SHELL=sh; source $EUPS_DIR/bin/setups.sh; setup pyfits; echo $PYFITS_DIR')
envIsSet('PYFITS_DIR')
echo('Using PYFITS_DIR=$PYFITS_DIR')
shell('[[ -d "$PYFITS_DIR" ]]')

setenvShellTemp('SWIG_DIR', 'export SHELL=sh; source $EUPS_DIR/bin/setups.sh; setup swig; echo $SWIG_DIR')
envIsSet('SWIG_DIR')
echo('Using SWIG_DIR=$SWIG_DIR')
shell('[[ -d "$SWIG_DIR" ]]')

setenvShellTemp('WCSTOOLS_DIR', 'export SHELL=sh; source $EUPS_DIR/bin/setups.sh; setup wcstools; echo $WCSTOOLS_DIR')
envIsSet('WCSTOOLS_DIR')
echo('Using WCSTOOLS_DIR=$WCSTOOLS_DIR')
shell('[[ -d "$WCSTOOLS_DIR" ]]')

#
# begin installation assuming we are located in LSST_HOME
#
# available environment variables:
#   LSST_HOME           the root of the LSST installation (the current 
#                          directory)
#   LSST_BUILD          a directory where one can build the package
#
# EUPS_PATH and EUPS_FLAVOR should also be set.
#

cd('$LSST_BUILD')

#
#   download any tarballs and unzip
#
echo ("downloading and extracting m4_PACKAGE-m4_VERSION...")
downloadUntar('m4_PKGURL/m4_PKGPATH/m4_TARBALL','BUILDDIR')

#
#   cd into the untarred directory, configure, make and make install
#
cd('$BUILDDIR')
echo ("configuring m4_PACKAGE-m4_VERSION...")
shell('export SHELL=sh; source $EUPS_DIR/bin/setups.sh; setup pyfits; setup wcstools; setup swig; ./configure')

echo ("running make install")
shell('export SHELL=sh; source $EUPS_DIR/bin/setups.sh; setup pyfits; setup wcstools; setup swig; make installnowarn')
cd()

#
# Now download & install the EUPS table file and load package into EUPS
#
m4_include([PacmanLsst-post.m4])m4_dnl

uninstallShell('rm -rf $PWD/m4_PACKAGE/m4_VERSION')
uninstallShell('rmdir $PWD/m4_PACKAGE; true')



