#
#   afw 2.3
#
#
m4_changequote([, ])m4_dnl
#
m4_dnl
m4_dnl  For a simple external package that follows the configure-make pattern,
m4_dnl  it may only be necessary to update the values of the following macros.
m4_dnl  Only m4_PACKAGE and m4_VERSION are required.  
m4_dnl
m4_define([m4_PACKAGE], [afw])m4_dnl
m4_define([m4_VERSION], [2.3])m4_dnl
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
# Make sure scons is available
m4_ENSURE_SCONS()

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
echo ("building and installing m4_PACKAGE-m4_VERSION...")
m4_SCONS_BUILD()
cd('$LSST_HOME')

#
# Now download & install the EUPS table file and load package into EUPS
#
m4_include([PacmanLsst-post.m4])m4_dnl

uninstallShell('rm -rf $PWD/m4_PACKAGE/m4_VERSION')
uninstallShell('rmdir $PWD/m4_PACKAGE; true')



