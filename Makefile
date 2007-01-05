# Template Makefile for a package's top-level directory 
#
# Normally, the only editing this file needs is adding subdirectories 
# to the SUBDIRS variable
#
include build/installdefs

# set this to the subdirectories where make must run recursively.
# These must contain there own Makefiles.  When they exist, the following 
# subdirectories fall into this category:
#   
# SUBDIRS = python src java etc scripts doc
#
# The build and ups subdirectories usually do not.  Do not include directories 
# that do not exist. 
#
SUBDIRS = python 

SHELL = /bin/sh
MAKEFLAGS = I $(PWD)/build

all : build

.PHONY : build
build : $(SUBDIRS)

.PHONY : $(SUBDIRS)
$(SUBDIRS): configure
	$(MAKE) -C $@ -$(MAKEFLAGS) build

.PHONY : build
clean distclean : configure
	@for f in $(SUBDIRS); do \
		if [ ! -d $$f ]; then \
			echo No such directory: $$f >&2; \
		else \
			if [ ! \( $$f = test -a $@ = all \) ]; then \
				(cd $$f; echo $$f; $(MAKE) -$(MAKEFLAGS) $@); \
			fi; \
		fi \
	done

.PHONY : topClean
topClean :
	$(RM) -r TAGS config.{log,status} autom4te.cache
	$(RM) core core.[0-9]*[0-9] *~
clean : topClean
distclean: clean
#
# Rebuild configure; almost no-one should need to do this
#
configure : configure.ac
	@ echo "Rebuilding ./configure"
	autoconf
#
# Update Makefile dependencies
#
.PHONY: depend
depend :
	@for f in swig; do \
		(cd $$f; echo $$f; $(MAKE) $(MAKEFLAGS) $@); \
	done
#
# Install things in their proper places as specified in etc/installdefs
#
.PHONY : install declare installwarn installnowarn installups installsubs
install: installwarn installnowarn 
installnowarn: installsubs installups
installwarn :
	@:
	@: Check the inode number for . and $(prefix) to find out if two
	@: directories are the same\; they may have different names due to
	@: symbolic links and automounters
	@:
	@if [ -d $(prefix) ]; then \
	    if [ `ls -id $(prefix) | awk '{print $$1}'` = `ls -id . | awk '{print $$1}'` ]; then \
		echo "The destination directory is the same" \
			"as the current directory; aborting." >&2; \
		echo ""; \
		exit 1; \
	   fi; \
	fi
	@echo "You will be installing in \$$(prefix)=$(prefix)"
	@echo "You should be sure to have updated before doing this."
	@echo ""
	@echo "I'll give you 5 seconds to think about it"
	@sleep 5
	@echo
	@if [ "$(prefix)" = "" ]; then \
		echo You have not specified a destination directory >&2; \
		exit 1; \
	fi
	@:
	@: End of checks, time to go
	@:

installsubs:
	@ $(RM) -r $(prefix)
	@ mkdir -p $(prefix)
	@ for f in $(SUBDIRS) ; do \
		(cd $$f ; echo In $$f; $(MAKE) -$(MAKEFLAGS) install ); \
	done

installups: ups
	install -d $(UPSDIR)
	install --mode=644 ups/*.table $(UPSDIR)

declare :
	eups declare --flavor $(UPS_FLAVOR) --root $(prefix) $(UPS_PRODUCT) $(UPS_VERSION)
current :
	eups declare --flavor $(UPS_FLAVOR) --current $(UPS_PRODUCT) $(UPS_VERSION)

.PHONY : tags
tags:
	etags `find . ! -name \*#\* \( -name \*.[ch] -o -name \*.py \) -print | bin/ignoreSwigFiles`
