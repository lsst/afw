include build/installdefs

SHELL = /bin/sh
UPSDIR = ups
DIRS = python 
MAKEFLAGS = I $(PWD)/build

.PHONY : build
all build clean distclean : configure
	@for f in $(DIRS); do \
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
.PHONY : install declare installwarn installups installsubs
install: installwarn installsubs installups
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
	@ for f in $(DIRS) ; do \
		(cd $$f ; echo In $$f; $(MAKE) -$(MAKEFLAGS) install ); \
	done

installups: ups
	cp -rf ups $(prefix)

declare :
	eups declare --flavor $(UPS_FLAVOR) --root $(prefix) $(UPS_PRODUCT) $(UPS_VERSION)
current :
	eups declare --flavor $(UPS_FLAVOR) --current $(UPS_PRODUCT) $(UPS_VERSION)

.PHONY : tags
tags:
	etags `find . ! -name \*#\* \( -name \*.[ch] -o -name \*.py \) -print | bin/ignoreSwigFiles`
