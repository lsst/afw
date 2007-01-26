include Makefile.global

CCDIRS = `find . -name Makefile -print | grep -v '\./Makefile' | sed -e 's/\/Makefile$$//' -e 's/^\.\///'`
PYMODS = `ls -F | grep /$$`
PYTHONFILES = `find . \( -name \*.py -o -name \*.pyc -o -name \*.$(SO) \) -print`
LSSTMOD = lsst
TOUCHMOD =

# comment this out if the top module is not "lsst"
TOUCHMOD = $(LSSTMOD)/__init__.py

.PHONY : all
all: build

.PHONY : build
build :: $(TOUCHMOD)

build clean distclean ::
	@for f in $(CCDIRS); do \
	    if [ ! -d $$f ]; then \
	        echo No such directory: $$f >&2; \
	    else \
	        if [ ! \( $$f = test -a $@ = all \) ]; then \
	            $(MAKE) -C $$f -$(MAKEFLAGS) $@; \
	        fi; \
	    fi \
	done

build :: 
	python -c 'import compileall; compileall.compile_dir(".")'

.PHONY : topClean
topClean :
	$(RM) -r TAGS 
	$(RM) core core.[0-9]*[0-9] *~

.PHONY : clean
clean :: topClean
	find . -name \*.pyc -exec rm '{}' \;

.PHONY : distclean
distclean:: clean
	$(RM) $(TOUCHMOD)

#
# Install things in their proper places as specified in etc/installdefs
#
.PHONY : install 
install : build 
	install -d $(PYTHONDIR)
	tar cf - $(PYTHONFILES) | (cd $(PYTHONDIR); tar xf -)

.PHONY : tags
tags:
	etags `find . ! -name \*#\* \( -name \*.[ch] -o -name \*.py \) -print | bin/ignoreSwigFiles`
