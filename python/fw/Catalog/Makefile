include ../etc/Makefile.global

#CFLAGS += $(FW_CFLAGS) -I../SDSS
SWIGFLAGS = $(shell echo $(FW_CFLAGS) | perl -pe 's/-pthread\s+//g')

.PHONY : install all clean distclean

.PHONY : all
all : check_swig _ctgread.$(SO)
	@: nothing to do
#
# The ctgread module
#
OBJS = ctgread_wrap.o

_ctgread.$(SO) : $(OBJS)
	$(LD) -o _ctgread.$(SO) $(SO_LDFLAGS) $(OBJS) $(FW_LIBS)
ctgread_wrap.o : ctgread_wrap.c
	$(CC) -c $(CFLAGS) ctgread_wrap.c

ctgread.py ctgread_wrap.c : ctgread.i
	$(SWIG) $(SWIGFLAGS) ctgread.i
#
# Update Makefile dependencies
#
-include .Makefile.depend
.PHONY : depend
depend : check_swig
	@echo Rebuilding make dependencies
	@$(CC) $(CFLAGS) -MM \
		`echo $(OBJS:.o=.c) | perl -ne 'print join(" ", grep(!/_wrap/,split(" ")))'` | \
			fixDepend FW > .Makefile.depend
	for f in *.i; do \
		$(SWIG) -MM -I.  -I$(XPA_DIR)/include $(SWIGFLAGS) $$f; \
		: $(SWIG) -MM $(PS_INCLUDES) -I../src -I$(PSPHOT_DIR)/include -I$(PSLIB_DIR)/include -I$(PSMODULES_DIR)/include $$f; \
	done | fixDepend FW >> .Makefile.depend
#
install : 
	@ mkdir -p $(PYTHONDIR)/Catalog
	cp *.[hi] *.py *.$(SO) $(PYTHONDIR)/Catalog
#
clean :
	$(RM) *.o _*.{so,dylib} *_wrap.c {ctgread}.py *.pyc *Rename.i
	$(RM) core core.[0-9]*[0-9] *~

distclean: clean
	@ $(RM) .Makefile.depend
