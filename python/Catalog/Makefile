include ../build/Makefile.global

MODULE = PyCtgread.$(SO)
OBJS = extCtgread.o swigCtgread_wrap.o

SUPPORT_CFLAGS += -I$(WCSTOOLS_DIR)

.PHONY : all
all : $(MODULE)
	@: nothing to do
#
#
$(MODULE) : $(OBJS)
	$(LD) -o $(MODULE) $(SO_LDFLAGS) $(OBJS) $(SUPPORT_LIBS)
#
# Update Makefile dependencies
#
-include .Makefile.depend
.PHONY : depend
depend :
	@echo Rebuilding make dependencies
	@$(CC) $(CFLAGS) -MM $(OBJS:.o=.c) > .Makefile.depend
#
install : 
	@ mkdir -p $(PYTHONDIR)/PyWCSlib
	cp *.py *.$(SO) $(PYTHONDIR)/PyWCSlib
#
clean :
	$(RM) *.o _*.{so,dylib} *_wrap.c
	$(RM) core core.[0-9]*[0-9] *~

distclean: clean
	@ $(RM) .Makefile.depend
	$(RM) *.so *.pyc wcstools.py
