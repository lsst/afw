include ../etc/Makefile.global

all :
	@: no default actions

install :
	@ mkdir -p $(UPSDIR)
	for f in *.table; do \
		eups_expandtable $$f $(UPSDIR); \
	done

clean:
	$(RM) *~ core*

distclean: clean
	@: nothing extra to do
