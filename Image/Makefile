include ../etc/Makefile.global

.PHONY : all
all :
	@: nothing to do
#
install : 
	@ mkdir -p $(PYTHONDIR)/Image
	cp fits2fic.py $(BINDIR)
	cp *.py $(PYTHONDIR)/Image
#
clean :
	$(RM) *.pyc
	$(RM) core core.[0-9]*[0-9] *~

distclean: clean
	@ $(RM) .Makefile.depend
