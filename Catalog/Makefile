WCSTOOLSLIB = /home/mmiller/michelle/tools/wcstools-3.6.2/libwcs
PYTHONLIB = /usr/include/python2.4
INCLUDE = -I${WCSTOOLSLIB} -I${PYTHONLIB}
CFLAGS = ${INCLUDE}
LIB = -L${WCSTOOLSLIB}

all : 
	swig -python ctgread.i
	gcc -c -fpic $(CFLAGS) ctgread_wrap.c
	ld -shared ctgread_wrap.o $(LIB) -lwcs -lc -o _wcstools.so
