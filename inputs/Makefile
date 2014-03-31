all: basic.so

basic.o: basic.c basic.h

CFLAGS=-std=c99 -fPIC

%.so: %.o basic.o
	$(CC) -std=c99 -shared -fcolor-diagnostics $^ -o $@

clean:
	rm -f *.o *~ *.so

.PHONY: clean all
