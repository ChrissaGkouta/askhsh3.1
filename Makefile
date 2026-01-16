CC = mpicc
CFLAGS = -O3 -Wall
TARGET = poly_mult

all: $(TARGET)

$(TARGET): poly_mult.c
	$(CC) $(CFLAGS) -o $(TARGET) poly_mult.c

clean:
	rm -f $(TARGET)