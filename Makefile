CC = g++
MAINCFLAGS = -O3 -march=native -isystem glm -Wall -Wextra

.PHONY: all

all : main

main : main.cpp
	$(CC) main.cpp $(MAINCFLAGS) -o main

main.cpp : $(wildcard *.hpp)

image.png : image.ppm
	convert $< $@

image.ppm : main
	./$< > $@
