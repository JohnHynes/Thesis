CC = g++
MAINCFLAGS = -std=c++2a -isystem glm -Wall -Wextra

.PHONY: all

all : main

main : main.cpp
	$(CC) main.cpp $(MAINCFLAGS) -o main

main.cpp : $(wildcard *.hpp)

image.png : image.ppm
	convert $< $@

image.ppm : main
	./$< > $@
