# Galactic to Atomic Zoom Demo

This project is a demonstration developed for the demo competition organized by the University of Paris 8. The competition challenges participants to create compelling demos within strict size limits without game engine (for the tracks I'm intrested in).
This demo features a zoom transition from a galactic scale to an atomic scale and was made using C and OpenGL.

## About the Competition

The University of Paris 8 hosts a demo competition with specific tracks that dictate the size constraints of the executable:
- **64 KB Track**: The executable must be less than 64 KB.
- **64 MB Track**: The executable can be up to 64 MB.

The goal of this demo is to participate in both the 64 KB and 64 MB tracks of the competition. More details about the competition and the other tracks can be found on the [official API8 website](http://api8.fr).

## Features

Current executable size: 39684 bytes (using UPX)

### Done

- Galaxy rendering
- Sun rendering
- Planet and water rendering
- Underwater particle effect
- 'Atmosphere' rendering
- Underwater rendering
- Jellyfish rendering

### TODO

- Cells, DNA and atoms rendering (if I have time)
- Music

## Building the Project

To build this project, you will need GCC for compiling and OpenGL libraries for graphics.

### Prerequisites

- A computer with Ubuntu or WSL2 installed
- GCC or any C compiler
- OpenGL library
- X11 libraries

### Compile and Run

1. Install the necessary libraries:
	```bash
	sudo apt-get update
	sudo apt-get install -y libx11-dev libgl1-mesa-dev zlib1g-dev
	```
	You will also need libmikmod found at https://sourceforge.net/projects/mikmod/files/libmikmod/3.3.11.1/libmikmod-3.3.11.1.tar.gz/download

2. Compile the project:
	```bash
	gcc -I./include ./**.c -lm -lX11 -lGL -lmikmod -o zoomDemo
	```

3. Run the executable:
	```bash
	./zoomDemo
	```

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Acknowledgements
- University of Paris 8 for organizing the demo competition.
- Stackoverflow for help on many topics.
- [GarrettGunnell](https://github.com/GarrettGunnell/Water) and [gasgiant](https://github.com/gasgiant/FFT-Ocean) for the FFT water implementation.
- ChatGPT for X11 setup and and various programming tasks.