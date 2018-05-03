# Drift-diffusion-device-simulation
Pyhton algorithm for numerical simulation of heterostructure devices

This is a computational code I developed for the simulation of semiconductor heterojunctions, that is, structures made of different semiconductor materials stacked on top each other. Semiconductor heterojunctions are implemented in lot of devices we use daily, as for instance leds and solar cells. The importance of the simulation tools lies in their ability to predict the behavior of devices with specific characteristics, without the need of fabricating them. Obviously this fact conducts to saving valuable resources.

The implemented algorithm is based on the drift-diffusion approach, a set of equations which describes the microscopic operation of the semiconductor devices from a physics point of view. The algorithm solves the whole set of equations numerically, determining the electrostatic potential and carrier densities at each point within the one-dimensional geometry depicting the simulated structure.

The document "Simulation of semiconductor heterojunctions with Python" (in spanish) explains with more detail the model and the developed work, shows some results and exhibits a comparison between simulations and results extracted from the literature, obtained with similar approaches.
