#{The exchange rate regime, international bailouts and sovereign defaults

In this repository you can find code for my summer paper.

- ```TwoDimSD.jl``` is a module for a functions used to solve the model with flexible and fixed exchange rate regime. 
- ```OneDImSD.jl``` is a module for the models without IFI's debt (just for comparision not used in the paper)
To use the code, you can run the ```run script.jl```: it will start a routine for model calibration, simulation for float and peg and IRF
 computation for the last part of the paper (might take some time, about 2 hours on my machione using 16 threads)
 (or use OneDImSD or TwoDimSD  module in any way you like)

Two thigs to note:
- ```output``` contains all the plots (addistional value functions countour plots) and calibration results I used (sorry that it is messy)
- it is NOT a julia package: to run the code you need to install all the needed julia libraries by yourself