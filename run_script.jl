#include("OneDimSD.jl")
#using .OneDimSD #call the module with all the routines
include("TwoDimSD.jl")
using .TwoDimSD, JLD
using Plots
#twod_sol = load("val_func.jld")["model_sol"]
#run_spline(TD_assumed(),TD_calib(β = 0.85, δ_1 = -0.35,δ_2 =0.4403), TD_gird(n_B=25, n_y=21,B_min=-.1, B_max =1.5, itp = "spline_cub"), ERR ="peg") #solve and simulate the model, given the modified params
twod_sol = Solve_eq_peg2D(500)
simulation = simulate_eq_2D(twod_sol, ERR = "peg")


