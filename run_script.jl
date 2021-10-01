
include("TwoDimSD.jl")
using .TwoDimSD, JLD, DelimitedFiles
using Plots


run_2D_calib_whole_exp(TD_assumed(),TD_calib(δ_1 = 0.58,δ_2 = 2.2,k = 0.18, pen = 0.011), TD_gird())

