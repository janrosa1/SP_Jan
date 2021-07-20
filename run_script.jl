include("OneDimSD.jl")
using .OneDimSD #call the module with all the routines
run_spline(TD_assumed(),TD_calib(β = 0.85, δ_1 = -0.35,δ_2 =0.4403), TD_gird(n_B=30, n_y=21,B_min=-.01, B_max =1.5)) #solve and simulate the model, given the modified params