include("TwoDimSD.jl")
using .TwoDimSD, JLD, DelimitedFiles
using Plots
#loss = wrapper_SMM([0.07,0.5, 0.02, 0.9], [0.0,0.0,0.0,0.0])
sol = run_SMM([0.58, 2.0, 0.01, 0.3])
println(loss)