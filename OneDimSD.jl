module OneDimSD
using Base: Float64
using LinearAlgebra, Interpolations, Random, QuantEcon, Statistics, DataFrames, Distributions, CSV, Plots, Optim, Roots, SchumakerSpline
using DelimitedFiles
using Parameters: @with_kw, @unpack
export TD_assumed, TD_calib, TD_gird, run_spline, unpack_params


# params initiasl vals 

global β = 0.85
global σ = 2.0
global α = 0.75
global h_bar = 1.0
global a = 0.26
global ζ = 0.5
global y_t = 1.0
global θ = 0.0385
global δ_1 = -0.35
global δ_2 = 0.46
global ρ = 0.93
global σ_μ = 0.037
global r =0.01
global n_y = 21
global n_B = 700
global B_min = -0.5
global B_max = 2.5
global grid_B = B_min:(B_max-B_min)/n_B:B_max
global P_stat, y_t_stat
global c_t_min = 1.0
global itp, disc
#############################################
#functions to modify params (I dont play with OOP so far, might start it later) 
#############################################

@with_kw struct TD_assumed #assumed paramters from TwinD;s paper
    σ::Float64 = 2.0 #CRRA param
    α::Float64 = 0.75  # production function param
    a::Float64 = 0.26 #CES utility of the tradeable good in the CES function of the final good
    ζ::Float64 = 0.5 #CES final good parameter
    y_t::Float64 = 1.0 # tradeable goods long term expectation
    ρ::Float64 = 0.9317 #tradeables shock persitanece (Argentinan economy from TwinD's)
    σ_μ::Float64 = 0.037 #tradeables shock variance (Argentinan economy from TwinD's)
    r::Float64 = 0.01 #interest rate 
    θ::Float64 = 0.0385 # probability of the reentry to the market after the default
    h_bar::Float64 = 1.0 #maximal employment level
end

@with_kw struct TD_calib #values which should be calibrated
    β::Float64 = 0.85 #discount factor
    δ_1::Float64 = -0.35 #penalty function param for the default (in form δ_1 y_t+ δ_2 y_t^2 ) 
    δ_2::Float64 = 0.44 #penalty function param for the default (in form δ_1 y_t+ δ_2 y_t^2 ) 
    w_bar::Float64 = 0.95* α * (1.0-a)/a #for the peg model calibration, minimal wage 
end

@with_kw struct TD_gird #technical params for the asset/income grid structure, discertization, interpolation method 
    n_y::Int64 = 21 #number of grid points for income, 21 is magical number used in all Arellano papers 
    n_B::Int64 = 20 #number of grid points for the Schumker splines, about 30 works for cubic splines, for the vfi, fior a good precision at least 200 points 
    B_min::Float64 = -0.01 #minimal debt level for (works for float)
    B_max::Float64 = 1.5 #maximal debt level
    itp::String = "spline_sch" # possible options are vanila VFI: "vfi", cubic splines "spline_cub" and Schumker splines "spline_sch"
    disc:: String = "tauchen" # also possible arguments are "rouwenhorst" and if n_y =200 the simualted Schmitt and Uribe values: "SU" (read from data)

end

function unpack_params(non_calib, calib, grid_p) #pretty ugly way to unpack params, probably code would be faster without globals but for now speed is not an issue (for float) 

    @unpack σ, α, a, ζ, y_t, ρ, σ_μ, r, θ,h_bar = non_calib
    @unpack β,δ_1,δ_2,w_bar = calib
    @unpack n_y, n_B, B_min, B_max, itp, disc  = grid_p
    global σ = σ
    global a = a
    global ζ = ζ
    global α = α
    global y_t = y_t
    global ρ = ρ
    global σ_μ = σ_μ
    global r = r
    global θ = θ
    global h_bar = h_bar
    global β = β
    global δ_1 = δ_1
    global δ_2 = δ_2
    global w_bar = w_bar
    global n_y = n_y
    global n_B = n_B
    global B_min = B_min
    global B_max = B_max
    global grid_B = B_min:(B_max-B_min)/(n_B-1):B_max
    global itp = itp
    global disc = disc
end


#############################################
#some useful functions
##############################################
function final_good(c_t, c_n) #final good aggregation
    return (a*c_t^(1.0-1.0/ζ)+(1.0-a)*c_n^(1.0-1.0/ζ))^(1.0/(1.0-1.0/ζ))
end

function utilityCRRA(c) #CRRA utility 
    return (c^(1.0-σ)-1.0)/(1.0-σ)
end

function marg_utilityCRRA(c, σ) #not used so far
    return c^(-σ)
end

function L(y_t) #penalty function for default
    return max(δ_1*y_t + δ_2*y_t^2.0,0.0)
end

function DiscretizeAR(ρ, var, n, method)
    #n:number of states
    #\rho , \sigma : values for  
    @assert n > 0 
    @assert var > 0.0 
    @assert ρ >= 0.0  
    @assert ρ < 1.0    

    #var= var*(1-ρ^2)^(1/2)
    if method == "rouwenhorst"
        MC = rouwenhorst(n, ρ,var)
        return(P = MC.p, ϵ = exp.(MC.state_values))
    elseif method == "tauchen"
        MC = tauchen(n, ρ, var)
        return(P = MC.p, ϵ = exp.(MC.state_values))
    elseif method == "SU" && n_y == 200
        P =  readdlm("P.csv", ',', Float64)
        ygrid =  readdlm("y.csv", ',', Float64)
        ϵ = exp.(ygrid)
        return (P=P, ϵ = ϵ )
    else
        throw(DomainError(method, "this method is not supported, try tauchen, rouwenhorst or SU if n_y ==200 "))
    end
end


function mc_sample_path(P; init = 11, sample_size = 11000) #Simulate the Markov chain for the simulation
    @assert size(P)[1] == size(P)[2] # square required
    N = size(P)[1] # should be square

    # create vector of discrete RVs for each row
    dists = [Categorical(P[i, :]) for i in 1:N]

    # setup the simulation
    X = fill(0, sample_size) # allocate memory, or zeros(Int64, sample_size)
    init_prob = fill(1/N, N)
    X[1] = rand(Categorical(init_prob)) # set the initial state

    for t in 2:sample_size
        dist = dists[X[t-1]] # get discrete RV from last state's transition distribution
        X[t] = rand(dist) # draw new value
    end
    SIM = convert(Array{Int64},X)
    return SIM
end


function bisection(f::Function, a::Number, b::Number;
    tol::AbstractFloat=1e-7, maxiter::Integer=100)
fa = f(a)
fa*f(b) <= 0 || error("No real root in [a,b]")
i = 0
local c
while b-a > tol
i += 1
i != maxiter || error("Max iteration exceeded")
c = (a+b)/2
fc = f(c)
if fc == 0
break
elseif fa*fc > 0
a = c  # Root is in the right half of [a,b].
fa = fc
else
b = c  # Root is in the left half of [a,b].
end
end
return c
end
#simulate if the country continue to be exclulded after default
function simulate_exclusion(θ)
    dists = Categorical([θ , 1.0- θ])
    sim = rand(dists, 1)
    return sim[1]
end

function interp_alg_unconst(splines_v, spline_q, state_y,state_b,b) 
    
    """
    interpolate the value of choosing debt b, given:
    value functions (splines_v),
    price function (spline_q),
    endowment: state_y
    current debt: state_b    
    """

    c_n = 1.0 #this is version for float so far
    q_b = max(spline_q(b), 0.0) #compute debt price
    if b<=0
        q_b =1.0/(1.0+r)
    end

    c_t = -state_b + y_t_stat[state_y] +q_b*b #compute tradeable consumption (check if it's >=0)
    if(c_t<=0)
        c_t = 1e-8
    end

    #compute the  consumption with b
    c = final_good(c_t, c_n)
    val =0.0
    val = copy(utilityCRRA(c))

    #compute the next period's values
    for j in 1:n_y
        v_p = splines_v[j]
        v_b = β*P_stat[state_y, j]*v_p(b)
        val = copy(val) + v_b
    end
    return -1.0*val #minimizing function, so need a negative value 
end

function interp_alg_const(splines_v, spline_q, state_y,state_b,b) 
    
    """
    interpolate the value of choosing debt b, given:
    value functions (splines_v),
    price function (spline_q),
    endowment: state_y
    current debt: state_b    
    """
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    q_b = max(spline_q(b), 0.0) #compute debt price
    if b<=0
        q_b =1.0/(1.0+r)
    end

    c_t = -state_b + y_t_stat[state_y] +q_b*b #compute tradeable consumption (check if it's >=0)
    if(c_t<=0)
        c_t = 1e-8
    end
    c_n = max(min(const_w*copy(c_t)^power,1.0),1e-7)
    #compute the  consumption with b
    c = final_good(c_t, c_n)
    val =0.0
    val = copy(utilityCRRA(c))

    #compute the next period's values
    for j in 1:n_y
        v_p = splines_v[j]
        v_b = β*P_stat[state_y, j]*v_p(b)
        val = copy(val) + v_b
    end
    return -1.0*val #minimizing function, so need a negative value 
end


#############################################################
#functions for one value function iterations
#############################################################
function Vf_update_TD_float(V_f, q) 
    """
    INPUTS
    V_f: current value function
    q: current price function, 

    OUTPUTS:
    V_final: value function after update
    Default_mat: default region matrix
    Policy: policy matrix (chosen debt index)
    """
    #allocate memory
    E_V_f_prim = zeros(n_y, n_B) #expected value next period for continuation
    Policy =  zeros(n_y, n_B) # policy function
    V_f_prim = zeros(n_y, n_B) #value for continuation
    C_t_vec = zeros(n_y, n_B) #possible consumption
    C_n_vec = ones(n_B) # value of nontradeable consumption (always one for the float)
    C_vec = zeros(n_y,n_B) 
    Default_mat = zeros(n_y, n_B) #default decsion


    #find index with no assets
    zero_ind = searchsortedfirst(grid_B, 0.0)
    
    #compute expected values next-period for continuation
   for j in 1:n_y
        for i in 1: n_B
            futur_val = 0
            for jj in 1: n_y
                futur_val = copy(futur_val)+P_stat[j,jj]*V_f[jj,i] 
            end
            E_V_f_prim[j,i] = futur_val
        end
    end
    
    #compute optimal savings rule for continuation
    
    #this part of teh code is the most heavy, can use some multithreading 
    Threads.@threads  for j in 1: n_y
         for i in 1:n_B
            
            for ii in 1:n_B
                C_t_vec[j,ii] = -grid_B[i] + y_t_stat[j] + q[j,ii]*grid_B[ii]
            end
            C_t_vec[j,:] = max.(copy(C_t_vec[j,:]), 1e-7)
            C_vec[j,:] = final_good.(C_t_vec[j, :],C_n_vec)
            V_f_prim[j,i], Policy[j,i] = findmax(utilityCRRA.(C_vec) + β*copy(E_V_f_prim[j,:]))
            
        end
    end
    for j in 1:n_y
        for i in 1: n_B
            futur_val = 0
            for jj in 1: n_y
                futur_val = copy(futur_val)+P_stat[j,jj]*V_f_prim[jj,zero_ind] 
            end
            E_V_f_prim_def[j] = futur_val
        end
    end

    #compute default value
    V_def = (I(n_y) - β*(1.0-θ)*P_stat)\(utilityCRRA.(final_good.(max.(y_t_stat-L.(y_t_stat),1e-9),ones(n_y))) + β*θ*copy(E_V_f_prim_def)) 

    #now choose the maximum between value of continuation and default
    V_final = copy(V_f_prim)
    
     for j in 1: n_y
        for i in 1:n_B
           if(V_f_prim[j,i]< V_def[j] && i> zero_ind) 
              Default_mat[j,i] = 1.0 
              V_final[j,i] = V_def[j]
            end
         end
    end
    return V_final, Default_mat, Policy
end

function Vf_update_TD_float_spline(V_f, q; spline_type = "spline_sch") 
    """
    INPUTS
    V_f: current value function
    q: current price function, 

    OUTPUTS:
    V_final: value function after update
    Default_mat: default region matrix
    Policy: policy matrix (chosen debt index)
    
    OPTIONAL:

    spline_type: choose cubic or schumker splines
    """


    E_V_f_prim = zeros(n_y) #expected value next period for continuation
    E_V_f_prim_default  =  zeros(n_y) #expected value next period for default
    V_final = zeros(n_y,n_B) #final value function after update
    V_f_prim = zeros(n_y, n_B) #value for continuation
    V_def = zeros(n_y) #value of default
    Default_mat = zeros(n_y, n_B) #default decsion
    Dafault_border = zeros(n_y) #exact location when the default occurs
    Policy_num = zeros(n_y,n_B) #next period policies 

    #choose the spline type, define the value function interpolations matrix for each endomwnet schock
    if(spline_type == "spline_cub")
        V_f_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        q_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = CubicSplineInterpolation(grid_B, V_f[j,:],extrapolation_bc = Line())
            q_spline[j] = LinearInterpolation(grid_B, q[j,:]) #spline works very bad for the price function (as they are non-monotonic)
        end
    else
        V_f_spline = Array{Schumaker{Float64}}(undef,n_y)
        q_spline = Array{Schumaker{Float64}}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = Schumaker(collect(grid_B), V_f[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
            q_spline[j] = Schumaker(collect(grid_B), q[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear)) #schumker splines preserves monotonicity
        end

    end

    # Given value functions and price functions find updated value function, this step can be done muchg faster but this more stable
    for j in 1:n_y
        for i in 1:n_B
            interp_alg_unconst_1(x) = deepcopy(interp_alg_unconst(copy(V_f_spline), q_spline[j], j ,grid_B[i],x)) #define the value of choosing the debt x as function of 1 variable

            next_b_sol = optimize(interp_alg_unconst_1,grid_B[1], grid_B[n_B], GoldenSection())#find the optimal value, Golden section is slower but more stable than Brent method 
            Optim.converged(next_b_sol) || error("Failed to converge in $(iterations(result)) iterations")
            V_f_prim[j,i] = max(-1.0*deepcopy(Optim.minimum(next_b_sol)), -1000) #find new value of continuation 
            Policy_num[j,i] = deepcopy(Optim.minimizer(next_b_sol)[1]) #find policy if continuation 
        end
    end
    
    #compute the expectation of V_f for reentring the international market after default
     for j in 1:n_y
         futur_val = copy(0.0)
         E_V_f_prim_default[j] = copy(V_f_spline[j](0.0))
         for jj in 1: n_y
             futur_val = copy(futur_val)+P_stat[j,jj]*V_f_spline[jj](0.0) 
         end
         E_V_f_prim[j] = copy(futur_val)
     end

    #compute default value
    V_def = (I(n_y) - β*(1.0-θ)*P_stat)\(utilityCRRA.(final_good.(max.(y_t_stat-L.(y_t_stat),1e-6),ones(n_y))) .+ β*θ*copy(E_V_f_prim)) 

    #Choose over default and continuation 
    V_final  = V_f_prim
    for j in 1: n_y
        #interpolate continuation value
        if(spline_type == "spline_cub")
            V_prim_itp = CubicSplineInterpolation(grid_B, V_f_prim[j,:].- V_def[j],extrapolation_bc = Line()) 
        else
            V_prim_itp = Schumaker(collect(grid_B), V_f_prim[j,:].- V_def[j]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
        end

        if(V_f_prim[j,n_B]<V_def[j])
            Dafault_border[j] = find_zero(V_prim_itp, (grid_B[1],grid_B[n_B] ),Roots.Brent() )
        else
            Dafault_border[j] = grid_B[n_B]+0.2 #never will be chosen
        end
        for i in 1:n_B #small mistaekes in Brent methd can make convergence impossible on some inital iterations, so I compute this independly, though it should get to the same result
           if(V_f_prim[j,i]< V_def[j] && grid_B[i]> 0.0) 
              Default_mat[j,i] = 1.0 
              V_final[j,i] = V_def[j]
            end
         end
    end
    return V_final, Default_mat,Policy_num,Dafault_border
end


function Vf_update_TD_peg_spline(V_f, q; spline_type = "spline_sch") 
    """
    INPUTS
    V_f: current value function
    q: current price function, 

    OUTPUTS:
    V_final: value function after update
    Default_mat: default region matrix
    Policy: policy matrix (chosen debt index)
    
    OPTIONAL:

    spline_type: choose cubic or schumker splines
    """


    E_V_f_prim = zeros(n_y) #expected value next period for continuation
    E_V_f_prim_default  =  zeros(n_y) #expected value next period for default
    V_final = zeros(n_y,n_B) #final value function after update
    V_f_prim = zeros(n_y, n_B) #value for continuation
    V_def = zeros(n_y) #value of default
    Default_mat = zeros(n_y, n_B) #default decsion
    Dafault_border = zeros(n_y) #exact location when the default occurs
    Policy_num = zeros(n_y,n_B) #next period policies 
    C_n_vec_def = ones(n_y) # vector of tradeable consumption in 


    #choose the spline type, define the value function interpolations matrix for each endomwnet schock
    if(spline_type == "spline_cub")
        V_f_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        q_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = CubicSplineInterpolation(grid_B, V_f[j,:],extrapolation_bc = Line())
            q_spline[j] = LinearInterpolation(grid_B, q[j,:]) #spline works very bad for the price function (as they are non-monotonic)
        end
    else
        V_f_spline = Array{Schumaker{Float64}}(undef,n_y)
        q_spline = Array{Schumaker{Float64}}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = Schumaker(collect(grid_B), V_f[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
            q_spline[j] = Schumaker(collect(grid_B), q[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear)) #schumker splines preserves monotonicity
        end

    end

    # Given value functions and price functions find updated value function, this step can be done muchg faster but this more stable
    for j in 1:n_y
        sol_max_qB = optimize(x-> -q_spline[j](x)*x,grid_B[1], grid_B[n_B], GoldenSection())
        max_qB = Optim.minimizer(sol_max_qB)[1]

        for i in 1:n_B
            if(y_t_stat[j]-grid_B[i]+ q_spline[j](grid_B[1])*grid_B[1]<c_t_min) && (y_t_stat[j]-grid_B[i]+ q_spline[j](max_qB)*max_qB>c_t_min)
                #println((y_t_stat[j]-grid_B[i]+ q_spline[j](grid_B[1])*grid_B[1]-c_t_min)*(y_t_stat[j]-grid_B[i]+ q_spline[j](max_qB)*max_qB-c_t_min))
                b_bord = copy(find_zero(x->y_t_stat[j]-grid_B[i]+ q_spline[j](x)*x - c_t_min, (grid_B[1],max_qB)))
                #b_bord = bisection(x->y_t_stat[j]-grid_B[i]+ q_spline[j](x)*x - c_t_min,grid_B[1],  max_qB)
                interp_alg_unconst_1(x) = deepcopy(interp_alg_unconst(copy(V_f_spline), q_spline[j], j ,grid_B[i],x))
                interp_alg_const_1(x) = deepcopy(interp_alg_const(copy(V_f_spline), q_spline[j], j ,grid_B[i],x))
                    
                next_b_sol_unconst = optimize(interp_alg_unconst_1,b_bord, grid_B[n_B], GoldenSection())#find the optimal value, Golden section is slower but more stable than Brent method 
                Optim.converged(next_b_sol_unconst) || error("Failed to converge in $(iterations(result)) iterations")
                next_b_sol_const = optimize(interp_alg_const_1,grid_B[1],b_bord, GoldenSection())#find the optimal value, Golden section is slower but more stable than Brent method 
                Optim.converged(next_b_sol_const) || error("Failed to converge in $(iterations(result)) iterations")
                if(Optim.minimum(next_b_sol_unconst)< Optim.minimum(next_b_sol_const))
                        V_f_prim[j,i] = max(-1.0*deepcopy(Optim.minimum(next_b_sol_unconst)), -1000)
                        Policy_num[j,i] = deepcopy(Optim.minimizer(next_b_sol_unconst)[1])
                else
                        V_f_prim[j,i] = max(-1.0*deepcopy(Optim.minimum(next_b_sol_const)), -1000)
                        Policy_num[j,i] = deepcopy(Optim.minimizer(next_b_sol_const)[1])
                end
            elseif y_t_stat[j]-grid_B[i]+ q_spline[j](max_qB)*max_qB<c_t_min
                interp_alg_const_2(x) = deepcopy(interp_alg_const(copy(V_f_spline), q_spline[j], j ,grid_B[i],x))
                next_b_sol_const = optimize(interp_alg_const_2,grid_B[1],grid_B[n_B], GoldenSection())#find the optimal value, Golden section is slower but more stable than Brent method 
                Optim.converged(next_b_sol_const) || error("Failed to converge in $(iterations(result)) iterations")
                V_f_prim[j,i] = max(-1.0*deepcopy(Optim.minimum(next_b_sol_const)), -1000)
                Policy_num[j,i] = deepcopy(Optim.minimizer(next_b_sol_const)[1])
    
            else
                interp_alg_unconst_3(x) = deepcopy(interp_alg_unconst(copy(V_f_spline), q_spline[j], j ,grid_B[i],x)) #define the value of choosing the debt x as function of 1 variable
                next_b_sol = optimize(interp_alg_unconst_3,grid_B[1], grid_B[n_B], GoldenSection())#find the optimal value, Golden section is slower but more stable than Brent method 
                Optim.converged(next_b_sol) || error("Failed to converge in $(iterations(result)) iterations")
                V_f_prim[j,i] = max(-1.0*deepcopy(Optim.minimum(next_b_sol)), -1000) #find new value of continuation 
                Policy_num[j,i] = deepcopy(Optim.minimizer(next_b_sol)[1]) #find policy if continuation 
            end

            
            
        end
    end
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    #compute the expectation of V_f for reentring the international market after default
     for j in 1:n_y
         futur_val = copy(0.0)
         E_V_f_prim_default[j] = copy(V_f_spline[j](0.0))
         for jj in 1: n_y
             futur_val = copy(futur_val)+P_stat[j,jj]*V_f_spline[jj](0.0) 
         end
         E_V_f_prim[j] = copy(futur_val)
     end
    
     for j in 1:n_y
        if(y_t_stat[j]-L(y_t_stat[j]) <c_t_min)
            C_n_vec_def[j] = min(copy(y_t_stat[j]-L.(y_t_stat[j]))^(power)*const_w,1.0)
        end
    end
    #compute default value
    V_def = (I(n_y) - β*(1.0-θ)*P_stat)\(utilityCRRA.(final_good.(max.(y_t_stat-L.(y_t_stat),1e-8),C_n_vec_def)) + β*θ*copy(E_V_f_prim)) 
    #Choose over default and continuation 
    V_final  = V_f_prim
    for j in 1: n_y
        #interpolate continuation value
        if(spline_type == "spline_cub")
            V_prim_itp = CubicSplineInterpolation(grid_B, V_f_prim[j,:].- V_def[j],extrapolation_bc = Line()) 
        else
            V_prim_itp = Schumaker(collect(grid_B), V_f_prim[j,:].- V_def[j]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
        end

        if(V_f_prim[j,n_B]<V_def[j])
            Dafault_border[j] = find_zero(V_prim_itp, (grid_B[1],grid_B[n_B] ),Roots.Brent() )
        else
            Dafault_border[j] = grid_B[n_B]+0.2 #never will be chosen
        end
        for i in 1:n_B #small mistaekes in Brent methd can make convergence impossible on some inital iterations, so I compute this independly, though it should get to the same result
           if(V_f_prim[j,i]< V_def[j] && grid_B[i]> 0.0) 
              Default_mat[j,i] = 1.0 
              V_final[j,i] = V_def[j]
            end
         end
    end
    return V_final, Default_mat,Policy_num,Dafault_border
end

function Solve_Bellman_float(n; method = "spline_sch")
    """
    solve Bellman equation for float, given maximal number of iterations n 
    """
    V_f = zeros(n_y,n_B)
    q = 1.0/(1.0+r)*ones(n_y, n_B)
    for i in 1:n_B
        V_f[:,i] = utilityCRRA.(final_good.(max.(y_t_stat,1e-7),ones(n_y)))
    end
    #check size of the asset and income grids 
    δ = zeros(n_y, n_B) #default probability
    q_new = ones(n_y, n_B) #new q price matrix
    #Allocate memory
    V_f_new  = zeros(n_y, n_B) #
    V_f_def  = zeros(n_y) #
    V_f_def_new = zeros(n_y)
    Default_mat = zeros(n_y, n_B)
    Policy = zeros(n_y, n_B)
    Policy_index = zeros(n_y, n_B)
    default_border = zeros(Int64, n_y)
    Policy_c_n = ones(n_y,n_B)   
    #initialize iterations
    iter = 0
    
    
    while((maximum(abs.(V_f_new-V_f).+abs.(V_f_def-V_f_def_new) )>=1e-6 || maximum(abs.(q_new-q))>=1e-8)&& iter <n ) #for now I want the solution to converge with 1e-6  
        
        if(iter >=1)
         V_f = copy(V_f_new) 
         q = copy(q_new)
         V_f_def = copy(V_f_def_new)
        end
        if method == "vfi" 
            V_f_new, Default_mat, Policy_index = Vf_update_TD_float( V_f, q)
        else 
            V_f_new, Default_mat, Policy, default_border = Vf_update_TD_float_spline( V_f, q, spline_type = method)
        end
         #compute default probability
        for j in 1:n_y
            for i in 1:n_B
                default_prob = 0.0
                for jj in 1:n_y
                    default_prob = copy(default_prob) + P_stat[j,jj]*Default_mat[jj,i]
                end
                δ[j,i] = copy(default_prob)
                        
                
                
            end
        end
        
        #update the q
        q_new = 0.99*(1.0/(1.0+r)*(1.0.-copy(δ)))+0.01*copy(q)
        iter = iter+1
        if(mod(iter, 10)==0)
            println("Iteration: ", iter)
            println("V_f conv: ",maximum(abs.(V_f_new-V_f)))   
            println("q conv: ",maximum(abs.(copy(q).-copy(q_new))))   
        end
    end
    
    println("ITERATION ends at: ")
    println("Iteration: ", iter)
    println("V_f conv: ",maximum(abs.(V_f_new-V_f)))   
    println("q conv: ",maximum(abs.(copy(q).-copy(q_new)))) 

    if method =="vfi"
        for j in 1:n_y
            for i in 1:n_B
                Policy[j,i] = grid_B[Policy_index[j,i]]
            end
            default_border[j] = searchsortedfirst(Default_mat[j,:],1)
        end
    end


    return (q, Default_mat, Policy, Policy_c_n, V_f, default_border)
end


function Solve_Bellman_peg(n; method = "spline_sch")
    """
    solve Bellman equation for float, given maximal number of iterations n 
    """
    global c_t_min = (copy(w_bar)/(copy(α)*(1-copy(a))/copy(a)))^(ζ)

    V_f = zeros(n_y,n_B)
    q = 1.0/(1.0+r)*ones(n_y, n_B)
    for i in 1:n_B
        V_f[:,i] = utilityCRRA.(final_good.(max.(y_t_stat,1e-7),ones(n_y)))
    end
    #check size of the asset and income grids 
    δ = zeros(n_y, n_B) #default probability
    q_new = ones(n_y, n_B) #new q price matrix
    #Allocate memory
    V_f_new  = zeros(n_y, n_B) #
    V_f_def  = zeros(n_y) #
    V_f_def_new = zeros(n_y)
    Default_mat = zeros(n_y, n_B)
    Policy = zeros(n_y, n_B)
    Policy_index = zeros(n_y, n_B)
    default_border = zeros(Int64, n_y)
    Policy_c_n = ones(n_y,n_B)   
    #initialize iterations
    iter = 0
    
    
    while((maximum(abs.(V_f_new-V_f).+abs.(V_f_def-V_f_def_new) )>=1e-6 || maximum(abs.(q_new-q))>=1e-8)&& iter <n ) #for now I want the solution to converge with 1e-6  
        
        if(iter >=1)
         V_f = copy(V_f_new) 
         q = copy(q_new)
         V_f_def = copy(V_f_def_new)
        end
        if method == "vfi" 
            V_f_new, Default_mat, Policy_index = Vf_update_TD_float( V_f, q)
        else 
            V_f_new, Default_mat, Policy, default_border = Vf_update_TD_peg_spline( V_f, q, spline_type = method)
        end
         #compute default probability
        for j in 1:n_y
            for i in 1:n_B
                default_prob = 0.0
                for jj in 1:n_y
                    default_prob = copy(default_prob) + P_stat[j,jj]*Default_mat[jj,i]
                end
                δ[j,i] = copy(default_prob)
                        
                
                
            end
        end
        
        #update the q
        q_new = 0.99*(1.0/(1.0+r)*(1.0.-copy(δ)))+0.01*copy(q)
        iter = iter+1
        if(mod(iter, 10)==0)
            println("Iteration: ", iter)
            println("V_f conv: ",maximum(abs.(V_f_new-V_f)))   
            println("q conv: ",maximum(abs.(copy(q).-copy(q_new))))   
        end
    end

    println("ITERATION ends at: ")
    println("Iteration: ", iter)
    println("V_f conv: ",maximum(abs.(V_f_new-V_f)))   
    println("q conv: ",maximum(abs.(copy(q).-copy(q_new)))) 


    if method =="vfi"
        for j in 1:n_y
            for i in 1:n_B
                Policy[j,i] = grid_B[Policy_index[j,i]]
            end
            default_border[j] = searchsortedfirst(Default_mat[j,:],1)
        end
    end


    return (q, Default_mat, Policy, Policy_c_n, V_f, default_border)
end
########################################################
## SImulate equilibrium
########################################################
"""
Simulate the SD model, given the policy functions
INPUTS: (instructure of Model solution and model params)
     
P: stochatsic matrix ofthe Markov Chain
ϵ: values of the income schock for the given Markov Chain
grid: gird for assets
r: interest rate
θ: paramter for the exclusion form the finacial arket after default
grid: gird for assets 
Def_mat: default region
q: price matrix
Policy: savings policy function
Params:
n_sim: number of independent simulatioins
t_sim: size of each simulation
burnout: number of first obervations to ignore 
"""
function simulate_TD( Model_solution, w_bar; burnout = 10000, t_sim =1010000, n_sim = 10, ERR="float", method = "spline_sch")
    #unpack policy functions and paramters

    q = Model_solution[1]
    Def_mat = Model_solution[2]
    Policy = Model_solution[3]
    Policy_c_n  = Model_solution[4]
    Def_bord = Model_solution[6]
    #allocate memory
    
    SIM = zeros(Int64, n_sim, t_sim) #simulated income process
    A = ones(n_sim, t_sim) #assets history
    Y_t = zeros(n_sim, t_sim) # output history
    Trade_B = zeros(n_sim, t_sim) # trade balance history
    
    C_t = zeros(n_sim, t_sim) # consumption history tradeables
    C_n = ones(n_sim, t_sim) # non-tradeables
    h_t = ones(n_sim, t_sim)
    C = zeros(n_sim, t_sim) # final good
    
    D = zeros(n_sim, t_sim) #Defaults histiry
    D_state = zeros(n_sim, t_sim) #exclusion from the financial market histiry
    R = zeros(n_sim, t_sim) #q hiatory
    
    #exchange rate and minimal wage
   
    ϵ = ones(n_sim, t_sim)
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
   
    #interpolate policy functions
    if method == "spline_cub"
        Policy_fun = Array{Interpolations.Extrapolation}(undef,n_y)
        q_func = Array{Interpolations.Extrapolation}(undef,n_y)
        for j in 1:n_y
            Policy_fun[j] = CubicSplineInterpolation(grid_B, Policy[j,:],extrapolation_bc = Line())
            q_func[j] = LinearInterpolation(grid_B, q[j,:])
           
        end
    elseif method == "spline_sch" 
        Policy_fun = Array{Schumaker{Float64}}(undef,n_y)
        q_func = Array{Schumaker{Float64}}(undef,n_y)
        for j in 1:n_y
            Policy_fun[j] = Schumaker(collect(grid_B), Policy[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
            q_func[j] = Schumaker(grid_B, q[j,:])
            
        end
    else 
        Policy_fun = Array{Interpolations.Extrapolation}(undef,n_y)
        q_func = Array{Interpolations.Extrapolation}(undef,n_y)
        for j in 1:n_y
            Policy_fun[j] = LinearInterpolation(grid_B, Policy[j,:],extrapolation_bc = Line())
            q_func[j] = LinearInterpolation(grid_B, q[j,:])
            
        end
    end
    

    #define starting values
    y_0 = Int(floor(n_y/2) ) # start with 11th state
    Y_0 = ϵ[y_0]*ones(n_sim) #output start value
    A_0 = zeros(n_sim) #start with 0 assets
    
    Y_t[:,1] = Y_0 
    A[:,1] = A_0

    Default_flag = 0.0 # flag if the country was excluded from financial market in the previous period
    #simulate income process
    for i in 1:n_sim
        SIM[i,:] = mc_sample_path(P_stat,init = Int(floor(n_y/2) ), sample_size = t_sim)
    end
    
    #now use simulated income process to find trade balance, output, assets and q history
    for i in 1:n_sim
        for t in 2:t_sim
            
            #case when country was excluded from financial market in the previous period
           if(Default_flag==1.0) 
                simi = simulate_exclusion(θ)
                if(simi==1)
                    Default_flag = 0.0
                else
                    Default_flag = 1.0
                end
                if(Default_flag==1.0)
                    Y_t[i,t] = y_t_stat[SIM[i,t]]- L(y_t_stat[SIM[i,t]])
                    C_t[i,t] = Y_t[i,t] 
                    if C_t[i,t] < c_t_min && ERR =="peg"
                        C_n[i,t] =  min(copy(C_t[i,t])^(power)*const_w,1.0)
                        h_t[i,t] = C_n[i,t]^(1.0/α)
                    else
                        C_n[i,t] = 1.0
                    end

                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    A[i,t] = 0.0
                    R[i,t] = 0.0 #nop interest rate
                    D[i,t] = 0.0
                    D_state[i,t] = 1.0
                else
                    
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    A[i,t] = Policy_fun[SIM[i,t]](0.0)
                    C_t[i,t] = Y_t[i,t] - A[i,t-1] + q_func[SIM[i,t]](0.0)*A[i,t]
                    if C_t[i,t] < c_t_min && ERR =="peg"
                        C_n[i,t] =  min(copy(C_t[i,t])^(power)*const_w,1.0)
                        h_t[i,t] = C_n[i,t]^(1.0/α)
                    else
                        C_n[i,t] = 1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 
                    R[i,t] = 1.0/q_func[SIM[i,t]](0.0) - 1.0 - r
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                    
                end
            else
                if Def_bord[SIM[i,t]] <grid_B[n_B] && A[i,t-1] > Def_bord[SIM[i,t]]  #case of default
                    Y_t[i,t] = y_t_stat[SIM[i,t]]- L(y_t_stat[SIM[i,t]])
                    C_t[i,t] = Y_t[i,t]
                    C[i,t] = final_good(C_t[i,t],1.0) 
                    if C_t[i,t] < c_t_min && ERR =="peg"
                        C_n[i,t] =  min(copy(C_t[i,t])^(power)*const_w,1.0)
                        h_t[i,t] = C_n[i,t]^(1.0/α)
                    else
                        C_n[i,t] = 1.0
                    end
                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    A[i,t] = 0.0
                    R[i,t] = 0 #1.0/q[SIM[i,t],A_int[i,t]] - 1.0
                    D[i,t] = 1.0
                    D_state[i,t] = 1.0
                    Default_flag = 1.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                else
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    A[i,t] = Policy_fun[SIM[i,t]](A[i,t-1])
                    C_t[i,t] = Y_t[i,t] - A[i,t-1] + q_func[SIM[i,t]](A[i,t])*A[i,t]
                    if C_t[i,t] < c_t_min && ERR =="peg"
                        C_n[i,t] =  min(copy(C_t[i,t])^(power)*const_w,1.0)
                        h_t[i,t] = C_n[i,t]^(1.0/α)
                    else
                        C_n[i,t] = 1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 
                    R[i,t] = 1.0/q_func[SIM[i,t]](A[i,t]) - 1.0 -r
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                end
           end
           if α * (C_t[i,t])^(1.0/ζ)* (1.0-a)/a < w_bar && ERR == "float"
                ϵ[i,t] = w_bar/((α * (C_t[i,t])^(1.0/ζ))* (1.0-a)/a)
           end
        end
    end

    #compute stats for defaults
    n_defaults = sum(D[:, burnout:t_sim])
    println(n_defaults)
    non_defaults = D_state[:,burnout:t_sim].<1.0   
    println(sum(non_defaults))
    #default probability
    Def_prob  =  1.0 - (1.0-n_defaults/(sum(non_defaults)))^4.0   


    #stats after default:
    Y_ab = Y_t[:,burnout:t_sim]
    R_ab = R[:,burnout:t_sim]
    A_ab = A[:,burnout:t_sim]
    D_ab = D[:,burnout:t_sim]
    ϵ_ab = ϵ[:,burnout:t_sim]
    Trade_B_ab = Trade_B[:,burnout:t_sim]
    D_state_ab = D_state[:,burnout:t_sim]
    C_ab_t = C_t[:,burnout:t_sim]
    C_ab_n = C_n[:,burnout:t_sim]
    h_t_ab = h_t[:,burnout:t_sim]
    #choose number of defaults to compute statistics 
    n_chosen_def = 10000
    #allocate memory for the staoistics 
    pre_def_stats_Y_t = zeros(n_chosen_def , 90)
    pre_def_stats_R = zeros(n_chosen_def , 90)        
    pre_def_stats_TB = zeros(n_chosen_def , 90)        
    pre_def_stats_C_t = zeros(n_chosen_def , 90) 
    pre_def_stats_C_n = zeros(n_chosen_def , 90) 
    pre_def_stats_B = zeros(n_chosen_def , 90)  
    pre_def_stats_NB = zeros(n_chosen_def , 90)  
    pre_def_stats_P = zeros(n_chosen_def , 90)  
    pre_def_stats_D = zeros(n_chosen_def , 90) 
    pre_def_stats_h = zeros(n_chosen_def , 90) 
    stats = zeros(9, 90)
    ϵ_stats = zeros(n_chosen_def , 90)  
    
    
    
    
    #compute statistics for 74 periods before the default, for n_def defaults (without any defults in the 74 periods before)
    iter = 0
    for i in 1:n_sim, t in 76:t_sim-burnout
            if(D_ab[i,t]==1 && sum(D_state_ab[i,t-25:t-1 ])== 0.0 && t<=t_sim-burnout-100 ) #if default happen
                 iter =iter+1   
                 pre_def_stats_Y_t[iter,:] = Y_ab[i, t-74:t+15]
                 pre_def_stats_R[iter,:] = (1.0.+R_ab[i, t-74:t+15]).^4.0.- (1.0+r).^4.0
                 pre_def_stats_B[iter,:] = A_ab[i, t-75:t+14]
                 pre_def_stats_NB[iter,:] = A_ab[i, t-74:t+15] #new debt
                 pre_def_stats_TB[iter,:] = Trade_B_ab[i, t-74:t+15]
                 pre_def_stats_C_t[iter,:] = C_ab_t[i, t-74:t+15]
                 pre_def_stats_C_n[iter,:] = C_ab_n[i, t-74:t+15]
                 pre_def_stats_P[iter,:] = (1-a)/a*(pre_def_stats_C_t[iter,:]./pre_def_stats_C_n[iter,:])
                 ϵ_stats[iter, :] = ϵ_ab[i, t-74:t+15] 
                 pre_def_stats_h[iter,:] = h_t_ab[i, t-74:t+15] 
                 pre_def_stats_D[iter,:] =  D_state[i, t-74:t+15]
                
            end 
            if(iter == n_chosen_def)
                
                break
            end
    end
    

   #calibration targets 
    debt_tradeables = mean(pre_def_stats_B[50:74]./pre_def_stats_Y_t[50:74])
    
    stats[1,:] = median(pre_def_stats_Y_t, dims =  1)
    stats[2,:] = median(pre_def_stats_R, dims = 1)
    stats[3,:] = median(pre_def_stats_B, dims = 1)
    stats[4,:] = median(pre_def_stats_C_t, dims = 1)
    stats[5,:] = median(ϵ_stats, dims = 1)
    stats[6,:] = median(pre_def_stats_C_n, dims = 1)
    
    stats[7,:] = median(pre_def_stats_P, dims = 1)
    stats[8,:] = mean(pre_def_stats_D, dims = 1)
    stats[9,:] = median(pre_def_stats_h, dims = 1)
    output_loss = mean((pre_def_stats_Y_t[:,75].- pre_def_stats_Y_t[:,74])./(pre_def_stats_Y_t[:,74]))
    println("calibration results: ", " default probability: ", Def_prob)       
    println("prod_loss: " , output_loss)      
    println("debt to tradeable: " , debt_tradeables)      
    println("mean spread: " , mean(stats[2,1:74]))      
    println("std spread: ", mean(std(pre_def_stats_R[1:74], dims=1))  )     
    return (Def_prob, debt_tradeables, output_loss, stats)
end
    
#########################################################################
#Ploting and saving equilibrium
#########################################################################
function plot_solution(Model_solution)
    V_f = Model_solution[5]
    q = Model_solution[1]
    Policy = Model_solution[3]

    grid_fig = B_min:0.0001:B_max-0.04
    V_values = zeros(n_y, length(grid_fig))
    q_values = zeros(n_y, length(grid_fig))
    policy_values = zeros(n_y, length(grid_fig))
    #define solution for many 
    
    if(method == "spline_cub")
        V_f_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        q_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        Policy_fun = Array{Interpolations.Extrapolation}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = CubicSplineInterpolation(grid_B, V_f[j,:])
            Policy_fun[j] = CubicSplineInterpolation(grid_B, Policy[j,:],extrapolation_bc = Line())
            q_spline[j] = LinearInterpolation(grid_B, q[j,:])
            V_values[j,:] = V_f_spline[j].(grid_fig)
            q_values[j,:] = q_spline[j].(grid_fig)
            policy_values[j,:] =  Policy_fun[j].(grid_fig)
        end
    elseif(method == "spline_sch")
        V_f_spline = Array{Schumaker{Float64}}(undef,n_y)
        q_spline = Array{Schumaker{Float64}}(undef,n_y)
        Policy_fun = Array{Schumaker{Float64}}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = Schumaker(grid_B, V_f[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
            Policy_fun[j] = Schumaker(grid_B, Policy[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
            q_spline[j] = Schumaker(grid_B, q[j,:]; extrapolation = (SchumakerSpline.Linear, SchumakerSpline.Linear))
            V_values[j,:] = V_f_spline[j].(grid_fig)
            q_values[j,:] = q_spline[j].(grid_fig)
            policy_values[j,:] =  Policy_fun[j].(grid_fig)
        end
    else
        V_f_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        q_spline = Array{Interpolations.Extrapolation}(undef,n_y)
        Policy_fun = Array{Interpolations.Extrapolation}(undef,n_y)
        for j in 1:n_y
            V_f_spline[j] = LinearInterpolation(grid_B, V_f[j,:])
            Policy_fun[j] = LinearInterpolation(grid_B, Policy[j,:],extrapolation_bc = Line())
            q_spline[j] = LinearInterpolation(grid_B, q[j,:])
            V_values[j,:] = V_f_spline[j].(grid_fig)
            q_values[j,:] = q_spline[j].(grid_fig)
            policy_values[j,:] =  Policy_fun[j].(grid_fig)
        end

    end
    #plot value, price an policy functions
    plot(grid_fig, V_values[1,:], label = string("y_t: ", round(y_t_stat[1],digits=2)), xlabel = "B", title = "value function")
    for j in 2:n_y
        plot!(grid_fig, V_values[j,:], label =  string("y_t: ", round(y_t_stat[j],digits=2)))
    end
    png("output/val_functions")

    plot(grid_fig, q_values[1,:],label =  string("y_t: ",round(y_t_stat[1], digits=2)), xlabel = "B", title = "debt price function")
    for j in 2:n_y
        plot!(grid_fig, q_values[j,:],label = string("y_t: ", round(y_t_stat[j],digits=2)))
    end
    png("output/q_functions")

    plot(grid_fig, policy_values[1,:],label =  string("y_t: ", round(y_t_stat[1],digits=2)), xlabel = "B", title = "policy function")
    for j in 2:n_y
        plot!(grid_fig, policy_values[j,:],label = string("y_t: ", round(y_t_stat[j],digits=2)))
    end
    png("output/policy_functions")

    


    writedlm( "output/val_f_spline.csv",  Model_solution[5], ',')
    writedlm( "output/q_spline.csv",  Model_solution[1], ',')
    writedlm( "output/policy_spline.csv",  Model_solution[3], ',')
    
end

function plot_simulation(stats)

    p1 = plot(-15.0:1.0:14.0, stats[3,60:89], ylabel = "B'", xlabel = "T")
    p2 = plot(-15.0:1.0:14.0,stats[1,60:89], ylabel = "Y_t", xlabel = "T")
    p3 = plot(-15.0:1.0:14.0, stats[7,60:89], ylabel = "P_n/P_t", xlabel = "T")
    p4 = plot(-15.0:1.0:14.0, stats[4,60:89], ylabel = "C_t", xlabel = "T")
    p5 = plot(-15.0:1.0:14.0, stats[6,60:89], ylabel = "C_n'", xlabel = "T")
    p6 = plot(-15.0:1.0:14.0,stats[9,60:89], ylabel = "h", xlabel = "T")
    
    plot(p1,p2, p3,p4,p5, p6, layout = (3, 2))
    png("output/pre_def_stats_1")

    writedlm( "output/results_pre_def.csv",  stats, ',')

end


#main function 
function run_spline(non_calib::TD_assumed,calib::TD_calib, grid_param::TD_gird;ERR ="float")
    println("you are using: ", Threads.nthreads(), " threads")
    println("Itertations starts")

    unpack_params(non_calib, calib,grid_param )
    global P_stat, y_t_stat = DiscretizeAR(ρ, σ_μ, n_y, disc)
    if(ERR == "float")
        Model_sol = Solve_Bellman_float(600, method = itp)
        println("Siumulations starts")
        pd, dt, loss, stats = simulate_TD(Model_sol, w_bar, ERR="float")
    elseif(ERR == "peg")
        Model_sol = Solve_Bellman_peg(600, method = itp)
        println("Siumulations starts")
        pd, dt, loss, stats = simulate_TD(Model_sol, w_bar, ERR="peg")
    else
        throw(DomainError(method, "this exchange rate regime is not supported, try float or peg "))
    end
    plot_solution(Model_sol)
    plot_simulation(stats)
    
end


end