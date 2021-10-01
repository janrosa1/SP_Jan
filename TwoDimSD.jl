module TwoDimSD
using Base: Float64
using LinearAlgebra, Interpolations, Random, QuantEcon, Statistics, DataFrames, Distributions, CSV, Plots, Optim, Roots, SchumakerSpline,NLopt
using DelimitedFiles, NaNMath, JLD
using Parameters: @with_kw, @unpack
export Solve_eq_float2D,simulate_eq_2D, Solve_eq_peg2D, run_2D, unpack_params, TD_assumed, TD_calib, TD_gird, run_2D_calib, run_SMM, wrapper_SMM, run_2D_calib_whole_exp, simulate_eq_2D_default


global β = 0.9
global σ = 2.0
global α = 0.75
global h_bar = 1.0
global a = 0.26
global ζ = 0.5
global y_t = 1.0
global θ = 0.0385
global δ_1 = 0.58 #-0.35
global δ_2 = 2.15# 0.46
global ρ = 0.93
global σ_μ = 0.037
global r =0.01
global n_y = 21
global n_B = 35
global n_d = 15
global B_min = 0.0
global B_max = 1.0
global d_min = 0.0
global d_max = 0.4
global grid_B = B_min:(B_max-B_min)/(n_B-1):B_max
global grid_d = d_min:(d_max-d_min)/(n_d-1):d_max
global w_bar = 0.95* α * (1.0-a)/a
global c_t_min = 1.0
global itp, disc
global k = 0.3
global c_t_min = 1.0
global P_stat, y_t_stat
global pen = 0.03
global ERR_glob = "peg"

@with_kw struct TD_assumed #assumed paramters from TwinD;s paper
    σ::Float64 = 2.0 #CRRA param
    α::Float64 = 0.75  # production function param
    a::Float64 = 0.26 #CES utility of the tradeable good in the CES function of the final good
    ζ::Float64 = 0.5 #CES final good parameter
    y_t::Float64 = 1.0 # tradeable goods long term expectation
    ρ::Float64 = 0.9317 #tradeables shock persitanece (Argentinan economy from TwinD's)
    σ_μ::Float64 = 0.037 #tradeables shock variance (Argentinan economy from TwinD's)
    r::Float64 = 0.01 #interest rate 
    θ::Float64 = 1.0 # probability of the reentry to the market after the default
    h_bar::Float64 = 1.0 #maximal employment level
end

@with_kw struct TD_calib #values which should be calibrated
    β::Float64 = 0.96 #discount factor
    δ_1::Float64 = 0.015 #penalty function param for the default (in form δ_1 y_t+ δ_2 y_t^2 ) 
    δ_2::Float64 = 0.6 #penalty function param for the default (in form δ_1 y_t+ δ_2 y_t^2 ) 
    w_bar::Float64 = 0.99* α * (1.0-a)/a #for the peg model calibration, minimal wage 
    pen::Float64 = 0.025 #percentage of consumption due to 
    k::Float64 = 0.3
end

@with_kw struct TD_gird #technical params for the asset/income grid structure, discertization, interpolation method 
    n_y::Int64 = 21 #number of grid points for income, 21 is magical number used in all Arellano papers 
    n_B::Int64 = 50 #number of grid points for the Schumker splines, about 30 works for cubic splines, for the vfi, fior a good precision at least 200 points 
    n_d::Int64 = 20
    B_min::Float64 = 0.0 #minimal debt level for (works for float)
    B_max::Float64 = 1.0 #maximal debt level
    d_min = 0.0
    d_max = 0.4
end
-
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
        MC = tauchen(n, ρ, var,0.0, 3)
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


function unpack_params(non_calib, calib, grid_p) #pretty ugly way to unpack params, probably code would be faster without globals but for now speed is not an issue (for float) 

    @unpack σ, α, a, ζ, y_t, ρ, σ_μ, r, θ,h_bar = non_calib
    @unpack β,δ_1,δ_2,w_bar, pen, k = calib
    @unpack n_y, n_B, n_d, B_min, B_max, d_min, d_max  = grid_p
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
    global pen = pen
    global k = k
    global n_y = n_y
    global n_B = n_B
    global B_min = B_min
    global B_max = B_max
    global grid_B = B_min:(B_max-B_min)/(n_B-1):B_max
    global n_d = n_d
    global d_min = d_min
    global d_max = d_max
    global grid_d = d_min:(d_max-d_min)/(n_d-1):d_max
    global P_stat, y_t_stat = DiscretizeAR(ρ, σ_μ, n_y, "tauchen")
    global c_t_min = (copy(w_bar)/(copy(α)*(1-copy(a))/copy(a)))^(ζ)
end


function final_good(c_t, c_n) #final good aggregation
    return (a*c_t^(1.0-1.0/ζ)+(1.0-a)*c_n^(1.0-1.0/ζ))^(1.0/(1.0-1.0/ζ))
end

function utilityCRRA(c) #CRRA utility 
    x = max(c, 1e-6)
    if σ ==1
        return log(x)
    end
    return (x^(1.0-σ)-1.0)/(1.0-σ)
end

function marg_utilityCRRA(c, σ) #not used so far
    return c^(-σ)
end

function L(y_t) #penalty function for default
    #return max(δ_1*y_t + δ_2*y_t^2.0,0.0)
    return max(δ_1 + δ_2*log(y_t),0.0)
end



function q_d(d)
    return 1/(1+r+k*d)
end

function c_n_peg(c_t)
    if(ERR_glob == "float")
        c_n = 1.0
        return c_n
    end
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    c_n = max(min(const_w*c_t^power,1.0),1e-6)
    
    return c_n
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



#simulate if the country continue to be exclulded after default
function simulate_exclusion(θ)
    dists = Categorical([θ , 1.0- θ])
    sim = rand(dists, 1)
    return sim[1]
end
#####################################################
#functions for global optimization
#####################################################


function interp_alg_const(splines_v, spline_q, state_y,state_b,state_d, x) 
    
    """
    interpolate the value of choosing debt b, given:
    value functions (splines_v),
    price function (spline_q),
    endowment: state_y
    current debt: state_b    
    """
    b= x[1]
    d = x[2]
     #this is version for float so far
    q_b = min(max(spline_q(b,d), 0.0),1.0/(1.0+r)) #compute debt price
    if b<=0
        q_b =1.0/(1.0+r)
    end

    c_t = -state_b - state_d+ y_t_stat[state_y] +q_b*b+q_d(d)*d #compute tradeable consumption (check if it's >=0)
    
    if(c_t<=0)
        c_t = 1e-7
    end

    if( c_t < c_t_min)
        c_n  = (1.0-pen)*copy(c_n_peg(c_t)) #compute the consumption equivalent utility decrease (it does not mess with FOC)
    else
        c_n = (1.0-pen)
    end
    #compute the  consumption with b
    c = final_good(c_t, c_n)
    val =0.0
    val = copy(utilityCRRA(c))

    #compute the next period's values
    for j in 1:n_y
        val = copy(val) + β*P_stat[state_y, j]*splines_v[j](b,d)
    end
    return -1.0*val #minimizing function, so need a negative value 
end


function interp_alg_const_def(splines_v, splines_v_d, state_y,state_d, x) 
    
    """
    interpolate the value of choosing debt b, given:
    value functions (splines_v),
    price function (spline_q),
    endowment: state_y
    current debt: state_b    
    """
    d= x[1]
     #this is version for float so far

    c_t =  -state_d + y_t_stat[state_y] + q_d(d)*d #compute tradeable consumption (check if it's >=0)
    
    if(c_t<=0)
        c_t = 1e-6
    end

    if( c_t < c_t_min) #compute the consumption equivalent utility decrease (it does not mess with FOC)
        c_n  = (1.0-pen)*copy(c_n_peg(c_t))
    else
        c_n =(1.0-pen)
    end
    #compute the  consumption with b
    c = final_good(c_t, c_n)
   
    val = 0.0
    #compute the next period's values
    for j in 1:n_y
        val = copy(val) + β*P_stat[state_y, j]*(θ*splines_v[j](0.0,d) +(1.0-θ)*splines_v_d[j](d))
    end
    val = copy(utilityCRRA(c)) - copy(L(y_t_stat[state_y]))+copy(val)
    return -1.0*val #minimizing function, so need a negative value 
end

function interp_alg_const_noIMF(splines_v, spline_q, state_y,state_b,state_d, x) 
    
    """
    interpolate the value of choosing debt b, given:
    value functions (splines_v),
    price function (spline_q),
    endowment: state_y
    current debt: state_b    
    """
    
    b= x[1]
     #this is version for float so far
    q_b = min(max(spline_q(b,0), 0.0),1.0/(1.0+r)) #compute debt price
    if b<=0
        q_b =1.0/(1.0+r)
    end

    c_t = -state_b - state_d+ y_t_stat[state_y] +q_b*b #compute tradeable consumption (check if it's >=0)
    
    if(c_t<=0)
        c_t = 1e-7
    end

    if( c_t < c_t_min) #no utility decrease since no IFI debt
        c_n  = copy(c_n_peg(c_t))
    else
        c_n = 1.0
    end
    #compute the  consumption with b
    c = final_good(c_t, c_n)
    val =0.0
    val = copy(utilityCRRA(c))

    #compute the next period's values
    for j in 1:n_y
        val = copy(val) + β*P_stat[state_y, j]*splines_v[j](b,0.0)
    end
    return -1.0*val #minimizing function, so need a negative value 
end


####################################################################################
#Wrappers for the global optimization (define problem in NLopt library)
####################################################################################

function solve_const_prob(f, x_0)
    opt = Opt(:LN_BOBYQA , 2) #LN_BOBYQA , LN_NEWUOA, LN_COBYLA
    opt.lower_bounds = [grid_B[1], grid_d[1]]
    opt.upper_bounds = [grid_B[n_B], grid_d[n_d]]
    opt.xtol_rel = 1e-8
    opt.min_objective = f
    opt.maxeval = 2500
    opt.initial_step = 1e-3 #(grid_B[n_B]-grid_B[1])/500.0
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end


function solve_const_prob_2(f, x_0)
    opt = Opt(:LN_COBYLA , 2) #LN_BOBYQA , LN_NEWUOA, LN_COBYLA
    opt.lower_bounds = [grid_B[1], grid_d[1]]
    opt.upper_bounds = [grid_B[n_B], grid_d[n_d]]
    opt.xtol_rel = 1e-8
    opt.min_objective = f
    opt.maxeval = 2500
    opt.initial_step = 1e-3 #(grid_B[n_B]-grid_B[1])/500.0
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end


function solve_const_prob_1d(f, x_0)
    opt = Opt(:LN_COBYLA, 1) #LN_BOBYQA , LN_NEWUOA, LN_COBYLA
    opt.lower_bounds = [grid_d[1]]
    opt.upper_bounds = [grid_d[n_d]]
    opt.xtol_rel = 1e-8
    opt.min_objective = f
    opt.maxeval = 2500
    opt.initial_step = 1e-3
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end

function solve_const_prob_1b(f, x_0)
    opt = Opt(:LN_COBYLA, 1) #LN_BOBYQA , LN_NEWUOA, LN_COBYLA
    opt.lower_bounds = [grid_B[1]]
    opt.upper_bounds = [grid_B[n_B]]
    opt.xtol_rel = 1e-8
    opt.min_objective = f
    opt.maxeval = 2500
    opt.initial_step = 1e-3
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end

############################################################################################
#global max procedures
##########################################################################################

function global_max(f, q_splines,x_guess; loc_flag =1.0)
    
    if(loc_flag==1.0)
        x_0_1s = [0.0, 0.00]
        x_guess = [0.0,0.0]
        g = ones(2)
        max_glob = f(x_0_1s,g )
        x_guess[2] = min(x_guess[2], grid_d[n_d])
        #first stage
        val_proposals = 1000*ones(4)
        maximizer_proposals = zeros(2,4)
        #look for the possible max on the whole grid
        grid_id = grid_d[1]:(grid_d[n_d]-grid_d[1])/20: grid_d[n_d]
        grid_i = grid_B[1]:(grid_B[n_B]-grid_B[1])/40: grid_B[n_B]


        for ki in 1:length(grid_i)
            for kid in 1:length(grid_id)
                if f([grid_i[ki],grid_id[kid]],g)< max_glob
                    x_0_1s = copy([grid_i[ki],grid_id[kid]])
                    max_glob = f([grid_i[ki],grid_id[kid]],g)
                end
            end
        end
        
        #second stage
        
        #set kid and look for the optimal ki
        ff(x_b) = f([x_b,x_0_1s[2]], [0.1, 0.1])
        qq(b) = min(q_splines(b, x_0_1s[2]),1.0/(1.0+r))
        sol_max_qB = Optim.optimize(x-> -qq(x)*x,grid_B[1], grid_B[n_B], GoldenSection())
        max_qB = Optim.minimizer(sol_max_qB)[1]
        sol_2s  = Optim.optimize(ff, grid_B[1], max_qB, GoldenSection())
        x_0_2s = Optim.minimizer(sol_2s)[1] 
        # #third stage
        (val_proposals[1],maximizer_proposals[:,1],ret) = solve_const_prob(f,[x_0_1s[1],x_0_1s[2]] )
       
        (val_proposals[3],maximizer_proposals[:,3],ret2) = solve_const_prob(f, [x_0_2s,x_0_1s[2]] )

        val_proposals[isnan.(val_proposals)].=Inf
        

        (minf, ind) = findmin(val_proposals)
        
        if(minf>max_glob)
            
            println("second_shot")
            (minf,minx,ret) = solve_const_prob(f, [0.0, 0.01] )
            if(minf>max_glob)
                (minf,minx,ret) = solve_const_prob_2(f, [0.0, 0.01] )
                println("third_shot")
                if(minf>max_glob)
                    minf = copy(max_glob)
                    solve_const_prob_2(f, [0.0, 0.01] )
                    println("mistake")
                end
            end        

        end
        @assert minf == minimum(val_proposals)
        minx = maximizer_proposals[:,ind]
        return (minf,minx,ret)
    else
        (minf,minx,ret) = solve_const_prob_2(f,x_guess )
        return (minf,minx,ret)
    end
end

#same global maximization but duruing default
function global_max_def(f, x_guess; loc_flag =1.0)
   
    if(loc_flag==1.0)
        x_0 = [0.0]
        g = ones(1)
        max_glob = f(x_0,g )
        
        grid_id = grid_d[1]:(grid_d[n_d]-grid_d[1])/60: grid_d[n_d]
       

        
        for kid in 1:length(grid_id)
            if f([grid_id[kid]],g)< max_glob
                x_0 = copy([grid_id[kid]])
                max_glob = f(x_0,g)
            end
        end
       
        #second stage
        

        (minf,minx,ret) = solve_const_prob_1d(f,x_0)

        if(minf>max_glob)
            
            (minf,minx,ret) = (max_glob, x_0, 1)

        end
    else
        (minf,minx,ret) = solve_const_prob_1d(f,[x_guess] )
    end 
        return (minf,minx,ret)
end

#global maximization withpout IFI debt 
function global_max_noIMF(f, x_guess; loc_flag =1.0)
    
    if(loc_flag==1.0)
        x_0 = [0.0]
        g = ones(1)
        max_glob = f(x_0,g )
        
        grid_iB = grid_B[1]:(grid_B[n_B]-grid_B[1])/150: grid_B[n_B]
       

        
        for i in 1:length(grid_iB)
            
            if f([grid_iB[i]],g)< max_glob
                x_0 = copy([grid_iB[i]])
                max_glob = f(x_0,g)
            end
        end
       
        #second stage
        

        (minf,minx,ret) = solve_const_prob_1b(f,x_0)

        if(minf>max_glob)
            
            
            (minf,minx,ret) = (max_glob, x_0, 1)

        end
    else
        (minf,minx,ret) = solve_const_prob_1b(f,[x_guess])
    end 
   
    return (minf,minx,ret)
end

###############################################################################################
#Value function iterations
###############################################################################################

#Solve the bellman equation via finite time approximation  
function Solve_eq_peg2D(T)
    #define the process for discretization 
    global P_stat, y_t_stat = DiscretizeAR(ρ, σ_μ, n_y, "tauchen")
    
    #allocate memory
    n_B_q = 2*n_B
    n_d_q = 2*n_d
    grid_B_q = grid_B[1]: (grid_B[n_B]- grid_B[1])/(n_B_q-1):grid_B[n_B]
    grid_d_q = grid_d[1]: (grid_d[n_d]- grid_d[1])/(n_d_q-1):grid_d[n_d]
    v_f = zeros(T, n_y, n_B, n_d)
    v_c = zeros(T, n_y, n_B, n_d)
    v_d = zeros(T, n_y, n_d)
    q = zeros(T, n_y, n_B, n_d)
    c_t = zeros(T, n_y, n_B, n_d)
    c_n = ones(T, n_y, n_B, n_d)
    c = zeros(T, n_y, n_B, n_d)
    d = zeros(T, n_y, n_B, n_d)
    policy_d = zeros(T, n_y, n_B, n_d)
    policy_def = zeros(T, n_y, n_d)
    policy_B = zeros(T, n_y, n_B, n_d)
    default_border = zeros( n_y, n_d)
    #define spline matrix
    V_f_spline = Array{Interpolations.ScaledInterpolation}(undef,T,n_y)
    V_c_spline = Array{Interpolations.ScaledInterpolation}(undef,T,n_y)
    V_d_spline = Array{Interpolations.ScaledInterpolation}(undef,T,n_y)
    q_spline = Array{Interpolations.ScaledInterpolation}(undef,T, n_y)
    d_itp =  Array{Interpolations.Extrapolation}(undef,T, n_y)
    #define x_0 and max func
    x_0 = zeros(2,n_y)
    max_func = Array{Function}(undef,n_y)

    #define vector of solutions
    minf = zeros(n_y)
    minx = zeros(2,n_y)
    minf_IMF = zeros(n_y)
    minx_IMF = zeros(2,n_y)
    minf_noIMF = zeros(n_y)
    minx_noIMF = zeros(n_y)

    #start the loop for the last period
    for j in 1:n_y
        for i in 1:n_B
            for id in 1:n_d
                # check unconstrained case
                c_t_c = max(y_t_stat[j]-grid_d[id] - grid_B[i],1e-7)
                if(c_t_c<c_t_min)
                    c_n_c = c_n_peg(c_t_c)
                else
                    c_n_c = 1.0
                end
                c_c = final_good(c_t_c, c_n_c)
                
                c_t_d = max(y_t_stat[j]-grid_d[id],1e-7) 
                if(c_t_d<c_t_min)
                    c_n_d = c_n_peg(c_t_d)
                else
                    c_n_d = 1.0
                end
                c_d = final_good(c_t_d,c_n_d)
                if utilityCRRA(c_c)>utilityCRRA(c_d)-L(y_t_stat[j])
                    c_t[T,j,i,id] = c_t_c
                    c_n[T,j,i,id] = c_n_c
                    c[T,j,i,id] = c_c
                    v_f[T,j,i,id] = utilityCRRA(c[T,j,i,id])
                else
                    c_t[T,j,i,id] = c_t_d
                    c_n[T,j,i,id] = c_n_d
                    c[T,j,i,id] = c_d
                    d[T,j,i,id]  = 1.0
                    v_f[T,j,i,id] = utilityCRRA(c[T,j,i,id]) -L(y_t_stat[j])
                end
                v_c[T,j,i,id] = utilityCRRA(c_c)
                v_d[T,j,id] = utilityCRRA(c_d)-L(y_t_stat[j])
            end
        end
    end
    println(sum(d[T,:,:,:]))
    ##ITERTAIONS STARTS
    flag = 1.0
    for t in T-1:-1:1
        
        for j in 1:n_y
            V_f_spline[t+1,j ] = Interpolations.scale(interpolate(v_f[t+1,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
            V_c_spline[t+1,j ] = Interpolations.scale(interpolate(v_c[t+1,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
            V_d_spline[t+1, j] = Interpolations.scale(interpolate(v_d[t+1,j,:], BSpline(Cubic(Line(OnGrid())))), grid_d) 
            #CubicSplineInterpolation((grid_B, grid_d), v_f[t+1,j,:,:],extrapolation_bc = Line())
            
        end
        for j in 1:n_y
            for i in 1:n_B
                for id in 1:n_d
                    d_val = 0.0
                    for jj in 1:n_y
                        d_val = copy(d_val)+ P_stat[j,jj]*d[t+1,jj,i,id]                   
                    end   
                    q[t,j,i,id] = 1.0/(1.0+r)*(1.0-copy(d_val))
                end
            end
        end
        for j in 1:n_y
            
            q_spline[t, j] =  Interpolations.scale(interpolate(q[t,j,:,:], BSpline(Interpolations.Linear())), grid_B, grid_d) 
            
        end
        Threads.@threads for j in 1:n_y 
            
            for id in 1:n_d
                
                #compute default value

                x_0_def = max(min(policy_def[t+1,j,id], grid_d[n_d]),grid_d[1])
                func_def(x::Vector, grad::Vector) = deepcopy(interp_alg_const_def(copy(V_f_spline[t+1,:]),copy(V_d_spline[t+1,:]), j ,grid_d[id],x))
                (minf_def,minx_def,ret_def) = global_max_def(func_def, x_0_def,  loc_flag = flag)  #solve_const_prob(interp_alg_unconst_1, constraint, x_0)
               

                c_t_d = max(y_t_stat[j] -grid_d[id],1e-6)
                val_d = 0.0
                for jj in 1:n_y
                    val_d = copy(val_d)+ P_stat[j,jj]*β*(θ*V_f_spline[t+1,jj ](0.0,0.0)+ (1-θ)*v_d[t+1, jj,1]) #TODO         
                end
                if(c_t_d< c_t_min)
                    c_n_d = c_n_peg(c_t_d)
                else
                    c_n_d = 1.0
                end

                if utilityCRRA(final_good(c_t_d, c_n_d))  - L(y_t_stat[j])+ val_d > -  minf_def
                    v_d[t,j,id] = utilityCRRA(final_good(c_t_d, c_n_d))  - L(y_t_stat[j])+ val_d
                    policy_def[t,j, id] = 0.0
                else
                    policy_def[t,j, id] = minx_def[1]    
                    v_d[t,j,id] = -minf_def
                end

                
                
                
                
                #compute continuation
                for i in 1:n_B
                    if(t==T-1 )
                        x_0[:,j] = [0.0001,0.0]
                    else
                        x_0[:,j] = [max(min(policy_B[t+1,j,i,id], grid_B[n_B]),grid_B[1]), max(min(policy_d[t+1,j,i,id], grid_d[n_d]), grid_d[1])]
                    end
                    #define the maximized functions for both cases (with IFI debt and without)
                    func_IMF(x::Vector, grad::Vector) = deepcopy(interp_alg_const(copy(V_f_spline[t+1,:]), q_spline[t,j], j ,grid_B[i],grid_d[id],x))
                    func_noIMF(x::Vector, grad::Vector) = deepcopy(interp_alg_const_noIMF(copy(V_f_spline[t+1,:]), q_spline[t,j], j ,grid_B[i],grid_d[id],x))
        
                    (minf_IMF[j],minx_IMF[:,j],ret_IMF) = global_max(func_IMF, q_spline[t,j],x_0[:,j], loc_flag = flag)  #solve the problem with the IFI debt
                    
                    (minf_noIMF[j],b_sol,ret_noIMF) = global_max_noIMF(func_noIMF ,x_0[1,j],  loc_flag = flag) #solve the problem without IFI debt
                    minx_noIMF[j] = b_sol[1]
                    
                    #compare values
                    if minf_noIMF[j]<minf_IMF[j]
                        v_f[t,j,i,id] = -minf_noIMF[j]
                        policy_d[t,j,i,id] = 0.0
                        policy_B[t,j,i,id] = minx_noIMF[j]
                    else
                        v_f[t,j,i,id] = -minf_IMF[j]
                        policy_d[t,j,i,id] = minx_IMF[2,j]
                        policy_B[t,j,i,id] = minx_IMF[1,j]
                    end
                    #println(maxim_sol)
                    
                    
                    if v_f[t,j,i,id] <= v_d[t, j,id]
                        v_f[t,j,i,id] = v_d[t, j,id]
                        d[t,j,i,id]  = 1.0
                    end
                end
            end
            
        end
        if(mod(T-t, 10)==0)
            println("Iteration: ", T-t)
            println(maximum(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:])))
            println(mean(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:])))
            println(maximum(abs.(q[t,:,:,:].-q[t+1,:,:,:])))
            println(sum(d[t,:,:,:]))   
        end
        
        if(maximum(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:])) <=5e-2 && T-t>10) #+maximum(abs.(q[t,:,:,:].-q[t+1,:,:,:]))
            flag =0.0
        end
        if(maximum(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:]))<=1e-5 || t==1) #stop iterations
            v_f[1,:,:,:] = v_f[t,:,:,:]
            q[1,:,:,:] = q[t,:,:,:]
            policy_d[1,:,:,:] =policy_d[t,:,:,:]
            policy_B[1,:,:,:] = policy_B[t,:,:,:]
            d[1,:,:,:] = d[t,:,:,:]
            v_d[1,:,:] = v_d[t,:,:]
            policy_def[1,:,:] = policy_def[t,:,:]
            for j in 1:n_y
                V_f_spline[1,j ] = Interpolations.scale(interpolate(v_f[t,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
                #CubicSplineInterpolation((grid_B, grid_d), v_f[t+1,j,:,:],extrapolation_bc = Line())
                q_spline[1, j] =  Interpolations.scale(interpolate(q[t,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
                #LinearInterpolation((grid_B, grid_d), q[t,j,:,:],extrapolation_bc = Periodic()) #spline works very bad for the price function (as they are non-monotonic)
            end
            break
        end
    end
    #compute an exact default region

    for j in 1:n_y 
        for id in 1:n_d
            if abs(v_f[1,j,1, id]-v_d[1, j, id])<= 1e-6
                default_border[j, id] = 0.0+1e-6
            elseif v_f[1,j,n_B, id]-v_d[1, j, id]>1e-6
                default_border[j, id] = grid_B[n_B]+0.2 #never will be chosen    
            else
                f_def_bord(x) = V_f_spline[1,j](x, grid_d[id])- v_d[1, j, id]-1e-6
                default_border[j, id] = Roots.find_zero(f_def_bord, (grid_B[1], grid_B[n_B] ))
            end 
           
        end
    end

    
    

    
    return (q[1,:,:,:], d[1,:,:,:], policy_B[1,:,:,:], policy_d[1,:,:,:], v_f[1,:,:,:],default_border, v_d[1,:,:], policy_def[1,:,:] )
end


########################################################################################################################
#simulate economy
#########################################################################################################################
function simulate_eq_2D( Model_solution; burnout = 1000, t_sim =1200, n_sim = 10000, ERR="float", method = "spline_sch")
    global P_stat, y_t_stat = DiscretizeAR(ρ, σ_μ, n_y, "tauchen")
    
    #unpack policy functions and paramters
    n_B_long =10000
    q = Model_solution[1]
    #Def_mat = Model_solution[2]
    Policy_B = Model_solution[3]
    Policy_d  = Model_solution[4]
    Def_bord = Model_solution[6]
    Policy_def = Model_solution[8]
    grid_B_long = grid_B[1]:(grid_B[n_B] - grid_B[1])/(n_B_long-1):grid_B[n_B]
    Def_matrix = zeros(n_y, n_B_long, n_d)
    for j in 1:n_y
        for id in 1:n_d
            for i in 1:n_B_long
                if(Def_bord[j,id]>= grid_B_long[i])
                    Def_matrix[j,i,id] =1.0 
                end
            end
        end
    end
    
    
    #allocate memory
    
    SIM = zeros(Int64, n_sim, t_sim) #simulated income process
    B_hist = zeros(n_sim, t_sim) #private debt history
    d_hist = zeros(n_sim, t_sim) #assets history for the non-defautable debt
    Y_t = zeros(n_sim, t_sim) # output history
    Trade_B = zeros(n_sim, t_sim) # trade balance history
    
    C_t = zeros(n_sim, t_sim) # consumption history tradeables
    C_n = ones(n_sim, t_sim) # non-tradeables
    h_t = ones(n_sim, t_sim)
    C = zeros(n_sim, t_sim) # final good
    
    D = zeros(n_sim, t_sim) #Defaults histiry
    D_state = zeros(n_sim, t_sim) #exclusion from the financial market histiry
    R = zeros(n_sim, t_sim) #q hiatory
    R_d = zeros(n_sim, t_sim)
    #exchange rate and minimal wage
   
    ϵ = ones(n_sim, t_sim)
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    
    policy_B_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_d_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_def_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    q_spline = Array{Interpolations.ScaledInterpolation}(undef, n_y)
    Def_bord_itp =  Array{Interpolations.Extrapolation}(undef,n_y)
    #interpolate policy functions
    for j in 1:n_y
        q_spline[j] =  Interpolations.scale(interpolate(q[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_B_spline[j] =  Interpolations.scale(interpolate(Policy_B[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_d_spline[j] = Interpolations.scale(interpolate(Policy_d[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d)
        policy_def_spline[j] = Interpolations.scale(interpolate(Policy_def[j,:], BSpline(Cubic(Line(OnGrid())))), grid_d)
        Def_bord_itp[j] = LinearInterpolation((grid_B_long, grid_d), Def_matrix[j,:,:], extrapolation_bc = Periodic()) #def bord is given for a id points co that is why I change the 
    end

    #define starting values
    y_0 = Int(floor(n_y/2) ) # start with 11th state
    Y_0 = y_t_stat[y_0]*ones(n_sim) #output start value
    B_0 = zeros(n_sim) #start with 0 assets
    d_0 = zeros(n_sim)
    Y_t[:,1] = Y_0 
    B_hist[:,1] = B_0
    d_hist[:,1] = d_0

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
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    d_hist[i,t] = min(max(policy_def_spline[SIM[i,t]](d_hist[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t] + q_d(d_hist[i,t])*d_hist[i,t] - d_hist[i,t-1]
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 - pen 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 - pen 
                        end
                        R_d[i,t] = -1.0
                    end

                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    B_hist[i,t] = 0.0
                    
                    R[i,t] = -1.0 #nop interest rate
                    D[i,t] = 0.0
                    D_state[i,t] = 1.0
                else
                    
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = max(policy_B_spline[SIM[i,t]](0.0, d_hist[i,t-1] ), grid_B[1])
                    d_hist[i,t] = min(max(policy_d_spline[SIM[i,t]](0.0, d_hist[i,t-1]),0.0), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t] + q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t])*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t] - d_hist[i,t-1] 
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 - pen 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 - pen 
                        end
                        R_d[i,t] = -1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 
                    if(B_hist[i,t]>1e-6)
                        R[i,t] = 1.0/q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t]) - 1.0
                    else
                        R[i,t] = -1
                    end
                    
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                    
                end
            else
                if Def_bord_itp[SIM[i,t]](B_hist[i,t-1],d_hist[i,t-1] ) <1.0  #case of default
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    d_hist[i,t] =  min(max(policy_def_spline[SIM[i,t]](d_hist[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t]  - d_hist[i,t-1] + q_d(d_hist[i,t])*d_hist[i,t]
                    C[i,t] = final_good(C_t[i,t],1.0) 
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                           
                        end
                        R_d[i,t] = -1.0
                    end
                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    B_hist[i,t] = 0.0
                    R[i,t] = -1.0 #1.0/q[SIM[i,t],A_int[i,t]] - 1.0
                    D[i,t] = 1.0
                    D_state[i,t] = 1.0
                    Default_flag = 1.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                else
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = max(min(policy_B_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1]), grid_B[n_B]), grid_B[1])
                    d_hist[i,t] = min(max(policy_d_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1]),0.0), grid_d[n_d])
                    C_t[i,t] =max(Y_t[i,t]- B_hist[i,t-1] - d_hist[i,t-1]+ q_spline[SIM[i,t]](0.0, 0.0)*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t], 1e-6)
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) - 1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0
                        end
                        R_d[i,t] =  - 1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 

                    if(B_hist[i,t]>1e-6)
                        R[i,t] = 1.0/q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t]) - 1.0
                    else
                        R[i,t] = -1
                    end
                   
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
    
    non_defaults = D_state[:,burnout:t_sim].<1.0   

    #default probability
    Def_prob  =  1.0 - (1.0-n_defaults/(n_sim*(t_sim-burnout)))^4.0   
    n_chosen_def = 10000

    #stats after default:
    Y_ab = Y_t[:,burnout:t_sim]
    R_ab = R[:,burnout:t_sim]
    R_d_ab = R_d[:,burnout:t_sim]
    B_hist_ab = B_hist[:,burnout:t_sim]
    d_hist_ab = d_hist[:,burnout:t_sim]
    D_ab = D[:,burnout:t_sim]
    ϵ_ab = ϵ[:,burnout:t_sim]
    Trade_B_ab = Trade_B[:,burnout:t_sim]
    D_state_ab = D_state[:,burnout:t_sim]
    C_ab = C[:,burnout:t_sim]
    C_ab_t = C_t[:,burnout:t_sim]
    C_ab_n = C_n[:,burnout:t_sim]
    h_t_ab = h_t[:,burnout:t_sim]
    #choose number of defaults to compute statistics

    #allocate memory for the staoistics 
    pre_def_stats_Y_t = zeros(n_chosen_def , 56)
    pre_def_stats_R = zeros(n_chosen_def , 56)   
    pre_def_stats_R_d = zeros(n_chosen_def , 56)        
    pre_def_stats_TB = zeros(n_chosen_def , 56)        
    pre_def_stats_C_t = zeros(n_chosen_def , 56) 
    pre_def_stats_C_n = zeros(n_chosen_def , 56) 
    pre_def_stats_C = zeros(n_chosen_def , 56) 
    pre_def_stats_B = zeros(n_chosen_def , 56)
    pre_def_stats_d = zeros(n_chosen_def , 56)
    pre_def_stats_d_change = zeros(n_chosen_def , 56)
    pre_def_stats_NB = zeros(n_chosen_def , 56)  
    pre_def_stats_P = zeros(n_chosen_def , 56)  
    pre_def_stats_D = zeros(n_chosen_def , 56) 
    pre_def_stats_changes_D = zeros(n_chosen_def , 56) 
    pre_def_stats_h = zeros(n_chosen_def , 56) 
    stats = zeros(11, 56)
    ϵ_stats = zeros(n_chosen_def , 56)  
    
    pre_def_stats_Y_t_yearly = zeros(n_chosen_def , 10)
    pre_def_stats_C_t_yearly = zeros(n_chosen_def , 10)
    pre_def_stats_d_t_change_yearly = zeros(n_chosen_def , 10)

    mean_R = zeros(n_chosen_def)
    std_R = zeros(n_chosen_def)
    mean_R_IMF = zeros(n_chosen_def)
    std_R_IMF = zeros(n_chosen_def)
    std_C = zeros(n_chosen_def)
    std_Y_t =  zeros(n_chosen_def)
    mean_B = zeros(n_chosen_def)
    mean_d = zeros(n_chosen_def)
    cor_D_change_Y_t = zeros(n_chosen_def)
    
    cor_R_Y_t = zeros(n_chosen_def)
    d_non_zero = zeros(n_chosen_def)

    std_C_yearly =  zeros(n_chosen_def)
    std_Y_t_yearly = zeros(n_chosen_def)
    std_change_d_t_yearly = zeros(n_chosen_def)
    cor_D_change_Y_t_yearly = zeros(n_chosen_def)
    #compute statistics for 74 periods before the default, for n_def defaults (without any defults in the 74 periods before)
    iter = 0
    dists = Categorical(1/(t_sim-burnout)*ones(t_sim-burnout)) 
    #for i in 1:n_sim, t in 76:t_sim-burnout
    for i in 1:n_sim
        t= 100
          if(t>=100  )
                 iter =iter+1   
                
                 pre_def_stats_Y_t[iter,:] = Y_ab[i, t-40:t+15]
                 pre_def_stats_R[iter,:] = R_ab[i, t-40:t+15]
                 pre_def_stats_B[iter,:] = B_hist_ab[i, t-40:t+15]
                 pre_def_stats_R_d[iter,:] = R_d_ab[i, t-40:t+15]
                 pre_def_stats_d[iter,:] = d_hist_ab[i, t-40:t+15]
                 pre_def_stats_d_change[iter,:] = d_hist_ab[i, t-40:t+15] - d_hist_ab[i, t-41:t+14]
                 pre_def_stats_NB[iter,:] = B_hist_ab[i, t-40:t+15] #new debt
                 pre_def_stats_TB[iter,:] = Trade_B_ab[i, t-40:t+15]
                 pre_def_stats_C_t[iter,:] = C_ab_t[i, t-40:t+15]
                 pre_def_stats_C_n[iter,:] = C_ab_n[i, t-40:t+15]
                 pre_def_stats_C[iter,:] = C_ab[i, t-40:t+15]
                 pre_def_stats_P[iter,:] = (1-a)/a*(pre_def_stats_C_t[iter,:]./pre_def_stats_C_n[iter,:])
                 ϵ_stats[iter, :] = ϵ_ab[i, t-40:t+15] 
                 pre_def_stats_h[iter,:] = h_t_ab[i, t-40:t+15] 
                 pre_def_stats_D[iter,:] =  D_state[i, t-40:t+15]

                 k=0
                for tt in 1:40
                    
                 
                    if(mod(tt,4)==0)
                        k = k+1
                        pre_def_stats_Y_t_yearly[i,k ] = sum(pre_def_stats_Y_t[i,tt-3:tt])
                        
                        pre_def_stats_C_t_yearly[i,k ] = sum(pre_def_stats_C[i,tt-3:tt])
                        pre_def_stats_d_t_change_yearly[i,k] = d_hist_ab[i, tt] - d_hist_ab[i, tt-3]
                    end
                end 
                #means and std
                mean_R[iter] = mean(filter(x->x!=-1.0, pre_def_stats_R[iter,1:40]))
                std_R[iter] =  std(filter(x->x!=-1.0, pre_def_stats_R[iter,1:40]))
                mean_R_IMF[iter] = mean(filter(x->x!=-1.0, pre_def_stats_R_d[iter,1:40]))
                std_R_IMF[iter] = std(filter(x->x!=-1.0, pre_def_stats_R_d[iter,1:40]))
                std_C[iter] = std(log.(pre_def_stats_C[iter,1:40]))
                std_Y_t[iter] =  std(log.(pre_def_stats_Y_t[iter,1:40]))
                mean_B[iter] = mean(pre_def_stats_B[iter,1:40]./ pre_def_stats_Y_t[iter,1:40])
                mean_d[iter] = mean(pre_def_stats_d[iter,1:40]./ pre_def_stats_Y_t[iter,1:40])
                cor_D_change_Y_t[iter] = cor(pre_def_stats_d_change[iter,1:40], log.(pre_def_stats_Y_t[iter,1:40]))[1]
                d_non_zero[iter] = sum(pre_def_stats_d[iter, 1:40].>1e-6)/length(pre_def_stats_d[iter, 1:40])
                std_C_yearly[iter] = std(log.(pre_def_stats_C_t_yearly[iter,:])) 
                std_change_d_t_yearly[iter] = std(pre_def_stats_d_t_change_yearly[iter,:]) 
                std_Y_t_yearly[iter] = std(log.(pre_def_stats_Y_t_yearly[iter,:]))
                cor_D_change_Y_t_yearly[iter] =  cor(pre_def_stats_d_t_change_yearly[iter,:]./log.(pre_def_stats_Y_t_yearly[iter,:]), log.(pre_def_stats_Y_t_yearly[iter,:]))[1]
            end 
            if(iter == n_chosen_def)
                
                break
            end
       
            
    end
    

   #calibration targets 
    #debt_tradeables = mean(pre_def_stats_B[32:74]./pre_def_stats_Y_t[32:74])
    
    stats[1,:] = median(pre_def_stats_Y_t, dims =  1)
    stats[2,:] = median( pre_def_stats_R, dims = 1)
    stats[3,:] = median(pre_def_stats_B, dims = 1)
    stats[4,:] = median(pre_def_stats_C_t, dims = 1)
    stats[5,:] = median(ϵ_stats, dims = 1)
    stats[6,:] = median(pre_def_stats_C_n, dims = 1)
    
    stats[7,:] = median(pre_def_stats_P, dims = 1)
    stats[8,:] = median(pre_def_stats_D, dims = 1)
    stats[9,:] = median(pre_def_stats_h, dims = 1)
    stats[10,:] = median(pre_def_stats_d, dims = 1)
    stats[11,:] = median( pre_def_stats_R_d,dims = 1 )
    
    
    output_loss = -1
    #d_positive = sum(pre_def_stats_d[:, 32:75].>0)/(n_chosen_def*(75.0-32.0))
    println("calibration results: ", " default probability: ", Def_prob)    
    println("calibration results: ", " consumption std: ", NaNMath.median(std_C_yearly./std_Y_t_yearly))  
    println("debt to tradeable: " , NaNMath.median(mean_B))      
    println("mean spread: " , NaNMath.median(mean_R))      
    println("std spread: ", NaNMath.median(std_R))
    println("mean spread: IMF " ,  NaNMath.median(mean_R_IMF))  
    println("std spread: IMF " ,  NaNMath.median(std_R_IMF) )
    println("std change d: " ,  NaNMath.median(std_change_d_t_yearly) )
    println("cor d change,y  " ,  NaNMath.median(cor_D_change_Y_t_yearly)) 
    println("d higher than 0: ", NaNMath.median(d_non_zero) )
    println("mean d : ", NaNMath.median(mean_d))
    println("ratio d tp B ", NaNMath.median(mean_d)/NaNMath.median(mean_B.+mean_d))

    moments = [Def_prob, NaNMath.median(mean_B), NaNMath.median(mean_R), NaNMath.median(std_R), NaNMath.median(mean_R_IMF), NaNMath.median(std_R_IMF), NaNMath.median(std_C_yearly./std_Y_t_yearly), NaNMath.median(d_non_zero), NaNMath.median(mean_d)]
    
    if ERR == "float"
       
        writedlm( "output/B_debt_float.csv", B_hist_ab , ',')
        writedlm( "output/d_debt_float.csv", d_hist_ab , ',') 
    else

        writedlm( "output/B_debt_peg.csv", B_hist_ab , ',') 
        writedlm( "output/d_debt_peg.csv", d_hist_ab , ',') 
    end

    return (Def_prob, NaNMath.median(mean_B), output_loss, stats, moments)
end

######################################################################################
#fast solution plotting
#######################################################################################
function plot_model(Model_solution, stats; ERR ="peg")
    q = Model_solution[1]
    q_spline = Array{Interpolations.ScaledInterpolation}(undef, n_y)
    for j in 1:n_y
        q_spline[j] =  Interpolations.scale(interpolate(q[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
    end
    mean_d = 0.003122815106919373
    Policy_B = Model_solution[3]
    Policy_d  = Model_solution[4]
    V_f = Model_solution[5]
    Def_bord = Model_solution[6]
    V_d = Model_solution[7]
    for j in 1:n_y
        contour(grid_d, grid_B, V_f[j,:,:],fill=true)
        png("output/val_function_2d_$j")

        contour(grid_d, grid_B, Policy_d[1,j,:,:],fill=true)
        png("output/policy_function_2d_d_$j")

        contour(grid_d, grid_B, Policy_B[1,j,:,:],fill=true)
        png("output/policy_function_2d_B_$j")

        contour(grid_d, grid_B, q[j,:,:],fill=true)
        png("output/q_function_2d_$j")

        plot(grid_d,  V_d[j,:],fill=true)
        png("output/v_d$j")
    end


    p1 = plot(-15.0:1.0:14.0, stats[3,25:54], ylabel = "B'", xlabel = "T")
    p2 = plot(-15.0:1.0:14.0,stats[1,25:54], ylabel = "Y_t", xlabel = "T")
    p3 = plot(-15.0:1.0:14.0, stats[7,25:54], ylabel = "P_n/P_t", xlabel = "T")
    p4 = plot(-15.0:1.0:14.0, stats[4,25:54], ylabel = "C_t", xlabel = "T")
    p5 = plot(-15.0:1.0:14.0, stats[6,25:54], ylabel = "C_n'", xlabel = "T")
    p6 = plot(-15.0:1.0:14.0,stats[10,25:54], ylabel = "d", xlabel = "T")
    p7 = plot(-15.0:1.0:14.0,stats[2,25:54], ylabel = "R", xlabel = "T")
    p8 = plot(-15.0:1.0:14.0,stats[11,25:54], ylabel = "R_d", xlabel = "T")
    if ERR == "float"
        #writedlm( "output/results_pre_def_2D_float.csv",  stats, ',')  
        plot(p1,p2, p3,p4,p5, p6,p7,p8, layout = (4, 2))
        png("output/pre_def_stats_2D_float")
    else
        #writedlm( "output/results_pre_def_2D_peg1.csv",  stats, ',')  
        plot(p1,p2, p3,p4,p5, p6,p7,p8, layout = (4, 2))
        png("output/pre_def_stats_2D_peg")
    end
    chosen_j = [5,9,12,16]
    q_func = zeros(5, n_B)
    q_func[1,:] = grid_B
    for j in 2:length( chosen_j)+1
        for ii in 1: n_B
            q_func[j,ii] = q_spline[j](grid_B[ii],mean_d)
        end
    end 
    if ERR == "float"
        writedlm( "output/q_func_float.csv",  q_func, ',')  
    else
        writedlm( "output/q_func_peg.csv",  q_func, ',')  
    end   
    
end

##################################################################################
#functions to solve the model
#######################################################################################
function run_2D(non_calib::TD_assumed,calib::TD_calib, grid_param::TD_gird;ERR ="peg")

    println("Itertations starts")

    unpack_params(non_calib, calib,grid_param )
    if(ERR == "float")
        global ERR_glob = "float"
        Model_sol = Solve_eq_peg2D(200)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="float")
        #save("/output/float_val.jl", "Model_sol", Model_sol)
        plot_model(Model_sol, stats, ERR = "float")
        save("/output/float_val.jld", "Model_sol", Model_sol)
    elseif(ERR == "peg")
        global ERR_glob = "peg"
        Model_sol = Solve_eq_peg2D(500)

        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="peg")
        
        plot_model(Model_sol, stats, ERR = "peg")
        save("/output/peg_val.jld","Model_sol", Model_sol)

    else
        throw(DomainError(method, "this exchange rate regime is not supported, try float or peg "))
    end
   
    
end

function run_2D_calib(non_calib::TD_assumed,calib::TD_calib, grid_param::TD_gird;ERR ="peg")


    unpack_params(non_calib, calib,grid_param )
    println("checking: ", "d1 = ",δ_1, "d2 = ", δ_2, "pen = ", pen, "k = ", k )

    if(ERR == "float")
        global ERR_glob = "float"
        Model_sol = Solve_eq_peg2D(250)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="float")
    elseif(ERR == "peg")
        global ERR_glob = "peg"
        Model_sol = Solve_eq_peg2D(250)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="peg")
    else
        throw(DomainError(method, "this exchange rate regime is not supported, try float or peg "))
        moments = zeros(9)
    end
    return(moments)
end
    
#####################################################################
#special function for SMM (not finally used)
######################################################################
function solver_run_eq(non_calib::TD_assumed,calib::TD_calib, grid_param::TD_gird;ERR ="peg")
    unpack_params(non_calib, calib,grid_param )
    println("checking: ", "d1 = ",δ_1, "d2 = ", δ_2, "pen = ", pen, "k = ", k )

    if(ERR == "float")
        Model_sol = Solve_eq_float2D(100)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="float")
    elseif(ERR == "peg")
        Model_sol = Solve_eq_peg2D(200)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="peg")
    else
        throw(DomainError(method, "this exchange rate regime is not supported, try float or peg "))
        moments = zeros(9)
    end
    def_prob = moments[1]
    spread_IMF = moments[5]
    if isnan(spread_IMF)
        spread_IMF = 10000
    end
    debt_IMF_ratio = moments[9]/moments[2]
    if isnan(debt_IMF_ratio)
        debt_IMF_ratio = 10
    end
    non_zero_debt = moments[8]
    loss = ((100*def_prob -3.0))^2 + 2*(100*spread_IMF-1.5)^2 +2*(100*debt_IMF_ratio-4.6)^2+ (10*non_zero_debt-0.75)^2
    return(loss)
end

function wrapper_SMM(x::Vector, grad::Vector )
    del_1 = x[1]
    del_2 = x[2]
    pen_prop = x[3]
    k_prop = x[4]
    #moments = run_2D_calib(TD_assumed(),TD_calib(δ_1 = del_1,δ_2 = del_2,k = k_prop, pen = pen_prop), TD_gird(), ERR ="peg")

    loss = solver_run_eq(TD_assumed(),TD_calib(δ_1 = del_1,δ_2 = del_2,k = k_prop, pen = pen_prop), TD_gird(), ERR ="peg")

    println(loss)
    return loss
end


function SMM(f, x_0)
    opt = Opt(:LN_BOBYQA , 4) #LN_BOBYQA , LN_NEWUOA, LN_COBYLA
    opt.lower_bounds = [0.2, 0.5, 0.0, 0.05]
    opt.upper_bounds = [5.0, 6.0, 0.05, 2.0]
    opt.xtol_rel = 1e-2
    opt.min_objective = f
    opt.maxeval = 250
    #opt.initial_step = 1e-3 #(grid_B[n_B]-grid_B[1])/500.0
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end

function run_SMM(x_0)
    (minf,minx,ret) = SMM(wrapper_SMM, x_0)
    return (minf,minx,ret)
end

#####################################################################################################################
#Special simulations (default epiosode of IRF computations)
#######################################################################################################################

function simulate_IRF( Model_solution_1, Model_solution_2; burnout = 1000, t_sim =1200, n_sim = 100000, ERR="float", method = "spline_sch", t_break = 1100)
    global P_stat, y_t_stat = DiscretizeAR(ρ, σ_μ, n_y, "tauchen")
    
    #unpack policy functions and paramters
    n_B_long =10000
    q = Model_solution_1[1]
    #Def_mat = Model_solution[2]
    Policy_B = Model_solution_1[3]
    Policy_d  = Model_solution_1[4]
    Def_bord = Model_solution_1[6]
    Policy_def = Model_solution_1[8]

    q_2 = Model_solution_2[1]
    Policy_B_2 = Model_solution_2[3]
    Policy_d_2  = Model_solution_2[4]
    Def_bord_2 = Model_solution_2[6]
    Policy_def_2 = Model_solution_2[8]



    grid_B_long = grid_B[1]:(grid_B[n_B] - grid_B[1])/(n_B_long-1):grid_B[n_B]
    Def_matrix = zeros(n_y, n_B_long, n_d)
    for j in 1:n_y
        for id in 1:n_d
            for i in 1:n_B_long
                if(Def_bord[j,id]>= grid_B_long[i])
                    Def_matrix[j,i,id] =1.0 
                end
            end
        end
    end

    Def_matrix_2 = zeros(n_y, n_B_long, n_d)
    for j in 1:n_y
        for id in 1:n_d
            for i in 1:n_B_long
                if(Def_bord_2[j,id]>= grid_B_long[i])
                    Def_matrix_2[j,i,id] =1.0 
                end
            end
        end
    end
    
    
    #allocate memory
    
    SIM = zeros(Int64, n_sim, t_sim) #simulated income process
    B_hist = zeros(n_sim, t_sim) #private debt history
    d_hist = zeros(n_sim, t_sim) #assets history for the non-defautable debt
    Y_t = zeros(n_sim, t_sim) # output history
    Trade_B = zeros(n_sim, t_sim) # trade balance history
    
    C_t = zeros(n_sim, t_sim) # consumption history tradeables
    C_n = ones(n_sim, t_sim) # non-tradeables
    h_t = ones(n_sim, t_sim)
    C = zeros(n_sim, t_sim) # final good
    
    D = zeros(n_sim, t_sim) #Defaults histiry
    D_state = zeros(n_sim, t_sim) #exclusion from the financial market histiry
    R = zeros(n_sim, t_sim) #q hiatory
    R_d = zeros(n_sim, t_sim)


    B_hist_2 = zeros(n_sim, t_sim) #private debt history
    d_hist_2 = zeros(n_sim, t_sim) #assets history for the non-defautable debt
    Y_t_2 = zeros(n_sim, t_sim) # output history
    Trade_B_2 = zeros(n_sim, t_sim) # trade balance history
    
    C_t_2 = zeros(n_sim, t_sim) # consumption history tradeables
    C_n_2 = ones(n_sim, t_sim) # non-tradeables
    h_t_2 = ones(n_sim, t_sim)
    C_2 = zeros(n_sim, t_sim) # final good
    
    D_2 = zeros(n_sim, t_sim) #Defaults histiry
    D_state_2 = zeros(n_sim, t_sim) #exclusion from the financial market histiry
    R_2 = zeros(n_sim, t_sim) #q hiatory
    R_d_2 = zeros(n_sim, t_sim)
    
    #exchange rate and minimal wage
   
    ϵ = ones(n_sim, t_sim)
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    
    policy_B_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_d_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_def_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    q_spline = Array{Interpolations.ScaledInterpolation}(undef, n_y)
    Def_bord_itp =  Array{Interpolations.Extrapolation}(undef,n_y)

    policy_B_spline_2 = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_d_spline_2 = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_def_spline_2 = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    q_spline_2 = Array{Interpolations.ScaledInterpolation}(undef, n_y)
    Def_bord_itp_2 =  Array{Interpolations.Extrapolation}(undef,n_y)
    #interpolate policy functions
    for j in 1:n_y
        q_spline[j] =  Interpolations.scale(interpolate(q[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_B_spline[j] =  Interpolations.scale(interpolate(Policy_B[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_d_spline[j] = Interpolations.scale(interpolate(Policy_d[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d)
        policy_def_spline[j] = Interpolations.scale(interpolate(Policy_def[j,:], BSpline(Cubic(Line(OnGrid())))), grid_d)
        Def_bord_itp[j] = LinearInterpolation((grid_B_long, grid_d), Def_matrix[j,:,:], extrapolation_bc = Periodic()) #def bord is given for a id points co that is why I change the 

        q_spline_2[j] =  Interpolations.scale(interpolate(q_2[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_B_spline_2[j] =  Interpolations.scale(interpolate(Policy_B_2[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_d_spline_2[j] = Interpolations.scale(interpolate(Policy_d_2[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d)
        policy_def_spline_2[j] = Interpolations.scale(interpolate(Policy_def_2[j,:], BSpline(Cubic(Line(OnGrid())))), grid_d)
        Def_bord_itp_2[j] = LinearInterpolation((grid_B_long, grid_d), Def_matrix_2[j,:,:], extrapolation_bc = Periodic())
    end

    #define starting values
    y_0 = Int(floor(n_y/2) ) # start with 11th state
    Y_0 = y_t_stat[y_0]*ones(n_sim) #output start value
    B_0 = zeros(n_sim) #start with 0 assets
    d_0 = zeros(n_sim)
    Y_t[:,1] = Y_0 
    B_hist[:,1] = B_0
    d_hist[:,1] = d_0

    Default_flag = 0.0 # flag if the country was excluded from financial market in the previous period
    Default_flag_2 = 0.0
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
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    d_hist[i,t] = min(max(policy_def_spline[SIM[i,t]]( d_hist[i,t-1])), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t] + q_d(d_hist[i,t])*d_hist[i,t] - d_hist[i,t-1]
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0
                        end
                        R_d[i,t] = -1.0
                    end

                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    B_hist[i,t] = 0.0
                    
                    R[i,t] = -1.0 #nop interest rate
                    D[i,t] = 0.0
                    D_state[i,t] = 1.0
                else
                    
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = max(policy_B_spline[SIM[i,t]](0.0, d_hist[i,t-1]), grid_B[1])
                    d_hist[i,t] = min(max(policy_d_spline[SIM[i,t]](0.0,d_hist[i,t-1]),grid_d[1]), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t] + q_spline[SIM[i,t]](0.0, 0.0)*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t] - d_hist[i,t-1] 
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0  
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0  
                        end
                        R_d[i,t] = -1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 
                    if(B_hist[i,t]>1e-6)
                        R[i,t] = 1.0/q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t]) - 1.0
                    else
                        R[i,t] = -1
                    end
                    
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                    
                end
            else
                if Def_bord_itp[SIM[i,t]](B_hist[i,t-1],d_hist[i,t-1] ) <1.0  #case of default
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    d_hist[i,t] = min(max(policy_def_spline[SIM[i,t]](d_hist[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t]  - d_hist[i,t-1] + q_d(d_hist[i,t])*d_hist[i,t]
                    C[i,t] = final_good(C_t[i,t],1.0) 
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                           
                        end
                        R_d[i,t] = -1.0
                    end
                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    B_hist[i,t] = 0.0
                    R[i,t] = -1.0 #1.0/q[SIM[i,t],A_int[i,t]] - 1.0
                    D[i,t] = 1.0
                    D_state[i,t] = 1.0
                    Default_flag = 1.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                else
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = max(min(policy_B_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1]), grid_B[n_B]), grid_B[1])
                    d_hist[i,t] = min(max(policy_d_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1]),0.0), grid_d[n_d])
                    C_t[i,t] =max(Y_t[i,t]- B_hist[i,t-1] - d_hist[i,t-1]+ q_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1])*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t], 1e-6)
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) - 1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)-pen
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] =  - 1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 

                    if(B_hist[i,t]>1e-6)
                        R[i,t] = 1.0/q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t]) - 1.0
                    else
                        R[i,t] = -1
                    end
                   
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                end
           end
           if α * (C_t[i,t])^(1.0/ζ)* (1.0-a)/a < w_bar && ERR == "float"
                ϵ[i,t] = w_bar/((α * (C_t[i,t])^(1.0/ζ))* (1.0-a)/a)
           end

           #now the post_reform simulations 

           if t<=t_break #before ifi policy change 
            B_hist_2[i,t] = B_hist[i,t]
            d_hist_2[i,t]= d_hist[i,t] #assets history for the non-defautable debt
            Y_t_2[i,t] = Y_t[i,t] # output history
            Trade_B_2[i,t] = Trade_B[i,t] # trade balance history
            
            C_t_2[i,t] = C_t[i,t] # consumption history tradeables
            C_n_2[i,t] = C_n[i,t] # non-tradeables
            h_t_2[i,t] = h_t[i,t]
            C_2[i,t] = C[i,t] # final good
            
            D_2[i,t] = D[i,t] #Defaults histiry
            D_state_2[i,t] = D_state[i,t] #exclusion from the financial market histiry
            R_2[i,t] = R[i,t] #q hiatory
            R_d_2[i,t] =  R_d[i,t]
            Default_flag_2 = Default_flag
           else #after t_break: new policy starts in
            if(Default_flag_2==1.0) 
                simi = simulate_exclusion(θ)
                if(simi==1)
                    Default_flag_2 = 0.0
                else
                    Default_flag_2 = 1.0
                end
                if(Default_flag_2==1.0)
                    Y_t_2[i,t] = y_t_stat[SIM[i,t]]
                    d_hist_2[i,t] = min(max(policy_def_spline_2[SIM[i,t]](d_hist_2[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t_2[i,t] = Y_t_2[i,t] + q_d(d_hist_2[i,t])*d_hist_2[i,t] - d_hist_2[i,t-1]
                    if d_hist_2[i,t]>1e-6
                        R_d_2[i,t] = 1.0/q_d(d_hist_2[i,t]) -1.0
                    else
                        R_d_2[i,t] = -1.0
                    end

                    if C_t_2[i,t] < c_t_min && ERR =="peg"
                        C_n_2[i,t] =  max(min(copy(C_t_2[i,t])^(power)*const_w,1.0),1e-6)
                        h_t_2[i,t] = C_n_2[i,t]^(1.0/α)
                    else
                        C_n_2[i,t] = 1.0 
                    end

                    C_2[i,t] = final_good(C_t_2[i,t],C_n_2[i,t]) 
                    Trade_B_2[i,t] = (Y_t_2[i,t] - C_t_2[i,t])/Y_t[i,t]
                    B_hist_2[i,t] = 0.0
                    
                    R_2[i,t] = -1.0 #nop interest rate
                    D_2[i,t] = 0.0
                    D_state_2[i,t] = 1.0
                else
                    
                    Y_t_2[i,t] = y_t_stat[SIM[i,t]]
                    B_hist_2[i,t] = max(policy_B_spline_2[SIM[i,t]](0.0,  d_hist_2[i,t-1]), grid_B[1])
                    d_hist_2[i,t] = min(max(policy_d_spline_2[SIM[i,t]](0.0, d_hist_2[i,t-1]),grid_d[1]), grid_d[n_d])
                    C_t_2[i,t] = Y_t_2[i,t] + q_spline_2[SIM[i,t]](B_hist_2[i,t],  d_hist_2[i,t])*B_hist_2[i,t] + q_d(d_hist_2[i,t])*d_hist_2[i,t] - d_hist_2[i,t-1] 
                    if d_hist_2[i,t]>1e-6
                        R_d_2[i,t] = 1.0/q_d(d_hist_2[i,t]) -1.0
                    else
                        R_d_2[i,t] = -1.0
                    end

                    if C_t_2[i,t] < c_t_min && ERR =="peg"
                        C_n_2[i,t] =  max(min(copy(C_t_2[i,t])^(power)*const_w,1.0),1e-6)
                        h_t_2[i,t] = C_n_2[i,t]^(1.0/α)
                    else
                        C_n_2[i,t] = 1.0 
                    end
                    C_2[i,t] = final_good(C_t_2[i,t], C_n_2[i,t]) 
                    if(B_hist_2[i,t]>1e-6)
                        R_2[i,t] = 1.0/q_spline_2[SIM[i,t]](B_hist_2[i,t], d_hist_2[i,t]) - 1.0
                    else
                        R_2[i,t] = -1
                    end
                    
                    D_2[i,t] = 0.0
                    D_state_2[i,t] = 0.0
                    Trade_B_2[i,t] = (Y_t_2[i,t] - C_t_2[i,t])/Y_t_2[i,t]
                    Default_flag_2 = 0.0
                    
                end
            else
                if Def_bord_itp_2[SIM[i,t]](B_hist_2[i,t-1],d_hist_2[i,t-1] ) <1.0  #case of default
                    Y_t_2[i,t] = y_t_stat[SIM[i,t]]
                    d_hist_2[i,t] = min(max(policy_def_spline_2[SIM[i,t]](d_hist_2[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t_2[i,t] = Y_t_2[i,t]  - d_hist_2[i,t-1] + q_d(d_hist_2[i,t])*d_hist_2[i,t]
                        if C_t_2[i,t] < c_t_min && ERR =="peg"
                            C_n_2[i,t] =  max(min(copy(C_t_2[i,t])^(power)*const_w,1.0),1e-6)
                            h_t_2[i,t] = C_n_2[i,t]^(1.0/α)
                        else
                            C_n_2[i,t] = 1.0 
                        end
                        if d_hist_2[i,t]>1e-6 
                            R_d_2[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                        else
                            R_d_2[i,t] = -1.0
                        end
                   
                    C_2[i,t] = final_good(C_t_2[i,t],C_n_2[i,t]) 
                    B_hist_2[i,t] = 0.0
                    R_2[i,t] = -1.0 #1.0/q[SIM[i,t],A_int[i,t]] - 1.0
                    D_2[i,t] = 1.0
                    D_state_2[i,t] = 1.0
                    Default_flag_2 = 1.0
                    Trade_B_2[i,t] = (Y_t_2[i,t] - C_t_2[i,t])/Y_t_2[i,t]
                else
                    Y_t_2[i,t] = y_t_stat[SIM[i,t]]
                    B_hist_2[i,t] = max(min(policy_B_spline_2[SIM[i,t]](B_hist_2[i,t-1], d_hist_2[i,t-1]), grid_B[n_B]), grid_B[1])
                    d_hist_2[i,t] = min(max(policy_d_spline_2[SIM[i,t]](B_hist_2[i,t-1], d_hist_2[i,t-1]),0.0), grid_d[n_d])
                    C_t_2[i,t] =max(Y_t_2[i,t]- B_hist_2[i,t-1] - d_hist_2[i,t-1]+ q_spline[SIM[i,t]](B_hist_2[i,t], d_hist_2[i,t])*B_hist_2[i,t] + q_d(d_hist_2[i,t])*d_hist_2[i,t], 1e-6)
                    if C_t_2[i,t] < c_t_min && ERR =="peg"
                        C_n_2[i,t] =  max(min(copy(C_t_2[i,t])^(power)*const_w,1.0),1e-6)
                        h_t_2[i,t] = C_n_2[i,t]^(1.0/α)
                    else
                        C_n_2[i,t] = 1.0  
                    end
                    if d_hist[i,t]>1e-6 
                        R_d_2[i,t] = 1.0/q_d(d_hist_2[i,t]) -1.0
                    else
                        R_d_2[i,t] = -1.0
                    end
                    C_2[i,t] = final_good(C_t_2[i,t], C_n_2[i,t]) 

                    if(B_hist_2[i,t]>1e-6)
                        R_2[i,t] = 1.0/q_spline_2[SIM[i,t]](B_hist_2[i,t], d_hist_2[i,t]) - 1.0
                    else
                        R_2[i,t] = -1
                    end
                   
                    D_2[i,t] = 0.0
                    D_state_2[i,t] = 0.0
                    Trade_B_2[i,t] = (Y_t_2[i,t] - C_t_2[i,t])/Y_t_2[i,t]
                    Default_flag_2 = 0.0
                end
           end

           end

        end
    end

    #compute stats for defaults
    n_defaults = sum(D[:, burnout:t_sim])
    
    non_defaults = D_state[:,burnout:t_sim].<1.0   

    #default probability
    Def_prob  =  1.0 - (1.0-n_defaults/(n_sim*(t_sim-burnout)))^4.0   
    n_chosen_def = 100000

    #stats after default:
    Y_ab = Y_t[:,burnout:t_sim]
    R_ab = R[:,burnout:t_sim]
    R_d_ab = R_d[:,burnout:t_sim]
    B_hist_ab = B_hist[:,burnout:t_sim]
    d_hist_ab = d_hist[:,burnout:t_sim]
    D_ab = D[:,burnout:t_sim]
    Trade_B_ab = Trade_B[:,burnout:t_sim]
    D_state_ab = D_state[:,burnout:t_sim]
    C_ab = C[:,burnout:t_sim]
    C_ab_t = C_t[:,burnout:t_sim]
    C_ab_n = C_n[:,burnout:t_sim]
    h_t_ab = h_t[:,burnout:t_sim]


    Y_ab_2 = Y_t_2[:,burnout:t_sim]
    R_ab_2 = R_2[:,burnout:t_sim]
    R_d_ab_2 = R_d_2[:,burnout:t_sim]
    B_hist_ab_2 = B_hist_2[:,burnout:t_sim]
    d_hist_ab_2 = d_hist_2[:,burnout:t_sim]
    D_ab_2 = D_2[:,burnout:t_sim]
    Trade_B_ab_2 = Trade_B_2[:,burnout:t_sim]
    D_state_ab_2 = D_state_2[:,burnout:t_sim]
    C_ab_2 = C_2[:,burnout:t_sim]
    C_ab_t_2 = C_t_2[:,burnout:t_sim]
    C_ab_n_2 = C_n_2[:,burnout:t_sim]
    h_t_ab_2 = h_t_2[:,burnout:t_sim]
    #choose number of defaults to compute statistics

    #allocate memory for the staoistics 
    pre_def_stats_Y_t = zeros(n_chosen_def , 56)
    pre_def_stats_R = zeros(n_chosen_def , 56)   
    pre_def_stats_R_d = zeros(n_chosen_def , 56)        
    pre_def_stats_TB = zeros(n_chosen_def , 56)        
    pre_def_stats_C_t = zeros(n_chosen_def , 56) 
    pre_def_stats_C_n = zeros(n_chosen_def , 56) 
    pre_def_stats_C = zeros(n_chosen_def , 56) 
    pre_def_stats_B = zeros(n_chosen_def , 56)
    pre_def_stats_d = zeros(n_chosen_def , 56)
    pre_def_stats_h = zeros(n_chosen_def , 56)
    pre_def_stats_D  = zeros(n_chosen_def , 56)

    stats = zeros(11, 56)
    cor_D_change_Y_t_yearly = zeros(n_chosen_def)
    #compute statistics for 74 periods before the default, for n_def defaults (without any defults in the 74 periods before)
    iter = 0
    #for i in 1:n_sim, t in 76:t_sim-burnout
    for i in 1:n_sim
        t= 140
          if(t>=140  )
                 iter =iter+1   
                
                 pre_def_stats_R[iter,:] = R_ab[i, t-40:t+15]- R_ab_2[i, t-40:t+15]
                 pre_def_stats_B[iter,:] = B_hist_ab[i, t-40:t+15] -  B_hist_ab_2[i, t-40:t+15]
                 pre_def_stats_R_d[iter,:] = R_d_ab[i, t-40:t+15]-  R_d_ab_2[i, t-40:t+15]
                 pre_def_stats_d[iter,:] = d_hist_ab[i, t-40:t+15]-  d_hist_ab_2[i, t-40:t+15]
                 
                
                 pre_def_stats_TB[iter,:] = Trade_B_ab[i, t-40:t+15]-  Trade_B_ab_2[i, t-40:t+15]
                 pre_def_stats_C_t[iter,:] = C_ab_t[i, t-40:t+15]-  C_ab_t_2[i, t-40:t+15]
                 pre_def_stats_C_n[iter,:] = C_ab_n[i, t-40:t+15]-  C_ab_n_2[i, t-40:t+15]
                 pre_def_stats_C[iter,:] = C_ab[i, t-40:t+15]-  C_ab_2[i, t-40:t+15]
                 
                 pre_def_stats_h[iter,:] = h_t_ab[i, t-40:t+15]  -  h_t_ab_2[i, t-40:t+15]
                 pre_def_stats_D[iter,:] =  D_state_ab[i, t-40:t+15] -  D_state_ab_2[i, t-40:t+15]

            
            end 
            if(iter == n_chosen_def)
                
                break
            end
       
            
    end
    

   #calibration targets 
    #debt_tradeables = mean(pre_def_stats_B[32:74]./pre_def_stats_Y_t[32:74])
    
    stats[1,:] = mean(pre_def_stats_Y_t, dims =  1)
    stats[2,:] = mean( pre_def_stats_R, dims = 1)
    stats[3,:] = mean(pre_def_stats_B, dims = 1)
    stats[4,:] = mean(pre_def_stats_C_t, dims = 1)
    
    stats[6,:] = mean(pre_def_stats_C_n, dims = 1)
    
    
    stats[8,:] = mean(pre_def_stats_D, dims = 1)
    stats[9,:] = mean(pre_def_stats_h, dims = 1)
    stats[10,:] = mean(pre_def_stats_d, dims = 1)
    stats[11,:] = mean( pre_def_stats_R_d,dims = 1 )
    
    
    output_loss = -1
    moments = zeros(10)
    if ERR == "float"
        writedlm( "output/results_pre_def_2D_float_IRF.csv",  stats, ',')  
        writedlm( "output/B_debt_float.csv", B_hist_ab , ',')
        writedlm( "output/d_debt_float.csv", d_hist_ab , ',') 
        
    else
        writedlm( "output/results_pre_def_2D_peg_IRF.csv",  stats, ',')  
        writedlm( "output/B_debt_peg.csv", B_hist_ab , ',') 
        writedlm( "output/B_debt_peg_2.csv", B_hist_ab_2 , ',') 
        writedlm( "output/d_debt_peg.csv", d_hist_ab , ',') 
    end
    println("IRF computed")
    return (Def_prob, 1.0, output_loss, stats, moments)
end

function simulate_eq_2D_default( Model_solution; burnout = 10000, t_sim =100000, n_sim = 10, ERR="float", method = "spline_sch")
    global P_stat, y_t_stat = DiscretizeAR(ρ, σ_μ, n_y, "tauchen")
    
    #unpack policy functions and paramters
    n_B_long =10000
    q = Model_solution[1]
    #Def_mat = Model_solution[2]
    Policy_B = Model_solution[3]
    Policy_d  = Model_solution[4]
    Def_bord = Model_solution[6]
    Policy_def = Model_solution[8]
    grid_B_long = grid_B[1]:(grid_B[n_B] - grid_B[1])/(n_B_long-1):grid_B[n_B]
    Def_matrix = zeros(n_y, n_B_long, n_d)
    for j in 1:n_y
        for id in 1:n_d
            for i in 1:n_B_long
                if(Def_bord[j,id]>= grid_B_long[i])
                    Def_matrix[j,i,id] =1.0 
                end
            end
        end
    end
    
    
    #allocate memory
    
    SIM = zeros(Int64, n_sim, t_sim) #simulated income process
    B_hist = zeros(n_sim, t_sim) #private debt history
    d_hist = zeros(n_sim, t_sim) #assets history for the non-defautable debt
    Y_t = zeros(n_sim, t_sim) # output history
    Trade_B = zeros(n_sim, t_sim) # trade balance history
    
    C_t = zeros(n_sim, t_sim) # consumption history tradeables
    C_n = ones(n_sim, t_sim) # non-tradeables
    h_t = ones(n_sim, t_sim)
    C = zeros(n_sim, t_sim) # final good
    
    D = zeros(n_sim, t_sim) #Defaults histiry
    D_state = zeros(n_sim, t_sim) #exclusion from the financial market histiry
    R = zeros(n_sim, t_sim) #q hiatory
    R_d = zeros(n_sim, t_sim)
    #exchange rate and minimal wage
   
    ϵ = ones(n_sim, t_sim)
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    
    policy_B_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_d_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_def_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    q_spline = Array{Interpolations.ScaledInterpolation}(undef, n_y)
    Def_bord_itp =  Array{Interpolations.Extrapolation}(undef,n_y)
    #interpolate policy functions
    for j in 1:n_y
        q_spline[j] =  Interpolations.scale(interpolate(q[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_B_spline[j] =  Interpolations.scale(interpolate(Policy_B[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_d_spline[j] = Interpolations.scale(interpolate(Policy_d[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d)
        policy_def_spline[j] = Interpolations.scale(interpolate(Policy_def[j,:], BSpline(Cubic(Line(OnGrid())))), grid_d)
        Def_bord_itp[j] = LinearInterpolation((grid_B_long, grid_d), Def_matrix[j,:,:], extrapolation_bc = Periodic()) #def bord is given for a id points co that is why I change the 
    end

    #define starting values
    y_0 = Int(floor(n_y/2) ) # start with 11th state
    Y_0 = y_t_stat[y_0]*ones(n_sim) #output start value
    B_0 = zeros(n_sim) #start with 0 assets
    d_0 = zeros(n_sim)
    Y_t[:,1] = Y_0 
    B_hist[:,1] = B_0
    d_hist[:,1] = d_0

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
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    d_hist[i,t] = min(max(policy_def_spline[SIM[i,t]](d_hist[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t] + q_d(d_hist[i,t])*d_hist[i,t] - d_hist[i,t-1]
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0  
                        end
                        R_d[i,t] = -1.0
                    end

                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    B_hist[i,t] = 0.0
                    
                    R[i,t] = -1.0 #nop interest rate
                    D[i,t] = 0.0
                    D_state[i,t] = 1.0
                else
                    
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = max(policy_B_spline[SIM[i,t]](0.0, d_hist[i,t-1] ), grid_B[1])
                    d_hist[i,t] = min(max(policy_d_spline[SIM[i,t]](0.0, d_hist[i,t-1]),0.0), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t] + q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t])*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t] - d_hist[i,t-1] 
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0  
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0  
                        end
                        R_d[i,t] = -1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 
                    if(B_hist[i,t]>1e-6)
                        R[i,t] = 1.0/q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t]) - 1.0
                    else
                        R[i,t] = -1
                    end
                    
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                    
                end
            else
                if Def_bord_itp[SIM[i,t]](B_hist[i,t-1],d_hist[i,t-1] ) <1.0  #case of default
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    d_hist[i,t] =  min(max(policy_def_spline[SIM[i,t]](d_hist[i,t-1]), grid_d[1]), grid_d[n_d])
                    C_t[i,t] = Y_t[i,t]  - d_hist[i,t-1] + q_d(d_hist[i,t])*d_hist[i,t]
                    C[i,t] = final_good(C_t[i,t],1.0) 
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) -1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                           
                        end
                        R_d[i,t] = -1.0
                    end
                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    B_hist[i,t] = 0.0
                    R[i,t] = -1.0 #1.0/q[SIM[i,t],A_int[i,t]] - 1.0
                    D[i,t] = 1.0
                    D_state[i,t] = 1.0
                    Default_flag = 1.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                else
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = max(min(policy_B_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1]), grid_B[n_B]), grid_B[1])
                    d_hist[i,t] = min(max(policy_d_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1]),0.0), grid_d[n_d])
                    C_t[i,t] =max(Y_t[i,t]- B_hist[i,t-1] - d_hist[i,t-1]+ q_spline[SIM[i,t]](0.0, 0.0)*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t], 1e-6)
                    if d_hist[i,t]>1e-6
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0 
                        end
                        R_d[i,t] = 1.0/q_d(d_hist[i,t]) - 1.0
                    else
                        if C_t[i,t] < c_t_min && ERR =="peg"
                            C_n[i,t] =  max(min(copy(C_t[i,t])^(power)*const_w,1.0),1e-6)
                            h_t[i,t] = C_n[i,t]^(1.0/α)
                        else
                            C_n[i,t] = 1.0
                        end
                        R_d[i,t] =  - 1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 

                    if(B_hist[i,t]>1e-6)
                        R[i,t] = 1.0/q_spline[SIM[i,t]](B_hist[i,t], d_hist[i,t]) - 1.0
                    else
                        R[i,t] = -1
                    end
                   
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
    
    non_defaults = D_state[:,burnout:t_sim].<1.0   

    #default probability
    Def_prob  =  1.0 - (1.0-n_defaults/(n_sim*(t_sim-burnout)))^4.0   
    n_chosen_def = 1000

    #stats after default:
    Y_ab = Y_t[:,burnout:t_sim]
    R_ab = R[:,burnout:t_sim]
    R_d_ab = R_d[:,burnout:t_sim]
    B_hist_ab = B_hist[:,burnout:t_sim]
    d_hist_ab = d_hist[:,burnout:t_sim]
    D_ab = D[:,burnout:t_sim]
    ϵ_ab = ϵ[:,burnout:t_sim]
    Trade_B_ab = Trade_B[:,burnout:t_sim]
    D_state_ab = D_state[:,burnout:t_sim]
    C_ab = C[:,burnout:t_sim]
    C_ab_t = C_t[:,burnout:t_sim]
    C_ab_n = C_n[:,burnout:t_sim]
    h_t_ab = h_t[:,burnout:t_sim]
    #choose number of defaults to compute statistics

    #allocate memory for the staoistics 
    pre_def_stats_Y_t = zeros(n_chosen_def , 56)
    pre_def_stats_R = zeros(n_chosen_def , 56)   
    pre_def_stats_R_d = zeros(n_chosen_def , 56)        
    pre_def_stats_TB = zeros(n_chosen_def , 56)        
    pre_def_stats_C_t = zeros(n_chosen_def , 56) 
    pre_def_stats_C_n = zeros(n_chosen_def , 56) 
    pre_def_stats_C = zeros(n_chosen_def , 56) 
    pre_def_stats_B = zeros(n_chosen_def , 56)
    pre_def_stats_d = zeros(n_chosen_def , 56)
    pre_def_stats_d_change = zeros(n_chosen_def , 56)
    pre_def_stats_NB = zeros(n_chosen_def , 56)  
    pre_def_stats_P = zeros(n_chosen_def , 56)  
    pre_def_stats_D = zeros(n_chosen_def , 56) 
    pre_def_stats_changes_D = zeros(n_chosen_def , 56) 
    pre_def_stats_h = zeros(n_chosen_def , 56) 
    stats = zeros(11, 56)
    ϵ_stats = zeros(n_chosen_def , 56)  
    
    pre_def_stats_Y_t_yearly = zeros(n_chosen_def , 10)
    pre_def_stats_C_t_yearly = zeros(n_chosen_def , 10)
    pre_def_stats_d_t_change_yearly = zeros(n_chosen_def , 10)

    mean_R = zeros(n_chosen_def)
    std_R = zeros(n_chosen_def)
    mean_R_IMF = zeros(n_chosen_def)
    std_R_IMF = zeros(n_chosen_def)
    std_C = zeros(n_chosen_def)
    std_Y_t =  zeros(n_chosen_def)
    mean_B = zeros(n_chosen_def)
    mean_d = zeros(n_chosen_def)
    cor_D_change_Y_t = zeros(n_chosen_def)
    
    cor_R_Y_t = zeros(n_chosen_def)
    d_non_zero = zeros(n_chosen_def)

    std_C_yearly =  zeros(n_chosen_def)
    std_Y_t_yearly = zeros(n_chosen_def)
    std_change_d_t_yearly = zeros(n_chosen_def)
    cor_D_change_Y_t_yearly = zeros(n_chosen_def)
    #compute statistics for 74 periods before the default, for n_def defaults (without any defults in the 74 periods before)
    iter = 0
   
    #for i in 1:n_sim, t in 76:t_sim-burnout
    for i in 1:n_sim, t in 76:t_sim-burnout
        if(D_ab[i,t]==1 && sum(D_ab[i,t-40:t-1 ])== 0.0 && t<=t_sim-burnout-100 )
                 iter =iter+1   
                println(iter)
                 pre_def_stats_Y_t[iter,:] = Y_ab[i, t-40:t+15]
                 pre_def_stats_R[iter,:] = R_ab[i, t-40:t+15]
                 pre_def_stats_B[iter,:] = B_hist_ab[i, t-40:t+15]
                 pre_def_stats_R_d[iter,:] = R_d_ab[i, t-40:t+15]
                 pre_def_stats_d[iter,:] = d_hist_ab[i, t-40:t+15]
                 pre_def_stats_d_change[iter,:] = d_hist_ab[i, t-40:t+15] - d_hist_ab[i, t-41:t+14]
                 pre_def_stats_NB[iter,:] = B_hist_ab[i, t-40:t+15] #new debt
                 pre_def_stats_TB[iter,:] = Trade_B_ab[i, t-40:t+15]
                 pre_def_stats_C_t[iter,:] = C_ab_t[i, t-40:t+15]
                 pre_def_stats_C_n[iter,:] = C_ab_n[i, t-40:t+15]
                 pre_def_stats_C[iter,:] = C_ab[i, t-40:t+15]
                 pre_def_stats_P[iter,:] = (1-a)/a*(pre_def_stats_C_t[iter,:]./pre_def_stats_C_n[iter,:])
                 ϵ_stats[iter, :] = ϵ_ab[i, t-40:t+15] 
                 pre_def_stats_h[iter,:] = h_t_ab[i, t-40:t+15] 
                 pre_def_stats_D[iter,:] =  D_state[i, t-40:t+15]

                 k=0
                for tt in 1:40
                    
                 
                    if(mod(tt,4)==0)
                        k = k+1
                        pre_def_stats_Y_t_yearly[i,k ] = sum(pre_def_stats_Y_t[i,tt-3:tt])
                        
                        pre_def_stats_C_t_yearly[i,k ] = sum(pre_def_stats_C[i,tt-3:tt])
                        pre_def_stats_d_t_change_yearly[i,k] = d_hist_ab[i, tt] - d_hist_ab[i, tt-3]
                    end
                end 
                #means and std
                mean_R[iter] = mean(filter(x->x!=-1.0, pre_def_stats_R[iter,1:40]))
                std_R[iter] =  std(filter(x->x!=-1.0, pre_def_stats_R[iter,1:40]))
                mean_R_IMF[iter] = mean(filter(x->x!=-1.0, pre_def_stats_R_d[iter,1:40]))
                std_R_IMF[iter] = std(filter(x->x!=-1.0, pre_def_stats_R_d[iter,1:40]))
                std_C[iter] = std(log.(pre_def_stats_C[iter,1:40]))
                std_Y_t[iter] =  std(log.(pre_def_stats_Y_t[iter,1:40]))
                mean_B[iter] = mean(pre_def_stats_B[iter,1:40]./ pre_def_stats_Y_t[iter,1:40])
                mean_d[iter] = mean(pre_def_stats_d[iter,1:40]./ pre_def_stats_Y_t[iter,1:40])
                cor_D_change_Y_t[iter] = cor(pre_def_stats_d_change[iter,1:40], log.(pre_def_stats_Y_t[iter,1:40]))[1]
                d_non_zero[iter] = sum(pre_def_stats_d[iter, 1:40].>1e-6)/length(pre_def_stats_d[iter, 1:40])
                std_C_yearly[iter] = std(log.(pre_def_stats_C_t_yearly[iter,:])) 
                std_change_d_t_yearly[iter] = std(pre_def_stats_d_t_change_yearly[iter,:]) 
                std_Y_t_yearly[iter] = std(log.(pre_def_stats_Y_t_yearly[iter,:]))
                cor_D_change_Y_t_yearly[iter] =  cor(pre_def_stats_d_t_change_yearly[iter,:]./log.(pre_def_stats_Y_t_yearly[iter,:]), log.(pre_def_stats_Y_t_yearly[iter,:]))[1]
            end 
            if(iter == n_chosen_def)
                
                break
            end
       
            
    end
    

   #calibration targets 
    #debt_tradeables = mean(pre_def_stats_B[32:74]./pre_def_stats_Y_t[32:74])
    
    stats[1,:] = median(pre_def_stats_Y_t, dims =  1)
    stats[2,:] = median( pre_def_stats_R, dims = 1)
    stats[3,:] = median(pre_def_stats_B, dims = 1)
    stats[4,:] = median(pre_def_stats_C_t, dims = 1)
    stats[5,:] = median(ϵ_stats, dims = 1)
    stats[6,:] = median(pre_def_stats_C_n, dims = 1)
    
    stats[7,:] = median(pre_def_stats_P, dims = 1)
    stats[8,:] = median(pre_def_stats_D, dims = 1)
    stats[9,:] = median(pre_def_stats_h, dims = 1)
    stats[10,:] = median(pre_def_stats_d, dims = 1)
    stats[11,:] = median( pre_def_stats_R_d,dims = 1 )
    
    
    output_loss = -1
    #d_positive = sum(pre_def_stats_d[:, 32:75].>0)/(n_chosen_def*(75.0-32.0))
    println("calibration results: ", " default probability: ", Def_prob)    
    println("calibration results: ", " consumption std: ", NaNMath.median(std_C_yearly./std_Y_t_yearly))  
    println("debt to tradeable: " , NaNMath.median(mean_B))      
    println("mean spread: " , NaNMath.median(mean_R))      
    println("std spread: ", NaNMath.median(std_R))
    println("mean spread: IMF " ,  NaNMath.median(mean_R_IMF))  
    println("std spread: IMF " ,  NaNMath.median(std_R_IMF) )
    println("std change d: " ,  NaNMath.median(std_change_d_t_yearly) )
    println("cor d change,y  " ,  NaNMath.median(cor_D_change_Y_t_yearly)) 
    println("d higher than 0: ", NaNMath.median(d_non_zero) )
    println("mean d : ", NaNMath.median(mean_d))
    println("ratio d tp B ", NaNMath.median(mean_d)/NaNMath.median(mean_B.+mean_d))

    moments = [Def_prob, NaNMath.median(mean_B), NaNMath.median(mean_R), NaNMath.median(std_R), NaNMath.median(mean_R_IMF), NaNMath.median(std_R_IMF), NaNMath.median(std_C_yearly./std_Y_t_yearly), NaNMath.median(d_non_zero), NaNMath.median(mean_d)]
    
    if ERR == "float"
        writedlm( "output/results_pre_def_2D_float.csv",  stats, ',')  
        writedlm( "output/B_debt_float.csv", B_hist_ab , ',')
        writedlm( "output/d_debt_float.csv", d_hist_ab , ',') 
    else
        writedlm( "output/results_pre_def_2D_peg.csv",  stats, ',')  
        writedlm( "output/B_debt_peg.csv", B_hist_ab , ',') 
        writedlm( "output/d_debt_peg.csv", d_hist_ab , ',') 
    end

    return (Def_prob, NaNMath.median(mean_B), output_loss, stats, moments)
end


function run_2D_calib_whole_exp(non_calib::TD_assumed,calib::TD_calib, grid_param::TD_gird;ERR ="peg")


    unpack_params(non_calib, calib,grid_param )
    println("checking: ", "d1 = ",δ_1, "d2 = ", δ_2, "pen = ", pen, "k = ", k )

        global ERR_glob = "peg"
        Model_sol = Solve_eq_peg2D(250)
        save("output/peg_val.jld","Model_sol", Model_sol)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol, ERR="peg")
        pd, dt, loss, stats, moments =  simulate_eq_2D_default(Model_sol, ERR="peg")
        global ERR_glob = "float"
        Model_sol_float = Solve_eq_peg2D(250)
        save("output/float_val.jld","Model_sol", Model_sol_float)
        println("Siumulations starts for float")
        pd, dt, loss, stats, moments = simulate_eq_2D(Model_sol_float, ERR="float")
        pd, dt, loss, stats, moments =  simulate_eq_2D_default(Model_sol_float, ERR="float")

        global ERR_glob = "peg"
        #starts experiment
        global pen = 0.015
        Model_sol_new_k = Solve_eq_peg2D(250)
        save("output/peg_val_new_k.jld","Model_sol", Model_sol_new_k)
        println("Siumulations starts")
        pd, dt, loss, stats, moments = simulate_IRF(Model_sol,Model_sol_new_k, ERR="peg" )

        global ERR_glob = "float"
        Model_sol_float_new_k = Solve_eq_peg2D(250)
        save("output/float_val_new_k.jld","Model_sol", Model_sol_float_new_k)
        println("Siumulations starts for float")
        pd, dt, loss, stats, moments = simulate_IRF(Model_sol_float,Model_sol_float_new_k, ERR="float" )
end


end