module TwoDimSD2
using Base: Float64
using LinearAlgebra, Interpolations, Random, QuantEcon, Statistics, DataFrames, Distributions, CSV, Plots, Optim, Roots, SchumakerSpline,NLopt
using DelimitedFiles
using Parameters: @with_kw, @unpack
export Solve_eq_float2D,simulate_eq_2D


global β = 0.9
global σ = 1.0
global α = 0.75
global h_bar = 1.0
global a = 0.26
global ζ = 0.5
global y_t = 1.0
global θ = 0.0385
global δ_1 = 0.32 #-0.35
global δ_2 = 2.42 # 0.46
global ρ = 0.93
global σ_μ = 0.037
global r =0.01
global n_y = 20
global n_B = 30
global n_d = 15
global B_min = -0.1
global B_max = 1.5
global d_min = 0.0
global d_max = 0.5
global grid_B = B_min:(B_max-B_min)/(n_B-1):B_max
global grid_d = d_min:(d_max-d_min)/(n_d-1):d_max
global w_bar = 0.95* α * (1.0-a)/a
global c_t_min = 1.0
global itp, disc
global k = 0.175
global c_t_min = (copy(w_bar)/(copy(α)*(1-copy(a))/copy(a)))^(ζ)
global P_stat, y_t_stat

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


function q_d(d)
    return 1/(1+r+k*d)
end


#####################################################
#functions for global optimization
#####################################################

function interp_alg_unconst(splines_v, spline_q, state_y,state_b,state_d, x) 
    
    """
    interpolate the value of choosing debt b, given:
    value functions (splines_v),
    price function (spline_q),
    endowment: state_y
    current debt: state_b    
    """
    b= x[1]
    d = x[2]
    c_n = 1.0 #this is version for float so far
    q_b = min(max(spline_q(b,d), 0.0),1.0/(1.0+r)) #compute debt price
    if b<=0
        q_b =1.0/(1.0+r)
    end

    c_t = -state_b - state_d+ y_t_stat[state_y] +q_b*b+q_d(d)*d #compute tradeable consumption (check if it's >=0)
    if(c_t<=0)
        c_t = 1e-6
    end

    #compute the  consumption with b
    c = final_good(c_t, c_n)
    val =0.0
    val = copy(utilityCRRA(c))

    #compute the next period's values
    for j in 1:n_y
        v_b = β*P_stat[state_y, j]*splines_v[j](b,d)
        val = copy(val) + v_b
    end
    return -1.0*val #minimizing function, so need a negative value 
end



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
    opt.xtol_rel = 1e-7
    opt.min_objective = f
    opt.maxeval = 2000
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end

function solve_const_prob_1d(f, x_0, maxqb)
    opt = Opt(:LN_BOBYQA , 1) #LN_BOBYQA , LN_NEWUOA, LN_COBYLA
    opt.lower_bounds = [grid_B[1]]
    opt.upper_bounds = [maxqb]
    opt.xtol_rel = 1e-6
    opt.min_objective = f
    opt.maxeval = 2000
    #inequality_constraint!(opt, (x,g)->c(x,g), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x_0)
    return (minf,minx,ret)
end

function global_max(f, q_splines,x_guess; loc_flag =1.0)
    x_0_1s = [0.0, 0.01]
    if(loc_flag==1.0)
        x_guess = [0.2,0.15]
        g = zeros(2)
        max_glob = f(x_0_1s,g )
        x_guess[2] = min(x_guess[2], grid_d[n_d])
        #first stage
        val_proposals = 1000*ones(4)
        maximizer_proposals = zeros(2,4)
        #look for the possible max on the whole grid
        grid_id = grid_d[1]:(grid_d[n_d]-grid_d[1])/30: grid_d[n_d]
        grid_i = grid_B[1]:(grid_B[n_B]-grid_B[1])/70: grid_B[n_B]
        #println(grid_i)
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
        sol_2s  = Optim.optimize(ff, grid_B[1], max_qB, GoldenSection())#solve_const_prob_1d(ff, [min(x_0_1s[1], max_qB)], max_qB)
        x_0_2s = Optim.minimizer(sol_2s)[1] 
        # #third stage
        (val_proposals[1],maximizer_proposals[:,1],ret) = solve_const_prob(f,[x_0_1s[1],x_0_1s[2]] )
        #(val_proposals[2],maximizer_proposals[:,2],ret) = solve_const_prob(f, [x_0_1s[1],x_0_1s[2]] )
        (val_proposals[3],maximizer_proposals[:,3],ret2) = solve_const_prob(f, [x_0_2s,x_0_1s[2]] )

        #sol_max_2 = Optim.optimize(x->f(x, [0.1,0.1]),[x_0_2s[1],x_0_1s[2]], BFGS())
        #val_proposals[4] =  Optim.minimum(sol_max_2)
        #maximizer_proposals[:,4] = Optim.minimizer(sol_max_2)
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
        (minf,minx,ret) = solve_const_prob(f,x_guess )
        return (minf,minx,ret)
    end
end
function global_max_2(f, x0)
    res = bboptimize(f; SearchRange = [(grid_B[1], grid_B[n_B]), (grid_d[1], grid_d[n_d])], NumDimensions = 2)
    println(best_fitness(res))
    return (best_fitness(res),best_candidate(res), 1000)
end

function Solve_eq_float2D(T)
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
    policy_B = zeros(T, n_y, n_B, n_d)
    default_border = zeros(T, n_y, n_d)
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

    #start the loop for the last period
    for j in 1:n_y
        for i in 1:n_B
            for id in 1:n_d
                if utilityCRRA(max(y_t_stat[j] -grid_d[id] - grid_B[i],1e-6))>utilityCRRA(max(y_t_stat[j]  -grid_d[id],1e-6))-L(y_t_stat[j])
                    c_t[T,j,i,id] = max(y_t_stat[j]-grid_d[id] - grid_B[i],1e-6)
                    c[T,j,i,id] = final_good(c_t[T,j,i,id], c_n[T,j,i,id])
                    v_f[T,j,i,id] = utilityCRRA(c[T,j,i,id])
                    v_c[T,j,i,id] = utilityCRRA(c[T,j,i,id])
                else
                    c_t[T,j,i,id] = max(y_t_stat[j] -grid_d[id],1e-6)
                    c[T,j,i,id] = final_good(c_t[T,j,i,id], c_n[T,j,i,id])
                    d[T,j,i,id]  = 1.0
                    v_f[T,j,i,id] = utilityCRRA(c[T,j,i,id]) -L(y_t_stat[j])
                end
                c_t_c = max(y_t_stat[j]-grid_d[id] - grid_B[i],1e-6)
                c_c = final_good(c_t_c, c_n[T,j,i,id])
                v_c[T,j,i,id] = utilityCRRA(c_c)
                v_d[T,j,id] = utilityCRRA(final_good(max(y_t_stat[j] -grid_d[id],1e-6), c_n[T,j,i,id]))-L(y_t_stat[j])
            end
        end
    end
    
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
            #V_f_spline[t+1,j ] = Interpolations.scale(interpolate(v_f[t+1,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
            #CubicSplineInterpolation((grid_B, grid_d), v_f[t+1,j,:,:],extrapolation_bc = Line())
            q_spline[t, j] =  Interpolations.scale(interpolate(q[t,j,:,:], BSpline(Interpolations.Linear())), grid_B, grid_d) 
            #LinearInterpolation((grid_B, grid_d), q[t,j,:,:],extrapolation_bc = Periodic()) #spline works very bad for the price function (as they are non-monotonic)
        end
        Threads.@threads for j in 1:n_y #@Threads.@threads
            
            for id in 1:n_d
                val_d = 0.0
                for jj in n_y
                    val_d = copy(val_d)+ P_stat[j,jj]*β*(θ*V_f_spline[t+1,jj ](0.0,0.0)+ (1-θ)*v_d[t+1, jj,1]) #TODO         
                end
                v_d[t, j,id] = utilityCRRA(final_good(max(y_t_stat[j] -grid_d[id],1e-6), 1.0))  - L(y_t_stat[j])+ val_d
                for i in 1:n_B
                    if(t==T-1 )
                        x_0[:,j] = [0.0,0.001]
                    else
                        x_0[:,j] = [max(min(policy_B[t+1,j,i,id], grid_B[n_B]),grid_B[1]), max(min(policy_d[t+1,j,i,id], grid_d[n_d]), grid_d[1])]
                    end
                    
                    func(x::Vector, grad::Vector) = deepcopy(interp_alg_unconst(copy(V_f_spline[t+1,:]), q_spline[t,j], j ,grid_B[i],grid_d[id],x))
                    #func(x::Vector) = deepcopy(interp_alg_unconst(copy(V_f_spline[t+1,:]), q_spline[t,j], j ,grid_B[i],grid_d[id],x))
                    max_func[j] = func
                    #constraint(x::Vector, grad::Vector) = y_t_stat[j]-grid_B[i]-grid_d[id]+ q_spline[j](x[1],x[2])*x[1] +q_d(x[2])*x[2]- c_t_min 
                    #constraint(x::Vector, grad::Vector) = x[1] - grid_B[n_B]
                    (minf[j],minx[:,j],ret) = global_max(max_func[j], q_spline[t,j],x_0[:,j], loc_flag =flag)  #solve_const_prob(interp_alg_unconst_1, constraint, x_0)
                    #(minf[j],minx[:,j],ret) = global_max_2(max_func[j],x_0[:,j])
                    v_f[t,j,i,id] = -minf[j]
                    policy_d[t,j,i,id] = minx[2,j]
                    policy_B[t,j,i,id] = minx[1,j]
                    #println(maxim_sol)
                    
                    
                    if v_f[t,j,i,id] <= v_d[t, j,id]
                        v_f[t,j,i,id] = v_d[t, j,id]
                        d[t,j,i,id]  = 1.0
                    end
                end
            end
            
        end
        println("Iteration: ", T-t)
        println(maximum(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:])))
        println(mean(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:])))
        println(maximum(abs.(q[t,:,:,:].-q[t+1,:,:,:])))
        println(sum(d[t,:,:,:]))
        if(maximum(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:]))<=1e-2)
            flag =0.0
        end
        if(maximum(abs.(v_f[t,:,:,:].-v_f[t+1,:,:,:]))<=1e-5)
            v_f[1,:,:,:] = v_f[t,:,:,:]
            q[1,:,:,:] = q[t,:,:,:]
            policy_d[1,j,:,:] =policy_d[t,j,:,:]
            policy_B[1,j,:,:] = policy_B[t,j,:,:]
            d[1,j,:,:] = d[t,j,:,:]
            for j in 1:n_y
                V_f_spline[1,j ] = Interpolations.scale(interpolate(v_f[t,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
                #CubicSplineInterpolation((grid_B, grid_d), v_f[t+1,j,:,:],extrapolation_bc = Line())
                q_spline[1, j] =  Interpolations.scale(interpolate(q[t,j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
                #LinearInterpolation((grid_B, grid_d), q[t,j,:,:],extrapolation_bc = Periodic()) #spline works very bad for the price function (as they are non-monotonic)
            end
            break
        end
    end
    #define the exact B,d for which the economy defults 
    
    
    for j in 1:n_y
        contour(grid_d, grid_B, v_f[1,j,:,:],fill=true)
        png("output/val_function_2d_$j")

        contour(grid_d, grid_B, policy_d[1,j,:,:].*(1.0.-d[1,j,:,:]),fill=true)
        png("output/policy_function_2d_d_$j")

        contour(grid_d, grid_B, policy_B[1,j,:,:].*(1.0.-d[1,j,:,:]),fill=true)
        png("output/policy_function_2d_B_$j")

        contour(grid_d, grid_B, q[1,j,:,:],fill=true)
        png("output/q_function_2d_$j")
    end


    

    
    return (q[1,:,:,:], d[1,:,:,:], policy_B[1,:,:,:], policy_d[1,:,:,:], v_f,default_border )
end

function simulate_eq_2D( Model_solution, w_bar; burnout = 10000, t_sim =1010000, n_sim = 10, ERR="float", method = "spline_sch")
    #unpack policy functions and paramters
    n_B_long =10000
    q = Model_solution[1]
    #Def_mat = Model_solution[2]
    Policy_B = Model_solution[3]
    Policy_d  = Model_solution[4]
    Def_bord = Model_solution[6]
    grid_B_long = gird_B[1]:(gird_B[n_B] - gird_B[1])/(n_B_long-1):grid[n_B]
    Def_matrix = zeros(n_y, 1000, n_d)
    for j in 1:n_y
        for id in 1:n_d
            for i in 1:n_B_long
                if(Def_bord[j,id]>= grid_B_long[])
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
    
    #exchange rate and minimal wage
   
    ϵ = ones(n_sim, t_sim)
    const_w = (α*(1.0-a)/a/w_bar)^(1.0/(1.0/ζ - (1.0-α)/α))
    power = copy(1.0/ζ/(1.0/ζ - (1.0-α)/α))
    
    policy_B_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    policy_d_spline = Array{Interpolations.ScaledInterpolation}(undef,n_y)
    q_spline = Array{Interpolations.ScaledInterpolation}(undef, n_y)
    Def_bord_itp =  Array{Interpolations.Extrapolation}(undef,n_y)
    #interpolate policy functions
    for j in 1:n_y
        q_spline[j] =  Interpolations.scale(interpolate(q[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_B_spline[j] =  Interpolations.scale(interpolate(Policy_B[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        policy_d_spline[j] = Interpolations.scale(interpolate(Policy_B[j,:,:], BSpline(Cubic(Line(OnGrid())))), grid_B, grid_d) 
        Def_bord_itp[j] = LinearInterpolation((grid_B_long, grid_d), Def_matrix[j,:,:]) #def bord is given for a id points co that is why I change the 
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
                    B_hist[i,t] = 0.0
                    d_hist[i,t] = 0.0
                    R[i,t] = 0.0 #nop interest rate
                    D[i,t] = 0.0
                    D_state[i,t] = 1.0
                else
                    
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = policy_B_spline[SIM[i,t]](0.0, 0.0)
                    d_hist[i,t] = policy_d_spline[SIM[i,t]](0.0,0.0)
                    C_t[i,t] = Y_t[i,t] + q_func[SIM[i,t]](0.0, 0.0)*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t]
                    if C_t[i,t] < c_t_min && ERR =="peg"
                        C_n[i,t] =  min(copy(C_t[i,t])^(power)*const_w,1.0)
                        h_t[i,t] = C_n[i,t]^(1.0/α)
                    else
                        C_n[i,t] = 1.0
                    end
                    C[i,t] = final_good(C_t[i,t], C_n[i,t]) 
                    R[i,t] = 1.0/q_func[SIM[i,t]](0.0,0.0) - 1.0 - r
                    D[i,t] = 0.0
                    D_state[i,t] = 0.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                    Default_flag = 0.0
                    
                end
            else
                if Def_bord_itp[SIM[i,t]](B_hist[i,t-1],d_hist[i,t-1] ) .<1.0  #case of default
                    Y_t[i,t] = y_t_stat[SIM[i,t]]- L(y_t_stat[SIM[i,t]])
                    C_t[i,t] = Y_t[i,t]  - d_hist[i,t-1]
                    C[i,t] = final_good(C_t[i,t],1.0) 
                    if C_t[i,t] < c_t_min && ERR =="peg"
                        C_n[i,t] =  min(copy(C_t[i,t])^(power)*const_w,1.0)
                        h_t[i,t] = C_n[i,t]^(1.0/α)
                    else
                        C_n[i,t] = 1.0
                    end
                    C[i,t] = final_good(C_t[i,t],C_n[i,t]) 
                    B_hist[i,t] = 0.0
                    d_hist[i,t] = 0.0
                    R[i,t] = 0 #1.0/q[SIM[i,t],A_int[i,t]] - 1.0
                    D[i,t] = 1.0
                    D_state[i,t] = 1.0
                    Default_flag = 1.0
                    Trade_B[i,t] = (Y_t[i,t] - C_t[i,t])/Y_t[i,t]
                else
                    Y_t[i,t] = y_t_stat[SIM[i,t]]
                    B_hist[i,t] = policy_B_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1])
                    d_hist[i,t] = policy_d_spline[SIM[i,t]](B_hist[i,t-1], d_hist[i,t-1])
                    C_t[i,t] =Y_t[i,t]- B_hist[i,t-1] - d_hist[i,t-1]+ q_func[SIM[i,t]](0.0, 0.0)*B_hist[i,t] + q_d(d_hist[i,t])*d_hist[i,t]
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
    B_hist_ab = B_hist[:,burnout:t_sim]
    d_hist_ab = d_hist[:,burnout:t_sim]
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
    pre_def_stats_d = zeros(n_chosen_def , 90) 
    pre_def_stats_NB = zeros(n_chosen_def , 90)  
    pre_def_stats_P = zeros(n_chosen_def , 90)  
    pre_def_stats_D = zeros(n_chosen_def , 90) 
    pre_def_stats_h = zeros(n_chosen_def , 90) 
    stats = zeros(10, 90)
    ϵ_stats = zeros(n_chosen_def , 90)  
    
    
    
    
    #compute statistics for 74 periods before the default, for n_def defaults (without any defults in the 74 periods before)
    iter = 0
    for i in 1:n_sim, t in 76:t_sim-burnout
            if(D_ab[i,t]==1 && sum(D_state_ab[i,t-25:t-1 ])== 0.0 && t<=t_sim-burnout-100 ) #if default happen
                 iter =iter+1   
                 pre_def_stats_Y_t[iter,:] = Y_ab[i, t-74:t+15]
                 pre_def_stats_R[iter,:] = (1.0.+R_ab[i, t-74:t+15]).^4.0.- (1.0+r).^4.0
                 pre_def_stats_B[iter,:] = B_hist_ab[i, t-75:t+14]
                 pre_def_stats_d[iter,:] = d_hist_ab[i, t-75:t+14]
                 pre_def_stats_NB[iter,:] = B_hist_ab[i, t-74:t+15] #new debt
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
    stats[10,:] = median(pre_def_stats_d, dims = 1)
    output_loss = mean((pre_def_stats_Y_t[:,75].- pre_def_stats_Y_t[:,74])./(pre_def_stats_Y_t[:,74]))
    println("calibration results: ", " default probability: ", Def_prob)       
    println("prod_loss: " , output_loss)      
    println("debt to tradeable: " , debt_tradeables)      
    println("mean spread: " , mean(stats[2,1:74]))      
    println("std spread: ", mean(std(pre_def_stats_R[1:74], dims=1))  )     
    return (Def_prob, debt_tradeables, output_loss, stats)
end

end