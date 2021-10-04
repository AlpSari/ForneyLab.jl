export ruleSPCVIOutVD, ruleSPCVIIn1MV, ruleSPCVIOutVDX, ruleSPCVIInX
function ruleSPCVIIn1MV(node_id::Symbol,
                        msg_out::Message{<:FactorFunction, <:VariateType},
                        msg_in::Message{<:FactorNode, <:VariateType})

    thenode = currentGraph().nodes[node_id]

    η = deepcopy(naturalParams(msg_in.dist))
    if thenode.online_inference == false
        λ_init = deepcopy(η)
    else
        λ_init = deepcopy(naturalParams(thenode.q[1]))
    end

    logp_nc(z) = (thenode.dataset_size/thenode.batch_size)*logPdf(msg_out.dist, thenode.g(z))
    λ = renderCVI(logp_nc,thenode.num_iterations,thenode.opt,λ_init,msg_in)

    λ_message = λ.-η
    # Implement proper message check for all the distributions later on.
    thenode.q = [standardDist(msg_in.dist,λ)]
    if thenode.online_inference == false thenode.q_memory = deepcopy(thenode.q) end
    return standardMessage(msg_in.dist,λ_message)
end

function ruleSPCVIIn1MV(node_id::Symbol,
                        msg_out::Message{<:FactorFunction, <:VariateType},
                        msg_in::Message{<:Gaussian, Univariate})

    thenode = currentGraph().nodes[node_id]

    η = deepcopy(naturalParams(msg_in.dist))
    if thenode.online_inference == false
        λ_init = deepcopy(η)
    else
        λ_init = deepcopy(naturalParams(thenode.q[1]))
    end

    logp_nc(z) = (thenode.dataset_size/thenode.batch_size)*logPdf(msg_out.dist, thenode.g(z))
    λ = renderCVI(logp_nc,thenode.num_iterations,thenode.opt,λ_init,msg_in)

    λ_message = λ.-η

    if thenode.proper_message λ_message = convertToProperMessage(msg_in, λ_message) end
    thenode.q = [standardDist(msg_in.dist,λ)]
    if thenode.online_inference == false thenode.q_memory = deepcopy(thenode.q) end
    return standardMessage(msg_in.dist,λ_message)
end

function ruleSPCVIIn1MV(node_id::Symbol,
                        msg_out::Message{<:FactorFunction, <:VariateType},
                        msg_in::Message{<:Gaussian, Multivariate})

    thenode = currentGraph().nodes[node_id]

    η = deepcopy(naturalParams(msg_in.dist))
    if thenode.online_inference == false
        λ_init = deepcopy(η)
    else
        λ_init = deepcopy(naturalParams(thenode.q[1]))
    end

    logp_nc(z) = (thenode.dataset_size/thenode.batch_size)*logPdf(msg_out.dist, thenode.g(z))
    λ = renderCVI(logp_nc,thenode.num_iterations,thenode.opt,λ_init,msg_in)

    λ_message = λ.-η

    if thenode.proper_message λ_message = convertToProperMessage(msg_in, λ_message) end
    thenode.q = [standardDist(msg_in.dist,λ)]
    if thenode.online_inference == false thenode.q_memory = deepcopy(thenode.q) end
    return standardMessage(msg_in.dist,λ_message)
end

function ruleSPCVIOutVD(node_id::Symbol,
                        msg_out::Nothing,
                        msg_in::Message)

    thenode = currentGraph().nodes[node_id]

    sampl = thenode.g(sample(msg_in.dist))
    if length(sampl) == 1
        variate = Univariate
    else
        variate = Multivariate
    end
    return Message(variate,SetSampleList,node_id=node_id)

end

function ruleSPCVIInX(node_id::Symbol,
                      inx::Int64,
                      msg_out::Message{<:FactorFunction, <:VariateType},
                      msgs_in::Vararg{Message})

    thenode = currentGraph().nodes[node_id]

    local_messages, global_messages = [], []
    local_inx, global_inx = [], []
    for i=1:length(thenode.online_inference)
        if thenode.online_inference[i]
            push!(global_messages, msgs_in[i])
            push!(global_inx, i)
        else
            push!(local_messages, msgs_in[i])
            push!(local_inx, i)
        end
    end
    msgs_list = [local_messages;global_messages]
    inx_list = [local_inx;global_inx]

    arg_sample = (z,j) -> begin
        samples_in = []
        for k=1:length(msgs_in)
            if k==j
                push!(samples_in,collect(Iterators.repeat([ z ], thenode.num_samples)))
            else
                push!(samples_in,sample(thenode.q_memory[k], thenode.num_samples))
            end
        end

        return samples_in
    end

    λ_list = []

    if thenode.infer_memory == 0
        for j=1:length(msgs_list)
            msg_in = msgs_list[j]
            logp_nc(z) = sum(logPdf.([msg_out.dist],thenode.g.(arg_sample(z,inx_list[j])...)))/thenode.num_samples
            if thenode.online_inference[inx_list[j]]
                λ_init = deepcopy(naturalParams(thenode.q[inx_list[j]]))
                logp_nc(z) = (thenode.dataset_size/thenode.batch_size)*sum(logPdf.([msg_out.dist],thenode.g.(arg_sample(z,inx_list[j])...)))/thenode.num_samples
            else
                λ_init = deepcopy(naturalParams(msg_in.dist))
            end
            λ = renderCVI(logp_nc,thenode.num_iterations[inx_list[j]],thenode.opt[inx_list[j]],λ_init,msg_in)
            thenode.q[inx_list[j]] = standardDist(msg_in.dist,λ)
            if thenode.online_inference[inx_list[j]] == false thenode.q_memory[inx_list[j]] = deepcopy(thenode.q[inx_list[j]]) end
            push!(λ_list, λ)
        end
        thenode.infer_memory = length(msgs_in) - 1
    else
        thenode.infer_memory -= 1
    end

    # Send the message
    λ = naturalParams(thenode.q[inx])
    η = naturalParams(msgs_in[inx].dist)
    λ_message = λ.-η

    if isUnivariateGaussian(thenode.q[inx]) || isMultivariateGaussian(thenode.q[inx])
        if thenode.proper_message λ_message = convertToProperMessage(msg_in, λ_message) end
    end
    return standardMessage(msgs_in[inx].dist,λ_message)

end

function ruleSPCVIOutVDX(node_id::Symbol,
                         msg_out::Nothing,
                         msgs_in::Vararg{Message})

    thenode = currentGraph().nodes[node_id]

    sampl_in = [sample(msg_in.dist) for msg_in in msgs_in]
    sampl = thenode.g(sampl_in...)
    if length(sampl) == 1
        variate = Univariate
    else
        variate = Multivariate
    end
    return Message(variate,SetSampleList,node_id=node_id)

end

#---------------------------
# CVI implementations
#---------------------------
import StatsBase:percentile,autocor
import Flux.Optimise:update!
import Statistics:median,var
# Additional definitions for iBLR implementation of CVI
mutable struct iBLR
    eta::Float64
    state::Int64
    stable_params::Any
end
mutable struct iBLR_AdaGrad
    field::Any
end
iBLR(eta)=iBLR(eta,1,nothing)
#--- Struct definitions
abstract type ConvergenceOptimizer end
Base.@kwdef mutable struct DefaultOptim <: ConvergenceOptimizer
    max_iterations ::Int64
end
Base.@kwdef mutable struct ConvergenceStatsFE
    F_prev::Float64 # FE from previous iterate
    F_best::Float64 # Minimum attained FE
    F_best_idx::Int64 # Index of Minimum attained FE
    F_converge_idx::Int64 # Index of Minimum attained FE
    ΔF_rel::Float64 # Relative difference between F_now and F_prev
    ΔF_vect::Vector{Float64}
    FE_vect::Vector{Float64}#TODO: delete this field later, this is for debugging
end # struct
Base.@kwdef mutable struct ConvergenceParamsFE <: ConvergenceOptimizer
    max_iterations::Int64
    eval_FE_window::Float64
    burn_in_min ::Float64
    burn_in_max ::Float64
    tolerance_mean ::Float64
    tolerance_median ::Float64
    stats::ConvergenceStatsFE #NEW
end # struct
Base.@kwdef mutable struct ConvergenceParamsMC <: ConvergenceOptimizer
    max_iterations::Int64
    pareto_k_thr ::Float64
    pareto_num_samples ::Float64
    mcmc_num_chains ::Int64
    mcmc_window_len ::Float64
    rhat_cutoff ::Float64
    mcse_cutoff ::Float64
    ess_threshold ::Float64
end # struct
Base.@kwdef mutable struct StepSizeParams
    init_stepsize::Float64
    current_stepsize::Float64
    stepsize_update_func::Function
end # struct
Base.@kwdef mutable struct ParamStr2
    is_initialized::Bool
    eta::Float64
    state::Int64
    stable_params::Union{Vector,Nothing}
    convergence_algo::String # Determines the convergence algorithm
    auto_init_stepsize::Bool # Determines the initial stepsize
    convergence_optimizer::ConvergenceOptimizer
    stepsize_optimizer::StepSizeParams
end

#--- Helper functions for struct initalizations
function check_all_keys(x::Dict,dtype::DataType)
    list = [key in fieldnames(dtype) for key in keys(x)]
    length(fieldnames(dtype))==count(x->x==true,list) ? true : false
end
#--- Constructors for structs
function DefaultOptim(x::Dict)
    if length(keys(x)) == 0
        return DefaultOptim()
    end
    have_all_keys = check_all_keys(x,DefaultOptim)
    if have_all_keys
        # This code will run eventually from other outer constructors which provide x::Dict with all fields
        return DefaultOptim(x[:max_iterations])
    else
        return DefaultOptim(;x...) # refer to different outer constructor
    end
end
function DefaultOptim(;kwargs...)
    defaults=(max_iterations =Int64(1e6),)
    ntuple= merge(defaults,kwargs)
    settings = Dict(pairs(ntuple))
    return DefaultOptim(settings)
end
#
function ConvergenceStatsFE(x::Dict)
    if length(keys(x)) == 0
        return ConvergenceStatsFE()
    end
    have_all_keys = check_all_keys(x,ConvergenceStatsFE)
    if have_all_keys
        # This code will run eventually from other outer constructors which provide x::Dict with all fields
        return ConvergenceStatsFE(x[:F_prev],x[:F_best],x[:F_best_idx],x[:F_converge_idx],x[:ΔF_rel],x[:ΔF_vect],x[:FE_vect])
    else
        return ConvergenceStatsFE(;x...) # refer to different outer constructor
    end
end
function ConvergenceStatsFE(;kwargs...)
    defaults= (;F_prev = Inf,
    F_best = Inf,
    F_best_idx = -1,
    F_converge_idx = -1,
    ΔF_rel  = Inf,
    ΔF_vect = Vector{Float64}(),
    FE_vect = Vector{Float64}()) # namedtuple with default params
    ntuple= merge(defaults,kwargs)
    settings = Dict(pairs(ntuple))
    return ConvergenceStatsFE(settings)
end
#
function ConvergenceParamsFE(;kwargs...)
    defaults =(max_iterations= Int64(1e6),
            eval_FE_window=0.002,
            burn_in_min = 0.05,
            burn_in_max = 0.5,
            tolerance_mean = 0.1,
            tolerance_median = 0.1,
            stats=ConvergenceStatsFE())
    ntuple= merge(defaults,kwargs)
    settings = Dict(pairs(ntuple))
    stats_struct =ConvergenceStatsFE(;settings...)
    settings[:stats] =stats_struct
    return ConvergenceParamsFE(settings)
end
function ConvergenceParamsFE(x::Dict)
    if length(keys(x)) == 0
        return ConvergenceParamsFE()
    end
    have_all_keys=check_all_keys(x,ConvergenceParamsFE)
    if have_all_keys
        # This code will run eventually from other outer constructors which provide x::Dict with all fields
        return ConvergenceParamsFE(x[:max_iterations],x[:eval_FE_window],x[:burn_in_min],x[:burn_in_max],x[:tolerance_mean],x[:tolerance_median],x[:stats])
    else
        return ConvergenceParamsFE(;x...) # refer to different outer constructor
    end
end
#
function ConvergenceParamsMC(;kwargs...)
    defaults= (max_iterations= Int64(1e6),pareto_k_thr= 0.7,
        pareto_num_samples = 1000.0,
        mcmc_num_chains= 10,
        mcmc_window_len= 500,
        rhat_cutoff= 1.2,
        mcse_cutoff= 1.0,
        ess_threshold= 100.0)
    ntuple= merge(defaults,kwargs)
    settings = Dict(pairs(ntuple))
    return ConvergenceParamsMC(settings)
end
function ConvergenceParamsMC(x::Dict)
    if length(keys(x)) == 0
        return ConvergenceParamsMC()
    end
    have_all_keys = check_all_keys(x,ConvergenceParamsMC)
    if have_all_keys
        # This code will run eventually from other outer constructors which provide x::Dict with all fields
        return ConvergenceParamsMC(x[:max_iterations],x[:pareto_k_thr],x[:pareto_num_samples],x[:mcmc_num_chains],x[:mcmc_window_len],x[:rhat_cutoff],
        x[:mcse_cutoff],x[:ess_threshold])
    else
        return ConvergenceParamsMC(;x...) # refer to different outer constructor
    end
end
#
function StepSizeParams(;kwargs...)
    defaults = (init_stepsize= 0.0,
            current_stepsize = 0.0,
            stepsize_update_func= x->x)
    belongsto(s) = first(s) in fieldnames(StepSizeParams)
    ntuple= merge(defaults,kwargs)
    settings = Dict(pairs(ntuple))
    return StepSizeParams(settings)
end
function StepSizeParams(x::Dict)
    if length(keys(x)) == 0
        return StepSizeParams()
    end
    have_all_keys = check_all_keys(x,StepSizeParams)
    if have_all_keys
        # This code will run eventually from other outer constructors which provide x::Dict with all fields
        return StepSizeParams(x[:init_stepsize],x[:current_stepsize],x[:stepsize_update_func])
    else
        return StepSizeParams(;x...) # refer to different outer constructor
    end
end
#
function ParamStr2(x::Dict)
    if length(keys(x)) == 0
        return ParamStr2()
    end
    have_all_keys_params = check_all_keys(x,ParamStr2)
    if have_all_keys_params
        # This code will run eventually from other outer constructors which provide x::Dict with all fields
        x[:is_initialized] =false # Always false when constructed
        return ParamStr2(x[:is_initialized],x[:eta],x[:state],x[:stable_params],x[:convergence_algo],x[:auto_init_stepsize],x[:convergence_optimizer],x[:stepsize_optimizer])
    else
        return ParamStr2(;x...) # refer to different outer constructor
    end
end

function ParamStr2(;kwargs...)
    defaults = (is_initialized=false,eta = 0.0,state=1,stable_params=nothing,convergence_algo="free_energy",auto_init_stepsize=true,
            convergence_optimizer=ConvergenceParamsFE(),
            stepsize_optimizer=StepSizeParams())
    if haskey(kwargs,:eta) # If eta is set, stepsize is manually set
        kwargs = (kwargs...,auto_init_stepsize=false)
    end
    ntuple= merge(defaults,kwargs)
    settings = Dict(pairs(ntuple))
    convergence_algo_str = settings[:convergence_algo]
    # 1) Check if initial stepsize is given, otherwise determine it by Auto (Wolfe Condition)
    if !settings[:auto_init_stepsize]
        # Assert that user did not specify an invalid stepsize
        # TODO: Maybe negative stepsize should be asserted along with 0.0 stepsize
        settings[:eta] == 0.0 ? throw(ArgumentError("Stepsize ':eta' cannot be $(settings[:eta]) when step size is manually provided.")) : nothing
    else
        nothing #Determine by Wolfe Condition
    end
    # 2) Check convergence algo string to determine convergence_optimizer type
    if convergence_algo_str == "free_energy"
        if ! (typeof(settings[:convergence_optimizer]) == ConvergenceParamsFE) #throw error since this is the default choice
            throw(ArgumentError("Invalid choice of optimizer ($convergence_optimizer) for the given convergence algorithm."))
        end
        optim_alias = ConvergenceParamsFE
    elseif convergence_algo_str == "MonteCarlo"
        optim_alias = ConvergenceParamsMC
    elseif convergence_algo_str == "none"
        optim_alias = DefaultOptim
    else
        throw(ArgumentError("Invalid string ('$convergence_algo_str') for convergence algorithm specification"))
    end
        #Init Optim struct
        optimizer = optim_alias(settings)
        #Init StepSize struct
        ss_struct = StepSizeParams(settings)
        # Make Adjustments to Settings
        settings[:convergence_optimizer] = optimizer
        settings[:stepsize_optimizer] = ss_struct
        return ParamStr2(settings)
end
#--- renderCVI function
function inexactLineSearch()
    #TODO: Implement WolfeConditons for inexactLineSearch
    return 1.0
end
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::ParamStr2,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})
    # TODO: Make this function is for any msg_in and assert msg_in for ConvergenceLoop
    # 1) Check if the optimizer is initialized
    stepsize_optimizer = opt.stepsize_optimizer
    convergence_optimizer = opt.convergence_optimizer
    if !(opt.is_initialized)
        # Check if auto stepsize detection is requested
        if opt.auto_init_stepsize
            opt.eta = inexactLineSearch()
            stepsize_optimizer.init_stepsize = deepcopy(opt.eta)
            stepsize_optimizer.current_stepsize = deepcopy(opt.eta)
        else
            # Manual setting of stepsize
            stepsize_optimizer.init_stepsize = deepcopy(opt.eta)
            stepsize_optimizer.current_stepsize = deepcopy(opt.eta)
        end
        opt.is_initialized = true
    end
    # # 2) Initialize parameter λ_q and prior λ_0
    # # 3) Set Stable params
    # 4) Run Optimization Loop depending on the type of ConvergenceOptimizer
    if typeof(convergence_optimizer) == ConvergenceParamsFE
        λ_natural_posterior = renderCVI_ΔFE(logp_nc,opt,λ_init,msg_in)
    elseif typeof(convergence_optimizer) == ConvergenceParamsMC
        λ_natural_posterior = renderCVI_MCMC(logp_nc,opt,λ_init,msg_in)
    elseif typeof(convergence_optimizer) == DefaultOptim
        #TODO : CHANGE THIS FUNCTION to renderCVI_Default
        λ_natural_posterior=renderCVI(logp_nc,convergence_optimizer.max_iterations,iBLR(opt.eta),λ_init,msg_in)
    else
        throw(ArgumentError("Invalid convergence_optimizer($convergence_optimizer)"))
    end

    return λ_natural_posterior
end

function renderCVI_ΔFE(logp_nc::Function,
                   opt::ParamStr2,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})

    convergence_optimizer = opt.convergence_optimizer
    # RESET STATS
    convergence_optimizer.stats = ConvergenceStatsFE()
    stats = convergence_optimizer.stats
    #
    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)
    # 1) Initialize parameter λ_q and prior λ_0
    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    # 2) Set Stable params
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point
    # 3) Init Free Energy Optimization Parameters
    eval_FE_window = Int64(floor(convergence_optimizer.max_iterations*convergence_optimizer.eval_FE_window))
    burn_in_min = Int64(floor(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_min))
    burn_in_max = Int64(floor(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_max))
    FE_check = false
    is_FE_converged = false
    tolerance_mean = convergence_optimizer.tolerance_mean
    tolerance_median = convergence_optimizer.tolerance_median
    #λ_iblr_best = Vector{Float64}(undef,2)

    FE_max_threshold = 0.0

    for i=1:convergence_optimizer.max_iterations
        q = bcToStandardDist(msg_in.dist,λ_iblr)
        z_s = sample(q)
        g_i = df_m(z_s)
        H_i=  df_H(z_s)
        g_μ_1 = (g_i+η[2]*(η[1]-λ_iblr[1]))/λ_iblr[2]
        g_μ_2= -H_i+η[2]-λ_iblr[2]
        g̃=[g_μ_1;g_μ_2]
        update!(opt,λ_iblr,g̃,msg_in.dist)
        if 1<= i<=burn_in_min
            FE = KL_bc(λ_iblr,η,msg_in.dist) - logp_nc(z_s)
            FE_max_threshold += FE
            if i == burn_in_min
                stats.F_best = FE_max_threshold/(burn_in_min)
                stats.F_prev = deepcopy(stats.F_best)
                stats.F_best_idx = burn_in_min
            end
        # End of burn in period
        elseif mod(i,eval_FE_window) == 0
            FE = KL_bc(λ_iblr,η,msg_in.dist) - logp_nc(z_s)
            #First condition to start tests: FE is smaller than initial FE
            if FE_check == false && FE < stats.F_best
                FE_check = true # FE dropped below initial ELBO
                println("FE ($FE) is smaller now,FE_check starts when i=$i")
            #Second condition to start tests: i exceeds burn_in_max period
            elseif FE_check == false && i > burn_in_max
                FE_check = true
                println("Burninmax reached ($burn_in_max), FE_check starts when i=$i")
            end
            # If tests start, check convergence
            if FE_check
                is_FE_converged = ΔFE_check!(stats,FE,i,tolerance_mean,tolerance_median)
                # Store λ yielding minimum Free Energy
                if stats.F_best_idx == i
                    λ_iblr_best = deepcopy(λ_iblr)
                end
            end
            # If converged, stop iterations
            if is_FE_converged
                println("Algorithm converged at iteration $i")
                break
            end
        end
    end
    if is_FE_converged && @isdefined λ_iblr_best
        # return best iterate
        λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr_best)
    else
        #return last iterate
        λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    end
    return λ_natural_posterior
end
function update!(opt::ParamStr2,params::Vector,natgrad::Vector,prior::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
    #TODO:#1) Use StepSizeParams's stepsize for update
    #TODO #2) Update currentstepsize after each iteration
    #params = [μ,S]
    if any(isnan.(natgrad)) || any(isinf.(natgrad))
        params = deepcopy(opt.stable_params)
        return # no update
    else
        opt.stable_params = deepcopy(params)
    end
    params[1] += opt.eta*natgrad[1]
    params[2] += opt.eta*natgrad[2]+0.5*(opt.eta*natgrad[2])^2/params[2]
    if isProper(bcToStandardDist(prior,params)) == false
        # not proper after update
        params = deepcopy(opt.stable_params)
    end
    return params
end
function renderCVI_MCMC(logp_nc::Function,
                   opt::ParamStr2,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})

    convergence_optimizer = opt.convergence_optimizer
    #
    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)
    # 1) Initialize parameter λ_q and prior λ_0
    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    # 2) Set Stable params
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point


    W = convergence_optimizer.mcmc_window_len
    J = convergence_optimizer.mcmc_num_chains
    max_iter = Int64(floor(convergence_optimizer.max_iterations/W))
    simulations_done = 0
    is_mcmc_converged= false
    #First one
    last_params,opt_matrix,stats_dict=oneWindowSimulation_MCMC(J,Int64(W),opt,η,λ_iblr,logp_nc,true,msg_in)
    for iter= 2:max_iter
        last_params,opt_matrix,stats_dict=oneWindowSimulation_MCMC(J,Int64(W),opt_matrix,η,last_params,logp_nc,false,msg_in)
        if all(stats_dict[:rhat] .< convergence_optimizer.rhat_cutoff)
            simulations_done = iter
            break
        end
    end
    for i=simulations_done:max_iter
        last_params,opt_matrix,stats_dict=oneWindowSimulation_MCMC(J,Int64(W),opt_matrix,η,last_params,logp_nc,false,msg_in)
        # Check MCSE and ESS
        if all(stats_dict[:mcse] .< convergence_optimizer.mcse_cutoff) && all(stats_dict[:ess] .> convergence_optimizer.ess_threshold)
            is_mcmc_converged = true
            println("Converged Parameters using Iterate Averaging = $(stats_dict[:λ_bar])")
            break
        end
    end
    if !is_mcmc_converged
        println("Warning! The algorithm might not have been converged!")
    end
    return bcToNaturalParams(msg_in.dist,stats_dict[:λ_bar])
end




#---
function ΔFE_check!(stats::ConvergenceStatsFE,F_now::Float64,idx_now::Int64,tolerance::Float64)
    ΔFE_check!(stats,F_now,idx_now,tolerance,tolerance)
end
function ΔFE_check!(stats::ConvergenceStatsFE,F_now::Float64,idx_now::Int64,tolerance_mean::Float64,tolerance_median::Float64)
    # Return true if Free Energy is converged, return false else
    push!(stats.FE_vect,F_now)
    if F_now < stats.F_best
        stats.F_best = deepcopy(F_now)
        stats.F_best_idx = deepcopy(idx_now)
    end
    stats.ΔF_rel = abs(100*(F_now-stats.F_prev)/(stats.F_prev))
    push!(stats.ΔF_vect,stats.ΔF_rel)
    # Calculate mean/median
    vect_mean = mean(stats.ΔF_vect)
    vect_median = median(stats.ΔF_vect)
    if (vect_mean < tolerance_mean && length(stats.ΔF_vect)>20) || (vect_median < tolerance_median && length(stats.ΔF_vect)>20)
        stats.F_converge_idx = deepcopy(idx_now)
        stats.F_prev = deepcopy(F_now) #Also becomes F at convergence
        return true
    else # No convergence, get ready for the next call of the function
    stats.F_prev = deepcopy(F_now)
        return false
    end
end

bcParams(dist::ProbabilityDistribution{Univariate, F}) where F<:Gaussian = [unsafeMean(dist),unsafePrecision(dist)]
bcParams(dist::ProbabilityDistribution{Multivariate, F}) where F<:Gaussian = [vec(unsafeMean(dist)); vec(unsafePrecision(dist))]
function bcToStandardDist(dist::ProbabilityDistribution{Univariate, F}, η::Vector) where F<:Gaussian
    ProbabilityDistribution(Univariate, GaussianWeightedMeanPrecision,xi=η[2]*η[1],w=η[2])
end
function bcToNaturalParams(dist::ProbabilityDistribution{Univariate, F}, η::Vector) where F<:Gaussian
    λ = [η[2]*η[1],-0.5*η[2]]
end
function bcToStandardDist(dist::ProbabilityDistribution{Multivariate, F}, η::Vector) where F<:Gaussian
    #TODO: CHECK if tiny should be added before XI definition
    d = dims(dist)
    W = reshape(η[d+1:end],d,d)
    XI = W*η[1:d]
    W = Matrix(Hermitian(W + tiny*diageye(d))) # Ensure precision is always invertible
    ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision,xi=XI,w=W)
end
# update method for iBLR, Univariate Gaussian case
function update!(opt::iBLR,params,natgrad,prior::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
    #params = [μ,S]
    if any(isnan.(natgrad)) || any(isinf.(natgrad))
        # Gradients are non-numeric
        if opt.state == 1 # Return to previous parameters which give numeric g̃
            params = deepcopy(opt.stable_params)
        end
        return # no update
    else
        # Gradients are numeric
        if opt.state ==1
            opt.stable_params = deepcopy(params)
        end
    end
    params[1] += opt.eta*natgrad[1]
    params[2] += opt.eta*natgrad[2]+0.5*(opt.eta*natgrad[2])^2/params[2]

    if isProper(bcToStandardDist(prior,params)) == false
        # not proper after update
        if opt.state == 1
            params = deepcopy(opt.stable_params)
        end
    end
    return params
end
# update method for iBLR_AdaGrad, Univariate Gaussian case
function update!(opt::iBLR_AdaGrad,params,natgrad,prior::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
    #params = [μ,S]
    if any(isnan.(natgrad)) || any(isinf.(natgrad))
        # Gradients are non-numeric
        if opt.state == 1 # Return to previous parameters which give numeric g̃
            params = deepcopy(opt.stable_params)
        end
        return # no update
    else
        # Gradients are numeric
        if opt.state ==1
            opt.stable_params = deepcopy(params)
        end
    end

    denominator = 1+(opt.grad_acc-1)^(0.5)
    #denominator = (1.01)^(opt.grad_acc)
    #opt.eta = opt.eta*0.95
    params[1] += (opt.eta/denominator)*natgrad[1]
    params[2] += (opt.eta/denominator)*natgrad[2]+0.5*((opt.eta/denominator)*natgrad[2])^2/params[2]
    #println("Current β_t = $(opt.eta/denominator)")

    if isProper(bcToStandardDist(prior,params)) == false
        # not proper after update
        if opt.state == 1
            params = deepcopy(opt.stable_params)
        end
    end
end
# update method for iBLR, Multivariate Gaussian case
function update!(opt::iBLR,params,natgrad,prior::ProbabilityDistribution{Multivariate, F},s_inv::Array{Float64,2}) where F <: Gaussian
    #params = [vec(μ),mat(S)]
    any_nan_value = any(any.((x->isnan.(x)).(natgrad)))
    any_inf_value = any(any.((x->isinf.(x)).(natgrad)))
    if any_nan_value || any_inf_value
        # Gradients are non-numeric
        if opt.state == 1 # Return to previous parameters which give numeric g̃
            params = deepcopy(opt.stable_params)
        end
        return # no update
    else
        # Gradients are numeric
        if opt.state ==1
            opt.stable_params = deepcopy(params)
        end
    end

    params[1] += opt.eta*natgrad[1]
    params[2] += opt.eta*natgrad[2]+0.5*(opt.eta)^2*natgrad[2]*s_inv*natgrad[2]

    if isProper(bcToStandardDist(prior,[params[1];vec(params[2])])) == false
        # not proper after update
        if opt.state == 1
            params = deepcopy(opt.stable_params)
        end
    end
end
# update method for iBLR, Multivariate Gaussian case
function update!(opt::iBLR,params,natgrad,prior::ProbabilityDistribution{Multivariate, F}) where F <: Gaussian
    s_inv = deepcopy(cholinv(params[2]))
    update!(opt,params,natgrad,prior,s_inv)
end

# KL Divergence between two Univariate Gaussian distributions (BCN parameterization)
function KL_bc(λ::Vector,λ_0::Vector,dist::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
    # 0.5 [log(σ^2/σ_0^2)+(σ^2+(μ-μ_0)^2)/(σ_0)^2-1]
    return 0.5 *(log(λ[2]/λ_0[2])+ (λ[2]+(λ[1]-λ_0[1])^2)/(λ_0[2])-1)
end
# CVI Original for Univariate Gaussian
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.derivative(df_m,z)
    #F(z,) = -(logp_nc(z)-KL(q||p))
    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)
    # CVI with naturalParams [μS,-0.5S]
    for i=1:num_iterations
        q = standardDist(msg_in.dist,λ)
        z_s = sample(q)
        df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
        df_μ2 = df_v(z_s)
        ∇f = [df_μ1, df_μ2]
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(opt,λ,∇)
        if isProper(standardDist(msg_in.dist,λ)) == false
            λ = λ_old
        end
    end
    return λ
end
# iBLR Original for Univariate Gaussian
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})

    """
    improved Bayesian Learning Rule implementation for CVI node
        BC Parameters are mean and precision(=[μ,S]) for Gaussian
    """

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)

    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point
    for i=1:num_iterations
        q = bcToStandardDist(msg_in.dist,λ_iblr)
        z_s = sample(q)
        g_i = df_m(z_s)
        H_i=  df_H(z_s)
        g_μ_1 = (g_i+η[2]*(η[1]-λ_iblr[1]))/λ_iblr[2]
        g_μ_2= -H_i+η[2]-λ_iblr[2]
        g̃=[g_μ_1;g_μ_2]
        update!(opt,λ_iblr,g̃,msg_in.dist)
    end
    λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    return λ_natural_posterior
end
# iBLR with ADAGrad
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR_AdaGrad,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})

    """
    improved Bayesian Learning Rule implementation for CVI node
        BC Parameters are mean and precision(=[μ,S]) for Gaussian
    """

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)

    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point
    for i=1:num_iterations
        q = bcToStandardDist(msg_in.dist,λ_iblr)
        z_s = sample(q)
        g_i = df_m(z_s)
        H_i=  df_H(z_s)
        g_μ_1 = (g_i+η[2]*(η[1]-λ_iblr[1]))/λ_iblr[2]
        g_μ_2= -H_i+η[2]-λ_iblr[2]
        g̃=[g_μ_1;g_μ_2]
        opt.grad_acc = deepcopy(i)
        update!(opt,λ_iblr,g̃,msg_in.dist)
    end
    λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    return λ_natural_posterior
end
# iBLR with Early Stopping Criteria based on Free Energy
function renderCVI_Δ_FE(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate},
                   tolerance::Float64)
   """
   improved Bayesian Learning Rule implementation for CVI node
       BC Parameters are mean and precision(=[μ,S]) for Gaussian
       with early stopping criterion based on mean/median of relative difference of Free Energy
       similar to STAN's algorithm
   """
    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)

    # iBLR Initialization
    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point

    # Δ_FE Initialization
    stats = FE_stats() #initialize ΔElbo object
    eval_elbo_window = 10
    burn_in_min = 9
    burn_in_max = floor(num_iterations/10)
    FE_check = false
    is_FE_converged = false
    λ_iblr_best = deepcopy(λ_iblr)
    F = Dict{Int64,Float64}()

    println("---Δ_FE Parameters---")
    println("Evaluate FE every $eval_elbo_window'th iteration")
    println("Tolerance (%):$tolerance")
    println("Burn_in_min:$burn_in_min ,Burn_in_max:$burn_in_max")
    println("-------")
    # ---
    for i=1:num_iterations
        q = bcToStandardDist(msg_in.dist,λ_iblr)
        z_s = sample(q)
        # ForwardDiff.hessian!(result, logp_nc, [z_s]);
        # g_i = DiffResults.gradient(result)
        # H_i = DiffResults.hessian(result)
        g_i = df_m(z_s)
        H_i=  df_H(z_s)
        g_μ_1 = (g_i+η[2]*(η[1]-λ_iblr[1]))/λ_iblr[2]
        g_μ_2= -H_i+η[2]-λ_iblr[2]
        g̃=[g_μ_1;g_μ_2]
        update!(opt,λ_iblr,g̃,msg_in.dist)

        # ---- START Δ_FE Algo
        #TODO : Note that FE is calculated after First update, it can be calculated
        # with prior params (but then KL part becomes 0)
        if i ==1
            F_first = KL_bc(λ_iblr,η,msg_in.dist) - logp_nc(z_s)
            stats.F_best = F_first
            stats.F_prev = deepcopy(stats.F_best)
            stats.F_best_idx = 1
            push!(F,1=>F_first)
        end
        # If iteration # is smaller then burn_in_min, dont calculate FE
        if i <= burn_in_min
            nothing
        # First,Calc FE every eval_elbo_window'th step
        elseif mod(i,eval_elbo_window) == 0
            FE = KL_bc(λ_iblr,η,msg_in.dist) - logp_nc(z_s)
            push!(F,i=>FE)
            #First condition to start tests: FE is smaller than initial FE
            if FE_check == false && FE < stats.F_best
                FE_check = true # FE dropped below initial ELBO
                println("FE_check starts when i=$i")
            #Second condition to start tests: i exceeds burn_in_max period
            elseif FE_check == false && i > burn_in_max
                FE_check = true
                println("FE_check starts when i=$i")
            end
            # If tests start, check convergence
            if FE_check
                is_FE_converged = ΔFE_check!(stats,FE,i,tolerance)
                # Store λ yielding minimum Free Energy
                if stats.F_best_idx == i
                    λ_iblr_best = deepcopy(λ_iblr)
                end
            end
            # If converged, stop iterations
            if is_FE_converged
                println("Algorithm converged at iteration $i")
                break
            end
        end
        # ---- End Δ_FE ALGO
    end # End for loop
    if is_FE_converged
        # return best iterate
        #λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr_best)
        #return last iterate
        λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    else
        #return last iterate
        λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    end
    return λ_natural_posterior,stats
end
# iBLR with Rhat diagnostic for convergence
function oneWindowSimulation_MCMC(J::F,Window::F,opts,η::Vector,λ_initials::Vector,logp_nc::Function,is_first_sim :: Bool,msg_in::Message{<:Gaussian, Univariate}) where F<:Number
    # Chain Initialization
    if is_first_sim
        # First sim -> initial point is the same for all chains
        params_container = [[λ_initials] for j=1:J]
        opt_matrix =[deepcopy(opts) for j =1:J]
    else
        # Not first -> chains are at different points
        params_container = [[λ_initials[j]] for j=1:J]
        opt_matrix =deepcopy(opts)
    end
    # Run one simulation window for all chains
    # Start simulation --- This part is specific for each msg_in
    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)
    for n =1:Window
        for j=1:J
            # push λ_t as λ_t+1 so that it will get updated in-place
            # θ_matrix[j,end] is the last λ_iblr for chain j
            push!(params_container[j],deepcopy(params_container[j][end]))
            q = bcToStandardDist(msg_in.dist,params_container[j][end])
            z_s = sample(q)
            g_i = df_m(z_s)
            H_i=  df_H(z_s)
            g_μ_1 = (g_i+η[2]*(η[1]-params_container[j][end][1]))/(params_container[j][end][2])
            g_μ_2= -H_i+η[2]-(params_container[j][end][2])
            g̃=[g_μ_1;g_μ_2]
            # This updates λ_t+1 from λ_t
            update!(opt_matrix[j],params_container[j][end],g̃,msg_in.dist)
        end
    end
    last_params = [params_container[j][end] for j=1:J]
    # --- End of Simulation
    stats_dict = Dict()
    # Calculate Where to split the chain
    total_len =Window+1
    k =  mod(Window,4) # how many extra samples to include in burn_in
    burn_in = Int64((Window-k)/2)+1+k  # Half of a chain
    half_chain_end = 3*Int64((Window-k)/4)+1+k # Mid of last half
    n = half_chain_end - burn_in + 1 #Samples per split-chain
    # Split chains in half
    split_chains = [params_container[j][start_idx:end_idx] for j=1:J for (start_idx,end_idx) in zip((burn_in+1,half_chain_end+1),(half_chain_end,total_len))]
    # Calculate mean/variance of all split chains
    chain_means = [mean(chain) for chain in split_chains]
    chain_vars = [var(chain) for chain in split_chains]
    # Calculate necessary statistics for Rhat,ESS and MCSE
    W = mean(chain_vars) # mean of within chain variances
    B_n = var(chain_means) # B/n : variance of means of chains
    σ_2_plus = ((n-1)/n)*W+B_n
    # Calculate Rhat
    rhat = sqrt.(σ_2_plus./W)
    # Calculate Effective Sample Size (For Mean parameter only)
    # Eq.(10) in An improved Rhat for assessing convergence of MCMC paper
    # 2*J comes since we split each chain in half
    # Index [1] is for the mean parameter
    meanparam_ρ_tj= [autocor([λ[1] for λ in split_chains[j]]) for j=1:2*J];#ρ_tj
    meanparam_var_j=[chain_vars[j][1] for j=1:length(chain_vars)] #sj^2
    weighted_corr = mean(meanparam_var_j.*meanparam_ρ_tj,dims=1)[1]
    ρ_t = 1.0 .- (W[1] .- weighted_corr)./(σ_2_plus[1])
    # Calculate ESS
    ess = (2*J)*n/(1+2*sum(ρ_t))
    # Calculate Monte Carlo Standard Error (MCSE)
    mcse = sqrt(σ_2_plus[1]/ess)
    # Calculate Iterate Average
    # Note: Mean of means of chains is only true since chains have same sample size
    λ_bar = mean(chain_means)
    stats_dict=Dict(:rhat=>rhat,:ess=>ess,:mcse=>mcse,:λ_bar=>λ_bar)

    return last_params,opt_matrix,stats_dict
end

function renderCVI_Rhat(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate},
                   num_simulations,tau,J,Window)

    """
    improved Bayesian Learning Rule implementation for CVI node
        BC Parameters are mean and precision(=[μ,S]) for Gaussian
    """

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)

    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point
    # Rhat Diagnostic Params
    # J = 25
    # Window = 1000 #also n
    last_params,opt_matrix,stats_dict=oneWindowSimulation_MCMC(J,Window,opt,η,λ_iblr,logp_nc,true,msg_in)
    println("i=0,Last_params = $(last_params[1]),Rhat=$(stats_dict[:rhat])")
    for i = 1:num_simulations
        last_params,opt_matrix,stats_dict=oneWindowSimulation_MCMC(J,Window,opt_matrix,η,last_params,logp_nc,false,msg_in)
        println("i=$i,Last_params = $(last_params[1]),Rhat=$(stats_dict[:rhat])")
        if all(stats_dict[:rhat] .< tau)
            break
        end
    end
    #TODO: DO the below iteration when Rhat converges or WARN THE USER that VI might not be converged
    # After stationary point has been found / Max number of simulations
    for i=1:num_simulations
        last_params,opt_matrix,stats_dict=oneWindowSimulation_MCMC(J,Window,opt_matrix,η,last_params,logp_nc,false,msg_in)
        # Check MCSE and ESS
        if all(stats_dict[:mcse] .< 0.1) && all(stats_dict[:ess] .> 15.0)
            println("Converged Parameters using Iterate Averaging = $(stats_dict[:λ_bar])")
            break
        end
    end
    #TODO: Return Natural Params in Normal Implementation
    return last_params,stats_dict

end
# Pareto shape parameter fit function
function Pareto_k_fit(logp_nc::Function,msg_in::Message{<:Gaussian, Univariate},λ_bc::Vector,S::F) where F<:Number
    #S:number of samples
    λ_natural  = bcToNaturalParams(msg_in.dist,λ_bc)
    logp(z)= logPdf(msg_in.dist,naturalParams(msg_in.dist),z)
    joint_pdf(z) = exp(logp(z)+logp_nc(z))
    q = bcToStandardDist(msg_in.dist,λ_bc)
    q_pdf(z) = exp(logPdf(q,λ_natural,z))
    rs(z) = joint_pdf(z)/q_pdf(z)
    if S<=225
        M = Int64(ceil(S/5))
    else
        M = Int64(ceil(3*sqrt(S)))
    end
    #
    samples = sample(q,S)
    rs_samples = rs.(samples)
    data = rs_samples[partialsortperm(rs_samples, 1:M,rev=true)]
    n=length(data)
    # Fit using M largest

    # MLE
    temp = 0.0
    l_xm = log(minimum(data))
    for i=1:n
        temp += log(data[i]) - l_xm
    end
    k̂ = n/temp

    # Zhang and Stephens(2009) method
    X =  percentile(data,25) #1st quartile
    X_n = maximum(data)
    m = 20+floor(sqrt(n))
    θj_func(j) = (1/X_n)+floor(1-sqrt((m)/(j-0.5)))/(3*X)
    k_func(θ)=-1.0*mean([log(1-θ*Xi) for Xi in data])
    l_func(θ) = n*(log(θ\k_func(θ))+k_func(θ)-1)
    θj_arr = θj_func.(1:m)
    wj_func(θ_j) = 1/sum(exp.(l_func.(θj_arr).-l_func(θ_j)))
    wj_arr =wj_func.(θj_arr)

    θ_new_est =sum(θj_arr.*wj_arr)
    k̂_new = -1.0*mean(log.(1.0 .- θ_new_est*data))
    #Bias Correction
    if S<1000
        k̂_new = (M*k̂_new+5)/(M+10)
    end
    return k̂,k̂_new,data
end
# iBLR with Pareto shape parameter fit return
function renderCVI_Pareto(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate},
                   S::F) where F<:Number

    """
    improved Bayesian Learning Rule implementation for CVI node
        BC Parameters are mean and precision(=[μ,S]) for Gaussian
    """

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_H(z) = ForwardDiff.derivative(df_m,z)

    η = deepcopy(bcParams(msg_in.dist)) # Prior BC parameters
    λ_iblr = Vector{Float64}(undef,2) # Posterior BC parameter vector
    λ_iblr[2] = deepcopy(-2*λ_init[2]) # Precision
    λ_iblr[1] = deepcopy(λ_init[1]/λ_iblr[2]) # Mean
    opt.stable_params = deepcopy(λ_iblr) # initialize stable point
    for i=1:num_iterations
        q = bcToStandardDist(msg_in.dist,λ_iblr)
        z_s = sample(q)
        g_i = df_m(z_s)
        H_i=  df_H(z_s)
        g_μ_1 = (g_i+η[2]*(η[1]-λ_iblr[1]))/λ_iblr[2]
        g_μ_2= -H_i+η[2]-λ_iblr[2]
        g̃=[g_μ_1;g_μ_2]
        update!(opt,λ_iblr,g̃,msg_in.dist)
    end

    λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    k̂,k̂_new,max_samples  = Pareto_k_fit(logp_nc,msg_in,λ_iblr,S)
    return λ_natural_posterior,k̂,k̂_new,max_samples
end
# iBLR original for Multivariate Gaussian
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Multivariate})

    df_m(z) = ForwardDiff.gradient(logp_nc,z)
    df_H(z) = ForwardDiff.jacobian(df_m,z)

    # λ_init are Natural Parameters for MV Gaussian
    #params = [vec(μ),mat(S)]
    n = dims(msg_in.dist)
    m_prior = deepcopy(unsafeMean(msg_in.dist))
    S_prior = deepcopy(unsafePrecision(msg_in.dist))
    S_t = deepcopy(reshape(-2*λ_init[n+1:end],n,n))
    m_t = deepcopy(S_t*λ_init[1:n])
    params = [m_t,S_t]
    opt.stable_params= deepcopy(params)
    for i=1:num_iterations
        q = bcToStandardDist(msg_in.dist,[params[1];vec(params[2])])
        z_s = sample(q)
        g_i = df_m(z_s)
        H_i = df_H(z_s)
        # Compute natural gradients of BCN parametrization
        s_inv = deepcopy(cholinv(params[2]))
        g_μ_1 = s_inv*(g_i+S_prior*(m_prior-params[1]))
        g_μ_2= -H_i+S_prior-params[2]
        g̃=[g_μ_1,g_μ_2]
        update!(opt,params,g̃,msg_in.dist,s_inv)

    end
    λ_natural_posterior = [params[2]*params[1];vec(-0.5*params[2])]
    return λ_natural_posterior
end
# CVI original for Multivariate Gaussian
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Multivariate})

    df_m(z) = ForwardDiff.gradient(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.jacobian(df_m,z)
    # CVI
    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)
    # CVI
    for i=1:num_iterations
        q = standardDist(msg_in.dist,λ)
        z_s = sample(q)
        df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
        df_μ2 = df_v(z_s)
        ∇f = [df_μ1; vec(df_μ2)]
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(opt,λ,∇)
        if isProper(standardDist(msg_in.dist,λ)) == false
            λ = λ_old
        end
    end
    return λ
end
# CVI original for Default FactorNode
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:FactorNode, <:VariateType})

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    A(η) = logNormalizer(msg_in.dist,η)
    gradA(η) = A'(η) # Zygote
    Fisher(η) = ForwardDiff.jacobian(gradA,η) # Zygote throws mutating array error
    for i=1:num_iterations
        q = standardDist(msg_in.dist,λ)
        z_s = sample(q)
        logq(λ) = logPdf(q,λ,z_s)
        ∇logq = logq'(λ)
        ∇f = Fisher(λ)\(logp_nc(z_s).*∇logq)
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(opt,λ,∇)
        if isProper(standardDist(msg_in.dist,λ)) == false
            λ = λ_old
        end
    end

    return λ

end
# CVI original for Categorical FactorNode
function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Categorical, <:VariateType})

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    A(η) = logNormalizer(msg_in.dist,η)
    gradA(η) = A'(η) # Zygote
    Fisher(η) = FIM(msg_in.dist,η) # To avoid Zygote's mutating array error
    for i=1:num_iterations
        q = standardDist(msg_in.dist,λ)
        z_s = sample(q)
        logq(λ) = logPdf(q,λ,z_s)
        ∇logq = logq'(λ)
        ∇f = Fisher(λ)\(logp_nc(z_s).*∇logq)
        λ_old = deepcopy(λ)
        ∇ = λ .- η .- ∇f
        update!(opt,λ,∇)
        if isProper(standardDist(msg_in.dist,λ)) == false
            λ = λ_old
        end
    end

    return λ

end

#---------------------------
# Some helpers
#---------------------------

isUnivariateGaussian(dist::ProbabilityDistribution{Univariate, F}) where F<:Gaussian = true
isUnivariateGaussian(dist::ProbabilityDistribution{V, F}) where {V<:VariateType, F<:FactorFunction} = false
isMultivariateGaussian(dist::ProbabilityDistribution{Multivariate, F}) where F<:Gaussian = true
isMultivariateGaussian(dist::ProbabilityDistribution{V, F}) where {V<:VariateType, F<:FactorFunction} = false

function convertToProperMessage(msg_in::Message{<:Gaussian, Univariate}, λ_message_org::Vector)
    λ_message = deepcopy(λ_message_org)
    w_message = -2*λ_message[2]
    if w_message<0 w_message = tiny end
    λ_message[2] = -0.5*w_message
    return λ_message
end

function convertToProperMessage(msg_in::Message{<:Gaussian, Multivariate}, λ_message_org::Vector)
    λ_message = deepcopy(λ_message_org)
    d = dims(msg_in.dist)
    W_message = -2*reshape(λ_message[d+1:end],(d,d))
    e_vals = eigvals(W_message)
    # below makes min eigen value zero. Later on in standardMessage(), we add tiny to ensure posdef
    if minimum(e_vals)<0 W_message -= minimum(e_vals)*diageye(d) end
    λ_message[d+1:end] = vec(-0.5*W_message)
    return λ_message
end


#---------------------------
# Custom inbounds collectors
#---------------------------

function collectSumProductNodeInbounds(node::CVI, entry::ScheduleEntry)
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]

    push!(inbounds, node.id)

    multi_in = (length(node.interfaces) > 2) # Boolean to indicate a multi-inbound nonlinear node
    inx = findfirst(isequal(entry.interface), node.interfaces) - 1 # Find number of inbound interface; 0 for outbound

    if (inx > 0) && multi_in # Multi-inbound backward rule
        push!(inbounds, Dict{Symbol, Any}(:inx => inx, # Push inbound identifier
                                          :keyword => false))
    end

    for node_interface in node.interfaces
        inbound_interface = ultimatePartner(node_interface)

        if (node_interface == entry.interface != node.interfaces[1])
            # Collect the incoming message

            if typeof(inbound_interface.node) == Equality
                # If CVI is connected to Equality node, incoming message is often not available
                # in time series models. The rules here allow us to use the message coming to
                # interface 1 of equality node.
                if inbound_interface == inbound_interface.node.interfaces[2]
                    push!(inbounds, interface_to_schedule_entry[ultimatePartner(inbound_interface.node.interfaces[1])])
                else
                    push!(inbounds, interface_to_schedule_entry[inbound_interface])
                end
            else
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            end
            #push!(inbounds, interface_to_schedule_entry[inbound_interface])
        # elseif (node_interface == node.interfaces[1] != entry.interface)
        #     # Collect the BP message from out interface
        #     push!(inbounds, interface_to_schedule_entry[inbound_interface])
        elseif node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        else
            # Collect entry from marginal schedule
            # try
            #     push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
            # catch
            #     # This rule is useful for the last time step in a time series model with Structured VMP
            #     push!(inbounds, interface_to_schedule_entry[inbound_interface])
            # end
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    return inbounds
end
