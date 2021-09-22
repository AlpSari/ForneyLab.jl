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

# Additional definitions for iBLR implementation of CVI
mutable struct iBLR
    eta::Float64
    state::Int64
    stable_params::Any
end

mutable struct FE_stats
    F_prev::Float64 # FE from previous iterate
    F_best::Float64 # Minimum attained FE
    F_best_idx:: Int64 # Index of Minimum attained FE
    F_converge_idx::Int64 # Index of Minimum attained FE
    ΔF_rel::Float64 # Relative difference between F_now and F_prev
    ΔF_vect::Vector{Float64}
end # struct
FE_stats()=FE_stats(Inf,Inf,-1,-1,Inf,Vector{Float64}())
function median(a::F) where F<:AbstractArray
    #TODO: TRY TO USE median of Statistics.jl package
    len = size(a)[1]
    if mod(len,2) ==1
        middle = Int64((len+1)/2)
        return sort(a)[middle]
    else
        middle = Int64(len/2)
        return mean(sort(a)[middle:middle+1])
    end
end

function ΔFE_check!(stats::FE_stats,F_now::Float64,idx_now::Int64,tolerance::Float64)
    # Return true if Free Energy is converged, return false else
    if F_now < stats.F_best
        stats.F_best = deepcopy(F_now)
        stats.F_best_idx = deepcopy(idx_now)
    end
    stats.ΔF_rel = abs(100*(F_now-stats.F_prev)/(stats.F_prev))
    push!(stats.ΔF_vect,stats.ΔF_rel)
    # Calculate mean/median
    vect_mean = mean(stats.ΔF_vect)
    vect_median = median(stats.ΔF_vect)
    if vect_mean < tolerance || vect_median < tolerance
        stats.F_converge_idx = deepcopy(idx_now)
        stats.F_prev = deepcopy(F_now) #Also becomes F at convergence
        return true
    else # No convergence, get ready for the next call of the function
    stats.F_prev = deepcopy(F_now)
        return false
    end
end


iBLR()=iBLR(0.1,0,nothing)
iBLR(eta::F) where F <: Number = iBLR(eta,0,nothing)
iBLR(eta,state) = iBLR(eta,state,nothing)
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
function update1!(opt::iBLR,params,natgrad,prior::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
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
end
function update1!(opt::iBLR,params,natgrad,prior::ProbabilityDistribution{Multivariate, F},s_inv::Array{Float64,2}) where F <: Gaussian
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
function update1!(opt::iBLR,params,natgrad,prior::ProbabilityDistribution{Multivariate, F}) where F <: Gaussian
    s_inv = deepcopy(cholinv(params[2]))
    update!(opt,params,natgrad,prior,s_inv)
end


function KL_bc(λ::Vector,λ_0::Vector,dist::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
    # 0.5 [log(σ^2/σ_0^2)+(σ^2+(μ-μ_0)^2)/(σ_0)^2-1]
    return 0.5 *(log(λ[2]/λ_0[2])+ (λ[2]+(λ[1]-λ_0[1])^2)/(λ_0[2])-1)
end

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
        update1!(opt,λ_iblr,g̃,msg_in.dist)
    end
    λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    return λ_natural_posterior
end

function renderCVI_Δ_FE(logp_nc::Function,
                   num_iterations::Int,
                   opt::iBLR,
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})
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
    tolerance = 0.005
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
        update1!(opt,λ_iblr,g̃,msg_in.dist)

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
        λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr_best)
    else
        #return last iterate
        λ_natural_posterior = bcToNaturalParams(msg_in.dist,λ_iblr)
    end
    return λ_natural_posterior
end

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
        update1!(opt,params,g̃,msg_in.dist,s_inv)

    end
    λ_natural_posterior = [params[2]*params[1];vec(-0.5*params[2])]
    return λ_natural_posterior
end

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
