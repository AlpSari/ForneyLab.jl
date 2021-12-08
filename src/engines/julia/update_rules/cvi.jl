export ruleSPCVIOutNFactorNode, ruleSPCVIIn1Factor, ruleSPCVIOutNFactorNodeX, ruleSPCVIInFactorX, ruleMCVIFactorX

function ruleSPCVIIn1Factor(node_id::Symbol,
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
    λ = renderCVI(logp_nc,thenode.num_iterations,thenode.opt,λ_init,msg_in,thenode.convergence_optimizer)

    λ_message = λ.-η
    # Implement proper message check for all the distributions later on.
    thenode.q = [standardDist(msg_in.dist,λ)]
    if thenode.online_inference == false thenode.q_memory = deepcopy(thenode.q) end
    return standardMessage(msg_in.dist,λ_message)
end

function ruleSPCVIIn1Factor(node_id::Symbol,
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
    λ = renderCVI(logp_nc,thenode.num_iterations,thenode.opt,λ_init,msg_in,thenode.convergence_optimizer)

    λ_message = λ.-η

    if thenode.proper_message λ_message = convertToProperMessage(msg_in, λ_message) end
    thenode.q = [standardDist(msg_in.dist,λ)]
    if thenode.online_inference == false thenode.q_memory = deepcopy(thenode.q) end
    return standardMessage(msg_in.dist,λ_message)
end

function ruleSPCVIIn1Factor(node_id::Symbol,
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
    λ = renderCVI(logp_nc,thenode.num_iterations,thenode.opt,λ_init,msg_in,thenode.convergence_optimizer)

    λ_message = λ.-η

    if thenode.proper_message λ_message = convertToProperMessage(msg_in, λ_message) end
    thenode.q = [standardDist(msg_in.dist,λ)]
    if thenode.online_inference == false thenode.q_memory = deepcopy(thenode.q) end
    return standardMessage(msg_in.dist,λ_message)
end

function ruleSPCVIOutNFactorNode(node_id::Symbol,
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

function ruleSPCVIInFactorX(node_id::Symbol,
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
            logp_nc = (z) -> sum(logPdf.([msg_out.dist],thenode.g.(arg_sample(z,inx_list[j])...)))/thenode.num_samples
            if thenode.online_inference[inx_list[j]]
                λ_init = deepcopy(naturalParams(thenode.q[inx_list[j]]))
                logp_nc = (z) -> (thenode.dataset_size/thenode.batch_size)*sum(logPdf.([msg_out.dist],thenode.g.(arg_sample(z,inx_list[j])...)))/thenode.num_samples
            else
                λ_init = deepcopy(naturalParams(msg_in.dist))
            end
            λ = renderCVI(logp_nc,thenode.num_iterations[inx_list[j]],thenode.opt[inx_list[j]],λ_init,msg_in,thenode.convergence_optimizer)
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

function ruleSPCVIOutNFactorNodeX(node_id::Symbol,
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

function ruleMCVIFactorX(node_id::Symbol,
                           msg_out::Message{<:FactorFunction, <:VariateType},
                           msgs_in::Vararg{Message})

    thenode = currentGraph().nodes[node_id]
    return ProbabilityDistribution(JointIndependentProbDist,marginals=thenode.q)
end

#---------------------------
# CVI implementations
#---------------------------

function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate},
                   convergence_optimizer::Nothing)

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.derivative(df_m,z)

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
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Multivariate},
                   convergence_optimizer::Nothing)

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.gradient(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.jacobian(df_m,z)

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
                   msg_in::Message{<:FactorNode, <:VariateType},
                   convergence_optimizer::Nothing)

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
                   msg_in::Message{<:Gaussian, Univariate},
                   convergence_optimizer::Conv) where Conv<:ConvergenceOptimizer

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.derivative(df_m,z)

    # Δ_FE method
    convergence_optimizer.stats = ConvergenceStatsFE() # RESET STATS
    stats = convergence_optimizer.stats
    # Init Free Energy Optimization Parameters
    eval_FE_window = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.eval_FE_window))
    burn_in_min = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_min))
    burn_in_max = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_max))
    FE_check = false
    is_FE_converged = false
    tolerance_mean = convergence_optimizer.tolerance_mean
    tolerance_median = convergence_optimizer.tolerance_median
    FE_max_threshold = 0.0

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

        #ΔFE
        # Burn in period
        if 1<= i<=burn_in_min
            #FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
            #FE_max_threshold += FE
            if i == 1 #burn_in_min
                FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
                #stats.F_best = FE_max_threshold/(burn_in_min)
                #stats.F_prev = deepcopy(stats.F_best)
                #stats.F_best_idx = burn_in_min
                stats.F_best = FE
                stats.F_prev = deepcopy(stats.F_best)
                stats.F_best_idx = 1
            end
        # End of burn in period
        elseif mod(i,eval_FE_window) == 0
            FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
            # First condition to start tests: FE is smaller than initial FE
            if FE_check == false && FE < stats.F_best
                FE_check = true # FE dropped below initial ELBO
            # Second condition to start tests: i exceeds burn_in_max period
            # elseif FE_check == false && i > burn_in_max
            #     FE_check = true
            end
            # If tests start, check convergence
            if FE_check
                is_FE_converged = ΔFE_check!(stats,FE,i,tolerance_mean,tolerance_median)
                # Store λ yielding minimum Free Energy
                if stats.F_best_idx == i
                    λ_best = deepcopy(λ)
                end
            end
            # If converged, stop iterations
            if is_FE_converged
                println("Algorithm converged at iteration $i")
                break
            end
        end
    end
    if is_FE_converged && @isdefined λ_best
        # return best iterate
        λ_natural_posterior = λ_best
    else
        #return last iterate
        λ_natural_posterior = λ
    end
    S = Int64(convergence_optimizer.pareto_num_samples)
    k̂_new  = Pareto_k_fit(logp_nc,msg_in,λ_natural_posterior,S)

    if isnan(k̂_new)
        #println("Importance ratios are 0, fitted Pareto shape parameter = $k̂_new")
        println("Warning! Convergence diagnostic indicator is = $k̂_new")
    elseif k̂_new >= convergence_optimizer.pareto_k_thr
        #println("Warning, fitted Pareto shape parameter = $k̂_new ")#≧ $(convergence_optimizer.pareto_k_thr)!")
        println("Warning! Convergence diagnostic indicator is = $k̂_new")
    else
        #println("Fitted Pareto shape parameter = $k̂_new")
        println("Convergence diagnostic indicator is = $k̂_new")
    end

    return λ_natural_posterior

end


function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Multivariate},
                   convergence_optimizer::Conv) where Conv<:ConvergenceOptimizer

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.gradient(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.jacobian(df_m,z)

    # Δ_FE method
    convergence_optimizer.stats = ConvergenceStatsFE() # RESET STATS
    stats = convergence_optimizer.stats
    # Init Free Energy Optimization Parameters
    eval_FE_window = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.eval_FE_window))
    burn_in_min = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_min))
    burn_in_max = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_max))
    FE_check = false
    is_FE_converged = false
    tolerance_mean = convergence_optimizer.tolerance_mean
    tolerance_median = convergence_optimizer.tolerance_median
    FE_max_threshold = 0.0

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
        #ΔFE
        # Burn in period
        if 1<= i<=burn_in_min
            #FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
            #FE_max_threshold += FE
            if i == 1 #burn_in_min
                FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
                #stats.F_best = FE_max_threshold/(burn_in_min)
                #stats.F_prev = deepcopy(stats.F_best)
                #stats.F_best_idx = burn_in_min
                stats.F_best = FE
                stats.F_prev = deepcopy(stats.F_best)
                stats.F_best_idx = 1
            end
        # End of burn in period
        elseif mod(i,eval_FE_window) == 0
            FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
            # First condition to start tests: FE is smaller than initial FE
            if FE_check == false && FE < stats.F_best
                FE_check = true # FE dropped below initial ELBO
            # Second condition to start tests: i exceeds burn_in_max period
            # elseif FE_check == false && i > burn_in_max
            #     FE_check = true
            end
            # If tests start, check convergence
            if FE_check
                is_FE_converged = ΔFE_check!(stats,FE,i,tolerance_mean,tolerance_median)
                # Store λ yielding minimum Free Energy
                if stats.F_best_idx == i
                    λ_best = deepcopy(λ)
                end
            end
            # If converged, stop iterations
            if is_FE_converged
                println("Algorithm converged at iteration $i")
                break
            end
        end
    end
    if is_FE_converged && @isdefined λ_best
        # return best iterate
        λ_natural_posterior = λ_best
    else
        #return last iterate
        λ_natural_posterior = λ
    end
    S = Int64(convergence_optimizer.pareto_num_samples)
    k̂_new  = Pareto_k_fit(logp_nc,msg_in,λ_natural_posterior,S)

    if isnan(k̂_new)
        #println("Importance ratios are 0, fitted Pareto shape parameter = $k̂_new")
        println("Warning! Convergence diagnostic indicator is = $k̂_new")
    elseif k̂_new >= convergence_optimizer.pareto_k_thr
        #println("Warning, fitted Pareto shape parameter = $k̂_new ")#≧ $(convergence_optimizer.pareto_k_thr)!")
        println("Warning! Convergence diagnostic indicator is = $k̂_new")
    else
        #println("Fitted Pareto shape parameter = $k̂_new")
        println("Convergence diagnostic indicator is = $k̂_new")
    end

    return λ_natural_posterior

end

function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:FactorNode, <:VariateType},
                   convergence_optimizer::Conv) where Conv<:ConvergenceOptimizer

    η = deepcopy(naturalParams(msg_in.dist))
    λ = deepcopy(λ_init)

    A(η) = logNormalizer(msg_in.dist,η)
    gradA(η) = A'(η) # Zygote
    Fisher(η) = ForwardDiff.jacobian(gradA,η) # Zygote throws mutating array error
    # Δ_FE method
    convergence_optimizer.stats = ConvergenceStatsFE() # RESET STATS
    stats = convergence_optimizer.stats
    # Init Free Energy Optimization Parameters
    eval_FE_window = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.eval_FE_window))
    burn_in_min = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_min))
    burn_in_max = Int64(ceil(convergence_optimizer.max_iterations*convergence_optimizer.burn_in_max))
    FE_check = false
    is_FE_converged = false
    tolerance_mean = convergence_optimizer.tolerance_mean
    tolerance_median = convergence_optimizer.tolerance_median
    FE_max_threshold = 0.0
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
        #ΔFE
        # Burn in period
        if 1<= i<=burn_in_min
            #FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
            #FE_max_threshold += FE
            if i == 1 #burn_in_min
                FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
                #stats.F_best = FE_max_threshold/(burn_in_min)
                #stats.F_prev = deepcopy(stats.F_best)
                #stats.F_best_idx = burn_in_min
                stats.F_best = FE
                stats.F_prev = deepcopy(stats.F_best)
                stats.F_best_idx = 1
            end
        # End of burn in period
        elseif mod(i,eval_FE_window) == 0
            FE = kl_Nat(λ,η,msg_in.dist) - logp_nc(z_s)
            # First condition to start tests: FE is smaller than initial FE
            if FE_check == false && FE < stats.F_best
                FE_check = true # FE dropped below initial ELBO
            # Second condition to start tests: i exceeds burn_in_max period
            # elseif FE_check == false && i > burn_in_max
            #     FE_check = true
            end
            # If tests start, check convergence
            if FE_check
                is_FE_converged = ΔFE_check!(stats,FE,i,tolerance_mean,tolerance_median)
                # Store λ yielding minimum Free Energy
                if stats.F_best_idx == i
                    λ_best = deepcopy(λ)
                end
            end
            # If converged, stop iterations
            if is_FE_converged
                println("Algorithm converged at iteration $i")
                break
            end
        end
    end
    if is_FE_converged && @isdefined λ_best
        # return best iterate
        λ_natural_posterior = λ_best
    else
        #return last iterate
        λ_natural_posterior = λ
    end
    S = Int64(convergence_optimizer.pareto_num_samples)
    k̂_new  = Pareto_k_fit(logp_nc,msg_in,λ_natural_posterior,S)

    if isnan(k̂_new)
        #println("Importance ratios are 0, fitted Pareto shape parameter = $k̂_new")
        println("Warning! Convergence diagnostic indicator is = $k̂_new")
    elseif k̂_new >= convergence_optimizer.pareto_k_thr
        #println("Warning, fitted Pareto shape parameter = $k̂_new ")#≧ $(convergence_optimizer.pareto_k_thr)!")
        println("Warning! Convergence diagnostic indicator is = $k̂_new")
    else
        #println("Fitted Pareto shape parameter = $k̂_new")
        println("Convergence diagnostic indicator is = $k̂_new")
    end

    return λ_natural_posterior

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

    multi_in = (length(node.interfaces) > 2) # Boolean to indicate a multi-inbound node
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
        elseif node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        else
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    return inbounds
end



# HELPERS

# Pareto shape parameter fit function - using naturalparameters
function Pareto_k_fit(logp_nc::Function,msg_in::Message{<:FactorNode, <:VariateType},λ_natural::Vector,S::F) where F<:Number
    #S:number of samples
    logp(z)= logPdf(msg_in.dist,naturalParams(msg_in.dist),z)
    joint_pdf(z) = exp(logp(z)+logp_nc(z))
    q = standardDist(msg_in.dist,λ_natural)
    q_pdf(z) = exp(logPdf(q,λ_natural,z))
    rs(z) = joint_pdf(z)/q_pdf(z) # importance ratio
    if S<=225
        M = Int64(ceil(S/5))
    else
        M = Int64(ceil(3*sqrt(S)))
    end
    #
    samples = sample(q,S)
    rs_samples = rs.(samples)
    if any(isnan.(rs_samples))
        k̂_new= NaN
        return k̂_new
    end
    data = rs_samples[partialsortperm(rs_samples, 1:M,rev=true)]# Fit using M largest
    n=length(data)
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
    return k̂_new
end
# KL(q||p)
function kl_Nat(λ::Vector,λ_0::Vector,dist::ProbabilityDistribution{Univariate, F}) where F <: Gaussian
    # 0.5 [log(σ^2/σ_0^2)+(σ^2+(μ-μ_0)^2)/(σ_0)^2-1]
    σ2_q = -0.5/λ[2]
    σ2_p = -0.5/λ_0[2]
    μ_q = λ[1]*σ2_q
    μ_p = λ_0[1]*σ2_p
    return 0.5 *(log(σ2_q/σ2_p)+ (σ2_q+(μ_q-μ_p)^2)/(σ2_p)-1)
end

function kl_Nat(λ::Vector,λ_0::Vector,dist::ProbabilityDistribution{Multivariate, F}) where F <: Gaussian
    d = dims(dist)
    XI_q, S = λ[1:d], reshape(-2*λ[d+1:end],d,d)
    XI_p, S_p = λ_0[1:d], reshape(-2*λ_0[d+1:end],d,d)
    Σ = cholinv(S)

    μ_q = Σ*XI_q
    μ_p = cholinv(S_p)*XI_p
    Δμ = μ_q-μ_p
    return 0.5*(logdet(S)-logdet(S_p)-d+tr(S_p*Σ)+dot(Δμ,S_p*Δμ))
end
function kl_Nat(λ::Vector,λ_0::Vector,dist::ProbabilityDistribution)
    q = standardDist(dist,λ)
    p = standardDist(dist,λ_0)
    z_s = sample(q)
    return logPdf(q,z_s)-logPdf(p,z_s)
end

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
    stats.ΔFE_rel = abs(100*(F_now-stats.F_prev)/(stats.F_prev))
    push!(stats.ΔFE_vect,stats.ΔFE_rel)
    # Calculate mean/median
    vect_mean = mean(stats.ΔFE_vect)
    vect_median = median(stats.ΔFE_vect)
    if (vect_mean < tolerance_mean && length(stats.ΔFE_vect)>100) || (vect_median < tolerance_median && length(stats.ΔFE_vect)>100)
        stats.F_converge_idx = deepcopy(idx_now)
        stats.F_prev = deepcopy(F_now) #Also becomes F at convergence
        return true
    else # No convergence, get ready for the next call of the function
        stats.F_prev = deepcopy(F_now)
        return false
    end
end
