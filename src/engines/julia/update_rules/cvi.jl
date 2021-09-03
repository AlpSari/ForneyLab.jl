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


# # TODO: Make λ_init in renderCVI functions BCN parameters
# # TODO: Test all functions defined here
# # Standard parameters to BC parameters
# bcParams(dist::ProbabilityDistribution{Univariate, F}) where F<:Gaussian = [unsafeMean(dist),unsafePrecision(dist)]
# bcParams(dist::ProbabilityDistribution{Multivariate, F}) where F<:Gaussian = [vec(unsafeMean(dist)); vec(unsafePrecision(dist))]
# # BC parameters to Natural parameters
# function bcToNaturalParams(dist::ProbabilityDistribution{Univariate, F}, η::Vector) where F<:Gaussian
#     λ = [η[1]*η[2],-0.5*η[2]]
# end
# function bcToNaturalParams(dist::ProbabilityDistribution{Multivariate, F}, η::Vector) where F<:Gaussian
#     d = dims(dist)
#     λ=[η[1:d].*η[d+1:end];-0.5*η[d+1:end]]
# end
# # BC parameters to standard dist. type
# function bcTostandardDist(dist::ProbabilityDistribution{Univariate, F}, η::Vector) where F<:Gaussian
#     ProbabilityDistribution(Univariate, GaussianWeightedMeanPrecision,xi=η[1]*η[2],w=η[2])
# end
#
# function bcTostandardDist(dist::ProbabilityDistribution{Multivariate, F}, η::Vector) where F<:Gaussian
#     d = dims(dist)
#     XI, W = η[d+1:end].*η[1:d], reshape(η[d+1:end],d,d)
#     W = Matrix(Hermitian(W + tiny*diageye(d))) # Ensure precision is always invertible
#     ProbabilityDistribution(Multivariate, GaussianWeightedMeanPrecision,xi=XI,w=W)
# end


function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Univariate})

    df_m(z) = ForwardDiff.derivative(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.derivative(df_m,z)

    flag_alp = true
    flag_same_samples = true
    flag_init_print = false
    if flag_alp
       #CVI Params
       η = deepcopy(naturalParams(msg_in.dist))
       λ = deepcopy(λ_init)
       # First Code Params
       m_prior = deepcopy(unsafeMean(msg_in.dist))
       m_t = deepcopy(unsafeMean(msg_in.dist))
       τ = η[2]*-2.0
       #prec_t = deepcopy(unsafePrecision(msg_in.dist))
       Σ = deepcopy(unsafeVar(msg_in.dist))
       S = 1/Σ-τ
       λ_alp = [(1/Σ)*m_t,-0.5*(1/Σ)]
       # Second Code Params
       m_prior_2 = deepcopy(unsafeMean(msg_in.dist))
       m_t_2 = deepcopy(unsafeMean(msg_in.dist))
       prec_prior_2 = η[2]*-2.0
       prec_t_2 = deepcopy(unsafePrecision(msg_in.dist))
       λ_alp2 = [prec_t_2*m_t_2,-0.5*prec_t_2]
       if flag_init_print
           println("Initially: [m_t,prec_t] = [$m_t,$prec_t],[m_t_2,prec_t_2] = [$m_t_2,$prec_t_2], λ_cvi = $λ")
           flag_init_print=false
       end
       if flag_same_samples
           for i=1:num_iterations
               #println("m_prior =$m_prior,Term = $(τ*m_prior)")
               q = standardDist(msg_in.dist,λ)
               z_s = sample(q)
               # CVI Gradient (descent direction)
               df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
               df_μ2 = df_v(z_s)
               ∇f = [df_μ1, df_μ2]
               ∇ = λ .- η .- ∇f
               # First code gradient (ascend direction)
               g_i = df_m(z_s)
               H_i = 2*df_v(z_s)
               #∇_1 = [(1.0/prec_t)*(g_i+prec_t*(m_prior-m_t)),(τ-H_i-prec_t)]
               S = 0.9*S+0.1*(-H_i)
               Σ = 1/(S+τ)
               m_t =m_t -0.1*Σ*(-g_i+τ*(m_t-m_prior))
               λ_alp = [(1/Σ)*m_t,-0.5*(1/Σ)]
               # Second code gradient (ascend direction)
               g_i_2 = df_m(z_s)
               H_i_2= 2*df_v(z_s)
               ∇_2 = [(1/prec_t_2)*(g_i_2+prec_prior_2*(m_prior_2-m_t_2)),-H_i_2+prec_prior_2-prec_t_2]
               # Update CVI
               λ -= 0.1*∇
               # Update first code params
               # m_t += 0.1*∇_1[1]
               # prec_t += 0.1*∇_1[2]
               # λ_alp = [prec_t*m_t,-0.5*prec_t]
               # Update second code params
               #NOTE: PRECISION must be updated before MEAN for this scheme to work
               #NOTE: This should be changed in Improved Bayesian Learning
               prec_t_2 += 0.1*∇_2[2]
               m_t_2 += 0.1*(1/prec_t_2)*(g_i_2+prec_prior_2*(m_prior_2-m_t_2))
               λ_alp2 = [prec_t_2*m_t_2,-0.5*prec_t_2]
           end
       else
           for i=1:num_iterations
               q = standardDist(msg_in.dist,λ_alp)
               z_s = sample(q)
               # grad + hessian for nonconj factor
               g_i = df_m(z_s)
               H_i = 2*df_v(z_s)
               # updates (from Nat Grad CVI Eqn16-17 + p(z) deki meani hesaba kat)
               m_t = m_t + 0.1*(1.0/prec_t)*(g_i+prec_t*(m_prior-m_t))
               prec_t = prec_t+0.1*(τ-H_i-prec_t)
               λ_alp = [prec_t*m_t,-0.5*prec_t]
               #m_t = m_t - 0.1*(1.0/prec_t)*(τ*m_t-g_i-m_prior)
           end
           λ_alp = [prec_t*m_t,-0.5*prec_t]
           # More organized code
           for i=1:num_iterations
               q_2 = standardDist(msg_in.dist,λ_alp2)
               z_s_2 = sample(q_2)
               # grad + hessian for nonconj factor
               g_i_2 = df_m(z_s_2)
               H_i_2= 2*df_v(z_s_2)
               df_μ1_2 = (1/prec_t_2)*g_i_2+m_prior_2-m_t_2
               df_μ2_2 = -H_i_2+prec_prior_2-prec_t_2
               m_t_2 += 0.1*df_μ1_2
               prec_t_2 += 0.1*df_μ2_2
               λ_alp2 = [prec_t_2*m_t_2,-0.5*prec_t_2]

           end
           λ_alp2 = [prec_t_2*m_t_2,-0.5*prec_t_2]
        end
        # CVI Original w/ naturalParams
        for i=1:num_iterations
           q = standardDist(msg_in.dist,λ)
           z_s = sample(q)
           df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
           df_μ2 = df_v(z_s)
           ∇f = [df_μ1, df_μ2]
           λ_old = deepcopy(λ)
           ∇ = λ .- η .- ∇f
           update!(opt,λ,∇)
           # if isProper(standardDist(msg_in.dist,λ)) == false
           #     λ = λ_old
           # end
        end
       end


    println("λ_alp = $λ_alp,λ_alp2 = $λ_alp2, λ_cvi = $λ")
    #println(λ)
    return λ

end

function renderCVI(logp_nc::Function,
                   num_iterations::Int,
                   opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent},
                   λ_init::Vector,
                   msg_in::Message{<:Gaussian, Multivariate})

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
