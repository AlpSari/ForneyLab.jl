export ruleSPSMPOutVD, ruleSPSMPIn1MV, ruleSPSMPOutVDX, ruleSPSMPInX

# function ruleSPSMPIn1MV(node_id::Symbol,
#                         msg_out::Message{<:FactorFunction, <:VariateType},
#                         msg_in::Nothing)
#
#     @show msg_in
#     @show typeof(msg_in)
#     @show msg_out
#     @show node_id
#     msg_in
# end

# function ruleSPSMPIn1MV(node_id::Symbol,
#                         msg_out::Message{<:FactorFunction, <:VariateType},
#                         msg_in::Nothing)
#
#     thenode = currentGraph().nodes[node_id]
#
#     η = deepcopy(naturalParams(msg_in.dist))
#     λ = deepcopy(η)
#
#     logp_nc(z) = logPdf(thenode.back_message_types, thenode.g(z))
#     df_m(z) = ForwardDiff.derivative(logp_nc,z)
#     df_v(z) = 0.5*ForwardDiff.derivative(df_m,z)
#     for i=1:thenode.num_iterations
#         q = standardDist(msg_in.dist,λ)
#         z_s = sample(q)
#         df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(q)
#         df_μ2 = df_v(z_s)
#         ∇f = [df_μ1, df_μ2]
#         ∇ = λ - η - ∇f
#         update!(thenode.opt,λ,∇)
#     end
#
#     return standardMessage(msg_in.dist,λ-η)
#
# end

function ruleSPSMPIn1MV(node_id::Symbol,
                        msg_out::Message{<:FactorFunction, <:VariateType},
                        msg_in::Nothing)

    thenode = currentGraph().nodes[node_id]

    λ = deepcopy(naturalParams(thenode.back_message_types))

    logp_nc(z) = logPdf(msg_out.dist, thenode.g(z))
    df_m(z) = ForwardDiff.gradient(logp_nc,z)
    df_v(z) = 0.5*ForwardDiff.jacobian(df_m,z)
    for i=1:thenode.num_iterations
        ν = standardDist(thenode.back_message_types,λ)
        z_s = sample(ν)
        df_μ1 = df_m(z_s) - 2*df_v(z_s)*mean(ν)
        df_μ2 = df_v(z_s)
        ∇f = [df_μ1; vec(df_μ2)]
        λ_old = deepcopy(λ)
        ∇ = λ - ∇f
        update!(thenode.opt,λ,∇)
        if isProper(standardDist(thenode.back_message_types,λ)) == false
            #@show node_id
            #@show i
            λ = deepcopy(λ_old)
        end
    end
    @show node_id
    @show standardMessage(thenode.back_message_types,λ)
    return standardMessage(thenode.back_message_types,λ)
end

# function ruleSPSMPOutVD(node_id::Symbol,
#                         msg_out::Nothing,
#                         msg_in::ProbabilityDistribution)
#
#     thenode = currentGraph().nodes[node_id]
#
#     samples = thenode.g.(sample(msg_in, thenode.num_samples))
#     weights = ones(thenode.num_samples)/thenode.num_samples
#
#     if length(samples[1]) == 1
#         variate = Univariate
#     else
#         variate = Multivariate
#     end
#
#     q=ProbabilityDistribution(variate, SampleList, s=samples, w=weights)
#     q.params[:entropy] = 0
#
#     return Message(variate,SetSampleList,q=q,node_id=node_id)
#
# end

function ruleSPSMPOutVD(node_id::Symbol,
                        msg_out::Nothing,
                        msg_in::Message)

    thenode = currentGraph().nodes[node_id]

    #samples = thenode.g.(sample(msg_in.dist, thenode.num_samples))
    samples = thenode.g.(sample(thenode.back_message_types, thenode.num_samples))
    weights = ones(thenode.num_samples)/thenode.num_samples

    if length(samples[1]) == 1
        variate = Univariate
    else
        variate = Multivariate
    end

    q=ProbabilityDistribution(variate, SampleList, s=samples, w=weights)
    q.params[:entropy] = 0

    return Message(variate,SetSampleList,q=q,node_id=node_id)

end

function ruleSPSMPInX(node_id::Symbol,
                      inx::Int64,
                      msg_out::Message{<:FactorFunction, <:VariateType},
                      msgs_in::Vararg{Union{Nothing,ProbabilityDistribution}})

    @show msgs_in
    msgs_in[inx]
end

function ruleSPSMPOutVDX(node_id::Symbol,
                         msg_out::Nothing,
                         msgs_in::Vararg{ProbabilityDistribution})

    thenode = currentGraph().nodes[node_id]

    samples_in = [sample(msg_in, thenode.num_samples) for msg_in in msgs_in]

    samples = thenode.g.(samples_in...)
    weights = ones(thenode.num_samples)/thenode.num_samples

    if length(samples[1]) == 1
        variate = Univariate
    else
        variate = Multivariate
    end

    q=ProbabilityDistribution(variate, SampleList, s=samples, w=weights)
    q.params[:entropy] = 0

    return Message(variate,SetSampleList,q=q,node_id=node_id)

end


#---------------------------
# Custom inbounds collectors
#---------------------------

function collectSumProductNodeInbounds(node::SMP, entry::ScheduleEntry)
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
        if node_interface === entry.interface
            # Ignore inbound message on outbound interface
            push!(inbounds, nothing)
        elseif isClamped(inbound_interface)
            # Hard-code outbound message of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Message))
        else
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    # for node_interface in node.interfaces
    #     inbound_interface = ultimatePartner(node_interface)
    #
    #     if (node_interface == node.interfaces[1] != entry.interface)
    #         # Collect the BP message from out interface
    #         push!(inbounds, interface_to_schedule_entry[inbound_interface])
    #     elseif node_interface === entry.interface
    #         # Ignore marginal of outbound edge
    #         push!(inbounds, nothing)
    #     elseif isClamped(inbound_interface)
    #         # Hard-code marginal of constant node in schedule
    #         push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
    #     else
    #         # Collect entry from marginal schedule
    #         try
    #             push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
    #         catch
    #             # This rule is useful for the last time step in a time series model with Structured VMP
    #             push!(inbounds, interface_to_schedule_entry[inbound_interface])
    #         end
    #     end
    # end

    return inbounds
end
