############################################
# EqualityNode
############################################
# Description:
#   Equality constraint node with variable
#   number of (symmetrical) interfaces (ports).
#   Interfaces don't have names since the number
#   of interfaces is variable.
#
#   Example:
#       EqualityNode(name="3_port_equ") # 3 interfaces is default
#       EqualityNode(5; name="5_port_equ")
#
# Interface ids, (names) and supported message types:
#   1. (none):
#       Message{Float64}
#       Message{Array{Float64}}
#       Message{GaussianDistribution}
#       Message{GammaDistribution}
#       Message{InverseGammaDistribution}
#   2. (none):
#       Message{Float64}
#       Message{Array{Float64}}
#       Message{GaussianDistribution}
#       Message{GammaDistribution}
#       Message{InverseGammaDistribution}
#   3. (none):
#       Message{Float64}
#       Message{Array{Float64}}
#       Message{GaussianDistribution}
#       Message{GammaDistribution}
#       Message{InverseGammaDistribution}
#   ...
#   N. (none):
#       Message{Float64}
#       Message{Array{Float64}}
#       Message{GaussianDistribution}
#       Message{GammaDistribution}
#       Message{InverseGammaDistribution}
############################################

export EqualityNode

type EqualityNode <: Node
    num_interfaces::Uint16
    interfaces::Array{Interface,1}
    name::ASCIIString

    function EqualityNode(num_interfaces::Integer=3; name="unnamed")
        @assert(num_interfaces>2, "An EqualityNode should have at least 3 interfaces")
        self = new(num_interfaces, Array(Interface, num_interfaces), name)
        # Create interfaces
        for interface_id=1:num_interfaces
            self.interfaces[interface_id] = Interface(self)
        end
        return self
    end
end

# Overload firstFreeInterface since EqualityNode is symmetrical in its interfaces
function firstFreeInterface(node::EqualityNode)
    # Return id of first free interface of a symmetrical node
    for interface_id = 1:length(node.interfaces)
        if node.interfaces[interface_id].partner == nothing
            return interface_id
        end
    end
    error("No free interface on $(typeof(node)) $(node.name)")
end

############################################
# GaussianDistribution methods
############################################

# Rule set
# From: Korl (2005), "A Factor graph approach to signal modelling, system identification and filtering", Table 4.1
equalityMRule{T<:Number}(m_x::Array{T, 1}, m_y::Array{T, 1}, W_x::Array{T, 2}, W_y::Array{T, 2}) = pinv(W_x+W_y)*(W_x*m_x+W_y*m_y)
equalityVRule{T<:Number}(V_x::Array{T, 2}, V_y::Array{T, 2}) = V_x*pinv(V_x+V_y)*V_y
equalityWRule{T<:Number}(W_x::Array{T, 2}, W_y::Array{T, 2}) = W_x+W_y
equalityXiRule{T<:Number}(xi_x::Array{T, 1}, xi_y::Array{T, 1}) = xi_x+xi_y

function updateNodeMessage!(node::EqualityNode,
                            outbound_interface_id::Int,
                            outbound_message_value_type::Type{GaussianDistribution},
                            inbounds...)
    # Calculate an outbound message based on the inbound messages and the node function.
    # This function is not exported, and is only meant for internal use.
    dist_result = getOrCreateMessage(node.interfaces[outbound_interface_id], GaussianDistribution).value

    # The following update rules correspond to node 1 from Table 4.1 in:
    # Korl, Sascha. “A Factor Graph Approach to Signal Modelling, System Identification and Filtering.” Hartung-Gorre, 2005.
    first_incoming_id = (outbound_interface_id==1) ? 2 : 1
    (typeof(inbounds[first_incoming_id])==Message{GaussianDistribution}) || error("EqualityNode: cannot calculate outbound Message{GaussianDistribution} from inbound $(typeof(inbounds[first_incoming_id]))")
    
    if length(node.interfaces)>3
        # Always calculate the (xi,W) parametrization of all incoming messages if it's not already present.

        # Create accumulators for W and xi of the correct size
        if !is(inbounds[first_incoming_id].value.W, nothing)
            W_sum = zeros(size(inbounds[first_incoming_id].value.W))
        else
            W_sum = zeros(size(inbounds[first_incoming_id].value.V))
        end
        if !is(inbounds[first_incoming_id].value.xi, nothing)
            xi_sum = zeros(size(inbounds[first_incoming_id].value.xi))
        else
            xi_sum = zeros(size(inbounds[first_incoming_id].value.m))
        end

        # Sum W and xi of all incoming messages
        for interface_id = 1:length(node.interfaces)
            (interface_id != outbound_interface_id) || continue
            (typeof(inbounds[interface_id])==Message{GaussianDistribution}) || error("EqualityNode: cannot calculate outbound Message{GaussianDistribution} from inbound $(typeof(inbounds[interface_id]))")
            ensureXiWParametrization!(inbounds[interface_id].value)
            W_sum += inbounds[interface_id].value.W
            xi_sum += inbounds[interface_id].value.xi
        end
        dist_result.xi = xi_sum
        dist_result.W = W_sum
        dist_result.V = nothing
        dist_result.m = nothing
    else # 3 interfaces
        dist_1 = inbounds[first_incoming_id].value
        dist_2 = inbounds[(outbound_interface_id==3) ? 2 : 3].value
        if dist_1.m != nothing && dist_1.W != nothing && dist_2.m != nothing && dist_2.W != nothing
            dist_result.m  = equalityMRule(dist_1.m, dist_2.m, dist_1.W, dist_2.W)
            dist_result.V  = nothing
            dist_result.W  = equalityWRule(dist_1.W, dist_2.W)
            dist_result.xi = nothing
        elseif dist_1.xi != nothing && dist_1.V != nothing && dist_2.xi != nothing && dist_2.V != nothing
            dist_result.m  = nothing
            dist_result.V  = equalityVRule(dist_1.V, dist_2.V)
            dist_result.W  = nothing
            dist_result.xi = equalityXiRule(dist_1.xi, dist_2.xi)
        else
            # Use (xi,W)
            ensureXiWParametrization!(dist_1)
            ensureXiWParametrization!(dist_2)
            dist_result.m  = nothing
            dist_result.V  = nothing
            dist_result.W  = equalityWRule(dist_1.W, dist_2.W)
            dist_result.xi = equalityXiRule(dist_1.xi, dist_2.xi)
        end
    end

    return node.interfaces[outbound_interface_id].message
end

############################################
# Number and Array{Number} methods
############################################

function updateNodeMessage!(node::EqualityNode,
                            outbound_interface_id::Int,
                            outbound_message_value_type::Union(Type{Float64}, Type{Vector{Float64}}, Type{Matrix{Float64}}),
                            inbounds...)
    # Calculate an outbound message based on the inbound messages and the node function.
    # This function is not exported, and is only meant for internal use.

    # Outbound message is equal to the inbound messages if not all inbound messages are equal.
    # Otherwise, the outbound message is Message{Float64}(0.0)
    first_incoming_id = (outbound_interface_id==1) ? 2 : 1
    constraint = inbounds[first_incoming_id].value
    (typeof(constraint)==outbound_message_value_type) || error("EqualityNode: cannot calculate outbound Message{$(outbound_message_value_type)} from inbound $(typeof(constraint))")
 
    for interface_id = 1:length(node.interfaces)
        (interface_id != outbound_interface_id) || continue
        if inbounds[interface_id].value != constraint
            msg_result = getOrCreateMessage(node.interfaces[outbound_interface_id], outbound_message_value_type, size(constraint))
            msg_result.value = zero(constraint)
            return node.interfaces[outbound_interface_id].message
        end
    end
    msg_result = getOrCreateMessage(node.interfaces[outbound_interface_id], outbound_message_value_type, size(constraint))
    msg_result.value = deepcopy(constraint)

    return node.interfaces[outbound_interface_id].message
end

############################################
# InverseGammaDistribution methods
############################################

function updateNodeMessage!(node::EqualityNode,
                            outbound_interface_id::Int,
                            outbound_message_value_type::Type{InverseGammaDistribution},
                            msg_1::Union(Message{InverseGammaDistribution}, Nothing),
                            msg_2::Union(Message{InverseGammaDistribution}, Nothing),
                            msg_3::Union(Message{InverseGammaDistribution}, Nothing))
    # Calculate an outbound message based on the inbound messages and the node function.
    # This function is not exported, and is only meant for internal use.
    dist_out = getOrCreateMessage(node.interfaces[outbound_interface_id], outbound_message_value_type).value

    # Definition from Korl table 5.2
    dist_out.a = 1.0
    dist_out.b = 0.0
    for inbound_msg in [msg_1, msg_2, msg_3]
        (inbound_msg != nothing) || continue
        (typeof(inbound_msg) == Message{outbound_message_value_type}) || error("EqualityNode: cannot calculate outbound Message{$(outbound_message_value_type)} from inbound $(typeof(inbound_msg))")
        dist_out.a += inbound_msg.value.a
        dist_out.b += inbound_msg.value.b
    end

    return node.interfaces[outbound_interface_id].message
end

############################################
# GammaDistribution methods
############################################

function updateNodeMessage!(node::EqualityNode,
                            outbound_interface_id::Int,
                            outbound_message_value_type::Type{GammaDistribution},
                            msg_1::Union(Message{GammaDistribution}, Nothing),
                            msg_2::Union(Message{GammaDistribution}, Nothing),
                            msg_3::Union(Message{GammaDistribution}, Nothing))
    # Calculate an outbound message based on the inbound messages and the node function.
    # This function is not exported, and is only meant for internal use.
    dist_out = getOrCreateMessage(node.interfaces[outbound_interface_id], outbound_message_value_type).value

    # Definition from Korl table 5.2
    dist_out.a = -1.0
    dist_out.b = 0.0
    for inbound_msg in [msg_1, msg_2, msg_3]
        (inbound_msg != nothing) || continue
        (typeof(inbound_msg) == Message{outbound_message_value_type}) || error("EqualityNode: cannot calculate outbound Message{$(outbound_message_value_type)} from inbound $(typeof(inbound_msg))")
        dist_out.a += inbound_msg.value.a
        dist_out.b += inbound_msg.value.b
    end

    return node.interfaces[outbound_interface_id].message
end
