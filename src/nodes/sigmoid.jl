############################################
# SigmoidNode
############################################
# Description:
#   Links a continuous, real-valued variable (X) to a binary (boolean) one (Y).
#
#     X       Y
#   -----[σ]-----
#
#   f(X,Y) = σ(X⋅Y)
#
# Interfaces:
#   1 i[:real], 2 i[:bin]
#
# Construction:
#   SigmoidNode(:normal_cdf, name="my_node")
#   The optional first argument specifies the sigmoid function.
#   Currently the only option is :normal_cdf.
#
############################################

export SigmoidNode


####################
# Sigmoid functions
####################

# Cummulative Gaussian (CDF of standard normal distribution)
Φ(x::Union{Float64, Vector{Float64}}) = 0.5*erfc(-x./sqrt(2.))


####################
# SigmoidNode
####################

type SigmoidNode <: Node
    sigmoid_func::Symbol
    id::Symbol
    interfaces::Array{Interface,1}
    i::Dict{Symbol,Interface}

    function SigmoidNode(sigmoid_func::Symbol=:normal_cdf; id=generateNodeId(SigmoidNode))
        (sigmoid_func == :normal_cdf) || error(":normal_cdf is the only supported sigmoid function at the moment")

        self = new(sigmoid_func, id, Array(Interface, 2), Dict{Symbol,Interface}())
        self.i[:real] = self.interfaces[1] = Interface(self)
        self.i[:bin] = self.interfaces[2] = Interface(self)

        addNode!(current_graph, self)

        return self
    end
end

####################
# Forward messages
####################

function sumProduct!{T<:Real}(node::SigmoidNode,
                              outbound_interface_id::Int,
                              msg_1::Message{DeltaDistribution{T}},
                              ::Void)
    # Generate Bernoulli message from incoming Delta message.
    (outbound_interface_id == 2) || error("Invalid call")
    dist_1 = msg_1.payload
    #(typeof(dist_1.m) <: Real) || error("The inbound message should be defined over the 1 dimensional real domain")
    dist_2 = ensureMessage!(node.interfaces[2], BernoulliDistribution).payload

    if node.sigmoid_func == :normal_cdf
        dist_2.p = Φ(dist_1.m)
    else
        error("Unsupported sigmoid function")
    end

    return (:sigmoid_delta_forward, node.interfaces[2].message)
end

function sumProduct!(node::SigmoidNode,
                     outbound_interface_id::Int,
                     msg_1::Message{GaussianDistribution},
                     ::Void)
    # Generate Bernoulli message from incoming Gaussian message.
    (outbound_interface_id == 2) || error("Invalid call")
    dist_1 = ensureParameters!(msg_1.payload, (:m, :V))
    (length(dist_1.m) == 1) || error("Only univariate messages are supported")
    dist_2 = ensureMessage!(node.interfaces[2], BernoulliDistribution).payload

    if node.sigmoid_func == :normal_cdf
        dist_2.p = Φ(dist_1.m / sqrt(1+dist_1.V))
    else
        error("Unsupported sigmoid function")
    end

    return (:sigmoid_gaussian_forward, node.interfaces[2].message)
end

############################################################
# Backward messages (expectation propagation)
############################################################

function ep!{T<:Bool}(
                    node::SigmoidNode,
                    outbound_interface_id::Int,
                    msg_cavity::Message{GaussianDistribution},
                    msg_bin::Message{DeltaDistribution{T}})
    # Convert incoming DeltaDistribution to BernoulliDistribution
    (outbound_interface_id == 1) || error("Invalid call")
    ep!(node, 1, msg_cavity, Message(BernoulliDistribution(msg_bin.payload.m)))
end

function ep!(
            node::SigmoidNode,
            outbound_interface_id::Int,
            msg_cavity::Message{GaussianDistribution},
            msg_bin::Message{BernoulliDistribution})
    # Calculate approximate (Gaussian) message towards i[:real]
    # The approximate message is an 'expectation' under the context (cavity distribution) encoded by incoming message msg_1.
    # Propagating the resulting approximate msg through the factor graph results in the expectation propagation (EP) algorithm.
    # Approximation procedure:
    #  1. Calculate exact (non-Gaussian) message towards i[:real].
    #  2. Combine exact outbound msg on i[:real] with exact inbound msg (cavity distribution) to find exact marginal.
    #  3. Approximate the exact (non-Gaussian) marginal with a Gaussian one using moment matching.
    #  4. Calculate back the Gaussian outbound msg on i[:real] that yields this approximate Gaussian marginal.
    # IMPORTANT NOTES:
    #  - This calculation results in an implicit cycle in the factor graph since the outbound message depends on the inbound message (cavity dist.).
    #  - The outbound message is not guaranteed to be proper iff 0 < msg_bin.payload.p < 1: variance/precision parameters might be negative.
    (outbound_interface_id == 1) || error("Invalid call")
    (node.sigmoid_func == :normal_cdf) || error("Unsupported sigmoid function")
    isProper(msg_bin.payload) || error("ep!: Incoming Bernoulli distribution should be proper")
    isProper(msg_cavity.payload) || error("ep!: Cavity distribution is improper")

    # Shordhand notations
    p = msg_bin.payload.p
    dist_cavity = ensureParameters!(msg_cavity.payload, (:m, :V))
    μ = dist_cavity.m; σ2 = dist_cavity.V

    # Calculate first and second moment (mp_1, mp_2) of the 'true' marginal p(x) on edge connected to i[:real]
    # p(x) = f(x) / Z
    # f(x) = (1-p)*N(x|μ,σ2) + (2p-1)*Φ(x)*N(x|μ,σ2)
    #      = (1-p)*N(x|μ,σ2) + (2p-1)*Φ(z)*(Φ(x)*N(x|μ,σ2)/Φ(z))
    #      = (1-p)*N(x|μ,σ2) + (2p-1)*Φ(z)*g(x)
    # See paper for detailed derivation

    z = μ / sqrt(1 + σ2)
    N = exp(-0.5*z^2)./sqrt(2*pi) # 𝓝(z)

    # Moments of g(x)
    mg_1 = Φ(z)*μ + σ2*N / sqrt(1+σ2)  # First moment of g
    mg_2 = 2*μ*mg_1 + Φ(z)*(σ2 - μ^2) - σ2^2*z*N / (1+σ2)  # Second moment of g

    # Moments of f(x) (exact marginal)
    Z = 1 - p + (2*p-1)*Φ(z)
    mp_1 = ((1-p)*μ + (2*p-1)*mg_1) / Z
    mp_2 = ((1-p)*(μ^2+σ2) + (2*p-1)*mg_2) / Z

    # Save Gaussian marginal with identical first and second moments (moment matching approximation)
    marginal = ensureMarginal!(node.interfaces[1].edge, GaussianDistribution)
    marginal.W = clamp(1/(mp_2 - mp_1^2), tiny, huge) # This quantity is guaranteed to be positive
    marginal.xi = marginal.W * mp_1
    marginal.m = marginal.V = NaN

    # Calculate the approximate message towards i[:real]
    dist_backward = ensureMessage!(node.interfaces[1], GaussianDistribution).payload
    ensureParameters!(dist_cavity, (:xi, :W))
    dist_backward.W = marginal.W - dist_cavity.W # This can be < 0, yielding an improper Gaussian msg
    if dist_backward.W < 0
        dist_backward.W = clamp(dist_backward.W, -1*huge, -1*tiny)
    else
        dist_backward.W = clamp(dist_backward.W, tiny, huge)
    end
    dist_backward.xi = marginal.xi - dist_cavity.xi
    dist_backward.m = dist_backward.V = NaN

    return (:sigmoid_backward_gaussian_expectation, node.interfaces[1].message)
end
