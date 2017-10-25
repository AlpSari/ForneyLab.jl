module ForneyLab

# Helpers
include("helpers.jl")
include("dependency_graph.jl")

# Other includes
import Base: show, convert, ==, mean, var, cov, *

# High level abstracts
abstract AbstractEdge # An Interface belongs to an Edge, so AbstractEdge has to be defined before Interface
abstract AbstractVariable
abstract InferenceAlgorithm

# Low-level internals
include("factor_node.jl")
include("interface.jl")
include("probability_distribution.jl")
include("edge.jl")
include("variable.jl")

# Factor nodes
include("factor_nodes/clamp.jl")
include("factor_nodes/equality.jl")
include("factor_nodes/addition.jl")
include("factor_nodes/multiplication.jl")
include("factor_nodes/gaussian.jl")
include("factor_nodes/gaussian_mean_variance.jl")
include("factor_nodes/gaussian_mean_precision.jl")
include("factor_nodes/gamma.jl")
include("factor_nodes/bernoulli.jl")
include("factor_nodes/gaussian_mixture.jl")
include("factor_nodes/sigmoid.jl")

# include("nodes/gaussian_mixture.jl")
# include("nodes/exponential.jl")
# include("nodes/gain_addition.jl")
# include("nodes/gain_equality.jl")
# include("nodes/categorical.jl")

# Factor graph
include("factor_graph.jl")

# Composite nodes
include("factor_nodes/composite.jl")

# Generic methods
include("message_passing.jl")

# Utils
include("visualization.jl")

# InferenceAlgorithms
include("algorithms/sum_product/sum_product.jl")
include("algorithms/variational_bayes/recognition_factorization.jl")
include("algorithms/variational_bayes/variational_bayes.jl")
include("algorithms/expectation_propagation/expectation_propagation.jl")

# Update rules
include("update_rules/clamp.jl")
include("update_rules/equality.jl")
include("update_rules/addition.jl")
include("update_rules/multiplication.jl")
include("update_rules/gaussian_mean_variance.jl")
include("update_rules/gaussian_mean_precision.jl")
include("update_rules/gamma.jl")
include("update_rules/bernoulli.jl")
include("update_rules/gaussian_mixture.jl")
include("update_rules/sigmoid.jl")

*(x::ProbabilityDistribution, y::ProbabilityDistribution) = prod!(x, y) # * operator for probability distributions

# include("docstrings.jl")

# Engines
include("engines/julia/julia.jl")

end # module ForneyLab
