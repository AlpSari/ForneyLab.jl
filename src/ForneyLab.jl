module ForneyLab

using Optim
using YAML
using LaTeXStrings

export ProbabilityDistribution
export sumProduct!, expectation!, vmp!
export vague, self, ==
export setVerbosity

# Export algorithm modules
export SumProduct
export VMP

# Verbosity
setVerbosity(is_verbose=true) = global verbose = is_verbose
printVerbose(msg) = if verbose println(msg) end

# ForneyLab helpers
include("helpers.jl")

# Other includes
import Base.show, Base.convert

# Top-level abstracts
abstract AbstractEdge # An Interface belongs to an Edge, but Interface is defined before Edge. Because you can not belong to something undefined, Edge will inherit from AbstractEdge, solving this problem.
abstract ProbabilityDistribution # ProbabilityDistribution can be carried by a Message or an Edge (as marginal)

# Node
include("node.jl")

# Message type
include("message.jl")

# Distributions
include("distributions/delta.jl")
include("distributions/bernoulli.jl")
include("distributions/beta.jl")
include("distributions/gamma.jl")
include("distributions/gaussian.jl")
include("distributions/inverse_gamma.jl")
include("distributions/normal_gamma.jl")
include("distributions/students_t.jl")

# Basic ForneyLab building blocks and methods
include("interface.jl")
include("edge.jl")
include("schedule.jl")

# Nodes
include("nodes/addition.jl")
include("nodes/terminal.jl")
include("nodes/equality.jl")
include("nodes/fixed_gain.jl")
include("nodes/gaussian.jl")
include("nodes/exponential.jl")
include("nodes/gain_addition.jl")
include("nodes/gain_equality.jl")
include("nodes/sigmoid.jl")

# Graph, wraps and algorithm
include("factor_graph.jl")
include("wrap.jl")
include("algorithm.jl")

# Composite nodes
include("nodes/composite.jl")

# Methods for calculating marginals
include("distributions/calculate_marginal.jl")

# Generic methods
include("message_passing.jl")
include("step.jl")

# Utils
include("visualization.jl")

# Algorithms
include("algorithms/sum_product/sum_product.jl")
include("algorithms/vmp/vmp.jl")

# Functions for message post-processing
vague(dist::ProbabilityDistribution) = vague(typeof(dist))

function __init__()
    # Run-time initialization

    # Module-global variable for verbosity setting
    global verbose = false

    # Create an empty FactorGraph
    # Module-global variable current_graph keeps track of currently active FactorGraph
    global current_graph = FactorGraph(Dict{Symbol, Node}(),
                                        Dict{Symbol, Edge}(),
                                        Dict{Symbol, AbstractWrap}(),
                                        Dict{DataType, Int}(),
                                        false,
                                        Dict{TerminalNode, Vector}(),
                                        Dict{Union(Edge,Interface), Vector}())

    # Module-global variable to keep track of currently active Algorithm
    global current_algorithm = nothing
end

end # module ForneyLab
