import Base.show
export VariationalBayes

include("subgraph.jl")
include("recognition_distribution.jl")
include("recognition_factorization.jl")
include("scheduler.jl")
include("message_passing.jl")

"""
Variational message passing algorithm.

Usage:

    VariationalBayes(recognition_distribution_types::Dict, graph::Graph; n_iterations, message_types)
"""
type VariationalBayes <: InferenceAlgorithm
    graph::FactorGraph
    execute::Function
    factorization::RecognitionFactorization
    recognition_distributions::Dict{Tuple{Node,Subgraph},RecognitionDistribution}
    n_iterations::Int64
end

function show(io::IO, algo::VariationalBayes)
    println("VariationalBayes inference algorithm")
    println("    number of factors: $(length(algo.factorization.factors))")
    println("    number of iterations: $(algo.n_iterations)")
end

############################################
# VariationalBayes algorithm constructors
############################################

function VariationalBayes(  recognition_distribution_types::Dict,
                            graph::FactorGraph=currentGraph();
                            n_iterations::Int64=50,
                            message_types::Dict{Interface,DataType}=Dict{Interface,DataType}())

    # Generates a VariationalBayes algorithm that propagates messages to all write buffers and wraps.

    factorization = factorize(recognition_distribution_types, graph)
    generateVariationalBayesSchedule!(factorization, graph) # Generate and store internal and external schedules on factorization subgraphs
    recognition_distributions = initializeVagueRecognitionDistributions(factorization, recognition_distribution_types) # Initialize vague recognition distributions

    function exec(algorithm)
        resetRecognitionDistributions!(algorithm.recognition_distributions) # Reset recognition distributions before next step
        for iteration = 1:algorithm.n_iterations
            execute(algorithm.factorization, algorithm.recognition_distributions) # For all subgraphs, execute internal and external schedules
        end
    end

    algo = VariationalBayes(graph, exec, factorization, recognition_distributions, n_iterations)
    inferDistributionTypes!(algo, message_types)

    return algo
end


############################################
# Type inference and preparation
############################################

function inferDistributionTypes!(algo::VariationalBayes, message_types::Dict{Interface,DataType})
    # Infer the payload types for all messages in the internal schedules

    for factor in algo.factorization.factors
        # Fill schedule_entry.inbound_types and schedule_entry.outbound_type
        schedule = factor.internal_schedule
        schedule_entries = Dict{Interface, ScheduleEntry}()

        for entry in schedule
            collectInboundTypes!(entry, schedule_entries, algo) # VariationalBayes algorithm specific collection of inbound types
            outbound_interface = entry.node.interfaces[entry.outbound_interface_id]
            if outbound_interface in keys(message_types)
                setOutboundType!(entry, message_types[outbound_interface])
            end
            inferOutboundType!(entry) # If the outbound type is fixed, this will check if there is a suitable rule available
            schedule_entries[outbound_interface] = entry # Assign schedule entry to lookup dictionary
        end
    end

    return algo
end

function collectInboundTypes!(entry::ScheduleEntry, schedule_entries::Dict{Interface, ScheduleEntry}, algo::VariationalBayes)
    entry.inbound_types = []
    outbound_interface = entry.node.interfaces[entry.outbound_interface_id]

    # Collect references to all required inbound messages for executing message computation rule
    for (id, interface) in enumerate(entry.node.interfaces)
        # Should we require the inbound message or marginal?
        if id == entry.outbound_interface_id
            push!(entry.inbound_types, Void)
        elseif is(algo.factorization.edge_to_subgraph[interface.edge], algo.factorization.edge_to_subgraph[outbound_interface.edge]) && !is(interface, outbound_interface)
            # Both edges in same subgraph, require message
            push!(entry.inbound_types, Message{schedule_entries[interface.partner].outbound_type})
        else
            # A subgraph border is crossed, require marginal
            # The factor is the set of internal edges that are in the same subgraph
            sg = algo.factorization.edge_to_subgraph[interface.edge]
            push!(entry.inbound_types, typeof(algo.recognition_distributions[(entry.node, sg)].distribution))
        end
    end

    return entry
end

function prepare!(algo::VariationalBayes)
    for factor in algo.factorization.factors
        schedule = factor.internal_schedule

        # Populate the subgraph with vague messages of the correct types
        for entry in schedule
            ensureMessage!(entry.node.interfaces[entry.outbound_interface_id], entry.outbound_type)
        end

        # Compile the schedule (define schedule_entry.execute)
        compile!(schedule, algo)
    end

    return algo.graph.prepared_algorithm = algo
end

function compile!(entry::ScheduleEntry, ::Type{Val{symbol(variationalRule!)}}, algo::VariationalBayes)
    # Generate entry.execute for schedule entry with vmp update rule

    # Collect references to all required inbound messages for executing message computation rule
    node = entry.node
    outbound_interface_id = entry.outbound_interface_id
    outbound_interface = node.interfaces[outbound_interface_id]

    inbound_rule_arguments = []
    # Add inbound messages to inbound_rule_arguments
    for (id, interface) in enumerate(entry.node.interfaces)
        # Should we require the inbound message or marginal?
        if id == entry.outbound_interface_id
            # Require marginal because it is available (not used for vmp update)
            sg = algo.factorization.edge_to_subgraph[interface.edge]
            push!(inbound_rule_arguments, algo.recognition_distributions[(node, sg)].distribution)
        elseif is(algo.factorization.edge_to_subgraph[interface.edge], algo.factorization.edge_to_subgraph[outbound_interface.edge])
            # Both edges in same subgraph, require message
            push!(inbound_rule_arguments, interface.partner.message)
        else
            # A subgraph border is crossed, require marginal
            # The factor is the set of internal edges that are in the same subgraph
            sg = algo.factorization.edge_to_subgraph[interface.edge]
            push!(inbound_rule_arguments, algo.recognition_distributions[(node, sg)].distribution)
        end
    end

    return buildExecute!(entry, inbound_rule_arguments)
end
