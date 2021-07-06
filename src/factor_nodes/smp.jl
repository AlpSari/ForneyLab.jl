export SMP, Smp

"""
Description:

    SMP node allows VMP to be applied to nonconjugate factor pairs.

    Maps a variable through

    f(out,in1) = δ(out - g(in1))

Interfaces:

    1. out
    2. in1

Construction:


"""

mutable struct SMP <: DeltaFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Vector function that expresses the output as a function of the inputs
    opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent, Vector{Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent}}}
    num_iterations::Union{Int,Vector{Int}}
    num_samples::Union{Int,Vector{Int}}
    back_message_types::ProbabilityDistribution

    function SMP(id::Symbol, g::Function,
                    opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent, Vector{Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent}}},
                    num_iterations::Union{Int,Vector{Int}}, num_samples::Union{Int,Vector{Int}}, back_message_types::ProbabilityDistribution,
                    out::Variable, args::Vararg)
        @ensureVariables(out)
        n_args = length(args)
        for i=1:n_args
            @ensureVariables(args[i])
        end
        self = new(id, Array{Interface}(undef, n_args+1), Dict{Int,Interface}(), g, opt, num_iterations, num_samples, back_message_types)
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        for k = 1:n_args
            self.i[:in*k] = self.interfaces[k+1] = associate!(Interface(self), args[k])
        end

        return self
    end

end

slug(::Type{SMP}) = "smp"

function Smp(out::Variable, args::Vararg; g::Function, opt::Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent, Vector{Union{Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent}}},
                num_samples::Union{Int,Vector{Int}}, num_iterations::Union{Int,Vector{Int}},
                back_message_types=ProbabilityDistribution(Univariate,GaussianMeanPrecision,m=0,w=1), id=ForneyLab.generateId(CVI))
    SMP(id, g, opt, num_iterations, num_samples, back_message_types, out, args...)
end
