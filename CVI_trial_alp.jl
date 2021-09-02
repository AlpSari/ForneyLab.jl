# using Pkg
# Pkg.activate("./")
# Pkg.instantiate()
using LinearAlgebra, Flux.Optimise, Plots, ForneyLab

# Conjugate Model Forneylab

# graph = FactorGraph()
#
# @RV x ~ GaussianMeanVariance(2.,1.)
# @RV w ~ Gamma(2.,3.5)
# @RV y ~ GaussianMeanPrecision(x,w)
#
# placeholder(y, :y)
# pfz = PosteriorFactorization(x, w, ids=[:X, :W])
# algo = messagePassingAlgorithm(free_energy=true)
# source_code = algorithmSourceCode(algo, free_energy=true)
# eval(Meta.parse(source_code));
# #println(source_code)
#
# # Execute algorithm
# n_its = 5
# marginals = Dict()
# F = zeros(n_its)
# data = Dict(:y => 11.4)
#
# marginals[:x] = vague(GaussianMeanVariance)
# marginals[:w] = vague(Gamma)
#
# for i = 1:n_its
#     stepX!(data, marginals)
#     stepW!(data, marginals)
#
#     F[i] = freeEnergy(data, marginals)
# end
# println(F)
# println(marginals)


# CVI Conjugate Model

# CVI
graph = FactorGraph()

f(x) = x

@RV x ~ GaussianMeanVariance(2.,1.)
@RV x_ ~ Cvi(x,g=f,opt=Descent(0.1),num_samples=1000,num_iterations=100)
@RV w ~ Gamma(2.,3.5)
@RV y ~ GaussianMeanPrecision(x_,w)

placeholder(y, :y)
;
pfz = PosteriorFactorization(x, w, ids=[:X, :W])
algo = messagePassingAlgorithm(free_energy=true)
source_code = algorithmSourceCode(algo, free_energy=true)
eval(Meta.parse(source_code));
#println(source_code)
# Execute algorithm
n_its = 15
marginals = Dict()
F = zeros(n_its)
data = Dict(:y => 11.4)

marginals[:x] = vague(GaussianMeanVariance)
marginals[:w] = vague(Gamma)

for i = 1:n_its
    stepX!(data, marginals)
    stepW!(data, marginals)

    F[i] = freeEnergy(data, marginals)
end
println(marginals[:x])
