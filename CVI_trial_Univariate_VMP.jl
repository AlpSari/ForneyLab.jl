using Random,LinearAlgebra, Flux.Optimise, Plots, ForneyLab

# VMP Without CVI NODE
# Generate toy data set
Random.seed!(1234);
n = 5
m_data = 3.0
w_data = 4.0
y_data = sqrt(1/w_data)*randn(n) .+ m_data;
g = FactorGraph()

# Priors
@RV m ~ GaussianMeanVariance(0.0, 1.)
@RV w ~ Gamma(0.01, 0.01)

# Observarion model
y = Vector{Variable}(undef, n)
for i = 1:n
    @RV y[i] ~ GaussianMeanPrecision(m, w)
    placeholder(y[i], :y, index=i)
end

# Specify posterior factorization
q = PosteriorFactorization(m, w, ids=[:M, :W])
# Build the variational update algorithms for each posterior factor
algo = messagePassingAlgorithm(free_energy=true)

# Generate source code for the algorithms
source_code = algorithmSourceCode(algo, free_energy=true)

# And inspect the algorithm code
#println(source_code)
eval(Meta.parse(source_code));
data = Dict(:y => y_data)

# Initial posterior factors
marginals = Dict(:m => vague(GaussianMeanVariance),
                 :w => vague(Gamma))

n_its = 2*n
F = Vector{Float64}(undef, n_its) # Initialize vector for storing Free energy
m_est = Vector{Float64}(undef, n_its)
w_est = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepM!(data, marginals)
    stepW!(data, marginals)

    # Store free energy
    F[i] = freeEnergy(data, marginals)
    println(naturalParams(marginals[:m])[1:2])
    println(mean(marginals[:m]),mean(marginals[:w]))
end

mean(marginals[:m])

# CVI Implementation
f(x) = x
# Priors
graph2 = FactorGraph()
@RV m ~ GaussianMeanVariance(0.0, 100.0)
@RV m_bar ~ Cvi(m,g=f,opt=Descent(3),num_samples=1000,num_iterations=100)
@RV w ~ Gamma(0.01, 0.01)

# Observarion model
y = Vector{Variable}(undef, n)
for i = 1:n
    @RV y[i] ~ GaussianMeanPrecision(m_bar, w)
    placeholder(y[i], :y, index=i)
end

# Specify posterior factorization
q = PosteriorFactorization(m, w, ids=[:M, :W])
# Build the variational update algorithms for each posterior factor
algo = messagePassingAlgorithm(free_energy=true)

# Generate source code for the algorithms
source_code = algorithmSourceCode(algo, free_energy=true)

# And inspect the algorithm code
#println(source_code)
eval(Meta.parse(source_code));


data = Dict(:y => y_data)

# Initial posterior factors
marginals2 = Dict(:m => vague(GaussianMeanVariance),
                 :w => vague(Gamma))

n_its = 2*n
F = Vector{Float64}(undef, n_its) # Initialize vector for storing Free energy
m_est = Vector{Float64}(undef, n_its)
w_est = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepM!(data, marginals2)
    stepW!(data, marginals2)

    # Store free energy
    F[i] = freeEnergy(data, marginals2)
end

println(mean(marginals[:m]))
println(mean(marginals2[:m]))
