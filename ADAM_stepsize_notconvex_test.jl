using Random,Flux.Optimise
"""
This code shows when updating x[t] using ADAM optimizer using gradient
g=[ϵ-x[t]], the update might not be convex in the sense that it would not be
equivalently written as:

x[t+1] = x[t] + β_t*(ϵ-x[t]) = (1-β_t)x[t] = β_t*ϵ where 0<β_t<1

β_t can be still calculated and used for calculating the additional term for
IBL update
"""

Random.seed!(2)
Opt =ADAM(0.3)
function get_beta(x_new,x_old,g)
    beta= ((x_new-x_old)/(g))[1]
end
x = [5.]
N = 10
beta_vect = Vector{Float64}(undef,N)
for i = 1:N
    x_old = deepcopy(x)
    ϵ = i+randn()
    g = [ϵ-x_old[1]]
    g_old = deepcopy(g)
    update!(Opt,x,-g) # -g since optimizer goes in descent direction
    beta = get_beta(x,x_old,g_old)
    beta_vect[i]=beta
    println()
    println("x_old=$x_old,x=$x,ϵ=$(4*ϵ),g_old=$g_old,beta=$beta")

end
println("min=$(minimum(beta_vect)),max=$(maximum(beta_vect))")
