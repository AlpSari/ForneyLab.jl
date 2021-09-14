module CviTest
using Test
using ForneyLab,Flux.Optimise
using ForneyLab: renderCVI,isProper

@testset "iBLR with GaussianPrior" begin
    @testset "Univariate" begin
        μ = randn() # mean
        S = abs(randn()) # precision
        msg_in = Message(Univariate, GaussianWeightedMeanPrecision,xi=S*μ,w=S)
        log_pnc(z) = z
        num_iterations = 100
        opts = [Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent]
        λ_init = deepcopy(naturalParams(msg_in.dist))
        for opt in opts
            if opt == ForgetDelayDescent
            optimizer = opt(1,2.,3.)
            else
            optimizer = opt()
            end
        @test isProper(standardDist(msg_in.dist,renderCVI(log_pnc,num_iterations,optimizer,λ_init,msg_in)))
        end
    end #Univariate
    @testset "Multivarite" begin
        n = rand(2:10,1)[1]
        μ = randn(n,) # mean
        S = abs.(randn(n,)).*diageye(n) # precision
        msg_in = Message(Multivariate, GaussianWeightedMeanPrecision,xi=S*μ,w=S)
        log_pnc(z) = sum(z) #needs to be scalar
        num_iterations = 100
        opts = [Descent, Momentum, Nesterov, RMSProp, ADAM, ForgetDelayDescent]
        λ_init = deepcopy(naturalParams(msg_in.dist))
        for opt in opts
            if opt == ForgetDelayDescent
            optimizer = opt(1,2.,3.)
            else
            optimizer = opt()
            end
        @test isProper(standardDist(msg_in.dist,renderCVI(log_pnc,num_iterations,optimizer,λ_init,msg_in)))
        end
    end #Multivariate
end #GaussianPrior

end #module
