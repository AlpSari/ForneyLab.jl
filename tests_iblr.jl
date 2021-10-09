module iBLRTest
using Test
using ForneyLab
using LinearAlgebra:norm
using ForneyLab: DefaultOptim, ConvergenceStatsFE,ConvergenceParamsFE,ConvergenceParamsMC
using ForneyLab: check_all_keys,adaptStepSize,ΔFE_check!,oneWindowSimulation_MCMC,Pareto_k_fit,renderCVI
using ForneyLab: ParamStr2

@testset "Types" begin
    @testset "DefaultOptim" begin
        @test isa(DefaultOptim(),DefaultOptim)
        @test isa(DefaultOptim(invalidfield="xyz"),DefaultOptim)
        @test isa(DefaultOptim(Dict()),DefaultOptim)
        @test DefaultOptim(100).max_iterations == 100
        @test DefaultOptim(Dict(:max_iterations=>200)).max_iterations == 200
    end
    @testset "ConvergenceStatsFE" begin
        @test isa(ConvergenceStatsFE(),ConvergenceStatsFE)
        @test isa(ConvergenceStatsFE(invalidfield="xyz"),ConvergenceStatsFE)
        @test isa(ConvergenceStatsFE(Dict()),ConvergenceStatsFE)
        @test ConvergenceStatsFE().F_prev == Inf
        @test ConvergenceStatsFE().F_best == Inf
        @test ConvergenceStatsFE(ΔF_vect=[1.0,2.0]).ΔF_vect  == [1.0,2.0]
    end
    @testset "ConvergenceParamsFE" begin
        @test isa(ConvergenceParamsFE(),ConvergenceParamsFE)
        @test isa(ConvergenceParamsFE(invalidfield="xyz"),ConvergenceParamsFE)
        @test isa(ConvergenceParamsFE(Dict()),ConvergenceParamsFE)
        @test ConvergenceParamsFE(max_iterations=1e7).max_iterations == 1e7
        @test ConvergenceParamsFE(max_iterations=100.0).max_iterations == 100
        @test isa(ConvergenceParamsFE().stats,ConvergenceStatsFE)
    end
    @testset "ConvergenceParamsMC" begin
        @test isa(ConvergenceParamsMC(),ConvergenceParamsMC)
        @test isa(ConvergenceParamsMC(invalidfield="xyz"),ConvergenceParamsMC)
        @test isa(ConvergenceParamsMC(Dict()),ConvergenceParamsMC)
        @test ConvergenceParamsMC(max_iterations=1e7).max_iterations == 1e7
        @test ConvergenceParamsMC(max_iterations=100.0).max_iterations == 100
    end
    @testset "iBLR" begin
        @test isa(ParamStr2(),ParamStr2)
        @test isa(ParamStr2(invalidfield="xyz"),ParamStr2)
        @test isa(ParamStr2(Dict()),ParamStr2)
        @test ParamStr2().is_initialized == false
        @test ParamStr2().auto_init_stepsize == true
        @test ParamStr2().eta == 0.0
        @test isa(ParamStr2().convergence_optimizer,ConvergenceParamsFE)
        @test isa(ParamStr2(convergence_algo = "free_energy").convergence_optimizer,ConvergenceParamsFE)
        @test isa(ParamStr2(convergence_algo = "MonteCarlo").convergence_optimizer,ConvergenceParamsMC)
        @test isa(ParamStr2(convergence_algo = "none").convergence_optimizer,DefaultOptim)
    end
end # begin
@testset "check_all_keys" begin
    struct TestStruct
        x::Float64
        y::Int64
        z::String
    end
    @test check_all_keys(Dict(),TestStruct) == false
    @test check_all_keys(Dict(:x=>4),TestStruct) == false
    @test check_all_keys(Dict(:x=>3,:y=>2),TestStruct) == false
    @test check_all_keys(Dict(:x1=>3,:y2=>2,:z3=>"a"),TestStruct) == false
    @test check_all_keys(Dict(:x=>3,:y=>2,:z=>"a"),TestStruct) == true
    @test check_all_keys(Dict(:x=>3,:y=>2,:z=>"a",:t=>3),TestStruct) == true
end # begin

@testset "CommonMainFunctions" begin
    @testset "adaptStepSize" begin
        opt = ParamStr2(current_stepsize=1.0,stepsize_update="none");adaptStepSize(opt);
        @test opt.current_stepsize == 1.0
        opt = ParamStr2(current_stepsize=1.0,iteration_counter=2,stepsize_update="decay");adaptStepSize(opt);
        @test opt.current_stepsize != 1.0
        opt = ParamStr2(current_stepsize=1.0,stepsize_update="adaptive");adaptStepSize(opt);
        @test_skip opt.current_stepsize == 1.0 # not implemented
    end
    @testset "ΔFE_check!" begin
        stats=ConvergenceStatsFE(ΔF_vect=ones(1000,));F_now= 1.0;idx_now=3;tolerance=0.5;
        @test ΔFE_check!(stats,F_now,idx_now,tolerance) in [true,false]
        stats=ConvergenceStatsFE()
        @test ΔFE_check!(stats,F_now,idx_now,tolerance) in [true,false]
    end
end # begin



@testset "Univariate Gaussian" begin
    λ_bc=[1.0,2.0] #Bc
    msg_in = Message(Univariate, GaussianWeightedMeanPrecision,xi=λ_bc[2]*λ_bc[1],w=λ_bc[2])
    λ_natural = naturalParams(msg_in.dist)
    logp_nc(z) = norm(z)

    @testset "oneWindowSimulation_MCMC" begin
        J=2;Window=10;opt = ParamStr2(max_iterations = 50,convergence_algo="MonteCarlo");
        is_first_sim = true
        last_params,opt_matrix,stats_dict = oneWindowSimulation_MCMC(J,Window,opt,λ_bc,logp_nc,is_first_sim,msg_in)
        @test isa(last_params,Vector)
        @test isa(opt_matrix,Vector)
        @test isa(stats_dict,Dict)
    end
    @testset "Pareto_k_fit" begin
        S=100
        k = Pareto_k_fit(logp_nc,msg_in,λ_bc,S)
        @test isa(k,Float64)
    end
    @testset "renderCVI" begin
        num_iterations = 100
        for str in ["none","free_energy","MonteCarlo"]
            opt = ParamStr2(max_iterations=num_iterations,convergence_algo = str)
            λ_posterior = renderCVI(logp_nc,num_iterations,opt,λ_natural,msg_in)
            @test isa(λ_posterior,Vector)
        end
    end
end

@testset "Multivariate Gaussian" begin
    d = 3
    λ_bc=[ones(d,),diageye(d),diageye(d)]
    msg_in = Message(Multivariate, GaussianWeightedMeanPrecision,xi=λ_bc[2]*λ_bc[1],w=λ_bc[2])
    λ_natural = naturalParams(msg_in.dist)
    logp_nc(z) = norm(z)
    @testset "oneWindowSimulation_MCMC" begin
        J=2;Window=10;opt = ParamStr2(max_iterations = 50,convergence_algo="MonteCarlo");
        is_first_sim = true
        last_params,opt_matrix,stats_dict = oneWindowSimulation_MCMC(J,Window,opt,λ_bc,logp_nc,is_first_sim,msg_in)
        @test isa(last_params,Vector)
        @test isa(opt_matrix,Vector)
        @test isa(stats_dict,Dict)
    end
    @testset "Pareto_k_fit" begin
        S=100
        k = Pareto_k_fit(logp_nc,msg_in,λ_bc,S)
        @test isa(k,Float64)
    end
    @testset "renderCVI" begin
        num_iterations = 100
        for str in ["none","free_energy","MonteCarlo"]
            opt = ParamStr2(max_iterations=num_iterations,convergence_algo = str)
            λ_posterior = renderCVI(logp_nc,num_iterations,opt,λ_natural,msg_in)
            @test isa(λ_posterior,Vector)
        end
    end
end


end  # module iBLRTest
