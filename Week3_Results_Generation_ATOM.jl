using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Random,LinearAlgebra, Flux.Optimise, Plots, ForneyLab
using ForneyLab:iBLR
using JLD2 # Used to save data
using TimerOutputs
using Statistics:median
using ColorSchemes
gr()
#
mutable struct Week3Data
    method::String
    stepsize::Float64
    num_samples::Float64
    num_iterations::Float64
    elapsed_time::Float64
    data::Vector
    true_value::Float64
end
# LOAD ALL DATA
all_data=[]
methods_str = ["iBLR","Descent","ADAM"]
for method in methods_str,i=1:77
    if i>66 && method =="ADAM"
        continue
    end
    datafile_name = "W3_$(method)_results"
    data_exp = load_object("$datafile_name/$(method)_$(i).jld2")
    push!(all_data,data_exp)
end
#
num_iterations_array = [10^(i) for i in 1:6.]
stepsize_array = [10^(-i) for i in 0:10.]
stepsize_array[6]=0.00001
# FIX METHOD PLOTS

function plot_fixed_method(method_name::String)
    x  = exp10.(1:6)
    names = ["$stepsize" for stepsize in stepsize_array]
    exps_all = [data for data in all_data if data.method == method_name]
    fixed_step_size_curves = []
    for step_size in stepsize_array
        # Curve: Fixed stepsize , varying num_iterations
        one_curve = [x.data for x in exps_all if isapprox(x.stepsize,step_size)&& x.num_iterations != 10^7]
        # Just to make sure they are ordered with ascending order
        permutation_indexes = sortperm([x.num_iterations for x in exps_all if x.method ==method_name && isapprox(x.stepsize,step_size)&& x.num_iterations != 10^7])
        one_curve = one_curve[permutation_indexes] #sorted for increasing num_iterations
        # Calculate data as the median of the absolute values of 10 simulations
        # Abs is used since there is cyclical behaviour between -120,120
        one_curve_medians = median.([abs.(x) for x in one_curve])
        # Push one curve data to a container
        push!(fixed_step_size_curves,one_curve_medians)
    end
    # Plot the ones only if they converge to the real true value at least once
    plot()
    for i in 1:length(names)
        one_curve =  fixed_step_size_curves[i]
        is_converged = any([isapprox(120,x,atol=20) for x in one_curve])
        if is_converged
                c = get(ColorSchemes.rainbow,i./length(names))
                plot!(x,fixed_step_size_curves[i,:][1],linewidth=2,label=names[i],color=c,linestyle=:dash)
        end
    end
    title!("$method_name method")
    xlabel!("# of iterations, True value = 120")
    ylabel!("Converged median of 10 simulations")
    plot!(ylim=(0,200),yticks = 0:20:200)
    display(plot!(legendtitle = "stepsize",legend=:outertopright,xaxis=:log,xticks = x))
end


function plot_fixed_stepsize(stepsize::Float64)
    x  = exp10.(1:6)
    method_names = ["iBLR","Descent","ADAM"]
    method_names_legend = ["iBLR","CVI_Descent","CVI_ADAM"]
    exps_all = [data for data in all_data if isapprox(data.stepsize,stepsize)]
    fixed_method_curves = []

    for method_name in method_names
        one_curve = [x.data for x in exps_all if x.method == method_name && x.num_iterations != 10^7]
        # Just to make sure they are ordered with ascending order
        permutation_indexes = sortperm([x.num_iterations for x in exps_all if x.method ==method_name && isapprox(x.stepsize,stepsize) && x.num_iterations != 10^7])
        one_curve = one_curve[permutation_indexes] #sorted for increasing num_iterations
        # Calculate data as the median of the absolute values of 10 simulations
        # Abs is used since there is cyclical behaviour between -120,120
        one_curve_medians = median.([abs.(x) for x in one_curve])
        push!(fixed_method_curves,one_curve_medians)
    end
    #
    plot()
    for i in 1:length(method_names)
        if i==1
            line_style = :solid #iBLR
        elseif i==2
            line_style = :dot #Descent
        else
            line_style = :dash # CVI
        end
        one_curve =  fixed_method_curves[i]
        is_converged = any([isapprox(120,x,atol=20) for x in one_curve])
        c = get(ColorSchemes.rainbow,i./length(method_names))
        plot!(x,fixed_method_curves[i,:][1],linewidth=2,label=method_names_legend[i],color=c,linestyle=line_style)
        # if is_converged
        #         c = get(ColorSchemes.rainbow,i./length(method_names))
        #         plot!(x,fixed_method_curves[i,:][1],linewidth=2,label=method_names_legend[i],color=c,linestyle=:dash)
        # end
    end
    title!("Step size  = $stepsize")
    xlabel!("# of iterations, True value = 120")
    ylabel!("Converged median of 10 simulations")
    plot!(ylim=(0,200),yticks = 0:20:200)
    display(plot!(legendtitle = "methods",legend=:outertopright,xaxis=:log,xticks = exp10.(1:6)))
end



function plot_fixed_numiter(numiter::Float64)
    # One curve: showing a method
    # X axis: Stepsize
    # Fixed for entire plot : number of iterations
    x  = exp10.(-10:1:0) # Should be in ascending order
    method_names = ["iBLR","Descent","ADAM"]
    method_names_legend = ["iBLR","CVI_Descent","CVI_ADAM"]
    exps_all = [data for data in all_data if isapprox(data.num_iterations,numiter)]
    fixed_method_curves = []

    for method_name in method_names
        one_curve = [x.data for x in exps_all if x.method == method_name && x.num_iterations != 10^7]
        # Just to make sure they are ordered with ascending order
        permutation_indexes = sortperm([x.stepsize for x in exps_all if x.method ==method_name && isapprox(x.num_iterations,numiter) && x.num_iterations != 10^7])
        one_curve = one_curve[permutation_indexes] #sorted for increasing num_iterations
        # Calculate data as the median of the absolute values of 10 simulations
        # Abs is used since there is cyclical behaviour between -120,120
        one_curve_medians = median.([abs.(x) for x in one_curve])
        push!(fixed_method_curves,one_curve_medians)
    end

    #fixed_method_curves has 3x11 elements
    #3 is for iBLR,CVI and ADAM
    #11 is for data for varying stepsizes in ascending order
    plot()
    for i in 1:length(method_names)
        if i==1
            line_style = :solid #iBLR
        elseif i==2
            line_style = :dot #Descent
        else
            line_style = :dash # CVI
        end
        one_curve =  fixed_method_curves[i]
        is_converged = any([isapprox(120,x,atol=20) for x in one_curve])
        c = get(ColorSchemes.rainbow,i./length(method_names))
        plot!(x,fixed_method_curves[i,:][1],linewidth=2,label=method_names_legend[i],color=c,linestyle=line_style)
        # if is_converged
        #         c = get(ColorSchemes.rainbow,i./length(method_names))
        #         plot!(x,fixed_method_curves[i,:][1],linewidth=2,label=method_names_legend[i],color=c,linestyle=:dash)
        # end
    end
    title!("Number of iterations  = $numiter")
    xlabel!("Stepsize, True value = 120")
    ylabel!("Converged median of 10 simulations")
    plot!(ylim=(0,200),yticks = 0:20:200)
    display(plot!(legendtitle = "methods",legend=:outertopright,xaxis=:log,xticks = x))
end
plot_fixed_numiter(1e6)
plot_fixed_method("iBLR")
plot_fixed_stepsize(1e-4)
