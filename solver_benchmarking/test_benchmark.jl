# Adds packages that don't require special setup
using Pkg
#Pkg.develop(path = joinpath(@__DIR__, "McCormick.jl-master"))
Pkg.develop(path = joinpath(@__DIR__, "MINLPLib.jl"))
Pkg.develop(path = joinpath(@__DIR__, "EAGO.jl-SIPextension"))
Pkg.add("McCormick")
Pkg.add("JSON"); Pkg.add("DataFrames"); Pkg.add("CSV")
Pkg.add("JuMP"); Pkg.add("IntervalArithmetic"); Pkg.add("LaTeXStrings")
Pkg.add("Plots"); Pkg.add("StatsBase"); Pkg.add("BenchmarkProfiles")

# Loads solver_benchmarking module
include(joinpath(@__DIR__, "solver_benchmarking.jl"))

# Loads relevent modules
using JuMP, MINLPLib, SCIP, EAGO, IntervalArithmetic, GAMS
using CSV, DataFrames, LaTeXStrings, Plots, StatsBase, BenchmarkProfiles

function baron_factory()
    m = GAMS.Optimizer(GAMS.GAMSWorkspace("C:\\GAMS\\37"))

    m.gams_options["nlp"] = "BARON"
    m.gams_options["optca"] = 1E-4
    m.gams_options["optcr"] = 1E-4

    return m
end

function scip_factory()
    m = SCIP.Optimizer()
    MOI.set(m, SCIP.Param("limits/gap"), 1E-4)
    MOI.set(m, SCIP.Param("limits/absgap"), 1E-4)
    m
end

function eago_factory()
    m = EAGO.Optimizer()
    MOI.set(m, MOI.RawParameter("absolute_tolerance"), 1E-4)
    MOI.set(m, MOI.RawParameter("relative_tolerance"), 1E-4)
    MOI.set(m, MOI.RawParameter("mul_relax_style"), 0)
    m
end

function write_new_model(model_file_name, f, W, b, n, l, p, flag)

    # Gets interval lower bound
    function ANN(w, b, x)
        y = b[1] + W[1]*x
        for i = 2:length(b)
            y = b[i] + W[2]*y
        end
        return sum(y)
    end
    X = [Interval(-1,1) for i=1:n]
    qIntv = ANN(W, b, X)
    qL = qIntv.lo
    qU = qIntv.hi

    # Builds nn
    layer_val = [String[] for i=1:l]
    for k = 1:l               # compute first layer
        bl = b[k]
        Wl = W[k]
        for i = 1:p           # iterate over number of neurons per layer
            s = "$(bl[i])"
            if k != 1
                llayer = layer_val[k-1]
                for j = 1:p
                    s = s*" + $(Wl[i,j])*$(llayer[j])"
                end
            else
                for j = 1:n
                    s = s*" + $(Wl[i,j])*\$(x[$j])"
                end
            end
            s = f(s; lib = flag)
            push!(layer_val[k], s)
        end
    end

    llast = layer_val[l]
    nn = "$(llast[1])"
    for i = 2:p
        nn = nn*" + $(llast[i])"
    end

    open(model_file_name, "w") do file
        write(file, "using JuMP, EAGO\n
                     m = Model()\n
                     EAGO.register_eago_operators!(m)\n
                     @variable(m, -1 <= x[i=1:$n] <= 1)\n
                     @variable(m, $qL <= q <= $qU)\n
                     add_NL_constraint(m, :("*nn*" - \$q <= 0.0))\n
                     @objective(m, Min, q)\n
                     return m\n
                    "
              )
    end
    println("wrote instance to $model_file_name")
end

relu(x::String; lib = true) =     lib ? "relu("*x*")"     : "max("*x*", 0.0)"
silu(x::String; lib = true) =     lib ? "swish("*x*")"    : "("*x*")/(1 + exp(-("*x*")))"
gelu(x::String; lib = true) =     lib ? "gelu("*x*")"     : "("*x*")*(1 + erf(("*x*")/sqrt(2)))/2"
softsign(x::String; lib = true) = lib ? "softsign("*x*")" : "("*x*")/(1 + "*x*")"
sigmoid(x::String; lib = true) =  lib ? "sigmoid("*x*")"  : "1/(1 + exp(-("*x*")))"
softplus(x::String; lib = true) = lib ? "softplus("*x*")" : "log(1 + exp("*x*"))"
maxsig(x::String; lib = true) =   lib ? "maxsig("*x*")"   : "max("*x*", 1/(1 + exp(-("*x*"))))"

function create_lib()
    instance_number = 100
    variable_range = 2:5
    layer_range = 1:4
    neuron_per_layer_range = 2:5
    act_func = [sigmoid; silu; gelu; softplus]
    minlp_folder_instances = joinpath(@__DIR__, "MINLPLib.jl", "instances")

    for i = 1:instance_number
        n = rand(variable_range)
        layers = rand(layer_range)
        neuron_per_layer = rand(neuron_per_layer_range)
        W = []
        b = []
        for j = 1:layers
            if j == 1
                push!(W, rand(neuron_per_layer, n)*2 .- 1)
            else
                push!(W, rand(neuron_per_layer, neuron_per_layer)*2 .- 1)
            end
            push!(b, rand(neuron_per_layer)*2 .- 1)
        end
        j = i > 9 ? "$i" : "0$i"
        for f in act_func
            file_name_env = joinpath(minlp_folder_instances, "ANN_Env", "$(j)_"*String(Symbol(f))*"_$(n)_$(layers)_$(neuron_per_layer).jl")
            file_name_expr = joinpath(minlp_folder_instances, "ANN_Expr", "$(j)_"*String(Symbol(f))*"_$(n)_$(layers)_$(neuron_per_layer).jl")
            file_name_env_15 = joinpath(minlp_folder_instances, "ANN_Env_15", "$(j)_"*String(Symbol(f))*"_$(n)_$(layers)_$(neuron_per_layer).jl")
            file_name_expr_15 = joinpath(minlp_folder_instances, "ANN_Expr_15", "$(j)_"*String(Symbol(f))*"_$(n)_$(layers)_$(neuron_per_layer).jl")
            
            write_new_model(file_name_env, f, W, b, n, layers, neuron_per_layer, true)
            write_new_model(file_name_expr, f, W, b, n, layers, neuron_per_layer, false)
            write_new_model(file_name_env_15, f, W, b, n, layers, neuron_per_layer, true)
            write_new_model(file_name_expr_15, f, W, b, n, layers, neuron_per_layer, false)
        end
    end
end

# set create_lib_files to true to generate new ANNs in the problem libary
create_lib_files = false
create_lib_files && create_lib()

expr_solvers = Dict{String,Any}()
expr_solvers["SCIP"]  = scip_factory
expr_solvers["EAGO"]  = eago_factory
expr_solvers["BARON"] = baron_factory

env_solvers = Dict{String,Any}()
env_solvers["EAGO"] = eago_factory

function drop_num_underscore(x)
    y = replace(x, "_" => "")
    for i=0:9
        y = replace(y, "$i" => "")
    end
    return y
end

function print_summary_tables(df_env, df_expr, fig_name, env_lib, upper_limit)
    
    df_expr.Form = ["Exp" for i=1:size(df_expr,1)]
    df_env.Form = ["Env" for i=1:size(df_env,1)]

    df_comb = vcat(df_env, df_expr)
    df_comb[!,:FullSolverName] = map((x,y) -> string(x)*" "*string(y), df_comb.SolverName, df_comb.Form)
    df_comb.SolveTime = map((x,y) -> ifelse(occursin("INFEASIBLE",x), upper_limit, y), df_comb.TerminationStatus, df_comb.SolveTime)
    df_comb.ShiftedSolveTime = df_comb.SolveTime .+ 1
    df_comb.CorrectlySolved = map(x -> (occursin("OPTIMAL",x) || occursin("LOCALLY_SOLVED",x)), df_comb.TerminationStatus)

    df_comb.IncorrectFeasibility = map(x -> occursin("INFEASIBLE",x), df_comb.TerminationStatus)
    # Presolve infeasibility in SCIP is labelled as OPTIMIZE_NOT_CALLED...
    df_comb.IncorrectFeasibility = map((x,y,z) -> ifelse(occursin("SCIP",x) & occursin("NOT_CALLED",y), true, z), df_comb.SolverName, df_comb.TerminationStatus, df_comb.IncorrectFeasibility)

    df_comb.ActFunc = map(drop_num_underscore, df_comb.InstanceName)

    gdf_comb_cs = groupby(df_comb, Symbol[:FullSolverName, :ActFunc])
    @show combine(gdf_comb_cs, :ShiftedSolveTime => x -> StatsBase.geomean(x) - 1)

    gdf_combt_correct = groupby(df_comb , Symbol[:FullSolverName, :CorrectlySolved, :ActFunc])
    @show combine(gdf_combt_correct, :ShiftedSolveTime => x -> count(fill(true,length(x))))

    df_comb_infeas = df_comb[df_comb.IncorrectFeasibility,:]
    gdf_check_infeasible  = groupby(df_comb_infeas, Symbol[:FullSolverName, :ActFunc])
    @show combine(gdf_check_infeasible, :SolveTime => x -> count(fill(true,length(x))))

    #gdf_status = groupby(df_comb, Symbol[:FullSolverName, :SolvedInTime])
    env_folder = joinpath(@__DIR__, "MINLPLib.jl", "instances", env_lib)
    name_anns = filter(x -> !occursin("gelu",x), readdir(joinpath(env_folder)))

    df_comb.UnsolvedRelativeGap = map((x,y,z) -> (~x & ~y & isfinite(z)), df_comb.CorrectlySolved, df_comb.IncorrectFeasibility, df_comb.RelativeGap)
    df_comb_gap = df_comb[df_comb.UnsolvedRelativeGap,:]
    gdf_comb_gap  = groupby(df_comb_gap, Symbol[:FullSolverName, :ActFunc])
    @show combine(gdf_comb_gap, :RelativeGap => x -> mean(x))

    trunc_solved_time = rand(300, 4)
    for (i,n) in enumerate(name_anns)
        plt_sdf = df_comb[occursin.(n[1:end-3], string.(df_comb.InstanceName)), :]
        trunc_solved_time[i,3] = plt_sdf[plt_sdf.FullSolverName .== "SCIP Exp",:].SolveTime[1]  # SCIP
        trunc_solved_time[i,4] = plt_sdf[plt_sdf.FullSolverName .== "BARON Exp",:].SolveTime[1] # BARON
        trunc_solved_time[i,1] = plt_sdf[plt_sdf.FullSolverName .== "EAGO Env",:].SolveTime[1]  # EAGO Env Entry
        trunc_solved_time[i,2] = plt_sdf[plt_sdf.FullSolverName .== "EAGO Exp",:].SolveTime[1]  # EAGO Exp Entry
    end
    plt = performance_profile(PlotsBackend(), trunc_solved_time, ["EAGO (Envelope)", "EAGO (McCormick)", "SCIP", "BARON"], linewidth = 1.5, linestyles=[:solid, :dash, :dashdot, :dot], legend=:bottomright)
    xlabel!("\$\\tau\$")
    xlims!(0.0,6.5)
    ylabel!("\$P(r_{p,s} \\leq \\tau : 1 \\leq s \\leq n_s)\$")
    ylims!(0.2,1.0)
    savefig(plt, joinpath(result_path, fig_name*".pdf"))
    show(plt)
end

result_path = joinpath(@__DIR__, "solver_benchmark_result")

params_15 = SolverBenchmarking.BenchmarkParams(time = 15*60, rerun = false, has_obj_bnd = true)
#params = SolverBenchmarking.BenchmarkParams(time = 100, rerun = false, has_obj_bnd = true)

SolverBenchmarking.run_solver_benchmark(result_path, expr_solvers, "ANN_Expr", "ANN_Expr_15"; params = params_15)
SolverBenchmarking.run_solver_benchmark(result_path, env_solvers, "ANN_Env", "ANN_Env_15"; params = params_15)
#SolverBenchmarking.run_solver_benchmark(result_path, expr_solvers, "ANN_Expr", "ANN_Expr"; params = params)
#SolverBenchmarking.run_solver_benchmark(result_path, env_solvers, "ANN_Env", "ANN_Env"; params = params)

SolverBenchmarking.summarize_results("ANN_Expr_15", result_path)
SolverBenchmarking.summarize_results("ANN_Env_15", result_path)
#SolverBenchmarking.summarize_results("ANN_Expr", result_path)
#SolverBenchmarking.summarize_results("ANN_Env", result_path)

df_expr_15 = DataFrame(CSV.File(joinpath(result_path, "ANN_Expr_15", "result_summary_15.csv")))
df_env_15 = DataFrame(CSV.File(joinpath(result_path, "ANN_Env_15", "result_summary_15.csv")))
#df_expr = DataFrame(CSV.File(joinpath(result_path, "ANN_Expr", "result_summary.csv")))
#df_env = DataFrame(CSV.File(joinpath(result_path, "ANN_Env", "result_summary.csv")))

print_summary_tables(df_env, df_expr, "performance_profile_15", "ANN_Env_15", 15*60.0)
#print_summary_tables(df_env, df_expr, "performance_profile", "ANN_Env", 100.00)