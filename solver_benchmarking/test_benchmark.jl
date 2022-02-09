# Adds packages that don't require special setup
using Pkg
#Pkg.develop(path = joinpath(@__DIR__, "McCormick.jl-master"))
Pkg.develop(path = joinpath(@__DIR__, "MINLPLib.jl"))
Pkg.develop(path = joinpath(@__DIR__, "EAGO.jl-SIPextension"))
Pkg.add("McCormick")
Pkg.add("JSON"); Pkg.add("DataFrames"); Pkg.add("CSV")
Pkg.add("JuMP"); Pkg.add("IntervalArithmetic");

# Loads solver_benchmarking module
include(joinpath(@__DIR__, "solver_benchmarking.jl"))

# Loads relevent modules
using JuMP, MINLPLib, SCIP, EAGO, IntervalArithmetic, GAMS

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
    set_optimizer_attribute(m, "mul_relax_style", 0)
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
    instance_number = 50
    variable_range = 2:6
    layer_range = 1:4
    neuron_per_layer_range = 2:6
    act_func = [sigmoid; silu; softsign; softplus]
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
            write_new_model(file_name_env, f, W, b, n, layers, neuron_per_layer, true)
            write_new_model(file_name_expr, f, W, b, n, layers, neuron_per_layer, false)
        end
    end
end

# set create_lib_files to true to generate new ANNs in the problem libary
create_lib_files = false
create_lib_files && create_lib()

expr_solvers = Dict{String,Any}()
expr_solvers["SCIP"]  = scip_factory
#expr_solvers["EAGO"]  = eago_factory
expr_solvers["BARON"] = baron_factory

env_solvers = Dict{String,Any}()
env_solvers["EAGO"] = eago_factory

params = SolverBenchmarking.BenchmarkParams(time = 100, rerun = false, has_obj_bnd = false)

result_path = joinpath(@__DIR__, "solver_benchmark_result")

SolverBenchmarking.run_solver_benchmark(result_path, env_solvers, "ANN_Env", "ANN_Env"; params = params)
SolverBenchmarking.run_solver_benchmark(result_path, expr_solvers, "ANN_Expr", "ANN_Expr"; params = params)

SolverBenchmarking.summarize_results("ANN_Env", result_path)
SolverBenchmarking.summarize_results("ANN_Expr", result_path)

#=
new_lib = "ANN_Expr"
result_path_env = "C:\\Users\\matt\\Desktop\\JOGO Activation Function NEW\\JuliaScripts-main\\test_benchmark_env"
result_path_expr = "C:\\Users\\matt\\Desktop\\JOGO Activation Function NEW\\JuliaScripts-main\\test_benchmark_expr"

relu(x::String; lib = true) =     lib ? "relu("*x*")"     : "max("*x*", 0.0)"
silu(x::String; lib = true) =     lib ? "swish("*x*")"    : "("*x*")/(1 + exp(-("*x*")))"
gelu(x::String; lib = true) =     lib ? "gelu("*x*")"     : "("*x*")*(1 + erf(("*x*")/sqrt(2)))/2"
softsign(x::String; lib = true) = lib ? "softsign("*x*")" : "("*x*")/(1 + "*x*")"
sigmoid(x::String; lib = true) =  lib ? "sigmoid("*x*")"  : "1/(1 + exp(-("*x*")))"
softplus(x::String; lib = true) = lib ? "softplus("*x*")" : "log(1 + exp("*x*"))"
maxsig(x::String; lib = true) =   lib ? "maxsig("*x*")"   : "max("*x*", 1/(1 + exp(-("*x*"))))"


"""
write_new_model

Takes a model file name, a activation function, weight matrices, offset vectors, length of relevant matrices.
"""
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

function create_lib()
    instance_number = 50
    variable_range = 2:6
    layer_range = 1:4
    neuron_per_layer_range = 2:6
    act_func = [sigmoid; silu; softsign; softplus]

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
            file_name_env = result_path_env*"\\"*new_lib*"\\Enve_$(j)_"*String(Symbol(f))*"_$(n)_$(layers)_$(neuron_per_layer).jl"
            file_name_expr = result_path_expr*"\\"*new_lib*"\\Expr_$(j)_"*String(Symbol(f))*"_$(n)_$(layers)_$(neuron_per_layer).jl"
            write_new_model(file_name_env, f, W, b, n, layers, neuron_per_layer, true)
            write_new_model(file_name_expr, f, W, b, n, layers, neuron_per_layer, false)
        end
    end
end


if create_lib_files
    create_lib()
else
end

if !create_lib_files
    # create benchmark library
    s_eago = Dict{String,Any}()
    s_eago["EAGO"] = EAGO.Optimizer

    s_both = Dict{String,Any}()
    #s_both["EAGO"] = EAGO.Optimizer
    s_both["SCIP"] = SCIP.Optimizer

    # create benchmark library
    function baron_factory()
        m = GAMS.Optimizer(GAMS.GAMSWorkspace("C:\\GAMS\\37"))
  
        m.gams_options["nlp"] = "BARON"
        m.gams_options["optca"] = 1E-4
        m.gams_options["optcr"] = 1E-4
  
        return m
    end
    s_baron = Dict{String,Any}()
    s_baron["BARON"] = baron_factory

    params = SolverBenchmarking.BenchmarkParams(time = 100, rerun = false, has_obj_bnd = true)

    SolverBenchmarking.run_solver_benchmark(result_path_env, s_baron, "ANN_Expr"; params = params)
    SolverBenchmarking.summarize_results("ANN_Expr", result_path_env)

    #SolverBenchmarking.run_solver_benchmark(result_path_env, s_eago, "ANN_Env"; params = params)
    #SolverBenchmarking.summarize_results("ANN_Env", result_path_env)

    #SolverBenchmarking.run_solver_benchmark(result_path_expr, s_eago, "ANN_Expr"; params = params)
    #SolverBenchmarking.summarize_results("ANN_Expr", result_path_expr)
end
=#

#=
using DataFrames, CSV
csv_path = joinpath(result_path, "result_summary.csv")
df = CSV.read(csv_path, DataFrame)
dfm = DataFrame()
solver_names = String[]
for s in groupby(df, "SolverName")
    setfield(dfm, Symbol(s.SolverName[1]), s.CompletedSolveTime)
    push!(solver_names, s.SolverName[1])
end
T = convert(Matrix,df)
=#
