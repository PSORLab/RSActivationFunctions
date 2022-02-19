
module SolverBenchmarking

export summarize_results, run_solver_benchmark, BenchmarkParams

using JSON, DataFrames, CSV
using MINLPLib, JuMP, EAGO

"""
Holds parameters for solver benchmarking.
"""
Base.@kwdef mutable struct BenchmarkParams
    time::Float64 = 200.0
    threads_per::Int = 1
    rerun::Bool = true
    has_obj_bnd::Bool = false
end

"""
Custom error type to handle exceptions 
and log information that occur when solving.
"""
struct InstanceError <: Exception
    path::String
    solver_name::String
    model_name::String
end

"""
Defines storage routine when an instance error occurs.
"""
function store_error(d::InstanceError)
    data = Dict{String,Any}()
    data["SolverName"] = d.solver_name
    data["InstanceName"] = d.model_name
    data["TerminationStatus"] = string(MOI.OTHER_ERROR)
    data["PrimalStatus"] = string(MOI.OTHER_RESULT_STATUS)
    data["ObjBound"] = 0.0
    data["ObjValue"] = 0.0
    data["SolveTime"] = 200.0
    data["CompletedSolveTime"] = 200.0
    data["Error"] = true

    file_path = joinpath(d.path, "$(d.solver_name)_$(d.model_name).json")
    open(file_path,"w") do f
        JSON.print(f, data)
    end
    return
end

#=
function capture_stdout(f)
    stdout_orig = stdout
    (rd, wr) = redirect_stdout()
    f()
    close(wr)
    redirect_stdout(stdout_orig)
    read(rd, String)
end
=#

"""
Fetches the model at "lib\\model_name", attaches the optimizers factory "opt_factory", sets benchmark parameters,
solves the optimization problem, then tidies up the memory space by finalizing the background then garbage collecting.
"""
function fetch_and_optimize(lib::String, model_name::String, opt_factory::Any, params::BenchmarkParams)
    m = fetch_model(lib, model_name)
    set_optimizer(m, opt_factory)

    set_time_limit_sec(m, params.time)
    set_silent(m)

    t = time()
    try
        JuMP.optimize!(m)
    catch
    end
    t = time() - t

    finalize(backend(m))
    GC.gc()
    return m, t
end

"""
If the rerun parameter is set to `true` or no JSON file ("p\\\$(solver_name)_\$(model_name).json") exists
then fetch the model, optimize it and store results to a JSON file at the path ("p\\\$(solver_name)_\$(model_name).json").
"""
function benchmark_problem(opt_factory::Any, solver_name::String, lib::String, libout::String, model_name::String, p::String, params::BenchmarkParams)
    file_path = joinpath(p, libout, "$(solver_name)_$(model_name).json")
    if !isfile(file_path) || params.rerun
        m, t = fetch_and_optimize(lib, model_name, opt_factory, params)

        obj_val = Inf
        s_time = params.time
        t_status = MOI.OPTIMIZE_NOT_CALLED
        p_status = MOI.NO_SOLUTION
        try
            obj_val = objective_value(m)
            s_time = solve_time(m)
            t_status = termination_status(m)
            p_status = primal_status(m)
        catch e
            obj_val = Inf
            s_time = params.time
            t_status = MOI.OPTIMIZE_NOT_CALLED
            p_status = MOI.NO_SOLUTION
        end

        obj_bnd = -Inf
        try
            obj_bnd = objective_bound(m)            
        catch e
            obj_bnd = -Inf
        end


        data = Dict{String,Any}()
        data["SolverName"] = solver_name
        data["InstanceName"] = model_name
        data["TerminationStatus"] = string(t_status)
        data["PrimalStatus"] = string(p_status)
        data["ObjBound"] = params.has_obj_bnd ? obj_bnd : -Inf
        data["ObjValue"] = obj_val
        data["SolveTime"] = s_time
        data["SolveTimeRaw"] = t
        data["CompletedSolveTime"] = (t_status == MOI.OPTIMAL) ? s_time : 100.0
        data["Error"] = false

        open(file_path,"w") do f
            JSON.print(f, data)
        end
    end
    return
end

"""
Main routine for running solver benchmark.
- If no directory `p` exists, make it.
- If no directory `p\\lib` exists, make it.
- Write the parameter file to the file path.
For each solver factory in "s", and each problem in "lib".
"""
function run_solver_benchmark(p::String, s::Dict{String,Any}, lib::String, libout::String; params = BenchmarkParams())

    # store parameter file
    !isdir(p) && mkdir(p)
    !isdir(joinpath(p, lib)) && mkdir(joinpath(p, lib))
    file_path = joinpath(p, libout, "parameters.json")
    data = Dict{String,Any}()
    data["time"] = params.time
    data["threads_per"] = params.threads_per
    data["rerun"] = params.rerun
    open(file_path,"w") do f
        JSON.print(f, data)
    end

    # benchmark solvers
    n = MINLPLib.fetch_names(lib)
    for (k,v) in s
        println("Running all benchmarks problems for solver = $(k)")
        @show length(n)
        for ni in n
            println("Running problem = $ni in $lib")
            benchmark_problem(v, k, lib, libout, ni, p, params)
        end
    end
    println("Benchmark Suite Complete")
    return
end

"""
"""
function summarize_results(lib::String, p::String)
    folder_path = joinpath(p, lib)
    df = DataFrame(SolverName = String[], InstanceName = String[], TerminationStatus = String[], PrimalStatus = String[],
                   ObjBound = Float64[], ObjValue = Float64[], SolveTime = Float64[], SolveTimeRaw = Float64[],
                   CompletedSolveTime = Float64[], Error = Bool[])
    for file in readdir(folder_path)
        if !endswith(file, ".json") || file === "parameters.json"
            continue
        end
        d = joinpath(folder_path, file) |> open |> JSON.parse
        obj_bnd = isnothing(d["ObjBound"]) ? -Inf : d["ObjBound"]
        obj_val = isnothing(d["ObjValue"]) ? Inf : d["ObjValue"]
        push!(df, (d["SolverName"], d["InstanceName"], d["TerminationStatus"], d["PrimalStatus"], obj_bnd, obj_val, d["SolveTime"], d["SolveTimeRaw"], d["CompletedSolveTime"], d["Error"]))
    end

    param_dict = joinpath(p, lib, "parameters.json") |> open |> JSON.parse
    df.RelativeGap = abs.(df.ObjValue - df.ObjBound)./min.(abs.(df.ObjValue), abs.(df.ObjBound))
    df.AbsoluteGap = df.ObjValue - df.ObjBound
    df.SolvedInTime = df.SolveTime .< param_dict["time"]
    CSV.write(joinpath(folder_path, "result_summary.csv"), df)
    return
end

end
