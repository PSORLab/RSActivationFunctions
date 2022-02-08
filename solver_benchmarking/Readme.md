This folder contains a module used to benchmark solvers versus a library of optimization problems defined using the MINLPLib package. In order to generate results from a benchmark testset follow below steps:

## Step 1: Generate a test library using MINLPLib tools. 

For problems not included in MINLPLib, the instances may be copied into the folder once the library is constructed. An example script is included below:

```julia
# build small problem library named "ImprovedCompMult"
using MINLPLib
new_lib = "ImprovedCompMult"
instance_names = ["bearing", "ex6_2_10", "ex6_2_10", "ex6_2_11", "ex6_2_12", "ex6_2_13", "ex7_2_4", "ex7_3_1", "ex7_3_2", "ex14_1_8", "ex14_1_9"]
source_lib = "global"
for n in instance_names
    MINLPLib.add_to_lib(new_lib, source_lib, n)
end
```

## Step 2: Run the below script to benchmark each instance
Specify a path to save results (`result_path`). Create a dictionary containing names of solvers and pairs of associated optimizer factories (functions that take no arguments and create the desired optimizer) and model initializers (functions that mutate the model). Then call the `benchmark_solvers` function. This will spawn call the `benchmark_problem` function for each combination of optimizer factory and problem in the benchmarking set which stores a JSON file within `result_path` directory. If a JSON file with the target name is present the solver/instance is skipped. In order to only run missing instances, set the `params.rerun = false`.

```julia
include("C:\\Users\\wilhe\\Desktop\\Package Development\\JuliaScripts\\solver_benchmarking\\solver_benchmarking.jl")
using SCIP

result_path = "C:\\Users\\wilhe\\Desktop\\Package Development\\JuliaScripts\\test_benchmark"
lib = "ImprovedCompMult"

s = Dict{String,Any}()
scip_lo() = SCIP.Optimizer(limits_gap=1E-1, limits_absgap=1E-1)
s["SCIP-lo toleranece"] = scip_lo, x -> nothing
s["SCIP-reg tolerance"] = SCIP.Optimizer, x -> nothing
s["SCIP-hi tolerance"] = () -> SCIP.Optimizer(limits_gap=1E-5, limits_absgap=1E-5), x -> nothing

params = SolverBenchmarking.BenchmarkParams()
SolverBenchmarking.run_solver_benchmark(result_path, s, lib; params = params)
```

The `summarize_results` function is used to tabulate the results into a CSV file

```julia
summarize_results(lib, result_path)
```

The one can then generate generate the Dolan-More performance profiles using the following script 
```julia
using DataFrames, CSV, BenchmarkProfiles, Plots
csv_path = joinpath(result_path, "result_summary.csv")
df = CSV.read(csv_path,DataFrame)
dfm = DataFrame()
solver_names = String[]
for s in groupby(df, "SolverName")
    setfield(dfm, Symbol(s.SolverName[1]), s.CompletedSolveTime)
    push!(solver_names, s.SolverName[1])
end
T = convert(Matrix,df)
performance_profile(PlotsBackend(), T, solver_names, title="Insert Title Here")
```