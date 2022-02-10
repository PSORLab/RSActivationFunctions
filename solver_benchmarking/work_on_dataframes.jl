using DataFrames, CSV, StatsBase

result_path = joinpath(@__DIR__, "solver_benchmark_result")

df_env = DataFrame(CSV.File(joinpath(result_path, "ANN_Env", "result_summary.csv")))
df_expr = DataFrame(CSV.File(joinpath(result_path, "ANN_Expr", "result_summary.csv")))
df_env.Form = ["Env" for i=1:size(df_env,1)]
df_expr.Form = ["Exp" for i=1:size(df_expr,1)]

df_comb = vcat(df_env, df_expr)
df_comb[!,:FullSolverName] = map((x,y) -> string(x)*" "*string(y), df_comb.SolverName, df_comb.Form)
df_comb.ShiftedSolveTime = df_comb.SolveTime .+ 1

function drop_num_underscore(x)
    y = replace(x, "_" => "")
    for i=0:9
        y = replace(y, "$i" => "")
    end
    return y
end
df_comb.ActFunc = map(drop_num_underscore, df_comb.InstanceName)


gdf_comb = groupby(df_comb, Symbol[:FullSolverName, :ActFunc])
@show combine(gdf_comb, :ShiftedSolveTime => x -> StatsBase.geomean(x) - 1)
gdf_combt = groupby(df_comb, Symbol[:FullSolverName, :TerminationStatus, :ActFunc])
@show combine(gdf_combt, :ShiftedSolveTime => x -> count(x))

#gdf_status = groupby(df_comb, Symbol[:FullSolverName, :SolvedInTime])
