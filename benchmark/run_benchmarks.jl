import Pkg
Pkg.activate("benchmark/")

using BenchmarkTools
# using PProf
using JET
import Optim # to specify Optim.Options

Pkg.develop(Pkg.PackageSpec(path = "../Mlogit"))
using Mlogit  # Your package
using Dates
using CSV, DataFrames  # Example dependency for data handling

const mlogit_datadir = joinpath(dirname(@__FILE__), "..", "data/")

# mlogit
electrictiy_long = CSV.read(mlogit_datadir * "electricity_long.csv", DataFrame)
# subset!(electrictiy_long, :id => x -> x .∈ [1:100])
df_mlogit = electrictiy_long
transform!(groupby(df_mlogit, :chid), eachindex => :alt)
transform!(df_mlogit, :id => (id -> 2 * id .- 5) => :x1)
transform!(df_mlogit, :x1 => (x1 -> 1 / 2 * x1) => :x2)
transform!(df_mlogit, :id => (x -> x / sum(unique(df_mlogit.id))) => :weight)
transform!(df_mlogit, :id => (x -> ifelse.(x .> 30, "a", "b")) => :cluster)

# Define benchmark
formula = @formula(choice ~ pf + cl + loc + wk + tod + seas)

mlogit(formula, df_mlogit, weights=:weight)

# suite = BenchmarkGroup()
# suite["mlogit"] = @benchmarkable mlogit($formula, $df_mlogit, weights=:weight)

# Run benchmarks
# results = run(suite, verbose=true)

# b = @benchmarkable mlogit($formula, $df_mlogit, weights=:weight) seconds=30
# results = run(b)
# median_results = median(results)
# display(median_results)
# BenchmarkTools.save("benchmark/results_"*Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS") * "_" * string(median_results.allocs) * ".json", median(results))

# mat_X, vec_choice, vec_chid, vec_weights_choice, vec_nests, coef_start, coef_names, n_coefficients, n_id, n_chid, nested, formula, formula_origin, formula_schema = Mlogit.prepare_mlogit_inputs(formula, df_mlogit, Mlogit.xlogit_indices(), :weight, nothing, false)

# b_fit_mlogit = @benchmarkable Mlogit.fit_mlogit($mat_X, $vec_choice, $coef_start, $vec_chid, $vec_weights_choice; optim_options=Optim.Options()) seconds = 30
# results_fit_mlogit = run(b_fit_mlogit)
# BenchmarkTools.save("benchmark/results_"*Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS") * "_" * string(median(results_fit_mlogit).allocs) * ".json", median(results_fit_mlogit))

# reportopt_fit_mlogit = @report_opt Mlogit.fit_mlogit(mat_X, vec_choice, coef_start, vec_chid, vec_weights_choice; optim_options=Optim.Options())
# print(reportopt_fit_mlogit)

# Profile the mlogit function
# @profview mlogit(formula, df_mlogit, weights=:weight)

# reportopt = @report_opt mlogit(formula, df_mlogit, weights=:weight, optim_options=Optim.Options(iterations=1))

# println(reportopt)





# fmlogit
# df_fmlogit = CSV.read(mlogit_datadir * "fmlogit_data.csv", DataFrame)

# b_fmlogit = @benchmarkable Mlogit.fmlogit(@formula(y1 + y2 + y3 + y4 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8), $df_fmlogit, multithreading=false) seconds = 30
# results_fmlogit = run(b_fmlogit)
# BenchmarkTools.save("benchmark/results_"*Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS") * "_" * string(median(results_fmlogit).allocs) * ".json", median(results_fmlogit))

# reportopt_fmlogit = @report_opt Mlogit.fmlogit(@formula(y1 + y2 + y3 + y4 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8), df_fmlogit, multithreading=false)
# println(reportopt_fmlogit)

# lclogit
df_lclogit = CSV.read(mlogit_datadir * "statadata_lclogit2_classes7_seed10329.csv", DataFrame)
model_lclogit_em = lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 7, method=:em, varname_samplesplit=:samplesplit)
b_lclogit = @benchmarkable lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), $df_lclogit, 7, start_mnl=model_lclogit_em.coef_mnl, start_memb=model_lclogit_em.coef_memb, method=:gradient) seconds=30
results_lclogit = run(b_lclogit)
BenchmarkTools.save("benchmark/results_"*Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS") * "_" * string(median(results_lclogit).allocs) * ".json", median(results_lclogit))

# reportopt_lclogit = @report_opt lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 7, method=:em, varname_samplesplit=:samplesplit)
# println(reportopt_lclogit)

# @profview lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 7, method=:em, varname_samplesplit=:samplesplit)
