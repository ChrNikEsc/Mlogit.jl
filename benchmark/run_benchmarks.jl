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

b = @benchmarkable mlogit($formula, $df_mlogit, weights=:weight) seconds=30

results = run(b)
BenchmarkTools.save("benchmark/results_"*string(Dates.now())*".json", median(results))


# # Save results
# using JSON
# open("benchmark/results.json", "w") do f
#     JSON.print(f, results)
# end



# Profile the mlogit function
@profview mlogit(formula, df_mlogit, weights=:weight)

reportopt = @report_opt mlogit(formula, df_mlogit, weights=:weight, optim_options=Optim.Options(iterations=1))

println(reportopt)