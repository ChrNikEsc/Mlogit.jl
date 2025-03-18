using Mlogit
using Test

using CSV, DataFrames

const glm_datadir = joinpath(dirname(@__FILE__), "..", "data/")

electrictiy_long = CSV.read(glm_datadir * "electricity_long.csv", DataFrame)
# subset!(electrictiy_long, :id => x -> x .∈ [1:100])

df_mlogit = electrictiy_long
transform!(groupby(df_mlogit, :chid), eachindex => :alt)
transform!(df_mlogit, :id => (id -> 2 * id .- 5) => :x1)
transform!(df_mlogit, :x1 => (x1 -> 1 / 2 * x1) => :x2)
transform!(df_mlogit, :id => (x -> x / sum(unique(df_mlogit.id))) => :weight)
transform!(df_mlogit, :id => (x -> ifelse.(x .> 30, "a", "b")) => :cluster)


@testset "Mlogit.jl" begin
    # mlogit
    model_mlogit = mlogit(
        @formula(choice ~ pf + cl + loc + wk + tod + seas),
        df_mlogit,
        weights=:weight
    )
end
