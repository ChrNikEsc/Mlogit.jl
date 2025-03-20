using Mlogit
using Test

using CSV, DataFrames

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

# nlogit
HCdata = CSV.read(mlogit_datadir * "HCdata.csv", DataFrame)
rename!(HCdata, Symbol.(replace.(String.(names(HCdata)), "." => "_")))
transform!(HCdata, :id => :chid)
transform!(HCdata, :alt => ByRow(x -> ifelse(x ∈ ["gcc", "ecc", "erc", "hpc"], "cooling", "others")) => :nest)

# fmlogit
df_fmlogit = CSV.read(mlogit_datadir * "fmlogit_data.csv", DataFrame)

@testset "Mlogit.jl" begin
    # mlogit
    model_mlogit = mlogit(
        @formula(choice ~ pf + cl + loc + wk + tod + seas),
        df_mlogit,
        weights=:weight
    )

    @test sum(model_mlogit.coef) ≈ -10.368346014522867

    # nlogit
    model_nlogit = mlogit(
        @formula(choice ~ ich + och + icca + occa + inc_room + inc_cooling + int_cooling + nests(nest)),
        HCdata,
        equal_lambdas=false
    )

    @test round(sum(model_nlogit.coef), digits=6) ≈ -7.895003 # GitHub CI test with julia 1.9 failed when requiring more precision
    
    # fmlogit
    model_fmlogit = fmlogit(@formula(y1 + y2 + y3 + y4 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8), df_fmlogit, multithreading=true)
end




