using Mlogit
using Test

using CSV, DelimitedFiles, DataFrames, RegressionTables

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

# mixed logit
df_mixed = readdlm(mlogit_datadir * "datamixed.txt") |> x -> DataFrame(x, [:id, :chid, :choice, :price, :opcost, :range, :ev, :gas, :hybrid, :hiperf, :medhiperf])
# price and opcost negative for lognormal distribution
df_mixed[:, 4] .= -df_mixed[:, 4]
df_mixed[:, 5] .= -df_mixed[:, 5]
# XMAT[:, 4] .= XMAT[:, 4] ./ 10000 # To scale price to be in tens of thousands of dollars.
df_mixed[:, 4] .= df_mixed[:, 4] ./ 10000
df_mixed.weight = log2.(df_mixed.id)

# nlogit
HCdata = CSV.read(mlogit_datadir * "HCdata.csv", DataFrame)
rename!(HCdata, Symbol.(replace.(String.(names(HCdata)), "." => "_")))
transform!(HCdata, :id => :chid)
transform!(HCdata, :alt => ByRow(x -> ifelse(x ∈ ["gcc", "ecc", "erc", "hpc"], "cooling", "others")) => :nest)

# fmlogit
df_fmlogit = CSV.read(mlogit_datadir * "fmlogit_data.csv", DataFrame)

# lclogit
df_lclogit = CSV.read(mlogit_datadir * "statadata_lclogit2_classes7_seed10329.csv", DataFrame)

@testset "Mlogit.jl" begin
    # mlogit

    model_mlogit = mlogit(
        @formula(choice ~ pf + cl + loc + wk + tod + seas),
        df_mlogit;
        weights=:weight
    )
    
    @test sum(model_mlogit.coef) ≈ -10.368346014522867

    # nlogit
    model_nlogit = mlogit(
        @formula(choice ~ ich + och + icca + occa + inc_room + inc_cooling + int_cooling + nests(nest)),
        HCdata,
        equal_lambdas=false
    )

    @test round(sum(model_nlogit.coef), digits=3) ≈ -7.895 # This seems to be relatively unstable

    # fmlogit
    model_fmlogit = fmlogit(@formula(y1 + y2 + y3 + y4 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8), df_fmlogit, multithreading=true)

    # lclogit
    model_lclogit_em = lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 7, method=:em, varname_samplesplit=:samplesplit)
    @test model_lclogit_em.loglikelihood ≈ -1006.353793

    model_lclogit_grad = lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 7, start_mnl=model_lclogit_em.coef_mnl, start_memb=model_lclogit_em.coef_memb, method=:gradient)
    @test model_lclogit_grad.loglikelihood ≈ -1006.3534820949649

    # mixed logit
    model_mixed = mlogit(
        @formula(choice ~ price + opcost + range + ev + hybrid + hiperf + medhiperf),
        df_mixed,
        randdist=[:lognormal, :lognormal, :lognormal, :normal, :normal, nothing, nothing],
        start=[0.0, 0.0, # scalar coefs
            0.2, 0.1, 0.1, -1.44, 0.412, # b random
            0.01, 0.01, 0.01, 0.01, 0.01],
        # weights=:weight,
        draws=(100, "test/mydraws4.mat")
    )
    @test model_mixed.loglikelihood ≈ -1305.9443349232477
end


# lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 3, method=:em)
# Mlogit.regtable(model_lclogit_grad)

# model_lclogit_em = lclogit(@formula(choice ~ pf + cl + loc + wk + tod + seas + membership(x1)), df_lclogit, 7, method=:em, varname_samplesplit=:samplesplit)
# regtable(model_lclogit_em; render=LatexTable())
# Mlogit.coefplot(model_lclogit_em)
# Mlogit.coefplot(model_lclogit_em, by=:coef)
# Mlogit.coefplot(model_lclogit_grad)
