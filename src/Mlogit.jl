module Mlogit

using DataFrames, Pipe#, BenchmarkTools, Profile

using Reexport
@reexport using StatsModels

using StatsBase, StatsAPI, Optim, RegressionTables, DataStructures, Vcov, Distributions, Makie, CairoMakie, Combinatorics
using Optim, Distributions, LineSearches
using ForwardDiff
using FiniteDifferences
using LinearAlgebra: diag, dot
# using NamedArrays # used to be in fmlogit.jl but might no longer be necessary
# using Random # used to be in fmlogit.jl but might no longer be necessary



export mlogit
export nests
export fmlogit

include("utils.jl")

include("MNLmodel.jl")
include("mlogit.jl")
include("fmlogit.jl")

end
