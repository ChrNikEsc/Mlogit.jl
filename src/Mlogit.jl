module Mlogit

using DataFrames, Pipe#, BenchmarkTools, Profile

using Reexport
@reexport using StatsModels

using StatsBase, StatsAPI, Optim, RegressionTables, DataStructures, Vcov, Distributions, Makie, CairoMakie, Combinatorics
using Optim, Distributions, LineSearches
import ForwardDiff
using FiniteDifferences



export mlogit

include("utils.jl")

include("MNLmodel.jl")
include("mlogit.jl")

end
