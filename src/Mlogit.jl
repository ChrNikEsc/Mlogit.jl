module Mlogit

using DataFrames, Pipe#, BenchmarkTools, Profile

using Reexport
@reexport using StatsModels

using StatsBase, StatsAPI, Optim, RegressionTables, DataStructures, Vcov, Distributions, Makie, CairoMakie, Combinatorics
using Optim, Distributions, LineSearches
import ForwardDiff
using FiniteDifferences
using LinearAlgebra: diag, dot



export mlogit
export nests

include("utils.jl")

include("MNLmodel.jl")
include("mlogit.jl")

end
