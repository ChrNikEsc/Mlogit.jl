module Mlogit

using DataFrames, Pipe#, BenchmarkTools, Profile

using Reexport
@reexport using StatsModels
@reexport using Random
@reexport using Distributions

using StatsBase, StatsAPI, Optim, RegressionTables, DataStructures, Vcov, Combinatorics
using Makie, CairoMakie, ColorSchemes
using StaticArrays
using LineSearches
using ForwardDiff
using DiffResults
using FiniteDifferences
using LinearAlgebra: diag, diagm, dot, mul!
import LogExpFunctions
# using NamedArrays # used to be in fmlogit.jl but might no longer be necessary
# using Random # used to be in fmlogit.jl but might no longer be necessary
# using CategoricalArrays # used to be in lclogit.jl but might no longer be necessary
# using ColorSchemes # used to be in lclogit.jl but might no longer be necessary
using Printf
using Primes
using Sobol
using StatsFuns

# import Base.length # used to be in lclogit.jl but might no longer be necessary

export MNLmodel
export LCLmodel

export mlogit
export nests
export membership
export random
export fmlogit
export lclmodel
export lclogit
export robust_cluster_vcov
export coefplot

include("utils.jl")

include("MNLmodel.jl")
include("FMLmodel.jl")
include("LCLmodel.jl")
include("mlogit.jl")
include("fmlogit.jl")
include("lclogit.jl")
include("mlogit_mixed.jl")

end
