module Mlogit

using DataFrames, Pipe#, BenchmarkTools, Profile

using Reexport
@reexport using StatsModels

using StatsBase, StatsAPI, Optim, RegressionTables, DataStructures, Vcov, Distributions, Makie, CairoMakie, Combinatorics
using Optim, Distributions, LineSearches
using ForwardDiff
using FiniteDifferences
using LinearAlgebra: diag, dot, mul!
# using NamedArrays # used to be in fmlogit.jl but might no longer be necessary
# using Random # used to be in fmlogit.jl but might no longer be necessary
# using CategoricalArrays # used to be in lclogit.jl but might no longer be necessary
# using ColorSchemes # used to be in lclogit.jl but might no longer be necessary
using Printf

# import Base.length # used to be in lclogit.jl but might no longer be necessary


export mlogit
export nests
export fmlogit
export lclmodel
export lclogit
export membership

include("utils.jl")

include("MNLmodel.jl")
include("FMLmodel.jl")
include("LCLmodel.jl")
include("mlogit.jl")
include("fmlogit.jl")
include("lclogit.jl")

end
