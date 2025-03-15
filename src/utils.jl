using CSV, DataFrames, Pipe#, BenchmarkTools, Profile
using StatsBase, StatsModels, StatsAPI, Optim, RegressionTables, DataStructures, Vcov, Distributions, Makie, CairoMakie, Combinatorics

struct XlogitIndices
    chid::Symbol
    id::Symbol
    alt::Symbol
end

function xlogit_indices(;
    chid=:chid,
    id=:id,
    alt=:alt
)
    return XlogitIndices(chid, id, alt)
end

# function Base.collect(vn::VarnamesStructure)
#     return [value for value in (vn.choice, vn.chid, vn.id, vn.alt, vn.nest, vn.weight) if value !== nothing]
# end

function safe_log(x)
    # return x < 1e-20 ? 0 : log(x)
    # return x == 0 ? 0 : log(x)

    # changed it in order to fixed nlogit with scaled sample data. didn't fix but didn't hurt. 
    if x > 1e-20
        return log(x)
    elseif x < -1e-20
        return -log(abs(x))
    else
        return 0
    end
end

function safe_exp(b, e)
    return b < 1e-20 ? 0 : b^e
    return b^e
end

function safe_div(n, d)
    return isinf(d) || d < 1e-20 ? 0 : n/d
end

function my_log(text_to_append::String)
    # Open the log file in append mode, write the text, and close the file
    open(log_file_path, "a") do file
        write(file, text_to_append * "\n")
    end
end

function my_log(text_to_append::String, log_file_path::String)
    # Open the log file in append mode, write the text, and close the file
    open(log_file_path, "a") do file
        write(file, text_to_append * "\n")
    end
end

function create_index_map(indices)
    idx_map = Dict{Int,Vector{Int}}()

    for (pos, val) in enumerate(indices)
        push!(get!(idx_map, val, Vector{Int}()), pos)
    end

    # Collecting the values in the order of unique elements
    ordered_values = [idx_map[val] for val in unique(indices)]

    return ordered_values
end

function split_collection(input::T, indices) where {T}
    is_matrix = ndims(input) > 1

    idx_map = create_index_map(indices)

    if is_matrix
        return [input[idx, :] for idx in idx_map]
    else
        return [input[idx] for idx in idx_map]
    end
end

# function remap_to_indices_chid!(v)
#     unique_vals = unique(v)
#     for i in eachindex(v)
#         v[i] = findfirst(==(v[i]), unique_vals)
#     end
# end

function remap_to_indices_chid!(v)
    unique_vals = unique(v)
    val_to_index = Dict(unique_vals .=> 1:length(unique_vals))
    for i in eachindex(v)
        v[i] = val_to_index[v[i]]
    end
end

function remap_to_indices(v)
    unique_vals = unique(v)
    r = zeros(Int64, length(v))
    for i in eachindex(v)
        r[i] = findfirst(==(v[i]), unique_vals)
    end
    return r
end

function remap_to_indices_nest(v)
    # missing, nothing and 0 in the nest column are interpreted as no nest or being its own nest
        # same in xlogit.jl --> definition of coefnames_nests
    unique_vals = filter(x -> !ismissing(x) && !isnothing(x) && x != 0, unique(v))
    val_to_index = Dict(unique_vals .=> 1:length(unique_vals))
    result = Vector{Int}(undef, length(v))
    for i in eachindex(v)
        result[i] = ismissing(v[i]) ? 0 : get(val_to_index, v[i], 0)
    end
    return result
end

function get_X(varnames_X::Vector{String}, dta::DataFrame; varnames_structure::NamedTuple=(choice="choice", chid="chid", id="id", alt="alt"))
    # if column names in varnames_struture exist, rename
    try
        view(dta, !, collect(varnames_structure))
    catch e
        error("varnames_structure: $(e)")
    end

    # check if column names in varlist_X exist
    try
        view(dta, !, varnames_X)
    catch e
        error("varnames_X: $(e)")
    end

    vec_chid = deepcopy(dta[!, Symbol(varnames_structure.chid)])

    remap_to_indices_chid!(vec_chid)

    # Dependent Variables
    X = split_collection(convert.(Float64, Matrix(dta[!, varnames_X])), vec_chid)

    return X
end

function get_y(dta::DataFrame; varnames_structure::NamedTuple=(choice="choice", chid="chid", id="id", alt="alt"))
    # if column names in varnames_struture exist, rename
    try
        view(dta, !, collect(varnames_structure))
    catch e
        error("varnames_structure: $(e)")
    end

    vec_chid = deepcopy(dta[!, Symbol(varnames_structure.chid)])

    remap_to_indices_chid!(vec_chid)

    # Dependent Variables
    y = split_collection(convert.(Float64, (dta[!, Symbol(varnames_structure.choice)])), vec_chid)

    return y
end

function get_ids(dta::DataFrame; varnames_structure::NamedTuple=(choice="choice", chid="chid", id="id", alt="alt"))
    # if column names in varnames_struture exist, rename
    try
        view(dta, !, collect(varnames_structure))
    catch e
        error("varnames_structure: $(e)")
    end

    vec_choice = convert.(Bool, dta[!, Symbol(varnames_structure.choice)])

    return dta[!, Symbol(varnames_structure.id)][vec_choice]
end

function get_weights(varname_weights::Union{String,Nothing}, dta::DataFrame; varnames_structure::NamedTuple=(choice="choice", chid="chid", id="id", alt="alt"))
    # if column names in varnames_struture exist, rename
    try
        view(dta, !, collect(varnames_structure))
    catch e
        error("varnames_structure: $(e)")
    end

    n_chid = length(unique(dta[!, Symbol(varnames_structure.chid)]))
    vec_choice = convert.(Bool, dta[!, Symbol(varnames_structure.choice)])

    vec_weights = ones(n_chid)
    if !isnothing(varname_weights)
        vec_weights .= (dta[!, varname_weights][vec_choice] ./ sum(dta[!, varname_weights][vec_choice])) .* n_chid
    end

    return vec_weights
end

# ---------------------------------------------------------------------------- #
#                               Iterate on terms                               #
# ---------------------------------------------------------------------------- #

# Taken from FixedEffectModels.jl/src/utils/formula.jl

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x

# ---------------------------------------------------------------------------- #
#                            Parse nests in formula                            #
# ---------------------------------------------------------------------------- #

# Follows FixedEffectModels.jl/src/utils/formula.jl

struct NestsTerm <: AbstractTerm
    x::Symbol
end
StatsModels.termvars(t::NestsTerm) = [t.x]
nests(x::Term) = nests(Symbol(x))
nests(s::Symbol) = NestsTerm(s)

has_nests(::NestsTerm) = true
has_nests(::FunctionTerm{typeof(nests)}) = true
has_nests(@nospecialize(t::InteractionTerm)) = any(has_nests(x) for x in t.terms)
has_nests(::AbstractTerm) = false
has_nests(@nospecialize(t::FormulaTerm)) = any(has_nests(x) for x in eachterm(t.rhs))

function parse_nests(@nospecialize(f::FormulaTerm))
    if has_nests(f)
        formula_main = FormulaTerm(f.lhs, Tuple(term for term in eachterm(f.rhs) if !has_nests(term)))
        formula_nests = FormulaTerm(ConstantTerm(0), Tuple(term for term in eachterm(f.rhs) if has_nests(term)))
        return formula_main, formula_nests
    else
        return f, FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    end
end
nestssymbol(t::NestsTerm) = t.x
nestssymbol(t::FunctionTerm{typeof(nests)}) = Symbol(t.args[1])

# ---------------------------------------------------------------------------- #
#                    Parse membership (LCLogit) in formula                     #
# ---------------------------------------------------------------------------- #

# Follows FixedEffectModels.jl/src/utils/formula.jl

# struct MembershipTerm <: AbstractTerm
#     x::Symbol
# end
# StatsModels.termvars(t::MembershipTerm) = [t.x]
# membership(x::Term) = membership(Symbol(x))
# membership(s::Symbol) = MembershipTerm(s)

# has_membership(::MembershipTerm) = true
# has_membership(::FunctionTerm{typeof(membership)}) = true
# has_membership(@nospecialize(t::InteractionTerm)) = any(has_membership(x) for x in t.terms)
# has_membership(::AbstractTerm) = false
# has_membership(@nospecialize(t::FormulaTerm)) = any(has_membership(x) for x in eachterm(t.rhs))

# function parse_membership(@nospecialize(f::FormulaTerm))
#     if has_membership(f)
#         formula_main = FormulaTerm(f.lhs, Tuple(term for term in eachterm(f.rhs) if !has_membership(term)))
#         formula_membership = FormulaTerm(ConstantTerm(0), Tuple(term for term in eachterm(f.rhs) if has_membership(term)))
#         return formula_main, formula_membership
#     else
#         return f, FormulaTerm(ConstantTerm(0), ConstantTerm(0))
#     end
# end
# membershipsymbol(t::MembershipTerm) = t.x
# membershipsymbol(t::FunctionTerm{typeof(membership)}) = Symbol(t.args[1])

# Define a custom term for membership
struct MembershipTerm <: AbstractTerm
    x::Vector{Symbol}
end

# Define how to extract variables from the MembershipTerm
StatsModels.termvars(t::MembershipTerm) = t.x

# Define the membership function for constructing MembershipTerm
function membership(args::Symbol...)
    MembershipTerm(args)
end

# Check if a term contains a membership function
has_membership(::MembershipTerm) = true
has_membership(::FunctionTerm{typeof(membership)}) = true
has_membership(@nospecialize(t::InteractionTerm)) = any(has_membership(x) for x in t.terms)
has_membership(::AbstractTerm) = false
has_membership(@nospecialize(t::FormulaTerm)) = any(has_membership(x) for x in eachterm(t.rhs))

# Helper to get the symbols from membership terms
function membershipsymbol(t::MembershipTerm)
    return t.x
end

# Helper to get symbols from FunctionTerm for membership
function membershipsymbol(t::FunctionTerm{typeof(membership)})
    # Handle both single and multiple terms inside membership()
    if length(t.args) == 1
        return [t.args[1]]  # Single term (e.g., x1)
    else
        return t.args[1].args  # Multiple terms (e.g., x1 + x2)
    end
end

# Extract the non-membership and membership terms from a formula
function parse_membership(@nospecialize(f::FormulaTerm), n_classes)
    if has_membership(f)
        # Main formula: exclude membership terms
        formula_main = FormulaTerm(f.lhs, Tuple(term for term in eachterm(f.rhs) if !has_membership(term)))

        # Membership formula: expand membership terms into individual terms
        membership_terms = Tuple(term for term in eachterm(f.rhs) if has_membership(term))

        # Flatten and extract individual symbols
        predictors = vcat(map(membershipsymbol, membership_terms)...)
        # Construct the membership formula with individual terms
        formula_membership = FormulaTerm(foldl(+, term.("lcl_H$s") for s in 1:n_classes), foldl(+, predictors))

        return formula_main, formula_membership
    else
        return f, FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    end
end
