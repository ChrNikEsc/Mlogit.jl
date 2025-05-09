mutable struct MNLmodel <: RegressionModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    converged::Union{Bool,Nothing}
    depvar::BitVector
    df_hash::UInt64
    dof::Int64
    estfun::Matrix{Float64}
    fitted::Vector{Float64}
    formula::FormulaTerm
    formula_origin::FormulaTerm
    formula_schema::FormulaTerm
    hessian::Matrix{Float64}
    indices::XlogitIndices
    iter::Union{Int64,Nothing}
    loglikelihood::Float64
    mixed::Bool
    nchids::Int64
    nclusters::Union{NamedTuple,Nothing}
    nests::Union{Nothing,Dict}
    nids::Int64
    nullloglikelihood::Float64
    optim::Optim.OptimizationResults
    score::Vector{Float64}
    start::Vector{Float64}
    startloglikelihood::Float64
    time::Float64
    vcov::Matrix{Float64}
    vcov_type::Vcov.CovarianceEstimator
end

function MNLmodel(;
    coef::Vector{Float64},
    coefnames::Vector{String},
    converged::Bool,
    depvar::BitVector,
    df_hash::UInt,
    dof::Int64,
    estfun::Matrix{Float64},
    fitted::Vector{Float64},
    formula::FormulaTerm,
    formula_origin::FormulaTerm,
    formula_schema::FormulaTerm,
    hessian::Matrix{Float64},
    indices::XlogitIndices,
    iter::Union{Int64,Nothing},
    loglikelihood::Float64,
    mixed::Bool,
    nchids::Int64,
    nclusters::Union{NamedTuple,Nothing},
    nests::Union{Nothing,Dict},
    nids::Int64,
    nullloglikelihood::Float64,
    optim::Optim.OptimizationResults,
    score::Vector{Float64},
    start::Vector{Float64},
    startloglikelihood::Float64,
    time::Float64,
    vcov::Matrix{Float64},
    vcov_type::Vcov.CovarianceEstimator)
    return MNLmodel(coef, coefnames, converged, depvar, df_hash, dof, estfun, fitted, formula, formula_origin, formula_schema, hessian, indices, iter, loglikelihood, mixed, nchids, nclusters, nests, nids, nullloglikelihood, optim, score, start, startloglikelihood, time, vcov, vcov_type)
end

function StatsAPI.adjr2(model::MNLmodel, variant::Symbol)
    if variant == :McFadden
        return 1 - (loglikelihood(model) - dof(model)) / nullloglikelihood(model)
    else
        throw(ArgumentError("variant must be :McFadden"))
    end
end
StatsAPI.adjr2(model::MNLmodel) = adjr2(model, :McFadden)
StatsAPI.aic(model::MNLmodel) = -2 * loglikelihood(model) + dof(model) * 2
StatsAPI.aicc(model::MNLmodel) = -2 * loglikelihood(model) + 2 * dof(model) + 2 * dof(model) * (dof(model) - 1) / (nobs(model) - dof(model) - 1)
StatsAPI.bic(model::MNLmodel) = -2 * loglikelihood(model) + dof(model) * log(nobs(model))
caic(model::MNLmodel) = -2 * loglikelihood(model) + dof(model) * (log(nobs(model)) + 1)
StatsAPI.coef(model::MNLmodel) = model.coef
StatsAPI.coefnames(model::MNLmodel) = model.coefnames
function confint(model::MNLmodel; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    hcat(coef(model), coef(model)) + stderror(model, type=type, cluster=cluster) * quantile(Normal(), (1.0 - level) / 2.0) * [1.0 -1.0]
end
function StatsAPI.coeftable(model::MNLmodel; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    cc = coef(model)
    se = stderror(model, type=type, cluster=cluster)
    zz0 = cc ./ se
    # p0 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz0)) #same as, but faster than: p0 = 2.0 * ccdf.(TDist(dof_residual(model)), abs.(zz0))
    p0 = 2.0 * ccdf.(Normal(), abs.(zz0))

    if isnothing(model.nests)
        return CoefTable(hcat(cc, se, zz0, p0, confint(model; level=level, type=type, cluster=cluster)),
            ["Estimate", "Std.Error", "z0 value", "Pr(>|z0|)", "Conf.Low $((level * 100))%", "Conf.High $((level * 100))%"],
            coefnames(model),
            4)
    end
    zz1 = (cc .- 1) ./ se
    p1 = 2.0 * ccdf.(Normal(), abs.(zz1))
    # p1 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz1))
    return CoefTable(hcat(cc, se, zz0, p0, zz1, p1, confint(model; level=level, type=type, cluster=cluster)),
        ["Estimate", "Std.Error", "z0 value", "Pr(>|z0|)", "z1 value", "Pr(>|z1|)", "Conf.Low $((level * 100))%", "Conf.High $((level * 100))%"],
        coefnames(model), 4)
end
# TODO clustering hier verstehen und ggf. mit robust_... einheitlich machen
# deviance
StatsAPI.dof(model::MNLmodel) = model.dof
StatsAPI.dof_residual(model::MNLmodel) = nobs(model) - dof(model)
# fit
# fit!
StatsAPI.fitted(model::MNLmodel; chosen=false) = chosen ? model.fitted[model.depvar] : model.fitted
StatsModels.formula(model::MNLmodel) = model.formula_schema
function informationmatrix(model::MNLmodel; expected::Bool=true)
    if expected
        @warn("Fisher (expected) information matrix not implemented. Returning observed information matrix.")
        return model.hessian
    else
        return model.hessian
    end
end
StatsAPI.isfitted(model::MNLmodel) = model.converged
StatsAPI.islinear(model::MNLmodel) = false
StatsAPI.loglikelihood(model::MNLmodel) = model.loglikelihood
# mss
StatsAPI.nobs(model::MNLmodel; use_nids::Bool=false) = use_nids ? model.nids : model.nchids
# nulldeviance
StatsAPI.nullloglikelihood(model::MNLmodel) = model.nullloglikelihood
function StatsAPI.predict(model::MNLmodel, newDf::DataFrame; chosen=false)

    df = DataFrame(newDf; copycols=false)
    nrows = size(df, 1)
    indices = model.indices
    coef_names = coefnames(model)

    # ---------------------------------------------------------------------------- #
    #                                 Parse formula                                #
    # ---------------------------------------------------------------------------- #

    formula_origin = model.formula_origin
    formula, formula_nests = parse_nests(formula_origin)

    # ---------------------------------------------------------------------------- #
    #                             DataFrame --> Matrix                             #
    # ---------------------------------------------------------------------------- #

    s = schema(formula, df)

    formula_schema = apply_schema(formula, s)
    vec_choice = convert(BitVector, response(formula_schema, df))
    mat_X = convert(Matrix{Float64}, modelmatrix(formula_schema, df))
    response_name, coefnames_utility = coefnames(formula_schema)

    nested = formula_nests != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    vec_nests = nested ? df[:, nestssymbol(formula_nests.rhs[1])] : zeros(Int64, nrows)
    coefnames_nests = if !nested
        String[]
    else
        setdiff(coef_names, coefnames_utility)
    end

    # Ids
    n_id = length(unique(df[!, indices.id]))

    # Chids
    vec_chid = df[!, indices.chid]
    # make sure that vec_chid can be used to index vectors of length length(unique(vec_chid))
    # unique(vec_chid) != 1:length(unique(vec_chid)) && 
    remap_to_indices_chid!(vec_chid)
    idx_map = create_index_map(vec_chid)
    n_chid = length(unique(vec_chid))

    # Nests
    if nested
        # transform vec_nest such that 0 means an alternative is in its own nest (don't estimate lambda)..
        # ..and other nests are re-labelled with integers for indexing in loglik_fun
        vec_nests_indices = remap_to_indices_nest(vec_nests)
        vec_nest_indices_choice = vec_nests_indices[vec_choice]
        n_nests_model = length(model.nests)
    end

    mat_X_choice = mat_X[vec_choice, :]

    # Retrieve coefficients
    theta = coef(model)
    beta = theta[1:length(coefnames_utility)]
    if nested
        lambda = if length(coefnames_nests) == 1 # equal_lambdas
            [1.0; fill(theta[length(coefnames_utility)+1], n_nests_model)]
        else
            [1.0; theta[length(coefnames_utility)+1:end]]
        end
    end


    if !nested
        exb = exp.(mat_X * beta)
        # Initialize objects in fgh!
        sexb = zeros(Float64, n_chid)
        Pni = zeros(Float64, length(vec_chid))

        @inbounds for i in eachindex(vec_chid)
            sexb[vec_chid[i]] += exb[i]
        end

        @inbounds for i in eachindex(vec_chid)
            Pni[i] = exb[i] / sexb[vec_chid[i]]
        end

        return chosen ? Pni[vec_choice] : Pni
    else
        V = mat_X * beta
        exp_adj_V = [exp(V[i] / lambda[vec_nests_indices[i]+1]) for i in eachindex(V)]

        sum_chid_nest = zeros(n_chid, n_nests_model + 1)
        @inbounds for i in eachindex(vec_chid)
            sum_chid_nest[vec_chid[i], vec_nests_indices[i]+1] += exp_adj_V[i]
        end

        sum_sum_chid_nest_ttl = sum(sum_chid_nest .^ lambda', dims=2)

        # Calculate choice probabilities for all alternatives
        fitted_values_all = [(exp_adj_V[i] * (sum_chid_nest[vec_chid[i], vec_nests_indices[i]+1]^(-1 + lambda[vec_nests_indices[i]+1])) / sum_sum_chid_nest_ttl[vec_chid[i]]) for i in eachindex(V)]

        return chosen ? fitted_values_all[vec_choice] : fitted_values_all
    end




















    #     varnames_X = [cn for cn in coefnames(model) if !startswith(cn, "lambda")]

    #     theta = coef(model)

    #     # if column names in varnames_struture exist, rename
    #     # if column names in varnames_struture exist, rename
    #     try
    #         view(newDf, !, collect(model.varnames_structure))
    #     catch e
    #         error("varnames_structure: $(e)")
    #     end
    #     rename!(newDf, Symbol(model.varnames_structure.choice) => :choice, Symbol(model.varnames_structure.chid) => :chid, Symbol(model.varnames_structure.id) => :id, Symbol(model.varnames_structure.alt) => :alt, Symbol(model.varnames_structure.nest) => :nest)

    #     # check if column names in varlist_X exist
    #     try
    #         view(newDf, !, varnames_X)
    #     catch e
    #         error("varnames_X: $(e)")
    #     end

    #     # Ids
    #     n_id = length(unique(newDf.id))

    #     # Chids
    #     vec_chid = newDf.chid
    #     # make sure that vec_chid can be used to index vectors of length length(unique(vec_chid))
    #     # unique(vec_chid) != 1:length(unique(vec_chid)) && 
    #     # function remap_to_indices_chid!(v)
    #     #     unique_vals = unique(v)
    #     #     val_to_index = Dict(unique_vals .=> 1:length(unique_vals))
    #     #     for i in eachindex(v)
    #     #         v[i] = val_to_index[v[i]]
    #     #     end
    #     # end
    #     remap_to_indices_chid!(vec_chid)
    #     n_chid = length(unique(vec_chid))

    #     vec_choice = convert.(Bool, (newDf[!, :choice]))

    #     # Weights
    #     if isnothing(model.varnames_structure.weight)
    #         vec_weights = ones(length(vec_choice))
    #     else
    #         vec_weights = ((newDf[!, model.varnames_structure.weight] ./ sum(newDf[!, model.varnames_structure.weight])) .* length(vec_chid))
    #     end
    #     vec_weights_choice = vec_weights[vec_choice]

    #     # Nests
    #     # TODO check that only nests known to the model are used

    #     # transform vec_nest such that 0 means an alternative is in its own nest (don't estimate lambda)..
    #     # ..and other nests are re-labelled with integers for indexing in loglik_fun
    #     vec_nest = [model.nests[dfn] for dfn in newDf.nest]

    #     n_nests = length(model.nests)   # Number of nests (excluding the outside option)

    #     # Dependent Variables
    #     mat_X = Matrix(newDf[!, varnames_X])

    #     # mat_X_choice = mat_X[vec_choice, :]


    #     beta = theta[1:length(varnames_X)]

    #     lambda = ones(n_nests + 1)

    #     if sum(startswith.(coefnames(model), "lambda")) == 1 # then we only have 1 lambda
    #         lambda[2:end] .= theta[end]
    #     else
    #         lambda[2:end] = theta[(length(varnames_X)+1):end]
    #     end
    #     # lambda_choice = [lambda[i+1] for i in vec_nest_choice]

    #     V = mat_X * beta
    #     exp_adj_V = [exp(V[i] / lambda[vec_nest[i]+1]) for i in eachindex(V)]

    #     sum_chid_nest = zeros(n_chid, n_nests + 1)
    #     @inbounds for i in eachindex(vec_chid)
    #         sum_chid_nest[vec_chid[i], vec_nest[i]+1] += exp_adj_V[i]
    #     end

    #     sum_sum_chid_nest_ttl = sum(sum_chid_nest .^ lambda', dims=2)
    #     display(sum_sum_chid_nest_ttl)

    #     # Calculate choice probabilities for all alternatives
    #     fitted_values_all = [(exp_adj_V[i] * (sum_chid_nest[vec_chid[i], vec_nest[i]+1]^(-1 + lambda[vec_nest[i]+1])) / sum_sum_chid_nest_ttl[vec_chid[i]]) for i in eachindex(V)]

    #     return chosen ? fitted_values_all[vec_choice] : fitted_values_all
    # end
end
StatsAPI.predict(model::MNLmodel; chosen=false) = fitted(model, chosen=chosen)
function StatsAPI.r2(model::MNLmodel, variant::Symbol)
    if variant == :McFadden
        return 1 - StatsBase.loglikelihood(model) / StatsBase.nullloglikelihood(model)
    else
        throw(ArgumentError("variant must be :McFadden"))
    end
end
StatsAPI.r2(model::MNLmodel) = r2(model, :McFadden)
StatsAPI.responsename(model::MNLmodel) = formula(model).lhs
# rss
StatsAPI.score(model::MNLmodel) = model.score

function StatsAPI.stderror(model::MNLmodel; type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    isnothing(model.vcov) && return zeros(length(model.coef))

    if isnothing(type) && isnothing(cluster)
        return sqrt.(diag(model.vcov))
    elseif isnothing(cluster)
        return sqrt.(diag(sandwich(model, type=type)))
    else
        return sqrt.(diag(vcovCL(model, cluster, type=type)))
    end
end

StatsAPI.vcov(model::MNLmodel) = model.vcov

function Base.show(io::IO, m::MNLmodel)
    # Your custom display logic
    println(io, coeftable(m))
    println(io, "Loglikelihood: ", round(loglikelihood(m), digits=4))
    # Add more details as needed
end

# Sandwich: Robust Covariance Matrix Estimators

estfun(model::MNLmodel) = model.estfun

# https://rdrr.io/cran/sandwich/src/R/sandwich.R
# https://stackoverflow.com/questions/66412110/sandwich-mlogit-error-in-ef-x-non-conformable-arrays-when-using-vcovhc


function meat(model::MNLmodel; type::Union{String,Nothing}="HC1")
    psi = estfun(model)
    n, k = size(psi)
    if type == "HC1"
        adjust_term = 1 / (n - k)
        # elseif adjust == "1/(n-1)"
        #     adjust_term = 1/(n-1)
    elseif type == "HC0"
        adjust_term = 1 / n
    else
        error("Unknown type, choose either HC0 or HC1")
    end
    rval = (psi' * psi) .* adjust_term
    return rval
end

function bread(model::MNLmodel)
    return vcov(model) .* size(estfun(model), 1)
end

function sandwich(model::MNLmodel, bread::Matrix{Float64}, meat::Matrix{Float64})
    n = size(estfun(model), 1)
    return 1 / n * (bread * meat * bread)
end
function sandwich(model::MNLmodel; type="HC1")
    n = size(estfun(model), 1)
    b = bread(model)
    m = meat(model, type=type)
    return 1 / n * (b * m * b)
end


function vcovCL(model::MNLmodel, cluster; type="HC1")
    meat = meatCL(model, cluster, type=type)

    return sandwich(model, bread(model), meat)
end

# Helper function to obtain the data frame that is required for the "cluster" argument in meatCL
get_cluster_df(varnames_cluster...; df=df, varname_chid=:chid) = string.(select(unique(df, varname_chid), varnames_cluster...))

# model_mlogit_vcov.nclusters = (; cluster=length(unique(get_cluster_df(:cluster, df=df))))
function robust_cluster_vcov(model::MNLmodel, type, df, varnames_cluster...)
    model_cluster_vcov = deepcopy(model)

    if model.df_hash != hash(df)
        @warn "DataFrame provided to robust_cluster_vcov() is not identical to the DataFrame used for estimation."
    end

    model_cluster_vcov.vcov = vcovCL(model, get_cluster_df(varnames_cluster..., df=df), type=type)
    model_cluster_vcov.nclusters = NamedTuple{Tuple(varnames_cluster)}([length(unique(get_cluster_df(vc, df=df)[!, 1])) for vc in varnames_cluster])
    # not necessary to make RegressionTables show clusters, but FixedEffectModels uses it as well, so just keep it
    model_cluster_vcov.vcov_type = Vcov.cluster(varnames_cluster...)

    # model_cluster_vcov.vcov = vcovCL(model, get_cluster_df(varnames_cluster..., df=df), type=type)
    return model_cluster_vcov
end

# no clustering, only robust
function robust_cluster_vcov(model::MNLmodel, type)
    model_cluster_vcov = deepcopy(model)

    model_cluster_vcov.vcov = sandwich(model, type=type)
    model_cluster_vcov.vcov_type = Vcov.robust()

    # model_cluster_vcov.vcov = vcovCL(model, get_cluster_df(varnames_cluster..., df=df), type=type)
    return model_cluster_vcov
end

function meatCL(model::MNLmodel, cluster; type="HC1", cadjust=true, multi0=false)
    ef = estfun(model)
    n, k = size(ef)
    n_clustervars = size(cluster, 2)
    cluster_interactions = collect(combinations(1:n_clustervars))
    sign = [(-1)^(length(ci) + 1) for ci in cluster_interactions]

    cluster = string.(Matrix(cluster))

    for i in eachindex(cluster_interactions)
        if i > n_clustervars
            cluster = hcat(cluster, [join(row, "_") for row in eachrow(cluster[:, cluster_interactions[i]])])
        end
    end
    unique_cluster_values = [unique(cluster[:, i]) for i in axes(cluster, 2)]
    g = length.(unique_cluster_values)

    rval = zeros(k, k)

    for i in eachindex(cluster_interactions)

        # add cluster adjustment g/(g - 1) or not?
        # only exception: HC0 adjustment for multiway clustering at "last interaction"
        if multi0 && (i == length(cluster_interactions))
            adj = type == "HC1" ? (n - k) / (n - 1) : 1
        else
            adj = cadjust ? g[i] / (g[i] - 1) : 1
        end

        if g[i] < n
            efi = reduce(vcat, [sum(ef[cluster[:, i].==uci, :], dims=1) for uci in unique_cluster_values[i]])
            rval .+= (sign[i] .* adj .* (efi' * efi) ./ n)
        else
            rval .+= (sign[i] .* adj .* (ef' * ef) ./ n)
        end

    end

    if type == "HC1"
        rval .*= (n - 1) / (n - k)
    end

    return rval
end

function coefplot(model::MNLmodel; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    # coefficient plot
    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)
    resolution = (1600, 600)

    coefs = coef(model)

    ci = confint(model, level=level, type=type, cluster=cluster)
    ci_lo = ci[:, 1]
    ci_hi = ci[:, 2]
    significant = vec(sum(sign.(ci), dims=2) .== 0)

    # Create the plot
    fig_coefs = Figure(resolution=resolution)
    ax = Axis(fig_coefs[1, 1], xlabel="Coefficient", ylabel="Variable", yreversed=true)
    # Red line at x=0
    vlines!(ax, [0], color=:red, linewidth=3)
    # Add horizontal lines for confidence intervals
    for i in 1:length(coefs)
        linesegments!(ax, [(ci_lo[i], i), (ci_hi[i], i)],
            color=:black, label=i == 1 ? "Confidence Interval" : nothing)
    end
    # Scatter plot for coefficients
    scatter!(ax, coefs, 1:length(coefs), color=significant, colormap=[:gray60, :black], label="Coefficient", markersize=20)
    # scatter!(ax, coefs, 1:length(coefs), label="Coefficient", markersize=20)
    # Customizing the y-axis to show variable names
    ax.yticks = (1:length(coefs), coefnames(model))
    # Add a legend
    # axislegend(ax)
    fig_coefs[1, 2] = Legend(fig_coefs, ax, framevisible=false)
    # Show the plot
    return fig_coefs
end

function RegressionTables.default_regression_statistics(model::MNLmodel)
    [Nobs, R2McFadden, AdjR2McFadden, LogLikelihood, AIC, BIC]
end