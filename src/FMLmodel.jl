mutable struct FMLsubmodel <: RegressionModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    formula::FormulaTerm
    responsename::String
    vcov::Union{Matrix{Float64},Nothing}
    vcov_type::Vcov.CovarianceEstimator
end

function FMLsubmodel(;
    coef::Vector{Float64},
    coefnames::Vector{String},
    formula::FormulaTerm,
    responsename::String,
    vcov::Union{Matrix{Float64},Nothing},
    vcov_type::Vcov.CovarianceEstimator
    )
    return FMLsubmodel(coef, coefnames, formula, responsename, vcov, vcov_type)
end

function RegressionTables._coefnames(x::FMLsubmodel)
    out = x.coefnames
    if !isa(out, AbstractVector)
        out = [out]
    end
    out
end

mutable struct FMLmodel <: RegressionModel
    coef::Matrix{Float64}
    coefnames::Matrix{String}
    converged::Union{Bool,Nothing}
    depvar::Matrix{Float64}
    df_hash::UInt64
    dof::Int64
    estfun::Union{Matrix{Float64},Nothing}
    fitted::Union{Vector{Float64}, Matrix{Float64}}
    formula::FormulaTerm
    formula_schema::FormulaTerm
    hessian::Union{Matrix{Float64},Nothing}
    hessian_method::Union{String,Nothing}
    indices::Union{XlogitIndices,Nothing}
    iter::Union{Int64,Nothing}
    loglikelihood::Float64
    nchids::Int64
    nclusters::Union{NamedTuple,Nothing}
    nids::Int64
    nullloglikelihood::Float64
    optim::Union{Optim.OptimizationResults,Nothing}
    score::Vector{Float64}
    submodels::Vector{FMLsubmodel}
    start::Matrix{Float64}
    startloglikelihood::Float64
    time::Float64
    vcov::Union{Matrix{Float64},Nothing}
    vcov_type::Vcov.CovarianceEstimator
end

function FMLmodel(;
    coef::Matrix{Float64},
    coefnames::Matrix{String},
    converged::Union{Bool,Nothing}=nothing,
    depvar::Matrix{Float64},
    df_hash::UInt,
    dof::Int64,
    estfun::Union{Matrix{Float64},Nothing},
    fitted::Union{Vector{Float64}, Matrix{Float64}},
    formula::FormulaTerm,
    formula_schema::FormulaTerm,
    hessian::Union{Matrix{Float64},Nothing}=nothing,
    hessian_method::Union{String,Nothing}=nothing,
    indices::Union{XlogitIndices,Nothing},
    iter::Union{Int64,Nothing}=nothing,
    loglikelihood::Float64,
    nchids::Int64,
    nclusters::Union{NamedTuple,Nothing},
    nids::Int64,
    nullloglikelihood::Float64,
    optim::Union{Optim.OptimizationResults,Nothing}=nothing,
    score::Vector{Float64},
    submodels::Vector{FMLsubmodel},
    start::Matrix{Float64},
    startloglikelihood::Float64,
    time::Float64,
    vcov::Union{Matrix{Float64},Nothing}=nothing,
    vcov_type::Vcov.CovarianceEstimator)
    return FMLmodel(coef, coefnames, converged, depvar, df_hash, dof, estfun, fitted, formula, formula_schema, hessian, hessian_method, indices, iter, loglikelihood, nchids, nclusters, nids, nullloglikelihood, optim, score, submodels, start, startloglikelihood, time, vcov, vcov_type)
end

function prepare_data_fmlogit(formula, df)
    s = schema(formula, df)

    formula_schema = apply_schema(formula, s)
    y = convert(Matrix{Float64}, hcat(response(formula_schema, df)...))
    X = convert(Matrix{Float64}, modelmatrix(formula_schema, df))
    varnames_y, varnames_X = coefnames(formula_schema)
    n::Int64 = nrow(df)
    j::Int64 = size(y, 2)
    k::Int64 = size(X, 2)
    X = [X ones(n)]
    (sum(isapprox.(sum(y, dims=2), 1)) == n) || error("Not all rows of y sum to 1")

    return formula_schema, y, X, varnames_y, varnames_X, n, j, k
end

function compute_fitted_values_fmlogit(betas, k, j, X)
    # Reshape the beta coefficients and prepare the betamat matrix
    betamat = [transpose(reshape(betas, k + 1, j - 1)); zeros(k + 1)']
    
    # Compute X * betamat' for the linear predictor (eta)
    exp_eta = exp.(X * transpose(betamat))
    
    # Use the softmax function to calculate probabilities (fitted values)
    sum_exp_eta = sum(exp_eta, dims=2)  # Sum across the categories (columns)
    
    # Fitted values are the normalized probabilities for each observation and category
    fitted_values = exp_eta ./ sum_exp_eta
    
    return fitted_values
end

function coef_mat(model::FMLmodel)
    betas = coef(model)

    k, j = length.(coefnames(model.formula_schema))
    return permutedims([transpose(reshape(model.coef, k + 1, j - 1)); zeros(k + 1)'])
end

function StatsAPI.adjr2(model::FMLmodel, variant::Symbol)
    if variant == :McFadden
        return 1 - (loglikelihood(model) - dof(model)) / nullloglikelihood(model)
    else
        throw(ArgumentError("variant must be :McFadden"))
    end
end
StatsAPI.adjr2(model::FMLmodel) = adjr2(model, :McFadden)
StatsAPI.aic(model::FMLmodel) = -2 * loglikelihood(model) + dof(model) * 2
StatsAPI.aicc(model::FMLmodel) = -2 * loglikelihood(model) + 2 * dof(model) + 2 * dof(model) * (dof(model) - 1) / (nobs(model) - dof(model) - 1)
StatsAPI.bic(model::FMLmodel) = -2 * loglikelihood(model) + dof(model) * log(nobs(model))
caic(model::FMLmodel) = -2 * loglikelihood(model) + dof(model) * (log(nobs(model)) + 1)
StatsAPI.coef(model::Union{FMLmodel, FMLsubmodel}) = vec(model.coef)
StatsAPI.coefnames(model::Union{FMLmodel, FMLsubmodel}) = vec(model.coefnames)
function confint(model::Union{FMLmodel, FMLsubmodel}; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    hcat(coef(model), coef(model)) + stderror(model, type=type, cluster=cluster) * quantile(Normal(), (1.0 - level) / 2.0) * [1.0 -1.0]
end
function StatsAPI.coeftable(model::Union{FMLmodel, FMLsubmodel}; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    cc = coef(model)
    se = stderror(model, type=type, cluster=cluster)
    zz0 = cc ./ se
    # p0 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz0)) #same as, but faster than: p0 = 2.0 * ccdf.(TDist(dof_residual(model)), abs.(zz0))
    p0 = 2.0 * ccdf.(Normal(), abs.(zz0))

    zz1 = (cc .- 1) ./ se
    p1 = 2.0 * ccdf.(Normal(), abs.(zz1))
    # p1 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz1))
    return CoefTable(hcat(cc, se, zz0, p0, zz1, p1, confint(model; level=level, type=type, cluster=cluster)),
        ["Estimate", "Std.Error", "z0 value", "Pr(>|z0|)", "z1 value", "Pr(>|z1|)", "Conf.Low $((level * 100))%", "Conf.High $((level * 100))%"],
        coefnames(model), 4)
end
# TODO clustering hier verstehen und ggf. mit robust_... einheitlich machen
# deviance
StatsAPI.dof(model::FMLmodel) = model.dof
StatsAPI.dof_residual(model::FMLmodel) = nobs(model) - dof(model)
StatsAPI.dof_residual(model::FMLsubmodel) = 1
# fit
# fit!
StatsAPI.fitted(model::FMLmodel) = model.fitted
StatsModels.formula(model::Union{FMLmodel, FMLsubmodel}) = model.formula
function informationmatrix(model::FMLmodel; expected::Bool=true)
    if expected
        @warn("Fisher (expected) information matrix not implemented. Returning observed information matrix.")
        return model.hessian
    else
        return model.hessian
    end
end
StatsAPI.isfitted(model::FMLmodel) = model.converged
StatsAPI.islinear(model::Union{FMLmodel, FMLsubmodel}) = false
StatsAPI.loglikelihood(model::FMLmodel) = model.loglikelihood
# mss
StatsAPI.nobs(model::FMLmodel; use_nids::Bool=false) = use_nids ? model.nids : model.nchids
# nulldeviance
StatsAPI.nullloglikelihood(model::FMLmodel) = model.nullloglikelihood
function StatsAPI.predict(model::FMLmodel, newDf::DataFrame)
    _, _, X, _, _, _, j, k = prepare_data_fmlogit(model.formula, newDf)
    return compute_fitted_values_fmlogit(coef(model), k, j, X)
end
StatsAPI.predict(model::FMLmodel) = fitted(model)
function StatsAPI.r2(model::FMLmodel, variant::Symbol)
    if variant == :McFadden
        return 1 - StatsBase.loglikelihood(model) / StatsBase.nullloglikelihood(model)
    else
        throw(ArgumentError("variant must be :McFadden"))
    end
end
StatsAPI.r2(model::FMLmodel) = r2(model, :McFadden)
StatsAPI.responsename(model::FMLmodel) = ""
# rss
StatsAPI.score(model::FMLmodel) = model.score

function StatsAPI.stderror(model::Union{FMLmodel, FMLsubmodel}; type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
    isnothing(model.vcov) && return zeros(length(vec(coef(model))))

    if isnothing(type) && isnothing(cluster)
        return [sqrt.(diag(model.vcov)); zeros(size(model.coef, 1))]
    elseif isnothing(cluster)
        return [sqrt.(diag(sandwich(model, type=type))); zeros(size(model.coef, 1))]
    else
        return [sqrt.(diag(vcovCL(model, cluster, type=type))); zeros(size(model.coef, 1))]
    end
end

StatsAPI.vcov(model::FMLmodel) = model.vcov

function Base.show(io::IO, m::FMLmodel)
    # Your custom display logic
    println(io, coeftable(m))
    println(io, "Loglikelihood: ", round(loglikelihood(m), digits=4))
    # Add more details as needed
end

# Sandwich: Robust Covariance Matrix Estimators

estfun(model::FMLmodel) = model.estfun

# https://rdrr.io/cran/sandwich/src/R/sandwich.R
# https://stackoverflow.com/questions/66412110/sandwich-mlogit-error-in-ef-x-non-conformable-arrays-when-using-vcovhc


function meat(model::FMLmodel; type::Union{String,Nothing}="HC1")
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

function bread(model::FMLmodel)
    return vcov(model) .* size(estfun(model), 1)
end

function sandwich(model::FMLmodel, bread::Matrix{Float64}, meat::Matrix{Float64})
    n = size(estfun(model), 1)
    return 1 / n * (bread * meat * bread)
end
function sandwich(model::FMLmodel; type="HC1")
    n = size(estfun(model), 1)
    b = bread(model)
    m = meat(model, type=type)
    return 1 / n * (b * m * b)
end


function vcovCL(model::FMLmodel, cluster; type="HC1")
    meat = meatCL(model, cluster, type=type)

    return sandwich(model, bread(model), meat)
end

# Helper function to obtain the data frame that is required for the "cluster" argument in meatCL
# get_cluster_df(varnames_cluster...; df=df, varname_chid=:chid) = string.(select(unique(df, varname_chid), varnames_cluster...))

# model_mlogit_vcov.nclusters = (; cluster=length(unique(get_cluster_df(:cluster, df=df))))
function robust_cluster_vcov(model::FMLmodel, type, df, varnames_cluster...)
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
function robust_cluster_vcov(model::FMLmodel, type)
    model_cluster_vcov = deepcopy(model)

    model_cluster_vcov.vcov = sandwich(model, type=type)
    model_cluster_vcov.vcov_type = Vcov.robust()

    # model_cluster_vcov.vcov = vcovCL(model, get_cluster_df(varnames_cluster..., df=df), type=type)
    return model_cluster_vcov
end

function meatCL(model::FMLmodel, cluster; type="HC1", cadjust=true, multi0=false)
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

function coefplot(model::FMLmodel; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing)
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

function RegressionTables.regtable(model::FMLmodel)
    j = length(model.submodels)
    regtable(model.submodels..., number_regressions=false, extralines = [
        DataRow(["N", nobs(model) => 2:(j+1)], align="lc"),
        DataRow(["Pseudo R2", PseudoR2(model) => 2:(j+1)], align="lc"),
        DataRow(["Pseudo Adjusted R2", adjr2(model) => 2:(j+1)], align="lc"),
        DataRow(["Log Likelihood", LogLikelihood(model) => 2:(j+1)], align="lc"),
        DataRow(["AIC", AIC(model) => 2:(j+1)], align="lc"),
        DataRow(["BIC", BIC(model) => 2:(j+1)], align="lc"),
    ],)
end

function RegressionTables.default_regression_statistics(model::FMLsubmodel)
    []
end

