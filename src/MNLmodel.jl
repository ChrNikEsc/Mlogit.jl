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
    coef_dist::AbstractVector
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
    coef_dist::AbstractVector,
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
    return MNLmodel(coef, coefnames, converged, depvar, df_hash, dof, estfun, fitted, formula, formula_origin, formula_schema, hessian, indices, iter, loglikelihood, mixed, coef_dist, nchids, nclusters, nests, nids, nullloglikelihood, optim, score, start, startloglikelihood, time, vcov, vcov_type)
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

function coefplot(model::MNLmodel; level::Real=0.95, type::Union{String,Nothing}=nothing, cluster::Union{DataFrame,Nothing}=nothing, printstandardMNL=true, axissettings=(;))
    # --- 1. Setup the plot theme and figure ---
    fontsize_theme = Theme(fontsize=18)
    set_theme!(fontsize_theme)
    required_lines::Int64 = Base.length(coefnames(model))
    size = (1600, required_lines * 60 + 100) # Adjust height based on number of coefficients + 100 for legend

    fig = Figure(size=size)
    ax = Axis(fig[1, 1], xlabel="Coefficient Value", ylabel="Variable", yreversed=true; axissettings...)
    vlines!(ax, [0], color=:black, linewidth=2) # Global reference line at zero

    # --- 2. Extract data from the model ---
    coefs = coef(model)
    dists = model.coef_dist
    ci = confint(model, level=level, type=type, cluster=cluster)

    # --- 3. Filter to get one entry per variable (for plotting logic) ---
    plot_info = []
    for i in 1:length(dists)
        if dists[i] !== nothing
            push!(plot_info, (dist_or_val=dists[i], original_index=i))
        end
    end
    num_vars = length(plot_info)

    # --- 4. Configure axes with variable names from formula ---
    varnames = if isnothing(model.nests)
        StatsModels.termnames(model.formula.rhs)
    else
        # Assuming in the nested case, the detailed names are more descriptive
        # and we need to identify lambdas from them.
        coefnames(model)
    end
    @assert length(varnames) == num_vars "Mismatch between number of formula names ($(length(varnames))) and plot variables ($num_vars)."

    # --- NEW: Identify which variables are lambda/nesting parameters by name ---
    # Adjust the "lambda" string to match your naming convention (e.g., "tau")
    is_lambda = startswith.(string.(varnames), "lambda")

    ax.yticks = (1:num_vars, string.(varnames))

    # set up trackers for legend
    has_random = false
    has_fixed = false
    has_lambda = false

    # --- 5. Iterate through variables and plot ---
    for (y_pos, data) in enumerate(plot_info)

        # --- CASE A: Random Coefficient (plot density) ---
        if data.dist_or_val[1] isa Distribution
            has_random = true

            dist = data.dist_or_val[1]
            q_low = quantile(dist, 0.001)
            q_high = quantile(dist, 0.999)
            x_range = range(q_low, q_high, length=200)
            y_pdf = pdf.(dist, x_range)
            max_pdf_val = maximum(y_pdf)
            y_pdf_scaled = max_pdf_val > 0 ? (y_pdf / max_pdf_val) * 0.4 : zeros(length(y_pdf))

            band!(ax, x_range, y_pos .- y_pdf_scaled, y_pos .+ y_pdf_scaled, color=(:gray85, 0.8))
            lines!(ax, x_range, y_pos .+ y_pdf_scaled, color=:black, linewidth=1.5)
            lines!(ax, x_range, y_pos .- y_pdf_scaled, color=:black, linewidth=1.5)

            # Plot point estimate of MNL model for reference
            printstandardMNL && scatter!(ax, [data.dist_or_val[2]], [y_pos], marker='o', color=:black, markersize=20)

            # --- CASE B: Fixed Coefficient (plot point and CI) ---
        elseif data.dist_or_val[1] isa Real
            has_fixed = true

            idx = data.original_index
            c = coefs[idx]
            ci_lo = ci[idx, 1]
            ci_hi = ci[idx, 2]

            # --- NEW: Set null value based on whether it's a lambda param ---
            is_current_var_lambda = is_lambda[y_pos]
            null_value = is_current_var_lambda ? 1.0 : 0.0

            # --- NEW: Add a reference marker at x=1 for lambda coefficients ---
            if is_current_var_lambda
                has_lambda = true
                scatter!(ax, [1.0], [y_pos], color=:black, marker='I', markersize=20)
            end

            # --- NEW: Significance test is now against the dynamic null_value ---
            # Insignificant if the confidence interval contains the null value
            is_significant = !(ci_lo <= null_value <= ci_hi)

            point_color = is_significant ? :black : :gray60

            linesegments!(ax, [(ci_lo, y_pos), (ci_hi, y_pos)], color=point_color, linewidth=3)
            scatter!(ax, [c], [y_pos], color=point_color, markersize=20)

            # Plot point estimate of MNL model for reference - also for fixed coefficients as these may be different
            printstandardMNL && has_random && scatter!(ax, [data.dist_or_val[2]], [y_pos], marker='o', color=:black, markersize=20)
        end
    end

    # --- 6. Legend ---
    legend_elements = []
    legend_labels = []

    has_random && begin
        push!(legend_elements, [
            PolyElement(color=:gray85, strokecolor=:black, strokewidth=1.5)
        ])
        push!(legend_labels, "Random Coefficient Density")
    end
    printstandardMNL && has_random && begin
        push!(legend_elements, [
            MarkerElement(color=:black, marker='o', markersize=15)
        ])
        push!(legend_labels, "Coefficient in a Standard MNL Model")
    end
    has_fixed && begin
        push!(legend_elements, [LineElement(color=:gray60, linewidth=3), MarkerElement(color=:gray60, marker=:circle, markersize=20)])
        push!(legend_labels, "Insignificant Fixed Coefficient")
    end
    has_fixed && begin
        push!(legend_elements, [LineElement(color=:black, linewidth=3), MarkerElement(color=:black, marker=:circle, markersize=20)])
        push!(legend_labels, "Significant Fixed Coefficient")
    end
    has_lambda && begin
        push!(legend_elements, MarkerElement(color=:black, marker='I', markersize=20))
        push!(legend_labels, "Lambda H₀ = 1")
    end

    fig[2, 1] = Legend(fig, legend_elements, legend_labels, "Legend", framevisible=false, orientation=:horizontal, tellheight=true)

    return fig
end

function RegressionTables.default_regression_statistics(model::MNLmodel)
    [Nobs, R2McFadden, AdjR2McFadden, LogLikelihood, AIC, BIC]
end

function coef_dist(model::MNLmodel; quantile_levels=[0.05, 0.25, 0.50, 0.75, 0.95])

    data = model.coef_dist
    data = data[data.!=nothing] # Remove Nothings from the data
    results_list = []

    mnl_varnames = StatsModels.termnames(model.formula.rhs)

    for (i, item) in enumerate(data)
        local stats # Ensure stats is local to the loop

        if item[1] isa Distribution
            # For Distribution objects
            dist_mean = round(mean(item[1]), digits=3)
            dist_std = round(std(item[1]), digits=3)
            share_neg = round(cdf(item[1], 0.0), digits=3)
            quants = [round(quantile(item[1], q), digits=3) for q in quantile_levels]
            var_name = "$(mnl_varnames[i]) ($(typeof(item[1]).name.name))"

            stats = (
                Variable=var_name,
                Mean=dist_mean,
                MNL=item[2],
                StdDev=dist_std,
                ShareBelowZero=share_neg,
                Quantiles=quants
            )

        elseif item[1] isa Real
            # For Float64 or other real numbers (treated as a point mass)
            var_name = "$(mnl_varnames[i])"
            # A single number has no standard deviation, and its mean is itself.
            # Quantiles are all equal to the number.
            stats = (
                Variable=var_name,
                MNL=item[2],
                Mean=round(item[1], digits=3),
                StdDev=0.0,
                ShareBelowZero=item[1] < 0 ? 1.0 : 0.0,
                Quantiles=[round(item[1], digits=3) for q in quantile_levels]
            )
        else
            # Handle other potential types if necessary
            @warn "Item at index $i of type $(typeof(item[1])) is not supported and will be skipped."
            continue
        end

        push!(results_list, stats)
    end

    # 5. Create the DataFrame from the list of results
    results = DataFrame(results_list)

    # 6. Format the final DataFrame for better presentation
    # Unpack the quantiles array into separate columns.
    quantile_names = ["Q_$(Int(q*100))" for q in quantile_levels]
    for (i, name) in enumerate(quantile_names)
        results[!, name] = [q[i] for q in results.Quantiles]
    end

    # Select and reorder columns for the final output
    final_df = select(results, :Variable, :MNL, :Mean, :StdDev, :ShareBelowZero, Symbol.(quantile_names)...)

    return final_df
end