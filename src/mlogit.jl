function mlogit(
    formula::FormulaTerm,
    df;
    vcov_type::CovarianceEstimator=Vcov.simple(),
    weights::Union{Symbol,Nothing}=nothing,
    start::Union{Nothing,Vector{Float64}}=nothing,
    indices::XlogitIndices=xlogit_indices(),
    # specific to nested logit
    equal_lambdas::Bool=false,
    # specifitic to mixed logit
    randdist::Union{Nothing,Vector{Union{Nothing,Symbol}}}=nothing,
    draws::Tuple{Int64,Union{Symbol,String}}=(100, :MLHS),
    optim_options=Optim.Options()
)
    StatsAPI.fit(MNLmodel, formula, df; vcov_type=vcov_type, weights=weights, start=start, indices=indices, equal_lambdas=equal_lambdas, randdist=randdist, draws=draws, optim_options=optim_options)
end

function StatsAPI.fit(::Type{MNLmodel},
    formula::FormulaTerm,
    df;
    vcov_type::CovarianceEstimator=Vcov.simple(),
    weights::Union{Symbol,Nothing}=nothing,
    start::Union{Nothing,Vector{Float64}}=nothing,
    indices::XlogitIndices=xlogit_indices(),
    # specific to nested logit
    equal_lambdas::Bool=false,
    # specifitic to mixed logit
    randdist::Union{Nothing,Vector{Union{Nothing,Symbol}}}=nothing,
    draws::Tuple{Int64,Union{Symbol,String}}=(100, :MLHS),
    optim_options=Optim.Options()
)

    mixed::Bool = !isnothing(randdist)

    if vcov_type != Vcov.simple()
        throw("Other types of covariance estimators than Vcov.simple() are not yet implemented. Use robust_cluster_vcov() post-estimation.")
    end

    start_time = time()

    df = DataFrame(df; copycols=false)
    nrows::Int64 = size(df, 1)

    mat_X, vec_choice, randdist, vec_id, vec_chid, vec_weights_chid, vec_nests, coef_start, coef_names, n_coefficients, n_id, n_chid, nested, formula, formula_origin, formula_schema =
        prepare_mlogit_inputs(formula, df, indices, weights, start, equal_lambdas, randdist)

    if !nested & !mixed
        opt, coefficients, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values = fit_mlogit(mat_X, vec_choice, coef_start, vec_chid, vec_weights_chid; optim_options=optim_options)
    elseif nested & !mixed
        opt, coefficients, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values, nests = fit_mlogit(mat_X, vec_choice, coef_start, vec_chid, vec_weights_chid, vec_nests, equal_lambdas; optim_options=optim_options)
    elseif mixed & !nested
        @warn("Ignoring Optim.Options() if provided. Must ensure that extended_trace=true and store_trace=true.")
        opt, coefficients, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values = fit_mlogit(mat_X, vec_choice, randdist, coef_start, vec_id, vec_chid, vec_weights_chid, draws)
    else
        error("Mixed logit with nests is not implemented.")
    end

    vcov = inv(hessian)

    r = MNLmodel(
        coef=coefficients,
        coefnames=coef_names,
        converged=converged,
        depvar=vec_choice,
        df_hash=hash(df),
        dof=n_coefficients,
        estfun=estfun,
        fitted=fitted_values,
        formula=formula,
        formula_origin=formula_origin,
        formula_schema=formula_schema,
        hessian=hessian,
        indices=indices,
        iter=iter,
        loglikelihood=loglik,
        mixed=false,
        nclusters=nothing,
        nchids=n_chid,
        nests=nested ? nests : nothing,
        nids=n_id,
        nullloglikelihood=loglik_0,
        optim=opt,
        score=gradient,
        start=coef_start,
        startloglikelihood=loglik_start,
        time=time() - start_time,
        vcov=vcov,
        vcov_type=vcov_type
    )

    return r
end

function prepare_mlogit_inputs(formula::FormulaTerm, df, indices::XlogitIndices,
    weights::Union{Symbol,Nothing}, start::Union{Nothing,Vector{Float64}},
    equal_lambdas::Bool,
    randdist::Union{Nothing,Vector{Union{Nothing,Symbol}}})# TODO: Should be inferred from formula later

    formula_origin = formula
    formula, formula_nests = parse_nests(formula_origin)
    s = schema(formula, df)
    formula_schema = apply_schema(formula, s)
    vec_choice::BitVector, mat_X::Matrix{Float64} = StatsModels.modelcols(formula_schema, df)
    response_name::String, coefnames_utility::Vector{String} = coefnames(formula_schema)
    nested::Bool = formula_nests != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    mixed::Bool = !isnothing(randdist)
    vec_nests::Vector{String} = nested ? df[:, nestssymbol(only(formula_nests.rhs))] : repeat(["x"], inner=Base.length(vec_choice))

    coefnames_nests::Vector{String} = if !nested
        []
    elseif equal_lambdas
        ["lambda"]
    else
        ["lambda_$k" for k in filter(x -> !ismissing(x) && !isnothing(x) && x != 0, unique(vec_nests))]
    end

    coef_names::Vector{String} = if nested && !mixed
        vcat(coefnames_utility, coefnames_nests)
    elseif !nested && !mixed
        coefnames_utility
    elseif !nested && mixed
        vcat(
            coefnames_utility[randdist.==nothing],
            ["b_" * string(randdist[idx]) * "_" * string(coefnames_utility[idx]) for idx in eachindex(randdist) if !isnothing(randdist[idx])],
            ["w_" * string(randdist[idx]) * "_" * string(coefnames_utility[idx]) for idx in eachindex(randdist) if !isnothing(randdist[idx])]
        )
    else
        error("Mixed logit with nests is not implemented.")
    end
    n_coefficients_util::Int64 = Base.length(coefnames_utility)
    n_coefficients::Int64 = Base.length(coef_names)
    n_id::Int64 = Base.length(unique(df[!, indices.id]))

    vec_id::Vector{Int64} = df[!, indices.id]
    remap_to_indices_chid!(vec_id)

    vec_chid::Vector{Int64} = df[!, indices.chid]
    remap_to_indices_chid!(vec_chid)
    n_chid::Int64 = Base.length(unique(vec_chid))

    vec_weights::Vector{Float64} = isnothing(weights) ?
                                   ones(Float64, Base.length(vec_choice)) :
                                   let w::Vector{Float64} = df[!, weights]
        (w ./ sum(w)) .* Base.length(vec_choice)
    end
    vec_weights_chid = vec_weights[vec_choice]

    coef_start::Vector{Float64} = if isnothing(start)
        if !nested && !mixed
            zeros(Float64, n_coefficients_util)
        elseif nested && !mixed
            [zeros(Float64, n_coefficients_util); ones(Float64, equal_lambdas + !equal_lambdas * Base.length(unique(vec_nests)))]
        elseif !nested && mixed
            vcat(
            zeros(Float64, length(coefnames_utility[randdist.==nothing])),
            [1 * rand(1)[1] - 1 for idx in eachindex(randdist) if !isnothing(randdist[idx])], # b random coefs, with Train's sample data, this produced very good results in the majority of tries. DOWNSIDE: have two make few tries and then take the best model
            # [0.0 for idx in eachindex(randdist) if !isnothing(randdist[idx])], # b random coefs
            [0.01 for idx in eachindex(randdist) if !isnothing(randdist[idx])] # w random coefs
        )
        else
            zeros(Float64, n_coefficients)
        end
    else
        start
    end
# display(coef_start)
    return mat_X, vec_choice, randdist, vec_id, vec_chid, vec_weights_chid, vec_nests, coef_start, coef_names, n_coefficients, n_id, n_chid, nested, formula, formula_origin, formula_schema
end

# No nests
function fit_mlogit(mat_X::Matrix{Float64}, vec_choice::BitVector, coef_start::Vector{Float64}, vec_chid::Vector{Int64}, vec_weights_chid::Vector{Float64}; method=Newton(), optim_options=Optim.Options())

    tmp_mul = similar(vec_chid, Float64, size(mat_X, 1)) # Temporary storage for X*theta

    n_chid::Int64 = Base.length(unique(vec_chid)) # Number of unique choice sets
    idx_map = create_index_map(vec_chid) # Maps choice set ID to indices of alternatives in mat_X
    n_coefficients::Int64 = Base.length(coef_start) # Number of parameters

    # Initialize objects used within fgh! - these will be repurposed for stable calculations
    # exb will be repurposed to store log(Pni). Initial value doesn't matter as it's overwritten.
    log_Pni::Vector{Float64} = similar(tmp_mul) # Original initialization: exp.(mat_X * coef_start) - removing to avoid redundant exp
    # sexb will be repurposed to store log-denominators per choice set.
    log_denom::Vector{Float64} = zeros(Float64, n_chid) # Original initialization
    # Pni will be repurposed to store stable probabilities.
    Pni::Vector{Float64} = zeros(Float64, Base.length(vec_chid)) # Original initialization

    # Preallocate for terms used in Gradient and Hessian calculations - keep original usage
    Px::Matrix{Float64} = zeros(Float64, Base.length(coef_start), n_chid)
    yx::Matrix{Float64} = zeros(Float64, Base.length(coef_start), n_chid)
    gradi::Matrix{Float64} = zeros(Float64, n_chid, Base.length(coef_start)) # Dimensions based on original code's usage

    # Preallocate for terms used in Hessian calculation - keep original usage
    dxpx::Matrix{Float64} = zeros(Float64, Base.length(vec_chid), Base.length(coef_start))
    dxpx_Pni::Matrix{Float64} = similar(dxpx)

    # Precompute unique choice set IDs (used in the choice set loop within fgh!)
    unique_chids = unique(vec_chid)

    function fgh!(F, G, H, theta::Vector{Float64})
        # Common computations within the optimization step

        # Compute log-utilities (X*theta) - store in tmp_mul as in original pattern
        mul!(tmp_mul, mat_X, theta) # in-place multiplication, tmp_mul now holds X*theta (log-utilities)

        @inbounds for c in unique_chids # unique_chids is available outside fgh!
            # Get the global indices of alternatives belonging to this choice set
            alt_indices = idx_map[c] # idx_map is available outside fgh!

            # Get the log-utilities for alternatives in this choice set (from tmp_mul)
            log_util_set = @view tmp_mul[alt_indices]

            # Compute the log-denominator for this choice set using logsumexp for numerical stability
            log_denom[c] = LogExpFunctions.logsumexp(log_util_set)
        end

        log_Pni .= tmp_mul .- log_denom[vec_chid] # exb now stores log(Pni) for each alternative

        Pni .= exp.(log_Pni)

        if !(isnothing(G) && isnothing(H))
            # If G or H is required by the optimizer, calculate terms needed for both.
            fill!(Px, zero(eltype(theta))) # Px[j, c] = sum over alternatives i in set c of Pni[i] * mat_X[i, j]
            fill!(yx, zero(eltype(theta))) # yx[j, c] = sum over chosen alternatives i in set c of mat_X[i, j]

            @inbounds for i in eachindex(vec_chid), j in eachindex(theta)
                # Accumulate contributions per choice set (indexed by vec_chid[i])
                Px[j, vec_chid[i]] += Pni[i] * mat_X[i, j]
                yx[j, vec_chid[i]] += vec_choice[i] * mat_X[i, j] # vec_choice[i] is 1 if chosen, 0 otherwise
            end

            # Original dxpx and dxpx_Pni calculation using Px and Pni
            dxpx .= zero(eltype(theta)) # dimensions num_alternatives x num_coefficients
            dxpx_Pni .= zero(eltype(theta)) # dimensions num_alternatives x num_coefficients

            @inbounds for i in eachindex(vec_chid), j in eachindex(theta)
                dxpx[i, j] = mat_X[i, j] - Px[j, vec_chid[i]]
                dxpx_Pni[i, j] = dxpx[i, j] * Pni[i]
            end

            @inbounds for c in 1:n_chid
                @inbounds for j in eachindex(theta)
                    gradi[c, j] = (yx[j, c] - Px[j, c]) * vec_weights_chid[c] # Use stable Px and yx, assume vec_weights_chid[c] is weight for set c
                end
            end

        end # End if !(isnothing(G) && isnothing(H))


        if G !== nothing
            # gradi has dimensions (n_chid, num_coefficients). Sum over dim 1 results in (1, num_coefficients).
            G .= -vec(sum(gradi, dims=1)) # Apply original negative sign.
        end

        if H !== nothing
            H .= zero(eltype(theta)) # dimensions num_coefficients x num_coefficients
            # Assuming vec_weights_chid weights choice sets based on original H loop index 'i' (which corresponds to chid).
            @inbounds @simd for c in 1:n_chid
                # Assuming vec_weights_chid[c] is the weight for choice set c
                weight_c = vec_weights_chid[c] # This implies weights are per choice set.
                # Get the indices for alternatives in choice set c
                alt_indices_in_set = idx_map[c]
                # Get the relevant parts of dxpx_Pni and dxpx for this choice set
                dxpx_Pni_set = @view dxpx_Pni[alt_indices_in_set, :] # Matrix for set c (num_alts_in_set x num_coefs)
                dxpx_set = @view dxpx[alt_indices_in_set, :] # Matrix for set c (num_alts_in_set x num_coefs)

                # The product sum for the Hessian contribution of set c is dxpx_Pni_set' * dxpx_set
                H .+= weight_c * dxpx_Pni_set' * dxpx_set # Accumulate weighted Hessian contribution per choice set
            end
        end

        if F !== nothing
            # Sum over chosen alternatives i of - vec_weights_chid[i] * log_Pni[i]
            neg_log_lik = -sum(vec_weights_chid .* log_Pni[vec_choice])

            return neg_log_lik
        end
    end

    # The outer fit_mlogit function calls the optimizer with the fgh! closure.
    # Your existing code calls Optim.only_fgh! with fgh!
    opt = Optim.optimize(Optim.only_fgh!(fgh!), coef_start::Vector{Float64}, method, optim_options)

    # --- Extract results using the stable fgh! ---

    coefficients_scaled::Vector{Float64} = Optim.minimizer(opt)
    converged::Bool = Optim.converged(opt)
    iter::Int64 = Optim.iterations(opt)

    # Compute loglik using the stable fgh! at the converged parameters
    # fgh! called with F = 1 requests the function value.
    loglik::Float64 = -fgh!(1, nothing, nothing, coefficients_scaled)

    # Compute loglik_0 at zero parameters using the stable fgh!
    loglik_0::Float64 = -fgh!(1, nothing, nothing, zeros(n_coefficients))

    # Compute loglik_start at start parameters using the stable fgh!
    loglik_start::Float64 = -fgh!(1, nothing, nothing, coef_start)


    # Compute gradient and Hessian at converged parameters using the stable fgh!
    # Call fgh! requesting G and H to populate gradient, hessian, and gradi.
    gradient::Vector{Float64} = similar(coefficients_scaled)
    hessian::Matrix{Float64} = Matrix{Float64}(undef, n_coefficients, n_coefficients)
    fgh!(nothing, gradient, hessian, coefficients_scaled) # Request G and H

    # gradi is populated by the fgh! call above (when G or H is requested).
    estfun::Matrix{Float64} = gradi # gradi has dims (n_chid, n_coefficients)

    # fitted_values is the stable probabilities Pni at the converged parameters.
    # Pni was populated by the last fgh! call (above, to get G and H).
    fitted_values::Vector{Float64} = Pni


    return opt, coefficients_scaled, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values
end

# With nests
function fit_mlogit(mat_X::Matrix{Float64}, vec_choice::BitVector, coef_start::Vector{Float64}, vec_chid::Vector{Int64}, vec_weights_chid, vec_nests, equal_lambdas; method=LBFGS(linesearch=LineSearches.BackTracking()), optim_options=Optim.Options())

    # Nests
    # transform vec_nest such that 0 means an alternative is in its own nest (don't estimate lambda)..
    # ..and other nests are re-labelled with integers for indexing in loglik_fun
    vec_nests_indices = remap_to_indices_nest(vec_nests)
    vec_nests_indices_choice = vec_nests_indices[vec_choice]
    n_nests = maximum(vec_nests_indices)  # Number of nests (excluding the outside option)
    nests = Dict(zip(unique(vec_nests), unique(vec_nests_indices)))

    n_chid = Base.length(unique(vec_chid))
    idx_map = create_index_map(vec_chid)
    n_coefficients = Base.length(coef_start)
    n_coefficients_utility = n_coefficients - !equal_lambdas * n_nests - equal_lambdas

    mat_X_choice = mat_X[vec_choice, :]

    # Initialize objects in fg!
    beta::Vector{Float64} = coef_start[1:n_coefficients_utility]
    lambda = ones(n_nests + 1)
    V = mat_X * beta
    sum_chid_nest = zeros(n_chid, n_nests + 1)
    choice_prob = zeros(n_chid)
    gradi = zeros(Float64, n_chid, n_coefficients_utility + !equal_lambdas * n_nests + equal_lambdas)
    gradi_lambda = zeros(Float64, n_chid, n_nests)

    function fg!(F, G, theta::Vector{Float64})
        beta = theta[1:n_coefficients_utility]
        if equal_lambdas
            lambda[2:end] .= theta[end]
        else
            lambda[2:end] .= theta[n_coefficients_utility+1:end]
        end

        lambda_choice = [lambda[i+1] for i in vec_nests_indices_choice]

        V .= mat_X * beta
        V_choice = V[vec_choice]
        exp_adj_V = exp.([V[i] / lambda[vec_nests_indices[i]+1] for i in eachindex(V)])

        exp_adj_V = ifelse.(isinf.(exp_adj_V), 1e6, exp_adj_V)
        if any(isinf, exp_adj_V)
            display([V[i] / lambda[vec_nests_indices[i]+1] for i in eachindex(V)])
            throw("")
        end

        exp_adj_V_choice = exp_adj_V[vec_choice]

        fill!(sum_chid_nest, zero(eltype(theta)))
        @inbounds for i in eachindex(vec_chid)
            sum_chid_nest[vec_chid[i], vec_nests_indices[i]+1] += exp_adj_V[i]
        end
        sum_chid_nest = ifelse.(isinf.(sum_chid_nest), 1e6, sum_chid_nest)
        vec_sum_chid_nest = [sum_chid_nest[i, vec_nests_indices_choice[i]+1] for i in eachindex(V_choice)]


        sum_sum_chid_nest_ttl = sum(sum_chid_nest .^ lambda', dims=2)

        # choice_prob .= (exp_adj_V_choice .* vec_sum_chid_nest .^ (-1 .+ lambda_choice)) ./ sum_sum_chid_nest_ttl
        choice_prob .= (exp_adj_V_choice .* safe_exp.(vec_sum_chid_nest, -1 .+ lambda_choice)) ./ sum_sum_chid_nest_ttl

        if G !== nothing
            fill!(gradi, zero(eltype(theta)))
            fill!(gradi_lambda, zero(eltype(theta)))
            fill!(G, zero(eltype(theta)))

            # Precompute things
            precomputed_multiplication = exp.(-V_choice ./ lambda_choice) .* safe_exp.(vec_sum_chid_nest, 1 .- lambda_choice) .* sum_sum_chid_nest_ttl

            exp_adj_V_times_mat_X_nestsums = zeros(eltype(theta), n_chid, n_nests + 1, Base.length(beta))
            @inbounds for i in eachindex(vec_chid), b in eachindex(beta)
                exp_adj_V_times_mat_X_nestsums[vec_chid[i], vec_nests_indices[i]+1, b] += exp_adj_V[i] * mat_X[i, b]
            end

            exp_adj_V_times_V_nestsums = zeros(eltype(theta), n_chid, n_nests + 1)
            @inbounds for i in eachindex(vec_chid)
                exp_adj_V_times_V_nestsums[vec_chid[i], vec_nests_indices[i]+1] += exp_adj_V[i] * V[i]
            end

            part1_beta = (choice_prob .* mat_X_choice) ./ lambda_choice

            @inbounds for j in 1:n_chid
                chosen_nest_j = vec_nests_indices_choice[j] + 1

                # Compute gradient for betas
                @inbounds for g in eachindex(beta)
                    tmp_g = part1_beta[j, g]
                    @inbounds for n in 1:n_nests+1
                        if chosen_nest_j == n
                            tmp_g += (exp_adj_V_choice[j] * (-1 + lambda_choice[j]) * safe_exp(vec_sum_chid_nest[j], -2 + lambda_choice[j]) * exp_adj_V_times_mat_X_nestsums[j, n, g]) / (lambda_choice[j] * sum_sum_chid_nest_ttl[j])
                            tmp_g -= (exp_adj_V_choice[j] * safe_exp(vec_sum_chid_nest[j], -1 + lambda_choice[j]) * (safe_exp(vec_sum_chid_nest[j], -1 + lambda_choice[j]) * exp_adj_V_times_mat_X_nestsums[j, n, g])) / (sum_sum_chid_nest_ttl[j])^2
                        else
                            tmp_g -= safe_div(exp_adj_V_choice[j] * safe_exp(vec_sum_chid_nest[j], -1 + lambda_choice[j]) * safe_exp(sum_chid_nest[j, n], (-1 + lambda[n])) * exp_adj_V_times_mat_X_nestsums[j, n, g], (sum_sum_chid_nest_ttl[j])^2)
                        end
                    end

                    gradi[j, g] += vec_weights_chid[j] * tmp_g * precomputed_multiplication[j]
                end

                # Compute gradient for lambdas
                @inbounds for n in 2:n_nests+1
                    g = n - 1
                    if vec_nests_indices_choice[j] + 1 == n
                        tmp_g = -(choice_prob[j] * V_choice[j]) / lambda[n]^2
                        tmp_g += choice_prob[j] * (-((-1 + lambda[n]) * exp_adj_V_times_V_nestsums[j, n]) / (lambda[n]^2 * vec_sum_chid_nest[j]) + safe_log(vec_sum_chid_nest[j]))
                        tmp_g += (exp_adj_V_choice[j] * safe_exp(vec_sum_chid_nest[j], -2 + 2 * lambda[n]) * (exp_adj_V_times_V_nestsums[j, n] - lambda[n] * vec_sum_chid_nest[j] * safe_log(vec_sum_chid_nest[j]))) / (lambda[n] * sum_sum_chid_nest_ttl[j]^2)
                        gradi_lambda[j, g] += vec_weights_chid[j] * tmp_g * precomputed_multiplication[j]
                    else
                        # gradi_lambda[j, g] += vec_weights_chid[j] * (sum_chid_nest[j, n]^(-1 + lambda[n]) * (exp_adj_V_times_V_nestsums[j, n] - lambda[n] * sum_chid_nest[j, n] * safe_log(sum_chid_nest[j, n]))) / (lambda[n] * sum_sum_chid_nest_ttl[j])
                        gradi_lambda[j, g] += vec_weights_chid[j] * (safe_exp(sum_chid_nest[j, n], -1 + lambda[n]) * (exp_adj_V_times_V_nestsums[j, n] - lambda[n] * sum_chid_nest[j, n] * safe_log(sum_chid_nest[j, n]))) / (lambda[n] * sum_sum_chid_nest_ttl[j])
                    end
                end
            end

            if equal_lambdas
                gradi[:, (Base.length(beta)+1):end] = sum(gradi_lambda, dims=2)
            else
                gradi[:, (Base.length(beta)+1):end] = gradi_lambda
            end
            G .= -vec(sum(gradi, dims=1))
        end

        if F !== nothing
            return -sum(vec_weights_chid .* log.(choice_prob))
        end
    end

    # objective function in case of automatic or finite differentiation
    f(theta) = fg!(1, nothing, theta)
    function gr(theta)
        GG = similar(theta)
        fg!(nothing, GG, theta)
        return GG
    end

    opt = Optim.optimize(Optim.only_fg!(fg!), coef_start::Vector{Float64}, method, optim_options)

    converged = Optim.converged(opt)
    iter = Optim.iterations(opt)
    coefficients_scaled::Vector{Float64} = convert.(Float64, Optim.minimizer(opt))

    function compute_fitted_values(theta::Vector{Float64})::Vector{Float64}
        beta .= theta[1:n_coefficients_utility]

        if equal_lambdas
            lambda[2:end] .= theta[end]
        else
            lambda[2:end] = theta[(n_coefficients_utility+1):end]
        end

        V .= mat_X * beta
        exp_adj_V = [exp(V[i] / lambda[vec_nests_indices[i]+1]) for i in eachindex(V)]

        fill!(sum_chid_nest, zero(eltype(theta)))
        @inbounds for i in eachindex(vec_chid)
            sum_chid_nest[vec_chid[i], vec_nests_indices[i]+1] += exp_adj_V[i]
        end

        sum_sum_chid_nest_ttl = sum(sum_chid_nest .^ lambda', dims=2)

        # Calculate choice probabilities for all alternatives
        fitted_values_all = [(exp_adj_V[i] * (sum_chid_nest[vec_chid[i], vec_nests_indices[i]+1]^(-1 + lambda[vec_nests_indices[i]+1])) / sum_sum_chid_nest_ttl[vec_chid[i]]) for i in eachindex(V)]

        return fitted_values_all
    end

    loglik = -f(coefficients_scaled)
    loglik_0 = -f([zeros(n_coefficients_utility); ones(n_nests)])
    loglik_start = -f(coef_start)

    gradient = gr(coefficients_scaled)
    estfun::Matrix{Float64} = gradi
    # hessian = ForwardDiff.jacobian(gr, coefficients_scaled)
    # TODO check whether ForwardDiff would have significant advantages. Overwriting things in fg! would likely not work then
    hessian::Matrix{Float64} = FiniteDifferences.jacobian(central_fdm(3, 1), gr, coefficients_scaled)[1]
    fitted_values = compute_fitted_values(coefficients_scaled)
    return opt, coefficients_scaled, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values, nests
end