function mlogit(
    formula::FormulaTerm,
    df,
    vcov_type::CovarianceEstimator=Vcov.simple();
    weights::Union{Symbol,Nothing}=nothing,
    start::Union{Nothing,Vector{Float64}}=nothing,
    indices::XlogitIndices=xlogit_indices(),
    method=nothing,
    # specific to nested logit
    equal_lambdas::Bool=false,
    optim_options=Optim.Options(),
)
    StatsAPI.fit(MNLmodel, formula, df, vcov_type; weights=weights, start=start, indices=indices, method=method, equal_lambdas=equal_lambdas, optim_options=optim_options)
end

function StatsAPI.fit(::Type{MNLmodel},
    formula::FormulaTerm,
    df,
    vcov_type::CovarianceEstimator=Vcov.simple();
    weights::Union{Symbol,Nothing}=nothing,
    start::Union{Nothing,Vector{Float64}}=nothing,
    indices::XlogitIndices=xlogit_indices(),
    method=nothing,
    # specific to nested logit
    equal_lambdas::Bool=false,
    optim_options=Optim.Options()
)

    if vcov_type != Vcov.simple()
        throw("Other types of covariance estimators than Vcov.simple() are not yet implemented. Use robust_cluster_vcov() post-estimation.")
    end

    start_time = time()

    df = DataFrame(df; copycols=false)
    nrows = size(df, 1)

    # ---------------------------------------------------------------------------- #
    #                                 Parse formula                                #
    # ---------------------------------------------------------------------------- #

    formula_origin = formula
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
        []
    elseif equal_lambdas
        ["lambda"]
    else
        # missing, nothing and 0 in the nest column are interpreted as no nest or being its own nest
        # same in remap_to_indices_nest
        ["lambda_$k" for k in filter(x -> !ismissing(x) && !isnothing(x) && x != 0, unique(vec_nests))]
    end

    coef_names = nested ? vcat(coefnames_utility, coefnames_nests) : coefnames_utility

    n_coefficients = Base.length(coef_names)

    # Ids
    n_id = Base.length(unique(df[!, indices.id]))

    # Chids
    vec_chid = convert.(Int64, df[!, indices.chid])
    # make sure that vec_chid can be used to index vectors of length length(unique(vec_chid))
    remap_to_indices_chid!(vec_chid)
    n_chid = Base.length(unique(vec_chid))

    # Weights
    if isnothing(weights)
        vec_weights = ones(Float64, Base.length(vec_choice))
    else
        vec_weights = convert.(Float64, ((df[!, weights] ./ sum(df[!, weights])) .* nrows))
    end
    vec_weights_choice = vec_weights[vec_choice]

    # Start values
    coef_start = if isnothing(start)
        [zeros(Base.length(coefnames_utility)); ones(nested * (equal_lambdas + !equal_lambdas * Base.length(unique(vec_nests))))]
    else
        copy(start) # to prevent start from being mutated in place
    end
    coef_start = convert.(Float64, coef_start)

    # Scaling
    # Standardizing the matrix column-wise
    mean_X = vec(mean(mat_X, dims=1))
    std_X = vec(std(mat_X, dims=1))
    extended_std_X = vcat(std_X, fill(1.0, Base.length(coef_start) - Base.length(std_X)))
    mat_X .= (mat_X .- mean_X') ./ std_X'
    coef_start .*= extended_std_X

    if !nested
        opt, coefficients_scaled, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values = fit_mlogit_nonests(mat_X, vec_choice, coef_start, vec_chid, vec_weights_choice; optim_options=optim_options)
    else
        opt, coefficients_scaled, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values, nests = fit_mlogit_nests(mat_X, vec_choice, coef_start, vec_chid, vec_weights_choice, vec_nests, equal_lambdas; optim_options=optim_options)
    end

    # revert standardization
    coefficients = coefficients_scaled ./ extended_std_X
    estfun .= estfun .* extended_std_X'
    # revert standardization
    hessian .*= extended_std_X * extended_std_X'
    vcov = fill(NaN, n_coefficients, n_coefficients)

    # if any(isinf, hessian) || any(isnan, hessian)
    #     # BHHH estimator
    #     hessian .= gradi' * gradi
    #     hessian_method = "BHHH"
    # end

    # if both fail, don't compute vcov
        vcov = inv(hessian)

    r = MNLmodel(
        coef=coefficients,
        coef_scaled=coefficients_scaled,
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

function fit_mlogit_nonests(mat_X, vec_choice, coef_start, vec_chid, vec_weights_choice; method=Newton(), optim_options=Optim.Options())

    n_chid = Base.length(unique(vec_chid))
    idx_map = create_index_map(vec_chid)
    n_coefficients = Base.length(coef_start)

    # Initialize objects in fgh!
    exb = exp.(mat_X * coef_start)
    sexb = zeros(Float64, n_chid)
    Pni = zeros(Float64, Base.length(vec_chid))
    Px = zeros(Float64, Base.length(coef_start), n_chid)
    yx = zeros(Float64, Base.length(coef_start), n_chid)
    gradi = zeros(Float64, n_chid, Base.length(coef_start))

    # further preallocation as suggested by deepseek
    dxpx = zeros(Float64, Base.length(vec_chid), Base.length(coef_start))
    dxpx_Pni = similar(dxpx)

    function fgh!(F, G, H, theta)
        # Common computations
        exb .= exp.(mat_X * theta)

        fill!(sexb, zero(eltype(theta)))
        @inbounds for i in eachindex(vec_chid)
            sexb[vec_chid[i]] += exb[i]
        end

        # restructuring loops and checks as suggested by deepseek; gives marginal time improvements
        if isnothing(G) && isnothing(H)
            @inbounds for i in eachindex(vec_chid)
                if vec_choice[i]
                    Pni[i] = exb[i] / sexb[vec_chid[i]]
                end
            end
        else
            @inbounds for i in eachindex(vec_chid)
                Pni[i] = exb[i] / sexb[vec_chid[i]]
            end
        end

        if !(isnothing(G) && isnothing(H))
            # if one of G or H is required
            fill!(Px, zero(eltype(theta)))
            fill!(yx, zero(eltype(theta)))
            @inbounds for i in eachindex(vec_chid), j in eachindex(theta)
                Px[j, vec_chid[i]] += Pni[i] * mat_X[i, j]
                yx[j, vec_chid[i]] += vec_choice[i] * mat_X[i, j]
            end
        end

        if G !== nothing
            gradi .= (yx .- Px)' .* vec_weights_choice
            G .= -vec(sum(gradi, dims=1))
        end

        if H !== nothing
            # further preallocation as suggested by deepseek
            dxpx .= zero(eltype(theta))
            dxpx_Pni .= zero(eltype(theta))

            # dxpx = zeros(eltype(theta), Base.length(vec_chid), Base.length(coef_start))
            @inbounds for i in eachindex(vec_chid), j in eachindex(theta)
                dxpx[i, j] += mat_X[i, j] - Px[j, vec_chid[i]]
            end

            dxpx_Pni .= dxpx .* Pni

            H .= zero(eltype(theta))
            @inbounds @simd for i in 1:n_chid
                H .+= dxpx_Pni[idx_map[i], :]' * dxpx[idx_map[i], :] # hessian per chid, sum up
            end
        end

        if F !== nothing
            return -sum(vec_weights_choice .* log.(Pni[vec_choice])) # TODO used to be Pni[indices_choice], was this important?
        end
    end

    f(theta) = fgh!(1, nothing, nothing, theta)
    function gr(theta)
        GG = similar(theta)
        fgh!(nothing, GG, nothing, theta)
        return GG
    end

    opt = Optim.optimize(Optim.only_fgh!(fgh!), coef_start, method, optim_options)

    coefficients_scaled::Vector{Float64} = Optim.minimizer(opt)
    converged = Optim.converged(opt)
    iter = Optim.iterations(opt)
    loglik = -f(coefficients_scaled)
    loglik_0 = -fgh!(1, nothing, nothing, zeros(n_coefficients))
    loglik_start = -f(coef_start)
    gradient = gr(coefficients_scaled)
    estfun = gradi
    hessian = fill(NaN, n_coefficients, n_coefficients)
    fgh!(1, nothing, hessian, coefficients_scaled)
    fitted_values = Pni

    return opt, coefficients_scaled, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values
end

function fit_mlogit_nests(mat_X, vec_choice, coef_start, vec_chid::Vector{Int64}, vec_weights_choice, vec_nests, equal_lambdas; method=LBFGS(linesearch=LineSearches.BackTracking()), optim_options=Optim.Options())

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
    beta = coef_start[1:n_coefficients_utility]
    lambda = ones(n_nests + 1)
    V = mat_X * beta
    sum_chid_nest = zeros(n_chid, n_nests + 1)
    choice_prob = zeros(n_chid)
    gradi = zeros(Float64, n_chid, n_coefficients_utility + !equal_lambdas * n_nests + equal_lambdas)
    gradi_lambda = zeros(Float64, n_chid, n_nests)

    function fg!(F, G, theta)
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

                    gradi[j, g] += vec_weights_choice[j] * tmp_g * precomputed_multiplication[j]
                end

                # Compute gradient for lambdas
                @inbounds for n in 2:n_nests+1
                    g = n - 1
                    if vec_nests_indices_choice[j] + 1 == n
                        tmp_g = -(choice_prob[j] * V_choice[j]) / lambda[n]^2
                        tmp_g += choice_prob[j] * (-((-1 + lambda[n]) * exp_adj_V_times_V_nestsums[j, n]) / (lambda[n]^2 * vec_sum_chid_nest[j]) + safe_log(vec_sum_chid_nest[j]))
                        tmp_g += (exp_adj_V_choice[j] * safe_exp(vec_sum_chid_nest[j], -2 + 2 * lambda[n]) * (exp_adj_V_times_V_nestsums[j, n] - lambda[n] * vec_sum_chid_nest[j] * safe_log(vec_sum_chid_nest[j]))) / (lambda[n] * sum_sum_chid_nest_ttl[j]^2)
                        gradi_lambda[j, g] += vec_weights_choice[j] * tmp_g * precomputed_multiplication[j]
                    else
                        # gradi_lambda[j, g] += vec_weights_choice[j] * (sum_chid_nest[j, n]^(-1 + lambda[n]) * (exp_adj_V_times_V_nestsums[j, n] - lambda[n] * sum_chid_nest[j, n] * safe_log(sum_chid_nest[j, n]))) / (lambda[n] * sum_sum_chid_nest_ttl[j])
                        gradi_lambda[j, g] += vec_weights_choice[j] * (safe_exp(sum_chid_nest[j, n], -1 + lambda[n]) * (exp_adj_V_times_V_nestsums[j, n] - lambda[n] * sum_chid_nest[j, n] * safe_log(sum_chid_nest[j, n]))) / (lambda[n] * sum_sum_chid_nest_ttl[j])
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
            return -sum(vec_weights_choice .* log.(choice_prob))
        end
    end

    # objective function in case of automatic or finite differentiation
    f(theta) = fg!(1, nothing, theta)
    function gr(theta)
        GG = similar(theta)
        fg!(nothing, GG, theta)
        return GG
    end

    opt = Optim.optimize(Optim.only_fg!(fg!), coef_start, method, optim_options)

    converged = Optim.converged(opt)
    iter = Optim.iterations(opt)
    coefficients_scaled = convert.(Float64, Optim.minimizer(opt))

    function compute_fitted_values(theta)
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
    estfun = gradi
    # hessian = ForwardDiff.jacobian(gr, coefficients_scaled)
    # TODO check whether ForwardDiff would have significant advantages. Overwriting things in fg! would likely not work then
    hessian = FiniteDifferences.jacobian(central_fdm(3, 1), gr, coefficients_scaled)[1]
    fitted_values = compute_fitted_values(coefficients_scaled)
    return opt, coefficients_scaled, converged, iter, loglik, loglik_0, loglik_start, gradient, estfun, hessian, fitted_values, nests
end