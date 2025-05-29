function lclogit(
    formula::FormulaTerm,
    df,
    n_classes::Int64;
    start_memb::Union{Nothing,Matrix{Float64}}=nothing,
    start_mnl::Union{Nothing,Matrix{Float64}}=nothing,
    indices::XlogitIndices=xlogit_indices(),
    method::Symbol=:em,
    quietly::Bool=false,
    standardize=true,
    # relevant for em
    varname_samplesplit=nothing,
    max_iter::Int64=1000,
    ltolerance::Float64=1e-7,
    multithreading::Bool=false, # appears to be only beneficial for high numbers of classes due to overhead
    # relevant for gradient
    optim_method=BFGS(),
    optim_options=Optim.Options(),
    max_retries::Int=5, # to be used if, e.g., the Hessian after the EM algorithm cannot be inverted,
    chunksize_func=(n) -> ceil(Int, sqrt(n)) # for Hessian chunksize
)

    StatsAPI.fit(LCLmodel, formula, df, n_classes, start_memb, start_mnl, indices, method, quietly, standardize, varname_samplesplit, max_iter, ltolerance, multithreading, optim_method, optim_options, max_retries, chunksize_func)
end

# Function to compute chid_map
function create_chid_map(vec_chid, vec_id, n_id)
    id_map = Dict(n => findall(==(n), vec_id) for n in 1:n_id)
    chid_map = Dict()

    for n in 1:n_id
        idx_n = id_map[n]  # Get indices for this subject
        chid_n = vec_chid[idx_n]  # Extract chid values
        chid_map[n] = Dict(t => findall(==(t), chid_n) for t in unique(chid_n))
    end

    return id_map, chid_map
end

function loglik_lc(theta::Vector, mat_X::Matrix{Float64}, mat_memb::Matrix{Float64},
    n_id::Int64, vec_choice::BitVector, n_classes::Int64,
    k_utility::Int64, k_membership::Int64, nrows::Int64, ll_n, mat_utils, Xb_share,
    id_map::Dict, chid_map::Dict)

    # Preallocate memory for log-likelihood
    if eltype(ll_n) ≠ eltype(theta)
        ll_n = zeros(eltype(theta), n_id)
    end
    if eltype(mat_utils) ≠ eltype(theta)
        mat_utils = zeros(eltype(theta), size(mat_X, 1), n_classes)
    end
    if eltype(Xb_share) ≠ eltype(theta)
        Xb_share = zeros(eltype(theta), nrows, n_classes)
    else
        Xb_share[:, n_classes] .= 0.0
    end


    # Reshape coefficients
    mat_coefs_mlogit_ll = reshape(theta[begin:(n_classes*k_utility)], k_utility, n_classes)
    coefs_memb_ll = reshape(theta[(n_classes*k_utility+1):end], (k_membership + 1), (n_classes - 1))

    # Efficient matrix multiplication
    mul!(mat_utils, mat_X, mat_coefs_mlogit_ll)

    # Compute membership probabilities efficiently
    @inbounds for c in 1:(n_classes-1)
        Xb_share[:, c] .= mat_memb * coefs_memb_ll[:, c]
    end
    Xb_share[:, n_classes] .= 0.0

    # Class shares
    log_probs_memb = Xb_share .- LogExpFunctions.logsumexp(Xb_share, dims=2)

    # Preallocate for conditional probabilities
    cond_probs_memb = zeros(eltype(theta), nrows, n_classes)

    # Initialize vector to sum log-probabilities over choice sets for each class
    log_prob_choices_given_class = zeros(eltype(theta), n_classes) # Size (n_classes,)
    log_sum_utils_t = zeros(eltype(theta), 1, n_classes) # Size (1, n_classes)
    tmp_exp = zeros(eltype(theta), n_classes) # Size (n_classes,)

    # Loop over subjects
    @inbounds for n in 1:n_id
        idx_n = id_map[n] # Precomputed indices
        Y_n = @view vec_choice[idx_n]
        utils_n = @view mat_utils[idx_n, :] # Log-utilities for individual n
        log_probs_memb_n = @view log_probs_memb[idx_n, :] # Log-membership probabilities for individual n (repeated for each alternative)
        # log_probs_memb_n = Xb_share[idx_n, :] .- LogExpFunctions.logsumexp(Xb_share[idx_n, :], dims=2)

        # Get chid mapping for this individual
        chid_map_n = chid_map[n]

        fill!(log_prob_choices_given_class, zero(eltype(theta)))

        # Loop over choice sets
        @inbounds for t in keys(chid_map_n)
            idx_t = chid_map_n[t] # Indices within idx_n for alternatives in choice set t
            utils_nt = @view utils_n[idx_t, :] # Log-utilities for choice set t

            # Fill choice probabilities (log-probabilities using logsoftmax) for the current choice set
            # This calculates log(P(alternative | choice set t, class k)) for all alternatives in set t
            # log_choiceprobs_n_t = utils_nt .- LogExpFunctions.logsumexp(utils_nt, dims=1) # Size (num_alternatives_in_t, n_classes)
            log_sum_utils_t .= LogExpFunctions.logsumexp(utils_nt, dims=1)

            # Identify the chosen alternative within this choice set t
            Y_nt = @view Y_n[idx_t] # Boolean vector for chosen alternative in choice set t
            chosen_alt_local_idx = findfirst(Y_nt) # Index within idx_t and log_choiceprobs_n_t

            # Accumulate the log-probability of the chosen alternative for each class
            @inbounds for k in 1:n_classes
                # log(P(chosen alternative in set t | set t, class k)) is log_choiceprobs_n_t[chosen_alt_local_idx, k]
                # log_prob_choices_given_class[k] += log_choiceprobs_n_t[chosen_alt_local_idx, k]
                log_prob_choices_given_class[k] += utils_nt[chosen_alt_local_idx, k] - log_sum_utils_t[1, k]
            end
        end

        # Stable likelihood using logsumexp with log(P(choices_n | class k))
        ll_n[n] = LogExpFunctions.logsumexp(log_prob_choices_given_class .+ vec(log_probs_memb_n[1, :]))

        log_prob_choices_given_class .+= vec(log_probs_memb_n[1, :]) # Use log_prob_choices_given_class here
        # exponentiate only once, to get the actual conditional probabilities
        tmp_exp .= exp.(log_prob_choices_given_class .- LogExpFunctions.logsumexp(log_prob_choices_given_class)) # Result is (n_classes,), no need for [1, :] if log_joint is (n_classes,)
        for i in idx_n
            cond_probs_memb[i, :] .= tmp_exp # Transpose tmp_exp for broadcasting assignment
        end
    end

    return -sum(ll_n)
end

function cond_probs_ll(coefs_mlogit::Matrix{Float64}, coefs_memb::Matrix{Float64}, mat_X::Matrix{Float64}, mat_memb::Matrix{Float64},
    n_id::Int64, vec_choice::BitVector, n_classes::Int64,
    ll_n::Vector{Float64}, mat_utils::Matrix{Float64}, Xb_share::Matrix{Float64}, cond_probs_memb::Matrix{Float64}, log_ProbSeq_n::Matrix{Float64},
    id_map::Dict, chid_map::Dict, sumll::Vector{Float64})

    mul!(mat_utils, mat_X, coefs_mlogit)

    # Compute membership probabilities
    @inbounds for c in 1:(n_classes-1)
        Xb_share[:, c] .= mat_memb * coefs_memb[:, c]
    end
    Xb_share[:, n_classes] .= 0.0

    # Class shares
    log_probs_memb = Xb_share .- LogExpFunctions.logsumexp(Xb_share, dims=2)

    # Initialize vector to sum log-probabilities over choice sets for each class
    log_prob_choices_given_class = zeros(Float64, n_classes) # Size (n_classes,)
    log_sum_utils_t = zeros(Float64, 1, n_classes) # Size (1, n_classes)
    tmp_exp = zeros(Float64, n_classes) # Temporary vector for conditional probabilities

    # Loop over subjects
    @inbounds for n in 1:n_id
        idx_n = id_map[n] # Precomputed indices
        Y_n = @view vec_choice[idx_n]
        utils_n = @view mat_utils[idx_n, :] # Log-utilities for individual n
        log_probs_memb_n = @view log_probs_memb[idx_n, :] # Log-membership probabilities for individual n (repeated for each alternative)
        # log_probs_memb_n = Xb_share[idx_n, :] .- LogExpFunctions.logsumexp(Xb_share[idx_n, :], dims=2)

        # Get chid mapping for this individual
        chid_map_n = chid_map[n]

        fill!(log_prob_choices_given_class, 0.0)

        # Loop over choice sets
        @inbounds for t in keys(chid_map_n)
            idx_t = chid_map_n[t] # Indices within idx_n for alternatives in choice set t
            utils_nt = @view utils_n[idx_t, :] # Log-utilities for choice set t

            # Fill choice probabilities (log-probabilities using logsoftmax) for the current choice set
            # This calculates log(P(alternative | choice set t, class k)) for all alternatives in set t
            # log_choiceprobs_n_t = utils_nt .- LogExpFunctions.logsumexp(utils_nt, dims=1) # Size (num_alternatives_in_t, n_classes)
            log_sum_utils_t .= LogExpFunctions.logsumexp(utils_nt, dims=1)

            # Identify the chosen alternative within this choice set t
            Y_nt = @view Y_n[idx_t] # Boolean vector for chosen alternative in choice set t
            chosen_alt_local_idx = findfirst(Y_nt) # Index within idx_t and log_choiceprobs_n_t

            # Accumulate the log-probability of the chosen alternative for each class
            @inbounds for k in 1:n_classes
                # log(P(chosen alternative in set t | set t, class k)) is log_choiceprobs_n_t[chosen_alt_local_idx, k]
                log_prob_choices_given_class[k] += utils_nt[chosen_alt_local_idx, k] - log_sum_utils_t[1, k]
            end
        end

        # We use log_prob_choices_given_class (size (n_classes,)) and log_probs_memb_n[1, :] (size (1, n_classes))
        ll_n[n] = LogExpFunctions.logsumexp(log_prob_choices_given_class .+ vec(log_probs_memb_n[1, :]))

        log_prob_choices_given_class .+= vec(log_probs_memb_n[1, :]) # Use log_prob_choices_given_class here
        # exponentiate only once, to get the actual conditional probabilities
        tmp_exp .= exp.(log_prob_choices_given_class .- LogExpFunctions.logsumexp(log_prob_choices_given_class)) # Result is (n_classes,), no need for [1, :] if log_joint is (n_classes,)
        for i in idx_n
            cond_probs_memb[i, :] .= tmp_exp
        end
    end

    # Store results
    push!(sumll, sum(ll_n))
end

function StatsAPI.fit(::Type{LCLmodel},
    formula::FormulaTerm,
    df,
    n_classes::Int64,
    start_memb::Union{Nothing,Matrix{Float64}},
    start_mnl::Union{Nothing,Matrix{Float64}},
    indices::XlogitIndices,
    method::Symbol,
    quietly::Bool,
    standardize::Bool,
    varname_samplesplit,
    max_iter::Int64,
    ltolerance::Float64,
    multithreading::Bool,
    optim_method,
    optim_options,
    max_retries::Int,
    chunksize_func
)
    # start time
    start_time = time()

    # prevent provided data from being modified (is this the best solution?)
    # df = DataFrame(df; copycols=false)
    nrows = size(df, 1)

    # # check that no column in df starts with lcl_ as this will be used later
    # if maximum(startswith.(names(df), "lcl_"))
    #     error("Column names must not start with \"lcl_\" as this is reserved for columns created by the algorithm")
    # end

    # ---------------------------------------------------------------------------- #
    #                                 Parse formula                                #
    # ---------------------------------------------------------------------------- #

    formula_origin = formula
    formula, formula_membership = parse_membership(formula_origin, n_classes)

    # ---------------------------------------------------------------------------- #
    #                             DataFrame --> Matrix                             #
    # ---------------------------------------------------------------------------- #

    s = schema(formula, df)
    s_memb = schema(formula_membership, df)

    formula_schema = apply_schema(formula, s)
    formula_schema_memb = apply_schema(formula_membership, s_memb)
    vec_choice::BitVector, mat_X::Matrix{Float64} = modelcols(formula_schema, df)

    mat_memb::Matrix{Float64} = convert(Matrix{Float64}, modelmatrix(formula_schema_memb, df))

    if standardize
        mat_X_mean = mean(mat_X, dims=1)
        mat_X .-= mat_X_mean
        mat_X_std = std(mat_X, dims=1)
        mat_X ./= mat_X_std

        mat_memb_mean = mean(mat_memb, dims=1)
        mat_memb .-= mat_memb_mean
        mat_memb_std = std(mat_memb, dims=1)
        mat_memb ./= mat_memb_std
    end

    mat_memb = hcat(mat_memb, ones(Float64, nrows)) # add constant column. maybe this should be incorporated in formula but fmlogit would not expect that (always assumes constant)

    response_name::String, coefnames_utility::Vector{String} = StatsModels.coefnames(formula_schema)
    _, coefnames_membership = StatsModels.coefnames(formula_schema_memb)
    coefnames_membership = [coefnames_membership;] # to ensure that it is a vector even if it has only one element
    k_membership::Int64 = Base.length(coefnames_membership)
    k_utility::Int64 = Base.length(coefnames_utility)

    vec_id::Vector{Int64} = Vector{Int64}(df.id)
    n_id::Int64 = Base.length(unique(vec_id))
    n_coefficients = (k_membership + 1) * (n_classes - 1) + k_utility * n_classes

    # for changing to array-based code
    lcl_first_by_id::BitVector = let seen = Set()
        [id in seen ? false : (push!(seen, id); true) for id in vec_id]
    end

    # Chids
    vec_chid::Vector{Int64} = df[!, indices.chid]
    # make sure that vec_chid can be used to index vectors of length length(unique(vec_chid))
    # unique(vec_chid) != 1:length(unique(vec_chid)) && 
    remap_to_indices_chid!(vec_chid)
    n_chid::Int64 = Base.length(unique(vec_chid))

    probs_memb = Matrix{Float64}(undef, nrows, n_classes)

    # Start values
    # TODO seems to ignore start values when using method=:em
    coefs_mlogit::Matrix{Float64} = if isnothing(start_mnl)
        zeros(Float64, k_utility, n_classes) # one column represents the coefs of a class's MNL model
    else
        copy(start_mnl) # to prevent start from being mutated in place
    end
    coefs_memb::Matrix{Float64} = if isnothing(start_memb)
        zeros(Float64, k_membership + 1, n_classes - 1)
    else
        copy(start_memb) # to prevent start from being mutated in place
    end

    if standardize
        for s in 1:n_classes
            coefs_mlogit[:, s] .*= vec(mat_X_std)
        end
        for s in 1:(n_classes-1)
            view(coefs_memb, 1:k_membership, s) .*= mat_memb_std'
            coefs_memb[end, s] = coefs_memb[end, s] + dot((view(coefs_memb, 1:k_membership, s) ./ mat_memb_std'), mat_memb_mean')
        end
    end

    # Initialize values

    cond_probs_memb::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # Class share indices (= membership coefficients? )
    Xb_share::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # matrix of utilities
    mat_utils::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # [1 x nclasses] vector of the likelihood of actual choice sequence
    log_ProbSeq_n::Matrix{Float64} = zeros(Float64, 1, n_classes)

    ll_n::Vector{Float64} = zeros(Float64, n_id)

    id_map, chid_map = create_chid_map(vec_chid, vec_id, n_id)

    function loglik_obj(theta)
        return loglik_lc(theta, mat_X, mat_memb, n_id, vec_choice, n_classes, k_utility, k_membership, nrows, ll_n, mat_utils, Xb_share, id_map, chid_map)
    end

    loglik_0 = -loglik_lc(zeros(k_utility * n_classes + (k_membership + 1) * (n_classes - 1)), mat_X, mat_memb, n_id, vec_choice, n_classes, k_utility, k_membership, nrows, ll_n, mat_utils, Xb_share, id_map, chid_map)
    loglik_start = -loglik_lc([vec(coefs_mlogit); vec(coefs_memb)], mat_X, mat_memb, n_id, vec_choice, n_classes, k_utility, k_membership, nrows, ll_n, mat_utils, Xb_share, id_map, chid_map)
    quietly || println("Null log-likelihood: $loglik_0")
    quietly || println("Start log-likelihood: $loglik_start")

    iter = 1 # for inner EM algorithm loop
    converged = false

    if method == :em

        iter = 1 # for inner EM algorithm loop

        ### split sample

        function create_lcl_s(vec_id, n_classes)
            # Get unique IDs and their first positions
            unique_ids = unique(vec_id)
            prop = 1 / n_classes

            # Assign random class probabilities to each ID
            id_to_class = Dict{eltype(vec_id),Int}()
            for id in unique_ids
                rand_val = rand(Uniform())
                class = findfirst(>(rand_val), cumsum(repeat([prop], n_classes)))
                id_to_class[id] = isnothing(class) ? n_classes : class
            end

            # Expand to full vector
            [id_to_class[id] for id in vec_id]
        end

        lcl_s::Vector{Float64} = if !isnothing(varname_samplesplit)
            df[!, Symbol(varname_samplesplit)]
        else
            create_lcl_s(vec_id, n_classes)
        end

        sumll::Vector{Float64} = Float64[]

        for s in 1:n_classes
            vec_chid_s = vec_chid[lcl_s.==s]
            remap_to_indices_chid!(vec_chid_s) # mlogit would do this in the data prep step, so has to be done before calling fit_mlogit
            # use fit_mlogit because it takes the (standardized) mat_X
            _, coefs_mlogit[:, s], _, _, _, _, _, _, _, _, _ = fit_mlogit(mat_X[lcl_s.==s, :], vec_choice[lcl_s.==s], coefs_mlogit[:, s], vec_chid_s, ones(Float64, sum(vec_choice[lcl_s.==s])))
        end

        cond_probs_ll(coefs_mlogit, coefs_memb, mat_X, mat_memb,
            n_id, vec_choice, n_classes,
            ll_n, mat_utils, Xb_share, cond_probs_memb, log_ProbSeq_n,
            id_map, chid_map, sumll)

        quietly || println("Iteration 0 - Log likelihood: $(last(sumll))")

        ### Loop
        llincrease = 9999.9

        while iter <= max_iter
            call_mlogit_coef(s) = fit_mlogit(mat_X, vec_choice, coefs_mlogit[:, s], vec_chid, cond_probs_memb[:, s][vec_choice])
            # Update the probability of the agent's sequence of choices
            if multithreading
                Threads.@threads for s in 1:n_classes
                    _, coefs_mlogit[:, s], _, _, _, _, _, _, _, _, _ = call_mlogit_coef(s)
                end
            else
                for s in 1:n_classes
                    _, coefs_mlogit[:, s], _, _, _, _, _, _, _, _, _ = call_mlogit_coef(s)
                end
            end

            # Update the class share probabilities
            if k_membership == 0
                Share = sum(cond_probs_memb, dims=1) / sum(cond_probs_memb)
                coefs_memb .= log.(Share / Share[n_classes])[:, 1:(n_classes-1)]
            else
                opt_fmlogit = Optim.optimize(theta -> loglik_fmlogit(theta, cond_probs_memb[lcl_first_by_id, :], mat_memb[lcl_first_by_id, :], fill(1.0, n_id), k_membership, n_classes; multithreading=multithreading), vec(coefs_memb), Newton(), Optim.Options(), autodiff=:forward)
                coefs_memb .= reshape(Optim.minimizer(opt_fmlogit), k_membership + 1, n_classes - 1)
            end

            cond_probs_ll(coefs_mlogit, coefs_memb, mat_X, mat_memb,
                n_id, vec_choice, n_classes,
                ll_n, mat_utils, Xb_share, cond_probs_memb, log_ProbSeq_n,
                id_map, chid_map, sumll)

            quietly || println("Iteration $iter - Log likelihood: $(last(sumll))")

            # Check for convergence
            if iter >= 6
                llincrease = -(sumll[iter] - sumll[iter-5]) / sumll[iter-5]
                if llincrease <= ltolerance
                    converged = true
                    break
                end
            end

            # If not converged, restart loop
            iter += 1
        end
    elseif method == :gradient

        # Compute the gradient configuration
        cfg = ForwardDiff.GradientConfig(loglik_obj, [vec(coefs_mlogit); vec(coefs_memb)], ForwardDiff.Chunk{n_coefficients}())

        # Define the gradient function
        function loglik_grad!(G, theta)
            ForwardDiff.gradient!(G, loglik_obj, theta, cfg)
        end

        # Wrap function and gradient in OnceDifferentiable
        fdf = Optim.OnceDifferentiable(loglik_obj, loglik_grad!, [vec(coefs_mlogit); vec(coefs_memb)])

        # Run optimization
        opt = Optim.optimize(fdf, [vec(coefs_mlogit); vec(coefs_memb)], optim_method, optim_options)

        coefficients = Optim.minimizer(opt)
        coefs_mlogit .= reshape(coefficients[1:(k_utility*n_classes)], k_utility, n_classes)
        coefs_memb .= reshape(coefficients[(k_utility*n_classes+1):end], (k_membership + 1), (n_classes - 1))

        converged = Optim.converged(opt)
        iter = Optim.iterations(opt)
    else
        error("Unknown method. Choose :em or :gradient")
    end


    # TODO this chunksize calculation is not necessarily optimal. n_coefficients is definitely worse, however
    chunksizeH::Int64 = chunksize_func(n_coefficients)
    quietly || println("Starting Hessian calculation with $n_coefficients coefficients and a chunksize of $chunksizeH.")
    diffresult = DiffResults.HessianResult([vec(coefs_mlogit); vec(coefs_memb)])
    cfgH = ForwardDiff.HessianConfig(loglik_obj, diffresult, [vec(coefs_mlogit); vec(coefs_memb)], ForwardDiff.Chunk{chunksizeH}())
    diffresult = ForwardDiff.hessian!(diffresult, loglik_obj, [vec(coefs_mlogit); vec(coefs_memb)], cfgH)

    gradient::Vector{Float64} = DiffResults.gradient(diffresult)
    hessian::Matrix{Float64} = DiffResults.hessian(diffresult)

    if standardize
        for s in 1:n_classes
            coefs_mlogit[:, s] ./= vec(mat_X_std)
        end
        for s in 1:(n_classes-1)
            view(coefs_memb, 1:k_membership, s) ./= mat_memb_std'
            # re-scale constant
            coefs_memb[end, s] = coefs_memb[end, s] - dot(view(coefs_memb, 1:k_membership, s), mat_memb_mean')
        end

        mat_std_ext = vcat(repeat(vec(mat_X_std), outer=n_classes), repeat(vcat(vec(mat_memb_std), [1.0]), outer=n_classes - 1))
        J = diagm(mat_std_ext)
        for (i, mean) in enumerate(mat_memb_mean)
            # The current 'mean' (which is mu_i for the i-th membership variable)
            # affects the constant term of *every* membership model equation.
            # Therefore, we loop through all (n_classes - 1) membership equations.
            for eq_loop_idx in 0:(n_classes-2)  # Loop from 0 (for 1st eq) to (num_membership_equations - 1)

                # Calculate the starting index in the full J matrix for the block of parameters
                # belonging to the current membership equation (identified by eq_loop_idx).
                # Each membership equation has (k_membership variables + 1 constant) parameters.
                base_idx_for_current_memb_equation = n_classes * k_utility + eq_loop_idx * (k_membership + 1)

                # Identify the row in J corresponding to the standardized constant (beta_const_std)
                # of the current membership equation. This is the (k_membership + 1)-th
                # parameter within this equation's specific block.
                row_J_for_beta_const_std = base_idx_for_current_memb_equation + k_membership + 1

                # Identify the column in J corresponding to the i-th original variable coefficient
                # (beta_var_orig_i) of the current membership equation. The index 'i' comes
                # from the outer 'enumerate(mat_memb_mean)' loop and refers to the
                # local index (1 to k_membership) of the variable whose 'mean' we are currently processing.
                col_J_for_beta_var_orig_i = base_idx_for_current_memb_equation + i

                # Assign the mean value to the J matrix.
                # This sets J[row_for_std_const, col_for_orig_var_i] = mu_i
                # This fills the off-diagonal elements in the rows corresponding to the
                # standardized constants of the membership model equations.
                J[row_J_for_beta_const_std, col_J_for_beta_var_orig_i] = mean
            end
        end
        gradient .= transpose(J) * gradient
        hessian .= transpose(J) * hessian * J
    end

    vcov = inv(hessian)

    if any(diag(vcov) .< 0.0)
        @warn "Main diagonale of VCOV has negative entries. Try gradient-based optimization."
    end

    loglik = -DiffResults.value(diffresult)::Float64

    # shares calculation
    for c in 1:(n_classes-1)
        Xb_share[:, c] .= mat_memb * coefs_memb[:, c]
    end

    # Class shares
    log_probs_memb = Xb_share .- LogExpFunctions.logsumexp(Xb_share, dims=2)
    probs_memb = exp.(log_probs_memb)
    shares::Vector{Float64} = vec(mean(probs_memb[lcl_first_by_id, :], dims=1))

    r = LCLmodel(
        coef_memb=coefs_memb,
        coef_mnl=coefs_mlogit,
        coefnames_memb=isempty(coefnames_membership) ? ["constant";] : [coefnames_membership; "constant"],
        coefnames_mnl=coefnames_utility,
        converged=converged,
        dof=n_coefficients,
        formula=formula,
        formula_origin=formula_origin,
        formula_schema=formula_schema,
        hessian=hessian,
        iter=iter,
        loglikelihood=loglik,
        method=method,
        nchids=n_chid,
        nclasses=n_classes,
        nids=n_id,
        nullloglikelihood=loglik_0,
        optim=(@isdefined opt) ? opt : nothing,
        responsename=response_name,
        score=gradient,
        shares=shares,
        start=zeros(k_utility * n_classes + (k_membership + 1) * (n_classes - 1)),
        startloglikelihood=loglik_start,
        time=time() - start_time,
        vcov=vcov
    )
    return r
end