function lclogit(
    formula::FormulaTerm,
    df,
    n_classes::Int64;
    start_memb::Union{Nothing,Matrix{Float64}}=nothing,
    start_mnl::Union{Nothing,Matrix{Float64}}=nothing,
    indices::XlogitIndices=xlogit_indices(),
    method::Symbol=:em,
    quietly::Bool=false,
    # relevant for em
    varname_samplesplit=nothing,
    max_iter::Int64=1000,
    ltolerance::Float64=1e-6,
    multithreading::Bool=false, # appears to be only beneficial for high numbers of classes due to overhead
    # relevant for gradient
    optim_method=BFGS(),
    optim_options=Optim.Options()
)

    StatsAPI.fit(LCLmodel, formula, df, n_classes; start_memb=start_memb, start_mnl=start_mnl, indices=indices, method=method, quietly=quietly, varname_samplesplit=varname_samplesplit, max_iter=max_iter, ltolerance=ltolerance, multithreading=multithreading, optim_method=optim_method, optim_options=optim_options)
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
    k_utility::Int64, k_membership::Int64, nrows::Int64, ll_n, exp_mat_utils, exp_Xb_share,
    id_map::Dict, chid_map::Dict)

    # Preallocate memory for log-likelihood
    if eltype(ll_n) ≠ eltype(theta)
        ll_n = zeros(eltype(theta), n_id)
    end
    if eltype(exp_mat_utils) ≠ eltype(theta)
        exp_mat_utils = zeros(eltype(theta), size(mat_X, 1), n_classes)
    end
    if eltype(exp_Xb_share) ≠ eltype(theta)
        exp_Xb_share = zeros(eltype(theta), nrows, n_classes)
    else
        exp_Xb_share[:, n_classes] .= 0.0
    end


    # Reshape coefficients
    mat_coefs_mlogit_ll = reshape(theta[begin:(n_classes*k_utility)], k_utility, n_classes)
    coefs_memb_ll = reshape(theta[(n_classes*k_utility+1):end], (k_membership + 1), (n_classes - 1))

    # exp_mat_utils = zeros(eltype(theta), size(mat_X, 1), n_classes)
    # exp_Xb_share = zeros(eltype(theta), nrows, n_classes)

    # Efficient matrix multiplication
    mul!(exp_mat_utils, mat_X, mat_coefs_mlogit_ll)

    # Element-wise exponentiation (preserving Dual types)
    exp_mat_utils .= exp.(exp_mat_utils)

    # Compute membership probabilities efficiently
    @inbounds for c in 1:(n_classes-1)
        exp_Xb_share[:, c] .= mat_memb * coefs_memb_ll[:, c]
    end
    exp_Xb_share .= exp.(exp_Xb_share)

    # Class shares
    probs_memb = exp_Xb_share ./ sum(exp_Xb_share, dims=2)

    # Preallocate for conditional probabilities
    cond_probs_memb = zeros(eltype(theta), nrows, n_classes)

    # Loop over subjects
    @inbounds for n in 1:n_id
        idx_n = id_map[n]  # Precomputed indices
        Y_n = @view vec_choice[idx_n]
        EXP_n = @view exp_mat_utils[idx_n, :]
        probs_memb_n = @view probs_memb[idx_n, :]

        # Preallocate for this subject
        cond_probs_memb_n = zeros(eltype(theta), Base.length(Y_n), n_classes)

        # Get chid mapping for this individual
        chid_map_n = chid_map[n]

        # Loop over choice sets t
        @inbounds for t in keys(chid_map_n)
            idx_t = chid_map_n[t]
            EXP_nt = @view EXP_n[idx_t, :]

            # Fill choice probabilities
            cond_probs_memb_n[idx_t, :] .= EXP_nt ./ sum(EXP_nt, dims=1)
        end

        # Compute likelihood
        ProbSeq_n = exp.(sum(log.(cond_probs_memb_n) .* Y_n, dims=1))
        ll_n[n] = log.(dot(ProbSeq_n, probs_memb_n[1, :]))

        # Compute conditional membership probabilities
        denom = dot(ProbSeq_n, probs_memb_n[1:1, :]')
        cond_probs_memb[idx_n, :] .= (ProbSeq_n .* probs_memb_n[1:1, :]) ./ denom
    end

    return -sum(ll_n)
end

function cond_probs_ll(coefs_mlogit::Matrix{Float64}, coefs_memb::Matrix{Float64}, mat_X::Matrix{Float64}, mat_memb::Matrix{Float64},
    n_id::Int64, vec_choice::BitVector, n_classes::Int64,
    ll_n::Vector{Float64}, exp_mat_utils::Matrix{Float64}, exp_Xb_share::Matrix{Float64}, cond_probs_memb::Matrix{Float64}, ProbSeq_n::Matrix{Float64},
    id_map::Dict, chid_map::Dict, sumll::Vector{Float64}, lcl_H::Matrix{Float64})

    # Compute exponentiated utilities
    mul!(exp_mat_utils, mat_X, coefs_mlogit)
    exp_mat_utils .= exp.(exp_mat_utils)

    # Compute membership probabilities
    @inbounds for c in 1:(n_classes-1)
        exp_Xb_share[:, c] .= mat_memb * coefs_memb[:, c]
    end
    exp_Xb_share[:, n_classes] .= 0.0
    exp_Xb_share .= exp.(exp_Xb_share)

    # Class shares
    probs_memb = exp_Xb_share ./ sum(exp_Xb_share, dims=2)

    # Loop over subjects
    @inbounds for n in 1:n_id
        idx_n = id_map[n]  # Precomputed indices
        Y_n = @view vec_choice[idx_n]
        EXP_n = @view exp_mat_utils[idx_n, :]
        probs_memb_n = @view probs_memb[idx_n, :]

        # Preallocate for this subject
        cond_probs_memb_n = zeros(Float64, Base.length(Y_n), n_classes)

        # Get chid mapping for this individual
        chid_map_n = chid_map[n]

        # Loop over choice sets
        @inbounds for t in keys(chid_map_n)
            idx_t = chid_map_n[t]
            EXP_nt = @view EXP_n[idx_t, :]

            # Fill choice probabilities
            cond_probs_memb_n[idx_t, :] .= EXP_nt ./ sum(EXP_nt, dims=1)
        end

        # Compute likelihood
        ProbSeq_n .= exp.(sum(log.(cond_probs_memb_n) .* Y_n, dims=1))
        ll_n[n] = log.(dot(ProbSeq_n, probs_memb_n[1, :]))

        # Compute conditional membership probabilities
        denom = dot(ProbSeq_n, probs_memb_n[1:1, :]')
        cond_probs_memb[idx_n, :] .= (ProbSeq_n .* probs_memb_n[1:1, :]) ./ denom
    end

    # Store results
    push!(sumll, sum(ll_n))
    for x in 1:n_classes
        lcl_H[:, x] .= cond_probs_memb[:, x]
    end
end


function StatsAPI.fit(::Type{LCLmodel},
    formula::FormulaTerm,
    df,
    n_classes::Int64;
    start_memb::Union{Nothing,Matrix{Float64}}=start_memb,
    start_mnl::Union{Nothing,Matrix{Float64}}=start_mnl,
    indices::XlogitIndices=xlogit_indices(),
    method::Symbol=:em,
    quietly::Bool=false,
    varname_samplesplit=nothing,
    max_iter::Int64=1000,
    ltolerance::Float64=1e-7,
    multithreading::Bool=false,
    optim_method=LBFGS(),
    optim_options=Optim.options()
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


    # Initialize values

    cond_probs_memb::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # Class share indices (= membership coefficients? )
    exp_Xb_share::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # matrix of utilities
    exp_mat_utils::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # [1 x nclasses] vector of the likelihood of actual choice sequence
    ProbSeq_n::Matrix{Float64} = zeros(Float64, 1, n_classes)

    ll_n::Vector{Float64} = zeros(Float64, n_id)

    id_map, chid_map = create_chid_map(vec_chid, vec_id, n_id)

    function loglik_obj(theta)
        return loglik_lc(theta, mat_X, mat_memb, n_id, vec_choice, n_classes, k_utility, k_membership, nrows, ll_n, exp_mat_utils, exp_Xb_share, id_map, chid_map)
    end

    if method == :em

        ### split sample

        function create_lcl_s(vec_id, n_classes)
            # Get unique IDs and their first positions
            unique_ids = unique(vec_id)
            prop = 1/n_classes

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

        lcl_H::Matrix{Float64} = Matrix(undef, nrows, n_classes)

        sumll::Vector{Float64} = Float64[]

        for s in 1:n_classes
            coefs_mlogit[:, s] .= coef(mlogit(formula, df[lcl_s.==s, :]))
        end

        cond_probs_ll(coefs_mlogit, coefs_memb, mat_X, mat_memb,
            n_id, vec_choice, n_classes,
            ll_n, exp_mat_utils, exp_Xb_share, cond_probs_memb, ProbSeq_n,
            id_map, chid_map, sumll, lcl_H)

        quietly || println("Iteration 0 - Log likelihood: $(last(sumll))")

        ### Loop

        iter = 1
        converged = false
        llincrease = 9999.9

        while iter <= max_iter
            call_mlogit_coef(s) = fit_mlogit(mat_X, vec_choice, coefs_mlogit[:, s], vec_chid, lcl_H[:, s][vec_choice])
            # Update the probability of the agent's sequence of choices
            if multithreading
                Threads.@threads for s in 1:n_classes
                    _, coefficients_scaled, _, _, _, _, _, _, _, _, _ = call_mlogit_coef(s)
                    coefs_mlogit[:, s] .= coefficients_scaled
                end
            else
                for s in 1:n_classes
                    _, coefficients_scaled, _, _, _, _, _, _, _, _, _ = call_mlogit_coef(s)
                    coefs_mlogit[:, s] .= coefficients_scaled
                end
            end

            # Update the class share probabilities
            if k_membership == 0
                Share = sum(cond_probs_memb, dims=1) / sum(cond_probs_memb)
                coefs_memb .= log.(Share / Share[n_classes])[:, 1:(n_classes-1)]
            else
                opt_fmlogit = Optim.optimize(theta -> loglik_fmlogit(theta, lcl_H[lcl_first_by_id, :], mat_memb[lcl_first_by_id, :], fill(1.0, n_id), 1, n_classes; multithreading=multithreading), vec(coefs_memb), Newton(), Optim.Options(), autodiff=:forward)
                coefs_memb .= reshape(Optim.minimizer(opt_fmlogit), 2, n_classes - 1)
            end

            cond_probs_ll(coefs_mlogit, coefs_memb, mat_X, mat_memb,
            n_id, vec_choice, n_classes,
            ll_n, exp_mat_utils, exp_Xb_share, cond_probs_memb, ProbSeq_n,
            id_map, chid_map, sumll, lcl_H)

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

    diffresult = DiffResults.HessianResult([vec(coefs_mlogit); vec(coefs_memb)])
    cfgH = ForwardDiff.HessianConfig(loglik_obj, diffresult, [vec(coefs_mlogit); vec(coefs_memb)], ForwardDiff.Chunk{18}())
    diffresult = ForwardDiff.hessian!(diffresult, loglik_obj, [vec(coefs_mlogit); vec(coefs_memb)], cfgH)

    gradient = DiffResults.gradient(diffresult)::Vector{Float64}
    hessian = DiffResults.hessian(diffresult)::Matrix{Float64}
    vcov = inv(hessian)
    if any(diag(vcov) .< 0.0)
        @warn "Main diagonale of VCOV has negative entries. Try gradient-based optimization."
    end

    loglik = -DiffResults.value(diffresult)::Float64
    loglik_0 = -loglik_lc(zeros(k_utility * n_classes + (k_membership + 1) * (n_classes - 1)), mat_X, mat_memb, n_id, vec_choice, n_classes, k_utility, k_membership, nrows, ll_n, exp_mat_utils, exp_Xb_share, id_map, chid_map)

    # shares calculation
    for c in 1:(n_classes-1)
        exp_Xb_share[:, c] .= mat_memb * coefs_memb[:, c]
    end

    exp_Xb_share = exp.(exp_Xb_share)

    # Class shares
    probs_memb = exp_Xb_share ./ sum(exp_Xb_share, dims=2)
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
        score=nothing,
        shares=shares,
        start=zeros(k_utility * n_classes + (k_membership + 1) * (n_classes - 1)),
        startloglikelihood=loglik_0,
        time=time() - start_time,
        vcov=vcov
    )
    return r
end