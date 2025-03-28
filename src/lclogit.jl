function get_used_memory_percentage()
    # Command to get memory usage percentage
    cmd = `bash -c "free | awk '/Mem:/ {print \$3/\$2 * 100}'"`

    # Execute the command and read the output
    output = read(cmd, String)

    # Convert output to a float and return
    return parse(Float64, output)
end


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

function loglik_lc(theta::Vector, mat_X::Matrix{Float64}, mat_memb::Matrix{Float64}, Xb_share_tmp::Matrix{Real}, vec_id::Vector{Int64}, n_id::Int64, vec_chid::Vector{Int64}, vec_choice::BitVector, n_classes::Int64, k_utility::Int64, k_membership::Int64, nrows::Int64)
    # initialise [N_subject x 1] vector of each subject's log-likelihood 
    ll_n = zeros(eltype(theta), n_id)

    # coefficients mlogit
    mat_coefs_mlogit_ll = reshape(theta[begin:(n_classes*k_utility)], k_utility, n_classes)
    # coefficients membership
    coefs_memb_ll = reshape(theta[(n_classes * k_utility+1):end], (k_membership + 1), (n_classes - 1))

    # matrix of utilities
    exp_mat_utils = exp.(mat_X * mat_coefs_mlogit_ll)

    for c in 1:(n_classes-1)
        Xb_share_tmp[:, c] .= mat_memb * coefs_memb_ll[:, c]
    end

    exp_Xb_share = exp.(Xb_share_tmp)

    # Class shares
    probs_memb = exp_Xb_share ./ sum(exp_Xb_share, dims=2)

    # Compute conditional probabilities and log-likelihood
    cond_probs_memb = zeros(eltype(theta), nrows, n_classes)

    # loop over subjects
    for n in unique(vec_id)
        # read in data rows pertaining to subject n & store in a matrix suffixed _n
        Y_n = vec_choice[vec_id.==n]
        EXP_n = exp_mat_utils[vec_id.==n, :]
        probs_memb_n = probs_memb[vec_id.==n, :]
        chid_n = vec_chid[vec_id.==n]

        # initialise [N_n x nclasses] matrix of conditional choice probabilities where N_n is # of data rows for subject n
        cond_probs_memb_n = zeros(eltype(theta), Base.length(Y_n), n_classes)

        # loop over choice sets t
        for t in unique(chid_n)
            # read in data rows pertaining to choice set t
            EXP_nt = EXP_n[chid_n.==t, :]

            # fill in choice probabilities	
            cond_probs_memb_n[chid_n.==t, :] = EXP_nt ./ sum(EXP_nt, dims=1) #colsum
        end

        # [1 x nclasses] vector of the likelihood of actual choice sequence
        ProbSeq_n = exp.(sum(log.(cond_probs_memb_n) .* Y_n, dims=1)) #colsum

        # compute subject n's log-likelihood
        ll_n[n] = log.(ProbSeq_n * probs_memb_n[1, :])[1, 1]

        # fill in subject n's conditional membership probabilities
        cond_probs_memb[vec_id.==n, :] .= (ProbSeq_n .* probs_memb_n[1:1, :]) ./ (ProbSeq_n * probs_memb_n[1:1, :]')
    end

    return -sum(ll_n)
end

function StatsAPI.fit(::Type{LCLmodel},
    formula::FormulaTerm,
    df,
    n_classes::Int64;
    start_memb::Union{Nothing,Matrix{Float64}} = start_memb,
    start_mnl::Union{Nothing,Matrix{Float64}} = start_mnl,
    indices::XlogitIndices = xlogit_indices(),
    method::Symbol = :em,
    quietly::Bool = false,
    varname_samplesplit = nothing,
    max_iter::Int64 = 1000,
    ltolerance::Float64 = 1e-7,
    multithreading::Bool = false,
    optim_method = BFGS(),
    optim_options = Optim.options()
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

    # for changing to array-based code
    lcl_first_by_id::BitVector = let seen = Set()
        [id in seen ? false : (push!(seen, id); true) for id in vec_id]
    end

    # Chids
    vec_chid::Vector{Int64} = df[!, indices.chid]
    # make sure that vec_chid can be used to index vectors of length length(unique(vec_chid))
    # unique(vec_chid) != 1:length(unique(vec_chid)) && 
    remap_to_indices_chid!(vec_chid)
    # idx_map = create_index_map(vec_chid)
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
    Xb_share_tmp::Matrix{Real} = zeros(Real, nrows, n_classes)
    # matrix of utilities
    exp_mat_utils::Matrix{Float64} = zeros(Float64, nrows, n_classes)
    # [1 x nclasses] vector of the likelihood of actual choice sequence
    ProbSeq_n::Matrix{Float64} = zeros(Float64, 1, n_classes)

    if method == :em

        ### split sample

        prop = 1 / n_classes

        function create_lcl_s(vec_id, lcl_first_by_id, n_classes, prop)
            # Get unique IDs and their first positions
            unique_ids = unique(vec_id)
            first_positions = findall(true, lcl_first_by_id)
            
            # Assign random class probabilities to each ID
            id_to_class = Dict{eltype(vec_id), Int}()
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
            create_lcl_s(vec_id, lcl_first_by_id, n_classes, prop)
        end

        lcl_H::Matrix{Float64} = Matrix(undef, nrows, n_classes)

        sumll = Float64[]

        for s in 1:n_classes
            coefs_mlogit[:, s] .= coef(mlogit(formula, df[lcl_s .== s, :]))
        end

        function cond_probs_ll()
            # initialise [N_subject x 1] vector of each subject's log-likelihood 
            ll_n = zeros(n_id)

            # matrix of utilities
            exp_mat_utils .= exp.(mat_X * coefs_mlogit)

            for c in 1:(n_classes-1)
                Xb_share_tmp[:, c] .= mat_memb * coefs_memb[:, c]
            end

            exp_Xb_share = exp.(Xb_share_tmp)

            # Class shares
            probs_memb = exp_Xb_share ./ sum(exp_Xb_share, dims=2)

            # loop over subjects
            for n in unique(vec_id)
                # read in data rows pertaining to subject n & store in a matrix suffixed _n
                Y_n = vec_choice[vec_id.==n]
                EXP_n = exp_mat_utils[vec_id.==n, :]
                probs_memb_n = probs_memb[vec_id.==n, :]
                chid_n = vec_chid[vec_id.==n]

                # initialise [N_n x nclasses] matrix of conditional choice probabilities where N_n is # of data rows for subject n
                cond_probs_memb_n = zeros(Base.length(Y_n), n_classes)

                # loop over choice sets t
                for t in unique(chid_n)
                    # read in data rows pertaining to choice set t
                    EXP_nt = EXP_n[chid_n.==t, :]

                    # fill in choice probabilities	
                    cond_probs_memb_n[chid_n.==t, :] .= EXP_nt ./ sum(EXP_nt, dims=1) #colsum
                end

                # [1 x nclasses] vector of the likelihood of actual choice sequence
                ProbSeq_n .= exp.(sum(log.(cond_probs_memb_n) .* Y_n, dims=1)) #colsum

                # compute subject n's log-likelihood
                ll_n[n] = log.(ProbSeq_n * probs_memb_n[1, :])[1, 1]

                # fill in subject n's conditional membership probabilities
                cond_probs_memb[vec_id.==n, :] .= (ProbSeq_n .* probs_memb_n[1:1, :]) ./ (ProbSeq_n * probs_memb_n[1:1, :]')
            end
            push!(sumll, sum(ll_n))
            for x in 1:n_classes
                lcl_H[:, x] .= cond_probs_memb[:, x]
            end
        end

        cond_probs_ll()

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
                opt_fmlogit = Optim.optimize(theta -> loglik_fmlogit(theta, lcl_H[lcl_first_by_id, :], mat_memb[lcl_first_by_id, :], fill(1.0, n_id), 1, n_classes), vec(coefs_memb), Newton(), Optim.Options(), autodiff=:forward)
                coefs_memb .= reshape(Optim.minimizer(opt_fmlogit), 2, n_classes-1)
            end

            cond_probs_ll()

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

        opt = Optim.optimize(theta -> loglik_lc(theta, mat_X, mat_memb, Xb_share_tmp, vec_id, n_id, vec_chid, vec_choice, n_classes, k_utility, k_membership, nrows), [vec(coefs_mlogit); vec(coefs_memb)], optim_method, autodiff=:forward, optim_options)
        coefficients = Optim.minimizer(opt)
        coefs_mlogit .= reshape(coefficients[1:(k_utility*n_classes)], k_utility, n_classes)
        coefs_memb .= reshape(coefficients[(k_utility*n_classes+1):end], (k_membership + 1), (n_classes - 1))

        converged = Optim.converged(opt)
        iter = Optim.iterations(opt)

        gradient = ForwardDiff.gradient(theta -> loglik_lc(theta, mat_X, mat_memb, Xb_share_tmp, vec_id, n_id, vec_chid, vec_choice, n_classes, k_utility, k_membership, nrows), coefficients)
        hessian::Matrix{Float64} = ForwardDiff.hessian(theta -> loglik_lc(theta, mat_X, mat_memb, Xb_share_tmp, vec_id, n_id, vec_chid, vec_choice, n_classes, k_utility, k_membership, nrows), coefficients)
        vcov = inv(hessian)

    else
        error("Unknown method. Choose :em or :gradient")
    end

    shares::Vector{Float64} = let m = vec(mean(probs_memb[lcl_first_by_id, :], dims=1))
        ForwardDiff.value.(ForwardDiff.value.(m)) # no clue why, but for some reason it needs two of those
    end
    
    n_coefficients = (k_membership+1)*(n_classes-1) + k_utility*n_classes

    loglik = -loglik_lc([vec(coefs_mlogit); vec(coefs_memb)], mat_X, mat_memb, Xb_share_tmp, vec_id, n_id, vec_chid, vec_choice, n_classes, k_utility, k_membership, nrows)
    loglik_0 = -loglik_lc(zeros(k_utility * n_classes + (k_membership + 1) * (n_classes - 1)), mat_X, mat_memb, Xb_share_tmp, vec_id, n_id, vec_chid, vec_choice, n_classes, k_utility, k_membership, nrows)

    r = LCLmodel(
        # coef=coefficients,
        coef_memb=coefs_memb,
        coef_mnl=coefs_mlogit,
        # coefnames=coefnames,
        coefnames_memb=isempty(coefnames_membership) ? ["constant";] : [coefnames_membership; "constant"],
        coefnames_mnl=coefnames_utility,
        converged=converged,
        dof=n_coefficients,
        formula=formula,
        formula_origin=formula_origin,
        formula_schema=formula_schema,
        hessian=(@isdefined hessian) ? hessian : fill(0.0, n_coefficients, n_coefficients),
        iter=iter,
        loglikelihood=loglik,
        method=method,
        # model_membership=model_memb,
        # models_mnl=models_mnl,
        nchids=n_chid,
        nclasses=n_classes,
        nids=n_id,
        nullloglikelihood=loglik_0,
        optim= (@isdefined opt) ? opt : nothing,
        responsename=response_name,
        score=nothing,
        shares=shares,
        start=zeros(k_utility * n_classes + (k_membership + 1) * (n_classes - 1)),
        startloglikelihood=loglik_0,
        time=time() - start_time,
        vcov=(@isdefined vcov) ? vcov : fill(0.0, n_coefficients, n_coefficients)
    )
    return r
end