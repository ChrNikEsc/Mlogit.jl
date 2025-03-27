function fmlogit(
    formula::FormulaTerm,
    df::DataFrame;
    start::Union{Matrix{Float64},Nothing}=nothing,
    weights::Union{Symbol,Nothing}=nothing,
    return_Hessian_vcov::Bool=true,
    method=Newton(),
    multithreading::Bool=false,
    skip_optimization::Bool=false,
    optim_options=optim_options=Optim.Options(),
    )::FMLmodel

    # Start time
    start_time = time()

    formula_schema, y::Matrix{Float64}, X::Matrix{Float64}, varnames_y::Vector{String}, varnames_X::Vector{String}, n::Int64, j::Int64, k::Int64 = prepare_data_fmlogit(formula, df)

    start::Matrix{Float64} = isnothing(start) ? zeros(Float64, (k + 1), (j - 1)) : start[:, 1:(j-1)]

    # Weights
    vec_weights::Vector{Float64} = isnothing(weights) ?
                                   ones(Float64, n) :
                                   let w::Vector{Float64} = df[!, weights]
                                        (w ./ sum(w)) .* n
                                    end

    if !skip_optimization
        opt = Optim.optimize(theta -> loglik_fmlogit(theta, y, X, vec_weights, k, j), vec(start), method, optim_options, autodiff=:forward)
    end

    coefficients::Matrix{Float64} = if skip_optimization
        start
    else
        reshape(Optim.minimizer(opt), (k + 1), (j - 1))
    end
    
    converged::Bool = if skip_optimization
        false
    else
        Optim.converged(opt)
    end
    
    iter::Int64 = if skip_optimization
        -1
    else
        Optim.iterations(opt)
    end


    # Pre-allocate gradient array
    gradient = zeros(Float64, Base.length(coefficients))
    ForwardDiff.gradient!(gradient, theta -> loglik_fmlogit(theta, y, X, vec_weights, k, j), vec(coefficients))  # In-place gradient computation

    loglik::Float64 = -loglik_fmlogit(coefficients, y, X, vec_weights, k, j)
    loglik_0::Float64 = -loglik_fmlogit(zeros((k + 1) * (j - 1)), y, X, vec_weights, k, j)
    loglik_start::Float64 = -loglik_fmlogit(start, y, X, vec_weights, k, j)
    n_coefficients::Int64 = Base.length(coefficients)

    if !converged && !skip_optimization
        @warn "fmlogit did not converge! Not returning Hessian and Variance-Covariance Matrix"
    end

    hessian = vcov = fill(NaN, n_coefficients, n_coefficients)

    # Return the FMLmodel object
    r = FMLmodel(
        coef=hcat(coefficients, zeros(k + 1)),
        coefnames=reshape(["y_" * y * ".X_" * x for y in varnames_y[begin:end] for x in [varnames_X; "constant"]], (k + 1), j),
        converged=converged,
        depvar=y,
        df_hash=hash(df),
        dof=n_coefficients,
        estfun=nothing,
        fitted=compute_fitted_values_fmlogit(coefficients, k, j, X),
        formula=formula,
        formula_schema=formula_schema,
        hessian=hessian,
        hessian_method="ForwardDiff",
        indices=nothing,
        iter=iter,
        loglikelihood=loglik,
        nclusters=nothing,
        nchids=n,
        nids=n,
        nullloglikelihood=loglik_0,
        optim=skip_optimization ? nothing : opt,
        score=gradient,
        submodels=[FMLsubmodel(coefficients[:, i], [varnames_X; "constant"], term(Symbol(varnames_y[i])) ~ foldl(+, term.(([varnames_X; "constant"]))), varnames_y[i], nothing, Vcov.simple()) for i in 1:(j-1)],
        start=start,
        startloglikelihood=loglik_start,
        time=time() - start_time,
        vcov=vcov,
        vcov_type=Vcov.simple()
    )
    return r
end;

function loglik_fmlogit(theta, y, X, vec_weights, k, j)
    betamat = [transpose(reshape(theta, k + 1, j - 1)); zeros(k + 1)']
    L = zeros(eltype(theta), j)

    X_betamat = X * transpose(betamat)

    log_sum_exp_Xb = log.(sum(exp.(X_betamat), dims=2))

    function call_L(i)
        return sum(view(y, :, i) .* (view(X_betamat, :, i) - log_sum_exp_Xb))
    end

    # if multithreading
    #     Threads.@threads for i in 1:j
    #         L[i] = call_L(i) * vec_weights[i]
    #     end
    # else
        for i in 1:j
            L[i] = call_L(i) * vec_weights[i]
        end
    # end

    return -sum(L)
end


# fit_fmlogit()