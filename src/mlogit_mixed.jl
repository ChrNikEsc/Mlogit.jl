# doit.jl
# This script checks data, transforms it, performs estimation, and prints results.
# It is intended to be included by mxlmsl.jl.
# Translated from Matlab code by Kenneth Train.

using MAT     # MAT.jl is used if DRAWTYPE=5

"""
    fit_mlogit_mixed(df, XMAT, NDRAWS, DRAWTYPE, WANTWGT, IDWGT,
         NV, NF, IDV, IDF, F, B, W,
         MAXITERS, PARAMTOL, LLTOL, output_filename,
         NAMESF, NAMES)

Main processing function that encapsulates the original script's logic.
It performs data preparation, simulation, optimization, and results reporting.

Returns:
    A tuple containing key results:
    (paramhat_final, fval_final, grad_final_at_opt, hessian_at_opt, inv_hessian_at_opt, stderr_final,
     NPARAM, WGT, X, XF, S, DR,
     NALTMAX, n_chidMAX,, inds_B_to_estimate, elapsed_time_minutes, opt_result_obj)
"""
function fit_mlogit(
    mat_X,
    vec_choice::BitVector,
    randdist::Vector{Union{Nothing,Symbol}},
    coef_start::Vector{Float64},          # Initial values for fixed coefficients
    vec_id::Vector{Int64},
    vec_chid::Vector{Int64},
    vec_weights_chid::Vector{Float64},
    draws::Tuple{Int,Union{Symbol,String}};
    optim_options=Optim.Options(
        extended_trace=true,
        show_trace=false,
        store_trace=true,
        f_abstol=-1,# in tests with Train's data, this led to premature convergence even if at 0 when starting values were not perfect
        f_reltol=-1,
        # x_abstol=-1,
        # x_reltol=-1,
        # g_abstol=1e-15
    )
)
    n_id::Int64 = maximum(vec_id)
    n_chid::Int64 = maximum(vec_chid)


    # # --- Check input data and specifications ---
    # println("Checking inputs.") # Should also go to output_filename
    # ok = check_inputs(n_id, n_chid, IDV, NV, NAMES, B, W, IDF, NF, NAMESF, F,
    #     DRAWTYPE, NDRAWS, WANTWGT, IDWGT, XMAT) # Call the main function from check.jl

    # if ok
    #     println("Inputs have been checked and look fine.") # To output_filename
    # else
    #     println("Input check failed. Terminating.") # To output_filename
    #     error("Terminating due to input check failure.")
    # end

    allowed_dist_types = [:normal, :lognormal, :truncnormal, :S_B, :normal0mn, :triangular]

    if !all((randdist .== nothing) .| (randdist .∈ Ref(allowed_dist_types)))
        error("Error: randdist contains invalid distribution types. Allowed types are: $allowed_dist_types, or 'nothing'")
    end

    NDRAWS::Int64 = draws[1] # Number of draws
    DRAWTYPE::Union{Symbol,String} = draws[2] # Type of draws

    NV::Int64 = sum(randdist .!= nothing) # Number of random coefficients
    NF::Int64 = sum(randdist .== nothing) # Number of fixed coefficients

    # F::Vector{Float64} = coef_start[1:NF] # Fixed coefficients
    # B::Vector{Float64} = coef_start[NF+1:NF+NV] # Random coefficients
    # W::Vector{Float64} = coef_start[NF+NV+1:end] # Scale parameters for random coefficients

    # Train's Matlab code works with (F, B, W) param vector
    # The following lines extract these from coef_start which follows the formula order
    # perm_coefs saves this order for returning the coefficients in the original order
    perm_coefs::Vector{Int64} = Int64[]

    F::Vector{Float64} = Float64[]
    B::Vector{Float64} = Float64[]
    W::Vector{Float64} = Float64[]
    for i in eachindex(randdist)
        lf = length(F)
        lw = length(W)
        l = length(F) + length(B) + length(W)
        if isnothing(randdist[i]) # not random coef
            push!(F, coef_start[l+1])
            push!(perm_coefs, lf + 1)
        else # if random coef
            if randdist[i] .!= :normal0mn
                push!(B, coef_start[l+1]) # b, but not if :normal0mmn
                push!(perm_coefs, NF + lw + 1) # Store the index of the random coefficient
                push!(W, coef_start[l+2]) # w
                push!(perm_coefs, NF + lw + 1 + NV) # Store the index of the scale parameter
            else
                push!(W, coef_start[l+1]) # w
                push!(perm_coefs, NF + lw + NV) # Store the index of the scale parameter
            end
        end
    end

    randdist_random::Vector{Symbol} = randdist[randdist.!=nothing] # Filter out non-random coefficients

    # --- Initial values for the log-likelihood function ---
    initial_neg_ll_val = 0.0
    if NV > 0
        initial_neg_ll_val = -1e10
    end

    WANTWGT = Int64(any(vec_weights_chid .!= 1.0)) # If all weights are 1, set to 0


    # --- Create variables based on inputs ---
    # Most  variables can be used directly if their names don't clash
    # with variables created inside. For clarity, one might rename, but here we use them.

    # --- Calculate NALTMAX: Maximum number of alternatives in any choice situation ---
    # Person identifier column from XMAT.
    # vec_id = XMAT[:, 1] # Local variable

    nn_cs_calc = zeros(Int, n_chid) # Stores count of alternatives for each choice situation.
    for n = 1:n_chid
        nn_cs_calc[n] = sum(vec_chid .== n)
    end
    NALTMAX::Int64 = maximum(nn_cs_calc)

    # --- Calculate n_chidMAX: Maximum number of choice situations faced by any person ---
    nn_p_calc = zeros(Int, n_id) # Stores count of choice situations for each person.
    for n = 1:n_id
        kdices = (vec_id .== n)
        cs_person_n = vec_chid[kdices]
        if !isempty(cs_person_n)
            nn_p_calc[n] = 1 + Int(round(cs_person_n[end])) - Int(round(cs_person_n[1]))
        else
            nn_p_calc[n] = 0
        end
    end
    n_chidMAX::Int64 = maximum(nn_p_calc)


    chids_per_id = let
        id_to_chids_map = Dict{eltype(vec_id),Vector{eltype(vec_chid)}}()
        for i in eachindex(vec_id)
            push!(get!(id_to_chids_map, vec_id[i], Vector{eltype(vec_chid)}()), vec_chid[i])
        end
        [unique(id_to_chids_map[ind]) for ind in unique(vec_id)]
    end

    # --- WGT: Weights for each person ---
    WGT = Float64[]
    if WANTWGT == 1
        @warn("Averaging weights per person, using one weight per person.")
        temp_wgt = [mean(vec_weights_chid[chids_per_id[id]]) for id in 1:n_id]
        sum_temp_wgt = sum(temp_wgt)
        if sum_temp_wgt == 0.0 && n_id > 0
            println("Warning: Sum of weights is zero. Cannot normalize. Using raw weights (possibly all zero).") # To console
            WGT = temp_wgt
        elseif n_id > 0 # Ensure n_id > 0 before division
            WGT = temp_wgt .* (n_id / sum_temp_wgt)
        else # n_id is 0
            WGT = Float64[]
        end
    else
        WGT = ones(Float64, n_id) # Default to unit weights if WANTWGT is not 1
    end

    # --- Data arrays ---
    dim1_X_val = NALTMAX > 0 ? NALTMAX - 1 : 0
    X = zeros(Float64, NALTMAX - 1, n_chidMAX, NV, n_id) # Explanatory variables with random coefficients for each choice situation, for each person
    XF = zeros(Float64, NALTMAX - 1, n_chidMAX, NF, n_id) # Explanatory variables with fixed coefficients for all choice situations, for each person
    S = zeros(Float64, NALTMAX - 1, n_chidMAX, n_id) # Identification of the alternatives in each choice situation, for each person

    @inbounds Threads.@threads for n_person_loop = 1:n_id  # Loop over people
        person_indices = (vec_id .== n_person_loop)
        if !any(person_indices)
            continue
        end

        local_person_choice_situations = convert(Vector{Int}, round.(vec_chid[person_indices]))
        local_person_chosen_alt_flags = convert(Vector{Int}, round.(vec_choice[person_indices]))

        local_person_vars_rand = mat_X[person_indices, .!isnothing.(randdist)]

        if NF > 0
            local_person_vars_fixed = mat_X[person_indices, isnothing.(randdist)]
        else
            local_person_vars_fixed = Matrix{Float64}(undef, sum(person_indices), 0)
        end

        if isempty(local_person_choice_situations)
            continue
        end

        cs_start_person_n = local_person_choice_situations[1]
        # cs_end_person_n = local_person_choice_situations[end] # This might not be robust if CS are not contiguous
        # Number of unique choice situations for this person
        num_cs_for_person = nn_p_calc[n_person_loop]


        cs_idx_person = 0
        # Iterate based on the actual number of choice situations for this person
        # from nn_p_calc, assuming they are numbered 1 to num_cs_for_person internally for X, S, XF
        # The original code iterates t_cs_loop = cs_start_person_n:cs_end_person_n,
        # which implies choice situations are contiguous and start from cs_start_person_n.
        # We'll stick to that logic for now.
        # A more robust way would be to map original CS numbers to 1:num_cs_for_person.

        unique_cs_numbers = unique(local_person_choice_situations)

        for t_cs_original_number in unique_cs_numbers # Iterate over unique choice situations for the person
            cs_idx_person += 1 # This is the 1-based index for X, S, XF for this person
            if cs_idx_person > n_chidMAX || cs_idx_person > num_cs_for_person
                # println("Warning: cs_idx_person ($cs_idx_person) exceeds n_chidMAX ($n_chidMAX) or num_cs_for_person ($num_cs_for_person) for person $n_person_loop.")
                continue
            end

            in_current_cs = (local_person_choice_situations .== t_cs_original_number)
            num_alts_cs = sum(in_current_cs)

            if num_alts_cs <= 1 || dim1_X_val == 0
                continue
            end
            k_non_chosen = num_alts_cs - 1
            if k_non_chosen > size(S, 1)
                k_non_chosen = size(S, 1)
            end
            if k_non_chosen <= 0
                continue
            end

            S[1:k_non_chosen, cs_idx_person, n_person_loop] .= 1.0

            chosen_alt_flag_cs = (local_person_chosen_alt_flags[in_current_cs] .== 1)

            if sum(chosen_alt_flag_cs) != 1
                S[1:k_non_chosen, cs_idx_person, n_person_loop] .= 0.0
                # println("Warning: Person $n_person_loop, CS $t_cs_original_number has $(sum(chosen_alt_flag_cs)) chosen alts. Expected 1.")
                continue
            end

            if NV > 0 && size(X, 3) > 0
                vars_rand_current_cs = local_person_vars_rand[in_current_cs, :]
                chosen_vars_rand = reshape(vars_rand_current_cs[chosen_alt_flag_cs, :], 1, NV)
                nonchosen_vars_rand = vars_rand_current_cs[.!chosen_alt_flag_cs, :]

                actual_k_non_chosen_data = size(nonchosen_vars_rand, 1)
                k_to_use = min(k_non_chosen, actual_k_non_chosen_data)

                if k_to_use > 0 && k_to_use <= size(X, 1)
                    X[1:k_to_use, cs_idx_person, :, n_person_loop] = nonchosen_vars_rand[1:k_to_use, :] .- repeat(chosen_vars_rand, k_to_use, 1)
                    if k_to_use < k_non_chosen && k_non_chosen <= size(S, 1)
                        S[k_to_use+1:k_non_chosen, cs_idx_person, n_person_loop] .= 0.0
                    end
                elseif k_non_chosen > 0 && k_non_chosen <= size(S, 1)
                    S[1:k_non_chosen, cs_idx_person, n_person_loop] .= 0.0
                end
            end

            if NF > 0 && size(XF, 3) > 0
                vars_fixed_current_cs = local_person_vars_fixed[in_current_cs, :]
                chosen_vars_fixed = reshape(vars_fixed_current_cs[chosen_alt_flag_cs, :], 1, NF)
                nonchosen_vars_fixed = vars_fixed_current_cs[.!chosen_alt_flag_cs, :]

                actual_k_non_chosen_data_f = size(nonchosen_vars_fixed, 1)
                k_to_use_f = min(k_non_chosen, actual_k_non_chosen_data_f)

                if k_to_use_f > 0 && k_to_use_f <= size(XF, 1)
                    XF[1:k_to_use_f, cs_idx_person, :, n_person_loop] = nonchosen_vars_fixed[1:k_to_use_f, :] .- repeat(chosen_vars_fixed, k_to_use_f, 1)
                end
            end
        end
    end



    # --- Draw Generation ---
    local DR::Array{Float64,3}

    # If DRAWTYPE=5, load the draws here.
    if typeof(DRAWTYPE) == String
        draws_filename = DRAWTYPE
        println("Attempting to load draws from '$draws_filename'.")
        # Ensure NV, n_id, NDRAWS are defined before this block if used for validation here.
        # They are defined above, so this is okay.
        try
            mat_contents = matread(draws_filename)
            if haskey(mat_contents, "DR")
                DR_loaded_from_mat = mat_contents["DR"]
                # Perform dimension check
                if ndims(DR_loaded_from_mat) == 3 && size(DR_loaded_from_mat, 1) == NV && size(DR_loaded_from_mat, 2) == n_id && size(DR_loaded_from_mat, 3) == NDRAWS
                    DR = convert(Array{Float64,3}, DR_loaded_from_mat)
                    println("Draws loaded successfully from $draws_filename with dimensions ($NV x $n_id x $NDRAWS).")
                    # Add to output file
                    # open(output_filename, "a") do io
                    #     println(io, "INFO: Using draws loaded from $draws_filename.")
                    # end
                else
                    println("Error: Loaded DR from $draws_filename has incorrect dimensions.")
                    println("Expected: ($NV x $n_id x $NDRAWS), Got: $(size(DR_loaded_from_mat))")
                    println("Please ensure 'mydraws.mat' contains a variable 'DR' with these dimensions.")
                    error("Terminating.")
                end
            else
                println("Error: Variable 'DR' not found in $draws_filename.")
                error("Terminating.")
            end
        catch e
            println("Error loading $draws_filename: $e")
            println("Ensure 'mydraws.mat' is in the current directory and readable.")
            error("Terminating.")
        end
    elseif typeof(DRAWTYPE) == Symbol
        println("Creating draws (DRAWTYPE=$string(DRAWTYPE)).")
        # Assuming makedraws_core is refactored to take arguments
        DR_temp = makedraws_core(NDRAWS, n_id, NV, DRAWTYPE, randdist_random)

        if DR_temp !== nothing && ndims(DR_temp) == 3 && size(DR_temp, 1) == NDRAWS && size(DR_temp, 2) == n_id && size(DR_temp, 3) == NV
            DR = permutedims(DR_temp, [3, 2, 1]) # NV x n_id x NDRAWS
        else
            println("Error: makedraws_core did not return expected 3D array (NDRAWS x n_id x NV). Got: $(DR_temp === nothing ? "nothing" : size(DR_temp))")
            DR = Array{Float64,3}(undef, NV, n_id, NDRAWS) # Fallback empty
        end
    end

    # Checks for DR (used to be in check.jl)
    if !isa(DR, AbstractArray) || ndims(DR) != 3
        print_error("DR for DRAWTYPE=5 is not correctly defined as a 3D array.")
        return false
    end
    if size(DR, 1) != NV || size(DR, 2) != n_id || size(DR, 3) != NDRAWS
        print_error("The DR array has dimensions $(size(DR)), but should be NVxn_idxNDRAWS ($(NV)x$(n_id)x$(NDRAWS)).")
        return false
    end

    # --- Initial Parameters ---
    param_F_local = F
    inds_B_to_estimate = [v ∈ [:normal, :lognormal, :S_B, :truncnormal, :triangular] for v in randdist_random]
    param_B_local = B#[inds_B_to_estimate]
    param_W_local = W

    initial_param_local = Vector{Float64}()
    if NV > 0 && NF > 0
        initial_param_local = [param_F_local; param_B_local; param_W_local]
    elseif NV > 0 && NF == 0
        initial_param_local = [param_B_local; param_W_local]
    elseif NV == 0 && NF > 0
        initial_param_local = param_F_local
    else # NV == 0 && NF == 0
        println("Model has no explanatory variables (NF=0, NV=0).")
        # Depending on desired behavior, could error out or proceed if NPARAM becomes 0.
        # error("Terminating due to no explanatory variables.")
    end
    NPARAM = length(initial_param_local)

    if NPARAM == 0 && (NV > 0 || NF > 0)
        println("Warning: NPARAM is 0, but NV or NF is positive. Check parameter setup (F, B, W lengths and dist types).")
    end

    # init c_coeffs and derivatives
    c_coeffs = similar(DR)
    V_diff = zeros(eltype(DR), NALTMAX - 1, n_chidMAX, n_id, NDRAWS)
    p_person = zeros(Float64, n_id)
    pp_choice_sit_draw = zeros(eltype(DR), 1, n_chidMAX, n_id, NDRAWS)
    deriv_C_b = similar(DR)
    deriv_C_w = similar(DR)

    optim_fg! = (F, G, param) -> fg!(F, G, param, NF, NV, randdist_random, WANTWGT, WGT,
        n_id, NDRAWS, NALTMAX, n_chidMAX, X, S, XF, DR, c_coeffs, V_diff, p_person, pp_choice_sit_draw, deriv_C_b, deriv_C_w)

    start_time = time()

    # Initialize result variables
    paramhat_final_val = zeros(Float64, NPARAM)
    fval_final_val = NaN
    grad_final_at_opt = Float64[]
    inv_hessian_at_opt = zeros(0, 0)
    elapsed_time_minutes_val = 0.0
    opt_result_obj::Union{Optim.OptimizationResults,Nothing} = nothing

    println("Using BFGS with HagerZhang line search.")
    # could try α0=1e-7 or so in InitialHagerZhang() if problems with exp() overflow
    optimizer_obj = BFGS(alphaguess=Optim.LineSearches.InitialHagerZhang(), linesearch=Optim.LineSearches.HagerZhang())

    opt_result_obj = Optim.optimize(Optim.only_fg!(optim_fg!), initial_param_local, optimizer_obj, optim_options)

    display(opt_result_obj)

    elapsed_time_minutes_val = (time() - start_time) / 60

    if opt_result_obj !== nothing
        paramhat_final_val = Optim.minimizer(opt_result_obj)
        fval_final_val = -Optim.minimum(opt_result_obj)
        grad_final_at_opt = opt_result_obj.trace[end].metadata["g(x)"]
        inv_hessian_at_opt = opt_result_obj.trace[end].metadata["~inv(H)"]
        # diag_ihess = diag(inv_hessian_at_opt)
        # stderr_final_val = sqrt.(max.(0.0, diag_ihess))
    else # Optimization failed or did not run
        println("Optimization did not converge or failed.")
        paramhat_final_val = copy(initial_param_local) # Or fill with NaNs
        fval_final_val = initial_neg_ll_val
    end

    hessian = inv(inv_hessian_at_opt) # Inverse of the inverse Hessian

    # --- Post-estimation processing (standard errors, etc.) ---
    # open(output_filename, "a") do io
    #     @printf(io, "\nEstimation took %.4f minutes.\n\n", elapsed_time_minutes_val)

    #     if NPARAM == 0
    #         @printf(io, "No optimization performed as NPARAM = 0.\n")
    #         @printf(io, "Value of the (negative) log-likelihood function: %.6f\n\n", fval_final_val)
    #     elseif opt_result_obj !== nothing
    #         if Optim.converged(opt_result_obj)
    #             @printf(io, "Convergence achieved.\n")
    #         else
    #             @printf(io, "Convergence not achieved. Reason: %s\n", Optim.iteration_limit_reached(opt_result_obj) ? "Iteration limit reached" : string(Optim.summary(opt_result_obj)))
    #         end
    #         if !isnan(fval_final_val)
    #             @printf(io, "Value of the log-likelihood function at result: %.6f\n\n", -fval_final_val)
    #         else
    #             @printf(io, "Final log-likelihood value not available.\n\n")
    #         end
    #     else
    #         @printf(io, "Optimization did not run or complete successfully.\n")
    #     end

    #     @printf(io, "Taking inverse of Hessian for standard errors.\n\n")


    #     neg_diagdices = diag_ihess .< -1e-9
    #     if any(neg_diagdices)
    #         @printf(io, "Warning: Negative elements on diagonal of inverse Hessian at indices %s. SEs will be NaN for these.\n", findall(neg_diagdices))
    #     end


    #     if !isempty(grad_final_at_opt) && size(inv_hessian_at_opt) == (NPARAM, NPARAM)
    #         opg_val = dot(grad_final_at_opt, inv_hessian_at_opt * grad_final_at_opt)
    #         @printf(io, "The value of grad'*inv(hessian)*grad is: %.6e\n\n", opg_val)
    #     else
    #         @printf(io, "Final gradient or inverse Hessian not available/compatible for OPG check.\n\n")
    #     end

    #     # --- Prepare parameters for printing ---
    #     fhat_print = Float64[]
    #     fsd_print = Float64[]
    #     bhat_print = NV > 0 ? zeros(Float64, NV) : Float64[] # Full B, including fixed ones
    #     bsd_print = NV > 0 ? zeros(Float64, NV) : Float64[]
    #     what_print = NV > 0 ? zeros(Float64, NV) : Float64[]
    #     wsd_print = NV > 0 ? zeros(Float64, NV) : Float64[]

    #     if NPARAM > 0 && !isempty(paramhat_final_val) && length(paramhat_final_val) == NPARAM
    #         current_idx_param = 0
    #         if NF > 0
    #             fhat_print = paramhat_final_val[1:NF]
    #             fsd_print = length(stderr_final_val) >= NF ? stderr_final_val[1:NF] : fill(NaN, NF)
    #             current_idx_param = NF
    #         end

    #         if NV > 0
    #             num_b_estimated = sum(inds_B_to_estimate)

    #             # Populate full bhat_print with original B values for non-estimated params
    #             bhat_print .= B
    #             bsd_print .= NaN # Default SE for non-estimated to NaN

    #             if num_b_estimated > 0
    #                 if length(paramhat_final_val) >= current_idx_param + num_b_estimated
    #                     bhat_estimated_part = paramhat_final_val[current_idx_param+1:current_idx_param+num_b_estimated]
    #                     bhat_print[inds_B_to_estimate] .= bhat_estimated_part
    #                 else
    #                     bhat_print[inds_B_to_estimate] .= NaN # Should not happen if NPARAM is correct
    #                 end
    #                 if length(stderr_final_val) >= current_idx_param + num_b_estimated
    #                     bsd_estimated_part = stderr_final_val[current_idx_param+1:current_idx_param+num_b_estimated]
    #                     bsd_print[inds_B_to_estimate] .= bsd_estimated_part
    #                 else
    #                     bsd_print[inds_B_to_estimate] .= NaN
    #                 end
    #             end
    #             current_idx_param += num_b_estimated

    #             if length(paramhat_final_val) >= current_idx_param + NV # W params are all estimated
    #                 what_print = paramhat_final_val[current_idx_param+1:current_idx_param+NV]
    #                 if length(stderr_final_val) >= current_idx_param + NV
    #                     wsd_print = stderr_final_val[current_idx_param+1:current_idx_param+NV]
    #                 else
    #                     wsd_print = fill(NaN, NV)
    #                 end
    #             else
    #                 what_print = fill(NaN, NV)
    #                 wsd_print = fill(NaN, NV)
    #             end
    #         end
    #     end

    #     # --- Print results to file ---
    #     @printf(io, "RESULTS\n\n")
    #     if NF > 0
    #         @printf(io, "FIXED COEFFICIENTS\n\n")
    #         @printf(io, "                        F\n")
    #         @printf(io, "                 ------------------\n")
    #         @printf(io, "                   Est        SE\n")
    #         for r_print = 1:NF
    #             @printf(io, "%-10s %10.4f %10.4f\n", NAMESF[r_print],
    #                 length(fhat_print) >= r_print ? fhat_print[r_print] : NaN,
    #                 length(fsd_print) >= r_print ? fsd_print[r_print] : NaN)
    #         end
    #         @printf(io, "\n")
    #     end

    #     if NV > 0
    #         @printf(io, "RANDOM COEFFICIENTS\n\n")
    #         @printf(io, "                               B                                      W\n")
    #         @printf(io, "                 ----------------------          -----------------------\n")
    #         @printf(io, "                   Est        SE                  Est        SE\n")
    #         for r_print = 1:NV
    #             @printf(io, "%-10s %10.4f %10.4f          %10.4f %10.4f\n",
    #                 NAMES[r_print],
    #                 length(bhat_print) >= r_print ? bhat_print[r_print] : NaN,
    #                 length(bsd_print) >= r_print ? bsd_print[r_print] : NaN,
    #                 length(what_print) >= r_print ? what_print[r_print] : NaN,
    #                 length(wsd_print) >= r_print ? wsd_print[r_print] : NaN)
    #         end
    #         @printf(io, "\n")

    #         if !isempty(c_coeffs) && NV > 0 && n_id > 0 && NDRAWS > 0
    #             C_for_stats = reshape(c_coeffs, NV, n_id * NDRAWS)
    #             @printf(io, "Distribution of coefficients in population implied by B-hat and W-hat.\n")
    #             @printf(io, "Using all %d draws.\n\n", NDRAWS)

    #             # dist_names_map = ["normal", "lognormal", "truncnormal", "S_B", "normal0mn", "triangular"]
    #             @printf(io, "                         Mean      StdDev     Share<0     Share=0\n")

    #             mean_C = vec(mean(C_for_stats, dims=2))
    #             std_C = vec(std(C_for_stats, dims=2))
    #             share_neg_C = vec(mean(C_for_stats .< 0, dims=2))
    #             share_zero_C = vec(mean(C_for_stats .== 0, dims=2))

    #             for r_print = 1:NV
    #                 dist_type = randdist[randdist.!==nothing][r_print]
    #                 @printf(io, "%-10s %-11s %10.4f %10.4f %10.4f %10.4f\n",
    #                     NAMES[r_print], string(dist_type),
    #                     mean_C[r_print], std_C[r_print], share_neg_C[r_print], share_zero_C[r_print])
    #             end
    #             @printf(io, "\n")
    #         end
    #     end # End if NV > 0

    #     @printf(io, "\nESTIMATED PARAMETERS AND FULL COVARIANCE MATRIX.\n")
    #     @printf(io, "The estimated values of the parameters are:\n")
    #     param_names_combined = Vector{String}()
    #     if NF > 0
    #         append!(param_names_combined, NAMESF)
    #     end
    #     if NV > 0 && !isempty(NAMES)
    #         temp_b_param_names = String[]
    #         for i_name = 1:NV
    #             if inds_B_to_estimate[i_name]
    #                 push!(temp_b_param_names, "B_" * NAMES[i_name])
    #             end
    #         end
    #         append!(param_names_combined, temp_b_param_names)

    #         w_param_names = ["W_" * NAMES[i] for i in 1:NV]
    #         append!(param_names_combined, w_param_names)
    #     end

    #     if NPARAM > 0 && !isempty(paramhat_final_val) && length(paramhat_final_val) == NPARAM && length(paramhat_final_val) == length(param_names_combined)
    #         for i = 1:NPARAM
    #             @printf(io, "%-20s : %12.6f\n", param_names_combined[i], paramhat_final_val[i])
    #         end
    #     elseif NPARAM > 0 && !isempty(paramhat_final_val) && length(paramhat_final_val) == NPARAM
    #         @printf(io, "Parameter names list length mismatch with NPARAM. Printing generic names.\n")
    #         for i = 1:NPARAM
    #             @printf(io, "PARAM_%-14d : %12.6f\n", i, paramhat_final_val[i])
    #         end
    #     else
    #         @printf(io, "(No parameters estimated or paramhat_final_val not available)\n")
    #     end

    #     @printf(io, "\nThe covariance matrix for these parameters is (inv(Hessian)):\n")
    #     if NPARAM > 0 && !isempty(inv_hessian_at_opt) && size(inv_hessian_at_opt, 1) == NPARAM && size(inv_hessian_at_opt, 2) == NPARAM && length(param_names_combined) == NPARAM
    #         header_str = "        " * join([@sprintf("%-10.10s", param_names_combined[idx]) for idx in 1:NPARAM], " ")
    #         @printf(io, "%s\n", header_str)
    #         for i_row = 1:NPARAM
    #             @printf(io, "%-8.8s", param_names_combined[i_row])
    #             for j_col = 1:NPARAM
    #                 @printf(io, " %10.4e", inv_hessian_at_opt[i_row, j_col])
    #             end
    #             @printf(io, "\n")
    #         end
    #     elseif NPARAM > 0 && !isempty(inv_hessian_at_opt) && size(inv_hessian_at_opt, 1) == NPARAM && size(inv_hessian_at_opt, 2) == NPARAM
    #         @printf(io, "Parameter names list length mismatch. Printing generic VCV matrix.\n")
    #         for i_row = 1:NPARAM
    #             @printf(io, "P%-7d", i_row)
    #             for j_col = 1:NPARAM
    #                 @printf(io, " %10.4e", inv_hessian_at_opt[i_row, j_col])
    #             end
    #             @printf(io, "\n")
    #         end
    #     else
    #         @printf(io, "(Covariance matrix not available or NPARAM=0)\n")
    #     end

    #     @printf(io, "\n\nYou can access the estimated parameters as variable paramhat_final_val,\n")
    #     @printf(io, "the gradient of the negative of the log-likelihood function as variable grad_final_at_opt,\n")
    #     @printf(io, "the Hessian of the negative of the log-likelihood function as variable hessian_at_opt,\n")
    #     @printf(io, "and the inverse of the Hessian as variable inv_hessian_at_opt.\n")

    # end # End open(output_filename, "a")

    # Return all relevant results
    # return paramhat_final_val, fval_final_val, grad_final_at_opt, hessian_at_opt, inv_hessian_at_opt, stderr_final_val,
    # NPARAM, WGT, X, XF, S, DR,
    # NALTMAX, n_chidMAX, inds_B_to_estimate, elapsed_time_minutes_val, opt_result_obj

    # Train's matlab code uses (F, B, W) order for parameters
    # permute back to order implied by formula
    paramhat_final_val .= paramhat_final_val[perm_coefs]
    grad_final_at_opt .= grad_final_at_opt[perm_coefs]
    hessian .= hessian[perm_coefs, perm_coefs]

    return opt_result_obj, paramhat_final_val, Optim.converged(opt_result_obj), Optim.iterations(opt_result_obj), fval_final_val, 0.0, 0.0, grad_final_at_opt, zeros(Float64, NF + NV + NV, NF + NV + NV), hessian, [0.0]

end

# Defines the loglik_fcn_grad function for calculating the negative log-likelihood
# and its gradient for the mixed logit model.
# This function is intended to be used with Optim.jl.
# Translated from Matlab code by Kenneth Train.
# llgrad2.m calculations have been integrated into this function.
# Re-written for improved performance with help of Google Gemini.

function fg!(
    F, # Argument from Optim.jl, not used for storing F since we return it
    G,          # Gradient vector to be mutated by this function
    param::AbstractVector{T_param},
    # Arguments corresponding to MATLAB globals
    NF::Int,
    NV::Int,
    # IDV::Matrix{Int},      # Assuming IDV is an NV x K matrix where IDV[:,2] contains distribution types
    randdist_random::Vector{Symbol}, # Vector of distribution types for each variable
    WANTWGT::Int,   # True if weights are to be used, false otherwise
    WGT::AbstractVector{T_data},
    n_id::Int,
    NDRAWS::Int,
    NALTMAX::Int,
    n_chidMAX::Int,
    X::AbstractArray{T_data,4},
    S::AbstractArray{T_data,3},
    XF::AbstractArray{T_data,4},
    DR::Union{AbstractArray{T_data,3},Nothing}, # Should be (NV, n_id, NDRAWS)
    c_coeffs::AbstractArray{T_param,3},  # Mutable: (NV, n_id, NDRAWS)
    V_diff::AbstractArray{T_param,4}, # Mutable: (NALTMAX-1, n_chidMAX, n_id, NDRAWS) # will store V_diff and exp(V_diff)
    p_person::Vector{T_param}, # Mutable: (n_id)
    pp_choice_sit_draw::AbstractArray{T_param,4}, # Mutable: (1, n_chidMAX, n_id, NDRAWS)
    deriv_C_b::AbstractArray{T_param,3}, # Mutable: (NV, n_id, NDRAWS)
    deriv_C_w::AbstractArray{T_param,3}  # Mutable: (NV, n_id, NDRAWS)
) where {T_param<:Real,T_data<:Real}

    # --- 1. Unpack parameters from `param` into f_params, b_full, and w_for_llgrad ---
    f_params = Vector{T_param}()
    b_full = zeros(T_param, NV)
    w_for_llgrad = Vector{T_param}() # Will be populated if NV > 0

    current_idx = 0
    if NF > 0
        f_params = param[1:NF]
        current_idx = NF
    end

    inds_B_to_estimate = [v ∈ [:normal, :lognormal, :truncnormal, :S_B, :triangular] for v in randdist_random]
    # inds_B_to_estimate = fill(true, NV)
    # num_b_estimated = NV

    num_b_estimated = sum(inds_B_to_estimate)

    if num_b_estimated > 0
        b_estimated_part = param[current_idx+1:current_idx+num_b_estimated]
        b_full[inds_B_to_estimate] .= b_estimated_part
    end
    current_idx += num_b_estimated

    w_for_llgrad = param[current_idx+1:current_idx+NV]

    # --- Start of inlined llgrad2_calc logic ---

    # p_person = zeros(T_param, n_id) # Corresponds to p_person in llgrad2_calc before averaging
    fill!(p_person, 0.0) # Initialize p_person to zero

    # Intermediate arrays needed for calculations
    # V_diff = zeros(T_param, NALTMAX - 1, n_chidMAX, n_id, NDRAWS) # Will also store exp(V_diff) later
    fill!(V_diff, 0.0) # Initialize V_diff to zero

    pp_person_draw_segment = zeros(T_param, n_id, NDRAWS) # Stores P_nr for the current take


    if G !== nothing
        local g_person::Matrix{T_param}
        local gg_alt_cs_draw::Array{T_param,4}
        local gr_segment_full::Array{T_param,3} = Array{T_param,3}(undef, NF + NV + NV, n_id, NDRAWS)

        g_person = zeros(T_param, NF + NV + NV, n_id) # Corresponds to g_person before final divisions

        # Initialize gradient specific intermediate arrays
        gg_alt_cs_draw = zeros(T_param, NALTMAX - 1, n_chidMAX, n_id, NDRAWS)
    end

    # --- llgrad2_calc Step 1: Transform standardized draws to coefficient draws ---
    trans_coeffs!(c_coeffs, b_full, w_for_llgrad, DR, NV, randdist_random) # Modifies c_coeffs

    # --- llgrad2_calc Step 2: Calculate Utilities (V_ijt - V_i*t) ---

    if NF > 0
        @inbounds for idx_naltmax in 1:(NALTMAX-1) # NALTMAX-1 dimension
            for idx_ncsmax in 1:n_chidMAX             # n_chidMAX dimension
                for idx_np in 1:n_id                         # n_id dimension
                    @simd for idx_nf in 1:NF # Sum over the NF dimension
                        V_diff[idx_naltmax, idx_ncsmax, idx_np, :] .+= XF[idx_naltmax, idx_ncsmax, idx_nf, idx_np] * f_params[idx_nf]
                    end
                end
            end
        end
    end

    @inbounds Threads.@threads for idx_ndraws in 1:NDRAWS                 # Outermost loop: NDRAWS
        for idx_naltmax in 1:(NALTMAX-1) # Innermost loop for V_diff elements: NALTMAX-1
            for idx_ncsmax in 1:n_chidMAX         # Next loop: n_chidMAX
                for idx_np in 1:n_id                     # Next loop: n_id
                    @simd for idx_nv in 1:NV
                        V_diff[idx_naltmax, idx_ncsmax, idx_np, idx_ndraws] += X[idx_naltmax, idx_ncsmax, idx_nv, idx_np] * c_coeffs[idx_nv, idx_np, idx_ndraws]
                    end
                end
            end
        end
    end

    # --- llgrad2_calc Step 3: Calculate Choice Probabilities (P_itn for each draw r) ---
    # V_diff .= exp.(V_diff)
    # V_diff[isinf.(V_diff)] .= T_param(1e200)

    @inbounds Threads.@threads for i in eachindex(V_diff)
        V_diff[i] = min(exp(V_diff[i]), T_param(1e200))
    end
    V_diff .*= S # S is a mask for the choice set

    # fill!(pp_choice_sit_draw, 0.0) # Fill with zeros
    @inbounds Threads.@threads for idx_ndraws in 1:NDRAWS  # Iterate over the last dimension
        for idx_id in 1:n_id   # Iterate over the third dimension
            for cs_idx in 1:n_chidMAX # Iterate over the second dimension
                current_sum = zero(T_param) # Accumulator for the sum

                # Innermost loop: sum over the first dimension of V_diff
                @simd for alt_idx in 1:(NALTMAX-1)
                    current_sum += V_diff[alt_idx, cs_idx, idx_id, idx_ndraws]
                end

                # Perform the final calculation and store directly
                pp_choice_sit_draw[1, cs_idx, idx_id, idx_ndraws] = T_param(1.0) / (T_param(1.0) + current_sum)
            end

            # --- llgrad2_calc Step 5 (Part 1): Aggregate Probabilities (P_nr) ---
            # temp_pp_reshaped = reshape(pp_choice_sit_draw, n_chidMAX, n_id, NDRAWS)

            current_product = T_param(1.0) # Initialize product for this (idx_np, idx_ndraws) combination

            # Innermost loop: calculate product along the n_chidMAX dimension of pp_choice_sit_draw
            # This corresponds to iterating through the elements that are multiplied together
            @simd for idx_ncsmax in 1:n_chidMAX
                current_product *= pp_choice_sit_draw[1, idx_ncsmax, idx_id, idx_ndraws]
            end

            # Store the computed product directly into the target array
            pp_person_draw_segment[idx_id, idx_ndraws] = current_product
        end
    end

    # p_person .+= vec(sum(pp_person_draw_segment, dims=2))
    @inbounds Threads.@threads for idx_np in 1:n_id
        # Accumulator for the sum over draws for the current person (idx_np)
        sum_for_current_person = zero(T_param) # Initialize sum to 0.0 of type T_param

        # Innermost loop: sum along the NDRAWS dimension for the current idx_np
        # This iterates across the columns of pp_person_draw_segment for row idx_np
        @simd for idx_ndraws in 1:NDRAWS # Candidate for @simd if NDRAWS is large enough
            sum_for_current_person += pp_person_draw_segment[idx_np, idx_ndraws]
        end

        # Add the calculated sum directly to the existing value in p_person
        p_person[idx_np] += sum_for_current_person
    end

    # --- Conditional Gradient Calculations (llgrad2_calc Steps 4 & Part of 5) ---
    if G !== nothing
        # --- llgrad2_calc Step 4: Calculate Gradient Components (per draw) ---
        gg_alt_cs_draw .= V_diff .* pp_choice_sit_draw
        # The number of elements in gg_alt_cs_draw is (NALTMAX-1) * n_chidMAX * n_id * NDRAWS.

        summed_base_grad_rand = Array{T_param}(undef, NV, n_id, NDRAWS)

        @inbounds Threads.@threads for idx_ndraws in 1:NDRAWS         # Dimension corresponding to NDRAWS
            for idx_np in 1:n_id             # Dimension corresponding to n_id

                if NF > 0
                    for idx_nf in 1:NF         # Dimension corresponding to NF (first dim of the slice)

                        current_sum_val = zero(T_param) # Initialize accumulator for the sum

                        # Sum over the common dimensions (NALTMAX-1 and n_chidMAX)
                        for idx_ncsmax in 1:n_chidMAX
                            @simd for idx_naltmax in 1:(NALTMAX-1)
                                current_sum_val += gg_alt_cs_draw[idx_naltmax, idx_ncsmax, idx_np, idx_ndraws] * XF[idx_naltmax, idx_ncsmax, idx_nf, idx_np]
                            end
                        end

                        # Assign the negated sum to the corresponding element in gr_segment_full
                        # Target slice indexing: gr_segment_full[idx_nf, idx_np, idx_ndraws]
                        gr_segment_full[idx_nf, idx_np, idx_ndraws] = -current_sum_val
                    end
                end

                for idx_nv in 1:NV
                    # Accumulator for the sum over (NALTMAX-1) and n_chidMAX dimensions
                    current_sum_val = zero(T_param)

                    # Sum over the first two effective dimensions (NALTMAX-1 and n_chidMAX)
                    for idx_ncsmax in 1:n_chidMAX
                        @simd for idx_naltmax in 1:(NALTMAX-1)
                            current_sum_val += gg_alt_cs_draw[idx_naltmax, idx_ncsmax, idx_np, idx_ndraws] * X[idx_naltmax, idx_ncsmax, idx_nv, idx_np]
                        end
                    end
                    # Store the negated sum in the output array
                    # Indices for summed_base_grad_rand: (nv, np, ndraws)
                    summed_base_grad_rand[idx_nv, idx_np, idx_ndraws] = -current_sum_val
                end
            end
        end

        derivatives_calc!(deriv_C_b, deriv_C_w, c_coeffs, randdist_random, DR, NV) # Modifies deriv_C_b, deriv_C_w

        # The number of elements in gg_alt_cs_draw must be (NALTMAX-1) * n_chidMAX * n_id * NDRAWS.
        # The number of elements in X must be (NALTMAX-1) * n_chidMAX * NV * n_id.

        # --- llgrad2_calc Step 5 (Part 2): Aggregate Gradients ---

        gr_segment_full[NF+1:NF+NV, :, :] .= summed_base_grad_rand .* deriv_C_b # gr_b
        gr_segment_full[NF+NV+1:NF+NV+NV, :, :] .= summed_base_grad_rand .* deriv_C_w # gr_w

        gr_segment_full .*= reshape(pp_person_draw_segment, 1, n_id, NDRAWS)
        # g_person .+= sum(gr_segment_full, dims=3)[:, :, 1]
        @inbounds for idx_np in 1:n_id
            for idx_nparam in 1:(NF+NV+NV)
                sum_over_draws = zero(T_param) # Accumulator for the sum

                # Innermost loop: sum along the NDRAWS dimension (third dim of gr_segment_full)
                @simd for idx_ndraws in 1:NDRAWS # Candidate for @simd
                    sum_over_draws += gr_segment_full[idx_nparam, idx_np, idx_ndraws]
                end

                # Add the computed sum to the corresponding element in g_person
                g_person[idx_nparam, idx_np] += sum_over_draws
            end
        end
    end # End of G !== nothing for gradient calculations

    # --- llgrad2_calc Step 6: Final Averaging for person_probs ---
    p_person ./= NDRAWS

    # --- Calculate total negative log-likelihood (Original fg! Step 3) ---
    log_p_person = log.(p_person)
    local neg_log_likelihood::T_param
    if WANTWGT == 0 || n_id == 0
        neg_log_likelihood = -sum(log_p_person)
    else
        neg_log_likelihood = -dot(WGT, log_p_person)
    end

    # --- Gradient Finalization (Original fg! Step 4, using inlined components) ---
    if G !== nothing
        # Calculate final person_grads_full (formerly g_person from llgrad2_calc)
        g_person ./= NDRAWS # Average over draws

        # Reshape person_probs (which is P_n) for broadcasting
        p_n_reshaped_denom = reshape(p_person, 1, n_id)

        g_person ./= p_n_reshaped_denom # Element-wise division

        # Sum gradients over people (n_id) - from original fg!
        local neg_summed_grads::Vector{T_param} = zeros(T_param, NF + NV + NV)
        if size(g_person, 2) == 0 # n_id == 0
        # neg_summed_grads remains zeros
        elseif WANTWGT == 0
            neg_summed_grads .= -vec(sum(g_person, dims=2))
        else # WANTWGT is true and n_id > 0
            # WGT is n_id x 1. Reshape for broadcasting: 1 x n_id
            neg_summed_grads .= -vec(sum(g_person .* reshape(WGT, 1, n_id), dims=2))
        end

        # Filter gradient to match the structure of `param` - from original fg!
        if (NF + NV + NV) == 0 # No parameters at all
        # G should be empty if param is empty. Optim.jl usually handles G's size.
        else
            grad_mask = Vector{Bool}(undef, NF + NV + NV)
            mask_idx = 1
            if NF > 0
                fill!(view(grad_mask, mask_idx:mask_idx+NF-1), true)
                mask_idx += NF
            end
            if NV > 0
                copyto!(view(grad_mask, mask_idx:mask_idx+NV-1), inds_B_to_estimate)
                mask_idx += NV
                fill!(view(grad_mask, mask_idx:mask_idx+NV-1), true)
            end
            G .= neg_summed_grads[grad_mask]
        end
    end # End G !== nothing for gradient finalization

    return neg_log_likelihood
end

# Defines the derivatives_calc function to compute the derivatives of
# random coefficients with respect to their mean (b) and standard deviation (w) parameters.
# Translated from Matlab code by Kenneth Train.

"""
    derivatives_calc(b_coeffs::Vector{Float64},
                     w_coeffs::Vector{Float64},
                     standard_draws::Array{Float64,3})

Calculates the partial derivatives of the transformed random coefficient draws (C)
with respect to their underlying mean parameters (b) and standard deviation
parameters (w).

Args:
    b_coeffs (Vector{Float64}): Mean parameters for each random coefficient (NV elements).
    w_coeffs (Vector{Float64}): Standard deviation parameters for each random coefficient (NV elements).
    standard_draws (Array{Float64,3}): Standardized draws (e.g., N(0,1) or triangular).
                                       Dimensions: NV x n_id x NMEM_current.
    NV (Int): Number of random coefficients.

Returns:
    Tuple{Array{Float64,3}, Array{Float64,3}}:
        - deriv_C_b (Array{Float64,3}): Derivatives dC/db, same dimensions as `DR`.
        - deriv_C_w (Array{Float64,3}): Derivatives dC/dw, same dimensions as `DR`.
"""
function derivatives_calc!(deriv_C_b::AbstractArray{T,3},
    deriv_C_w::AbstractArray{T,3},
    c_coeffs::AbstractArray{T,3},
    randdist_random::Vector{Symbol},
    DR::Array{Float64,3},
    NV::Int) where T<:Real

    # Initialize deriv_C_b. For C = beta = b + w*draw, dC/db = d(beta)/db = 1.
    # This is the base case for Normal, Normal0Mean, and Triangular distributions.
    fill!(deriv_C_b, 1.0) # Initialize all elements to 1.0
    # deriv_C_b = ones(T, size(DR)...)
    # deriv_C_w = ones(T, size(DR)...)

    # Only need to calculate beta_draws if there are non-linear transformations.
    # Check if any distributions are type 2, 3, or 4.
    needs_beta_calc = any(randdist_random .∈ Ref([:lognormal, :truncnormal, :S_B]))

    if needs_beta_calc
        # Calculate beta_draws = b + w * DR, as these are needed for some derivatives.
        # beta_draws = b_coeffs .+ w_coeffs .* DR

        @inbounds for i = 1:NV
            dist_type = randdist_random[i]

            # Slice of beta_draws for the i-th variable: n_id x NMEM_current
            c_coeffs_slice_i = view(c_coeffs, i, :, :)

            if dist_type == :lognormal # Lognormal: C = exp(beta_i). dC/d(beta_i) = exp(beta_i).
                # So, dC/db = exp(beta_i)
                deriv_C_b[i, :, :] = c_coeffs_slice_i # Already exp() in trans_coeffs!

            elseif dist_type == :truncnormal # Truncated Normal: C = max(0, beta_i). dC/d(beta_i) = 1 if beta_i > 0, else 0.
                # So, dC/db = (beta_i > 0)
                deriv_C_b[i, :, :] = (c_coeffs_slice_i .> 0.0)

            elseif dist_type == :S_B # S_B: C = exp(beta_i) / (1 + exp(beta_i)).
                # dC/d(beta_i) = C * (1 - C).
                # So, dC/db = C * (1 - C)
                c_val = c_coeffs_slice_i # Already exp() / (1 + exp() in trans_coeffs!
                deriv_C_b[i, :, :] = c_val .* (1.0 .- c_val)

                # For dist_type 1, 5, 6, dC/db remains 1, already initialized.
            end
        end
    end # End if needs_beta_calc

    # Calculate deriv_C_w: dC/dw = (dC/d(beta)) * (d(beta)/dw)
    # Since d(beta)/dw = standard_draws, and dC/d(beta) is what we stored in deriv_C_b,
    # then dC/dw = deriv_C_b .* standard_draws.
    deriv_C_w .= deriv_C_b .* DR

    return deriv_C_b, deriv_C_w
end


# Defines the trans_coeffs function to transform standardized random draws
# into draws of actual random coefficients based on specified distributions.
# Translated from Matlab code by Kenneth Train.


"""
    trans_coeffs!(c_coeffs::AbstractArray{T,3},
                  b_coeffs::AbstractVector{T},
                  w_coeffs::AbstractVector{T},
                  standard_draws::AbstractArray{T,3},
                  NV::Int,
                  randdist_random) where T<:Real

Transforms standardized draws (e.g., N(0,1) or other base distributions)
into draws of random coefficients according to their specified distributions,
mutating `c_coeffs` in-place.

Args:
    c_coeffs (AbstractArray{T,3}): Output array for transformed coefficients.
                                      This array will be mutated.
                                      Dimensions: NV x n_id x NMEM_current.
    b_coeffs (AbstractVector{T}): Mean parameters for each random coefficient (NV elements).
                                  For type 5 (zero mean normal), b_coeffs[i] should be 0.
    w_coeffs (AbstractVector{T}): Standard deviation parameters for each random coefficient (NV elements).
    standard_draws (AbstractArray{T,3}): Standardized draws.
                                          Dimensions: NV x n_id x NMEM_current.
                                          (NV: number of random variables,
                                           n_id: number of persons,
                                           NMEM_current: number of draws in this segment).
    NV (Int): Number of random coefficients.
    randdist_random

Returns:
    Nothing. The `c_coeffs` array is modified in-place.
"""
function trans_coeffs!(c_coeffs::AbstractArray{T,3},
    b_coeffs::AbstractVector{T},
    w_coeffs::AbstractVector{T},
    DR::AbstractArray{Float64,3},
    NV::Int,
    randdist_random::Vector{Symbol} # Vector of distribution types for each variable
) where {T<:Real}

    # Step 1: Basic linear transformation (beta = b + w * draw)
    # Broadcasting: (NV,1) .+ (NV,1) .* (NV,n_id,NMEM_current) results in (NV,n_id,NMEM_current).
    # This operation handles NV=0 correctly, where all arrays involved will have 0 as their first dimension.
    c_coeffs .= b_coeffs .+ w_coeffs .* DR

    # Step 2: Apply specific distribution transformations row-wise (for each variable i)
    # This loop only executes if NV > 0.
    @inbounds for i = 1:NV
        dist_type = randdist_random[i] # Get distribution type for the i-th random variable

        # Get a view of the current slice from c_coeffs.
        # current_slice now holds the result of b_i + w_i * draw_i (i.e., beta_i for this variable)
        # Operations on current_slice will modify c_coeffs in-place.
        current_slice = view(c_coeffs, i, :, :)

        if dist_type == :normal || dist_type == :normal0mn || dist_type == :triangular
            # Type 1: Normal N(b, w^2)
            # Type 5: Normal N(0, w^2) (b_coeffs[i] is 0)
            # Type 6: Triangular (DR for triangular are assumed pre-scaled)
            # For these types, c_coeffs[i,:,:] (i.e., current_slice) already holds the final coefficient draw.
            # No action needed.
        elseif dist_type == :lognormal # Lognormal: coefficient is exp(beta_i)
            current_slice .= exp.(current_slice)
        elseif dist_type == :truncnormal # Truncated Normal: coefficient is max(0, beta_i)
            current_slice .= max.(T(0.0), current_slice)
        elseif dist_type == :S_B # S_B (Johnson SB) distribution: exp(beta_i) / (1 + exp(beta_i))
            # current_slice currently holds beta_i
            current_slice .= exp.(current_slice) # current_slice now stores exp(beta_i)
            # Now current_slice is exp(beta_i), let's call it val = exp(beta_i)
            # We want val / (1 + val)
            current_slice .= current_slice ./ (T(1.0) .+ current_slice)
        else
            error("Unknown distribution type $dist_type encountered in trans_coeffs! for variable $i.")
        end
    end

    return nothing # Mutating functions typically return nothing
end



# Contains makedraws_core() function to generate standardized random draws.
# Translated from Matlab code by Kenneth Train.

# Helper function to transform uniform draws to triangular draws on [-1, 1]
function _generate_triangular_draws(uniform_draws::AbstractArray{Float64})
    # Ensure draws are in [0,1] for the formula
    clipped_draws = clamp.(uniform_draws, 0.0, 1.0)

    term1 = (sqrt.(2 .* clipped_draws) .- 1) .* (clipped_draws .<= 0.5)
    term2 = (1 .- sqrt.(2 .* (1 .- clipped_draws))) .* (clipped_draws .> 0.5)
    return term1 .+ term2
end

# Helper function to generate Halton-like sequence as in Matlab code
# Generates `num_points_to_keep` points after discarding `discard_first_n`.
function _generate_matlab_halton_sequence(base_prime::Int, num_points_needed_total::Int)
    # num_points_needed_total includes points to be discarded + points to keep.

    # Initialize sequence with 0
    # Using a Vector of Float64, grows dynamically. For very large sequences, pre-allocation or a more direct calculation would be better.
    current_draws = [0.0]

    exponent = 1
    while length(current_draws) < num_points_needed_total
        draws_old_exponent = copy(current_draws)
        # For each point generated with previous exponent, add (m / base_prime^exponent)
        # This loop structure adds (base_prime - 1) * length(draws_old_exponent) new points
        for m = 1:(base_prime-1)
            term_to_add = m / (base_prime^exponent)
            append!(current_draws, draws_old_exponent .+ term_to_add)
            if length(current_draws) >= num_points_needed_total
                break # Stop if enough points are generated
            end
        end
        exponent += 1
    end

    # Return the required number of points from the generated sequence
    return current_draws[1:num_points_needed_total]
end


function makedraws_core(NDRAWS::Int, n_id::Int, NV::Int, DRAWTYPE::Symbol, randdist_random::Vector{Symbol})
    # dr_array will store draws
    # Dimensions: NDRAWS x n_id x NV
    dr_array::Array{Float64,3} = zeros(Float64, NDRAWS, n_id, NV)

    # --- Generate Draws based on DRAWTYPE ---

    # DRAWTYPE == 1: Pseudo-random draws
    if DRAWTYPE == :pseudo
        for j = 1:NV # For each random variable
            is_triangular = (randdist_random[j] == :triangular)
            temp_draws_uniform = rand(Float64, NDRAWS, n_id) # Uniform draws for transformation
            if is_triangular
                dr_array[:, :, j] = _generate_triangular_draws(temp_draws_uniform)
            else # Normal draws
                # randn generates from N(0,1) directly
                dr_array[:, :, j] = randn(Float64, NDRAWS, n_id)
            end
        end
    end # End DRAWTYPE == 1

    # DRAWTYPE == 2 (Standard Halton) or DRAWTYPE == 3 (Shifted/Shuffled Halton)
    if DRAWTYPE == :halton || DRAWTYPE == :shiftedhalton
        # Find NV prime numbers
        prime_bases = Int[]
        limit = 200 # Initial limit for prime search, adjust if NV is very large
        while length(prime_bases) < NV
            prime_bases = Primes.primes(limit)
            if length(prime_bases) < NV
                limit *= 2 # Increase limit if not enough primes found
            end
        end
        prime_bases = prime_bases[1:NV]

        points_to_discard = 10 # As per Matlab code
        num_draws_per_var = n_id * NDRAWS
        total_halton_points_needed = num_draws_per_var + points_to_discard

        for j = 1:NV # For each random variable
            base_p = prime_bases[j]

            # Generate Halton sequence (uniform on [0,1))
            halton_seq_raw = _generate_matlab_halton_sequence(base_p, total_halton_points_needed)
            uniform_draws_for_var = halton_seq_raw[(points_to_discard+1):end] # Keep n_id*NDRAWS points

            if DRAWTYPE == :shiftedhalton # Shifted and Shuffled Halton
                # Shift: one shift for the entire sequence for this variable
                shift_val = rand()
                uniform_draws_for_var .+= shift_val
                uniform_draws_for_var .-= floor.(uniform_draws_for_var) #Equivalent to mod(x,1)

                # Reshape for shuffling per person
                # Data is currently a vector of length n_id*NDRAWS.
                # Reshape to NDRAWS x n_id (column-major by default in Julia's reshape)
                # This means column `p` has `NDRAWS` draws for person `p`.
                draws_reshaped_for_shuffle = reshape(copy(uniform_draws_for_var), NDRAWS, n_id)
                for p = 1:n_id # Shuffle for each person separately
                    shuffle!(@view draws_reshaped_for_shuffle[:, p])
                end
                # Reshape back to a single vector (column-major concatenation)
                uniform_draws_for_var = vec(draws_reshaped_for_shuffle)
            end

            # Transform draws (uniform to normal or triangular)
            is_triangular = (randdist_random[j] == :triangular)
            local final_transformed_draws_for_var::Vector{Float64}
            if is_triangular
                final_transformed_draws_for_var = _generate_triangular_draws(uniform_draws_for_var)
            else # Normal
                final_transformed_draws_for_var = quantile.(Normal(0, 1), uniform_draws_for_var)
            end

            # Store or write
            # Reshape final_transformed_draws_for_var (length n_id*NDRAWS) to NDRAWS x n_id
            # and assign to dr_array[:, :, j]
            dr_array[:, :, j] = reshape(final_transformed_draws_for_var, NDRAWS, n_id)
        end
    end # End DRAWTYPE == 2 or 3

    # DRAWTYPE == 4: Modified Latin Hypercube Sampling (MLHS)
    if DRAWTYPE == :MLHS
        base_lhs_points = (0:(NDRAWS-1)) ./ NDRAWS # Vector of NDRAWS base points

        for j = 1:NV # For each random variable
            is_triangular = (randdist_random[j] == :triangular)
            for p = 1:n_id # For each person
                # Create draws for this person, for this variable
                # Shift: Different shift for each person-variable combination
                shifted_lhs_points = base_lhs_points .+ (rand() / NDRAWS)

                # Shuffle
                shuffled_draws_uniform = shuffle(shifted_lhs_points) # Length NDRAWS

                # Transform (uniform to normal or triangular)
                local final_mlhs_draws_person_var::Vector{Float64}
                if is_triangular
                    final_mlhs_draws_person_var = _generate_triangular_draws(shuffled_draws_uniform)
                else # Normal
                    final_mlhs_draws_person_var = quantile.(Normal(0, 1), shuffled_draws_uniform)
                end

                # Store or write
                dr_array[:, p, j] = final_mlhs_draws_person_var
            end
        end
    end # End DRAWTYPE == 4

    # The returned dr_array is NDRAWS x n_id x NV.
    # doit.jl will permute it to NV x n_id x NDRAWS.
    return dr_array
end

# This script defines the function check_inputs() to validate input data and specifications.
# It is intended to be included by doit.jl.
# Translated from Matlab code by Kenneth Train.

function check_inputs(n_id, n_chid, randdist, NV, NAMES, B, W, IDF, NF, NAMESF, F,
    DRAWTYPE, NDRAWS, WANTWGT, IDWGT, XMAT)::Bool
    # This function accesses global variables defined in mxlmsl.jl.
    # These include: n_id, n_chid, IDV, NV, NAMES, B, W, IDF, NF, NAMESF, F,
    # DRAWTYPE, NDRAWS, WANTWGT, IDWGT, XMAT, DR (if DRAWTYPE==5).

    # Helper function for error messages (to reduce repetition)
    # In a larger application, this might also write to the output_filename
    function print_error(msg::String)
        println(msg) # Should also go to output_filename (e.g., myrun.out)
        println("Program terminated.") # To output_filename
    end

    # Check for positive integers
    if !isinteger(n_id) || n_id < 1
        print_error("n_id must be a positive integer, but it is set to $n_id.")
        return false
    end

    if !isinteger(n_chid) || n_chid < 1
        print_error("n_chid must be a positive integer, but it is set to $n_chid.")
        return false
    end

    if !isinteger(NDRAWS) || NDRAWS < 1
        print_error("NDRAWS must be a positive integer, but it is set to $NDRAWS.")
        return false
    end

    if !isinteger(DRAWTYPE) || DRAWTYPE < 1
        print_error("DRAWTYPE must be a positive integer, but it is set to $DRAWTYPE.")
        return false
    end

    if NV > 0 && any(x -> !isinteger(x) || x < 1, IDV)
        print_error("IDV must contain positive integers only, but it contains other values.")
        return false
    end

    # IDF is a Vector in mxlmsl.jl. If it were a Matrix, the original check would be sum(sum(...))
    # For a Vector, any(...) is sufficient.
    if NF > 0 && any(x -> !isinteger(x) || x < 1, IDF)
        print_error("IDF must contain positive integers only, but it contains other values.")
        return false
    end

    # Checking XMAT

    if any(XMAT[:, 1] .> n_id)
        print_error("The first column of XMAT has a value greater than n_id = $n_id.")
        return false
    end

    if any(XMAT[:, 1] .< 1)
        print_error("The first column of XMAT has a value less than 1.")
        return false
    end

    if any(vec_chid .> n_chid)
        print_error("The second column of XMAT has a value greater than n_chid = $n_chid.")
        return false
    end

    if any(vec_chid .< 1)
        print_error("The second column of XMAT has a value less than 1.")
        return false
    end

    if any((XMAT[:, 3] .!= 0) .& (XMAT[:, 3] .!= 1))
        print_error("The third column of XMAT has a value other than 1 or 0.")
        return false
    end

    for s = 1:n_chid
        rows_for_cs = (vec_chid .== s)
        if sum(rows_for_cs) == 0 # No alternatives for this choice situation
            # This might be an issue depending on data structure expectations.
            # Original code doesn't explicitly check for this before summing chosen.
            # If sum(XMAT[rows_for_cs,3]) is on an empty set, it might behave differently.
            # sum on empty vector is 0 in Julia.
            print_error("No data found for choice situation $s.")
            return false
        end
        sum_chosen = sum(XMAT[rows_for_cs, 3])
        if sum_chosen > 1
            print_error("The third column of XMAT indicates more than one chosen alternative for choice situation $s.")
            return false
        end
        if sum_chosen < 1
            print_error("The third column of XMAT indicates that no alternative was chosen for choice situation $s.")
            return false
        end
    end

    if any(isnan.(XMAT))
        print_error("XMAT contains missing data (NaN).")
        return false
    end

    if any(isinf.(XMAT))
        print_error("XMAT contains an infinite value.")
        return false
    end

    # Checks for IDV (Random Coefficients)
    if NV > 0
        if size(IDV, 2) != 2
            print_error("IDV must have 2 columns, but it has $(size(IDV,2)).")
            return false
        end

        if any(IDV[:, 1] .> size(XMAT, 2))
            print_error("IDV identifies a variable column index that is outside XMAT (max is $(size(XMAT,2))). Problematic indices: $(IDV[IDV[:,1] .> size(XMAT,2), 1]).")
            return false
        end

        if any(IDV[:, 1] .<= 3)
            print_error("Each element in the first column of IDV must exceed 3 (first three XMAT columns are reserved). Problematic indices: $(IDV[IDV[:,1] .<= 3, 1]).")
            return false
        end

        if any((IDV[:, 2] .< 1) .| (IDV[:, 2] .> 6)) # Distribution codes 1-6
            error("The second column of IDV must be integers 1-6. Problematic values: $(IDV[(IDV[:,2] .< 1 .| IDV[:,2] .> 6), 2]).")
        end

        # NAMES, B, W are Vectors in mxlmsl.jl
        # Check if they are column-vector like (true for Vector, or Matrix with 1 column)
        # And check lengths.
        if !(isa(NAMES, Vector)) && !(isa(NAMES, Matrix) && size(NAMES, 2) == 1)
            print_error("NAMES must be a column vector (or Julia Vector). Be sure to separate names by semicolons in Matlab original if it was a cell string array.")
            return false
        end
        if length(NAMES) != NV
            print_error("IDV and NAMES must have the same length (NV=$NV). NAMES has length $(length(NAMES)).")
            return false
        end

        if !(isa(B, Vector)) && !(isa(B, Matrix) && size(B, 2) == 1)
            print_error("B must be a column vector (or Julia Vector).")
            return false
        end
        if length(B) != NV
            print_error("B must have the same length as IDV (NV=$NV). B has length $(length(B)).")
            return false
        end

        if !(isa(W, Vector)) && !(isa(W, Matrix) && size(W, 2) == 1)
            print_error("W must be a column vector (or Julia Vector).")
            return false
        end
        if length(W) != NV
            print_error("W must have the same length as IDV (NV=$NV). W has length $(length(W)).")
            return false
        end
    end # End NV > 0 checks

    # Checks for IDF (Fixed Coefficients)
    if NF > 0
        # IDF is a Vector in mxlmsl.jl
        if !(isa(IDF, Vector)) && !(isa(IDF, Matrix) && size(IDF, 2) == 1)
            print_error("IDF must be a column vector (or Julia Vector).")
            return false
        end
        # Length of IDF is NF by definition in mxlmsl.jl

        if any(IDF .> size(XMAT, 2))
            print_error("IDF identifies a variable column index that is outside XMAT (max is $(size(XMAT,2))). Problematic indices: $(IDF[IDF .> size(XMAT,2)]).")
            return false
        end

        if any(IDF .<= 3)
            print_error("Each element of IDF must exceed 3 (first three XMAT columns are reserved). Problematic indices: $(IDF[IDF .<= 3]).")
            return false
        end

        # NAMESF, F are Vectors
        if !(isa(NAMESF, Vector)) && !(isa(NAMESF, Matrix) && size(NAMESF, 2) == 1)
            print_error("NAMESF must be a column vector (or Julia Vector).")
            return false
        end
        if length(NAMESF) != NF
            print_error("IDF and NAMESF must have the same length (NF=$NF). NAMESF has length $(length(NAMESF)).")
            return false
        end

        if !(isa(F, Vector)) && !(isa(F, Matrix) && size(F, 2) == 1)
            print_error("F must be a column vector (or Julia Vector).")
            return false
        end
        if length(F) != NF
            print_error("F must have the same length as IDF (NF=$NF). F has length $(length(F)).")
            return false
        end
    end # End NF > 0 checks

    # Draw type checks
    if DRAWTYPE < 1 || DRAWTYPE > 5 # Original Matlab was 1-5; current mxlmsl.m allows 1-6 for IDV dists, but DRAWTYPE is 1-5.
        print_error("DRAWTYPE must be an integer 1-5. It is set to $DRAWTYPE.")
        return false
    end

    # Weight checks
    if WANTWGT != 0 && WANTWGT != 1
        print_error("WANTWGT must be 0 or 1, but it is set to $WANTWGT.")
        return false
    end

    if WANTWGT == 1
        if IDWGT === nothing
            print_error("WANTWGT=1, but IDWGT is not set (it's 'nothing'). It must be a column index.")
            return false
        end
        # IDWGT is an Int if set. Check if it's a scalar.
        # In Julia, if IDWGT is Int, it's a scalar. No need for size(IDWGT,1) != 1.
        # The Matlab check size(IDWGT,1)~=1 was for if IDWGT was an array e.g. from user error.
        # Here, IDWGT is expected to be an Int.

        if IDWGT > size(XMAT, 2)
            print_error("IDWGT ($IDWGT) for weights identifies a variable outside XMAT (max col is $(size(XMAT,2))).")
            return false
        end

        if IDWGT < 1 # isinteger check already done implicitly if it's an Int type
            print_error("IDWGT ($IDWGT) must be a positive integer identifying a variable in XMAT.")
            return false
        end

        # Check if weight variable is constant for each person
        cp_weights = XMAT[:, 1] # Person ID column
        for r = 1:n_id
            person_rows = (cp_weights .== r)
            if sum(person_rows) > 0 # If person has data
                weights_for_person = XMAT[person_rows, IDWGT]
                if any(weights_for_person .!= mean(weights_for_person))
                    print_error("Weight variable (column $IDWGT) must be the same for all rows for each person. Person $r has varying weights.")
                    return false
                end
            end
        end
    end # End WANTWGT == 1 checks

    return true # All checks passed
end

