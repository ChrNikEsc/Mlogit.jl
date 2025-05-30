mutable struct LCLmodel <: RegressionModel
    # coef::Vector{Float64}
    coef_memb::Matrix{Float64}
    coef_mnl::Matrix{Float64}
    # coefnames::Vector{String}
    coefnames_memb::Vector{String}
    coefnames_mnl::Vector{String}
    converged::Union{Bool,Nothing}
    dof::Int64
    formula::FormulaTerm
    formula_origin::FormulaTerm
    formula_schema::FormulaTerm
    hessian::Union{Matrix{Float64},Nothing}
    iter::Union{Int64,Nothing}
    loglikelihood::Float64
    method::Symbol
    # model_membership::FMLmodel
    # models_mnl::Vector{MNLmodel}
    nchids::Int64
    nclasses::Int64
    nids::Int64
    nullloglikelihood::Float64
    optim::Union{Optim.OptimizationResults,Nothing}
    responsename::String
    score::Union{Vector{Float64},Nothing}
    shares::Vector{Float64}
    start::Vector{Float64}
    startloglikelihood::Float64
    time::Float64
    vcov::Union{Matrix{Float64},Nothing}
end

function LCLmodel(;
    # coef::Vector{Float64},
    coef_memb::Matrix{Float64},
    coef_mnl::Matrix{Float64},
    # coefnames::Vector{String},
    coefnames_memb::Vector{String},
    coefnames_mnl::Vector{String},
    converged::Union{Bool,Nothing},
    dof::Int64,
    formula::FormulaTerm,
    formula_origin::FormulaTerm,
    formula_schema::FormulaTerm,
    hessian::Union{Matrix{Float64},Nothing},
    iter::Union{Int64,Nothing},
    loglikelihood::Float64,
    method::Symbol,
    # model_membership::FMLmodel,
    # models_mnl::Vector{MNLmodel},
    nchids::Int64,
    nclasses::Int64,
    nids::Int64,
    nullloglikelihood::Float64,
    optim::Union{Optim.OptimizationResults,Nothing},
    responsename::String,
    score::Union{Vector{Float64},Nothing},
    shares::Vector{Float64},
    start::Vector{Float64},
    startloglikelihood::Float64,
    time::Float64,
    vcov::Union{Matrix{Float64},Nothing})
    return LCLmodel(coef_memb, coef_mnl, coefnames_memb, coefnames_mnl, converged, dof, formula, formula_origin, formula_schema, hessian, iter, loglikelihood, method, nchids, nclasses, nids, nullloglikelihood, optim, responsename, score, shares, start, startloglikelihood, time, vcov)
end


StatsBase.adjr2(model::LCLmodel) = 1 - (loglikelihood(model) - dof(model)) / nullloglikelihood(model)
StatsBase.aic(model::LCLmodel) = -2 * loglikelihood(model) + dof(model) * 2
StatsBase.aicc(model::LCLmodel) = -2 * loglikelihood(model) + 2 * dof(model) + 2 * dof(model) * (dof(model) - 1) / (nobs(model) - dof(model) - 1)
StatsBase.bic(model::LCLmodel) = -2 * loglikelihood(model) + dof(model) * log(nobs(model))
caic(model::LCLmodel) = -2 * loglikelihood(model) + dof(model) * (log(nobs(model)) + 1)
StatsBase.coef(model::LCLmodel) = [vec(model.coef_mnl); vec(model.coef_memb)]
StatsBase.coefnames(model::LCLmodel) = vcat(
    vec([model.coefnames_mnl[i] * " _" * string.(c) for i in 1:Base.length(model.coefnames_mnl), c in 1:model.nclasses]),
    vec([model.coefnames_memb[i] * " _" * string.(c) for i in 1:Base.length(model.coefnames_memb), c in 1:(model.nclasses-1)])
)
function confint(model::LCLmodel; level::Real=0.95)
    hcat(coef(model), coef(model)) + stderror(model) * quantile(Normal(), (1.0 - level) / 2.0) * [1.0 -1.0]
end
# function coeftable(model::MultinomLogitModel; level::Real=0.95)
#     cc = coef(model)
#     se = stderror(model)
#     zz0 = cc ./ se
#     # p0 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz0)) #same as, but faster than: p0 = 2.0 * ccdf.(TDist(dof_residual(model)), abs.(zz0))
#     p0 = 2.0 * ccdf.(Normal(), abs.(zz0))

#     if !model.nested 
#         return CoefTable(hcat(cc, se, zz0, p0, confint(model; level)),
#         ["Estimate", "Std.Error", "z0 value", "Pr(>|z0|)", "Conf.Low $((level * 100))%", "Conf.High $((level * 100))%"],
#         coefnames(model),
#         4)
#     end
#     zz1 = (cc .- 1) ./ se
#     p1 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz1))
#     return CoefTable(hcat(cc, se, zz0, p0, zz1, p1, confint(model; level)),
#         ["Estimate", "Std.Error", "z0 value", "Pr(>|z0|)", "z1 value", "Pr(>|z1|)", "Conf.Low $((level * 100))%", "Conf.High $((level * 100))%"],
#         coefnames(model), 4)
# end
# # deviance
StatsBase.dof(model::LCLmodel) = model.dof
StatsBase.dof_residual(model::LCLmodel) = nobs(model) - dof(model)
# # fit
# # fit!
# function informationmatrix(model::MultinomLogitModel; expected::Bool=true)
#     if expected
#         @warn("Fisher (expected) information matrix not implemented. Returning observed information matrix.")
#         return model.hessian
#     else
#         return model.hessian
#     end
# end
StatsBase.isfitted(model::LCLmodel) = model.converged
StatsBase.islinear(model::LCLmodel) = false
StatsBase.loglikelihood(model::LCLmodel) = model.loglikelihood
# # mss
StatsBase.nobs(model::LCLmodel; use_nids::Bool=true) = use_nids ? model.nids : model.nchids
# # nulldeviance
StatsBase.nullloglikelihood(model::LCLmodel) = model.nullloglikelihood
StatsBase.r2(model::LCLmodel) = 1 - StatsBase.loglikelihood(model) / StatsBase.nullloglikelihood(model)
StatsBase.responsename(model::LCLmodel) = model.responsename
# # rss
# StatsBase.score(model::MultinomLogitModel) = model.score
StatsBase.stderror(model::LCLmodel) = sqrt.(diag(model.vcov))
StatsBase.vcov(model::LCLmodel) = model.vcov


stats_LCLmodel = (:N => Int ∘ nobs, "McFadden \$R^2\$" => r2, "LL" => loglikelihood, "AIC" => aic, "BIC" => bic)

# # TexTables.TableCol(header, t::MultinomLogitModel) = TexTables.TableCol(header, t; stats=stats_MultionLogitModel)

# function TexTables.TableCol(header, m::MultinomLogitModel;
#     stats=stats_MultionLogitModel,
#     meta=("test" => r2), stderror::Function=StatsBase.stderror, kwargs...)

#     if maximum(startswith.(coefnames(m), "lambda"))==1
#         @info "Significance levels (stars) for lambda estimates in nested logit models refer to H0: lambda=1"
#     end

#     # Compute p-values
#     pval(m) = ccdf.(FDist(1, dof_residual(m)),
#         abs2.((coef(m) .- startswith.(coefnames(m), "lambda")) ./ stderror(m)))

#     # Initialize the column
#     col = RegCol(header)

#     # Add the coefficients
#     for (name, val, se, p) in zip([startswith(cn, "lambda") ? cn * " (H0: .=1)" : cn for cn in coefnames(m)], coef(m), stderror(m), pval(m))
#         setcoef!(col, name, val, se)
#         0.05 < p <= 0.1 && star!(col[name], 1)
#         0.01 < p <= 0.05 && star!(col[name], 2)
#         p <= 0.01 && star!(col[name], 3)
#     end

#     # Add in the fit statistics
#     setstats!(col, OrderedDict(p.first => p.second(m) for p in TexTables.tuplefy(stats)))

#     # Add in the metadata
#     setmeta!(col, OrderedDict(p.first => p.second(m) for p in TexTables.tuplefy(meta)))

#     return col
# end

# # TexTables.regtable(t::MultinomLogitModel...) = TexTables.regtable(t; stats=stats_MultionLogitModel)

# function Base.show(io::IO, m::MultinomLogitModel)
#     # Your custom display logic
#     println(io, coeftable(m))
#     println(io, "Loglikelihood: ", round(loglikelihood(m), digits=4))
#     # Add more details as needed
# end

Base.length(rrs::LCLmodel) = 1
function RegressionTables.regtable(
    rrs::LCLmodel;
    renderSettings=nothing,
    render::T=RegressionTables.default_render(renderSettings, rrs),
    labels::Dict{String,String}=RegressionTables.default_labels(render, rrs),
    align::Symbol=RegressionTables.default_align(render),
    regression_statistics=RegressionTables.default_regression_statistics(render, rrs),
    transform_labels::Union{Dict,Symbol}=RegressionTables.default_transform_labels(render, rrs),
    digits_stats=nothing,
    estimformat=nothing,
    statisticformat=nothing,
    use_relabeled_values=RegressionTables.default_use_relabeled_values(render, rrs),
) where {T<:AbstractRenderType}
    if isa(transform_labels, Symbol)
        transform_labels = RegressionTables._escape(transform_labels)
    end

    nclasses = rrs.nclasses

    coefnames_mnl = use_relabeled_values ? RegressionTables.replace_name.(rrs.coefnames_mnl, Ref(labels), Ref(transform_labels)) : rrs.coefnames_mnl
    n_coef_mnl = Base.length(coefnames_mnl)
    coefvalues_mnl = reshape([RegressionTables.CoefValue(rrs, i) for i in 1:(n_coef_mnl*nclasses)], n_coef_mnl, nclasses)
    coefvalues_mnl = RegressionTables.repr.(render, coefvalues_mnl)
    coefbelow_mnl = reshape([RegressionTables.StdError(rrs, i) for i in 1:(n_coef_mnl*nclasses)], n_coef_mnl, nclasses)
    coefbelow_mnl = RegressionTables.repr.(render, coefbelow_mnl)

    coefnames_memb = use_relabeled_values ? RegressionTables.replace_name.(rrs.coefnames_memb, Ref(labels), Ref(transform_labels)) : rrs.coefnames_memb
    n_coef_memb = Base.length(coefnames_memb)
    coefvalues_memb = reshape([RegressionTables.CoefValue(rrs, i) for i in (n_coef_mnl*nclasses+1):dof(rrs)], n_coef_memb, nclasses - 1)
    coefvalues_memb = RegressionTables.repr.(render, coefvalues_memb)
    coefbelow_memb = reshape([RegressionTables.StdError(rrs, i) for i in (n_coef_mnl*nclasses+1):dof(rrs)], n_coef_memb, nclasses - 1)
    coefbelow_memb = RegressionTables.repr.(render, coefbelow_memb)

    shares = rrs.shares

    out = Vector{DataRow{T}}()

    align = 'l' * join(fill(align, nclasses), "")
    wdths = fill(0, nclasses + 1)

    breaks = [1]

    RegressionTables.push_DataRow!(out, [DataRow([missing; ["($i)" for i in 1:nclasses]])], align, wdths, false, render)

    for i in 1:n_coef_mnl
        RegressionTables.push_DataRow!(out, DataRow([coefnames_mnl[i]; coefvalues_mnl[i, :]]), align, wdths, false, render)
        RegressionTables.push_DataRow!(out, DataRow([missing; coefbelow_mnl[i, :]]), align, wdths, false, render)
    end

    push!(breaks, maximum(breaks) + 2 * n_coef_mnl)

    RegressionTables.push_DataRow!(out, DataRow(["share"; shares]), align, wdths, false, render)

    push!(breaks, maximum(breaks) + 1)

    for i in 1:n_coef_memb
        RegressionTables.push_DataRow!(out, DataRow([coefnames_memb[i]; coefvalues_memb[i, :]; missing]), align, wdths, false, render)
        RegressionTables.push_DataRow!(out, DataRow([missing; coefbelow_memb[i, :]; missing]), align, wdths, false, render)
    end

    push!(breaks, maximum(breaks) + n_coef_memb * 2)

    stats = RegressionTables.combine_statistics([rrs], regression_statistics)
    if digits_stats !== nothing
        stats = repr.(render, stats; digits=digits_stats)
    elseif statisticformat !== nothing
        stats = repr.(render, stats; str_format=statisticformat)
    end

    for i in axes(stats, 1), j in axes(stats, 2)
        if j == 2
            stats[i, j] = stats[i, j] => 2:(nclasses+1)
        end
    end

    RegressionTables.push_DataRow!(out, stats, "lc", wdths, false, render)

    RegressionTable(out, align, breaks)
end

function RegressionTables.default_regression_statistics(render::AbstractRenderType, model::LCLmodel)
    [Nobs, R2McFadden, AdjR2McFadden, LogLikelihood, AIC, BIC]
end

# function coefplot(model::LCLmodel; level::Real=0.95)
#     model_data = lclmodel_data(model, level=level)


#     # coefficient plot
#     fontsize_theme = Theme(fontsize=30)
#     set_theme!(fontsize_theme)
#     size = (1600, 600)

#     row = model_data.row
#     coef = model_data.coef
#     coefname = model_data.coefname
#     coefname_index = model_data.coefname_index

#     ci = confint(model, level=level)
#     ci_lo = model_data.ci_lo
#     ci_hi = model_data.ci_hi
#     significant = vec(sum(sign.(ci), dims=2) .!= 0)
#     marker = ifelse.(significant, :circle, :vline)

#     colorscheme = get(ColorSchemes.Dark2_8, range(0.0, 1.0, length=maximum(coefname_index)))
#     transform!(model_data, :coefname_index => ByRow(i -> colorscheme[i]) => :coefname_color)

#     # Create the plot
#     fig_coefs = Figure(size=size)
#     ax = Axis(fig_coefs[1, 1], xlabel="Coefficient", ylabel="Variable", yreversed=true)
#     # Red line at x=0
#     vlines!(ax, [0], color=:black, linewidth=3)
#     # Add horizontal lines for confidence intervals
#     for i in 1:length(coef)
#         linesegments!(ax, [(ci_lo[i], i), (ci_hi[i], i)],
#             color=:black, label=i == 1 ? "Confidence Interval" : nothing)
#     end

#     # connect points of a coefname
#     for cn in unique(coefname)
#         data_subset = subset(model_data, :coefname => x -> x .== cn)
#         lines!(ax, data_subset.coef, data_subset.row, color=data_subset.coefname_color)
#     end

#     # Scatter plot for coefficients
#     scatter!(ax, coef, row, marker=marker, color=model_data.coefname_color, label="Coefficient", markersize=20)
#     # scatter!(ax, coefs, 1:length(coefs), label="Coefficient", markersize=20)
#     # Customizing the y-axis to show variable names
#     ax.yticks = (1:length(coef), coefnames(model))
#     # Add a legend
#     # axislegend(ax)
#     fig_coefs[1, 2] = Legend(fig_coefs, ax, framevisible=false)
#     # Show the plot
#     return fig_coefs
# end

function lclmodel_data(model::LCLmodel; level=0.95)
    ci = confint(model, level=level)
    cc = [vec(model.coef_mnl); vec(model.coef_memb)]
    se = stderror(model)
    zz0 = cc ./ se
    # p0 = ccdf.(FDist(1, dof_residual(model)), abs2.(zz0)) #same as, but faster than: p0 = 2.0 * ccdf.(TDist(dof_residual(model)), abs.(zz0))
    p0 = 2.0 * ccdf.(Normal(), abs.(zz0))


    df = DataFrame(
        row=1:dof(model),
        model=[repeat([:mnl], Base.length(model.coefnames_mnl) * model.nclasses); repeat([:memb], Base.length(model.coefnames_memb) * (model.nclasses - 1))],
        coefname=[repeat(model.coefnames_mnl, outer=model.nclasses); repeat(model.coefnames_memb, outer=model.nclasses - 1)],
        class=[repeat(1:model.nclasses, inner=Base.length(model.coefnames_mnl)); repeat(1:(model.nclasses-1), inner=Base.length(model.coefnames_memb))],
        coef=coef(model),
        ci_lo=ci[:, 1],
        ci_hi=ci[:, 2],
        p0=p0
    )

    transform!(df, :coefname => (coefname -> remap_to_indices(coefname)) => :coefname_index)
    transform!(df, Cols(:ci_lo, :ci_hi) => ByRow((l, h) -> sign(l) + sign(h) != 0) => :significant)

    return df
end

function coefplot(model::LCLmodel; level=0.95, by=:class, mnlaxissettings=(;), membaxissettings=(;))
    fontsize_theme = Theme(fontsize=18)
    set_theme!(fontsize_theme)
    # size = (1600, 900)
    required_lines::Int64 = maximum([Base.size(model.coef_mnl, 1), Base.size(model.coef_memb, 1)]) * model.nclasses
    size = (1600, by == :coefhist ? Base.size(model.coef_mnl, 1) * 100 : required_lines * 25)

    model_data = lclmodel_data(model, level=level)
    colorscheme = get(ColorSchemes.Dark2_8, range(0.0, 1.0, length=maximum(model_data.coefname_index)))
    transform!(model_data, :coefname_index => ByRow(i -> colorscheme[i]) => :coefname_color)
    transform!(model_data, :significant => ByRow(s -> ifelse(s, :circle, :vline)) => :marker)

    data_mnl = subset(model_data, :model => x -> x .== :mnl)
    data_memb = subset(model_data, :model => x -> x .== :memb)

    nclasses = model.nclasses
    nmnlcoef = Base.length(model.coefnames_mnl)
    nmembcoef = Base.length(model.coefnames_memb)


    fig = Figure(size=size)

    if by == :class

        ax_mnl = Axis(fig[1:nclasses, 1:10], yreversed=true, xlabel="Coefficient Value", ylabel="Variable"; mnlaxissettings...)
        xlims!(ax_mnl, minimum(data_mnl.coef) * 1.1, maximum(data_mnl.coef) * 1.1)
        ylims!(ax_mnl, nclasses * nmnlcoef + 0.5, 0.5)
        ax_mnl.yticks = (1:nrow(data_mnl), [rich(row.coefname * get_stars(row.p0), color=get_label_color(row.p0, level=level)) for row in eachrow(data_mnl)])
        # display(data_mnl)

        for (i, mnlcoefname) in enumerate(unique(model_data[model_data.model.==:mnl, :].coefname))
            data_mnlcoef = subset(model_data, :coefname => x -> x .== mnlcoefname)

            scatter!(ax_mnl, data_mnlcoef.coef, data_mnlcoef.class .* nmnlcoef .+ i .- nmnlcoef, color=data_mnlcoef.coefname_color, marker=data_mnlcoef.marker)
            for ii in 1:nrow(data_mnlcoef)
                linesegments!(ax_mnl, [(data_mnlcoef.ci_lo[ii], data_mnlcoef.class[ii] .* nmnlcoef .+ i .- nmnlcoef), (data_mnlcoef.ci_hi[ii], data_mnlcoef.class[ii] .* nmnlcoef .+ i .- nmnlcoef)], color=data_mnlcoef.coefname_color[ii])
            end
        end

        vlines!(ax_mnl, [0], color=:black, linewidth=2)
        hlines!(ax_mnl, [i * nmnlcoef + 0.5 for i in 1:nclasses], color=:black, linewidth=1)

        # text!(repeat([minimum(data_mnl.coef) * 1.2], nclasses), (1:nclasses) .* nmnlcoef .- ((nmnlcoef - 1) / 2), text="Class " .* string.(1:nclasses) .* [@sprintf(", %.2f %%", model.shares[i] * 100) for i in 1:nclasses], align=(:left, :center), fontsize=17)


        # shares
        axs_share = [Axis(fig[c, 11], xlabelpadding=0) for c in 1:nclasses]
        linkyaxes!(axs_share...)
        # colgap!(fig.layout, 0)
        rowgap!(fig.layout, 0)

        for c in 1:nclasses
            # tightlimits!(axs_share[c])
            hidexdecorations!(axs_share[c])
            # hidespines!(axs_share[c])
            # ylims!(axs_share[c], 0, nothing)
            barplot!(axs_share[c], 1, model.shares[c], color="grey")
        end

        # membership

        ax_memb = Axis(fig[1:nclasses, 12:15], yreversed=true, yaxisposition=:right; membaxissettings...)
        xlims!(ax_memb, minimum(data_memb.coef) * 1.1, maximum(data_memb.coef) * 1.1)
        ylims!(ax_memb, nclasses * nmembcoef + 0.5, 0.5)
        ax_memb.yticks = (1:nrow(data_memb), [rich(row.coefname * get_stars(row.p0), color=get_label_color(row.p0, level=level)) for row in eachrow(data_memb)])

        for (i, membcoefname) in enumerate(unique(data_memb.coefname))
            data_membcoef = subset(data_memb, :coefname => x -> x .== membcoefname)

            scatter!(ax_memb, data_membcoef.coef, data_membcoef.class .* nmembcoef .+ i .- nmembcoef, color=data_membcoef.coefname_color, marker=data_membcoef.marker, markersize=model.shares[data_membcoef.class] .* 10 .* nclasses)
            for ii in 1:nrow(data_membcoef)
                linesegments!(ax_memb, [(data_membcoef.ci_lo[ii], data_membcoef.class[ii] .* nmembcoef .+ i .- nmembcoef), (data_membcoef.ci_hi[ii], data_membcoef.class[ii] .* nmembcoef .+ i .- nmembcoef)], color=data_membcoef.coefname_color[ii])
            end
        end

        vlines!(ax_memb, [0], color=:black, linewidth=2)
        hlines!(ax_memb, [i * nmembcoef + 0.5 for i in 1:nclasses], color=:black, linewidth=1)

        ax_mnl.title = "Multinomial Logit Models"
        axs_share[1].title = "Class Shares"
        ax_memb.title = "Membership Model"

        fig

    elseif by == :coef
        ax_mnl = Axis(fig[1, 1], yreversed=true, xlabel="Coefficient Value", ylabel="Variable"; mnlaxissettings...)
        xlims!(ax_mnl, minimum(data_mnl.coef) * 1.1, maximum(data_mnl.coef) * 1.1)
        ylims!(ax_mnl, nclasses * nmnlcoef + 0.5, 0.5)

        vlines!(ax_mnl, [0], color=:black, linewidth=3)
        hlines!(ax_mnl, [i * nclasses + 0.5 for i in 1:nmnlcoef], color=:black, linewidth=1)
        # ax_mnl.yticks = (1:nrow(data_mnl), repeat(model.coefnames_mnl, inner=nclasses) .* " - Class " .* string.(repeat(1:nclasses, outer=nmnlcoef)))
        ytickindexs = Int64[]
        yticklabels = Makie.RichText[]

        for (i, c) in enumerate(unique(model_data[model_data.model.==:mnl, :].class))
            data_class = subset(model_data, :class => x -> x .== c, :model => x -> x .== :mnl)

            scatter!(ax_mnl, data_class.coef, data_class.coefname_index .* nclasses .+ i .- nclasses, color=data_class.coefname_color, marker=data_class.marker, markersize=model.shares[c] * 15 * nclasses)
            for ii in 1:nrow(data_class)
                linesegments!(ax_mnl, [(data_class.ci_lo[ii], data_class.coefname_index[ii] .* nclasses .+ i .- nclasses), (data_class.ci_hi[ii], data_class.coefname_index[ii] .* nclasses .+ i .- nclasses)], color=data_class.coefname_color[ii])
                push!(ytickindexs, data_class.coefname_index[ii] .* nclasses .+ i .- nclasses)
                push!(yticklabels, rich("Class " * string(c) * " - " * data_class.coefname[ii] * get_stars(data_class.p0[ii]), color=get_label_color(data_class.p0[ii], level=level)))
                # push!(yticklabels, rich(data_class.coefname[ii] * " - Class " * string(c) * get_stars(data_class.p0[ii]), color=get_label_color(data_class.p0[ii], level=level)))
            end
        end
        ax_mnl.yticks = (ytickindexs, yticklabels)

        # text!(repeat([minimum(data_mnl.coef) * 1.2], nmnlcoef), (1:nmnlcoef) .* nclasses .- ((nclasses - 1) / 2), text=model.coefnames_mnl, align=(:left, :center), fontsize=17)

        fig
    elseif by == :coefhist
        # --- 1. Setup Axis ---
        ax_mnl = Axis(fig[1, 1], yreversed=true, xlabel="Coefficient Value", ylabel="Variable"; mnlaxissettings...)

        # Filter to just the MNL data needed
        data_mnl = subset(model_data, :model => x -> x .== :mnl)

        # Use the base variable names for the MNL part
        mnl_varnames = StatsModels.termnames(model.formula.rhs)
        nmnlcoef = length(mnl_varnames)

        # --- 2. Set axis limits and reference lines ---
        vlines!(ax_mnl, [0], color=:black, linewidth=2)
        hlines!(ax_mnl, [i + 0.5 for i in 1:(nmnlcoef-1)], color=:black, linewidth=0.75, linestyle=:dot)
        ylims!(ax_mnl, nmnlcoef + 0.5, 0.5)
        ax_mnl.yticks = (1:nmnlcoef, string.(mnl_varnames))

        # --- 3. Define plotting parameters ---
        # Controls the maximum total height of the symmetric line
        height_scale = 2

        # --- 4. Main plotting loop (iterate by variable) ---
        for i in 1:nmnlcoef
            # Get all class data for the current variable
            data_var = subset(data_mnl, :coefname_index => x -> x .== i)

            # Inner loop for each class within the variable
            for row in eachrow(data_var)
                # Determine color based on significance
                plot_color = row.significant ? :black : :gray60

                # --- FINAL SYMMETRIC PLOTTING LOGIC with a single centered line ---

                # Total height of the line is proportional to the class share
                total_height = model.shares[row.class] * height_scale

                # Calculate the top and bottom y-coordinates, centered on the row `i`
                y_top = i - total_height / 2
                y_bottom = i + total_height / 2

                # Draw a single, centered vertical line segment for each class
                linesegments!(ax_mnl,
                    [(row.coef, y_bottom), (row.coef, y_top)],
                    color=plot_color, linewidth=3)
            end
        end
        # The figure is returned at the end of the parent function
        fig
    end
end


# Helper function to get significance stars based on p-value
function get_stars(p_value::Real)
    if p_value < 0.001
        return "***"
    elseif p_value < 0.01
        return "**"
    elseif p_value < 0.05
        return "*"
        # Add 0.1 threshold if desired
        # elseif p_value < 0.1
        #     return "†" # Or another symbol like "."
    else
        return "" # No star
    end
end

# Helper function to get label color based on p-value and significance level
function get_label_color(p_value::Real; level::Real=0.95)
    # A coefficient is considered insignificant if its p-value is >= the alpha level (1 - confidence level)
    alpha = 1 - level
    is_significant = p_value < alpha
    return is_significant ? :black : :grey
end