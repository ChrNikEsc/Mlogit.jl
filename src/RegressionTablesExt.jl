
# Copied and adapted from https://github.com/jmboehm/RegressionTables.jl/blob/0fb6d7dfda349a963efbd5702a2dc96e819b643b/ext/RegressionTablesFixedEffectModelsExt.jl#L47
# needed to show clusters in RegressionTables

function RegressionTables.other_stats(model::Union{MNLmodel, FMLmodel}, s::Symbol)
    if s == :clusters && model.nclusters !== nothing
        collect(RegressionTables.ClusterCoefName.(string.(keys(model.nclusters))) .=> RegressionTables.ClusterValue.(values(model.nclusters)))
    else
        nothing
    end
end

RegressionTables.default_transform_labels(render::AbstractLatex, rrs) = Dict("&" => "\\&", "%" => "\\%", "\$" => "\\\$", "#" => "\\#", "_" => "\\_", "{" => "\\{", "}" => "\\}", "^" => "\\^{}")

# RegressionTables.R2McFadden(model::MNLmodel) = r2(model)



# stats_MNLmodel = (:N => Int ∘ nobs, "McFadden \$R^2\$" => r2, "LL" => loglikelihood, "AIC" => aic, "BIC" => bic)

# TexTables.TableCol(header, t::MNLmodel) = TexTables.TableCol(header, t; stats=stats_MultionLogitModel)

# function TexTables.TableCol(header, m::MNLmodel;
#     stats=stats_MNLmodel,
#     meta=(), stderror::Function=StatsBase.stderror, type=nothing, cluster=nothing, kwargs...)

#     if maximum(startswith.(coefnames(m), "lambda")) == 1
#         @info "Significance levels (stars) for lambda estimates in nested logit models refer to H0: lambda=1"
#     end

#     # Compute p-values
#     pval(m) = ccdf.(FDist(1, dof_residual(m)),
#         abs2.((coef(m) .- startswith.(coefnames(m), "lambda")) ./ stderror(m, type=type, cluster=cluster)))

#     # Initialize the column
#     col = RegCol(header)

#     # Add the coefficients
#     for (name, val, se, p) in zip([startswith(cn, "lambda") ? cn * " (H0: .=1)" : cn for cn in coefnames(m)], coef(m), stderror(m, type=type, cluster=cluster), pval(m))
#         setcoef!(col, name, val, se)
#         0.05 < p <= 0.1 && star!(col[name], 1)
#         0.01 < p <= 0.05 && star!(col[name], 2)
#         p <= 0.01 && star!(col[name], 3)
#     end

#     # Add in the fit statistics
#     setstats!(col, OrderedDict(p.first => p.second(m) for p in TexTables.tuplefy(stats)))

#     # Add in the metadata
#     setmeta!(col, OrderedDict(p.first => p.second(m) for p in TexTables.tuplefy(meta)))
#     setmeta!(col, OrderedDict("Rob. SE." => isnothing(type) ? "-" : type))
#     setmeta!(col, OrderedDict("Cluster" => cluster isa DataFrame ? replace(join(string.(names(cluster)), ", "), "_" => "\\_") : ""))

#     return col
# end

# TexTables.regtable(t::MNLmodel...) = TexTables.regtable(t; stats=stats_MultionLogitModel)