import ModelingBase

function ModelingBase.fit(::Type{LinearModel}, 
            features::AbstractMatrix,
            labels::AbstractVector; 
            weights         ::  Union{Dict{<:Any, Float64}, Void}=nothing,
            solver_type     ::  Cint=L2R_L2LOSS_SVC_DUAL,
            eps             ::  Real=Inf,
            C               ::  Real=1.0,
            p               ::  Real=0.1,
            init_sol        ::  Ptr{Float64}=convert(Ptr{Float64}, C_NULL), # initial solutions for solvers L2R_LR, L2R_L2LOSS_SVC
            bias            ::  Real=-1.0,
            verbose         ::  Bool=false
    )
    
    linear_train(labels, features; 
            weights = weights,
            solver_type = solver_type,
            eps = eps,
            C = C,
            p = p,
            init_sol = init_sol,
            bias = bias,
            verbose = verbose,
    )
end



function ModelingBase.predict(model::LinearModel, features;
        probability_estimates   ::  Bool=false,
        verbose                 ::  Bool=false
    )
    classes, probs = linear_predict(model, features;
            probability_estimates=probability_estimates,
            verbose=verbose
    )
    vec(probs)
end

