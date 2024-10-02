function format_surv_data(x_data, t_data, e_data)
    ordering = sortperm(-t_data)
    x_ = gpu(Matrix(x_data[ordering,:]'));
    y_t_ = gpu(Matrix(t_data[ordering,:]'));
    y_e_ = gpu(Matrix(e_data[ordering,:]'));
    NE_frac = sum(y_e_ .== 1) != 0 ? 1 / sum(y_e_ .== 1) : 0
    return x_, y_t_, y_e_, NE_frac
end 
function coxnll(outs, Y_e_, NE_frac)
    hazard_ratios = exp.(outs)
    log_risk = log.(cumsum(hazard_ratios, dims = 2))
    uncensored_likelihood = outs .- log_risk
    censored_likelihood = uncensored_likelihood .* Y_e_
    # neg_likelihood = - sum(censored_likelihood) / sum(Y_e_ .== 1)
    neg_likelihood = - sum(censored_likelihood) * NE_frac
    return neg_likelihood
end 


function concordance_index(T, E, S)
    concordant_pairs = S .< S'
    admissable_pairs = T .< T'
    discordant_pairs = S .> S'
    tied_pairs = sum(S .== S') - length(T)
    concordant = sum(E .* (admissable_pairs .* concordant_pairs))
    discordant = sum(E .* (admissable_pairs .* discordant_pairs) )
    C_index = concordant / (concordant + discordant)  
    return C_index, concordant, discordant, tied_pairs
end

function train_cphdnn!(cphdnn_params, cphdnn, x_train, y_e_train, NE_frac_tr, x_test, y_e_test, NE_frac_tst)
    opt = Flux.setup(OptimiserChain(Flux.WeightDecay(cphdnn_params["l2"]), Flux.Optimise.Adam(cphdnn_params["lr"])), cphdnn);
    tst_c_index, tst_scores = 0, [] 
    for stepn in 1:cphdnn_params["nsteps"]
        grads = Flux.gradient(cphdnn) do m
            loss = coxnll(m(x_train), y_e_train, NE_frac_tr)
        end
        tr_lossval = coxnll(cphdnn(x_train), y_e_train, NE_frac_tr)
        tst_lossval = coxnll(cphdnn(x_test), y_e_test, NE_frac_tst)
        tr_c,_,_,_ = concordance_index(y_t_train, y_e_train, -1 * cphdnn(x_train))
        tst_c,_,_,_ = concordance_index(y_t_test, y_e_test, -1 * cphdnn(x_test))
        tst_c_index = tst_c
        Flux.update!(opt, cphdnn, grads[1])
        if stepn % cphdnn_params["printstep"] == 0
            println("$stepn TRAIN loss: $(round(tr_lossval, digits = 4)) c-index: $(round(tr_c, digits = 4)) TEST loss: $(round(tst_lossval,digits = 4)) c-index: $(round(tst_c, digits = 4))")
        end 
    end 
    return tst_c_index, cphdnn(x_test)
end