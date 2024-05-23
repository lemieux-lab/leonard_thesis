struct MLSurvDataset
    data::Matrix
    samples::Array 
    genes::Array
    biotypes::Array
    labels::Array
    survt::Array
    surve::Array
end 
function MLSurvDataset(infilename)
    infile = h5open(infilename, "r")
    data = infile["data"][:,:] 
    samples = infile["samples"][:]  
    labels = infile["labels"][:]  
    genes = infile["genes"][:]  
    biotypes = infile["biotypes"][:]
    survt = infile["survt"][:]  
    surve = infile["surve"][:]  
    close(infile)
    return MLSurvDataset(data, samples, genes, biotypes, labels, survt, surve)
end 
function load_tcga_datasets(infiles)
    
    out_dict = Dict()
    for infile in infiles
        DataSet = MLSurvDataset(infile)
        name_tag = split(infile,"_")[3]
        out_dict["$(name_tag)"] = Dict("dataset"=>DataSet,
        "name" => name_tag
        ) 
    end 
    return out_dict
end 


function prep_FE(data::Matrix,patients::Array,genes::Array, device=gpu)
    n = length(patients)
    m = length(genes)
    # k = length(tissues)
    values = Array{Float32,2}(undef, (1, n * m))
    patient_index = Array{Int64,1}(undef, max(n * m, 1))
    gene_index = Array{Int64,1}(undef, max(n * m, 1))
    # tissue_index = Array{Int32,1}(undef, n * m)
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1,index] = data[i,j] # to debug 
            patient_index[index] = i # Int
            gene_index[index] = j # Int 
            # tissue_index[index] = tissues[i] # Int 
        end
    end 
    shfl = shuffle(collect(1:length(values)))
    return (device(patient_index[shfl]), device(gene_index[shfl])), device(vec(values[shfl]))
    # return (device(patient_index), device(gene_index)), device(vec(values))

end 


struct FE_model
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hls
    outpl::Flux.Dense
    opt
    lossf
end

# function l2_penalty(model::FE_model)
#     return sum(abs2, model.embed_1) + sum(abs2, model.embed_2) + sum.(abs2, model.hls)
# end
# function l2_penalty(ps::Flux.params)
#     return sum(p -> sum(abs2, p), ps)
# end 

# function mse_l2(model::FE_model, X, Y;weight_decay = 1e-6)
#     return Flux.mse(model.net(X), Y) + l2_penalty(model) * weight_decay
# end 


function FE_model(params::Dict)
    emb_size_1 = params["emb_size_1"]
    emb_size_2 = params["emb_size_2"]
    a = emb_size_1 + emb_size_2 
    # b, c = params["fe_hl1_size"], params["fe_hl2_size"]#, params["fe_hl3_size"] ,params["fe_hl4_size"] ,params["fe_hl5_size"] 
    emb_layer_1 = gpu(Flux.Embedding(params["nsamples"], emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(params["ngenes"], emb_size_2))
    hlayers = []
    for (i,layer_size) in enumerate(params["fe_layers_size"][1:end])
        i == 1 ? inpsize = a : inpsize = params["fe_layers_size"][i - 1]
        push!(hlayers, Flux.Dense(inpsize, layer_size, relu))
    end 
    # hl1 = gpu(Flux.Dense(a, b, relu))
    # hl2 = gpu(Flux.Dense(b, c, relu))
    # hl3 = gpu(Flux.Dense(c, d, relu))
    # hl4 = gpu(Flux.Dense(d, e, relu))
    # hl5 = gpu(Flux.Dense(e, f, relu))
    outpl = gpu(Flux.Dense(params["fe_layers_size"][end], 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hlayers..., outpl,
        vec))
    opt = Flux.ADAM(params["lr"])
    lossf = mse_l2
    FE_model(net, emb_layer_1, emb_layer_2, hlayers, outpl, opt, lossf)
end 
function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 

function train_SGD!(params, X, Y, model; printstep = 1_000)
    start_timer = now()
    batchsize = params["batchsize"]
    nminibatches = Int(floor(length(Y) / batchsize))
    tr_loss, tr_epochs, tr_cor, tr_elapsed = [], [], [], []
    opt = Flux.ADAM(params["lr"])
    println("1 epoch 1 - 1 /$nminibatches - TRAIN \t ELAPSED: $((now() - start_timer).value / 1000 ) Shuffling ...")         
        
    shuffled_ids = collect(1:length(Y)) # very inefficient
    for iter in 1:params["nsteps"]
        # Stochastic gradient descent with minibatches
        cursor = (iter -1)  % nminibatches + 1
        if cursor == 1 
            shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
        end 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
        ids = shuffled_ids[mb_ids]
        X_, Y_ = (X[1][ids],X[2][ids]), Y[ids]
        ps = Flux.params(model.net)
        # dump_cb(model, params, iter + restart)
        gs = gradient(ps) do 
            Flux.mse(model.net(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps) ## loss
        end
        lossval = Flux.mse(model.net(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
        pearson = my_cor(model.net(X_), Y_)
        Flux.update!(opt,ps, gs)
        push!(tr_cor, pearson)
        push!(tr_loss, lossval)
        push!(tr_epochs, Int(ceil(iter / nminibatches)))
        push!(tr_elapsed, (now() - start_timer).value / 1000 )
        (iter % 100 == 0) | (iter == 1) ? println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ELAPSED: $((now() - start_timer).value / 1000 )") : nothing        
            
        if (iter % printstep == 0) 
            CSV.write("$(params["outpath"])/$(params["modelid"])_loss_computing_times", DataFrame(:tr_epochs=>tr_epochs, :tr_loss=>tr_loss, :tr_elapsed=>tr_elapsed))
            # # save model 
            bson("$(params["outpath"])/$(params["modelid"])_in_training_model.bson", Dict("model"=> cpu(model.net)))
        end 
    end
    # save model 
    bson("$(params["outpath"])/$(params["modelid"])_model_$(params["nsteps"]).bson", Dict("model"=> cpu(model.net)))
    return tr_epochs, tr_loss, tr_cor, tr_elapsed
end 

function train_embed_SGD!(params, X, Y, model)
    batchsize = params["batchsize"]
    nminibatches = Int(floor(length(Y) / batchsize))
    tr_loss, tr_epochs, tr_cor = [], [], []
    opt = Flux.ADAM(params["lr"])
    shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
    for iter in 1:params["nsteps"]
        # Stochastic gradient descent with minibatches
        cursor = (iter -1)  % nminibatches + 1
        if cursor == 1 
            shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
        end 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
        ids = shuffled_ids[mb_ids]
        X_, Y_ = (X[1][ids],X[2][ids]), Y[ids]
        ps = Flux.params(model.net[1][1])
        # dump_cb(model, params, iter + restart)
        gs = gradient(ps) do 
            Flux.mse(model.net(X_), Y_) + params["l2"] * l2_penalty(model)
        end
        lossval = Flux.mse(model.net(X_), Y_) + params["l2"] * l2_penalty(model)
        pearson = my_cor(model.net(X_), Y_)
        Flux.update!(opt,ps, gs)
        push!(tr_cor, pearson)
        push!(tr_loss, lossval)
        push!(tr_epochs, Int(ceil(iter / nminibatches)))
        println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson")
    end
    return tr_epochs, tr_loss, tr_cor
end 

function generate_patient_embedding(X_data, patients, genes, params, tissue_labels)
    bson("$(params["outpath"])/$(params["modelid"])_params.bson", params)
    X, Y = prep_FE(X_data, patients, genes);
    ## init model
    model = FE_model(params);

    # train loop
    tr_epochs, tr_loss, tr_cor, tr_elapsed = train_SGD!(params, X, Y, model, printstep = params["printstep"])

    reconstruction_fig = plot_FE_reconstruction(model, X, Y, modelID=params["modelid"])
    CairoMakie.save("$(params["outpath"])/$(params["modelid"])_FE_reconstruction.pdf", reconstruction_fig)
    CairoMakie.save("$(params["outpath"])/$(params["modelid"])_FE_reconstruction.png", reconstruction_fig)


    ## plotting embed directly 
    patient_embed_fig = Figure(size = (1024,800));
    trained_patient_FE = cpu(model.net[1][1].weight)
    patient_embed_fig = plot_patient_embedding(trained_patient_FE, patient_embed_fig, tissue_labels, "trained 2-d embedding\n$(params["modelid"])", 1) 
    CairoMakie.save("$(params["outpath"])/$(params["modelid"])_trained_2D_factorized_embedding.pdf", patient_embed_fig)
    CairoMakie.save("$(params["outpath"])/$(params["modelid"])_trained_2D_factorized_embedding.png", patient_embed_fig)
    
    ### plotting training curves
    training_curve_fig = Figure(size = (512,512));
    ax1 = Axis(training_curve_fig[1,1], title = "Training Pearson correlation by step\n$(params["modelid"])",
    xlabel = "step", ylabel = "Pearson correlation")
    ax2 = Axis(training_curve_fig[2,1], title = "Training loss by step",
    xlabel = "step", ylabel = "loss (log scale)")
    lines!(ax1, collect(1:size(tr_cor)[1]), Float32.(tr_cor))
    lines!(ax2, collect(1:size(tr_loss)[1]), log10.(Float32.(tr_loss)))

    CairoMakie.save("$(params["outpath"])/$(params["modelid"])_training_curve_factorized_embedding.pdf", training_curve_fig)
    CairoMakie.save("$(params["outpath"])/$(params["modelid"])_training_curve_factorized_embedding.png", training_curve_fig)

    return model, tr_epochs,tr_loss, tr_cor
end 
function reset_embedding_layer(trained_FE, params, size)
    hlayers = deepcopy(trained_FE[2:end])
    test_FE = Flux.Chain(
        Flux.Parallel(vcat, 
            Flux.Embedding(size,params["emb_size_1"]),
            deepcopy(trained_FE[1][2])
        ),
        hlayers...
    ) |> gpu
    return test_FE    
end 


function reset_embedding_layer_sample_init(FE_net, params, test_size)
    hlayers = deepcopy(FE_net[2:end])
    embed_layer_train = FE_net[1][1].weight
    x_ids = collect(1:size(embed_layer_train)[2])
    shuffled_ids = shuffle(x_ids)
    init_embed = embed_layer_train[:,shuffled_ids[1:test_size]]
    test_FE = Flux.Chain(
        Flux.Parallel(vcat, 
            Flux.Embedding(init_embed),
            deepcopy(FE_net[1][2])
        ),
        hlayers...
    ) |> gpu
    return test_FE    
end 

function do_inference(trained_FE, params, test_data, test_patients, genes; pre_trained_init = true)
    X_test, Y_test = prep_FE(test_data, test_patients, genes);
    # reset patient embedding layer
    pre_trained_init ? inference_model = reset_embedding_layer_sample_init(trained_FE, params, size(test_data)[1]) : inference_model = reset_embedding_layer(trained_FE, params, size(test_data)[1]) 
    # do inference 
    batchsize = params["batchsize"]
    nminibatches = Int(floor(length(Y_test) / batchsize))
    opt = Flux.ADAM(params["lr"])
    for iter in 1:params["nsteps_inference"]
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_test)))
        X_, Y_ = (X_test[1][mb_ids],X_test[2][mb_ids]), Y_test[mb_ids]
        ps = Flux.params(inference_model[1][1])
        gs = gradient(ps) do 
            Flux.mse(inference_model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
        end
        lossval = Flux.mse(inference_model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
        pearson = my_cor(inference_model(X_), Y_)
        Flux.update!(opt,ps, gs)
        iter % 100 == 0 ?  println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ") : nothing
    end 
    return inference_model
end 