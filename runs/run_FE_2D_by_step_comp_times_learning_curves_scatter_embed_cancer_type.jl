include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/gpu_utils.jl")
device!()
outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
X_data = TCGA_data[:,CDS]
# split train / test 
# folds = split_train_test(X_data, nfolds = 5) # split 80-20
# train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
for gene_embsize in [10, 25, 75,100,1000]
generate_params(X_data;nsamples_batchsize=4) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>1000, 
    "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt",
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    "nsamples_batchsize"=> nsamples_batchsize, 
    ## optim infos 
    "lr" => 1e-2, "l2" => 0,"nsteps" => 100_000, "nsteps_inference" => 10_000, "batchsize" => nsamples_batchsize * size(X_data)[2],
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => gene_embsize, "fe_layers_size"=> [100,50,50]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# train whole dataset 
# params = generate_params(X_data)
# trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# train with training set
params = generate_params(X_data, nsamples_batchsize = 10)
# output_labels_legend(params)

# save IDs
# bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
#     Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
#     "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))

trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labs)

trained_patient_FE_df = DataFrame(Dict([("embed_$i",cpu(trained_FE.net[1][1].weight)[i,:]) for i in 1:params["emb_size_1"]]))
CSV.write("figures/tables/$(params["modelid"])_trained_2D_factorized_embedding.csv", trained_patient_FE_df)
bson("figures/tables/$(params["modelid"])_params.bson", params)
end     
# X, Y = prep_FE(X_data, patients, genes[CDS], order = "per_sample");

# # Option 1 : quickest TOO SLOW 
# # by patient in training set
# new_embed = zeros((params["nsamples"], 2));
# nsamples_batchsize = 200
# batchsize = params["ngenes"] * nsamples_batchsize
# nminibatches = Int(floor(params["nsamples"] / nsamples_batchsize))

# for target_id in 1:params["nsamples"]
#     #generate imputed profile
#     target_range = (target_id -1) * params["ngenes"] + 1 : min(target_id * params["ngenes"], params["nsamples"] * params["ngenes"])
#     Y_ = Y[target_range]
#     best_MSE = 10e8
#     for iter in 1:nminibatches
#         batch_range = (iter -1) * batchsize + 1 : min(iter * batchsize, params["nsamples"] * batchsize)
#         X_ = (X[1][batch_range],X[2][batch_range]) 
#         @time trained_FE.net(X_)
#         # @time MM = reshape(trained_FE.net(X_), (nsamples_batchsize, params["ngenes"]))
#         # diff_2_MM = vec(sum((MM .- Y_') .^ 2, dims = 2))
#         # diff_2_MM_min_val = minimum(diff_2_MM)
#         # diff_2_MM_min_index = (iter - 1) * nsamples_batchsize +  findfirst(diff_2_MM .== diff_2_MM_min_val)
        
#         # if diff_2_MM_min_val <= best_MSE
#         #     new_embed[target_id, :] .= vec(cpu(trained_FE.net[1][1].weight[:,diff_2_MM_min_index]))
#         #     best_MSE = diff_2_MM_min_val
#         # end 
#     end 
#     target_id % 10 == 0 ? println("$target_id $(patients[target_id]) \t ") : nothing
# end 

# # implement patient embedding restart 
# # copy 
# infer_model = reset_embedding_layer_sample_init(trained_FE.net,params,params["nsamples"])

# # impute training set patient embed, compute MSE to target, take lowest 20%
# # fine-tune patient embed 2000 steps
# # 


# # trained_patient_FE = cpu(trained_FE.net[1][1].weight)
# trained_patient_FE = cpu(infer_model[1][1].weight)
# TCGA_colors_labels_df = CSV.read("Data/GDC_processed/TCGA_colors_def.txt", DataFrame)

# patient_embed_fig = plot_patient_embedding(trained_patient_FE, labs, "trained 2-d embedding\n$(params["modelid"])", params["colorsFile"]) 
# CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_$(iter).png", patient_embed_fig)
# # inference         

# function train_restart_model_SGD_per_sample!(params, X, Y, model; printstep = 1_000)
#     start_timer = now()
#     nsamples_batchsize = params["nsamples_batchsize"]
#     batchsize = params["ngenes"] * nsamples_batchsize
#     nminibatches = Int(floor(params["nsamples"] / nsamples_batchsize))
#     tr_loss, tr_epochs, tr_cor, tr_elapsed = [], [], [], []
#     opt = Flux.ADAM(params["lr"])
#     println("1 epoch 1 - 1 /$nminibatches - TRAIN \t ELAPSED: $((now() - start_timer).value / 1000 )")         
        
#     # shuffled_ids = collect(1:length(Y)) # very inefficient
#     for iter in 1:params["nsteps"]
#         # Stochastic gradient descent with minibatches
#         cursor = (iter -1)  % nminibatches + 1
#         # if cursor == 1 
#         #     shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
#         # end 
#         # id_range = (cursor -1) * batchsize + 1:min(cursor * batchsize, params["nsamples"])
#         # ids = shuffled_ids[id_range]
#         batch_range = (cursor -1) * batchsize + 1 : min(cursor * batchsize, params["nsamples"] * batchsize)
#         X_, Y_ = (X[1][batch_range],X[2][batch_range]), Y[batch_range]
#         ps = Flux.params(model[1][1])
#         # dump_cb(model, params, iter + restart)
#         gs = gradient(ps) do 
#             Flux.mse(model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps) ## loss
#         end
#         lossval = Flux.mse(model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
#         pearson = my_cor(model(X_), Y_)
#         Flux.update!(opt,ps, gs)
#         push!(tr_cor, pearson)
#         push!(tr_loss, lossval)
#         push!(tr_epochs, Int(ceil(iter / nminibatches)))
#         push!(tr_elapsed, (now() - start_timer).value / 1000 )
#         (iter % 100 == 0) | (iter == 1) ? println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ELAPSED: $((now() - start_timer).value / 1000 )") : nothing        
            
#         if (iter % printstep == 0) 
#             CSV.write("$(params["outpath"])/$(params["modelid"])_loss_computing_times", DataFrame(:tr_epochs=>tr_epochs, :tr_loss=>tr_loss, :tr_elapsed=>tr_elapsed))
#             # # save model 
#             # bson("$(params["outpath"])/$(params["modelid"])_restart_in_training_model.bson", Dict("model"=> cpu(model)))
#             trained_patient_FE = cpu(model[1][1].weight)
#             patient_embed_fig = plot_patient_embedding(trained_patient_FE, labs, "trained 2-d embedding\n$(params["modelid"]) \n- step $(iter)", params["colorsFile"]) 
#             CairoMakie.save("$(params["outpath"])/$(params["modelid"])_restart_2D_embedding_$(iter).png", patient_embed_fig)
#         end 
#     end
#     # save model 
#     # bson("$(params["outpath"])/$(params["modelid"])_model_$(params["nsteps"]).bson", Dict("model"=> cpu(model.net)))
#     return tr_epochs, tr_loss, tr_cor, tr_elapsed
# end

# train_restart_model_SGD_per_sample!(params, X, Y, infer_model)
# trained_patient_FE = cpu(infer_model[1][1].weight)
# TCGA_colors_labels_df = CSV.read("Data/GDC_processed/TCGA_colors_def.txt", DataFrame)

# patient_embed_fig = plot_patient_embedding(trained_patient_FE, labs, "trained 2-d embedding\n$(params["modelid"])", params["colorsFile"]) 

# #test_model = do_inference(trained_FE.net, params, test_data,  patients[test_ids], genes[CDS] )
# # inference with pre-training init embed  
# #test_model = do_inference(trained_FE.net, params, test_data,  patients[test_ids], genes[CDS], pre_trained_init = true)

# # plot final 2D patient embed (train + test)
# train_test_2d_fig = plot_train_test_patient_embed(trained_FE, test_model, labels, train_ids, test_ids)
# CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.png", train_test_2d_fig)
# CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.pdf", train_test_2d_fig)


# ## PlotlyJS
# plot_interactive(params, cpu(trained_FE.net[1][1].weight), cpu(test_model[1][1].weight), train_ids, test_ids,  labels)

# sample_profile = test_data[125,:]
# gen_grid(nb_p, span) = collect(span[1]:(span[2] - span[1]) / nb_p:span[2]) 
# grid = gen_grid(100,[-3,3])
# grid_vals = zeros(size(grid)[1] , size(grid)[1])
# gene_embed = trained_FE.net[1][2](gpu(collect(1:size(genes[CDS])[1])))
# Xs = []
# Ys = []
# for i in 1:size(grid)[1]
#     println(i)
#     for j in 1:size(grid)[1]
#         grid_val = vcat(gpu(ones(size(genes[CDS])[1]) * grid[i])', gpu(ones(size(genes[CDS])[1]) * grid[j])', gene_embed)
        
#         infer_profile = vec(trained_FE.net[2:end](grid_val))
#         grid_vals[i,j] =  cpu(my_cor(gpu(sample_profile), infer_profile))
#         push!(Xs, grid[i])
#         push!(Ys, grid[j])
#     end 
# end 
# fig = Figure(size = (512,512));
# ax = Axis(fig[1,1], title = "Contour", xlabel = "EMBED-1", ylabel = "EMBED-2" )
# # grid_vals
# co = contourf!(ax, grid, grid, grid_vals', levels = 20)
# Colorbar(fig[1,2], co)
# fig