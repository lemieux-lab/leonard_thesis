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
folds = split_train_test(X_data, nfolds = 5) # split 80-20
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
generate_params(X_data;nsamples_batchsize=1) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>1000, 
    "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt",
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    "nsamples_batchsize"=> nsamples_batchsize, 
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 100_000, "nsteps_inference" => 10_000, "batchsize" => size(X_data)[2] * nsamples_batchsize,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [150,100,50]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# train whole dataset 
# params = generate_params(X_data)
# trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

model = FE_model(params)
model.net[1]
# train with training set
params = generate_params(X_data, nsamples_batchsize=4)
# save IDs
# bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
#     Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
#     "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labs)
inference_model = do_inference(trained_FE.net, params,X_data,patients,genes[CDS])

trained_patient_FE = cpu(inference_model[1][1].weight)
TCGA_colors_labels_df = CSV.read("Data/GDC_processed/TCGA_colors_def.txt", DataFrame)

patient_embed_fig = plot_patient_embedding(trained_patient_FE, labs, "trained 2-d embedding\n$(params["modelid"])", params["colorsFile"]) 
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_$(iter).png", patient_embed_fig)
# inference         
#test_model = do_inference(trained_FE.net, params, test_data,  patients[test_ids], genes[CDS] )
# inference with pre-training init embed  
#test_model = do_inference(trained_FE.net, params, test_data,  patients[test_ids], genes[CDS], pre_trained_init = true)

# plot final 2D patient embed (train + test)
train_test_2d_fig = plot_train_test_patient_embed(trained_FE, test_model, labels, train_ids, test_ids)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.png", train_test_2d_fig)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.pdf", train_test_2d_fig)


## PlotlyJS
plot_interactive(params, cpu(trained_FE.net[1][1].weight), cpu(test_model[1][1].weight), train_ids, test_ids,  labels)

sample_profile = test_data[125,:]
gen_grid(nb_p, span) = collect(span[1]:(span[2] - span[1]) / nb_p:span[2]) 
grid = gen_grid(100,[-3,3])
grid_vals = zeros(size(grid)[1] , size(grid)[1])
gene_embed = trained_FE.net[1][2](gpu(collect(1:size(genes[CDS])[1])))
Xs = []
Ys = []
for i in 1:size(grid)[1]
    println(i)
    for j in 1:size(grid)[1]
        grid_val = vcat(gpu(ones(size(genes[CDS])[1]) * grid[i])', gpu(ones(size(genes[CDS])[1]) * grid[j])', gene_embed)
        
        infer_profile = vec(trained_FE.net[2:end](grid_val))
        grid_vals[i,j] =  cpu(my_cor(gpu(sample_profile), infer_profile))
        push!(Xs, grid[i])
        push!(Ys, grid[j])
    end 
end 
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], title = "Contour", xlabel = "EMBED-1", ylabel = "EMBED-2" )
# grid_vals
co = contourf!(ax, grid, grid, grid_vals', levels = 20)
Colorbar(fig[1,2], co)
fig