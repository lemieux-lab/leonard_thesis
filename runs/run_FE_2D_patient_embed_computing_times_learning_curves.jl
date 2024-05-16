include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
X_data = TCGA_data[:,CDS]
# split train / test 
folds = split_train_test(X_data, nfolds = 5) # split 80-20
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
generate_params(X_data) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>10_000, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 20_000, "nsteps_inference" => 10_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [25]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# train whole dataset 
# params = generate_params(X_data)
# trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# train with training set
params = generate_params(train_data)
# save IDs
bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
    Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
    "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, patients[train_ids], genes[CDS], params, labels[train_ids])

# inference         
test_model = do_inference(trained_FE.net, params, test_data,  patients[test_ids], genes[CDS] )
# plot final 2D patient embed (train + test)
train_test_2d_fig = plot_train_test_patient_embed(trained_FE, test_model, labels, train_ids, test_ids)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.png", train_test_2d_fig)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.pdf", train_test_2d_fig)


## PlotlyJS
plot_interactive(params, cpu(trained_FE.net[1][1].weight), cpu(test_model[1][1].weight), train_ids, test_ids,  labels)

sample_profile = test_data[250,:]
grid = collect(-25:1:25) ./ 10
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
ax = Axis(fig[1,1], title = "Heatmap", xlabel = "EMBED-1", ylabel = "EMBED-2" )
scatter!(ax, Float64.(Xs), Float64.(Ys) , color=vec(grid_vals), colormap=:viridis)
# fig[1,2] = Colorbar(scat)
Xs[vec(grid_vals) .== maximum(vec(grid_vals))]
Ys[vec(grid_vals) .== maximum(vec(grid_vals))]
# Xs[vec(grid_vals) .== maximum(vec(grid_vals))]
# Ys[vec(grid_vals) .== maximum(vec(grid_vals))]
# grid_vals[]
# grid_vals
fig