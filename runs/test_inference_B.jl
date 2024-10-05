include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/pca.jl")
include("engines/coxphdnn.jl")
include("engines/gpu_utils.jl")
# CUDA.device!()
outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, samples, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS = biotypes .== "protein_coding"

dim_redux_size = 2
input_type = "FE"
nsteps_dim_redux = 100_000
printstep_FE = 1_000 

generate_params(X_data, emb_size) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>printstep_FE, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 5e-3, "l2" => 1e-7,"nsteps" => nsteps_dim_redux, "nsteps_inference" => Int(floor(nsteps_dim_redux * 0.1)), "nsamples_batchsize" => 1,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => 100, "fe_layers_size"=> [250, 100], #, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    ## plotting infos 
    "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt"
    )

nfolds = 5
folds = split_train_test(TCGA_data, nfolds = nfolds) # split 80-20
ACCs = []
modelid = "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
# outdir = "tmp"
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
params_dict = generate_params(train_data, dim_redux_size)
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, samples[train_ids], genes, params_dict, labs[train_ids])
inference_model, part1_fig, part2_fig = do_inference_B(trained_FE, train_data, train_ids, test_data, test_ids, samples, genes, params_dict)
cpu(trained_FE[1][1].weight)
CairoMakie.save("figures/FE_2D_inference.pdf", part1_fig)

fig = plot_train_test_patient_embed(trained_FE, inference_model, labs, train_ids, test_ids, params_dict)
