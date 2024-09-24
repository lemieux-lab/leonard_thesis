include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/pca.jl")
outpath, session_id = set_dirs("FE_RES")
BRCA_data, labs, samples, genes, biotypes = load_tcga_dataset("Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5")
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
embs = 50
# Train

folds = split_train_test(BRCA_data[:,CDS], nfolds = 5) # split 80-20
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
generate_params(X_data, emb_size) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>1000, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-8,"nsteps" => 100_000, "nsteps_inference" => 10_000, "nsamples_batchsize" => 4,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => 100, "fe_layers_size"=> [250, 100], #, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    ## plotting infos 
    "colorsFile"=> "Data/GDC_processed/BRCA_colors_def.txt"
    )
# train with training set
params_dict = generate_params(train_data, embs)
# save IDs
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, samples[train_ids], genes[CDS], params_dict, labs[train_ids])
# infer FE
inference_model, part1_fig, part2_fig = do_inference_B(trained_FE, train_data, train_ids, test_data, test_ids, samples, genes[CDS], params_dict)

# feed to COX-DNN
# infer 