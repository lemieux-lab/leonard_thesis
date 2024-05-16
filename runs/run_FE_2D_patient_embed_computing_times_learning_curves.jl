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
    "printstep"=>10000, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 100_000, "nsteps_inference" => 10_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [250,75,50,25,10]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
params = generate_params(X_data)
# save IDs
# splits_d = BSON.load("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson")

# train whole dataset 
X_train, Y_train = prep_FE(X_data, patients, genes[CDS]);
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_train,Y_train, params, labels)
# train with training set
params = generate_params(train_data)
bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
    Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
    "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))
X_train, Y_train = prep_FE(X_data, patients[train_ids], genes[CDS]);
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_train,Y_train, params, labels[train_ids])

# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# prep test data points 
X_test, Y_test = prep_FE(test_data,)
# inference 
test_model = do_inference(trained_FE, params, test_data,  patients[test_ids], genes[CDS] )
# plot final 2D patient embed (train + test)
train_test_2d_fig = plot_train_test_patient_embed(trained_FE, test_model, labels, train_ids, test_ids)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.png", train_test_2d_fig)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.pdf", train_test_2d_fig)


## PlotlyJS
plot_interactive(params, train_embed, test_embed, train_ids, test_ids,  labels)
