include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/gpu_utils.jl")
include("engines/classification.jl")
device!()
outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS = biotypes .== "protein_coding"

# X_data = TCGA_data[:,CDS]
X_data = TCGA_data[:,CDS]
nfolds = 5
folds = split_train_test(X_data, nfolds = nfolds) # split 80-20
ACCs = []
modelid = "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
outdir = "inference_B_FE_models"
# outdir = "tmp"

for foldn in 1:nfolds
    train_ids, train_data, test_ids, test_data = folds[foldn]["train_ids"], folds[foldn]["train_x"], folds[foldn]["test_ids"], folds[foldn]["test_x"]
    # set params 
    generate_params(X_data;emb_size_2 =75, emb_size_1 = 2, nsamples_batchsize=4) = return Dict( 
        ## run infos 
        "session_id" => session_id,  "modelid" =>  modelid,
        "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
        "printstep"=>20_000, "foldn"=>foldn, 
        "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt",
        ## data infos 
        "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
        "nsamples_batchsize"=> nsamples_batchsize, 
        ## optim infos 
        "lr" => 1e-2, "l2" => 1e-8,"nsteps" => 100_000, "nsteps_inference" => 10_000, "batchsize" => nsamples_batchsize * size(X_data)[2],
        ## model infos
        "emb_size_1" => emb_size_1, "emb_size_2" => emb_size_2, "fe_layers_size"=> [100,50,50]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
        )

    # train with training set
    params_dict = generate_params(train_data, emb_size_1 = 128, emb_size_2 = 75, nsamples_batchsize = 4)

    # save IDs
    bson("$(params_dict["outpath"])/$(params_dict["modelid"])_train_test_ids.bson", 
        Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
        "model_prefix"=> "$(params_dict["outpath"])/$(params_dict["modelid"])"))
    trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, patients[train_ids], genes[CDS], params_dict, labs[train_ids])
    inference_model, part1_fig, part2_fig = do_inference_B(trained_FE, train_data, train_ids, test_data, test_ids, patients, genes[CDS], params_dict)

    ACC = test_performance_FE(trained_FE, inference_model, labs, train_ids, test_ids)
    params_dict["classif_ACC"] = ACC
    dump_patient_embedding_CSV(cpu(trained_FE.net[1][1].weight), "$(foldn)_train_$(params_dict["emb_size_1"])D_factorized_embedding", params_dict, labs, train_ids,outdir=outdir)
    dump_patient_embedding_CSV(cpu(inference_model[1][1].weight), "$(foldn)_inference_B_$(params_dict["emb_size_1"])D_factorized_embedding", params_dict, labs, test_ids,outdir=outdir)
end 
