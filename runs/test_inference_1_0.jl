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
nfolds = 5
folds = split_train_test(X_data, nfolds = nfolds) # split 80-20
ACCs = []
for foldn in 1:nfolds
    train_ids, train_data, test_ids, test_data = folds[foldn]["train_ids"], folds[foldn]["train_x"], folds[foldn]["test_ids"], folds[foldn]["test_x"]
    ## train for 80% 
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
        "lr" => 1e-2, "l2" => 1e-8,"nsteps" => 100_000, "nsteps_inference" => 10_000, "batchsize" => nsamples_batchsize * size(X_data)[2],
        ## model infos
        "emb_size_1" => 2, "emb_size_2" => 75, "fe_layers_size"=> [100,50,50]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
        )

    # train with training set
    params = generate_params(train_data, nsamples_batchsize = 10)

    trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, patients[train_ids], genes[CDS], params, labs[train_ids])
    test_FE = do_inference(trained_FE.net, params, test_data, patients[test_ids], genes[CDS])

    ## visualize 
    fig = plot_train_test_patient_embed(trained_FE, inference_model, labs, train_ids, test_ids)
    CairoMakie.save("figures/FE_inference_1_0.png", fig)
    CairoMakie.save("figures/FE_inference_1_0.pdf", fig)

    #fig[1,2] = axislegend(ax, position =:rc, labelsize = 8, rowgap=0)
    # train classifier 
    include("engines/classification.jl")

    ACC = test_performance_FE(trained_FE, inference_model, labs, train_ids, test_ids)
    push!(ACCs, ACC)
    break
end 
# report training 
