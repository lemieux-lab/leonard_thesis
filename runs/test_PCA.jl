include("engines/init.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/classification.jl")
include("engines/factorized_embeddings.jl")
include("engines/pca.jl")
outpath, session_id = set_dirs("UMAP_RES")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
CDS = biotypes .== "protein_coding"
X_data = TCGA_data[:,CDS]
nfolds = 5
folds = split_train_test(X_data, nfolds = nfolds) # split 80-20
ACCs = []
modelid = "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
outdir = "inference_PCA_models"
foldn = 1

for foldn in 1:nfolds
    train_ids, train_data, test_ids, test_data = folds[foldn]["train_ids"], folds[foldn]["train_x"], folds[foldn]["test_ids"], folds[foldn]["test_x"]
    generate_pca_params(X_data; dim_redux_size = 4) = return Dict( 
        ## run infos 
        "session_id" => session_id,  "modelid" =>  modelid,
        "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
        "printstep"=>10000, "foldn" => foldn, "modeltype"=>"PCA",
        "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt",
        ## data infos 
        "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
        ## optim infos 
        
        ## model infos
        "n_components"=>dim_redux_size, 
        )
    params = generate_pca_params(train_data; dim_redux_size = 64)
    start_timer = now()
    train_PCA = fit_pca(train_data', params["n_components"])
    elapsed = (now() - start_timer).value / 1000
    train_PCA_x = transform_pca(train_data', train_PCA)
    test_PCA_x = transform_pca(test_data', train_PCA)
    ACC = test_performance_PCA(train_PCA_x, test_PCA_x, labs, train_ids, test_ids)

    params["classif_ACC"] = ACC
    params["total_time"] = elapsed
    dump_patient_embedding_CSV(train_PCA_x, "$(foldn)_train_$(params["n_components"])D_PCA", params, labs, train_ids,outdir=outdir)
    dump_patient_embedding_CSV(test_PCA_x, "$(foldn)_test_$(params["n_components"])D_PCA", params, labs, test_ids, outdir=outdir)
    println("$(ACC)\tElapsed time: $(elapsed)")
end 

# ## plotting embed using UMAP 
# fig = Figure(size = (512,512));
# ax2 = Axis(fig[1,1], xlabel = "UMAP-1", ylabel="UMAP-2", aspect = 1);
# for group_lab in unique(labs)
#     group = labs .== group_lab
#     plot!(umap_model.embedding[1,group],umap_model.embedding[2,group], label = group_lab)
# end 
