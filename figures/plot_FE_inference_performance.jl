include("engines/init.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/classification.jl")
include("engines/factorized_embeddings.jl")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
## gather params from inference output dir 
## plot by model id 
# df = gather_params("inference_B_FE_models")
df = gather_params("inference_B_FE_models")
minidf = df[:,["emb_size_1", "l2", "emb_size_2","nsteps", "foldn", "modelid","classif_ACC"]]
minidf = minidf[minidf[:,"emb_size_1"] .!= 512,:]
FE_results = unique(minidf[(minidf[:,"nsteps"] .== 100_000),:], view=true)
FE_stats = combine(groupby(FE_results,:emb_size_1), "classif_ACC"=>mean, "classif_ACC"=>std)
fig = Figure(size = (750,512));
ax = Axis(fig[1,1], 
        title = "Average accuracy of cancer type prediction \nby sample embedding layer dimension",
        xlabel = "Size of dimensionality of reduction", 
        ylabel = "Accuracy (%)",
        limits = (nothing, nothing, nothing, 100),
        xticks = (log2.(sort(unique(FE_results[:,"emb_size_1"]))),string.(sort(unique(FE_results[:,"emb_size_1"])))),
        yticks = (collect(30:10:100),string.(collect(30:10:100))) );
fig, ax = data_to_axis(ax, 
    FE_results[:,"emb_size_1"], # X
    FE_results[:, "classif_ACC"], # Y 
    FE_stats[:,"emb_size_1"], # X_mean
    FE_stats[:, "classif_ACC_mean"], # Y_mean
    FE_stats[:, "classif_ACC_std"], # Y_std
    "Factorized Embeddings"); # lbl 
umap_data = gather_params("inference_UMAP_models")
umap_mini_df = umap_data[:,["n_components", "foldn", "modelid","classif_ACC"]]
keep = [in(x, [2,4,8,16,32,64,128,256]) for x in umap_mini_df[:,"n_components"]]
umap_results = unique(umap_mini_df[keep,:], view=true)
umap_stats =  combine(groupby(umap_results,:n_components), "classif_ACC"=>mean, "classif_ACC"=>std)
umap_stats[:,"n_components"]
umap_stats[:, "classif_ACC_std"]
umap_stats[:, "classif_ACC_mean"]
fig, ax = data_to_axis(ax, 
    umap_results[:,"n_components"], # X
    umap_results[:, "classif_ACC"], # Y 
    umap_stats[:,"n_components"], # X_mean
    umap_stats[:, "classif_ACC_mean"], # Y_mean
    umap_stats[:, "classif_ACC_std"], # Y_std
    "UMAP"); # lbl 
pca_results = gather_params("inference_PCA_models")
pca_mini_df = pca_results[:,["n_components", "foldn", "modelid","classif_ACC"]]
pca_results = unique(pca_mini_df, view=true)
pca_stats = combine(groupby(pca_results,:n_components), "classif_ACC"=>mean, "classif_ACC"=>std)
fig, ax = data_to_axis(ax, 
    pca_results[:,"n_components"], # X
    pca_results[:, "classif_ACC"], # Y 
    pca_stats[:,"n_components"], # X_mean
    pca_stats[:, "classif_ACC_mean"], # Y_mean
    pca_stats[:, "classif_ACC_std"], # Y_std
    "PCA"); # lbl 
axislegend(ax, position = :rb)
fig
CairoMakie.save("figures/FE_inference_classification_by_embed_size.pdf", fig)
CairoMakie.save("figures/FE_inference_classification_by_embed_size.png", fig)

