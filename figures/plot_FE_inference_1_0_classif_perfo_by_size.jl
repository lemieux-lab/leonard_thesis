include("engines/init.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/classification.jl")
include("engines/factorized_embeddings.jl")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
## gather params from inference output dir 
## plot by model id 
df = gather_params("inference_1_0_FE_models")
minidf = df[:,["emb_size_1", "nsteps", "foldn", "modelid","classif_ACC"]]
results = unique(minidf[minidf[:,"nsteps"] .== 100_000,:], view=true)

umap_results = gather_params("inference_UMAP_models")
umap_mini_df = umap_results[:,["n_components", "foldn", "modelid","classif_ACC"]]
umap_results = unique(umap_mini_df, view=true)


emb_sizes_sorted = sort(unique(results[:, "emb_size_1"]))
means = [round(median(results[results[:,"emb_size_1"] .== emb, "classif_ACC"]) * 100, digits = 1)  for emb in  emb_sizes_sorted]

comp_size_sorted = sort(unique(umap_results[:, "n_components"]))
umap_vals = [round(median(umap_results[umap_results[:,"n_components"] .== emb, "classif_ACC"]) * 100, digits = 1)  for emb in  comp_size_sorted]

posit = Dict([(embsize, i) for (i, embsize) in  enumerate(emb_sizes_sorted)])
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], 
    title = "Average accuracy of cancer type prediction \nby number of nodes in the patient embedding layer",
    xlabel = "number of nodes in patient embedding layer", 
    ylabel = "Accuracy", 
    limits = (0, nothing, nothing, 100),
    xticks = (collect(1:length(posit)),string.(emb_sizes_sorted)),
    yticks = (collect(40:10:100),string.(collect(40:10:100))) );
boxplot!(ax, Int.([posit[emb] for emb in results[:,"emb_size_1"]]) .- 0.2, Float32.(results[:,"classif_ACC"]) * 100, width = 0.5, show_outliers = false, label = "FE")
scatter!(ax,  Int.([posit[emb] for emb in results[:,"emb_size_1"]]) .- 0.2, Float32.(results[:,"classif_ACC"]) * 100, strokewidth=2)
text!(ax, collect(1:length(emb_sizes_sorted)) .- 0.85, means .-1, text = string.(means), fontsize = 14)

boxplot!(ax, Int.([posit[emb] for emb in umap_results[:,"n_components"]]) .+ 0.2, Float32.(umap_results[:,"classif_ACC"]) * 100,width = 0.5, show_outliers = false, label = "UMAP")
scatter!(ax,  Int.([posit[emb] for emb in umap_results[:,"n_components"]]) .+ 0.2, Float32.(umap_results[:,"classif_ACC"]) * 100, strokewidth=2)
text!(ax, collect(1:length(comp_size_sorted)), umap_vals .+ 2, text = string.(umap_vals), fontsize = 14)

axislegend(ax, position=:rb)
fig
CairoMakie.save("figures/FE_inference_1_0_classification_by_embed_size.pdf", fig)
CairoMakie.save("figures/FE_inference_1_0_classification_by_embed_size.png", fig)

# num_labs = format_Y(labs)
# train_model_df = CSV.read("tmp/4c5ead59348d891df8d95__1_train_2D_factorized_embedding.csv", DataFrame)
# train_ids = train_model_df[:,"id"]
# train_Y = gpu(Matrix(num_labs[train_ids,:]'))
# train_X = gpu(Matrix(Matrix(train_model_df[:,1:50])'))

# inference_model_df = CSV.read("tmp/4c5ead59348d891df8d95__1_inference_1_0_2D_factorized_embedding.csv", DataFrame)
# test_ids = inference_model_df[:,"id"]
# test_Y = gpu(Matrix(num_labs[test_ids,:]'))
# test_X = gpu(Matrix(Matrix(inference_model_df[:,1:50])'))

# model = train_DNN(train_X, train_Y, test_X, test_Y, nsteps = 2000)    
# using libcudnn