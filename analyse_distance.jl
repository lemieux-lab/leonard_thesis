include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/gpu_utils.jl")

# compute pairwise distances (euclidian) in TCGA (Gene Expression values)
# outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
ORIG_data = TCGA_data[:,CDS]
sample_size = 100
sample_ids = shuffle(1:size(ORIG_data)[1])[1:sample_size]
# prwd = sqrt.(sum(abs2, X_data[sample_ids,:] .- X_data[sample_ids[1],:]', dims = 2))
function compute_pairwise_dists(X_data, sample_ids)
    MM= zeros(sample_size, sample_size)
    for (row_id, sample_id) in enumerate(sample_ids)
        prwd = sqrt.(sum(abs2, X_data[sample_ids,:] .- X_data[sample_id,:]', dims = 2))
        MM[row_id,:] .= prwd
    end 
    return MM 
end 
orig_space_dists = vec(compute_pairwise_dists(ORIG_data, sample_ids))
# compute pairwise distances in TCGA in FE space
# model_params = gather_params("figures/tables/")
# model_params = model_params[model_params[:,"emb_size_2"] .!= 2,:]
# model_params = model_params[model_params[:,"l2"] .!= 1e-4,:]

FE_data = CSV.read("figures/tables/f1ee9a10160c95c93555f_trained_2D_factorized_embedding.csv", DataFrame)
FE_space_dists = vec(compute_pairwise_dists(Matrix(FE_data), sample_ids))

Pearson = my_cor(orig_space_dists[orig_space_dists .!= 0], FE_space_dists[FE_space_dists .!= 0])

fig = Figure(size = (512,512));
ax= Axis(fig[1,1], title = "Sample pairwise Euclidian distances preservation \nwith Factorized Embeddings in TCGA (n=10,000)\n Pearson : $(Pearson)",
xlabel = "Patient (2D) Embedding distance",
ylabel = "Gene Expression distance (19962 genes)")
# hexbin!(ax, orig_space_dists[orig_space_dists .!= 0], FE_space_dists[FE_space_dists .!= 0], bins = 100)
scatter!(ax, FE_space_dists[FE_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], color = :grey, markersize = 7, strokewidth = 0.5, bins = 100)
CairoMakie.save("figures/preservation_distances_FE.pdf",fig)
CairoMakie.save("figures/preservation_distances_FE.png",fig)

