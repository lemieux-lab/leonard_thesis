include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/gpu_utils.jl")
include("engines/pairwise_distances.jl")
# compute pairwise distances (euclidian) in TCGA (Gene Expression values)
# outpath, session_id = set_dirs("FE_RES")
infile = "/home/golem/scratch/munozc/DDPM/full_TCGA_GE.h5"
inf = h5open(infile, "r")
data = inf["data_matrix"][:,:]
genes = inf["gene_id"][:]
biotypes = inf["gene_type"][:]
close(inf)


size(data)
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
ORIG_data = TCGA_data[:,CDS]

# prwd = sqrt.(sum(abs2, X_data[sample_ids,:] .- X_data[sample_ids[1],:]', dims = 2))
 
# compute pairwise distances in TCGA in FE space
# model_params = gather_params("figures/tables/")
# model_params = model_params[model_params[:,"emb_size_2"] .!= 2,:]
# model_params = model_params[model_params[:,"l2"] .!= 1e-4,:]
# function make_figure()

    FE_data = CSV.read("models/f1ee9a10160c95c93555f_trained_2D_factorized_embedding.csv", DataFrame)
    UMAP_data = CSV.read("inference_UMAP_models/27d128a5e9d188fd8a4d8_1_train_2D_UMAP.csv", DataFrame)
    sample_size = 200
    random_ids = shuffle(collect(1:size(UMAP_data)[1]))[1:sample_size]
    sample_ids = vec(UMAP_data[:,"id"])[random_ids]
    FE_space_dists = vec(compute_pairwise_dists(Matrix(FE_data), sample_ids))
    UMAP_space_dists = vec(compute_pairwise_dists(Matrix(UMAP_data[:, 1:2]), random_ids))
    orig_space_dists = vec(compute_pairwise_dists(ORIG_data, sample_ids))

    FE_Pearson = my_cor(orig_space_dists[orig_space_dists .!= 0], FE_space_dists[FE_space_dists .!= 0])
    UMAP_Pearson = my_cor(orig_space_dists[orig_space_dists .!= 0], UMAP_space_dists[UMAP_space_dists .!= 0])

    fig = Figure(size = (1024,1024));
    ax_FE= Axis(fig[1,1], title = "Sample pairwise Euclidian distances preservation \nwith Factorized Embeddings in TCGA (n=$(sample_size ^2))\n Pearson : $(FE_Pearson)",
    xlabel = "Patient (2D) Embedding distance",

    ylabel = "Gene Expression distance (19962 genes)")
    # hexbin!(ax, orig_space_dists[orig_space_dists .!= 0], FE_space_dists[FE_space_dists .!= 0], bins = 100)
    col, alpha, ms_size, strk_w = :grey, 0.5, 3, 0
    # scatter!(ax_FE, FE_space_dists[FE_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], color = (col,alpha), markersize = ms_size, strokewidth = strk_w)
    hexbin!(ax_FE, FE_space_dists[FE_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], bins = 150)
    a = add_linear_reg!(ax_FE, FE_space_dists[FE_space_dists .!= 0], orig_space_dists[FE_space_dists .!= 0])
    
    ax_UMAP= Axis(fig[1,2], title = "Sample pairwise Euclidian distances preservation \nwith UMAP in TCGA (n=$(sample_size ^2))\n Pearson : $(UMAP_Pearson)",
    xlabel = "UMAP (2D) Embedding distance",
    ylabel = "Gene Expression distance (19962 genes)")

    # scatter!(ax_UMAP, UMAP_space_dists[UMAP_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], color = (col,alpha), markersize = ms_size, strokewidth = strk_w)
    hexbin!(ax_UMAP, UMAP_space_dists[UMAP_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], bins = 150)

    UMAP_50D_data = CSV.read("inference_UMAP_models/dec1dfb1b37305ec22b72_1_train_50D_UMAP.csv", DataFrame)
    FE_50D_data = CSV.read("inference_1_0_FE_models/f19c2d92a453b0c26daae__1_train_50D_factorized_embedding.csv", DataFrame)

    sample_size = 200
    random_ids = shuffle(collect(1:size(UMAP_data)[1]))[1:sample_size]
    sample_ids = vec(UMAP_50D_data[:,"id"])[random_ids]
    UMAP_50D_orig_space_dists = vec(compute_pairwise_dists(ORIG_data, sample_ids))
    UMAP_50D_space_dists = vec(compute_pairwise_dists(Matrix(UMAP_50D_data[:, 1:50]), random_ids))

    sample_size = 200
    random_ids = shuffle(collect(1:size(FE_50D_data)[1]))[1:sample_size]
    sample_ids = vec(FE_50D_data[:,"id"])[random_ids]
    FE_50D_orig_space_dists = vec(compute_pairwise_dists(ORIG_data, sample_ids))
    FE_50D_space_dists = vec(compute_pairwise_dists(Matrix(FE_50D_data[:,1:50]), random_ids))


    FE_50D_Pearson = my_cor(FE_50D_orig_space_dists[FE_50D_orig_space_dists .!= 0], FE_50D_space_dists[FE_50D_space_dists .!= 0])
    UMAP_50D_Pearson = my_cor(UMAP_50D_orig_space_dists[UMAP_50D_orig_space_dists .!= 0], UMAP_50D_space_dists[UMAP_50D_space_dists .!= 0])

    ax_FE_50D= Axis(fig[2,1], title = "Sample pairwise Euclidian distances preservation \nwith Factorized Embeddings 50D in TCGA (n=$(sample_size ^2))\n Pearson : $(FE_50D_Pearson)",
    xlabel = "FE (50D) Embedding distance",
    ylabel = "Gene Expression distance (19962 genes)")

    # scatter!(ax_UMAP, UMAP_space_dists[UMAP_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], color = (col,alpha), markersize = ms_size, strokewidth = strk_w)
    hexbin!(ax_FE_50D, FE_50D_space_dists[FE_50D_space_dists .!= 0], FE_50D_orig_space_dists[FE_50D_orig_space_dists .!= 0], bins = 150)
    a = add_linear_reg!(ax_FE_50D, FE_50D_space_dists[FE_50D_space_dists .!= 0], orig_space_dists[FE_50D_space_dists .!= 0])


    ax_UMAP_50D= Axis(fig[2,2], title = "Sample pairwise Euclidian distances preservation \nwith UMAP 50D in TCGA (n=$(sample_size ^2))\n Pearson : $(UMAP_50D_Pearson)",
    xlabel = "UMAP (50D) Embedding distance",
    ylabel = "Gene Expression distance (19962 genes)")

    # scatter!(ax_UMAP, UMAP_space_dists[UMAP_space_dists .!= 0], orig_space_dists[orig_space_dists .!= 0], color = (col,alpha), markersize = ms_size, strokewidth = strk_w)
    hexbin!(ax_UMAP_50D, UMAP_50D_space_dists[UMAP_50D_space_dists .!= 0], UMAP_50D_orig_space_dists[UMAP_50D_orig_space_dists .!= 0], bins = 150)
    a = add_linear_reg!(ax_UMAP_50D,  UMAP_50D_space_dists[UMAP_50D_space_dists .!= 0], UMAP_50D_orig_space_dists[UMAP_50D_orig_space_dists .!= 0])

    CairoMakie.save("figures/preservation_distances_FE_UMAP_2D_50D.png",fig)
    CairoMakie.save("figures/preservation_distances_FE_UMAP_2D_50D.pdf",fig)


    fig, orig_space_dists, FE_space_dists,UMAP_space_dists, FE_50D_space_dists, UMAP_50D_space_dists
# end
fig, orig_space_dists, FE_space_dists,UMAP_space_dists, FE_50D_space_dists, UMAP_50D_space_dists =  make_figure()
fig
