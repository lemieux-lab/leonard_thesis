include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
outpath, session_id = set_dirs("FE_RES")
### TEST FE Leucegene
# LGN_data = MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")
### TEST FE TCGA  
infile = h5open("Data/GDC_processed/TCGA_TPM_lab.h5")
TCGA_data = log10.(infile["data"][:,:] .+ 1)
labs = string.(infile["labels"][:])
patients = string.(infile["rows"][:])
genes = string.(infile["cols"][:]) 
close(infile)

tcga_abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame)
labels = innerjoin(DataFrame(:abbrv=>[x[2] for x in split.(labs,"-")]),tcga_abbrv, on = :abbrv )[:,"def"]
labels = ["$lab (n=$(sum(labels .== lab)))" for lab in labels]
### sanity check with t-sne 
X_embed_tsne = tsne(TCGA_data[:,CDS], 2, 50, 1000, 30, verbose = true, progress = true)
# TSNE_df = DataFrame(:TSNE1 => X_embed_tsne[:,1], :TSNE2 => X_embed_tsne[:,2], :group => labs)
# p = AlgebraOfGraphics.data(TSNE_df) * mapping(:TSNE1, :TSNE2, color = :group, marker=:group)
# draw(p; axis = (width = 1024, height = 512))
fig = Figure(size = (1024,800));
ax2 = Axis(fig[1,1],title="TSNE of TCGA", xlabel = "TSNE-1", ylabel="TSNE-2", aspect = 1);
markers = [:diamond, :circle, :utriangle, :rect]
nlabs = length(unique(labels))
for (i, group_lab) in enumerate(unique(labels))
    group = labels .== group_lab
    scatter!(ax2, X_embed_tsne[group,1], X_embed_tsne[group,2], strokewidth = 0.1, color = RGBf(rand(), rand(), rand()), marker = markers[i%4 + 1], label = group_lab)
end 
fig[1,2] = axislegend(ax2, position =:rc, labelsize = 8, rowgap=0)
fig

tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
TCGA_datasets = load_tcga_datasets(tcga_datasets_list)
BRCA_data = TCGA_datasets["BRCA"]
# LGG_data = TCGA_datasets["LGG"]
# OV_data = TCGA_datasets["OV"]
# TCGALAML_data = TCGA_datasets["LAML"]
# Composite_TCGA_data = vcat(BRCA_data["dataset"].data, LGG_data["dataset"].data, OV_data["dataset"].data, TCGALAML_data["dataset"].data)
# X_data =Composite_TCGA_data .- mean(Composite_TCGA_data, dims =1)
# tcga_labels = vcat(fill("BRCA", size(BRCA_data["dataset"].data)[1]),
# fill("LGG", size(LGG_data["dataset"].data)[1]),
# fill("OV", size(OV_data["dataset"].data)[1]),
# fill("TCGA-LAML", size(TCGALAML_data["dataset"].data)[1]))
# CDS = BRCA_data["dataset"].biotypes .== "protein_coding"
# Composite_TCGA_data = Composite_TCGA_data[:,CDS]
#projects_num = [findall(unique(labs) .== X)[1] for X in labs] 
# X_embed_tsne = tsne(Composite_TCGA_data, 2, 500, 3000,30, verbose = true, progress = true)
# TSNE_df = DataFrame(:TSNE1 => X_embed_tsne[:,1], :TSNE2 => X_embed_tsne[:,2], :group => tcga_labels)
# p = AlgebraOfGraphics.data(TSNE_df) * mapping(:TSNE1, :TSNE2, color = :group, marker=:group)
# draw(p)
CDS = BRCA_data["dataset"].biotypes .== "protein_coding"
# TCGA_genes = genes[CDS]
# X, Y = prep_FE(TCGA_data, patients, genes[CDS]);
# X, Y = prep_FE(LGN_data.data, LGN_data.samples, LGN_data.genes);
# X_data = TCGA_data[:,CDS] .- mean(TCGA_data[:,CDS], dims = 1)
X_data = TCGA_data[:,CDS]
X, Y = prep_FE(X_data, patients, genes[CDS]);

params = Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> length(genes[CDS]), 
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 5_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )

trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X,Y, params, labs)
DataFrame(:tr_epoch=>tr_epochs, :tr_loss=>tr_loss, tr_cor=>tr_cor)

cpu(model.embed_1.weight)
umap_model = UMAP_(TCGA_data', 2;min_dist = 0.99, n_neighbors = 200);

## plotting embed using UMAP 
fig = Figure(size = (512,512));
ax2 = Axis(fig[1,1], xlabel = "UMAP-1", ylabel="UMAP-2", aspect = 1);
for group_lab in unique(labs)
    group = labs .== group_lab
    plot!(umap_model.embedding[1,group],umap_model.embedding[2,group], label = group_lab)
end 
