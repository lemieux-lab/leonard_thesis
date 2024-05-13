include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
tcga_abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame)
labels = innerjoin(DataFrame(:abbrv=>[x[2] for x in split.(labs,"-")]),tcga_abbrv, on = :abbrv )[:,"def"]
labels = ["$lab (n=$(sum(labels .== lab)))" for lab in labels]
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
X_data = TCGA_data
X, Y = prep_FE(X_data, patients, genes);

params = Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> length(genes), 
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 100_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [250,75,50,25,10]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# model = FE_model(params)

trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X,Y, params, labels)


### TEST FE Leucegene
# LGN_data = MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")
### TEST FE TCGA  
### CORRUPTED !!
# infile = h5open("Data/GDC_processed/TCGA_TPM_lab.h5")
# TCGA_data = log10.(infile["data"][:,:] .+ 1)
# labs = string.(infile["labels"][:])
# patients1 = string.(infile["rows"][:])
# genes = string.(infile["cols"][:]) 
# close(infile)

# infile = h5open("Data/GDC_processed/TCGA_TPM_hv_subset.h5")
# TCGA_data = infile["data"][:,:]
# labs = string.(infile["labels"][:])
# patients2 = string.(infile["rows"][:])
# genes = string.(infile["cols"][:]) 
# close(infile)

# infile = h5open("Data/GDC_processed/TCGA_TPM_lab_corrupted.h5")
# patients1 = string.(infile["rows"][:])
# close(infile)
# TCGA_raw_files = readdir("Data/GDC_raw/TCGA")
# intersect(patients1, patients2)
# length(patients2)
# length(TCGA_raw_files)

### reassemble data
# new_patient_list = intersect(patients2, TCGA_raw_files)
# inputfile = CSV.read("Data/GDC_raw/TCGA/$(new_patient_list[1])", delim = "\t", header= 2, DataFrame)[5:end,:]
# output_data = fill(0., (length(new_patient_list), size(inputfile)[1]))
# for (i,star_data) in enumerate(new_patient_list)
#     if i % 10 == 0
#         println("$i / $(length(new_patient_list))")
#     end 
#     inputfile = CSV.read("Data/GDC_raw/TCGA/$star_data", delim = "\t", header= 2, DataFrame)[5:end,:]
#     output_data[i,:] .= Float64.(inputfile[:,"tpm_unstranded"])
# end 
# genes = inputfile[:,"gene_name"]
# biotypes = inputfile[:,"gene_type"]
# # new_labels = innerjoin(DataFrame(:case_id=>new_patient_list), DataFrame(:case_id=>patients2, :group=>labs), on = :case_id)[:,"group"]
# outfile = h5open("Data/GDC_processed/TCGA_TPM_lab.h5", "w")
# outfile["data"] = log10.(output_data .+ 1)
# outfile["genes"] = string.(genes) 
# outfile["biotypes"] = string.(biotypes) 
# outfile["samples"] = string.(new_patient_list) 
# outfile["labels"] = string.(new_labels)
# close(outfile)


# infile = h5open("Data/GDC_processed/TCGA_TPM_lab.h5")
# TCGA_data = infile["data"][:,:]
# labs = string.(infile["labels"][:])
# patients = string.(infile["samples"][:])
# genes = string.(infile["genes"][:]) 
# biotypes = string.(infile["biotypes"][:])
# close(infile)


# sum(patients1 .== patients2)
# labels_reorder = innerjoin(DataFrame(:case_id=>patients2), DataFrame(:case_id=>patients1, :labs=>labs),on = :case_id)[:,"labs"] 
### sanity check with t-sne 
# X_embed_tsne = tsne(TCGA_data[:,CDS], 2, 50, 1000, 30, verbose = true, progress = true)
# TSNE_df = DataFrame(:TSNE1 => X_embed_tsne[:,1], :TSNE2 => X_embed_tsne[:,2], :group => labs)
# p = AlgebraOfGraphics.data(TSNE_df) * mapping(:TSNE1, :TSNE2, color = :group, marker=:group)
# draw(p; axis = (width = 1024, height = 512))
# fig = Figure(size = (1024,800));
# ax2 = Axis(fig[1,1],title="TSNE of TCGA", xlabel = "TSNE-1", ylabel="TSNE-2", aspect = 1);
# markers = [:diamond, :circle, :utriangle, :rect]
# nlabs = length(unique(labels))
# for (i, group_lab) in enumerate(unique(labels))
#     group = labels .== group_lab
#     scatter!(ax2, X_embed_tsne[group,1], X_embed_tsne[group,2], strokewidth = 0.1, color = RGBf(rand(), rand(), rand()), marker = markers[i%4 + 1], label = group_lab)
# end 
# fig[1,2] = axislegend(ax2, position =:rc, labelsize = 8, rowgap=0)
# fig

# tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
# TCGA_datasets = load_tcga_datasets(tcga_datasets_list)
# BRCA_data = TCGA_datasets["BRCA"]
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
# TCGA_genes = genes[CDS]
# X, Y = prep_FE(TCGA_data, patients, genes[CDS]);
# X, Y = prep_FE(LGN_data.data, LGN_data.samples, LGN_data.genes);
# X_data = TCGA_data[:,CDS] .- mean(TCGA_data[:,CDS], dims = 1)
tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

using PlotlyJS
trained_FE_mat = cpu(trained_FE.net[1][1].weight)
# X_tr = fit_transform_pca(trained_FE_mat,2)
FE_to_plot = DataFrame(:ids => patients, :EMBED1=> trained_FE_mat[1,:], :EMBED2=>trained_FE_mat[2,:], :group=>labels) 
# FE_to_plot = DataFrame(:EMBED1=> X_tr[1,:], :EMBED2=>X_tr[2,:], :group=>labels) 

FE_to_plot = innerjoin(FE_to_plot, DataFrame(:group => unique(labels), :type=>[i%3 for i in 1:size(unique(labels))[1]]),on = :group)
CSV.write("interactive_figures/$(params["modelid"])_FE_2D_embedding.csv", FE_to_plot)
P = PlotlyJS.plot(
    FE_to_plot, x=:EMBED1, y=:EMBED2, color=:group, symbol = :type, ids = :ids,
    kind = "scatter", mode = "markers", 
        Layout(
            title = "FE 2D visualisation by subgroup"

))
PlotlyJS.savefig(P, "interactive_figures/$(params["modelid"])_FE_2D_visualisation.html")
cpu(model.embed_1.weight)
umap_model = UMAP_(TCGA_data', 2;min_dist = 0.99, n_neighbors = 200);

## plotting embed using UMAP 
fig = Figure(size = (512,512));
ax2 = Axis(fig[1,1], xlabel = "UMAP-1", ylabel="UMAP-2", aspect = 1);
for group_lab in unique(labs)
    group = labs .== group_lab
    plot!(umap_model.embedding[1,group],umap_model.embedding[2,group], label = group_lab)
end 
