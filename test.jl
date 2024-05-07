include("engines/init.jl")
include("engines/factorized_embeddings.jl")

### TEST FE Leucegene
LGN_data = MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")
### TEST FE TCGA  
infile = h5open("Data/GDC_processed/TCGA_TPM_lab.h5")
TCGA_data = log10.(infile["data"][:,:] .+ 1)
labs = string.(infile["labels"][:])
patients = string.(infile["rows"][:])
genes = string.(infile["cols"][:]) 
close(infile)

### sanity check with t-sne 
size(TCGA_data)
# X_embed_tsne = tsne(TCGA_data, 2, 50, 3000,30, verbose = true, progress = true)

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
TCGA_genes = genes[CDS]
# X, Y = prep_FE(TCGA_data, patients, genes[CDS]);
# X, Y = prep_FE(LGN_data.data, LGN_data.samples, LGN_data.genes);
X_data = TCGA_data[:,CDS] .- mean(TCGA_data[:,CDS], dims = 1)
X, Y = prep_FE(X_data, collect(1:size(X_data)[1]), TCGA_genes );

batchsize = 40_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))

params = Dict( "nsteps" => 40_000,
    "emb_size_1" => 2,
    "emb_size_2" => 50,
    "fe_hl1_size" => 50,
    "fe_hl2_size" => 50,
    # "fe_hl3_size" => 100,
    # "fe_hl4_size" => 50,
    # "fe_hl5_size" => 10,

    "nsamples" =>size(TCGA_data)[1],
    "ngenes"=> length(genes[CDS]), 
    "lr" => 1e-2,
    "wd" => 1e-3)
## 
model = FE_model(params);

## plotting embed directly 
patient_FE = cpu(model.embed_1.weight) 

fig = Figure(size = (1024,512));
ax2 = Axis(fig[1,1],title="Before-training embedding", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
markers = [:xcross, :hexagon, :utriangle, :cross]
for (i, group_lab) in enumerate(unique(labs))
    group = labs .== group_lab
    plot!(ax2, patient_FE[1,group],patient_FE[2,group], marker = markers[i%4 + 1], label = group_lab)
end 
fig[1,2] = axislegend(ax2)
fig


tr_loss = []
tr_epochs = []
opt = Flux.ADAM(params["lr"])
nminibatches = Int(floor(length(Y) / batchsize))
# shuffled_ids = shuffle(collect(1:length(Y)))

for iter in 1:params["nsteps"]
    ps = Flux.params(model.net)
    cursor = (iter -1)  % nminibatches + 1
    if cursor == 1 
        shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
    end 
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
    ids = shuffled_ids[mb_ids]
    X_, Y_ = (X[1][ids],X[2][ids]), Y[ids]
    
    # dump_cb(model, params, iter + restart)
    
    gs = gradient(ps) do 
        Flux.mse(model.net(X_), Y_) + params["wd"] * l2_penalty(model)
    end
    # if params.clip 
    #     g_norm = norm(gs)
    #     c = 0.5
    #     g_norm > c && (gs = gs ./ g_norm .* c)
    #     # if g_norm > c
    #     #     println("EPOCH: $(iter) gradient norm $(g_norm)")
    #     #     println("EPOCH: $(iter) new grad norm $(norm(gs ./ g_norm .* c))")
    #     # end 
    # end 
    lossval = mse_l2(model, X_, Y_; weight_decay = params["wd"])
    pearson = my_cor(model.net(X_), Y_)
    Flux.update!(opt,ps, gs)
    push!(tr_loss, lossval)
    push!(tr_epochs, Int(floor((iter - 1)  / nminibatches)) + 1)
    println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson")
end

OUTS_ = model.net((X[1][1:500_000], X[2][1:500_000]))
Y_ = Y[1:500_000]
pearson = my_cor(OUTS_, Y_)
fig = Figure();
ax1 = Axis(fig[1,1], title = "Gene expression reconstruction of FE model on TCGA data \n Pearson R = $(round(pearson,digits = 4))", ylabel = "Real Log TPM count", xlabel = "FE predicted Log TPM count", xticks = collect(0:5), yticks = collect(0:5));
hexbin!(ax1, cpu(OUTS_),cpu(Y_), cellsize = 0.05, colorscale = log10)
#Colorbar(fig[1,2])
lines!(ax1, [0,5],[0,5], linestyle=:dash, color =:black)
ax1.aspect = 1
#resize_to_layout!(fig)
fig
CairoMakie.save("figures/FE_reconstruction.pdf", fig)
CairoMakie.save("figures/FE_reconstruction.png", fig)
TCGA_datas

cpu(model.embed_1.weight)
umap_model = UMAP_(TCGA_data', 2;min_dist = 0.99, n_neighbors = 200);

## plotting embed using UMAP 
fig = Figure(size = (512,512));
ax2 = Axis(fig[1,1], xlabel = "UMAP-1", ylabel="UMAP-2", aspect = 1);
for group_lab in unique(labs)
    group = labs .== group_lab
    plot!(umap_model.embedding[1,group],umap_model.embedding[2,group], label = group_lab)
end 
#axislegend(ax2)
fig

## plotting embed directly 
# trained_patient_FE = cpu(model.embed_1.weight) 
trained_patient_FE = cpu(model.net[1][1].weight)
# fig = Figure(size = (1024,512));
ax2 = Axis(fig[1,3],title="trained 2-d embedding", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
markers = [:xcross, :hexagon, :utriangle, :cross]
for (i, group_lab) in enumerate(unique(labs))
    group = labs .== group_lab
    plot!(ax2, trained_patient_FE[1,group], trained_patient_FE[2,group], marker = markers[i%4 + 1], label = group_lab)
end 
fig[1,4] = axislegend(ax2)
fig
CairoMakie.save("figures/trained_2D_factorized_embedding.pdf", fig)
CairoMakie.save("figures/trained_2D_factorized_embedding.png", fig)

trained_patient_FE[1,1]
patient_FE[1,1]
