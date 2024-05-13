readdir(".")
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
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

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
