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
X_data = TCGA_data
# split train / test 
folds = split_train_test(X_data, nfolds = 5) # split 80-20
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
params = Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> length(genes), 
    "nsamples_train" => size(train_data)[1], "nsamples_test" => size(test_data)[1], 
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 400_000, "nsteps_inference" => 10_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [250,75,50,25,10]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# save IDs
bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
    Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
    "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))
# splits_d = BSON.load("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson")

# train 
X_train, Y_train = prep_FE(train_data, patients[train_ids], genes);
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_train,Y_train, params, labels[train_ids])
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# prep test data points 
X_test, Y_test = prep_FE(test_data, patients[test_ids], genes)
# inference 
test_model = do_inference(trained_FE, params, X_test, Y_test)
# plot final 2D patient embed (train + test)
train_test_2d_fig = plot_train_test_patient_embed(trained_FE, test_model, labels, train_ids, test_ids)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.png", train_test_2d_fig)
CairoMakie.save("$(params["outpath"])/$(params["modelid"])_2D_embedding_train_test.pdf", train_test_2d_fig)


## PlotlyJS
colors_dict = Dict([(lab, RGBf(rand(), rand(), rand())) for lab in unique(labels)])    
traces = [PlotlyJS.scatter(x=train_embed[1,tissue_labels[train_ids] .== group_lab], y=train_embed[2,tissue_labels[train_ids] .== group_lab], marker = attr(color= colors_dict[group_lab]), mode = "markers", name = group_lab) for group_lab in unique(labels)]
[push!(traces, PlotlyJS.scatter(x=test_embed[1,tissue_labels[test_ids] .== group_lab], y=test_embed[2,tissue_labels[test_ids] .== group_lab], marker = attr(symbol="diamond", line_width=1,color= colors_dict[group_lab]), mode = "markers", name = "TEST - $(group_lab)")) for group_lab in unique(labels)]
P = PlotlyJS.plot(traces, 
    Layout(title = "Patient Factorized Embedding 2D with train and test samples\nmodel ID: $(params["modelid"])",
    yaxis = attr(showgrid=true, gridwidth=1, gridcolor="black", zeroline=true, zerolinewidth=1, zerolinecolor="black"),
    xaxis = attr(showgrid=true, gridwidth=1, gridcolor="black", zeroline=true, zerolinewidth=1, zerolinecolor="black"),
    plot_bgcolor = :white, 
))
PlotlyJS.savefig(P, "interactive_figures/$(params["modelid"])_FE_2D_visualisation_train_test.html")
