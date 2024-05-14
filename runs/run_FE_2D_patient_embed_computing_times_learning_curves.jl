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

X, Y = prep_FE(X_data, patients, genes);

# split train / test 
# save IDs, save model 
# train 
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X,Y, params, labels)
# save model 
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# inference 
# plot final 2D patient embed (train + test)
