include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/pca.jl")
outpath, session_id = set_dirs("CPHDNN_RES")
BRCA_data, labs, samples, genes, biotypes = load_tcga_dataset("Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5")
CDS = biotypes .== "protein_coding"
inf  = h5open("Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5", "r")
survt = inf["survt"][:];
surve = inf["surve"][:];
close(inf)
## CPHDNN 
# split 
folds = split_train_test(BRCA_data[:,CDS], nfolds = 5) # split 80-20
test_c_ind_by_fold, tst_scores_by_fold, tst_t_by_fold, tst_e_by_fold = [], [], [], []
for fold in folds
    train_ids, train_data, test_ids, test_data = fold["train_ids"], fold["train_x"], fold["test_ids"], fold["test_x"]
    train_t = survt[train_ids]
    train_e = surve[train_ids]
    test_t = survt[test_ids]
    test_e = surve[test_ids]
    # train 
    # each printstep steps dump params withtrain loss, test_loss. train c ind, test c ind. 
    cphdnn_params_dict(X_data) = Dict(
        "insize" => size(X_data)[2],
        "cph_hl_size"=> 64,
        "lr"=>1e-6, "l2"=>1e-3, "nsteps"=>2000,
        "printstep"=>100,
    )
    # set hyperparams 
    cphdnn_params = cphdnn_params_dict(train_data)
    # format data 
    x_train, y_t_train, y_e_train, NE_frac_tr = format_surv_data(train_data, train_t, train_e)
    x_test, y_t_test, y_e_test, NE_frac_tst = format_surv_data(test_data, test_t, test_e)

    # init model 
    cphdnn = Chain(Dense(cphdnn_params["insize"],cphdnn_params["cph_hl_size"], leakyrelu), 
            Dense(cphdnn_params["cph_hl_size"], cphdnn_params["cph_hl_size"], leakyrelu), 
            Dense(cphdnn_params["cph_hl_size"], 1, sigmoid,  bias = false)) |> gpu
    # train loop 
    tst_c_ind, tst_scores = train_cphdnn!(cphdnn_params, cphdnn, x_train, y_e_train, NE_frac_tr, x_test, y_e_test, NE_frac_tst)
    push!(test_c_ind_by_fold, tst_c_ind)
    push!(tst_scores_by_fold, cpu(vec(tst_scores)))
    push!(tst_t_by_fold, cpu(vec(y_t_test)))
    push!(tst_e_by_fold, cpu(vec(y_e_test)))
end 
test_c_ind_by_fold
vcat(tst_t_by_fold...)
vcat(tst_e_by_fold...)
T, E, S = vcat(tst_t_by_fold...), vcat(tst_e_by_fold...), -1 * vcat(tst_scores_by_fold...)
T_g, E_g, S_g = gpu(T), gpu(E), gpu(S)
@time tst_c,_,_,_ = concordance_index(T, E, S)
@time tst_c,_,_,_ = concordance_index(T_g, E_g, S_g)
function bootstrap(fn, T, E, S; n=10_000)
    size = length(T)
    cs = zeros(n)
    for i in 1:n 
        sampling
        cs[i] = fn(T[sampling], E[sampling], S[sampling])
    end 
end 
# infer
# do inference. dump risk scores with surv.

# compute final performance. Merge splits inference scores. bootstrap x 10000, plot distribution. (KM curve?)


# X_data = TCGA_data[:,CDS]
embs = 2

# 
# Train

# set params 
generate_params(X_data, emb_size) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>10_000, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-8,"nsteps" => 100_000, "nsteps_inference" => 10_000, "nsamples_batchsize" => 4,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => 100, "fe_layers_size"=> [250, 100], #, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    ## plotting infos 
    "colorsFile"=> "Data/GDC_processed/BRCA_colors_def.txt"
    )
# train with training set
params_dict = generate_params(train_data, embs)
# save IDs
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, samples[train_ids], genes[CDS], params_dict, labs[train_ids])
# infer FE
inference_model, part1_fig, part2_fig = do_inference_B(trained_FE, train_data, train_ids, test_data, test_ids, samples, genes[CDS], params_dict)

# feed to COX-DNN
# infer 
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
TCGA_datasets = load_tcga_datasets(tcga_datasets_list);
BRCA_data = TCGA_datasets["BRCA"]
LGG_data = TCGA_datasets["LGG"]
OV_data = TCGA_datasets["OV"]