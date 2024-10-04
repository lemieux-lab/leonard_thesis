include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/pca.jl")
include("engines/coxphdnn.jl")
include("engines/gpu_utils.jl")
# CUDA.device!()
outpath, session_id = set_dirs("CPHDNN_RES")
BRCA_data, labs, samples, genes, biotypes = load_tcga_dataset("Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5")
CDS = biotypes .== "protein_coding"
inf  = h5open("Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5", "r")
survt = inf["survt"][:];
surve = inf["surve"][:];
close(inf)
#FIXED VARIABLES
dim_redux_size = 64
input_type = "FE"
nsteps_dim_redux = 120
printstep_FE = 10_000 
nsteps_CPHDNN = 100 
printstep_CPHDNN = 10_000 

generate_params(X_data, emb_size) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>printstep_FE, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-3, "l2" => 1e-8,"nsteps" => nsteps_dim_redux, "nsteps_inference" => Int(floor(nsteps_dim_redux * 0.1)), "nsamples_batchsize" => 4,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => 100, "fe_layers_size"=> [250, 100], #, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    ## plotting infos 
    "colorsFile"=> "Data/GDC_processed/BRCA_colors_def.txt"
    )

cphdnn_params_dict(X_data) = Dict(
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "input_type"=>input_type, "model_type" => "CPHDNN",
    "insize" => size(X_data)[2],
    "cph_hl_size"=> 64,
    "lr"=>1e-6, "l2"=>1e-3, "nsteps"=>nsteps_CPHDNN,
    "printstep"=> printstep_CPHDNN,
)

# split data 
folds = split_train_test(BRCA_data[:,CDS], nfolds = 5) # split 80-20
# init performance loggers
test_c_ind_by_fold, tst_scores_by_fold, tst_t_by_fold, tst_e_by_fold = [], [], [], []
# loop
for fold in folds
    # format data 
    train_ids, train_data, test_ids, test_data = fold["train_ids"], fold["train_x"], fold["test_ids"], fold["test_x"]
    train_t, train_e, test_t, test_e  = survt[train_ids],  surve[train_ids], survt[test_ids], surve[test_ids]
    # train FE 
    FE_params_dict = generate_params(train_data, dim_redux_size)
    trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, samples[train_ids], genes[CDS], FE_params_dict, labs[train_ids])
    # infer FE
    inference_model, part1_fig, part2_fig = do_inference_B(trained_FE, train_data, train_ids, test_data, test_ids, samples, genes[CDS], FE_params_dict)
    #### CPHDNN eval 
    tst_c_ind, tst_scores, y_t_test, y_e_test = CPHDNN_eval(trained_FE[1][1].weight', train_t, train_e, inference_model[1][1].weight', test_t, test_e)
    ### dump fold c-index     
    FE_params_dict["c_ind_test"] = tst_c_ind
    bson("$(FE_params_dict["outpath"])/$(FE_params_dict["modelid"])_params.bson", FE_params_dict)
    
    push!(test_c_ind_by_fold, tst_c_ind)
    push!(tst_scores_by_fold, cpu(vec(tst_scores)))
    push!(tst_t_by_fold, cpu(vec(y_t_test)))
    push!(tst_e_by_fold, cpu(vec(y_e_test)))
end 
T, E, S = vcat(tst_t_by_fold...), vcat(tst_e_by_fold...), -1 * vcat(tst_scores_by_fold...);
dump_surv_scores_c_index_bootstrap_h5(T, E, S, input_type, dim_redux_size, nsteps_dim_redux)

