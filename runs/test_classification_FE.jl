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
X_data = TCGA_data[:,CDS]
# split train / test 
folds = split_train_test(X_data, nfolds = 5) # split 80-20
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
generate_params(X_data) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>10_000, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 100_000, "nsteps_inference" => 10_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 125, "emb_size_2" => 50, "fe_layers_size"=> [250, 75, 50, 25, 10]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# train whole dataset 
# params = generate_params(X_data)
# trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# train with training set
params = generate_params(X_data)
# save IDs
# bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
#     Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
#     "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
patient_FE = cpu(trained_FE.net[1][1].weight)
DF = DataFrame(Dict([("EMBED$(i)", patient_FE[i,:]) for i in 1:size(patient_FE)[1]]))
CSV.write("$(params["outpath"])/$(params["modelid"])_patient_embed.csv", DF)

model_params = gather_params("figures/tables/")
model_params = model_params[model_params[:,"emb_size_2"] .!= 2,:]
model_params = model_params[model_params[:,"l2"] .!= 1e-4,:]

ACCs_table = []
hyperparams = []
for (row, emb_size_2) in enumerate(sort(unique(model_params[:,"emb_size_2"])))
    for (col, l2_val) in enumerate(sort(unique(model_params[:, "l2"])))
        group_data = (model_params[:,"emb_size_2"] .== emb_size_2) .& (model_params[:,"l2"] .== l2_val)
        if sum(group_data) != 0
            fname = model_params[group_data, "modelid"][1]
                
            data = CSV.read("figures/tables/$(fname)_trained_2D_factorized_embedding.csv", DataFrame)
            patient_FE = Matrix(data)'
            ACCs = test_classification_perf(patient_FE, labels)
            print((l2_val,emb_size_2), "\t$(mean(ACCs))")
            push!(hyperparams, (l2_val,emb_size_2))
            push!(ACCs_table, ACCs)
        end 
    end 
end 

df = DataFrame(Matrix(reshape(mean.(ACCs_table), (5,6))'), :auto)
names(df) .= string.(sort(unique(model_params[:,"l2"])))
df[:,"emb_size_2"] .= string.(sort(unique(model_params[:,"emb_size_2"])))
CSV.write("figures/tables/hypersearch_classif_acc_TCGA.csv", df)
full_profile_outfile = test_classification_perf(Matrix(X_data'), labels)
println(replace(full_profile_outfile, "\t" => ","))
println(replace(outfile, "\t" => ","))

input1 = CSV.read("figures/tables/full_profile_classification.txt", DataFrame)
input2 = CSV.read("figures/tables/FE_125_classification.txt", DataFrame)
fig = Figure(size=(512,512));
ax = Axis(fig[1,1], title ="Classification accuracy on TCGA",ylabel="Accuracy on test set", xticks=(collect(1:2), ["Full CDS profile", "FE Model (125D)"]));
boxplot!(ax, ones(size(input1)[1]), input1.acc,label = "Full profile", show_outliers = false)
scatter!(ax, rand(size(input1)[1]) / 5 .+ 0.9, input1.acc,markersize = 20, color = :white, strokewidth=2)
boxplot!(ax, ones(size(input2)[1]) * 2, input2.acc,label = "FE (125D)", show_outliers = false)
scatter!(ax, rand(size(input2)[1]) / 5 .+ 1.9, input2.acc,markersize = 20, color = :white, strokewidth=2)
axislegend(ax)
fig
