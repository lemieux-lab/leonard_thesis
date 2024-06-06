include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
outpath, session_id = set_dirs("FE_RES")
BRCA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5")
CDS = biotypes .== "protein_coding"
# X_data = TCGA_data[:,CDS]
X_data = BRCA_data[:,CDS]
# split train / test 
folds = split_train_test(X_data, nfolds = 5) # split 80-20
train_ids, train_data, test_ids, test_data = folds[1]["train_ids"], folds[1]["train_x"], folds[1]["test_ids"], folds[1]["test_x"]
# set params 
generate_params(X_data) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>500, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 1e-2, "l2" => 1e-7,"nsteps" => 20000, "nsteps_inference" => 10_000, "batchsize" => 40_000,
    ## model infos
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [25, 10], #, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    ## plotting infos 
    "colorsFile"=> "Data/GDC_processed/BRCA_colors_def.txt"
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
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labs)
patient_FE = cpu(trained_FE.net[1][1].weight)
DF = DataFrame(Dict([("EMBED$(i)", patient_FE[i,:]) for i in 1:size(patient_FE)[1]]))
CSV.write("$(params["outpath"])/$(params["modelid"])_patient_embed.csv", DF)

function test_classification_perf(X_data, labels;nsteps=2000)
    dimFE = size(X_data)[1] 
    npatients = size(X_data)[2]
    outsize = length(unique(labels))
    Y = zeros( npatients, outsize)
    unique_labs = unique(labels)
    [Y[i, unique_labs .== lab] .= 1 for (i,lab) in enumerate(labels)]
    Y_data_clf = gpu(Matrix(Y'))
    folds = split_train_test(Matrix(X_data'), nfolds = 5) # split 80-20
    outfile = "modelid\tfoldn\tacc\t"
    for (foldn, fold) in enumerate(folds)
    train_ids, train_x, test_ids, test_x = fold["train_ids"], gpu(Matrix(fold["train_x"])'), fold["test_ids"], gpu(Matrix(fold["test_x"]'))
    train_y = gpu(Matrix(Y')[:,train_ids])
    test_y = gpu(Matrix(Y')[:,test_ids])
    model = gpu(Flux.Chain(Dense(size(train_x)[1], 250, leakyrelu), Dense(250, 100, leakyrelu),
        Dense(100, 50, leakyrelu), Dense(50, outsize), softmax
        ))
    opt = Flux.setup(OptimiserChain(Flux.WeightDecay(1e-6), Flux.Optimise.Adam(0.0001)), model);
    for i in 1:nsteps
        grads = Flux.gradient(model) do m
            loss = sum(Flux.crossentropy(m(train_x), train_y))
        end 
        outs = model(train_x)
        tr_lossval = sum(Flux.crossentropy(outs, train_y))
        tr_acc = sum((maximum(outs, dims = 1) .== outs)' .& (Int.(train_y))') / size(train_x)[2]
        
        tst_outs = model(test_x)
        tst_lossval = Flux.mse(tst_outs, test_y)
        tst_acc = sum((maximum(tst_outs, dims = 1) .== tst_outs)' .& (Int.(test_y))') / size(test_x)[2]
        
        Flux.update!(opt, model, grads[1])
        if i % 100 == 0
            println("FOLDN $foldn - STEP $i - TRAIN : loss : $(tr_lossval) acc: $(tr_acc)\n TEST: loss: $(tst_lossval) acc: $(tst_acc)")
        end
    end
    tst_outs = model(test_x)
    ACC =  sum((maximum(tst_outs, dims = 1) .== tst_outs)' .& (Int.(test_y))') / size(test_x)[2]
    
    outfile = "$outfile\n$(params["modelid"])\t$foldn\t$ACC\t"
    end 
    return outfile
end 
FE_outfile = test_classification_perf(patient_FE, labels)
println(outfile)
size(patient_FE)
size(X_data)

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
