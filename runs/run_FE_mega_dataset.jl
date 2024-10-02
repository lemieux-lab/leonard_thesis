include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/gpu_utils.jl")
# device!()
outpath, session_id = set_dirs("FE_RES")

# display dataset sizes
using DataFrames
infile = readtable()
filesizes = CSV.read("tables/TCGA_bootstrap_sizes.txt", DataFrame, delim=' ', ignorerepeated=true)
bsize = [parse(Int, s[1:end-1]) for s in filesizes[:,5]]
nsamples = [parse(Int, split(split(s,"_")[end],".")[1][1:end-1]) for s in filesizes[:,9]]
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], xlabel = "number of samples", ylabel = "size (GB)", title = "Size in bytes of TCGA bootstraped datasets");
lines!(ax, sort(nsamples * 10346), sort(bsize), linestyle = :dash)
scatter!(ax, nsamples * 10346, bsize, markersize = 20)
CairoMakie.save("figures/TCGA_bootstrap_sizes.png",fig)
CairoMakie.save("figures/TCGA_bootstrap_sizes.pdf",fig)

@time dat, lab, smpl, gns = load_tcga_mega_dataset("Data/TCGA_bootstrap/TCGA_poisson_raw_counts_10x.h5");
shuffled_ids = shuffle(collect(1:size(smpl)[1]))
@time X_data = dat[shuffled_ids,:] # pretty long!
samples = smpl[shuffled_ids]
labs = lab[shuffled_ids]
# for 100Kx20K data
# @time X_data = log10.(X_data .+ 1) # < 1 min
X_data = log10.(dat .+ 1) 
samples = smpl
labs = lab 
# @time folds = split_train_test(X_data, nfolds = 5) # 1m30s 
# set params 
modelid = "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
outdir = "megadataset_FE_models"
# foldn = 1
# train_ids, train_data, test_ids, test_data = folds[foldn]["train_ids"], folds[foldn]["train_x"], folds[foldn]["test_ids"], folds[foldn]["test_x"]

generate_params(X_data;emb_size_2 =75, emb_size_1 = 2, nsamples_batchsize=4) = return Dict( 
        ## run infos 
        "session_id" => session_id,  "modelid" =>  modelid,
        "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
        "printstep"=>10_000,# "foldn"=>foldn, 
        "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt",
        ## data infos 
        "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
        "nsamples_batchsize"=> nsamples_batchsize, 
        ## optim infos 
        "lr" => 1e-2, "l2" => 1e-10,"nsteps" => 5_000_000, "nsteps_inference" => 10_000, "batchsize" => nsamples_batchsize * size(X_data)[2],
        ## model infos
        "emb_size_1" => emb_size_1, "emb_size_2" => emb_size_2, "fe_layers_size"=> [100,50,50]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
        )
# train whole dataset 
# params = generate_params(X_data)
# trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# train with training set
params_dict = generate_params(X_data, emb_size_1=2, emb_size_2=100, nsamples_batchsize = 10)

## init model
model = FE_model(params_dict);
start_timer = now()
smpl_batchsize = params_dict["nsamples_batchsize"]
nminibatches = Int(floor(params_dict["nsamples"] / smpl_batchsize))
tr_loss, tr_epochs, tr_cor, tr_elapsed = [], [], [], []
# opt = Flux.setup(OptimiserChain(Flux.WeightDecay(params_dict["l2"]), Flux.Optimise.Adam(params_dict["lr"])), model);    
opt = Flux.setup(Flux.Optimise.Adam(params_dict["lr"]), model);    
println("1 epoch 1 - 1 /$nminibatches - TRAIN \t ELAPSED: $((now() - start_timer).value / 1000 )")         
for iter in 1:params_dict["nsteps"]
    # Stochastic gradient descent with minibatches
    cursor = (iter -1)  % nminibatches + 1
    smpl_range = (cursor -1) * smpl_batchsize + 1,cursor * smpl_batchsize

    X_ = gen_minibatch_X(params_dict["ngenes"], smpl_range)
    Y_ = gpu(vec(X_data[smpl_range[1]:smpl_range[2],:]))

    OUTS = model(X_)
    grads = Flux.gradient(model) do m 
        loss = Flux.mse(m(X_), Y_) 
    end 
    lossval = Flux.mse(model(X_), Y_) 
    pearson = my_cor(model(X_), Y_)
    Flux.update!(opt, model, grads[1])
    push!(tr_cor, pearson)
    push!(tr_loss, lossval)
    push!(tr_epochs, Int(ceil(iter / nminibatches)))
    push!(tr_elapsed, (now() - start_timer).value / 1000 )
    (iter % 100 == 0) | (iter == 1) ? println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ELAPSED: $((now() - start_timer).value / 1000 )") : nothing        
        
    if (iter % params_dict["printstep"] == 0) 
        CSV.write("$(params_dict["outpath"])/$(params_dict["modelid"])_loss_computing_times", DataFrame(:tr_epochs=>tr_epochs, :tr_loss=>tr_loss, :tr_elapsed=>tr_elapsed))
        # # save model 
        bson("$(params_dict["outpath"])/$(params_dict["modelid"])_in_training_model.bson", Dict("model"=> cpu(model)))
        trained_patient_FE = cpu(model[1][1].weight)
        patient_embed_fig = plot_patient_embedding(trained_patient_FE, labs, "trained 2-d embedding\n$(params_dict["modelid"]) \n- step $(iter)", params_dict["colorsFile"]) 
        #patient_embed_fig = plot_patient_embedding(trained_patient_FE, labs, "trained 2-d embedding\n$(params["modelid"]) \n- step $(iter)", params["colorsFile"]) 
        CairoMakie.save("$(params_dict["outpath"])/$(params_dict["modelid"])_2D_embedding_$(iter).png", patient_embed_fig)
    end 
end