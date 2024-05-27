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
    "emb_size_1" => 2, "emb_size_2" => 50, "fe_layers_size"=> [250, 75, 50, 25, 10]#, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    )
# train whole dataset 
# params = generate_params(X_data)
# trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(X_data, patients, genes[CDS], params, labels)
# continue training on embedding layer only.
# tr_epochs , tr_loss, tr_cor =  train_embed_SGD!(params, X,Y, trained_FE)

# train with training set
params = generate_params(train_data)
# save IDs
bson("$(params["outpath"])/$(params["modelid"])_train_test_ids.bson", 
    Dict("train_ids"=> train_ids, "test_ids"=>test_ids, 
    "model_prefix"=> "$(params["outpath"])/$(params["modelid"])"))
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(train_data, patients[train_ids], genes[CDS], params, labels[train_ids])

## generate X and Y test data. 

function prep_FE_inference_2(infer_sample::Array,patients::Array,genes::Array, device=gpu)
    n = length(patients)
    m = length(genes)
    # k = length(tissues)
    values = Array{Float32,1}(undef, max(n * m, 1))
    patient_index = Array{Int64,1}(undef, max(n * m, 1))
    gene_index = Array{Int64,1}(undef, max(n * m, 1))
    # tissue_index = Array{Int32,1}(undef, n * m)
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[index] = infer_sample[j] # to debug 
            patient_index[index] = i # Int
            gene_index[index] = j # Int 
            # tissue_index[index] = tissues[i] # Int 
        end
    end 
    return (device(patient_index), device(gene_index)), device(vec(values))
    # return (device(patient_index), device(gene_index)), device(vec(values))

end
infer_patient_id = 3
X_infer, Y_infer = prep_FE_inference_2(vec(test_data[2,:]), patients[train_ids], genes[CDS]) 
## Compute : EMBED1, EMBED2, MSE, MSE+L2, PCORR 
infer_model = deepcopy(trained_FE.net)
# local_min = inference_2_0(infer_model, X_infer, Y_infer, train_ids, genes[CDS])
## Fine-tune patient inference embedding 5000 steps
batchsize = params["batchsize"]
nminibatches = Int(floor(length(Y_infer) / batchsize))
opt = Flux.ADAM(params["lr"])
for iter in 1:params["nsteps_inference"]
    cursor = (iter -1)  % nminibatches + 1
    if cursor == 1 
        shuffled_ids = shuffle(collect(1:length(Y_infer))) # very inefficient
    end 
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_infer)))
    ids = shuffled_ids[mb_ids]
    X_, Y_ = (X_infer[1][ids],X_infer[2][ids]), Y_infer[ids]
    ps = Flux.params(infer_model[1][1]) # patient embed only
    gs = gradient(ps) do 
        Flux.mse(infer_model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
    end
    lossval = Flux.mse(infer_model(X_), Y_) + params["l2"] * sum(p -> sum(abs2, p), ps)
    pearson = my_cor(infer_model(X_), Y_)
    Flux.update!(opt,ps, gs)
    iter % 100 == 0 ?  println("$(iter) epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson ") : nothing
end 
## before : 0.7591, -0.12578, 0.03646 MSE 
local_min = inference_2_0(infer_model, X_infer, Y_infer, train_ids, genes[CDS])

## after 10000 steps : 0.768057 -0.128536 0.03636
# local_min = inference_2_0(infer_model, X_infer, Y_infer, train_ids, genes[CDS])

fig = Figure(size = (1024,1024));
ax = Axis(fig[1,1],title="$(params["modelid"]) FE-training", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
markers = [:diamond, :circle, :utriangle, :rect]
tissue_labels = labels[train_ids]
patient_FE = cpu(trained_FE.net[1][1].weight)
for (i, group_lab) in enumerate(unique(tissue_labels))
    group = tissue_labels .== group_lab
    scatter!(ax, patient_FE[1,group],patient_FE[2,group], strokewidth = 0.1, color = RGBf(rand(), rand(), rand()), marker = markers[i%4 + 1], label = group_lab)
end 
# fig[1,2] = axislegend(ax, position =:rc, labelsize = 8, rowgap=0)
group_lab = labels[test_ids[infer_patient_id]]
patient_id = patients[test_ids[infer_patient_id]]
ax2 = Axis(fig[1,2],title="$(params["modelid"]) FE-training - \n $(patient_id) \n - $(group_lab) vs Rest", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
scatter!(ax2, patient_FE[1,:],patient_FE[2,:], color = :grey, label = "others")
scatter!(ax2, patient_FE[1,tissue_labels .== group_lab],patient_FE[2,tissue_labels .== group_lab], color = :red, label = group_lab)
fig  

sample_profile = test_data[infer_patient_id,:]
gen_grid(nb_p, span) = collect(span[1]:(span[2] - span[1]) / nb_p:span[2]) 
grid = gen_grid(100,[-3,3])
grid_vals = zeros(size(grid)[1] , size(grid)[1])
gene_embed = trained_FE.net[1][2](gpu(collect(1:size(genes[CDS])[1])))
Xs = []
Ys = []
for i in 1:size(grid)[1]
    println(i)
    for j in 1:size(grid)[1]
        grid_val = vcat(gpu(ones(size(genes[CDS])[1]) * grid[i])', gpu(ones(size(genes[CDS])[1]) * grid[j])', gene_embed)
        
        infer_profile = vec(trained_FE.net[2:end](grid_val))
        grid_vals[i,j] =  cpu(Flux.mse(gpu(sample_profile), infer_profile))
        push!(Xs, grid[i])
        push!(Ys, grid[j])
    end 
end 
ax3 = Axis(fig[2,1], title = "Contour patient $(patient_id) \n", xlabel = "EMBED-1", ylabel = "EMBED-2" )
# grid_vals
co = contourf!(ax3, grid, grid, grid_vals, colormap = Reverse(:viridis), levels = 100)
#Colorbar(fig[2,1], co)
fig
infer_sample_min = local_min[local_min.MSE .== minimum(local_min.MSE),:]
scatter!(ax3, infer_sample_min.EMBED1, infer_sample_min.EMBED2, color = :black)
fig
ax4 = Axis(fig[2,2],  title = "Contour patient $(patient_id) \n inference space", xlabel = "EMBED-1", ylabel = "EMBED-2" )
co = contourf!(ax4, grid, grid, grid_vals, levels = 20)
scatter!(ax4, local_min.EMBED1, local_min.EMBED2, color=local_min.MSE, colormap=Reverse(:viridis), colorscale = log10)
fig
## Compute : EMBED1, EMBED2, MSE, MSE+L2, PCORR 
## Find minimum. Return minimal patient embed, inference embedding.
## Trace A) Training Embedding B) inference embedding C) local minimizer for new patient

## Reproduce Assya's results on TCGA classification. (Only on trained FE)
## Test for 2D, 3D, 25D, 50D, 125D, Full CDS profile. 

## PCA + UMAP

sample_profile = test_data[39,:]
gen_grid(nb_p, span) = collect(span[1]:(span[2] - span[1]) / nb_p:span[2]) 
grid = gen_grid(100,[-3,3])
grid_vals = zeros(size(grid)[1] , size(grid)[1])
gene_embed = trained_FE.net[1][2](gpu(collect(1:size(genes[CDS])[1])))
Xs = []
Ys = []
for i in 1:size(grid)[1]
    println(i)
    for j in 1:size(grid)[1]
        grid_val = vcat(gpu(ones(size(genes[CDS])[1]) * grid[i])', gpu(ones(size(genes[CDS])[1]) * grid[j])', gene_embed)
        
        infer_profile = vec(trained_FE.net[2:end](grid_val))
        grid_vals[i,j] =  cpu(my_cor(gpu(sample_profile), infer_profile))
        push!(Xs, grid[i])
        push!(Ys, grid[j])
    end 
end 
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], title = "Contour", xlabel = "EMBED-1", ylabel = "EMBED-2" )
# grid_vals
co = contourf!(ax, grid, grid, grid_vals', levels = 20)
Colorbar(fig[1,2], co)
fig