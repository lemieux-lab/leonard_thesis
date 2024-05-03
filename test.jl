include("engines/init.jl")

### TEST FE 
struct MLSurvDataset
    data::Matrix
    samples::Array 
    genes::Array
    biotypes::Array
    labels::Array
    survt::Array
    surve::Array
end 
function MLSurvDataset(infilename)
    infile = h5open(infilename, "r")
    data = infile["data"][:,:] 
    samples = infile["samples"][:]  
    labels = infile["labels"][:]  
    genes = infile["genes"][:]  
    biotypes = infile["biotypes"][:]
    survt = infile["survt"][:]  
    surve = infile["surve"][:]  
    close(infile)
    return MLSurvDataset(data, samples, genes, biotypes, labels, survt, surve)
end 
function load_tcga_datasets(infiles)
    
    out_dict = Dict()
    for infile in infiles
        DataSet = MLSurvDataset(infile)
        name_tag = split(infile,"_")[3]
        out_dict["$(name_tag)"] = Dict("dataset"=>DataSet,
        "name" => name_tag
        ) 
    end 
    return out_dict
end 


function prep_FE(data::Matrix,patients::Array,genes::Array,tissues::Array, device=gpu)
    n = length(patients)
    m = length(genes)
    # k = length(tissues)
    values = Array{Float32,2}(undef, (1, n * m))
    patient_index = Array{Int64,1}(undef, max(n * m, 1))
    gene_index = Array{Int64,1}(undef, max(n * m, 1))
    # tissue_index = Array{Int32,1}(undef, n * m)
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1,index] = data[i,j]
            patient_index[index] = i # Int
            gene_index[index] = j # Int 
            # tissue_index[index] = tissues[i] # Int 
        end
    end 
    shfl = shuffle(collect(1:length(values)))
    return (device(patient_index[shfl]), device(gene_index)[shfl]), device(vec(values[shfl]))
end 


struct FE_model
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
    opt
    lossf
end

function l2_penalty(model::FE_model)
    return sum(abs2, model.embed_1.weight) + sum(abs2, model.embed_2.weight) + sum(abs2, model.hl1.weight) + sum(abs2, model.hl2.weight)
end

function mse_l2(model::FE_model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.net(X), Y) + l2_penalty(model) * weight_decay
end 


function FE_model(params::Dict)
    emb_size_1 = params["emb_size_1"]
    emb_size_2 = params["emb_size_2"]
    a = emb_size_1 + emb_size_2 
    b, c, d, e, f = params["fe_hl1_size"], params["fe_hl2_size"], params["fe_hl3_size"] ,params["fe_hl4_size"] ,params["fe_hl5_size"] 
    emb_layer_1 = gpu(Flux.Embedding(params["nsamples"], emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(params["ngenes"], emb_size_2))
    hl1 = gpu(Flux.Dense(a, b, relu))
    hl2 = gpu(Flux.Dense(b, c, relu))
    hl3 = gpu(Flux.Dense(c, d, relu))
    hl4 = gpu(Flux.Dense(d, e, relu))
    hl5 = gpu(Flux.Dense(e, f, relu))
    outpl = gpu(Flux.Dense(c, 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    opt = Flux.ADAM(params["lr"])
    lossf = mse_l2
    FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl, opt, lossf)
end 
function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 
infile = h5open("Data/GDC_processed/TCGA_TPM_lab.h5")
data = log10.(infile["data"][:,:] .+ 1)
labs = string.(infile["labels"][:])
patients = string.(infile["rows"][:])
genes = string.(infile["cols"][:]) 
close(infile)

BRCA = load_tcga_datasets(["Data/TCGA_datasets/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5"])["BRCA"]
CDS = BRCA["dataset"].biotypes .== "protein_coding"
TCGA_data = data[:,CDS]
projects_num = [findall(unique(labs) .== X)[1] for X in labs] 

X, Y = prep_FE(TCGA_data, patients, genes[CDS], projects_num);

batchsize = 500_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))

params = Dict( "nsteps" => 10_000,
    "emb_size_1" => 50,
    "emb_size_2" => 50,
    "fe_hl1_size" => 250,
    "fe_hl2_size" => 150,
    "fe_hl3_size" => 100,
    "fe_hl4_size" => 50,
    "fe_hl5_size" => 10,
    
    "nsamples" =>length(patients),
    "ngenes"=> sum(CDS), 
    "lr" => 1e-3,
    "wd" => 1e-5)
## 
model = FE_model(params);

tr_loss = []
tr_epochs = []
opt = Flux.ADAM(params["lr"])
nminibatches = Int(floor(length(Y) / batchsize))
# shuffled_ids = shuffle(collect(1:length(Y)))

for iter in 1:params["nsteps"]
    ps = Flux.params(model.net)
    cursor = (iter -1)  % nminibatches + 1
    # if cursor == 1 
    #     shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
    # end 
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
    # ids = shuffled_ids[mb_ids]
    X_, Y_ = (X[1][mb_ids],X[2][mb_ids]), Y[mb_ids]
    
    # dump_cb(model, params, iter + restart)
    
    gs = gradient(ps) do 
        mse_l2(model, X_, Y_, weight_decay = params["wd"])
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
    println("epoch $(Int(ceil(iter / nminibatches))) - $cursor /$nminibatches - TRAIN loss: $(lossval)\tpearson r: $pearson")
end

OUTS_ = model.net((X[1][1:500_000], X[2][1:500_000]))
Y_ = Y[1:500_000]
fig = Figure();
ax1 = Axis(fig[1,1], xticks = collect(0:5), yticks = collect(0:5));
hexbin!(ax1, cpu(OUTS_),cpu(Y_), cellsize = 0.05, colorscale = log10)
#Colorbar(fig[1,2])
lines!(ax1, [0,5],[0,5], linestyle=:dash, color =:black)
ax1.aspect = 1
#resize_to_layout!(fig)
fig
TCGA_datas

model.embed_1