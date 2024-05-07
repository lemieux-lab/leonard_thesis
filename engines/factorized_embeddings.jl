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


function prep_FE(data::Matrix,patients::Array,genes::Array, device=gpu)
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
    # return (device(patient_index), device(gene_index)), device(vec(values))

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
    b, c = params["fe_hl1_size"], params["fe_hl2_size"]#, params["fe_hl3_size"] ,params["fe_hl4_size"] ,params["fe_hl5_size"] 
    emb_layer_1 = gpu(Flux.Embedding(params["nsamples"], emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(params["ngenes"], emb_size_2))
    hl1 = gpu(Flux.Dense(a, b, relu))
    hl2 = gpu(Flux.Dense(b, c, relu))
    # hl3 = gpu(Flux.Dense(c, d, relu))
    # hl4 = gpu(Flux.Dense(d, e, relu))
    # hl5 = gpu(Flux.Dense(e, f, relu))
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