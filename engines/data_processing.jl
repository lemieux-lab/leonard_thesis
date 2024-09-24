function load_tcga_mega_dataset(infilename)
    inf = h5open(infilename, "r")
    data = inf["data"][:,:]
    genes = inf["genes"][:]
    samples = inf["samples"][:]
    labels = inf["labels"][:]
    close(inf)
    return data, labels, samples, genes
end 


function gather_params(basedir=".")
    df = DataFrame()
    for (root, dirs, files) in walkdir(basedir)
        for file in files
            if occursin("params.bson",file)
                # println("Loading $root/$file")
                d = BSON.load("$root/$file")
                push!(df, d, cols=:union)
            end
        end
    end
    return df
end


function split_train_test(X::Matrix; nfolds = 5)
    folds = Array{Dict, 1}(undef, nfolds)
    nsamples = size(X)[1]
    fold_size  = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples)
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds 
        tst_ids = shuffled_ids[collect((i-1) * fold_size +1: min(nsamples, i * fold_size))]
        tr_ids = setdiff(ids, tst_ids)
        train_x = X[tr_ids,:]
        # train_y = targets[tr_ids, :]
        test_x = X[tst_ids, :]
        # test_y = targets[tst_ids, :]
        folds[i] = Dict("foldn" => i, "train_x"=> train_x, "train_ids"=>tr_ids,"test_x"=> test_x, "test_ids" =>tst_ids)
    end
    return folds  
end

function load_tcga_dataset(infilename)
    infile = h5open(infilename)
    TCGA_data = infile["data"][:,:]
    labs = string.(infile["labels"][:])
    patients = string.(infile["samples"][:])
    genes = string.(infile["genes"][:]) 
    biotypes = string.(infile["biotypes"][:])
    close(infile)
    return TCGA_data, labs, patients, genes, biotypes
end 