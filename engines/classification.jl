function format_Y(labs)
    npatients = size(labs)[1]
    outsize = length(unique(labs))
    Y = zeros( npatients, outsize)
    unique_labs = unique(labs)
    [Y[i, unique_labs .== lab] .= 1 for (i,lab) in enumerate(labs)]
    return Y
end 

function train_DNN(train_x, train_y, test_x, test_y;nsteps=2000)
    outsize = size(train_y)[1]
    model = gpu(Flux.Chain(Dense(size(train_x)[1], 100, leakyrelu), Dense(100, 50, leakyrelu),
        Dense(50, outsize), softmax
        ))
    opt = Flux.setup(OptimiserChain(Flux.WeightDecay(1e-8), Flux.Optimise.Adam(0.01)), model);
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
        if i % 1000 == 0
            println("STEP $i - TRAIN : loss : $(tr_lossval) acc: $(tr_acc)\n TEST: loss: $(tst_lossval) acc: $(tst_acc)")
        end
    end
    return model
end 
   
function test_performance_FE(trained_FE, inference_model, labs, train_ids, test_ids)
    train_X_FE = deepcopy(trained_FE.net[1][1].weight)
    num_labs = format_Y(labs)
    train_Y = gpu(Matrix(num_labs[train_ids,:]'))
    test_X_FE = deepcopy(inference_model[1][1].weight)
    test_Y = gpu(Matrix(num_labs[test_ids,:]'))
    model = train_DNN(train_X_FE, train_Y, test_X_FE, test_Y, nsteps = 2000)    
    tst_outs = model(test_X_FE)
    ACC =  sum((maximum(tst_outs, dims = 1) .== tst_outs)' .& (Int.(test_Y))') / size(test_X_FE)[2]
    return ACC
end 


function test_performance_UMAP(train_UMAP, test_UMAP, labs, train_ids, test_ids)
    train_X = gpu(Matrix(train_UMAP))
    num_labs = format_Y(labs)
    train_Y = gpu(Matrix(num_labs[train_ids,:]'))
    test_X = gpu(Matrix(test_UMAP))
    test_Y = gpu(Matrix(num_labs[test_ids,:]'))
    model = train_DNN(train_X, train_Y, test_X, test_Y, nsteps = 2000)    
    tst_outs = model(test_X)
    ACC =  sum((maximum(tst_outs, dims = 1) .== tst_outs)' .& (Int.(test_Y))') / size(test_X)[2]
    return ACC
end 


function test_performance_PCA(train_PCA, test_PCA, labs, train_ids, test_ids)
    train_X = gpu(Matrix(train_PCA))
    num_labs = format_Y(labs)
    train_Y = gpu(Matrix(num_labs[train_ids,:]'))
    test_X = gpu(Matrix(test_PCA))
    test_Y = gpu(Matrix(num_labs[test_ids,:]'))
    model = train_DNN(train_X, train_Y, test_X, test_Y, nsteps = 2000)    
    tst_outs = model(test_X)
    ACC =  sum((maximum(tst_outs, dims = 1) .== tst_outs)' .& (Int.(test_Y))') / size(test_X)[2]
    return ACC
end 

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
    ACCs = []
    for (foldn, fold) in enumerate(folds)
    train_ids, train_x, test_ids, test_x = fold["train_ids"], gpu(Matrix(fold["train_x"])'), fold["test_ids"], gpu(Matrix(fold["test_x"]'))
    train_y = gpu(Matrix(Y')[:,train_ids])
    test_y = gpu(Matrix(Y')[:,test_ids])
    
    model = train_DNN(train_x, train_y, test_x, test_y, nsteps = nsteps)
    tst_outs = model(test_x)
    ACC =  sum((maximum(tst_outs, dims = 1) .== tst_outs)' .& (Int.(test_y))') / size(test_x)[2]
    push!(ACCs, ACC)
    outfile = "$outfile\n$foldn\t$ACC\t"
    end 
    return ACCs
end 