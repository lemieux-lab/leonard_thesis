include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")

TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS=  biotypes .== "protein_coding";
# load model
basepath = "FE_RES/2024-05-15T21:48:09.546/fba5cbb2b75a01c86eac9"
load_model_bson(model_path) = gpu(BSON.load("$(model_path)_in_training_model.bson")["model"])
model = load_model_bson(basepath);
# parse data 
load_train_test_ids = BSON.load("$(basepath)_train_test_ids.bson")
load_params = BSON.load("$(basepath)_params.bson")
train_ids = load_train_test_ids["train_ids"] 
test_ids = load_train_test_ids["test_ids"]

train_data = TCGA_data[train_ids,CDS];
test_data = TCGA_data[test_ids,CDS];
test_model = do_inference(model, load_params, test_data ,patients[test_ids], genes[CDS])
plot_interactive(load_params, cpu(model[1][1].weight), cpu(test_model[1][1].weight), train_ids, test_ids,  labels)
cpu(model[1][1])
cpu(test_model[1][1])