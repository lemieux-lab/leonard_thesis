include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
include("engines/utils.jl")
include("engines/pca.jl")
include("engines/coxphdnn.jl")
include("engines/gpu_utils.jl")
# CUDA.device!()
outpath, session_id = set_dirs("FE_RES")
TCGA_data, labs, samples, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")
labels = annotate_labels(labs, "Data/GDC_processed/TCGA_abbrev.txt")
CDS = biotypes .== "protein_coding"

dim_redux_size = 2
input_type = "FE"
nsteps_dim_redux = 100_000
printstep_FE = 1_000 

generate_params(X_data, emb_size) = return Dict( 
    ## run infos 
    "session_id" => session_id,  "modelid" =>  "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "printstep"=>printstep_FE, 
    ## data infos 
    "nsamples" =>size(X_data)[1], "ngenes"=> size(X_data)[2],  
    ## optim infos 
    "lr" => 5e-3, "l2" => 1e-7,"nsteps" => nsteps_dim_redux, "nsteps_inference" => Int(floor(nsteps_dim_redux * 0.1)), "nsamples_batchsize" => 1,
    ## model infos
    "emb_size_1" => emb_size, "emb_size_2" => 100, "fe_layers_size"=> [250, 100], #, "fe_hl1_size" => 50, "fe_hl2_size" => 50,
    ## plotting infos 
    "colorsFile"=> "Data/GDC_processed/TCGA_colors_def.txt"
    )

FE_params_dict = generate_params(TCGA_data, dim_redux_size)
trained_FE,  tr_epochs , tr_loss, tr_cor =  generate_patient_embedding(TCGA_data, samples, genes, FE_params_dict, labs)
# imputations (1 sample per cancer type)
# BRCA
cancer_type = "TCGA-BRCA"

# sample_id = samples[labs .== "TCGA-$(cancer_type)",:][1]
function imputation(target, model; grid_min = -2, grid_max=2, n=10) 
    # make grid 
    ngenes = size(model[1][2].weight)[2]
    range = grid_max - grid_min
    step = range / n
    grid = grid_min .+ collect(0:n) * step  
    GeneEmbeds = model[1][2](gpu(1:ngenes))
    # for loop 
    Ps = []
    Grid = [[],[]]
    for i in grid
        for j in grid
            push!(Grid[1],i)
            push!(Grid[2],j)
            Pearson = my_cor(gpu(target), model[2:end](vcat(gpu(i * ones(ngenes)'), gpu(j* ones(ngenes)'), GeneEmbeds)))
            push!(Ps, Pearson)
        end 
    end 
    return Grid, Ps
end 
function get_ctype_imputation(cancer_type, TCGA_data, trained_FE, FE_params_dict)
    avg_ctype_target = gpu(vec(mean(TCGA_data[labs .== "$(cancer_type)",:], dims = 1)))
    Grid, Ps = imputation(avg_ctype_target, trained_FE,grid_min=-2, grid_max=2, n = 75)
    fig = Figure(size = (1024,512));
    colors_labels_df = CSV.read(FE_params_dict["colorsFile"], DataFrame)
    ax1 = Axis(fig[1,1], title = "$cancer_type Scatter");
    # target type 
    sampleEmbed = cpu(trained_FE[1][1].weight)
    group = labs .!= cancer_type
    scatter!(ax1, sampleEmbed[1,group],sampleEmbed[2,group], strokewidth = 0.1, color = "grey")
    group = labs .== cancer_type
    col = colors_labels_df[colors_labels_df[:,"labs"] .== cancer_type,"hexcolor"][1]
    name_tag = colors_labels_df[colors_labels_df[:,"labs"] .== cancer_type,"name"][1]
    scatter!(ax1, sampleEmbed[1,group],sampleEmbed[2,group], strokewidth = 0.1, color = String(col), label = name_tag)
    axislegend(ax1)
    ax2 = Axis(fig[1,2], title="Imputation of $cancer_type average profile in trained FE.")
    co = contourf!(ax2, Float32.(Grid[1]), Float32.(Grid[2]), Float32.(Ps), 
        levels = Base.range(0.01, 0.99, length = 40),
        extendlow = :white, extendhigh = :magenta)
    tightlimits!(ax2)
    Colorbar(fig[1,3], co)
    return fig
end 
BRCA_im = get_ctype_imputation("TCGA-BRCA", TCGA_data, trained_FE, FE_params_dict)
LGG_im = get_ctype_imputation("TCGA-LGG", TCGA_data, trained_FE, FE_params_dict)
GBM_im = get_ctype_imputation("TCGA-GBM", TCGA_data, trained_FE, FE_params_dict)
LAML_im = get_ctype_imputation("TCGA-LAML", TCGA_data, trained_FE, FE_params_dict)

OV_im = get_ctype_imputation("TCGA-OV", TCGA_data, trained_FE, FE_params_dict)
HNSC_im = get_ctype_imputation("TCGA-HNSC", TCGA_data, trained_FE, FE_params_dict)
KIRC_im = get_ctype_imputation("TCGA-KIRC", TCGA_data, trained_FE, FE_params_dict)
KIRP_im = get_ctype_imputation("TCGA-KIRP", TCGA_data, trained_FE, FE_params_dict)

CairoMakie.save("figures/BRCA_imputation.pdf", BRCA_im)
CairoMakie.save("figures/LGG_imputation.pdf", LGG_im)
CairoMakie.save("figures/GBM_imputation.pdf", GBM_im)
CairoMakie.save("figures/LAML_imputation.pdf", LAML_im)

CairoMakie.save("figures/OV_imputation.pdf", OV_im)
CairoMakie.save("figures/HNSC_imputation.pdf", HNSC_im)
CairoMakie.save("figures/KIRC_imputation.pdf", KIRC_im)
CairoMakie.save("figures/KIRP_imputation.pdf", KIRP_im)
