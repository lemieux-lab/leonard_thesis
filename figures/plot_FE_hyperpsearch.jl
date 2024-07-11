include("engines/init.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")

## load TCGA data 
TCGA_data, labs, patients, genes, biotypes = load_tcga_dataset("Data/GDC_processed/TCGA_TPM_lab.h5")

## load params dict
model_params = gather_params("figures/tables/")
model_params = model_params[model_params[:,"emb_size_2"] .!= 2,:]
model_params = model_params[model_params[:,"l2"] .!= 1e-4,:]

function plot_FE_hyperparams_density(model_params;nbins = 25) 
    fig = Figure(size = (1100, 1400));
    for (row, emb_size_2) in enumerate(sort(unique(model_params[:,"emb_size_2"])))
        for (col, l2_val) in enumerate(sort(unique(model_params[:, "l2"])))
            group_data = (model_params[:,"emb_size_2"] .== emb_size_2) .& (model_params[:,"l2"] .== l2_val)
            if sum(group_data) != 0
                fname = model_params[(model_params[:,"emb_size_2"] .== emb_size_2) .& (model_params[:,"l2"] .== l2_val), "modelid"][1]
                data = CSV.read("figures/tables/$(fname)_trained_2D_factorized_embedding.csv", DataFrame)
                cellsize = (maximum(data[:,1]) - minimum(data[:,1])) / nbins
                ax = Axis(fig[row, col], title = "EMB. Size = $(emb_size_2), L2 = $(l2_val)", limits = (-5,5,-5,5));
                hexbin!(ax, data[:,1], data[:,2], cellsize = cellsize, colorscale = log10);
                hidespines!(ax)
            end 
        end 
    end 
    CairoMakie.save("figures/FE_hyperparams_density.pdf", fig)
    CairoMakie.save("figures/FE_hyperparams_density.png", fig)
    fig
end 
plot_FE_hyperparams_density(model_params,nbins = 30)

function plot_FE_hyperparams_scatter(model_params, labs, colorsFile)
    colors_labels_df = CSV.read(colorsFile, DataFrame)
    fig = Figure(size = (1100, 1400));
    for (row, emb_size_2) in enumerate(sort(unique(model_params[:,"emb_size_2"])))
        for (col, l2_val) in enumerate(sort(unique(model_params[:, "l2"])))
            group_data = (model_params[:,"emb_size_2"] .== emb_size_2) .& (model_params[:,"l2"] .== l2_val)
            if sum(group_data) != 0
                fname = model_params[(model_params[:,"emb_size_2"] .== emb_size_2) .& (model_params[:,"l2"] .== l2_val), "modelid"][1]
                data = CSV.read("figures/tables/$(fname)_trained_2D_factorized_embedding.csv", DataFrame)
                ax = Axis(fig[row, col], title = "EMB. Size = $(emb_size_2), L2 = $(l2_val)");
                for (i, group_lab) in enumerate(unique(labs))
                    group = labs .== group_lab
                    col = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"hexcolor"][1]
                    name = colors_labels_df[colors_labels_df[:,"labs"] .== group_lab,"name"][1]
                    scatter!(ax, data[group,1],data[group,2], markersize = 5, strokewidth = 0, color = String(col), label = name)
                end            
                hidespines!(ax, :t, :r)
            end 
        end 
    end 
    CairoMakie.save("figures/FE_hyperparams_scatter.pdf", fig)
    CairoMakie.save("figures/FE_hyperparams_scatter.png", fig)
    fig
end 

plot_FE_hyperparams_scatter(model_params, labs, "Data/GDC_processed/TCGA_colors_def.txt")
groups = groupby(model_params[:,["modelid", "l2", "emb_size_2"]], ["l2"])
