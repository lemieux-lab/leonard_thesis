include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/cphdnn_model_evaluation.jl")
include("engines/figures.jl")
include("engines/utils.jl")
outpath, session_id = set_dirs("RES_ISMB") ;

tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
TCGA_datasets = load_tcga_datasets(tcga_datasets_list);
BRCA_data = TCGA_datasets["BRCA"]
LGG_data = TCGA_datasets["LGG"]
OV_data = TCGA_datasets["OV"]

LGNAML_data = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")) 
keep_tcga_cds = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep_lgnaml_common = [gene in BRCA_data["dataset"].genes[keep_tcga_cds] for gene in LGNAML_data["dataset"].genes];

BRCA_data["CDS"] = keep_tcga_cds
BRCA_data["CF"] = zeros(size(BRCA_data["dataset"].data)[1],0)  
LGG_data["CDS"] = keep_tcga_cds  
LGG_data["CF"] = zeros(size(LGG_data["dataset"].data)[1],0) 
OV_data["CDS"] = keep_tcga_cds  
OV_data["CF"] = zeros(size(OV_data["dataset"].data)[1],0) 

LGNAML_data["CDS"] = keep_lgnaml_common 
LGNAML_data["CF"] = zeros(size(LGNAML_data["dataset"].data)[1],0) 

### Use LGG, BRCA, AML, OV 
### Use random vs pca 
### Use CPHDNN, Cox-ridge 
### 10 replicates


# Cox NL loss and concordance index do not always concord. (1A-BRCA)
# BRCA clinf 16 CPHDNN / Cox-ridge training report learning curve (c-index, loss)
# linear, adam, lr=1e-6, cph_wd=1e-3, 150K steps.
clinical_factors = Matrix(CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame))
BRCA_data["CF"] = clinical_factors

evaluate_model("cphdnn", BRCA_data, 0, cph_nb_hl = 2, hlsize = 512, sigmoid_output=false,
    nepochs = 150_000, cph_lr = 1e-6, cph_l2 = 1e-3, dim_redux_type = "CLINF")
BRCA_data["params"]
PARAMS = gather_params("RES_ISMB/")
LC = gather_learning_curves(basedir="RES_ISMB/", skip_steps=100)
PARAMS[:,"TYPE"] = replace.(["$x-$y" for (x,y) in  zip(PARAMS[:, "dim_redux_type"], PARAMS[:, "insize"])], "RDM"=>"CDS")

LOSSES_DF = innerjoin(LC, sort(PARAMS, ["TYPE"]), on = "modelid");
DATA_df = sort(LOSSES_DF[LOSSES_DF[:,"dataset"] .== "BRCA",:], ["TYPE"])
names(DATA_df)
DATA_df[:, "loss_vals_log"] .= log10.(DATA_df.loss_vals)
fig = Figure(size = (1600,400));
for (row_id, metric) in enumerate(["cind_vals", "loss_vals_log"])
    #DATA_df = sort(LOSSES_DF[LOSSES_DF[:,"dataset"] .== dataset,:], ["TYPE"])
    for fold_id in 1:5
        FOLD_data = DATA_df[DATA_df.foldns .== fold_id,:]
        ax = Axis(fig[row_id,fold_id]; 
            xlabel = "steps", ylabel = metric, 
            title = "BRCA - CPHDNN - CLINF (16) - FOLD ($fold_id)");
        lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train", metric], color = "blue", label = "train") 
        lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test", metric], color = "orange", label = "test")
    end    
end
fig 
CairoMakie.save("figures/PDF/ismb_two_pager_1a_brca.pdf", fig) 
CairoMakie.save("figures/PDF/ismb_two_pager_1a_brca.svg", fig) 
CairoMakie.save("figures/ismb_two_pager_1a_brca.png", fig) 
CSV.write("figures/ismb_two_pager_1a_params.csv", PARAMS) 

# Cox NL loss and concordance index do not always concord. (1A-LGN)
# LGN clinf 8 CPHDNN / Cox-ridge training report learning curve (c-index, loss)
# linear, adam, lr=1e-6, cph_wd=1e-3, 150K steps.
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
push!(lnames, "Age")
clinical_factors = hcat(CF_bin, lgn_CF[:,"Age_at_diagnosis"])
LGNAML_data["CF"] = clinical_factors
evaluate_model("cphdnn", LGNAML_data, 0, cph_nb_hl = 2, hlsize = 512, sigmoid_output=true,
    nepochs = 150_000, cph_lr = 1e-6, cph_l2 = 1e-3, dim_redux_type = "CLINF")
PARAMS = gather_params("RES_ISMB/")
LC = gather_learning_curves(basedir="RES_ISMB/", skip_steps=100)
PARAMS[:,"TYPE"] = replace.(["$x-$y" for (x,y) in  zip(PARAMS[:, "dim_redux_type"], PARAMS[:, "insize"])], "RDM"=>"CDS")

LOSSES_DF = innerjoin(LC, sort(PARAMS, ["TYPE"]), on = "modelid");
DATA_df = sort(LOSSES_DF[LOSSES_DF[:,"dataset"] .== "LgnAML",:], ["TYPE"])
DATA_df[:, "loss_vals_log"] .= log10.(DATA_df.loss_vals)
fig = Figure(size = (1600,400));
for (row_id, metric) in enumerate(["cind_vals", "loss_vals_log"])
    #DATA_df = sort(LOSSES_DF[LOSSES_DF[:,"dataset"] .== dataset,:], ["TYPE"])
    for fold_id in 1:5
        FOLD_data = DATA_df[DATA_df.foldns .== fold_id,:]
        ax = Axis(fig[row_id,fold_id]; 
            xlabel = "steps", ylabel = metric, 
            title = "Leucegene - CPHDNN - CLINF (8) - FOLD ($fold_id)");
        lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train", metric], color = "blue", label = "train") 
        lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test", metric], color = "orange", label = "test")
        if metric == "loss_vals_log"
            lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train", "l2_vals"], color = :black, label = "l2 penalty") 
        end
        if row_id == 1 & fold_id == 5
            axislegend(ax)
        end  
    end    
end
fig 
CairoMakie.save("figures/PDF/ismb_two_pager_1a_lgn.pdf", fig) 
CairoMakie.save("figures/PDF/ismb_two_pager_1a_lgn.svg", fig) 
CairoMakie.save("figures/ismb_two_pager_1a_lgn.png", fig) 
CSV.write("figures/ismb_two_pager_1a_params.csv", PARAMS) 


# Linear output (output range is unbound) vs sigmoid (output clamped between 0 and 1). (1B)
ngenes = sum(BRCA_data["CDS"])
evaluate_model("coxridge", BRCA_data, ngenes, cph_nb_hl = 0, hlsize = 0,  
    nepochs = 80_000, cph_lr = 1e-6, cph_l2 = 1e-7, dim_redux_type = "CDS", sigmoid_output = false)

prev_m, curr_m, lr_curves_df = evaluate_model_debug("cphdnn", BRCA_data, ngenes, cph_nb_hl = 2, hlsize = 512, 
    nepochs = 50_000, cph_lr = 1e-6, cph_l2 = 1e-2, dim_redux_type = "CDS", sigmoid_output = false)
lr_curves_df
names(lr_curves_df)
fig = Figure(size = (812,400));
ax1 = Axis(fig[1,1], ylabel = "concordance index", xlabel = "steps")
lines!(ax1, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"cind_tr"]), color = :blue, label = "train", linewidth = 3)
lines!(ax1, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"cind_tst"]), color = :orange, label = "test", linewidth = 3)
ax2 = Axis(fig[2,1], ylabel = "Cox NL Loss", xlabel = "steps")
lines!(ax2, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"tr_loss"]), color = :blue, label = "train", linewidth = 3)
lines!(ax2, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"tst_loss"]), color = :orange, label = "test", linewidth = 3)
axislegend(ax2, position = :rt)

  

fig = plot_hist_scores!(fig, BRCA_data, prev_m, log_tr = false)
tr_outs = cpu(vec(prev_m(BRCA_data["data_prep"]["train_x"])))
hist_df = sort(DataFrame(:outs=>tr_outs, :deceased=>vec(cpu(BRCA_data["data_prep"]["train_y_e"])) ), "outs")
hist_df[:,"II"] .= 1

hist_deceased = [size(CC)[1] for CC in partition(tr_outs[vec(cpu(BRCA_data["data_prep"]["train_y_e"])) .== 1], intervals(tr_outs, 30))]
hist_alive = [size(CC)[1] for CC in partition(tr_outs[vec(cpu(BRCA_data["data_prep"]["train_y_e"])) .== 0], intervals(tr_outs, 30))]
x_ticks = intervals(tr_outs, 30)[1:end-1]

fig = Figure(size = (1024,512));
ax = Axis(fig[1,1]);
fig
CairoMakie.save("figures/PDF/ismb_two_pager_1b_linear.pdf", fig)
CairoMakie.save("figures/PDF/ismb_two_pager_1b_linear.svg", fig)
CairoMakie.save("figures/ismb_two_pager_1b_linear.png", fig)

ngenes = sum(BRCA_data["CDS"])
prev_m, curr_m, lr_curves_df = evaluate_model_debug("cphdnn", BRCA_data, ngenes, cph_nb_hl = 2, hlsize = 512, 
    nepochs = 150_000, cph_lr = 1e-6, cph_l2 = 1e-3, dim_redux_type = "CDS", sigmoid_output = true)
fig = Figure(size = (812,400));
ax1 = Axis(fig[1,1], ylabel = "concordance index", xlabel = "steps")
lines!(ax1, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"cind_tr"]), color = :blue, label = "train", linewidth = 3)
lines!(ax1, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"cind_tst"]), color = :orange, label = "test", linewidth = 3)
ax2 = Axis(fig[2,1], ylabel = "Cox NL Loss", xlabel = "steps")
lines!(ax2, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"tr_loss"]), color = :blue, label = "train", linewidth = 3)
lines!(ax2, collect(1:size(lr_curves_df)[1]), Float32.(lr_curves_df[:,"tst_loss"]), color = :orange, label = "test", linewidth = 3)
axislegend(ax2, position = :rt)
fig = plot_hist_scores!(fig, BRCA_data, prev_m, log_tr = true)
CairoMakie.save("figures/PDF/ismb_two_pager_1b_sigmoid.pdf", fig)
CairoMakie.save("figures/PDF/ismb_two_pager_1b_sigmoid.svg", fig)
CairoMakie.save("figures/ismb_two_pager_1b_sigmoid.png", fig)
    

BRCA_data["params"]



# Using convergence criterion for unbiased determination of L2, learning rate, optimization steps hyper-parameters. (1C)
# Reporting cross-validation metrics with c-index scores aggregation + bootstrapping is more consistent than using the average. (1D) 