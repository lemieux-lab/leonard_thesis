function plot_train_test_patient_embed(trained_FE, test_model, tissue_labels, train_ids, test_ids)
    train_embed = cpu(trained_FE.net[1][1].weight)
    test_embed = cpu(test_model[1][1].weight)

    fig = Figure(size = (1024,800));
    ax = Axis(fig[1,1],title="Train and Test patient embedding\n$(params["modelid"])", xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
    colors_dict = Dict([(lab, RGBf(rand(), rand(), rand())) for lab in unique(tissue_labels)])
    # first plot train embed with circles.
    for (i, group_lab) in enumerate(unique(tissue_labels))
        group = tissue_labels[train_ids] .== group_lab
        scatter!(ax, train_embed[1,group], train_embed[2,group], strokewidth = 0, color = colors_dict[group_lab], marker = :circle, label = group_lab)
    end 
    # then plot test embed with triangles
    for (i, group_lab) in enumerate(unique(tissue_labels))
        group = tissue_labels[test_ids] .== group_lab
        scatter!(ax, test_embed[1,group], test_embed[2,group], strokewidth = 1, color = colors_dict[group_lab], marker = :utriangle)
    end 
    fig[1,2] = axislegend(ax, position =:rc, labelsize = 8, rowgap=0)
    return fig 
end 

function plot_UMAP(TCGA_data,labs)

    umap_model = UMAP_(TCGA_data', 2;min_dist = 0.99, n_neighbors = 200);

    ## plotting embed using UMAP 
    fig = Figure(size = (512,512));
    ax2 = Axis(fig[1,1], xlabel = "UMAP-1", ylabel="UMAP-2", aspect = 1);
    for group_lab in unique(labs)
        group = labs .== group_lab
        plot!(umap_model.embedding[1,group],umap_model.embedding[2,group], label = group_lab)
    end 
    return fig 
end 
function plot_interactive(params, train_embed, test_embed, train_ids, test_ids,  tissue_labels)
    colors_dict = Dict([(lab, RGBf(rand(), rand(), rand())) for lab in unique(tissue_labels)])    
    traces = [PlotlyJS.scatter(x=train_embed[1,tissue_labels[train_ids] .== group_lab], y=train_embed[2,tissue_labels[train_ids] .== group_lab], marker = attr(color= colors_dict[group_lab]), mode = "markers", name = group_lab) for group_lab in unique(tissue_labels)]
    [push!(traces, PlotlyJS.scatter(x=test_embed[1,tissue_labels[test_ids] .== group_lab], y=test_embed[2,tissue_labels[test_ids] .== group_lab], marker = attr(symbol="diamond", line_width=1,color= colors_dict[group_lab]), mode = "markers", name = "TEST - $(group_lab)")) for group_lab in unique(tissue_labels)]
    P = PlotlyJS.plot(traces, 
        Layout(title = "Patient Factorized Embedding 2D with train and test samples\nmodel ID: $(params["modelid"])",
        yaxis = attr(showgrid=true, gridwidth=1, gridcolor="black", zeroline=true, zerolinewidth=1, zerolinecolor="black"),
        xaxis = attr(showgrid=true, gridwidth=1, gridcolor="black", zeroline=true, zerolinewidth=1, zerolinecolor="black"),
        plot_bgcolor = :white, 
    ))
    PlotlyJS.savefig(P, "interactive_figures/$(params["modelid"])_FE_2D_visualisation_train_test.html")
end

function plot_interactive(trained_FE, params, patients, labels)
    trained_FE_mat = cpu(trained_FE.net[1][1].weight)
    # X_tr = fit_transform_pca(trained_FE_mat,2)
    FE_to_plot = DataFrame(:ids => patients, :EMBED1=> trained_FE_mat[1,:], :EMBED2=>trained_FE_mat[2,:], :group=>labels) 
    # FE_to_plot = DataFrame(:EMBED1=> X_tr[1,:], :EMBED2=>X_tr[2,:], :group=>labels) 

    FE_to_plot = innerjoin(FE_to_plot, DataFrame(:group => unique(labels), :type=>[i%3 for i in 1:size(unique(labels))[1]]),on = :group)
    CSV.write("interactive_figures/$(params["modelid"])_FE_2D_embedding.csv", FE_to_plot)
    P = PlotlyJS.plot(
        FE_to_plot, x=:EMBED1, y=:EMBED2, color=:group, symbol = :type, ids = :ids,
        kind = "scatter", mode = "markers", 
            Layout(
                title = "FE 2D visualisation by subgroup"

    ))
    PlotlyJS.savefig(P, "interactive_figures/$(params["modelid"])_FE_2D_visualisation.html")
end 

function plot_patient_embedding(patient_FE, fig, tissue_labels, title, i)
    
    ax2 = Axis(fig[1,i],title=title, xlabel = "Patient-FE-1", ylabel="Patient-FE-2", aspect = 1);
    markers = [:diamond, :circle, :utriangle, :rect]
    for (i, group_lab) in enumerate(unique(tissue_labels))
        group = tissue_labels .== group_lab
        scatter!(ax2, patient_FE[1,group],patient_FE[2,group], strokewidth = 0.1, color = RGBf(rand(), rand(), rand()), marker = markers[i%4 + 1], label = group_lab)
    end 
    fig[1,i+1] = axislegend(ax2, position =:rc, labelsize = 8, rowgap=0)
    return fig
end 

function plot_FE_reconstruction(model, X, Y;modelID="")
    OUTS_ = model.net((X[1][1:500_000], X[2][1:500_000]))
    Y_ = Y[1:500_000]
    pearson = my_cor(OUTS_, Y_)
    fig = Figure(size = (512,512));
    ax1 = Axis(fig[1,1], 
        title = "Gene expression reconstruction of FE model on TCGA data \nmodel: $modelID \n Pearson R = $(round(pearson,digits = 4))", 
        ylabel = "Real Log TPM count", xlabel = "FE predicted Log TPM count")
        #limits = (-2,2,-2,2),
        #xticks = collect(-2:2), yticks = collect(-2:2));
    hexbin!(ax1, cpu(OUTS_),cpu(Y_), cellsize = 0.05, colorscale = log10)
    #Colorbar(fig[1,2])
    lines!(ax1, [0,maximum(cpu(Y_))],[0,maximum(cpu(Y_))], linestyle=:dash, color =:black)
    ax1.aspect = 1
    #resize_to_layout!(fig)
    return fig
end 