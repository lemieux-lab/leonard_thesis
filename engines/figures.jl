
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

function plot_FE_reconstruction(model, X, Y)
    OUTS_ = model.net((X[1][1:500_000], X[2][1:500_000]))
    Y_ = Y[1:500_000]
    pearson = my_cor(OUTS_, Y_)
    fig = Figure(size = (512,512));
    ax1 = Axis(fig[1,1], 
        title = "Gene expression reconstruction of FE model on TCGA data \n Pearson R = $(round(pearson,digits = 4))", 
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