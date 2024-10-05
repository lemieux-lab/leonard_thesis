include("engines/init.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")
# gather all results 
basedir = "CPHDNN_RES"
median_lo_up_scores_FE = []
dr_sizes = []
for (root, dirs, files) in walkdir(basedir)
    for filename in files
        if occursin("_surv_scores_cphdnn.h5", filename) 
            inf = h5open("$root/$filename","r")
            # dimredux_type = inf["dim_redux_type"]
            m = read(inf["m"])
            up = read(inf["up_95"])
            lo= read(inf["lo_95"])
            nsteps = read(inf["nsteps_FE"])
            close(inf)
            if nsteps >= 120_000
                push!(dr_sizes, parse(Int, split(filename,"_")[2]))
                push!(median_lo_up_scores_FE, (m,lo,up, nsteps))
            end 
        end 
    end 
end 

RUNS = gather_params(basedir)
UMAP_res = RUNS[ismissing.(RUNS[:,"modeltype"]) .!= 1,["n_epochs","c_ind_test","modeltype", "n_components"]]
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], xlabel = "input size", ylabel = "c-index", xticks=(log2.(sort(unique(UMAP_res[:,"n_components"]))), string.(sort(unique(UMAP_res[:,"n_components"])))) )
boxplot!(ax, log2.(UMAP_res[:,"n_components"]), Float32.(UMAP_res[:,"c_ind_test"]))
scatter!(ax, log2.(UMAP_res[:,"n_components"]), Float32.(UMAP_res[:,"c_ind_test"]))
fig
median_lo_up_scores_FE
# filter 
# scatter + lines plot