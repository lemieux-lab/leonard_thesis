include("engines/init.jl")
include("engines/figures.jl")

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
RUNS[ismissing.(RUNS[:,"modeltype"]) .!= 1,["n_epochs","c_ind_test","modeltype", "n_components"]]

RUNS[:,"c_ind_test"] .> 0
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], xlabel = "input size", ylabel = "c-index")
scatter!(ax, log2.(dr_sizes), Float32.([score[1] for score in median_lo_up_scores_FE]))
fig
median_lo_up_scores_FE
# filter 
# scatter + lines plot