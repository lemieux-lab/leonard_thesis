include("engines/init.jl")
include("engines/figures.jl")

# gather all results 
basedir = "CPHDNN_RES"
median_lo_up_scores_FE = []
dr_sizes = []
for (root, dirs, files) in walkdir(basedir)
    for filename in files
        if occursin("_surv_scores_cphdnn.h5", filename) 
            println("$root/$filename")
            inf = h5open("$root/$filename","r")
            # dimredux_type = inf["dim_redux_type"]
            m = read(inf["m"])
            up = read(inf["up_95"])
            lo= read(inf["lo_95"])
            close(inf)
            push!(dr_sizes, parse(Int, split(filename,"_")[2]))
            push!(median_lo_up_scores_FE, (m,lo,up))
        end 
    end 
end 
inf = h5open("CPHDNN_RES/2024-10-02T19:02:17.261/FE_2_surv_scores_cphdnn.h5", "r")
m = read(inf["m"])
median_lo_up_scores_FE
# filter 
# scatter + lines plot