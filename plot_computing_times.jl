include("engines/init.jl")
include("engines/factorized_embeddings.jl")
include("engines/figures.jl")
include("engines/data_processing.jl")

# merge computing times files 
basepath = "interactive_figures/comp_times/"
to_files = readdir(basepath)
OUTFILE = DataFrame()
datas = []
repn = []
for (i, infile) in enumerate(to_files)
    data  = CSV.read("$basepath/$infile", DataFrame)[:,3]
    push!(datas, data)
    push!(repn, ones(length(data)) * i)
end 
OUTFILE = DataFrame(:t_second=>vcat(datas...), :repn=>Int64.(vcat(repn...)), :method=>fill("Julia", length(vcat(repn...))) )
CSV.write("interactive_figures/computing_times_Julia2024.csv",OUTFILE)

JuliaData = CSV.read("interactive_figures/computing_times_Julia2024.csv", DataFrame)
JuliaData = JuliaData[JuliaData[:,"repn"].==1,:]
JuliaData[:,"stepn"] = collect(1:size(JuliaData)[1])
JuliaData[:,"speed"] .= 1 ./ (JuliaData.t_second .- vcat(JuliaData.t_second[1], JuliaData.t_second[1:end-1]))
JuliaData = JuliaData[2:end,:]
PythonData = CSV.read("interactive_figures/computing_times_trofimov2020.csv", DataFrame)
PythonData[:,"method"] .= "Python"
PythonData = PythonData[PythonData.repn .== 5, :]
PythonData[:,"stepn"] .= collect(1:size(PythonData)[1])
PythonData[:,"speed"] .= 1 ./ ( PythonData.t_second .- vcat(PythonData.t_second[1], PythonData.t_second[1:end-1]))
PythonData = PythonData[2:end,:]
JuliaData
PythonData
merged_data = vcat(JuliaData, PythonData, cols =:union)
# plot Python data 
fig = Figure(size = (1024,512));
ax = Axis(fig[1,1], title = "Computing times between Julia and Python", xlabel = "Number of steps", ylabel="elapsed time (s)")
# [scatter!(ax, collect(1:sum(PythonData.repn .== i)), PythonData[PythonData.repn .== i,"t_second"], label = "Python $i") for i in unique(PythonData.repn)]
# [scatter!(ax, collect(1:sum(JuliaData.repn .== i)), JuliaData[JuliaData.repn .== i,"t_second"], label = "Julia $i") for i in unique(JuliaData.repn)]
scatter!(ax, merged_data.stepn, merged_data.t_second, color = Int64.(merged_data.method .== "Julia"), label = merged_data.method)
ax2 = Axis(fig[1,2],title="Speed comparison between the approaches",  xlabel = "Implementation", ylabel = "Iteration Speed (Hz)", xticks = (collect(0:1), ["Python", "Julia"]));
merged_data.speed
boxplot!(ax2, Int64.(merged_data.method .== "Julia"), merged_data.speed)
fig 

axislegend(ax)

fig
scatter!(ax, collect(1:sum(PythonData.repn .== i)), PythonData[PythonData.repn .== i,"t_second"], label = "Python $i")
i = 1
scatter!(ax, collect(1:sum(JuliaData.repn .== i)), JuliaData[JuliaData.repn .== i,"t_second"], label = "Julia $i")
axislegend(ax, position = :rt)
fig
fig

