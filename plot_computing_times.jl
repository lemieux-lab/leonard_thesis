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
losses = []
for (i, infile) in enumerate(to_files)
    data  = CSV.read("$basepath/$infile", DataFrame)
    push!(datas, data[:,3])
    push!(repn, ones(length(data[:,3])) * i)
    push!(losses, data[:,2])
end 
OUTFILE = DataFrame(:t_second=>vcat(datas...), :repn=>Int64.(vcat(repn...)), :lossv=>Float64.(vcat(losses...)), :method=>fill("Julia", length(vcat(repn...))))
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
median(JuliaData.speed)
median(PythonData.speed)
JuliaData.speed
tmp_data = JuliaData[end-10_000:end, "lossv"]
X = collect(1:size(tmp_data)[1])
PearsonR = my_cor(X, tmp_data)
std_data = std(tmp_data)
fig = Figure(size = (512,256));
ax = Axis(fig[1,1], title = "Last 10,000 steps of training\nPearson Correlation : $(round(PearsonR, digits = 4)) \n Standard deviation : $(round(std_data, digits = 8))", xlabel = "steps", ylabel = "loss value")
scatter!(ax, X, tmp_data,color= RGBAf(0,0,0,0), strokewidth=1,  strokecolor = :black,)
# hexbin!(ax, X, tmp_data, cellsize = 5)
fig
PythonData.speed
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

