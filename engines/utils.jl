function annotate_labels(labs, abbrv_filename)
    tcga_abbrv = CSV.read(abbrv_filename, DataFrame)
    labels = innerjoin(DataFrame(:abbrv=>[x[2] for x in split.(labs,"-")]),tcga_abbrv, on = :abbrv )[:,"def"]
    labels = ["$lab (n=$(sum(labels .== lab)))" for lab in labels]
    return labels 
end 
