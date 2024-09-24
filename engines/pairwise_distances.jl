function compute_pairwise_dists(X_data, sample_ids)
    sample_size = size(sample_ids)[1]
    MM= zeros(sample_size, sample_size)
    for (row_id, sample_id) in enumerate(sample_ids)
        prwd = sqrt.(sum(abs2, X_data[sample_ids,:] .- X_data[sample_id,:]', dims = 2))
        MM[row_id,:] .= prwd
    end 
    return MM 
end