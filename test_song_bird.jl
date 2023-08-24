using CSV
using DataFrames
using DSP
using Plots
using ProgressMeter
using UMAP
using JLD
using OnlineStats
using StatsBase
#using DelimitedFiles
import DelimitedFiles: readdlm
using Plots


#default(show = true)
#plotly()

"""
    compute_liquid_states(spikes, sim_times, kernel, dt)

Compute the convolution of the spike trains with the kernel provided.

# Arguments
- `spikes::Vector`: Each array i contains spike times (in ms) of neuron i
- `sim_times::Vector`: Goes from 0 to simulation time (in ms)
- `kernel::Vector`: The kernel
- `dt::Float32`: Time step
"""
function compute_liquid_states!(liquid_states,spikes, sim_times, kernel, dt)
    @showprogress for (i, spk_trains) in enumerate(spikes)
        conv_spks = zeros(Float32, length(sim_times))
        if !isempty(spk_trains)
            # Each index represent a time, but julia does not accept 0
            adjusted_times = spk_trains/dt .+ 1
            # For the same reason, a time slice starts after 0
            adjusted_times .-= minimum(sim_times)
            conv_spks[round.(Int, adjusted_times)] .= 1
            conv_spks = conv(conv_spks, kernel)
        end
        liquid_states[i, :] = conv_spks[1:length(sim_times)]
    end

end

"""
    preprocess_spike_data(neuron_spike_ids, times)

Convert an two arrays containing spike times and neuron ids into an array where
each row is a spike time.

Nonactive neurons are not included, so an array of ids associated with each
row is returned.
"""
function preprocess_spike_data(neuron_spike_ids, times)
    neuron_ids = sort(unique(neuron_spike_ids))
    trains_array = [Vector{Float32}() for _ in 1:length(neuron_ids)]
    @inbounds for (i, id) in enumerate(neuron_ids)
        trains_array[i] = times[neuron_spike_ids .== id]
    end
    trains_array, neuron_ids
end

function load_song_bird()
    spikes = Vector{Any}([])
    nn=Vector{UInt32}([])
    ttt=Vector{Float32}([])

    file_read_list = readdlm("datasets/songbird_spikes.txt", '\t', Float64, '\n')
    nodes = [n for (n, t) in eachrow(file_read_list)]
    @inbounds for _ in 1:maximum(unique(nodes))+1
        push!(spikes,[])
    end
    @inbounds for (n, t) in eachrow(file_read_list)
        push!(spikes[UInt32(n)],t)
    end
    @inbounds for (i, t) in enumerate(spikes)
        for tt in t
            if length(t)!=0
                push!(nn,i);
                push!(ttt,Float32(tt))
            end
        end
    end
    numb_neurons=Int(maximum(nodes))+1
    maxt = (maximum(ttt))    
    (nn::Vector{UInt32},ttt::Vector{Float32},spikes::Vector{Any},numb_neurons::Integer,maxt::Real)
end

function hist2dHeat(nodes::Vector{UInt32}, times::Vector{<:Real}, denom_for_bins::Real)
    t0 = times
    n0 = nodes
    stimes = sort(times)
    ns = maximum(unique(nodes))    
    temp_vec = collect(0:Float64(maximum(stimes)/denom_for_bins):maximum(stimes))
    templ = []
    for (cnt,n) in enumerate(collect(1:maximum(nodes)+1))
        push!(templ,[])
    end
    for (cnt,n) in enumerate(nodes)
        push!(templ[n+1],times[cnt])    
    end
    list_of_artifact_rows = [] # These will be deleted as they bias analysis.
    @inbounds @showprogress for (ind,t) in enumerate(templ)
        psth = fit(Histogram,t,temp_vec)
        if sum(psth.weights[:]) == 0.0
            append!(list_of_artifact_rows,ind)
        end
    end
    adjusted_length = ns+1-length(list_of_artifact_rows)
    data = Matrix{Float64}(undef, adjusted_length, Int(length(temp_vec)-1))
    cnt = 1
    @inbounds @showprogress  for t in templ
        psth = fit(Histogram,t,temp_vec)        
        if sum(psth.weights[:]) != 0.0
            data[cnt,:] = psth.weights[:]
            @assert sum(data[cnt,:])!=0
            cnt +=1
        end
    end
    @inbounds for (ind,col) in enumerate(eachcol(data))
        data[:,ind] .= (col.-mean(col))./std(col)
    end
    data[isnan.(data)] .= 0.0
    #LinearAlgebra.normalize(data)
    data::Matrix{Float64}
end

function initiatilize_data_pre_process()
    (nodes,times,spikes,numb_neurons,maxt) = load_song_bird()

    trains_array, neuron_ids = preprocess_spike_data(nodes, times)

    @time psth = hist2dHeat(nodes,times,50.0)
    println("Calculating time-surface")
    exp_kernel = Vector{Float32}([exp(-x/30) for x in range(start=0, stop=999)])
    step_kernel = Vector{Float32}([0, 1, 0])
    sim_times = range(start=minimum(times), stop=maximum(times))
    time_surface = zeros(Float32, length(spikes), length(sim_times))        
    compute_liquid_states!(time_surface,trains_array, sim_times, exp_kernel, 1)
    time_surface,trains_array,sim_times,psth
end


@time time_surface,trains_array,sim_times,psth = initiatilize_data_pre_process()

function GetOnlinePCA(time_surface,trains_array,sim_times,psth)

    println("Calculating UMAP")
    @time result_umap = umap(time_surface, 3)
    display(Plots.plot(scatter3d(result_umap[1, :],
              result_umap[2, :],
              result_umap[3, :],
              zcolor=1:size(result_umap, 2))))
    savefig("bird_song_UMAP3d.png")
    display(Plots.plot(scatter(result_umap[1, :],result_umap[2, :],
              zcolor=1:size(result_umap, 2))))
    savefig("bird_song_UMAP2d.png")

    println("Calculating histograms and PCA")
    
    #hist2dHeat(nodes::Vector{UInt32}, times::Vector{Float32},
    #psth = fit.(Histogram,
    #            trains_array,
    #            Ref(Vector{Float32}(sim_times[1]:2:sim_times[end])))
    #psth = [Vector{Float64}(x.weights) for x in psth]
    #psth = Matrix{Float32}(reduce(hcat, psth)')
    display(Plots.heatmap(psth))

    savefig("heatmap.png")
    @time result_umap = umap(psth, 3)
    display(Plots.plot(scatter3d(result_umap[1, :],
              result_umap[2, :],
              result_umap[3, :],
              zcolor=1:size(result_umap, 2))))
    savefig("bird_song_UMAP_PSTH_3d.png")
    display(Plots.plot(scatter(result_umap[1, :],result_umap[2, :],
              zcolor=1:size(result_umap, 2))))
    savefig("bird_song_UMAP_PSTH_2d.png")


    #pca_dims = 2
    #pca_projections = CCIPCA(2,1480)
    #psth_ = psth'[:]
    #@show(typeof(psth_))
    #fit!(pca_projections, reduce(hcat, psth)')

    #psth_transformed = zeros(pca_dims, pca_interval[end])
    #@showprogress for (i, col) in enumerate(eachcol(psth))
    #    psth_transformed[:, i] = vec(OnlineStats.transform(pca_projections, col))
    #end
    #OnlineStats.transform(pca_projections,psth)
    #println("Variances explained by PCA")
    #println(OnlineStats.relativevariances(pca_projections))

    #println("Calculating UMAP with PCA")
    #@time result_umap2 = umap(psth_transformed, 3)
    #display(Plots.plot(scatter(psth_transformed[1, :],
    #psth_transformed[2, :])))
    savefig("songbird_via_onlinePCA.png")
end
GetOnlinePCA(time_surface,trains_array,sim_times,psth)