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
    nnn=Vector{UInt32}([])
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
                push!(nnn,i);
                push!(ttt,Float32(tt))
            end
        end
    end
    numb_neurons=Int(maximum(nodes))+1

    maxt = (maximum(ttt))    
    (nnn::Vector{UInt32},ttt::Vector{Float32},spikes::Vector{Any},numb_neurons::Integer,maxt::Real)
end
function initiatilize_data_pre_process()
    (nodes,times,spikes,numb_neurons,maxt) = load_song_bird()
    trains_array, neuron_ids = preprocess_spike_data(nodes, times)

    println("Calculating time-surface")
    exp_kernel = Vector{Float32}([exp(-x/30) for x in range(start=0, stop=999)])
    step_kernel = Vector{Float32}([0, 1, 0])
    sim_times = range(start=minimum(times), stop=maximum(times))
    time_surface = zeros(Float32, length(spikes), length(sim_times))        
    compute_liquid_states!(time_surface,trains_array, sim_times, exp_kernel, 1)
    time_surface,trains_array,sim_times
end

@time time_surface,trains_array,sim_times = initiatilize_data_pre_process()
function GetOnlinePCA(time_surface,trains_array,sim_times)

    println("Calculating UMAP")
    @time result_umap = umap(time_surface, 3)
    display(Plots.plot(scatter3d(result_umap[1, :],
              result_umap[2, :],
              result_umap[3, :],
              zcolor=1:size(result_umap, 2), marker=(2, 2, :auto, stroke(0)))))
    savefig("bird_song_UMAP.png")
    println("Calculating histograms and PCA")
    psth = fit.(Histogram,
                trains_array,
                Ref(Vector{Float32}(sim_times[1]:2:sim_times[end])))
    psth = [Vector{Float64}(x.weights) for x in psth]
    psth = reduce(hcat, psth)'
    display(Plots.heatmap(psth))
    savefig("heatmap.png")
    pca_dims = 6
    #pca_interval = 1:length(psth)
    pca_projections = CCIPCA(pca_dims, size(psth, 1))
    #@showprogress for i in pca_interval
    fit!(pca_projections, psth)
    #end
    psth_transformed = zeros(pca_dims, pca_interval[end])
    @showprogress for (i, col) in enumerate(eachcol(copy(psth[:, pca_interval])))
        psth_transformed[:, i] = vec(OnlineStats.transform(pca_projections, col))
    end

    println("Variances explained by PCA")
    println(OnlineStats.relativevariances(pca_projections))

    println("Calculating UMAP with PCA")
    @time result_umap2 = umap(psth_transformed, 3)
    display(Plots.plot(scatter3d(result_umap2[1, :],
              result_umap2[2, :],
              result_umap2[3, :],
              zcolor=1:size(result_umap2, 2), marker=(2, 2, :auto, stroke(0)))))
    savefig("songbird_via_onlinePCA.png")
end
GetOnlinePCA(time_surface,trains_array,sim_times)