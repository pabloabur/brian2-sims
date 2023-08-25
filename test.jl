using CSV
using DataFrames
using DSP
using Plots
using ProgressMeter
using UMAP
using JLD
using OnlineStats
using StatsBase

default(show = true)
plotly()

"""
    compute_liquid_states(spikes, sim_times, kernel, dt)

Compute the convolution of the spike trains with the kernel provided.

# Arguments
- `spikes::Vector`: Each array i contains spike times (in ms) of neuron i
- `sim_times::Vector`: Goes from 0 to simulation time (in ms)
- `kernel::Vector`: The kernel
- `dt::Float32`: Time step
"""
function compute_liquid_states(spikes, sim_times, kernel, dt)
    liquid_states = zeros(Float32, length(spikes), length(sim_times))
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

    return liquid_states
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
    @showprogress for (i, id) in enumerate(neuron_ids)
        trains_array[i] = times[neuron_spike_ids .== id]
    end

    trains_array, neuron_ids
end

target_path = ARGS[1] * "/"
if !isfile(target_path * "output_spikes.jld")
    df = CSV.read(target_path * "output_spikes.csv", DataFrame)

    neuron_spike_ids = df.id
    neuron_spike_ids = Vector{Int32}(neuron_spike_ids)
    times = df.time_ms
    times = Vector{Float32}(times)

    println("Shaping arrays for time-surface calculation")
    trains_array, neuron_ids = preprocess_spike_data(neuron_spike_ids, times)
    save(target_path * "output_spikes.jld",
        "trains_array", trains_array,
        "neuron_ids", neuron_ids,
        "neuron_spike_ids", neuron_spike_ids,
        "times", times)
else
    trains_array = load(target_path * "output_spikes.jld", "trains_array")
    neuron_ids = load(target_path * "output_spikes.jld", "neuron_ids")
    neuron_spike_ids = load(target_path * "output_spikes.jld", "neuron_spike_ids")
    times = load(target_path * "output_spikes.jld", "times")
end

# Useful scenarios
# End of simulation, non changing weights: 194999 onwards
# Transition between plasticity active and turned off: between 138000 and 142000
# Beginning of simulation: between 0 and 4000
trains_array = map(filter(x -> x>138000 && x<142000), trains_array)
empty_elements = isempty.(trains_array)
trains_array = trains_array[.!empty_elements]
neuron_ids = neuron_ids[.!empty_elements]

println("Calculating time-surface")
exp_kernel = Vector{Float32}([exp(-x/30) for x in range(start=0, stop=999)])
step_kernel = Vector{Float32}([0, 1, 0])
sim_times = Vector{Int32}(range(start=minimum(minimum.(trains_array)),
                                stop=maximum(maximum.(trains_array))))
time_surface = compute_liquid_states(trains_array, sim_times, exp_kernel, 1)

println("Calculating UMAP")
@time result_umap = umap(time_surface, 3)
scatter3d(result_umap[1, :],
          result_umap[2, :],
          result_umap[3, :],
          zcolor=1:size(result_umap, 2), marker=(2, 2, :auto, stroke(0)))

println("Calculating histograms and PCA")
psth = fit.(Histogram,
            trains_array,
            Ref(Vector{Float32}(sim_times[1]:2:sim_times[end])))
psth = [Vector{Float64}(x.weights) for x in psth]
psth = reduce(hcat, psth)'

pca_dims = 6
pca_interval = 1:150
pca_projections = CCIPCA(pca_dims, size(psth, 1))
@showprogress for i in pca_interval
    fit!(pca_projections, psth[:, i])
end
psth_transformed = zeros(pca_dims, pca_interval[end])
@showprogress for (i, col) in enumerate(eachcol(copy(psth[:, pca_interval])))
    psth_transformed[:, i] = vec(OnlineStats.transform(pca_projections, col))
end

println("Variances explained by PCA")
println(OnlineStats.relativevariances(pca_projections))

println("Calculating UMAP with PCA")
@time result_umap2 = umap(psth_transformed, 3)
scatter3d(result_umap2[1, :],
          result_umap2[2, :],
          result_umap2[3, :],
          zcolor=1:size(result_umap2, 2), marker=(2, 2, :auto, stroke(0)))
