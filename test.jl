using CSV
using DataFrames
using DSP
using Plots
using ProgressMeter
using UMAP

function compute_liquid_states(spikes, sim_times, kernel, dt)
    """ 
    Parameters
    ----------
    spikes : array of arrays
        Each array i contains spike times (in ms) of neuron i
    sim_times : array of integers
        Goes from 0 to simulation time. Unit of ms is considered.
    """
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

function preprocess_spike_data(neuron_spike_ids, times)
    # Nonactive neurons below max are included to make easier to analyse output
    neuron_ids = sort(unique(neuron_spike_ids))
    trains_array = Vector{Float32}[Vector{Float32}[] for _ in 1:maximum(neuron_ids)]
    @showprogress for id in neuron_ids
        trains_array[id] = times[neuron_spike_ids .== id]
    end

    trains_array
end

df = CSV.read("bal_stdp_cudadgx/output_spikes.csv", DataFrame)
df = filter(row -> row.time_ms > 194999, df)

neuron_spike_ids = df.id
neuron_spike_ids = Vector{Int32}(neuron_spike_ids)
neuron_spike_ids = [n + 1 for n in neuron_spike_ids]

times = df.time_ms
times = Vector{Float32}(times)

println("Shaping arrays for time-surface calculation")
trains_array = preprocess_spike_data(neuron_spike_ids, times)

println("Calculating time-surface")
exp_kernel = Vector{Float32}([exp(-x/30) for x in range(start=0, stop=999)])
step_kernel = Vector{Float32}([0, 1, 0])
sim_times = Vector{Int32}(range(start=minimum(times), stop=maximum(times)))
time_surface = compute_liquid_states(trains_array, sim_times, step_kernel, 1)

println("Calculating UMAP")
@time result_umap = umap(time_surface, 3)
plot_range = 400:900
scatter3d(result_umap[1, plot_range],
          result_umap[2, plot_range],
          result_umap[3, plot_range],
          zcolor=plot_range, marker=(2, 2, :auto, stroke(0)))

# `plot_range` means time steps after slicing `times` array, so we adjustment
raster_plot_range = in.(times, Ref(minimum(times) .+ collect(plot_range)))
scatter(times[raster_plot_range],
        neuron_spike_ids[raster_plot_range],
        markersize=2,
        markerstrokewidth=0)
