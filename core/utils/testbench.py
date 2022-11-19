# -*- coding: utf-8 -*-
""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking neurons.
Unreleased software, Institute of Neuroinformatics, University of Zurich and ETH
Zurich, 2018.
"""
import numpy as np
import os
import sys
import operator

from brian2 import SpikeGeneratorGroup, PoissonGroup
from brian2 import Network, SpikeMonitor
from brian2 import second, ms, Hz

def create_item(input_indices, isi, num_spikes):
    """
    Parameters
    ----------
    input_indices : list of int
        Indices of input channels that will be used to create item of a sequence
    isi : int
        Interspike interval of the spikes of item, in ms
    num_spikes : int
        Indicates how many spikes for the given item
    """
    indices = [x for x in input_indices]*num_spikes
    times = np.repeat([x for x in range(0, num_spikes*isi, isi)],
                      len(input_indices)).tolist()

    return {'indices': indices, 'times': times}

def create_sequence(items, intra_seq_dt):
    """
    Parameters
    ----------
    items : dict
        Contains indices and relative times (keys should be 'indices' and 'time',
        respectively) of items of the sequence to be created.
    intra_seq_dt : int
        Time gap between items in a sequence, in ms.
    """
    sequence_indices = []
    sequence_times = []
    last_t = 0
    for i in items:
        sequence_indices.extend(i['indices'])
        sequence_times.extend([x+last_t for x in i['times']])
        last_t = sequence_times[-1] + intra_seq_dt

    return {'indices': sequence_indices,
            'times': sequence_times}

def create_testbench(sequences, labels, occurences, inter_seq_dt, num_seq, silence=None):
    """
    Parameters
    ----------
    sequences : list
        Contains sequences to be used as testbench
    labels : list of int
        Identifies labels of each sequence or class. Its order must be set
        according to the order of the parameter sequences and must also have
        the same length. When occurences is not None, label elements represent
        the index of a given sequence in sequences.
    occurences: list of float
        Probabilities that each sequence has appear in the final testbench. If
        it is None, parameter sequences is assumed to contain entire dataset
        and it is therefore shuffled.
    inter_seq_dt : int
        Time gap between sequences, in ms
    num_seq : int
        Total number of sequences. If parameter occurence is None, this will
        not be used for anything.
    silence : dict
        Determines whether there are arbitrary gaps with arbitrary duration in
        the testbench. Must have keys 'iteration' (at which it happens)
        and 'duration'. It is added at the end of iteration.

    Returns
    -------
    testbench_indices : list of int
        Each element is a channel
    testbench_times : list of int
        Contains instant of an event, in ms
    events : list of lists
        Each nested list contains label, start time, and stop time. Units
        of time are in ms and labels are int.
    """
    testbench_indices = []
    testbench_times = []
    if not silence:
        silence = {'iteration': []}
    rng = np.random.default_rng()
    if occurences:
        labels = rng.choice(labels, num_seq, p=occurences)
        testbench = [sequences[x] for x in labels]
    else:
        id_perm = rng.permutation(range(len(labels)))
        labels = [labels[x] for x in id_perm]
        testbench = [sequences[x] for x in id_perm]
    events = []

    last_t = 0
    for i, seq in enumerate(testbench):
        events.append([labels[i]])
        testbench_indices.extend(seq['indices'])
        testbench_times.extend([x+last_t for x in seq['times']])
        events[-1].extend([last_t*ms, max(testbench_times)*ms])
        last_t = max(testbench_times) + inter_seq_dt
        if i in silence['iteration']:
            ind = silence['iteration'].index(i)
            last_t += silence['duration'][ind]

    return testbench_indices, testbench_times, events

def delete_doublets(spiketimes, indices, verbose=False):
    """
    Removes spikes that happen at the same time and at the same index.
    This happens when you donwnsample, but Brian2 cannot cope with more that 1 spike per ts.
    :param spiketimes: numpy array of spike times
    :param indices:  numpy array of indices
    :return: same as input but with removed doublets
    """
    len_before = len(spiketimes)
    buff_data = np.vstack((spiketimes, indices)).T
    buff_data[:, 0] = buff_data[:, 0].astype(int)
    _, idx = np.unique(buff_data, axis=0, return_index=True)
    buff_data = buff_data[np.sort(idx), :]

    spiketimes = buff_data[:, 0]
    indices = np.asarray(buff_data[:, 1], dtype=int)

    if verbose:
        print(len_before - len(spiketimes), 'spikes removed')
        print(100 - (len(spiketimes) / len_before * 100), '% spikes removed')
    return spiketimes, indices

class STDP_Testbench():
    """This class provides a stimulus to test your spike-timing dependent plasticity algorithm.

    Attributes:
        N (int): Size of the pre and post neuronal population.
        stimulus_length (int): Length of stimuli in ms.
    """

    def __init__(self, N=1, stimulus_length=1200):
        """Initializes the testbench class.

        Args:
            N (int, optional): Size of the pre and post neuronal population.
            stimulus_length (int, optional): Length of stimuli in ms.
        """
        self.N = N  # Number of Neurons per input group
        self.stimulus_length = stimulus_length

    def stimuli(self, isi=10):
        """Stimulus gneration for STDP protocols.

        This function returns two brian2 objects.
        Both are Spikegeneratorgroups which hold a single index each
        and varying spike times.
        The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
        strong LTD, homoeostasis.

        Args:
            isi (int, optional): Interspike Interval. How many spikes per stimulus phase.

        Returns:
            SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the spiketimes and
                the respective neuron indices.
        """
        t_pre_homoeotasis_1 = np.arange(1, 202, isi)
        t_pre_weakLTP = np.arange(301, 502, isi)
        t_pre_weakLTD = np.arange(601, 802, isi)
        t_pre_strongLTP = np.arange(901, 1102, isi)
        t_pre_strongLTD = np.arange(1201, 1402, isi)
        t_pre_homoeotasis_2 = np.arange(1501, 1702, isi)
        t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                           t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

        # Normal distributed shift of spike times to ensure homoeotasis
        t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
            np.clip(np.random.randn(len(t_pre_homoeotasis_1)), -1, 1)
        t_post_weakLTP = t_pre_weakLTP + 5   # post neuron spikes 7 ms after pre
        t_post_weakLTD = t_pre_weakLTD - 5   # post neuron spikes 7 ms before pre
        t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
        t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
        t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
            np.clip(np.random.randn(len(t_pre_homoeotasis_2)), -1, 1)

        t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                            t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
        ind_pre = np.zeros(len(t_pre))
        ind_post = np.zeros(len(t_post))

        pre = SpikeGeneratorGroup(
            self.N, indices=ind_pre, times=t_pre * ms, name='gPre')
        post = SpikeGeneratorGroup(
            self.N, indices=ind_post, times=t_post * ms, name='gPost')
        return pre, post


class OCTA_Testbench():
    """This class holds all relevant stimuli to test modules provided with the
    Online Clustering of Temporal Activity (OCTA) framework.

    Attributes:
        angles (numpy.ndarray): List of angles of orientation.
        DVS_SHAPE (TYPE): Input shape of the simulated DVS/DAVIS vision sensor.
        end (TYPE): End pixel location of the line.
        events (TYPE): Attribute storing events of testbench stimulus.
        indices (TYPE): Attribute storing neuron index of testbench stimulus.
        line (TYPE): Stimulus of the testbench which is used to either generate an interactive
            plot to record stimulus with a DVS/DAVIS camera or coordinates are used to generate
            a SpikeGenerator.
        start (TYPE): Start pixel location of the line.
        times (list): Attribute storing spike times of testbench stimulus.
    """

    def __init__(self, DVS_SHAPE=(240, 180)):
        """Summary

        Args:
            DVS_SHAPE (tuple, optional): Dimension of pixel array of the simulated DVS/DAVIS vision sensor.
        """
        self.DVS_SHAPE = DVS_SHAPE
        self.angles = np.arange(-np.pi / 2, np.pi * 3 / 2, 0.01)

    def aedat2events(self, rec, camera='DVS128'):
        """Wrapper function of the original aedat2numpy function in teili.tools.converter.

        This function will save events for later usage and will directly return them if no
        SpikeGeneratorGroup is needed.

        Args:
            rec (str): Path to stored .aedat file.
            camera (str, optional): Can either be string ('DAVIS240') or int 240, which specifies
                the larger of the 2 pixel dimension to unravel the coordinates into indices.

        Returns:
            events (np.ndarray): 4D numpy array with #events entries. Array is organized as x, y, ts, pol. See aedat2numpy for more details.
        """
        assert(type(rec) == str), "rec has to be a string."
        assert(os.path.isfile(rec)), "File does not exist."
        events = aedat2numpy(datafile=rec, camera=camera)
        np.save(rec[:-5] + 'npy', events)
        return events

    def infinity(self, cAngle):
        """Given an angle cAngle this function returns the current position on an infinity trajectory.

        Args:
            cAngle (float): current angle in rad which determines position on infinity trajectory.

        Returns:
            position (tuple): Postion in x, y coordinates.
        """
        return np.cos(cAngle), np.sin(cAngle) * np.cos(cAngle)

    def dda_round(self, x):
        """Simple round funcion.

        Args:
            x (float): Value to be rounded.

        Returns:
            (int): Ceiled value of x.
        """
        if type(x) is np.ndarray:
            return (x + 0.5).astype(int)
        else:
            return int(x + 0.5)

    def rotating_bar(self, length=10, nrows=10, ncols=None, direction='ccw', ts_offset=10,
                     angle_step=10, artifical_stimulus=True, rec_path=None, save_path=None,
                     noise_probability=None, repetitions=1, debug=False):
        """This function returns a single SpikeGeneratorGroup (Brian object).

        The purpose of this function is to provide a simple test stimulus.
        A bar is rotating in the center. The goal is to learn necessary
        spatio-temporal features of the moving bar and be able to make predictions
        about where the bar will move.

        Args:
            length (int): Length of the bar in pixel.
            nrows (int, optional): X-Axis size of the pixel array.
            ncols (int, optional): Y-Axis size of the pixel array.
            orientation (str): Orientation of the bar. Can either be 'vertical'
                or 'horizontal'.
            ts_offset (int): time between two pixel location.
            angle_step (int, optional): Angular velocity. Sets step width in np.arrange.
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file.
            rec_path (str, optional): Path/to/stored/location/of/recorded/stimulus.aedat.
            save_path (str, optional): Path to store generated events.
            noise_probability (float, optional): Probability of noise events between 0 and 1.
            repetitions (int, optional): Number of revolutions of the rotating bar.
            debug (bool, optional): Flag to print more detailed output of testbench.

        Returns:
            SpikeGenerator obj: Brian2 objects which holds the spike times as well
                as the respective neuron indices

        Raises:
            UserWarning: If no filename is given but aedat recording should be loaded
        """
        if ncols is None:
            ncols = nrows

        num_neurons = nrows * ncols

        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            assert(os.path.isfile(rec_path + 'bar.aedat')
                   ), "No recording exists. Please record a stimulus first."
            self.events = aedat2numpy(
                datafile=rec_path + 'bar.aedat', camera='DVS240')
        else:
            x_coord = []
            y_coord = []
            pol = []
            self.times = []
            repetition_offset = 0
            center = (nrows / 2, ncols / 2)
            self.angles = np.arange(-np.pi / 2, np.pi *
                                    3 / 2, np.radians(angle_step))
            if direction == 'cw':
                self.angles = np.flip(self.angles, axis=0)
            for repetition in range(repetitions):
                if repetition_offset != 0:
                    repetition_offset += ts_offset
                for i, cAngle in enumerate(self.angles):
                    endy_1 = center[1] + ((length / 2.)
                                          * np.sin((np.pi / 2 + cAngle)))
                    endx_1 = center[0] + ((length / 2.)
                                          * np.cos((np.pi / 2 + cAngle)))
                    endy_2 = center[1] - ((length / 2.)
                                          * np.sin((np.pi / 2 + cAngle)))
                    endx_2 = center[0] - ((length / 2.)
                                          * np.cos((np.pi / 2 + cAngle)))
                    self.start = np.asarray((endx_1, endy_1))
                    self.end = np.asarray((endx_2, endy_2))
                    self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)),
                                                              key=operator.itemgetter(1))
                    dv = (self.end - self.start) / self.max_length
                    self.line = [self.dda_round(self.start)]
                    for step in range(int(self.max_length)):
                        self.line.append(self.dda_round(
                            (step + 1) * dv + self.start))
                    list_of_coord = []
                    for coord in self.line:
                        list_of_coord.append((coord[0], coord[1]))
                    for coord in self.line:
                        if coord[0] >= nrows or coord[1] >= nrows:
                            if debug:
                                print("Coordinate larger than input space. x: {}, y: {}".format(
                                    coord[0], coord[1]))
                            continue
                        x_coord.append(coord[0])
                        y_coord.append(coord[1])
                        self.times.append(repetition_offset + (i * ts_offset))
                        pol.append(1)
                        if noise_probability is not None and noise_probability >= np.random.rand():
                            noise_index = np.random.randint(0, num_neurons)
                            noise_x, noise_y = ind2xy(
                                noise_index, nrows, ncols)
                            if (noise_x, noise_y) not in list_of_coord:
                                # print(noise_x, noise_y)
                                # print(list_of_coord)
                                list_of_coord.append((noise_x, noise_y))
                                x_coord.append(noise_x)
                                y_coord.append(noise_y)
                                self.times.append(
                                    repetition_offset + (i * ts_offset))
                                pol.append(1)
                repetition_offset = np.max(self.times)
            self.events = np.zeros((4, len(x_coord)))
            self.events[0, :] = np.asarray(x_coord)
            self.events[1, :] = np.asarray(y_coord)
            self.events[2, :] = np.asarray(self.times)
            self.events[3, :] = np.asarray(pol)
        if debug:
            print("Max X: {}. Max Y: {}".format(
                np.max(self.events[0, :]), np.max(self.events[1, :])))
            print("Stimulus last from {} ms to {} ms".format(
                np.min(self.events[2, :]), np.max(self.events[2, :])))
        if not artifical_stimulus:
            self.indices, self.times = dvs2ind(self.events, scale=False)
        else:
            self.indices = xy2ind(np.asarray(self.events[0, :], dtype='int'), np.asarray(self.events[
                                  1, :], dtype='int'), nrows, ncols)
            if debug:
                print("Maximum index: {}, minimum index: {}".format(
                    np.max(self.indices), np.min(self.indices)))
        nPixel = np.int(np.max(self.indices))
        gInpGroup = SpikeGeneratorGroup(
            nPixel + 1, indices=self.indices, times=self.times * ms, name='bar')
        return gInpGroup

    def translating_bar_infinity(self, length=10, nrows=64, ncols=None, orientation='vertical', shift=32,
                                 ts_offset=10, artifical_stimulus=True, rec_path=None,
                                 return_events=False):
        """
        This function will either load recorded DAVIS/DVS recordings or generate artificial events
        of a bar moving on an infinity trajectory with fixed orientation, i.e. no super-imposed rotation.
        In both cases, the events are provided to a SpikeGeneratorGroup which is returned.

        Args:
            length (int, optional): length of the bar in pixel.
            nrows (int, optional): X-Axis size of the pixel array.
            ncols (int, optional): Y-Axis size of the pixel array.
            orientation (str, optional): lag which determines if bar is orientated vertically or horizontally.
            shift (int, optional): offset in x where the stimulus will start.
            ts_offset (int, optional): Time in ms between consecutive pixels (stimulus velocity).
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file.
            rec_path (str, optional): Path/to/stored/location/of/recorded/stimulus.aedat.
            return_events (bool, optional): Flag to return events instead of SpikeGenerator.

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes.
            events (numpy.ndarray, optional): If return_events is set, events will be returned.

        Raises:
            UserWarning: If no filename is given but aedat recording should be loaded.

        """
        if ncols is None:
            ncols = nrows

        num_neurons = nrows * ncols
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            if orientation == 'vertical':
                fname = rec_path + 'Inifity_bar_vertical.aedat'
            elif orientation == 'horizontal':
                fname = 'Infinity_bar_horizontal.aedat'
            assert(os.path.isfile(
                fname)), "No recording exists. Please record a stimulus first."
            self.events = aedat2numpy(datafile=fname, camera='DVS240')
        else:
            x_coord = []
            y_coord = []
            pol = []
            self.times = []
            for i, cAngle in enumerate(self.angles):
                x, y = self.infinity(cAngle)
                if orientation == 'vertical':
                    endy_1 = shift + shift * y + \
                        ((length / 2) * np.sin(np.pi / 2))
                    endx_1 = shift + shift * x + \
                        ((length / 2) * np.cos(np.pi / 2))
                    endy_2 = shift + shift * y - \
                        ((length / 2) * np.sin(np.pi / 2))
                    endx_2 = shift + shift * x - \
                        ((length / 2) * np.cos(np.pi / 2))
                elif orientation == 'horizontal':
                    endy_1 = shift + shift * y + ((length / 2) * np.sin(np.pi))
                    endx_1 = shift + shift * x + ((length / 2) * np.cos(np.pi))
                    endy_2 = shift + shift * y - ((length / 2) * np.sin(np.pi))
                    endx_2 = shift + shift * x - ((length / 2) * np.cos(np.pi))
                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)),
                                                          key=operator.itemgetter(1))
                dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round(
                        (step + 1) * dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    self.times .append(i * ts_offset)
                    pol.append(1)

            self.events = np.zeros((4, len(x_coord)))
            self.events[0, :] = np.asarray(x_coord)
            self.events[1, :] = np.asarray(y_coord)
            self.events[2, :] = np.asarray(self.times)
            self.events[3, :] = np.asarray(pol)

        if return_events:
            return self.events
        else:
            if not artifical_stimulus:
                self.indices, self.times = dvs2ind(self.events, scale=False)
            else:
                self.indices = xy2ind(np.asarray(self.events[0, :], dtype='int'),
                                      np.asarray(
                                          self.events[1, :], dtype='int'),
                                      nrows, ncols)
                print(np.max(self.indices), np.min(self.indices))
            nPixel = np.int(np.max(self.indices))
            gInpGroup = SpikeGeneratorGroup(
                nPixel + 1, indices=self.indices, times=self.times * ms, name='bar')
            return gInpGroup

    def rotating_bar_infinity(self, length=10, nrows=64, ncols=None, orthogonal=False, shift=32,
                              ts_offset=10, artifical_stimulus=True, rec_path=None,
                              return_events=False):
        """This function will either load recorded DAVIS/DVS recordings or generate artificial events
        of a bar moving on an infinity trajectory with fixed orientation, i.e. no super-imposed rotation.
        In both cases, the events are provided to a SpikeGeneratorGroup which is returned.

        Args:
            length (int, optional): Length of the bar in pixel.
            nrows (int, optional): X-Axis size of the pixel array.
            ncols (int, optional): Y-Axis size of the pixel array.
            orthogonal (bool, optional): Flag which determines if bar is kept always orthogonal to trajectory,
                if it kept aligned with the trajectory or if it returns in a "chaotic" way.
            shift (int, optional): Offset in x where the stimulus will start.
            ts_offset (int, optional): Time in ms between consecutive pixels (stimulus velocity).
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file.
            rec_path (str, optional): Path/to/stored/location/of/recorded/stimulus.aedat.
            return_events (bool, optional): Flag to return events instead of SpikeGenerator.

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes.
            events (numpy.ndarray, optional): If return_events is set, events will be returned.

        Raises:
            UserWarning: If no filename is given but aedat recording should be loaded.
        """
        if ncols is None:
            ncols = nrows

        num_neurons = nrows * ncols
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            if orthogonal == 0:
                fname = rec_path + 'Inifity_aligned_bar.aedat'
            elif orthogonal == 1:
                fname = rec_path + 'Infinity_orthogonal_bar.aedat'
            elif orthogonal == 2:
                fname = rec_path + 'Infinity_orthogonal_aligned_bar.aedat'
            assert(os.path.isfile(
                fname)), "No recording exists. Please record a stimulus first."
            self.events = aedat2numpy(datafile=fname, camera='DVS240')
            return self.events
        else:
            x_coord = []
            y_coord = []
            pol = []
            self.times = []
            flipped_angles = self.angles[::-1]
            for i, cAngle in enumerate(self.angles):
                x, y = self.infinity(cAngle)
                if orthogonal == 1:
                    if x >= shift:
                        endy_1 = shift + shift * y + \
                            ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                        endx_1 = shift + shift * x + \
                            ((length / 2) * np.cos((np.pi / 2 * cAngle)))
                        endy_2 = shift + shift * y - \
                            ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                        endx_2 = shift + shift * x - \
                            ((length / 2) * np.cos((np.pi / 2 * cAngle)))

                    else:
                        endy_1 = shift + shift * y - \
                            ((length / 2) * np.sin(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                        endx_1 = shift + shift * x - \
                            ((length / 2) * np.cos(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                        endy_2 = shift + shift * y + \
                            ((length / 2) * np.sin(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                        endx_2 = shift + shift * x + \
                            ((length / 2) * np.cos(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                elif orthogonal == 0:
                    endy_1 = shift + shift * y + \
                        ((length / 2) * np.sin(np.pi / 2 + cAngle))
                    endx_1 = shift + shift * x + \
                        ((length / 2) * np.cos(np.pi / 2 + cAngle))
                    endy_2 = shift + shift * y - \
                        ((length / 2) * np.sin(np.pi / 2 + cAngle))
                    endx_2 = shift + shift * x - \
                        ((length / 2) * np.cos(np.pi / 2 + cAngle))

                elif orthogonal == 2:
                    endy_1 = shift + shift * y + \
                        ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                    endx_1 = shift + shift * x + \
                        ((length / 2) * np.cos((np.pi / 2 * cAngle)))
                    endy_2 = shift + shift * y - \
                        ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                    endx_2 = shift + shift * x - \
                        ((length / 2) * np.cos((np.pi / 2 * cAngle)))

                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(
                    enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
                dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round(
                        (step + 1) * dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    self.times.append(i * ts_offset)
                    pol.append(1)

            self.events = np.zeros((4, len(x_coord)))
            self.events[0, :] = np.asarray(x_coord)
            self.events[1, :] = np.asarray(y_coord)
            self.events[2, :] = np.asarray(self.times)
            self.events[3, :] = np.asarray(pol)

        if return_events:
            return self.events
        else:
            if not artifical_stimulus:
                self.indices, self.times = dvs2ind(self.events, scale=False)
            else:
                self.indices = xy2ind(np.asarray(self.events[0, :], dtype='int'),
                                      np.asarray(
                                          self.events[1, :], dtype='int'),
                                      nrows, ncols)
            nPixel = np.int(np.max(self.indices))
            gInpGroup = SpikeGeneratorGroup(
                nPixel + 1, indices=self.indices, times=self.times * ms, name='bar')
            return gInpGroup

    def ball(self, rec_path):
        '''
        This function loads a simple recording of a ball moving in a small arena.
        The idea is to test the Online Clustering and Prediction module of OCTAPUS.
        The aim is to learn spatio-temporal features based on the ball's trajectory
        and learn to predict its movement.

        Args:
            rec_path (str, required): Path to recording.

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes

        Raises:
            UserWarning: If no filename is given but aedat reacording should be loaded

        '''
        if rec_path is None:
            raise UserWarning('No path to recording was provided')
        fname = rec_path + 'ball.aedat'
        assert(os.path.isfile(fname)), "No recording ball.aedat exists in {}. Please use jAER to record the stimulus and save it as ball.aedat in {}".format(
            rec_path, rec_path)
        events = aedat2numpy(datafile=fname, camera='DVS240')
        ind_on, ts_on, ind_off, ts_off = dvs2ind(
            Events=events, resolution=max(self.DVS_SHAPE), scale=True)
        # depending on how long conversion to index takes we might need to
        # savbe this as well
        input_on = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                       indices=ind_on, times=ts_on, name='input_on*')
        input_off = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                        indices=ind_off, times=ts_off, name='input_off*')
        return input_on, input_off


class SequenceTestbench():
    """ This class provides a simple poisson or deterministic encoded sequence testbench.
    This class returns neuron indices and spike times, which can used to
    instantiate a SpikeGeneratorGroup in brian2."""

    def __init__(self, n_channels, n_items,
                 item_length, superposition_length=0,
                 noise_probability=None, rate=None, cycle_repetitions=None,
                 surprise_item=False,
                 deterministic=False):
        """ Creates arrays of neuron index and spike times for
        a simple sequence learning benchmark. The sequence consists of
        a number of items which are encoded in spatially distinct input
        channels.

        Args:
            n_channels (int, required): Total numbers of channels.
            n_items (in, required): Number of items. The number of channels
                per item is determined automatically based on the available
                number of channels.
            item_length (int, required): The length in ms of a single item.
            superposition_length (int, optional): The duration of superposition
                in ms between each symbol.
            noise_probability (float, optional): Probability of noise spikes
                per time step (ms).
            rate (int, optional): Frequency of each item. Default is 30.
            cycle_repetitions=(int, optional): Number of times the cycle is
                repeated.
            surprise_item (boolean, optional): Determines that last item is
                not added in the sequence.
            deterministic (bool): Defines if spikes will be deterministic. If 
                not, they will be poissonian.

        Returns:
            indices (1darray, int): An array of neuron indices.
            times (1darray, int): An array of spike times in seconds.
        """
        self.n_channels = n_channels
        self.surprise_item = surprise_item
        self.n_items = n_items - 1 if self.surprise_item else n_items
        self.noise_probability = noise_probability
        self.rate = rate
        self.item_length = item_length
        self.cycle_length = item_length * n_items - superposition_length * (n_items-1)
        self.cycle_repetitions = cycle_repetitions
        self.superposition_length = superposition_length
        self.channels_per_item = n_channels / n_items
        self.deterministic = deterministic

    def create_poisson_items(self):
        """ This function creates the Poisson distributed items with the
        specificed rate.
        """
        if self.rate is None:
            self.rate = 30

        P = PoissonGroup(N=self.n_channels, rates=self.rate*Hz)
        monitor = SpikeMonitor(P, record=True)
        net = Network()
        net.add(P, monitor)
        net.run(duration=(self.cycle_length) * ms)
        # Use integer values to avoid floating point errors
        self.monitor_t = np.around(monitor.t/ms).astype(int)
        self.monitor_i = monitor.i

    def create_deterministic_items(self):
        """ This function creates deterministic items with the
        specificed rate.
        """
        # Use integer values to avoid floating point errors
        period = 1/self.rate*1000 # Considering ms
        self.monitor_t = np.repeat(
            np.arange(0, self.cycle_length, period), self.n_channels)
        self.monitor_i = np.repeat(
            [np.arange(0, self.n_channels)],
            len(self.monitor_t)/self.n_channels,
            axis=0).flatten()

    def add_noise(self):
        """
        This function adds noise spike given the noise_probability.
        """

        noise_spikes = np.random.rand(self.n_channels,
                          (self.cycle_length*self.cycle_repetitions
                           - self.superposition_length*(self.cycle_repetitions - 1))
                          )

        self.noise_indices = np.where(noise_spikes < self.noise_probability)[0]
        self.noise_times = np.where(noise_spikes < self.noise_probability)[1]

    def repeate_cycle(self):
        """
        This functions replicates same items throughout simulation.
        """
        spike_times, spike_indices = [], []
        init_time = self.cycle_length - self.superposition_length
        if self.surprise_item:
            init_time -= self.item_length
        for rep in range(self.cycle_repetitions):
            spike_indices.extend(self.indices)
            aux_t = [(x + rep*init_time) for x in self.times]
            spike_times.extend(aux_t)
        self.indices = np.array(spike_indices)
        self.times = np.array(spike_times)

    def sort_spikes(self):
        """ Sort spike indices according to spike times."""
        sorting_index = np.argsort(self.times)
        self.indices = self.indices[sorting_index]
        self.times = self.times[sorting_index]

    def stimuli(self):
        """ This function creates the stimuli and returns neuron indices and
        spike times.
        """

        if self.deterministic:
            self.create_deterministic_items()
        else:
            self.create_poisson_items()

        t_start = 0
        i_start = 0
        self.items = {}
        for item in range(self.n_items):
            t_stop = t_start + self.item_length
            i_stop = i_start + self.channels_per_item

            cInd = np.logical_and(np.logical_and(np.logical_and(
                self.monitor_t >= t_start, self.monitor_i >= i_start),
                self.monitor_t < t_stop), self.monitor_i < i_stop)

            t_start += self.item_length - self.superposition_length
            i_start += self.channels_per_item
            item_indices = np.asarray(self.monitor_i[cInd])
            item_times = np.asarray(self.monitor_t[cInd])
            self.items[item] = {'t': item_times, 'i': item_indices}
            try:
                self.indices = np.concatenate((self.indices, item_indices))
                self.times = np.concatenate((self.times, item_times))
            except AttributeError:
                self.indices = item_indices
                self.times = item_times
        if self.surprise_item:
            item += 1
            t_stop = t_start + self.item_length
            i_stop = i_start + self.channels_per_item

            cInd = np.logical_and(np.logical_and(np.logical_and(
                self.monitor_t >= t_start, self.monitor_i >= i_start),
                self.monitor_t < t_stop), self.monitor_i < i_stop)

            t_start += self.item_length - self.superposition_length
            i_start += self.channels_per_item
            item_indices = np.asarray(self.monitor_i[cInd])
            item_times = np.asarray(self.monitor_t[cInd])
            self.items[item] = {'t': item_times, 'i': item_indices}

        if self.cycle_repetitions is not None:
            self.repeate_cycle()

        if self.noise_probability is not None:
            self.add_noise()
            self.indices = np.concatenate((self.indices, self.noise_indices))
            self.times = np.concatenate((self.times, self.noise_times))
        self.sort_spikes()

        self.times, self.indices = delete_doublets(self.times, self.indices)
        return self.indices, self.times*ms
