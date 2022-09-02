import numpy as np
import sys
import matplotlib.pyplot as plt

def pad_matrices(n_rows, n_cols, matrix, target_ids):
    """ 
    
    Attributes:
        matrix (nd.array): weight matrix as provided by Brian2
            group. In case a monitor is provided, second dimension
            must represent time.
    """
    if len(np.shape(matrix)) == 1:
        matrix = np.reshape(matrix, (-1, 1))
    pad_mat = np.zeros((n_rows, n_cols, np.shape(matrix)[-1]))
    ref_id = 0
    for neu_id, targets in enumerate(target_ids):
        pad_mat[neu_id, targets, :] = matrix[ref_id:ref_id+len(targets), :]
        ref_id += len(targets)

    return pad_mat

def plot_weight_matrix(weight_matrix, title, xlabel, ylabel):
    # Plot matrices
    # Inferno colormap
    colors = [
        (0, 0, 4),
        (40, 11, 84),
        (101, 21, 110),
        (159, 42, 99),
        (212, 72, 66),
        (245, 125, 21),
        (250, 193, 39),
        (252, 255, 16)
    ]
    plt.imshow(weight_matrix)
    plt.show()
    #cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 8), color=colors)

    #win = QtGui.QMainWindow()
    #win.setWindowTitle(title)
    #image_axis = pg.PlotItem()
    #image_axis.setLabel(axis='bottom', text=xlabel)
    #image_axis.setLabel(axis='left', text=ylabel)
    ##image_axis.hideAxis('left')
    #imv = pg.ImageView(view=image_axis)
    #win.setCentralWidget(imv)
    #win.show()
    #imv.ui.histogram.hide()
    #imv.ui.roiBtn.hide()
    #imv.ui.menuBtn.hide() 
    #if len(np.shape(weight_matrix)) == 2:
    #    imv.setImage(weight_matrix, axes={'y':0, 'x':1})
    #elif len(np.shape(weight_matrix)) == 3:
    #    imv.setImage(weight_matrix, axes={'y':0, 'x':1, 't': 2})
    #imv.setColorMap(cmap)

    #return win

def raster_sort(mon, permutation_ids):
    sorted_i = np.asarray([np.where(np.asarray(permutation_ids) == int(i))[0][0] for i in mon.i])
    return sorted_i

def plot_all_w(conn, num, ch):
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(np.sqrt(num).astype(int), np.sqrt(num).astype(int)),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for idx in range(num):
        grid[idx].imshow(np.reshape(conn.w_plast[:, idx], (np.sqrt(ch).astype(int), np.sqrt(ch).astype(int))))
    plt.show()

def plot_EI_balance(idx=None, win_len=None, limits=None):
    if limits is None:
        limits = [max(statemon_net_current.t)/defaultclock.dt - 500, max(statemon_net_current.t)/defaultclock.dt]

    if idx is not None:
        if win_len:
            iin0 = np.convolve(statemon_net_current.Iin0[idx], np.ones(win_len)/win_len, mode='valid')
            iin1 = np.convolve(statemon_net_current.Iin1[idx], np.ones(win_len)/win_len, mode='valid')
            iin2 = np.convolve(statemon_net_current.Iin2[idx], np.ones(win_len)/win_len, mode='valid')
            iin3 = np.convolve(statemon_net_current.Iin3[idx], np.ones(win_len)/win_len, mode='valid')
            total_Iin = np.convolve(statemon_net_current.Iin[idx], np.ones(win_len)/win_len, mode='valid')
        else:
            iin0 = statemon_net_current.Iin0[idx]
            iin1 = statemon_net_current.Iin1[idx]
            iin2 = statemon_net_current.Iin2[idx]
            iin3 = statemon_net_current.Iin3[idx]
            total_Iin = statemon_net_current.Iin[idx]
    else:
        if win_len:
            iin0 = np.convolve(np.mean(statemon_net_current.Iin0, axis=0), np.ones(win_len)/win_len, mode='valid')
            iin1 = np.convolve(np.mean(statemon_net_current.Iin1, axis=0), np.ones(win_len)/win_len, mode='valid')
            iin2 = np.convolve(np.mean(statemon_net_current.Iin2, axis=0), np.ones(win_len)/win_len, mode='valid')
            iin3 = np.convolve(np.mean(statemon_net_current.Iin3, axis=0), np.ones(win_len)/win_len, mode='valid')
            total_Iin = np.convolve(np.mean(statemon_net_current.Iin, axis=0), np.ones(win_len)/win_len, mode='valid')
        else:
            iin0 = np.mean(statemon_net_current.Iin0, axis=0)
            iin1 = np.mean(statemon_net_current.Iin1, axis=0)
            iin2 = np.mean(statemon_net_current.Iin2, axis=0)
            iin3 = np.mean(statemon_net_current.Iin3, axis=0)
            total_Iin = np.mean(statemon_net_current.Iin, axis=0)
    plt.plot(iin0, 'r', label='pyr')
    plt.plot(iin1, 'g', label='pv')
    plt.plot(iin2, 'b', label='sst')
    plt.plot(iin3, 'k--', label='input')
    plt.plot(total_Iin, 'k', label='net current')
    plt.legend()
    if limits is not None:
        plt.xlim(limits)
    plt.ylabel('Current [amp]')
    plt.xlabel('time [ms]')
    plt.title('EI balance')
    plt.show()
