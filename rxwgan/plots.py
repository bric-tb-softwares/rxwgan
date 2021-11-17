

__all__ = ["plot_evolution"]

import matplotlib.pyplot as plt

def plot_evolution( history, output, notebook=False ):

    epochs = list(range(len(history['train_discr_loss'])))
    xmin = epochs[0] - 20
    xmax = epochs[-1] + 20

    #fig, ax = plt.subplots( 2, 1, figsize=(10, 5), sharex=True)
    #ax[0].plot( epochs, history['train_discr_loss'], label='train', color='red')
    #ax[0].plot( epochs, history['val_discr_loss'], label='val', color='lightcoral')
    #ax[0].set_ylabel('Critic Loss',fontsize=18,loc='top')
    #ax[0].set_xlim(xmin,xmax)
    #ax[0].legend()
    #ax[1].plot( epochs, history['train_gen_loss'], label='train', color='black')
    #ax[1].plot( epochs, history['val_gen_loss'], label='val', color='gray')
    #ax[1].set_ylabel('Generator Loss',fontsize=18,loc='top')
    #ax[1].set_xlabel("# Epochs",fontsize=18,loc='right')
    #ax[1].set_xlim(xmin,xmax)
    #ax[1].legend()

    fig = plt.figure(figsize=(10, 5))
    plt.plot( epochs, history['train_discr_loss'], label='train critit', color='red')
    plt.plot( epochs, history['val_discr_loss'], label='val critic', color='lightcoral')
    plt.plot( epochs, history['train_gen_loss'], label='train gen', color='black')
    plt.plot( epochs, history['val_gen_loss'], label='val gen', color='gray')
    plt.ylabel('Loss',fontsize=18,loc='top')
    plt.xlabel("# Epochs",fontsize=18,loc='right')
    ax = plt.gca()
    ax.set_xlim(xmin,xmax)
    plt.legend()

    if notebook:
      plt.show()
    fig.savefig(output)


