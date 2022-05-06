from matplotlib import pyplot as plt

from sarracen.interpolate import interpolate2D
from sarracen.kernels import BaseKernel


def render(data: 'SarracenDataFrame',
           x: str,
           y: str,
           target: str,
           kernel: 'BaseKernel',
           xmin: float = 0,
           ymin: float = 0,
           xmax: float = 1,
           ymax: float = 1,
           pixcountx: int = 480,
           pixcounty: int = 480):
    pixwidthx = (xmax - xmin) / pixcountx
    pixwidthy = (ymax - ymin) / pixcounty
    image = interpolate2D(data, x, y, target, kernel, pixwidthx, pixwidthy, xmin, ymin, pixcountx, pixcounty)

    # this figsize approximation seems to work well enough in most cases
    fig, ax = plt.subplots(figsize=(4, 3*((ymax - ymin) / (xmax - xmin))))
    img = ax.imshow(image, cmap='RdBu', origin='lower', extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(target)

    return fig, ax
