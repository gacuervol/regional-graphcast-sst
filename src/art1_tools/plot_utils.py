# import matplotlib.colors as mcolors
import datetime
import numpy as np
import xarray as xr
import IPython 
from typing import Optional
from IPython.display import HTML
from matplotlib import animation
# import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def scale(
    data: xr.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xr.Dataset, colors.Normalize, str]:
    """
    Escala los datos para la visualización.

    Args:
        data (xr.Dataset): El conjunto de datos a escalar.
        center (Optional[float], optional): El centro para la escala. Por defecto es None.
        robust (bool, optional): Indica si se debe utilizar una escala robusta. Por defecto es False.

    Returns:
        tuple[xr.Dataset, matplotlib.colors.Normalize, str]: Una tupla que contiene los datos escalados, 
        el objeto de normalización y el mapa de colores.

    """
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, colors.Normalize(vmin, vmax),
            ("RdBu_r" if center is not None else "viridis"))

def select(
    data: xr.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
) -> xr.Dataset:
    """
    Seleccione y filtre datos de un conjunto de datos Xr.

    Args:
        data (xr.Dataset): El conjunto de datos Xr del cual seleccionar.
        variable (str): El nombre de la variable a seleccionar.
        level (Optional[int], optional): El nivel específico a seleccionar (si corresponde). Por defecto es None.
        max_steps (Optional[int], optional): El número máximo de pasos de tiempo a seleccionar. Por defecto es None.

    Returns:
        xr.Dataset: El conjunto de datos seleccionado.

    """
    if type(data) == xr.DataArray:
        data = data.to_dataset()
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

# nb_path = IPython.extract_module_locals()[1]['__vsc_ipynb_file__']
#n_nb_test = re.search(r'test_(\d+)', nb_path).group(1)
# import matplotlib as mpl
"""class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))"""
def plot_data(
    data: dict[str, xr.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    ani_to_save: bool = False,
    device: int = None
) -> tuple[xr.Dataset, colors.Normalize, str]:
    """
    Visualiza los datos en un gráfico de animación.

    Args:
        data (dict[str, xr.Dataset]): Un diccionario de datos para visualizar.
        fig_title (str): El título de la figura.
        plot_size (float, optional): El tamaño de la trama. Por defecto es 5.
        robust (bool, optional): Indica si se debe utilizar una escala robusta. Por defecto es False.
        cols (int, optional): El número de columnas en el diseño de la trama. Por defecto es 4.

    Returns:
        tuple[xr.Dataset, matplotlib.colors.Normalize, str]: Una tupla que contiene los datos escalados, 
        el objeto de normalización y el mapa de colores.

    """
    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1)
               for d, _, _ in data.values())

    if all('device' in d.dims
            for d, _, _ in data.values()):
                assert device != None, f"You have to choose a device, currently {device=}"
                for k in data.keys():
                    data[k] = (data[k][0].sel(device=device), data[k][1], data[k][2])

    cols = min(cols, len(data))
    rows = int(np.ceil(len(data) / cols))
    # figure = plt.figure(figsize=(plot_size * 2 * cols,
    #                              plot_size * rows))
    # figure = plt.figure(figsize=(15, 7))
    figure = plt.figure(figsize=(plot_size * cols, plot_size * 0.85 * rows))
    figure.suptitle(fig_title, fontsize=16)
    #figure.subplots_adjust(wspace=0.05, hspace=0)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=0.01, hspace=0.15)
    # figure.tight_layout()
    if list(data.keys()) != (empty_keys := [' ']):
        vmin_value, vmax_value = data['Diff: Targets - Predictions'][0].min(), data['Diff: Targets - Predictions'][0].max()
        if vmin_value > 0: 
            vmin_value = -0.001
        if vmax_value < 0:
            vmax_value = 0.001
    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1,)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        if title == 'Diff: Targets - Predictions':
            # norm.vmin, norm.vmax = vmin_value, vmax_value
            norm = colors.TwoSlopeNorm(vmin=vmin_value, vcenter=0, vmax=vmax_value)
            # norm = MidpointNormalize(vmin=vmin_value, vmax=vmax_value, midpoint=0)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), 
            norm=norm,
            origin="lower", cmap=cmap, aspect='equal')
        plt.colorbar(
                mappable=im,
                ax=ax,
                orientation="vertical",
                pad=0.01,
                aspect=20,
                shrink=0.6,
                cmap=cmap,
                norm=norm,
                #extend='max', 
                #spacing='proportional',
                extend=("both" if robust else "neither"),
                ).ax.set_yscale('linear')
        images.append(im)

    def update(frame):
            
        """
        Actualiza los datos mostrados en la animación.

        Args:
            frame (int): El número del frame a mostrar.

        """
        if "time" in first_data.dims:
            t_0 = datetime.timedelta(
                microseconds=first_data["time"][0].item() / 1000)
            td = datetime.timedelta(
                microseconds=first_data["time"][frame].item() / 1000)

            figure.suptitle(f"{fig_title}, {td - t_0} days", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

    ani = animation.FuncAnimation(fig=figure, func=update, frames=max_steps, interval=250)
    #folder = ""
    #folder_linux = ""
    #ani.save(filename=folder_linux + f"test_{n_nb_test}.gif" if sys.platform.startswith("linux") else folder + f"test_{n_nb_test}.gif", 
    #         writer="pillow",)
    plt.close(figure.number)

    return (ani if ani_to_save else HTML(ani.to_jshtml()))

def create_animation(targets, use_norm=False):
    """
    Crea una animación de los datos de predicción.

    Args:
    - targets (np.ndarray): Datos de predicción con forma (time, height, width).

    Returns:
    - HTML: Animación en formato HTML.
    """
    time_steps = targets.shape[0]

    # Crear la figura y los ejes
    vmin_value, vmax_value = (
        np.nanmin(targets), 
        np.nanmax(targets)
        )
    if vmin_value > 0: 
            vmin_value = -0.001

    norm = colors.TwoSlopeNorm(
        vmin=vmin_value, 
        vcenter=0, 
        vmax=vmax_value,
        ) if use_norm else None

    fig, ax = plt.subplots()
    im = ax.imshow(
        targets[0, :, :], 
        cmap='RdBu_r', 
        aspect='auto', 
        norm=norm,
        origin='lower',
        )
    plt.colorbar(im, ax=ax, 
    norm=norm
    )
    title = ax.set_title(f'Epoch 0')

    # Función de actualización para la animación
    def update(frame):
        im.set_array(targets[frame, :, :])
        title.set_text(f'Epoch {frame}')
        return im, title

    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=time_steps, blit=True)

    # Renderizar la animación en HTML
    html_anim = HTML(ani.to_jshtml())
    plt.close(fig)  # Cerrar la figura para evitar que se muestre estáticamente

    return html_anim


class BarrierPlotDataExtractor:
    def __init__(self, model_rmse: xr.Dataset, n_lead_times: int) -> None:
        self.model_rmse = model_rmse
        self.LEAD_TIMES = n_lead_times
        self.WINDOW_STEP = 1
        self.dates_coord = (self.get_dates_coord())
        
    def __call__(self):
        empty_arr_to_fill = self.compute_empty_arr()
        for i_batch, date in enumerate(self.dates_coord):
            is_in_dates = (self.model_rmse.datetime == date)
            rmse_for_that_date = (
                self.model_rmse.where(is_in_dates, drop=True)
                .mean(dim=['batch'], skipna=True)
                )
            i_lead_time = rmse_for_that_date.time.data.astype('timedelta64[D]').astype('int')
            row = i_batch
            for col, i_rmse in zip(i_lead_time, rmse_for_that_date.data):
                empty_arr_to_fill[row, col] = i_rmse
        
        return xr.DataArray(
            empty_arr_to_fill,
            dims=['date', 'time'],
            coords={
                'date': self.dates_coord,
                'time': np.arange(1, self.LEAD_TIMES + 1, self.WINDOW_STEP)
                }
            )

    def get_dates_coord(self):
        min_date = self.model_rmse.datetime.min().data.astype('datetime64[D]')
        max_date = self.model_rmse.datetime.max().data.astype('datetime64[D]')
        
        return np.arange(min_date, max_date)

    def compute_empty_arr(self):
        empty_arr = np.empty((len(self.dates_coord), self.LEAD_TIMES//self.WINDOW_STEP))
        empty_arr[:] = np.nan

        return empty_arr