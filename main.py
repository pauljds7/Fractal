import math

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors, pyplot, rcParams
import tkinter as tk

import utils as util
import fractal as fractal


#default keys
NAME = 0
CC_ESTIMATE = 1

class ProgramGui():
    def __init__(self, window_name = "Fractal", limits = (-2.0, 1.0, -1.0, 1.0), image_dims = (800, 600), info_panel_height_px = 100, device_info = None, bg_color = 'grey'):
        self.root = tk.Tk()
        self.root.title(window_name)

        #Gets user screen DPI as per https://stackoverflow.com/questions/54271887/calculate-screen-dpi
        self.screen_dpi = int(self.root.winfo_fpixels('1i'))
        rcParams['figure.dpi'] = self.screen_dpi #Setting future DPI of our figure/image
        self.inch_per_pixels = 1.0 / self.screen_dpi #Inces per user screen pixel, for drawing image in mpl
        self.window_size_px = (image_dims[0], image_dims[1]+info_panel_height_px) #Window size based on iamge size + padding for info panel
        self.root.geometry(str(self.window_size_px[0]) + 'x' + str(self.window_size_px[1])) #
        self.img_size_inch = (math.ceil(image_dims[0] * self.inch_per_pixels*100)/100, image_dims[1] * self.inch_per_pixels)
        self.color_map = mpl.colormaps['magma'].resampled(8)


        self.image_dims = image_dims
        self.limits = limits
        self.info_panel_height_px = info_panel_height_px

        #
        #Setup initial GUI elements, their places and initial values
        #
        #First figure to hold fractal image(in axis image form) and canvas to draw it
        self.fig = pyplot.Figure(frameon=False, figsize=self.img_size_inch)
        self.axes = pyplot.Axes(self.fig, [0., 0., 1., 1.])
        self.axes.set_axis_off()
        self.fig.add_axes(self.axes)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        #Three frames to hold info/interaction:
        #1 - general info for GPU info
        #2 - calculation info/interaction
        #3 - bounds info/interaction info for showing coordiante region bounds etc
        self.frm_info = tk.Frame(master=self.root, bg=bg_color, height=self.info_panel_height_px)
        self.frm_calculations = tk.Frame(master = self.root, bg=bg_color, height=self.info_panel_height_px)
        self.frm_bounds = tk.Frame(master=self.root, bg=bg_color, height=self.info_panel_height_px)

        #First GPU info panel
        self.frm_info_info = tk.Frame(master=self.frm_info)
        self.frm_info_title = tk.Label(master=self.frm_info, text='Current GPU')

        self.frm_info_gpu_name_name = tk.Label(master=self.frm_info_info, text='Name', anchor=tk.W)
        self.frm_info_gpu_name_value = tk.Label(master=self.frm_info_info, text=device_info[NAME], anchor=tk.W)
        self.frm_info_gpu_cc_name = tk.Label(master=self.frm_info_info, text='Cuda cores', anchor=tk.W)
        self.frm_info_gpu_cc_value = tk.Label(master=self.frm_info_info, text=str(device_info[CC_ESTIMATE]), anchor=tk.W)
        self.frm_info_gpu_name_name.grid(row=0, column=0, sticky=tk.NSEW)
        self.frm_info_gpu_name_value.grid(row=0, column=1, sticky=tk.NSEW)
        self.frm_info_gpu_cc_name.grid(row=1, column=0, sticky=tk.NSEW)
        self.frm_info_gpu_cc_value.grid(row=1, column=1, sticky=tk.NSEW)

        self.frm_info_title.grid(row=0)
        self.frm_info_info.grid(row=1, sticky='we')

        #Second calculation info panel/frame
        self.frm_calc_info = tk.Frame(master = self.frm_calculations)
        self.frm_calc_info_title = tk.Label(master = self.frm_calculations, text='Calculations')

        self.frm_calc_time_title = tk.Label(master = self.frm_calc_info, text='Time', anchor=tk.W)
        self.frm_calc_time_value = tk.Label(master = self.frm_calc_info, text='<NA>'+'ms', anchor=tk.W)
        self.frm_calc_time_title.grid(row = 0, column = 0, sticky=tk.NSEW)
        self.frm_calc_time_value.grid(row = 0, column = 1, sticky=tk.NSEW)

        self.frm_calc_info_title.grid(row = 0, column=0)
        self.frm_calc_info.grid(row = 1, column = 0, sticky='we')

        #Third coordinate grid info panel/frame building
        self.frm_bounds_info = tk.Frame(master = self.frm_bounds)
        self.frm_bounds_info_title = tk.Label(master = self.frm_bounds, text='Bounds')

        self.frm_bounds_x_info_title = tk.Label(master = self.frm_bounds_info, text='Real', anchor=tk.W)
        self.frm_bounds_x_info_value = tk.Label(master = self.frm_bounds_info, text='(' + str(limits[0]) + ';' + str(limits[1]) + ')', anchor=tk.W)
        self.frm_bounds_x_info_title.grid(row=0,column=0, sticky=tk.NSEW)
        self.frm_bounds_x_info_value.grid(row=0, column=1, sticky=tk.NSEW)

        self.frm_bounds_y_info_title = tk.Label(master = self.frm_bounds_info, text='Imag', anchor=tk.W)
        self.frm_bounds_y_info_value_upper = tk.Label(master = self.frm_bounds_info, text=str(limits[3]), anchor=tk.W)
        self.frm_bounds_y_info_value_lower = tk.Label(master=self.frm_bounds_info, text=str(limits[2]), anchor=tk.W)
        self.frm_bounds_y_info_title.grid(row=1, column=0, rowspan=2, sticky=tk.NSEW)
        self.frm_bounds_y_info_value_upper.grid(row=1, column=1, sticky=tk.NSEW)
        self.frm_bounds_y_info_value_lower.grid(row=2, column=1, sticky=tk.NSEW)

        self.frm_bounds_info_title.grid(row = 0, column = 0)
        self.frm_bounds_info.grid(row = 1, column = 0)

        #General placing end
        self.frm_info.grid(row=0, column=0, sticky=tk.NSEW)
        self.frm_calculations.grid(row=0, column=1, sticky=tk.NSEW)
        self.frm_bounds.grid(row=0, column=2, sticky=tk.NSEW)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, sticky='s')

        #Calculate and draw the result in separete function
        self.calculate_draw()

        #TKinter mainloop init
        tk.mainloop()


    def calculate_draw(self):
        """
        Calculates and redraws the fractal image and updates the GUI window
        :return: None
        """
        results = fractal.get_fractal_data(iters=100, dims=self.image_dims, bounds=self.limits)
        result = np.transpose(results[fractal.CALC_RESULT])
        print(results[fractal.CALC_TIME], 'ms', sep='')
        self.frm_calc_time_value['text'] = str(round(results[fractal.CALC_TIME], 2))+'ms'
        self.axes.imshow(result, cmap=self.color_map, norm=colors.PowerNorm(gamma=0.90), aspect='equal',
                    interpolation='gaussian')
        self.canvas.draw()




def launch_program():
    """
    Launches the program
    :return: None
    """
    device = util.get_current_card()
    info = util.get_card_info(device)
    gui = ProgramGui(device_info={NAME:info[util.NAME],
                                  CC_ESTIMATE: info[util.CC_COUNT_ESTIMATE]
                                  })


if __name__ == '__main__':
    launch_program()