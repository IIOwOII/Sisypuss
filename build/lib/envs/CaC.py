#%%
import pygame as pg

import numpy as np
import pickle as pkl

import copy
import os

#%% Util Functions
def Util_find_directory():
    return os.path.split(os.path.abspath(__file__))[0]

def Util_load_image(file):
    """loads an image, prepares it for play"""
    file = os.path.join(Util_find_directory(), 'Data', file)
    try:
        surface = pg.image.load(file)
    except:
        pg.quit()
        raise SystemExit(f'Could not load image "{file}"')
    return surface.convert_alpha()

#%%
class Env_CaC():
    def __init__(self, TPS=20, resolution=(1920,1080),
                 grid_offset=(0,0), grid_size=16, grid_number=(33,33)):
        # Components
        self.TPS = TPS # Tick Rate (Default is 20)
        self.resolution = resolution
        
        # Components (Grid)
        self.grid_offset = grid_offset
        self.grid_size = grid_size
        self.grid_number = grid_number
        
        #
        self.init_render()
    
    def init_render(self):
        pg.init()
        pg.display.set_caption('Chasing and Chased')
        self.screen = pg.display.set_mode(self.resolution)
        self.clock = pg.time.Clock()
    
    def render(self):
        self.clock.tick(self.TPS)
        pg.display.update()

class CaC_Role():
    def __init__(self, movement_speed=0.5, speed_factor=1):
        #
        self.v_m = movement_speed
        self.v_f = speed_factor