# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:50:18 2020

@author: Haoran6
"""

from argparse import ArgumentParser
import h5py
from itertools import count
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input
import pickle
import torch

import cv2
import numpy as np
import os
from os.path import isfile, join
import os
import glob

    
def verts_animation(verts, jpg_dir=None):
    verts = verts.detach().cpu().numpy()
    f = open('./body_models/smpl/male/model.pkl', 'rb')
    params = pickle.load(f)
    faces = params['f']
    pd = tvtk.PolyData(points=verts[0], polys=faces)
    normals = tvtk.PolyDataNormals()
    configure_input(normals, pd)
    
    
    mapper = tvtk.PolyDataMapper()
    configure_input(mapper, normals)
    actor = tvtk.Actor(mapper=mapper)
    actor.property.set(edge_color=(0.5, 0.5, 0.5), ambient=0.0,
                       specular=0.15, specular_power=128., shading=True, diffuse=0.8)

    fig = mlab.figure(bgcolor=(1,1,1))
    fig.scene.add_actor(actor)
    
    @mlab.animate(delay=20, ui=False)
    def animation():
        for i in count():
            frame = i % len(verts)
            pd.points = verts[frame]
            fig.scene.render()
            if jpg_dir:
                mlab.savefig(jpg_dir.format(frame),magnification=5)
            yield

    a = animation()
        
    # if not jpg_dir:
    fig.scene.z_plus_view()
    mlab.show()


def png2video(pathIn):
    pathOut = 'video.avi'
    fps = 60
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key = lambda x: int(x[10:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    png2video('./dbs_obj/dbs/')