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

    
def verts_animation(verts):
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
    
    @mlab.animate(delay=40, ui=False)
    def animation():
        for i in count():
            frame = i % len(verts)
            pd.points = verts[frame]
            fig.scene.render()
            yield

    a = animation()
    fig.scene.z_minus_view()
    mlab.show()