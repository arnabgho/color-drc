#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
g_render4cnn_root_folder = os.path.dirname(os.path.abspath(__file__))
# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
g_blender_executable_path =   '/home/viveka/blender-2.78c-linux-glibc219-x86_64/blender'    #   '/home/arnab/blender-git/build_linux/bin/blender'       #'/home/eecs/shubhtuls/Downloads/blender-2.71/blender' #!! MODIFY if necessary
g_shapenet_root_folder =    '/home/viveka/ShapeNetCore.v2'            # '/data1/shubhtuls/cachedir/Datasets/shapeNetCoreV1'
g_blank_blend_file_path = os.path.join(g_render4cnn_root_folder, 'blank.blend')
g_blender_python_script = os.path.join(g_render4cnn_root_folder, 'render_model_views.py')
