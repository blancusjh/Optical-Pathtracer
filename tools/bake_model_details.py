"""Bake sculpted junctions and real font outlines to ordinary OWE triangle meshes.

Run by build_scene_models.py. No external assets or Blender add-ons are required.
"""
import json
import math
import os
from pathlib import Path
import sys
import bpy
from mathutils import Vector, Matrix
from mathutils import noise

manifest=Path(sys.argv[sys.argv.index('--')+1])
jobs=json.loads(manifest.read_text())
root=manifest.parent
scene=jobs['scene']

def export(objects,path):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects: obj.select_set(True)
    bpy.context.view_layer.objects.active=objects[0]
    bpy.ops.object.join()
    obj=bpy.context.object
    obj.data.calc_loop_triangles()
    with path.open('w') as out:
        out.write('# Blender-baked geometry; metres; Z up; no shader dependencies\n')
        for v in obj.data.vertices:
            p=obj.matrix_world@v.co
            out.write(f'v {p.x:.7f} {p.y:.7f} {p.z:.7f}\n')
        for tri in obj.data.loop_triangles:
            if tri.area>1e-15:
                out.write('f '+' '.join(str(i+1) for i in tri.vertices)+'\n')
    print('BAKED',path.name,len(obj.data.loop_triangles),'triangles',flush=True)
    bpy.data.objects.remove(obj,do_unlink=True)

font_path=Path('/usr/share/fonts/gnu-free/FreeSerif.otf')
if not font_path.exists():
    candidates=list(Path('/usr/share/fonts').rglob('*Serif*.ttf'))
    font_path=candidates[0] if candidates else None
font=bpy.data.fonts.load(str(font_path)) if font_path else None
groups={}
for job in jobs['text']:
    data=bpy.data.curves.new('Engraving','FONT')
    data.body=job['text'];data.size=job['size'];data.align_x=job['align']
    data.extrude=.000025;data.bevel_depth=.000008;data.bevel_resolution=1;data.resolution_u=5
    if font: data.font=font
    obj=bpy.data.objects.new('Engraving',data)
    bpy.context.collection.objects.link(obj)
    obj.location=job['position'];obj.rotation_euler=job['rotation']
    if 'basis' in job:
        obj.rotation_euler=Matrix(job['basis']).transposed().to_euler()
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True)
    bpy.context.view_layer.objects.active=obj
    bpy.ops.object.convert(target='MESH')
    if job.get('page'):
        for vertex in obj.data.vertices:
            world=obj.matrix_world@vertex.co
            t=min(1,abs(world.x-.95)/.215)
            world.z=.799+.019*(1-t)**2+.002*math.sin(math.pi*t)+.00010+vertex.co.z
            vertex.co=obj.matrix_world.inverted()@world
    groups.setdefault(job['group'],[]).append(bpy.context.object)
for group,objects in groups.items():
    export(objects,root/(scene+'_'+group+'.obj'))

for job in jobs['sculpt']:
    path=root/(scene+'_'+job['group']+'.obj')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.wm.obj_import(filepath=str(path),forward_axis='Y',up_axis='Z')
    obj=bpy.context.object
    obj.data.remesh_voxel_size=job.get('voxel',.025)
    bpy.ops.object.voxel_remesh()
    smooth=obj.modifiers.new('Organic junctions','SMOOTH');smooth.factor=1.05;smooth.iterations=4
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    # Small, baked bark relief; noise is deterministic in object coordinates.
    # Snapshot normals before editing coordinates. Reading v.normal after every
    # coordinate write otherwise makes Blender recompute the whole mesh per vertex.
    normals=[v.normal.copy() for v in obj.data.vertices]
    for v,normal in zip(obj.data.vertices,normals):
        p=v.co.copy()
        coarse=noise.noise_vector(p*3.7).x
        fine=noise.noise_vector(Vector((p.x*38,p.y*38,p.z*6))).x
        v.co+=normal*(.009*coarse+.004*fine)
    export([obj],path)

# This dedicated background worker has no unsaved scene. Avoid Blender's audio
# teardown, which can deadlock under a sandbox even with -noaudio. All OBJ files
# above are closed before signalling success; exceptions never reach this point.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
