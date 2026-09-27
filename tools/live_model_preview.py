"""Blender model-review workspace, watching the exported scene geometry.

blender --python tools/live_model_preview.py
blender -b --python tools/live_model_preview.py -- --render
Uses Workbench studio lighting: this is a geometry review, not an OWE optical render.
Only objects in this tool's own Blender scenes are replaced on refresh.
"""
import math
import os
from pathlib import Path
import sys
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "model-review"
WATCH = {}
VIEWS = {'temple':('Wide','Colonnade','OliveGrove','PrecinctEdge'),
         'observatory':('Architecture','Room','Desk','Books','Carpet','Mechanism'),
         'telescope':('Landscape','Woodland','Naked','Beside')}
LABELS = tuple(sys.argv[sys.argv.index('--scenes')+1].split(',')) if '--scenes' in sys.argv else tuple(VIEWS)


class OWE_OT_review_view(bpy.types.Operator):
    bl_idname = 'owe.review_view'
    bl_label = 'Show model detail'
    scene_label: bpy.props.StringProperty()
    camera_name: bpy.props.StringProperty()

    def execute(self, context):
        scene = bpy.data.scenes.get('OWE • ' + self.scene_label)
        if scene is None:
            return {'CANCELLED'}
        cam = next((o for o in scene.objects if o.type == 'CAMERA' and o.name.split('.')[0] == self.camera_name), None)
        if cam is None:
            return {'CANCELLED'}
        context.window.scene = scene
        scene.camera = cam
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}


class OWE_PT_review(bpy.types.Panel):
    bl_label = 'Live model review'
    bl_idname = 'OWE_PT_model_review'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Models'

    def draw(self, context):
        layout = self.layout
        layout.label(text='Geometry only • studio lighting')
        for key in LABELS:
            label=key.title();names=VIEWS[key]
            box = layout.box()
            box.label(text=label)
            for name in names:
                op = box.operator('owe.review_view', text=name)
                op.scene_label = label
                op.camera_name = name
        layout.label(text='Auto-refresh after model export')


def camera(scene, name, eye, target, angle=math.radians(48)):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = eye
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    data.angle = angle
    data.clip_end = 6000
    data.clip_start = .005
    return obj


def refresh(label):
    path = OUT / (label + ".obj")
    ready = path.with_suffix('.ready')
    if not ready.exists() or not path.exists() or WATCH.get(label) == ready.stat().st_mtime_ns:
        return
    name = "OWE • " + label.title()
    scene = bpy.data.scenes.get(name) or bpy.data.scenes.new(name)
    old_scene = bpy.context.window.scene
    bpy.context.window.scene = scene
    old_camera = scene.camera.name.split('.')[0] if scene.camera else None
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis='Y', up_axis='Z')
    for obj in list(scene.objects):
        if obj.type == 'MESH':
            if obj.data.materials:
                obj.color = obj.data.materials[0].diffuse_color
            # Curved primitives smooth; stonework and leaf folds retain their facets.
            if any(s in obj.name.lower() for s in ('ring', 'column', 'lathe', 'sphere', 'dome', 'trunk', 'branch', 'bark')):
                for face in obj.data.polygons:
                    face.use_smooth = True
    bpy.ops.object.select_all(action='DESELECT')
    cams = {}
    for line in path.with_suffix('.cameras').read_text().splitlines():
        fields = line.split()
        values = list(map(float, fields[1:]))
        cams[fields[0]] = camera(scene, fields[0], values[:3], values[3:6], values[6])
    ground = next((o for o in scene.objects if o.name.startswith('Floor.') or o.name.startswith('Precinct.')), None)
    base = max((ground.matrix_world @ Vector(v)).z for v in ground.bound_box) if ground else 0
    if label == 'observatory':
        cams['Architecture'] = camera(scene, 'Architecture', (12,-17,base+8), (0,-.8,base+3), math.radians(48))
        cams['Books'] = camera(scene, 'Books', (.25,2.68,base+1.23), (.67,3.25,base+.82), math.radians(48))
        cams['Carpet'] = camera(scene, 'Carpet', (2.3,-2.8,base+3.9), (.2,.2,base), math.radians(44))
        cams['Mechanism'] = camera(scene, 'Mechanism', (1.22,.63,base+2.55), (.40,.15,base+2.08), math.radians(46))
        scene.camera = cams['Architecture']
    elif label == 'temple':
        cams['OliveGrove'] = camera(scene, 'OliveGrove', (3,-15,base+3.2), (11,-4,base+2.5), math.radians(48))
        cams['PrecinctEdge'] = camera(scene, 'PrecinctEdge', (-32,-19,base+2), (-19,13,base+2.5), math.radians(57))
        scene.camera = cams.get('Wide') or next(iter(cams.values()))
    else:
        # Native Naked camera gives the unchanged terrain datum at the instrument.
        base=cams['Naked'].location.z-1.45
        cams['Landscape']=camera(scene,'Landscape',(-18,-21,base+8),(180,5,base+45),math.radians(63))
        cams['Woodland']=camera(scene,'Woodland',(-12,-12,base+2.6),(30,-47,base+6),math.radians(58))
        scene.camera=cams['Landscape']
    if old_camera in cams:
        scene.camera = cams[old_camera]
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.studiolight_rotate_z = .4
    shading.color_type = 'MATERIAL'
    # Workbench shadow maps cannot resolve a millimetre seam and a 3 km landscape
    # together. Studio shading + cavity gives a clean modelling view at both scales.
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = 'BOTH'
    shading.curvature_ridge_factor = 1.4
    shading.curvature_valley_factor = 1.2
    shading.show_specular_highlight = True
    shading.background_type = 'WORLD'
    scene.world = bpy.data.worlds.new(name + ' background')
    scene.world.color = (.055,.065,.08)
    scene.view_settings.view_transform = 'Standard'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'SOLID'
                area.spaces.active.shading.color_type = 'MATERIAL'
                area.spaces.active.shading.show_cavity = True
                area.spaces.active.shading.show_shadows = False
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.clip_end = 6000
    WATCH[label] = ready.stat().st_mtime_ns
    bpy.context.window.scene = old_scene if old_scene.name.startswith('OWE') else scene
    # Discard old imported datablocks after refresh; long sessions stay bounded.
    for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.worlds):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)
    print('MODEL REVIEW UPDATED:', label, flush=True)


def tick():
    try:
        for label in LABELS:
            refresh(label)
    except Exception as exc:
        print('Preview reload deferred:', exc, flush=True)
    return 3.0


tick()
bpy.context.window.scene = bpy.data.scenes.get('OWE • '+LABELS[0].title()) or bpy.context.scene
if '--render' in sys.argv:
    selected = sys.argv[sys.argv.index('--views')+1].split(',') if '--views' in sys.argv else None
    for label in LABELS:
        scene = bpy.data.scenes.get('OWE • ' + label.title())
        if not scene:
            continue
        bpy.context.window.scene = scene
        names = VIEWS[label]
        for name in names:
            if selected and name not in selected:
                continue
            cam = next((o for o in scene.objects if o.type == 'CAMERA' and o.name.split('.')[0] == name), None)
            if cam:
                scene.camera = cam
                scene.render.filepath = str(OUT / (label + '-' + name.lower() + '.png'))
                bpy.ops.render.render(write_still=True)
    for label, name in (('Temple', 'Wide'), ('Observatory', 'Architecture'), ('Telescope','Landscape')):
        scene = bpy.data.scenes.get('OWE • ' + label)
        if scene:
            scene.camera = next(o for o in scene.objects if o.type == 'CAMERA' and o.name.split('.')[0] == name)
    bpy.context.window.scene = bpy.data.scenes['OWE • '+LABELS[0].title()]
    filename='models.blend' if set(LABELS)==set(VIEWS) else '-'.join(LABELS)+'-review.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / filename))
    sys.stdout.flush()
    sys.stderr.flush()
    # All images and the review file are saved; avoid sandbox audio teardown hangs.
    os._exit(0)
else:
    if '--detail' in sys.argv:
        scene = bpy.context.window.scene
        scene.camera = next((o for o in scene.objects if o.type == 'CAMERA' and o.name.split('.')[0] == 'Books'), scene.camera)
    for cls in (OWE_OT_review_view, OWE_PT_review):
        bpy.utils.register_class(cls)
    bpy.app.timers.register(tick, persistent=True)
    print('LIVE MODEL WATCHER READY:', ', '.join(LABELS), flush=True)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.show_region_ui = True
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            area.spaces.active.region_3d.view_camera_zoom = 0
            # Blender 5 exposes active_panel_category as read-only. The Models
            # tab is available for selection without setting this UI property.
