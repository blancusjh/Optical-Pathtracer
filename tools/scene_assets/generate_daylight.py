"""Bake Hosek-Wilkie daylight with Cycles CUDA or CPU, with explicit owe direction conventions.
No photographic sky is rotated to an incompatible elevation. No solar disc.
Generated 4096x2048 HDR files are retained at their original resolution.
"""
import bpy, math, json, hashlib, re, argparse, sys
import numpy as np
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cpu', action='store_true', help='Use Cycles CPU; also the fallback when CUDA is unavailable.')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
root=Path.cwd(); out=root/'assets/daylight';out.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=1
s.cycles.device='CPU'
if not args.cpu:
 try:
  prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='CUDA';prefs.get_devices()
  cuda=[d for d in prefs.devices if d.type=='CUDA']
  if cuda:
   for d in prefs.devices:d.use=d.type=='CUDA'
   s.cycles.device='GPU'
 except (TypeError, RuntimeError):
  pass
print('Daylight bake device:',s.cycles.device,flush=True)
s.render.resolution_x=4096;s.render.resolution_y=2048;s.render.resolution_percentage=100
s.render.image_settings.file_format='HDR';s.render.image_settings.color_mode='RGB'
s.view_settings.view_transform='Raw';s.view_settings.exposure=0;s.view_settings.gamma=1
s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=.01
bpy.ops.mesh.primitive_plane_add(size=2);plane=bpy.context.object;plane.scale.y=.5
bpy.ops.object.camera_add(location=(0,0,1));cam=bpy.context.object;cam.data.type='ORTHO';cam.data.ortho_scale=2;s.camera=cam
mat=bpy.data.materials.new('AtmosphericRadiance');mat.use_nodes=True;plane.data.materials.append(mat)
nt=mat.node_tree;nt.nodes.clear();n=nt.nodes;l=nt.links
uv=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ');l.new(uv.outputs['UV'],sep.inputs[0])
def op(kind,a,b=None):
 x=n.new('ShaderNodeMath');x.operation=kind
 for i,v in enumerate([a,b]):
  if v is None:continue
  if isinstance(v,(int,float)):x.inputs[i].default_value=v
  else:l.new(v,x.inputs[i])
 return x.outputs[0]
phi=op('MULTIPLY',sep.outputs['X'],2*math.pi);theta=op('MULTIPLY',op('SUBTRACT',1,sep.outputs['Y']),math.pi)
v=n.new('ShaderNodeCombineXYZ');st=op('SINE',theta)
l.new(op('MULTIPLY',st,op('COSINE',phi)),v.inputs[0]);l.new(op('MULTIPLY',st,op('SINE',phi)),v.inputs[1]);l.new(op('COSINE',theta),v.inputs[2])
sky=n.new('ShaderNodeTexSky');sky.sky_type='HOSEK_WILKIE';sky.turbidity=2.4;sky.ground_albedo=.18;l.new(v.outputs[0],sky.inputs['Vector'])
em=n.new('ShaderNodeEmission');l.new(sky.outputs[0],em.inputs[0]);output=n.new('ShaderNodeOutputMaterial');l.new(em.outputs[0],output.inputs['Surface'])
records=json.loads((out/'MANIFEST.json').read_text()) if (out/'MANIFEST.json').exists() else []
def scene_sun(name):
 text=(root/'scenes'/('the_'+name+'.owe')).read_text()
 block=re.search(r'\bsun\s*\{([^}]+)\}',text).group(1)
 def value(key,default=None):
  m=re.search(r'\b'+key+r'\s*=\s*([-+0-9.eE]+)',block)
  if m:return float(m.group(1))
  if default is not None:return default
  raise ValueError('Missing solar '+key+' in '+name)
 return name,value('elevation'),value('azimuth'),value('luminance'),value('angular_diameter',.533)
for name,elev,az,solarY,diam in map(scene_sun,['kunstkammer','invisible_window']):
 if (out/(name+'_sun_aligned.hdr')).exists():
  print('Existing daylight preserved:',name,flush=True);continue
 records=[r for r in records if r.get('id')!=name]
 er,ar=map(math.radians,[elev,az]);direction=(math.cos(er)*math.sin(ar),math.cos(er)*math.cos(ar),math.sin(er));sky.sun_direction=direction
 raw=out/(name+'_hosek_raw.hdr');s.render.filepath=str(raw);bpy.ops.render.render(write_still=True)
 im=bpy.data.images.load(str(raw),check_existing=False);w,h=im.size;pixels=np.empty(w*h*4,np.float32);im.pixels.foreach_get(pixels);a=pixels.reshape(h,w,4)
 # Blender pixel rows bottom-first; upper hemisphere is the upper half of this array.
 theta=np.pi*(1-(np.arange(h)+.5)/h);weights=np.maximum(np.cos(theta),0)*np.sin(theta)
 Y=a[:,:,:3]@np.array([.2126,.7152,.0722],np.float32);E=float((Y.mean(axis=1)*weights).sum()*2*np.pi*np.pi/h)
 omega=2*np.pi*(1-math.cos(math.radians(diam)/2));direct_normal=solarY*omega
 # Explicit clear-day assumption: diffuse horizontal irradiance =10% of DNI.
 # This sets the light level, not a claim of site-specific measured weather.
 target=.1*direct_normal;scale=target/E;a[:,:,:3]*=scale
 # Lambertian distant terrain, no fictitious lower-hemisphere grey emission.
 terrain=np.array([.085,.12,.045],np.float32);ground=terrain*(target+direct_normal*math.sin(er))/np.pi
 a[:h//2,:,:3]=ground;a[:,:,3]=1
 final=bpy.data.images.new(name+'_sun_aligned',width=w,height=h,float_buffer=True,alpha=True);final.pixels.foreach_set(a.ravel());final.filepath_raw=str(out/(name+'_sun_aligned.hdr'));final.file_format='HDR';final.save()
 # Inspection thumbnail ONLY; not a scene texture or edited render.
 preview=bpy.data.images.new(name+'_inspection',width=w,height=h,float_buffer=True,alpha=True)
 disp=a.copy();gain=.18/(target/np.pi);disp[:,:,:3]=np.maximum(disp[:,:,:3]*gain,0);disp[:,:,:3]=disp[:,:,:3]/(1+disp[:,:,:3]);preview.pixels.foreach_set(disp.ravel());preview.scale(1024,512);preview.filepath_raw=str(out/(name+'_sky_preview.png'));preview.file_format='PNG';preview.save()
 records.append(dict(id=name,model='Hosek-Wilkie 2012, Blender Cycles '+s.cycles.device,reference='https://cgg.mff.cuni.cz/projects/SkylightModelling/',license='Project-generated radiance data; no third-party photograph',resolution=[w,h],sun_elevation_deg=elev,sun_azimuth_deg=az,sun_direction_xyz=direction,environment_rotation_deg=0,solar_disc_in_map=False,turbidity=2.4,model_ground_albedo=.18,terrain_reflectance=terrain.tolist(),diffuse_horizontal_to_direct_normal_ratio=.1,raw_horizontal_irradiance=E,radiance_scale=scale,direct_normal_irradiance=direct_normal,sky_horizontal_irradiance=target,files=[str(raw.relative_to(root)),str(Path(final.filepath_raw).relative_to(root))]))
 for img in [im,final,preview]:bpy.data.images.remove(img)
(out/'MANIFEST.json').write_text(json.dumps(records,indent=2)+'\n');print('DAYLIGHT_COMPLETE',json.dumps(records),flush=True)
