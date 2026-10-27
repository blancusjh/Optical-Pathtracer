"""Project curation recipe: classify original triangles without editing scenes. Run from repository root."""
from pathlib import Path
import json,copy
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
if Path('assets/sketchfab/armor_man_and_horse/curated/scene.gltf').exists():
    print('Curated armour already exists; preserving it.')
    raise SystemExit(0)
src=Path('assets/sketchfab/armor_man_and_horse');d=json.load(open(src/'scene.gltf'));bufs=[(src/b['uri']).read_bytes() for b in d['buffers']]
def arr(i):
 a=d['accessors'][i];v=d['bufferViews'][a['bufferView']];dt={5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']];nc={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];off=v.get('byteOffset',0)+a.get('byteOffset',0);return np.ndarray((a['count'],nc),dtype=dt,buffer=bufs[v['buffer']],offset=off,strides=(v.get('byteStride',np.dtype(dt).itemsize*nc),np.dtype(dt).itemsize))
d['materials']=[{'name':'Steel'},{'name':'CrimsonCloth'}];new=bytearray();bi=len(d['buffers']);count=[0,0];im=np.asarray(Image.open(src/'textures/Armor_Mat_baseColor.jpeg').convert('RGB'));h,w=im.shape[:2]
for mesh in d['meshes']:
 ps=[]
 for p in mesh['primitives']:
  idx=arr(p['indices']).reshape(-1,3);uv=arr(p['attributes']['TEXCOORD_0']);tuv=uv[idx].mean(axis=1);c=im[(tuv[:,1]*h).astype(int).clip(0,h-1),(tuv[:,0]*w).astype(int).clip(0,w-1)].astype(float);red=(c[:,0]>c[:,1]*1.35)&(c[:,0]>c[:,2]*1.4)&(c[:,0]>35)
  for mat,mask in enumerate([~red,red]):
   ix=idx[mask].astype('<u4').flatten();off=len(new);new.extend(ix.tobytes());vi=len(d['bufferViews']);d['bufferViews'].append({'buffer':bi,'byteOffset':off,'byteLength':ix.nbytes,'target':34963});ai=len(d['accessors']);d['accessors'].append({'bufferView':vi,'componentType':5125,'count':len(ix),'type':'SCALAR'});q=copy.deepcopy(p);q['indices']=ai;q['material']=mat;ps.append(q);count[mat]+=len(ix)//3
 mesh['primitives']=ps
out=Path('assets/sketchfab/armor_man_and_horse/curated');out.mkdir(exist_ok=True)
for b in d['buffers']:b['uri']='../'+b['uri']
for img in d.get('images',[]):
 if 'uri' in img:img['uri']='../'+img['uri']
d['buffers'].append({'uri':'classification.bin','byteLength':len(new)});(out/'classification.bin').write_bytes(new);(out/'scene.gltf').write_text(json.dumps(d));print('Armour triangle classification, steel/cloth:',count)
