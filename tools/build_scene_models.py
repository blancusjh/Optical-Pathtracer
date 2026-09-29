#!/usr/bin/env python3
"""Deterministic, original scene dressing; metres, Z up; Blender for detail baking.

Exports ordinary triangulated OBJ shells supported by the existing OWE loader.
Only the marked generated section of each scene is refreshed. Optical bodies,
detectors, lights and rendering settings are never generated or modified here.
"""
import argparse
from collections import defaultdict
import json
import gc
import math
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models' / 'crafted'
TAU = math.tau


def add(a, b): return tuple(x+y for x, y in zip(a, b))
def sub(a, b): return tuple(x-y for x, y in zip(a, b))
def mul(a, s): return tuple(x*s for x in a)
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return mul(a, 1/max(1e-12, math.sqrt(sum(x*x for x in a))))


class Mesh:
    def __init__(self):
        self.vertices, self.faces = [], []

    def polygon(self, points):
        start = len(self.vertices)
        self.vertices.extend(points)
        for i in range(1, len(points)-1):
            self.faces.append((start, start+i, start+i+1))

    def box(self, center, size, bevel=.008, angle=0, wear=0, rng=None):
        """Closed chamfered block with a real edge highlight, not a texture seam."""
        x, y, z = (v/2 for v in size)
        b = min(bevel, x*.35, y*.35, z*.45)
        rings = []
        ca, sa = math.cos(angle), math.sin(angle)
        jitter = [(rng.uniform(-wear, wear) if rng else 0) for _ in range(8)]
        for dz, inset in ((-z,b),(-z+b,0),(z-b,0),(z,b)):
            X,Y=x-inset,y-inset
            c=max(b*.7, .00001)
            xy=[(-X+c,-Y),(X-c,-Y),(X,-Y+c),(X,Y-c),(X-c,Y),(-X+c,Y),(-X,Y-c),(-X,-Y+c)]
            rings.append([(center[0]+u*ca-v*sa,center[1]+u*sa+v*ca,center[2]+dz+(jitter[k] if dz>0 else 0)) for k,(u,v) in enumerate(xy)])
        self.polygon(list(reversed(rings[0])))
        for a,b in zip(rings,rings[1:]):
            for i in range(8): self.polygon([a[i],a[(i+1)%8],b[(i+1)%8],b[i]])
        self.polygon(rings[-1])

    def branch(self, points, radii, segments=9, phase=0):
        rings=[]
        for j,(p,r) in enumerate(zip(points,radii)):
            axis=norm(sub(points[min(j+1,len(points)-1)],points[max(0,j-1)]))
            u=norm(cross(axis,(0,1,0) if abs(axis[1])<.9 else (1,0,0)))
            v=cross(axis,u)
            rings.append([add(p,add(mul(u,r*math.cos(TAU*i/segments+phase)*(1+.1*math.sin(i*3+j))),mul(v,r*math.sin(TAU*i/segments+phase)))) for i in range(segments)])
        self.polygon(list(reversed(rings[0])))
        for a,b in zip(rings,rings[1:]):
            for i in range(segments): self.polygon([a[i],a[(i+1)%segments],b[(i+1)%segments],b[i]])
        self.polygon(rings[-1])

    def ring(self, center, radius, thickness, axis=(0,0,1), start=0, end=TAU, steps=96):
        axis=norm(axis); u=norm(cross(axis,(0,1,0) if abs(axis[1])<.9 else (1,0,0))); v=cross(axis,u)
        points=[add(center,add(mul(u,radius*math.cos(start+(end-start)*i/steps)),mul(v,radius*math.sin(start+(end-start)*i/steps)))) for i in range(steps+1)]
        self.branch(points,[thickness]*len(points),8)

    def wood_branch(self, points, radii, segments=12):
        # Catmull-Rom centres give old timber a continuous bend, without elbow joints.
        smooth=[]; sizes=[]
        for i in range(len(points)-1):
            a,b,c,d=points[max(0,i-1)],points[i],points[i+1],points[min(len(points)-1,i+2)]
            for step in range(4):
                t=step/4
                smooth.append(tuple(.5*((2*b[k])+(-a[k]+c[k])*t+(2*a[k]-5*b[k]+4*c[k]-d[k])*t*t+(-a[k]+3*b[k]-3*c[k]+d[k])*t*t*t) for k in range(3)))
                sizes.append(radii[i]*(1-t)+radii[i+1]*t)
        self.branch(smooth+[points[-1]],sizes+[radii[-1]],segments)

    def leaf(self, p, length, width, rng, direction=None):
        a=rng.random()*TAU
        axis=norm(direction or (math.cos(a),math.sin(a),rng.uniform(-.65,.65)))
        side=norm(cross(axis,(rng.uniform(-.7,.7),rng.uniform(-.7,.7),1)))
        # Two gently folded halves with a tapered, curved lanceolate outline.
        normal=norm(cross(axis,side))
        centres=[add(p,add(mul(axis,length*t),mul(normal,width*.11*math.sin(math.pi*t)))) for t in (0,.25,.6,1)]
        for sign in (-1,1):
            edge=[add(q,mul(side,sign*width*w)) for q,w in zip(centres,(0,.72,1,0))]
            self.polygon([centres[0],edge[1],centres[1]])
            self.polygon([centres[1],edge[1],edge[2],centres[2]])
            self.polygon([centres[2],edge[2],centres[3]])

    def write(self, path):
        # Share coincident corners in the OBJ; retain sharp normals in OWE (which
        # uses geometric face normals). This substantially reduces asset size.
        unique={}
        remap=[]
        for p in self.vertices:
            key=tuple(round(v,6) for v in p)
            if key not in unique: unique[key]=len(unique)+1
            remap.append(unique[key])
        with path.open('w') as f:
            f.write('# Original procedural model; metres; Z up. Generated by tools/build_scene_models.py\n')
            for p in unique: f.write('v '+' '.join(f'{v:.6f}' for v in p)+'\n')
            for t in self.faces: f.write('f '+' '.join(str(remap[i]) for i in t)+'\n')


class Model:
    def __init__(self, name, terrain=True, variants=(), folder=''):
        # variants: other scenes built in the same room, which share this crafted section.
        # folder: the scene's group under scenes/ (scenes/glass/...), '' for scenes/ itself.
        self.name, self.terrain, self.variants, self.folder = name, terrain, variants, folder
        self.meshes=defaultdict(Mesh)
        self.colors={}
        self.rng=random.Random(7301)
        self.text_jobs=[]
        self.sculpt_jobs=[]

    def lettering(self, group, text, position, size, rotation=(0,0,0), align='CENTER'):
        self.text_jobs.append(dict(group=group,text=text,position=position,size=size,rotation=rotation,align=align))
        self.part(group,(.55,.365,.12))

    def part(self, name, rgb):
        self.colors[name]=rgb
        return self.meshes[name]

    def palette(self, name, color, count=5, spread=.12):
        return [self.part(f'{name}_{i}',tuple(min(.95,c*(1+spread*(i-(count-1)/2))) for c in color)) for i in range(count)]

    def save(self):
        OUT.mkdir(parents=True,exist_ok=True)
        lines=['# BEGIN CRAFTED MODELS', '# Generated by tools/build_scene_models.py; edit its recipes to regenerate.']
        total=0
        text_groups={job['group'] for job in self.text_jobs}
        for name,m in self.meshes.items():
            if not m.faces and name not in text_groups: continue
            stem=self.name+'_'+name
            if m.faces: m.write(OUT/(stem+'.obj'))
            rgb=', '.join(f'{c:.4f}' for c in self.colors[name])
            if 'brass' in name.lower():
                lines.append(f'material Crafted_{name} {{ type = conductor metal = gold roughness = 0.24 tint = rgb({rgb}) }}')
            else:
                # Fine, low-contrast material variation complements the actual bevels.
                low=', '.join(f'{c*.91:.4f}' for c in self.colors[name])
                scale=.006 if any(s in name for s in ('wood','Walnut','Oak','floorboards')) else .028
                lines.append(f'material Crafted_{name} {{ type = diffuse texture = noise(a = rgb({rgb}), b = rgb({low}), scale = {scale}) }}')
            pos='on_terrain("Ground", 0, 0, 0)' if self.terrain else '(0, 0, 0)'
            scale=' scale = 1000' if self.name=='optical_bench' else ''
            finish='medium = N-BK7 closed = true' if name=='Window_glass' else f'material = Crafted_{name}'
            up='../'*(2 if self.folder else 1)
            lines.append(f'body Crafted_{name} {{ type = mesh file = "{up}models/crafted/{stem}.obj" {finish} position = {pos}{scale} }}')
            total+=len(m.faces)
        lines.append('# END CRAFTED MODELS')
        for scene in (self.name,)+tuple(self.variants):
            path=ROOT/'scenes'/self.folder/f'{scene}.owe'
            source=path.read_text()
            start=source.index('# BEGIN CRAFTED MODELS')
            end=source.index('# END CRAFTED MODELS')+len('# END CRAFTED MODELS')
            path.write_text(source[:start]+'\n'.join(lines)+source[end:])
        print(f'{self.name}: {len(self.meshes)} material groups, {total:,} triangles',flush=True)
        if self.text_jobs or self.sculpt_jobs:
            # Release procedural meshes before Blender starts its bake, keeping
            # peak memory bounded when the live review is open on the same machine.
            for mesh in self.meshes.values():
                mesh.vertices.clear();mesh.faces.clear()
            gc.collect()
            manifest=OUT/(self.name+'.bakes.json')
            manifest.write_text(json.dumps(dict(scene=self.name,text=self.text_jobs,sculpt=self.sculpt_jobs),indent=2))
            subprocess.run(['blender','--background','--factory-startup','-noaudio','--threads','4','--python-exit-code','1','--python',str(ROOT/'tools'/'bake_model_details.py'),'--',str(manifest)],check=True)


def paving(model, bounds, top=.0, cell=(1.2,.82), name='Limestone', wells=()):
    from scene_details import relief_slab
    rng=model.rng
    stones=model.palette(name,(.57,.52,.43),7,.055)
    xmin,xmax,ymin,ymax=bounds
    row=0; y=ymin
    while y<ymax-.01:
        h=min(cell[1]*rng.uniform(.87,1.13),ymax-y)
        x=xmin-cell[0]*(.5 if row%2 else 0)
        while x<xmax:
            w=cell[0]*rng.uniform(.78,1.25)
            left=max(x,xmin);right=min(x+w,xmax)
            in_well=any(abs((left+right)/2-wx)<radius+(right-left)/2 and abs(y+h/2-wy)<radius+h/2 for wx,wy,radius in wells)
            if right-left>.05 and not in_well:
                relief_slab(rng.choice(stones),((left+right)/2,y+h/2,top-.055),(right-left-.014,h-.014,.11),rng,depth=.0015,n=8)
            x+=w
        row+=1;y+=h


def temple():
    from scene_details import natural_olive, natural_cypress
    m=Model('the_temple'); rng=m.rng
    planting=random.Random(710)
    trees=[(-15+planting.uniform(-.6,.6),8+k*7.5,planting.uniform(10.4,12.8),110+k) for k in range(4)]
    trees += [(15.5+planting.uniform(-.4,.4),11+k*9,planting.uniform(10,12),120+k) for k in range(3)]
    wells=[(x,y,.48) for x,y,_,_ in trees]+[(11+k*5.5,-4-k*6,.80) for k in range(3)]
    paving(m,(-22,22,-12,48),.012,wells=wells)
    for x,y,h,seed in trees: natural_cypress(m,(x,y,0),h,seed)
    for k in range(3): natural_olive(m,(11+k*5.5,-4-k*6,0),4.8-k*.24,140+k)
    # Doric frieze: raised triglyphs, incised channels, regulae and guttae.
    stone=m.part('Carved_marble',(.76,.72,.63)); shadow=m.part('Frieze_recess',(.45,.425,.37))
    for y in (8.77,31.23):
        for i in range(11):
            x=-6.5+i*1.3
            stone.box((x,y,8.68),(.42,.12,.84),.018)
            for dx in (-.125,0,.125): shadow.box((x+dx,y+(-.067 if y<20 else .067),8.7),(.04,.012,.66),.004)
            stone.box((x,y,8.18),(.52,.19,.09),.012)
            for dx in (-.18,-.06,.06,.18): stone.branch([(x+dx,y,8.07),(x+dx,y,8.14)],[.025,.034],8)
    for x in (-7.33,7.33):
        for i in range(17):
            y=9.6+i*1.3
            stone.box((x,y,8.68),(.12,.42,.84),.016)
            for dy in (-.125,0,.125): shadow.box((x+(-.067 if x<0 else .067),y+dy,8.7),(.012,.04,.66),.004)
    # Roof tiles laid on both pitches; ridge caps and antefixes break the silhouette.
    tiles=m.palette('Roof_terracotta',(.48,.285,.17),5,.065)
    for side in (-1,1):
        for j in range(42):
            y=8.35+j*.558
            for i in range(15):
                x=side*(.26+i*.52)
                z=11.685-abs(x)*.26795
                # Slope follows the existing 150 degree gable exactly.
                p1=(x-side*.25,y,z+.067);p2=(x+side*.25,y,z-.067)
                rng.choice(tiles).polygon([p1,p2,add(p2,(0,.55,0)),add(p1,(0,.55,0))])
                # Raised barrel seam down each tile strip.
                seam=(x+side*.25,y,z-.067+.022)
                tiles[(j+i)%5].branch([seam,add(seam,(0,.535,0))],[.029,.033],7)
            tiles[j%5].branch([(0,y,11.75),(0,y+.53,11.75)],[.13,.13],10)
    for side in (-1,1):
        for j in range(25):
            x=side*7.78;y=8.6+j*.94
            stone.branch([(x,y,9.66),(x,y,9.92),(x,y,10.06)],[.095,.12,.01],9)
    # Small tufts, leaf litter and rubble at the precinct margins.
    grass=m.palette('Dry_grass',(.32,.32,.15),3,.12)
    for _ in range(950):
        x=rng.uniform(-24,24);y=rng.uniform(-14,50)
        if abs(x)<21.65 and -11.7<y<47.7: continue
        for _ in range(7): rng.choice(grass).leaf((x+rng.uniform(-.07,.07),y+rng.uniform(-.07,.07),.014),rng.uniform(.12,.32),.008,rng)
    # Amphora handles, thick lips and foot rings.
    clay=m.part('Amphora_handles',(.53,.275,.14))
    for x in (-8.6,8.6):
        for side in (-1,1):
            clay.branch([(x+side*.10,7.2,1.05),(x+side*.31,7.2,1.03),(x+side*.37,7.2,.76),(x+side*.25,7.2,.61)],[.03,.04,.042,.04],12)
        clay.ring((x,7.2,1.142),.113,.022)
    from scene_details import detail_temple
    detail_temple(m)
    m.save()


def observatory():
    m=Model('the_observatory');rng=m.rng
    bricks=m.palette('Handmade_brick',(.38,.19,.125),7,.035)
    sandstone=m.palette('Sandstone',(.68,.54,.36),4,.04)
    walnut=m.palette('Walnut_panelling',(.245,.125,.06),5,.10)
    brass=m.part('Engraved_brass',(.60,.405,.145))
    bronze=m.part('Dome_bronze',(.135,.19,.17))
    # Tube rings are outside the physical optical bore. The original lenses and
    # blackened tubes remain the only geometry on each optical axis.
    for origin,az,el in (((.4188,-.6984,2.6556),176,33),((-.4591,-.9377,2.7498),164,38)):
        az,el=math.radians(az),math.radians(el)
        backward=(-math.sin(az)*math.cos(el),-math.cos(az)*math.cos(el),-math.sin(el))
        for distance in (.06,.17,.78,.84,1.12,1.18,1.80):
            brass.ring(add(origin,mul(backward,distance)),.091,.0035,axis=backward,steps=64)
    # A woven border in actual geometry, keeping the rug's centre softly coloured.
    thread=m.part('Rug_border',(.49,.30,.13))
    for x in (-1.03,1.43): thread.box((x,.2,.0024),(.038,1.68,.0005),.0001)
    for y in (-.63,1.03): thread.box((.2,y,.0024),(2.49,.038,.0005),.0001)
    for i in range(30):
        x=-.94+i*.078
        for y in (-.54,.94):
            thread.polygon([(x,y,.0028),(x+.03,y+.037,.0028),(x+.06,y,.0028),(x+.03,y-.037,.0028)])
    # Exterior running bond, with the existing lunar window left completely clear.
    for row in range(37):
        z=.055+row*.081
        for i in range(124):
            a=TAU*(i+.5*(row%2))/124
            az=math.atan2(math.cos(a),math.sin(a))%TAU
            delta=abs((az-math.radians(250)+math.pi)%TAU-math.pi)
            if delta<.155 and 1.08<z<2.72: continue
            if any(abs((az-math.radians(azimuth)+math.pi)%TAU-math.pi)<.095 and .76<z<2.56 for azimuth in (75,345,225,135)): continue
            rng.choice(bricks).box((4.615*math.cos(a),4.615*math.sin(a),z),(.224,.19,.072),.0028,a+math.pi/2,wear=.0013,rng=rng)
    # Layered plinth and cornice, voussoirs and pilasters at each bay.
    for z,r,w in ((.08,4.60,.09),(.24,4.62,.045),(2.94,4.62,.045),(3.10,4.70,.085),(3.24,4.74,.055)):
        sandstone[1].ring((0,0,z),r,w,steps=192)
    for k in range(12):
        a=k*TAU/12
        for z,width,height in ((.42,.44,.28),(1.52,.29,1.96),(2.64,.44,.20),(2.81,.52,.12)):
            sandstone[k%4].box((4.65*math.cos(a),4.65*math.sin(a),z),(width,.22,height),.016,a+math.pi/2)
        # Tall recessed exterior panels have stone surrounds and divided glazing.
        angle=a+math.pi/12
        window_az=(math.pi/2-angle)%TAU
        if abs((window_az-math.radians(250)+math.pi)%TAU-math.pi)<.23: continue
        radial=(math.cos(angle),math.sin(angle),0); tangent=(-math.sin(angle),math.cos(angle),0)
        center=mul(radial,4.69)
        if k in (0,3,7,10):
            # A flat pane on the curved course: 0.60 m keeps its edges ~2 cm clear of the
            # nearest bricks, and 0.77-2.51 m clears the courses below and above. Closed
            # glass must not overlap other matter (the renderer reports region inconsistencies).
            glass=m.part('Window_glass',(.75,.85,.86))
            glass.box(add(mul(radial,4.56),(0,0,1.64)),(.60,.006,1.74),.0001,angle+math.pi/2)
        else:
            dark=m.part('Recessed_glazing',(.07,.125,.145))
            dark.box(add(center,(0,0,1.65)),(.64,.05,1.60),.014,angle+math.pi/2)
        for offset in (-.39,.39):
            p=add(mul(radial,4.74),mul(tangent,offset))
            sandstone[2].box(add(p,(0,0,1.65)),(.095,.15,1.85),.012,angle+math.pi/2)
        for z in (.74,2.56): sandstone[2].box(add(mul(radial,4.75),(0,0,z)),(.95,.21,.13),.014,angle+math.pi/2)
        for z in (1.15,1.65,2.15): sandstone[0].box(add(mul(radial,4.74),(0,0,z)),(.66,.065,.025),.004,angle+math.pi/2)
        sandstone[0].box(add(mul(radial,4.74),(0,0,1.65)),(.028,.065,1.69),.004,angle+math.pi/2)
        # Broken-pediment style triangular hood.
        for side in (-1,1):
            p=add(mul(radial,4.75),mul(tangent,side*.48))
            sandstone[2].branch([add(p,(0,0,2.68)),add(mul(radial,4.75),(0,0,2.94))],[.055,.055],8)
    # Frame the actual lunar observing window, with its centre completely open.
    angle=math.radians(200);radial=(math.cos(angle),math.sin(angle),0);tangent=(-math.sin(angle),math.cos(angle),0)
    for offset in (-.72,.72):
        sandstone[2].box(add(add(mul(radial,4.67),mul(tangent,offset)),(0,0,1.9)),(.13,.30,1.65),.012,angle+math.pi/2)
    for z in (1.10,2.69): sandstone[2].box(add(mul(radial,4.67),(0,0,z)),(1.59,.30,.12),.012,angle+math.pi/2)
    # Inner walnut wainscot: shallow framed panels and continuous turned rails.
    for k in range(40):
        a=TAU*k/40
        walnut[k%5].box((4.43*math.cos(a),4.43*math.sin(a),.52),(.665,.095,1.03),.012,a+math.pi/2)
        for da in (-.064,.064):
            b=a+da
            walnut[3].box((4.35*math.cos(b),4.35*math.sin(b),.53),(.034,.045,.88),.006,b+math.pi/2)
    for z,r in ((.09,4.35),(.99,4.34),(1.075,4.36),(3.07,4.36)):
        walnut[3].ring((0,0,z),r,.045,steps=180)
    # Dome seams on both sides, interrupted by the original slit.
    for k in range(32):
        a=TAU*k/32
        points=[]
        for j in range(65):
            e=j*math.pi/128
            p=(4.48*math.cos(e)*math.sin(a),4.48*math.cos(e)*math.cos(a),3.2+4.48*math.sin(e))
            delta=abs((a-math.radians(170)+math.pi)%TAU-math.pi)
            in_slit=delta<math.pi/2 and abs(4.48*math.cos(e)*math.sin(delta))<.93 and e<math.radians(80)
            if in_slit:
                if len(points)>1: bronze.branch(points,[.027]*len(points),7)
                points=[]
            else: points.append(p)
        if len(points)>1: bronze.branch(points,[.027]*len(points),7)
    # Copper outer standing seams, set outside the shell.
    for k in range(32):
        a=TAU*k/32; pts=[]
        for j in range(60):
            e=j*math.pi/120;delta=abs((a-math.radians(170)+math.pi)%TAU-math.pi)
            if delta<math.pi/2 and abs(4.54*math.cos(e)*math.sin(delta))<.95 and e<math.radians(80): continue
            pts.append((4.54*math.cos(e)*math.sin(a),4.54*math.cos(e)*math.cos(a),3.2+4.54*math.sin(e)))
        if len(pts)>1: bronze.branch(pts,[.022]*len(pts),6)
    # Sawn oak boards with staggered end joints, recessed seams and subtle edge wear.
    from scene_details import relief_slab, board_grain
    boards=m.palette('Oak_boards',(.34,.205,.103),8,.04)
    for row in range(46):
        y=-4.49+row*.196
        if abs(y+.098)>4.45: continue
        half=math.sqrt(max(0,4.45**2-max(abs(y),abs(y+.19))**2))
        x=-half
        while x<half-.03:
            size=min(rng.uniform(.7,1.65),2*half if x==-half and row%2 else 1.65,half-x)
            if size<.02: break
            relief_slab(rng.choice(boards),(x+size/2,y+.093,-.0208),(size-.003,.189,.044),rng,depth=.00008,n=10)
            board_grain(m,x,y+.003,size-.003,.187,.00148,rng)
            x+=size
    # Brass meridian line and an inlaid compass rose flush in the floor.
    inlay=m.part('Floor_inlay',(.13,.09,.052))
    for radius in (.67,.74,1.0): inlay.ring((0,0,.0014),radius,.0007,steps=128)
    for k in range(16):
        a=k*TAU/16;length=.68 if k%2==0 else .43
        inlay.polygon([(0,0,.0015),(length*math.cos(a),length*math.sin(a),.0015),(.15*math.cos(a+.18),.15*math.sin(a+.18),.0015)])
    # Turned desk legs, apron, drawers and small brass handles.
    for x in (.26,1.74):
        for y in (2.99,3.61):
            walnut[2].branch([(x,y,.04),(x,y,.14),(x,y,.19),(x,y,.27),(x,y,.56),(x,y,.67),(x,y,.735)],[.045,.033,.054,.035,.024,.06,.05],16)
    walnut[1].box((1,3.0,.655),(1.46,.085,.16),.012)
    for x in (.58,1.40):
        walnut[3].box((x,2.944,.655),(.68,.025,.125),.008)
        brass.ring((x,2.918,.65),.025,.004,axis=(0,1,0),steps=24)
    # Folio orbital diagram: real thin line geometry on the open page.
    ink=m.part('Folio_ink',(.09,.06,.035))
    for radius in (.019,.03,.046,.061,.075): ink.ring((1.066,3.18,.817),radius,.0006,steps=64)
    for row in range(19):
        y=3.04+row*.013
        for col in range(4):
            x=.753+col*.04
            ink.box((x,y,.817),(.025+rng.random()*.01,.001,.0003),.00005)
    # Armillary graduation ticks, axle and brass horizon rim.
    center=(1.62,3.22,1.1455)
    for k in range(72):
        a=TAU*k/72
        brass.branch([add(center,(.144*math.cos(a),.144*math.sin(a),.004)),add(center,((.155 if k%6 else .16)*math.cos(a),(.155 if k%6 else .16)*math.sin(a),.004))],[.0009,.0009],5)
    brass.branch([(1.62,3.22,.97),(1.62,3.22,1.32)],[.003,.003],8)
    # Reference-inspired instrument cabinet at the north-west wall.
    for z in (.12,.92,1.70,2.42): walnut[1].box((-2.1,3.22,z),(1.42,.43,.08),.012)
    for x in (-2.82,-1.38): walnut[3].box((x,3.22,1.27),(.085,.43,2.50),.012)
    leather=m.palette('Library_bindings',(.24,.105,.075),5,.16)
    parchment=m.part('Parchment_edges',(.64,.56,.39))
    for shelf in (.18,.98,1.76):
        x=-2.71
        while x<-1.50:
            w=rng.uniform(.055,.10);h=rng.uniform(.36,.55)
            rng.choice(leather).box((x,3.18,shelf+h/2),(w,.27,h),.004)
            for z in (shelf+.06,shelf+h-.06): brass.box((x,3.037,z),(w*.8,.005,.007),.001)
            parchment.box((x,3.18,shelf+h+.001),(w*.80,.22,.002),.0002)
            x+=w+.013
    # Stone terrace: radially cut blocks, outside the tower's original interior.
    for row in range(4):
        radius=4.94+row*.42
        for k in range(84):
            a=TAU*(k+.5*(row%2))/84
            sandstone[k%4].box((radius*math.cos(a),radius*math.sin(a),-.065),(.34+row*.025,.405,.18),.012,a+math.pi/2)
    # Greenwich-inspired entrance pavilion: rusticated corners, a recessed oak door,
    # mansard roof and an oculus dormer. Below the southern observing sight lines.
    for row in range(17):
        z=.15+row*.155
        for side in (-1,1):
            for j in range(7):
                rng.choice(bricks).box((side*1.50,-4.52-j*.29,z),(.19,.275,.14),.008)
        for j in range(11):
            x=-1.45+j*.29
            if abs(x)<.72 and z<2.22: continue
            rng.choice(bricks).box((x,-6.39,z),(.275,.20,.14),.008)
    for x in (-1.54,1.54):
        for row in range(9): sandstone[row%4].box((x,-6.40,.18+row*.29),(.34,.32,.27),.012)
    for x in (-.79,.79): sandstone[2].box((x,-6.43,1.16),(.18,.28,2.32),.013)
    sandstone[2].box((0,-6.43,2.37),(1.87,.33,.18),.012)
    walnut[1].box((0,-6.35,1.09),(1.39,.10,2.13),.008)
    for x in (-.34,.34):
        for z in (.48,1.18,1.80): walnut[3].box((x,-6.416,z),(.54,.04,.45),.016)
    brass.ring((.16,-6.457,1.18),.043,.006,axis=(0,1,0),steps=32)
    for y in (-6.60,-6.80): sandstone[1].box((0,y,.055),(2.2,.42,.10),.012)
    for z,w in ((2.76,3.35),(2.87,3.48)):
        sandstone[1].box((0,-5.43,z),(w,2.4,.10),.012)
    slate=m.palette('Mansard_slate',(.15,.19,.205),4,.06)
    # Hip roof in strips: lower steep pitch then shallow upper crown.
    corners=[(-1.73,-6.63),(1.73,-6.63),(1.73,-4.22),(-1.73,-4.22)]
    upper=[(-1.16,-6.10),(1.16,-6.10),(1.16,-4.72),(-1.16,-4.72)]
    for i in range(4):
        a,b=corners[i],corners[(i+1)%4];c,d=upper[i],upper[(i+1)%4]
        for j in range(17):
            t=j/17;s=(j+.94)/17
            p=(a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t,2.93)
            q=(a[0]*(1-s)+b[0]*s,a[1]*(1-s)+b[1]*s,2.93)
            r=(c[0]*(1-s)+d[0]*s,c[1]*(1-s)+d[1]*s,3.63)
            u=(c[0]*(1-t)+d[0]*t,c[1]*(1-t)+d[1]*t,3.63)
            slate[j%4].polygon([p,q,r,u])
        slate[1].polygon([(x,y,3.63) for x,y in upper])
    # Round dormer centred over the door; its dark centre is recessed behind trim.
    sandstone[2].box((0,-6.49,3.18),(.66,.18,.66),.035)
    sandstone[2].ring((0,-6.60,3.23),.22,.045,axis=(0,1,0),steps=64)
    m.part('Recessed_glazing',(.07,.125,.145)).polygon([(.195*math.cos(TAU*k/64),-6.601,3.23+.195*math.sin(TAU*k/64)) for k in range(64)])
    from scene_details import detail_observatory
    detail_observatory(m)
    m.save()


def camera_obscura_floor():
    # Boards lie on the room's floor slab (top at z = 0), above the garden lawn (z = -0.001), which
    # stays hidden inside the slab: a board top level with the lawn would leave two coincident
    # surfaces for the renderer to choose between. The first row stops 0.5 mm short of the back wall.
    m=Model('camera_obscura',False)
    boards=m.palette('Worn_floorboards',(.07,.047,.03),5,.1)
    for j in range(21):
        for i in range(3):
            boards[(i+j)%5].box((-1+i,-3.812+j*.18,.018),(.995,.175,.034),.002)
    m.save()


def other_floors():
    camera_obscura_floor()
    # The telescope's slope-grounded terrain/forest is owned by build_landscapes.py.
    # Joinery for close-up demonstrations: board tops stay at the old support plane.
    for scene,folder,width,depth,ycenter in (('the_lens','',1.4,.9,0),('glass_of_water','glass',1.2,.8,.1)):
        m=Model(scene,False,folder=folder);rng=m.rng
        wood=m.palette('Table_boards',(.42,.255,.13),6,.055)
        for j in range(6):
            wood[j].box((0,ycenter-depth/2+(j+.5)*depth/6,-.014),(width,depth/6-.0006,.028),.0006)
        wood[2].box((0,ycenter-depth/2+.025,-.055),(width-.07,.04,.08),.004)
        wood[2].box((0,ycenter+depth/2-.025,-.055),(width-.07,.04,.08),.004)
        for x in (-width/2+.035,width/2-.035): wood[3].box((x,ycenter,-.055),(.045,depth-.06,.08),.003)
        grain=m.part('Open_woodgrain',(.30,.175,.08))
        for _ in range(350):
            x=rng.uniform(-width*.48,width*.48);y=rng.uniform(ycenter-depth*.48,ycenter+depth*.48)
            length=min(rng.uniform(.02,.14),width/2-x-.005)
            grain.polygon([(x,y,.000001),(x+length,y+.00008,.000001),(x+length*.3,y+.00022,.000001)])
        m.save()
    m=Model('optical_bench',False)
    table=m.part('Bench_top',(.14,.145,.15));table.box((0,.20,-.070),(.48,.68,.04),.006)
    feet=m.part('Bench_feet',(.25,.27,.28))
    for y in (-.015,.415): feet.box((0,y,-.046),(.125,.043,.014),.003)
    m.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene',choices=['all','temple','observatory','floors'],default='all',nargs='?')
    args=parser.parse_args()
    if args.scene in ('all','temple'): temple()
    if args.scene in ('all','observatory'): observatory()
    if args.scene in ('all','floors'): other_floors()
