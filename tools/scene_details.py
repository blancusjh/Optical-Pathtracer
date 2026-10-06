"""Close-range modelling recipes. Every detail is exported as scene geometry."""
import math
import random
from build_scene_models import Mesh, add, sub, mul, norm, cross, TAU


def path(mesh, points, radius=.0005, sides=6):
    mesh.branch(points,[radius]*len(points),sides)


def transform(point, origin, angle=0):
    x,y,z=point;c,s=math.cos(angle),math.sin(angle)
    return (origin[0]+c*x-s*y,origin[1]+s*x+c*y,origin[2]+z)


def scroll(mesh, center, size, angle=0, radius=.00035):
    points=[]
    for i in range(49):
        t=i/48; a=TAU*1.35*t
        r=size*(1-.89*t)
        points.append(transform((r*math.cos(a),r*math.sin(a),0),center,angle))
    path(mesh,points,radius)
    # Engraved leaflets fan off the main curl.
    for k in range(6):
        a=angle+k*.7
        root=add(center,(size*.62*math.cos(a),size*.62*math.sin(a),0))
        mesh.leaf(root,size*.31,size*.09,random.Random(k),(-math.sin(a),math.cos(a),.05))


def relief_slab(mesh, center, size, rng, angle=0, depth=.002, n=8):
    """Chipped rim and lightly dished, tessellated top; closed beneath the surface."""
    w,h,t=size;cx,cy,cz=center
    phase=rng.uniform(0,TAU)
    ztop=cz+t/2
    grid=[]
    for j in range(n+1):
        row=[]
        for i in range(n+1):
            u=i/n;v=j/n
            edge=min(u,1-u,v,1-v)
            rough=depth*(.35*math.sin(31*u+17*v+phase)+.25*math.sin(13*u-29*v+phase))
            wear=depth*1.8*max(0,1-edge*15)
            z=ztop+rough-wear-depth*.4*math.sin(math.pi*u)*math.sin(math.pi*v)
            row.append(transform(((u-.5)*w,(v-.5)*h,z-cz),center,angle))
        grid.append(row)
    for j in range(n):
        for i in range(n): mesh.polygon([grid[j][i],grid[j][i+1],grid[j+1][i+1],grid[j+1][i]])
    rim=grid[0][:]+[grid[j][-1] for j in range(1,n+1)]+list(reversed(grid[-1][:-1]))+[grid[j][0] for j in range(n-1,0,-1)]
    bottom=[(p[0],p[1],cz-t/2) for p in rim]
    for i in range(len(rim)): mesh.polygon([bottom[i],bottom[(i+1)%len(rim)],rim[(i+1)%len(rim)],rim[i]])
    # A centre fan avoids zero-area triangles along the subdivided straight rim.
    for i in range(len(bottom)):
        mesh.polygon([(cx,cy,cz-t/2),bottom[(i+1)%len(bottom)],bottom[i]])


def book(m, center, width, depth, height, angle, title, color, seed, base=True):
    rng=random.Random(seed)
    leather=m.part('Tooled_leather_'+str(seed),color)
    gilt=m.part('Book_brass_tooling',(.55,.37,.12))
    paper=m.part('Book_deckled_paper',(.69,.60,.43))
    edge=m.part('Book_page_shadows',(.36,.285,.19))
    def p(q): return transform(q,center,angle)
    if base:
        for z in (.002,height-.002): leather.box(p((0,0,z)),(width,depth,.004),.001,angle)
        paper.box(p((.001,0,height/2)),(width-.012,depth-.011,height-.008),.001,angle)
    # A convex leather spine, headcaps and raised cords replace a square binding.
    leather.branch([p((-width/2+.004,y,height/2)) for y in (-depth/2+.004,0,depth/2-.004)],[height*.52]*3,16)
    for y in (-depth*.37,-depth*.23,depth*.23,depth*.37):
        points=[p((-width/2-.006-height*.12*math.sin(math.pi*j/16),y,.003+(height-.006)*j/16)) for j in range(17)]
        path(leather,points,.002)
        path(gilt,[add(q,(0,0,.00015)) for q in points],.00028)
    # Slightly different exposed page edges; each line sits against the paper block.
    for layer in range(max(12,int(height/.00075))):
        z=.004+layer*.00075
        if z>height-.005: break
        points=[p((-width*.46+width*.92*k/16,-depth/2+.004+rng.uniform(-.0006,.0006),z)) for k in range(17)]
        path(edge,points,.00007,4)
    # Double fillet, corner arabesques and a central cartouche.
    z=height+.00015
    for inset in (.009,.012):
        x=width/2-inset;y=depth/2-inset
        path(gilt,[p(q) for q in ((-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z),(-x,-y,z))],.00032)
    for k in range(20):
        x=-width*.38+k*width*.76/19
        for sy in (-1,1):
            motif_center=p((x,sy*(depth*.5-.017),z))
            points=[add(motif_center,(.0026*math.cos(a*TAU/16),.0026*math.sin(a*TAU/16),0)) for a in range(17)]
            path(gilt,points,.00015,4)
    for sx in (-1,1):
        for sy in (-1,1):
            scroll(gilt,p((sx*(width/2-.028),sy*(depth/2-.026),z)),.013,angle+sx*sy*.7,.00028)
    for inset in (0,.002):
        points=[p(((width*.30-inset)*math.cos(TAU*k/96),(depth*.235-inset)*math.sin(TAU*k/96),z)) for k in range(97)]
        path(gilt,points,.00025)
    m.lettering('Book_brass_lettering',title,p((0,-.003,z)),min(.015,width/max(8,len(title))*.95),(0,0,angle))
    m.lettering('Book_brass_lettering','ASTRONOMIA',p((0,-depth*.115,z)),.008,(0,0,angle))
    for x in (-width*.38,width*.38):
        for y in (-depth*.37,depth*.37):
            gilt.ring(p((x,y,z)),.0013,.00023,steps=16)
    # Sewn headband at the top of the page block.
    for k in range(15):
        q=p((-width/2+.006+k*.0014,depth/2-.004,height-.004))
        path(gilt,[q,add(q,(0,.0013,.001))],.00025,4)


def folio(m):
    m.meshes['Folio_ink']=Mesh()
    pages=m.part('Folio_curved_pages',(.79,.71,.54))
    ink=m.part('Folio_printed_ink',(.065,.043,.022))
    def p(x,y,offset=0):
        t=min(1,abs(x-.95)/.215)
        return (x,y,.799+.019*(1-t)**2+.002*math.sin(math.pi*t)+offset)
    for side in (-1,1):
        for layer in range(5):
            for i in range(32):
                for j in range(12):
                    x0=.95+side*(.007+i*.0065);x1=.95+side*(.007+(i+1)*.0065)
                    y0=3.023+j*.0245;y1=y0+.0245
                    pages.polygon([p(x0,y0,-layer*.00055),p(x1,y0,-layer*.00055),p(x1,y1,-layer*.00055),p(x0,y1,-layer*.00055)])
    def text(value,x,y,size):
        m.lettering('Folio_ink_lettering',value,(x,y,.82),size)
        m.text_jobs[-1]['page']=True
    text('CL. PTOLEMAEI',.836,3.282,.014)
    text('DE MOTIBVS CAELESTIBVS',.836,3.263,.0058)
    prose=['ORDO ET RATIO COELI','Sol medium mundi tenet.','Luna circa terram volvitur.','Stellae suis orbibus moventur.','Tempus observatione notatur.','Mensura et proportio rerum.']
    for row in range(18): text(prose[row%len(prose)],.837,3.240-row*.0108,.0054)
    for r in (.015,.026,.039,.053,.068,.082):
        points=[p(1.066+r*math.cos(TAU*k/128),3.164+r*math.sin(TAU*k/128),.00010) for k in range(129)]
        path(ink,points,.00028,4)
    for k,(r,label) in enumerate(((.026,'I'),(.039,'II'),(.053,'III'),(.068,'IV'),(.082,'V'))):
        a=.8+k*.91;x=1.066+r*math.cos(a);y=3.164+r*math.sin(a)
        ink.ring(p(x,y,.00012),.0016,.00045,steps=16)
        text(label,x+.004,y+.003,.004)
    text('SOL',1.066,3.16,.006)
    text('SYSTEMA MVNDI',1.066,3.278,.009)
    m.colors['Folio_ink_lettering']=(.055,.038,.021)


def carpet(m):
    # Dense, interlaced ribbon geometry with a floral field, medallion and guard borders.
    m.meshes['Rug_border']=Mesh()
    colors=[(.285,.045,.031),(.11,.022,.018),(.035,.07,.078),(.48,.31,.13),(.65,.54,.34),(.14,.20,.13)]
    meshes=[m.part('Carpet_pile_'+str(i),c) for i,c in enumerate(colors)]
    def pattern(x,y):
        X=abs(x)/1.3;Y=abs(y)/.9;edge=min(1-X,1-Y)
        if edge<.025: return 1
        if .025<edge<.052 or .18<edge<.195: return 3
        if edge<.18:
            if X>Y: along=y*27;across=(1-X)*35
            else: along=x*27;across=(1-Y)*35
            rose=math.sin(along)**2+math.sin(across)**2
            return 4 if rose<.34 else (3 if rose<.75 else 2)
        u=x/.63;v=y/.53;r=math.hypot(u,v);a=math.atan2(v,u)
        rim=1+.09*math.cos(a*12)
        if r<rim:
            if r>rim-.08: return 4
            if r>.74: return 3 if math.sin(a*24)>.30 else 2
            if r<.16+.035*math.cos(8*a): return 4
            petal=abs(math.sin(a*8))
            return 3 if petal<.23 or .35<r<.42 else (5 if petal<.43 else 2)
        # Stylised palmettes and curling vines in the red field.
        px=(x+.065)% .22-.11;py=(y+.03)%.20-.10
        rho=math.hypot(px/.063,py/.073)
        if rho<.38: return 3
        if .55<rho<.78 and py>-.045: return 4 if px*py>0 else 5
        return 1 if abs(py-.035*math.sin(x*24))<.009 else 0
    nx,ny=650,450
    for j in range(ny):
        y=-.9+(j+.5)*1.8/ny
        for i in range(nx):
            x=-1.3+(i+.5)*2.6/nx
            mesh=meshes[pattern(x,y)]
            z=.00255+.00011*math.sin(i*1.7+j*2.3)
            # Crossing warp/weft ribbons; the high thread alternates at each knot.
            for half in range(2):
                crest=.00019 if (i+j+half)%2 else -.00010
                def knot(u,v,h):
                    return (.2+x+(v if half else u),.2+y+(u if half else v),z+h)
                mesh.polygon([knot(-.00082,-.002,0),knot(.00082,-.002,0),knot(.00082,0,crest),knot(-.00082,0,crest)])
                mesh.polygon([knot(-.00082,0,crest),knot(.00082,0,crest),knot(.00082,.002,0),knot(-.00082,.002,0)])
    fringe=m.part('Carpet_cotton_fringe',(.62,.54,.39));rng=random.Random(553)
    for side in (-1,1):
        for i in range(260):
            x=-1.08+i*.00985;y=.2+side*.90
            points=[(x+.0015*math.sin(j*.9+i),y+side*j*.006,.0024+.0006*math.sin(j*.6+i)) for j in range(rng.randint(7,12))]
            path(fringe,points,.00055,5)


def telescope_details(m):
    brass=m.part('Machined_brass',(.53,.34,.12))
    dark=m.part('Engraving_black',(.035,.025,.016))
    iron=m.part('Mount_cast_iron',(.065,.073,.071))
    for index,(origin,az,el) in enumerate((((.4188,-.6984,2.6556),176,33),((-.4591,-.9377,2.7498),164,38))):
        az,el=map(math.radians,(az,el))
        axis=(-math.sin(az)*math.cos(el),-math.cos(az)*math.cos(el),-math.sin(el))
        side=norm(cross(axis,(0,0,1))); up=cross(side,axis)
        def point(t,u,v): return add(origin,add(mul(axis,t),add(mul(side,u),mul(up,v))))
        # Retaining collars, split clamps, and real external screws.
        for t in (.06,.17,.80,1.16,1.79):
            for dt in (-.009,0,.009): brass.ring(point(t+dt,0,0),.089,.0024,axis=axis,steps=96)
            for k in range(6):
                a=TAU*k/6;n=add(mul(side,math.cos(a)),mul(up,math.sin(a)))
                c=add(point(t,0,0),mul(n,.094))
                brass.branch([c,add(c,mul(n,.004))],[.0028,.0028],8)
                tang=norm(cross(n,axis));path(dark,[add(c,add(mul(n,.0042),mul(tang,-.0017))),add(c,add(mul(n,.0042),mul(tang,.0017)))],.00028,4)
        # Maker's plaque: lettering follows the tube's own coordinate frame.
        center=point(.46,.091,0)
        brass.polygon([add(center,add(mul(axis,u),mul(up,v))) for u,v in ((-.105,-.026),(.105,-.026),(.105,.026),(-.105,.026))])
        for label,offset,size in (('OBSERVATORIVM',.006,.009),('REFRACTOR  •  150',-.009,.006)):
            m.lettering('Telescope_ink_lettering',label,add(center,add(mul(up,offset),mul(side,.0002))),size)
            m.text_jobs[-1]['basis']=[axis,up,side]
        # Worm wheel and a graduated declination annulus on the right of the saddle.
        center=point(.99,.155,-.11)
        basis1=axis;basis2=up
        # Central hub, swept spokes and a connected bearing saddle.
        brass.branch([add(center,mul(side,-.022)),add(center,mul(side,.012))],[.027,.027],32)
        for k in range(3):
            a=k*TAU/3
            points=[add(center,add(mul(axis,r*math.cos(a+.2*r/.11)),mul(up,r*math.sin(a+.2*r/.11)))) for r in (.023,.05,.08,.111)]
            path(brass,points,.008,10)
        pier=((.3603,.1383,1.969),(-.6763,-.1802,1.985))[index]
        iron.branch([pier,point(.99,0,-.11),add(center,mul(side,-.022))],[.045,.038,.028],20)
        saddle=[point(.99,.096*math.cos(a*math.pi/48),-.096*math.sin(a*math.pi/48)) for a in range(49)]
        path(iron,saddle,.009,10)
        for k in range(192):
            a=TAU*k/192;b=TAU*(k+1)/192
            def q(r,t,d=0): return add(center,add(mul(side,d),add(mul(basis1,r*math.cos(t)),mul(basis2,r*math.sin(t)))))
            brass.polygon([q(.108,a),q(.145,a),q(.145,b),q(.108,b)])
            brass.polygon([q(.108,b,-.018),q(.145,b,-.018),q(.145,a,-.018),q(.108,a,-.018)])
            for r in (.108,.145):
                brass.polygon([q(r,a,0),q(r,a,-.018),q(r,b,-.018),q(r,b,0)])
            if k%2==0: path(dark,[q(.126 if k%16 else .116,a,.00015),q(.141,a,.00015)],.00032,4)
            if k%16==0:
                pos=q(.115,a,.0002)
                m.lettering('Telescope_ink_lettering',str(int(k/192*360)),pos,.0045)
                m.text_jobs[-1]['basis']=[axis,up,side]
            if k%3==0:
                half=TAU/192*.68
                tooth=[(.143,a-half),(.155,a-half*.6),(.155,a+half*.6),(.143,a+half)]
                front=[q(r,t,-.002) for r,t in tooth]
                back=[q(r,t,-.016) for r,t in tooth]
                brass.polygon(front);brass.polygon(list(reversed(back)))
                for j in range(4):
                    brass.polygon([front[j],back[j],back[(j+1)%4],front[(j+1)%4]])
        # Knurled slow motion and focus knobs on external shafts.
        for t,v in ((1.02,-.26),(1.69,-.02)):
            c=point(t,.16,v)
            iron.branch([point(t,.097,v),c],[.007,.007],12)
            brass.branch([c,add(c,mul(side,.018))],[.022,.022],32)
            for k in range(36):
                a=k*TAU/36;n=add(mul(axis,math.cos(a)),mul(up,math.sin(a)))
                path(brass,[add(c,mul(n,.0225)),add(c,add(mul(n,.0225),mul(side,.018)))],.0007,4)
    m.colors['Telescope_ink_lettering']=(.035,.025,.016)


def detail_observatory(m):
    book(m,(.42,3.47,.7854),.27,.20,.05,math.radians(4),'PTOLEMAEI',(.24,.05,.029),1,False)
    book(m,(.43,3.46,.8355),.25,.18,.042,math.radians(-9),'COELESTIS',(.045,.13,.095),2,False)
    book(m,(.41,3.48,.8776),.22,.16,.036,math.radians(15),'KEPLERI',(.22,.11,.045),3,False)
    book(m,(.40,3.135,.7854),.28,.25,.042,math.radians(-10),'URANOGRAPHIA',(.16,.029,.020),4)
    folio(m)
    carpet(m)
    telescope_details(m)
    decorative_joinery(m)


def board_grain(m,x,y,width,height,z,rng):
    grain=m.part('Oak_grain',(.28,.165,.084))
    nails=m.part('Floor_forged_nails',(.13,.12,.10))
    knot=rng.uniform(.15,.85)*width
    for i in range(13):
        sy=(i+.5)*height/13
        phase=rng.uniform(0,TAU)
        start=rng.uniform(.01,.07)*width;end=rng.uniform(.84,.98)*width
        points=[]
        for k in range(29):
            u=start+(end-start)*k/28
            bend=.0032*math.sin(u*22+phase)+.005*math.sin((sy/height-.5)*4)*math.exp(-((u-knot)/.075)**2)
            points.append((x+u,y+sy+bend,z+.000012*math.sin(k*.3)))
        path(grain,points,rng.uniform(.00010,.00022),4)
    for u in (.025,width-.025):
        if u<0 or u>width: continue
        for v in (.026,height-.026):
            nails.branch([(x+u,y+v,z-.0001),(x+u,y+v,z+.00006)],[.0014,.0014],8)


def decorative_joinery(m):
    wood=m.part('Carved_walnut',(.20,.095,.041))
    brass=m.part('Furniture_brass',(.55,.36,.13))
    wax=m.part('Candle_wax_drips',(.78,.71,.53))
    wick=m.part('Charred_wicks',(.024,.019,.013))
    # Fluted and gadrooned candlestick feet, drip pans, and hanging wax runs.
    for x,y,h in ((.70,3.56,.10),(1.16,3.58,.13),(1.86,3.50,.16)):
        for k in range(24):
            a=k*TAU/24
            points=[(x+r*math.cos(a),y+r*math.sin(a),.7854+z) for r,z in ((.040,.008),(.028,.012),(.016,.020))]
            path(brass,points,.00065,5)
        for k in range(6):
            a=k*TAU/6+.7
            points=[(x+.0108*math.cos(a),y+.0108*math.sin(a),.8555+h-v) for v in (.001,.01,.014+k*.003)]
            path(wax,points,.00125,7)
        path(wick,[(x,y,.8555+h-.001),(x+.001,y,.8555+h+.005)],.00045,6)
    # Carved rosettes and scrollwork on the apron, in the vertical front plane.
    for x in (.58,1.40):
        for dx in (-.24,.24):
            # Build an XY carving, then rotate its relief into the drawer front.
            temp=Mesh();scroll(temp,(0,0,0),.025,0,.0011)
            base=len(wood.vertices)
            wood.vertices.extend((x+dx+p[0],2.927-p[2],.655+p[1]) for p in temp.vertices)
            wood.faces.extend(tuple(base+i for i in face) for face in temp.faces)
    # Leg collars and stretcher joinery: exposed turned details instead of bare poles.
    for x in (.26,1.74):
        for y in (2.99,3.61):
            for z,r in ((.14,.033),(.19,.054),(.27,.035),(.65,.055)):
                wood.ring((x,y,z),r+.001,.0035,steps=48)
            wood.box((x,y,.075),(.076,.076,.05),.008)
        wood.box((x,3.30,.20),(.035,.64,.043),.005)
    # Thin drawer edging and escutcheon plates.
    for x in (.58,1.40):
        brass.box((x,2.915,.661),(.033,.0018,.018),.001)
        for dx in (-.013,.013): brass.branch([(x+dx,2.912,.66),(x+dx,2.909,.66)],[.0016,.0016],8)
