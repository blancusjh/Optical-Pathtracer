"""Fast authoring invariants, independent of the tracing implementation."""
from array import array
from bisect import bisect_right
import json
import random
import unittest
from build_scene_models import Mesh, cross, sub, ROOT
from build_landscapes import Terrain, rock, graded_axis, ground


class LandscapeTests(unittest.TestCase):
    def terrain(self):
        t=Terrain.__new__(Terrain)
        t.scene='the_telescope';t.sx=t.sy=8;t.n=2;t.base=0
        t.heights=array('d',[0,1,2,2,3,4,4,5,6])
        return t

    def test_native_support_triangle_interpolation(self):
        t=self.terrain()
        self.assertAlmostEqual(t.support(-2,-2),1.5)
        self.assertAlmostEqual(t.support(1,2),4.25)

    def test_surface_sampling_follows_generated_triangles(self):
        t=self.terrain();t.xs=t.ys=[-4,0,4]
        t.surface_mesh=Mesh();t.surface_mesh.vertices=[(x,y,t.support(x,y)) for y in t.ys for x in t.xs]
        for x,y in ((-3,-1),(1,3),(2,-1)):
            self.assertAlmostEqual(t.height(x,y),t.support(x,y))

    def test_protected_optical_and_architectural_footprints(self):
        t=self.terrain()
        self.assertEqual(t.protected(0,0),0)
        self.assertEqual(t.protected(450,0),0)
        t.scene='the_temple'
        for x,y in ((0,20),(-22,0),(22,48),(0,-12)):self.assertEqual(t.protected(x,y),0)
        self.assertEqual(t.protected(-30,0),1)
        t.scene='the_observatory';self.assertEqual(t.protected(0,-8),0)

    def test_graded_grid_has_no_duplicate_edges(self):
        axis=graded_axis(2000)
        self.assertEqual(axis[0],-2000);self.assertEqual(axis[-1],2000)
        self.assertTrue(all(b>a for a,b in zip(axis,axis[1:])))

    def test_boulder_is_closed_and_non_degenerate(self):
        m=Mesh();rock(m,(2,3,4),(1,.7,.8),random.Random(5),.3)
        edges={}
        for face in m.faces:
            a,b,c=[tuple(round(v,6) for v in m.vertices[k]) for k in face]
            self.assertGreater(sum(v*v for v in cross(sub(b,a),sub(c,a))),1e-15)
            for p,q in ((a,b),(b,c),(c,a)):
                key=tuple(sorted((p,q)));edges[key]=edges.get(key,0)+1
        self.assertTrue(all(v==2 for v in edges.values()))

    def test_exported_tree_roots_match_exported_ground(self):
        folder=ROOT/'models'/'landscapes'
        manifest=folder/'the_telescope.placement.json'
        if not manifest.exists():self.skipTest('Generate the landscape first')
        axis=graded_axis(2000);w=len(axis);heights=array('d')
        with (folder/'the_telescope_Ground_relief.obj').open() as f:
            for line in f:
                if line.startswith('f '):break
                if line.startswith('v '):heights.append(float(line.split()[3]))
        self.assertEqual(len(heights),w*w)
        for tree in json.loads(manifest.read_text()):
            x,y,z=tree['position'];i=bisect_right(axis,x)-1;j=bisect_right(axis,y)-1
            u=(x-axis[i])/(axis[i+1]-axis[i]);v=(y-axis[j])/(axis[j+1]-axis[j]);k=j*w+i
            a,b,c,d=heights[k],heights[k+1],heights[k+w+1],heights[k+w]
            surface=a+(b-a)*u+(c-b)*v if u>=v else a+(c-d)*u+(d-a)*v
            self.assertAlmostEqual(z+.08,surface,places=5)


if __name__=='__main__':unittest.main()
