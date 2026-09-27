"""Fast checks for the modelling recipes; no Blender or renderer needed."""
import random
import unittest

from build_scene_models import Mesh, Model, cross, sub
from scene_details import book, relief_slab


class DetailTests(unittest.TestCase):
    def test_book_tooling_stays_on_cover(self):
        model = Model('test', False)
        book(model, (.4, 3.1, .8), .28, .25, .042, -.17, 'URANOGRAPHIA', (.2,.04,.02), 4)
        for job in model.text_jobs:
            self.assertAlmostEqual(job['position'][2], .84215)
        for mesh in model.meshes.values():
            for x,y,z in mesh.vertices:
                self.assertLess(abs(x-.4), .22)
                self.assertLess(abs(y-3.1), .22)
                self.assertGreater(z, .79)
                self.assertLess(z, .85)

    def test_relief_slab_closed_and_non_degenerate_after_quantization(self):
        mesh = Mesh()
        relief_slab(mesh, (2,3,.03), (1.2,.8,.06), random.Random(7), angle=.2)
        vertices = [tuple(round(c,6) for c in p) for p in mesh.vertices]
        edges = {}
        for face in mesh.faces:
            a,b,c = [vertices[i] for i in face]
            self.assertGreater(sum(v*v for v in cross(sub(b,a),sub(c,a))), 1e-20)
            for p,q in ((a,b),(b,c),(c,a)):
                key = tuple(sorted((p,q)))
                edges[key] = edges.get(key,0)+1
        self.assertTrue(all(count==2 for count in edges.values()))


if __name__ == '__main__':
    unittest.main()
