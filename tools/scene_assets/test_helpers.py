"""Small temporary fixtures; never read or regenerate production assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('asset_checker', HERE / 'check_assets.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class Helpers(unittest.TestCase):
    def test_recursive_missing_buffer_and_provider_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'scenes').mkdir(); (root / 'assets').mkdir()
            for name in checker.SCENES:
                (root / 'scenes' / (name + '.owe')).write_text('body X { file = "../assets/model.gltf" }')
            (root / 'assets/model.gltf').write_text(json.dumps({'buffers': [{'uri':'missing.bin'}]}))
            missing = [p for p in checker.dependencies(root) if not p.exists()]
            self.assertEqual(missing, [root / 'assets/missing.bin'])
            d = root / 'docs/assets/polyhaven/fixture'; d.mkdir(parents=True)
            (d / 'files.json').write_text(json.dumps({'url':'https://example.com/a', 'md5':hashlib.md5(b'a').hexdigest()}))
            self.assertEqual(checker.provider_hashes(root)['https://example.com/a'], hashlib.md5(b'a').hexdigest())
            (root / 'docs/assets/polyhaven.json').write_text(json.dumps([{'files':[{'path':'assets/missing.bin','url':'https://example.com/a'}]}]))
            (root / 'assets/missing.bin').write_bytes(b'b')
            result = subprocess.run([sys.executable, str(HERE/'check_assets.py'),'--root',str(root),'--integrity'],capture_output=True,text=True)
            self.assertEqual(result.returncode, 1); self.assertIn('MD5 mismatch',result.stdout)

    def test_derivative_preserves_dimensions_existing_output_and_scene(self):
        from PIL import Image
        spec = importlib.util.spec_from_file_location('texture_derivatives', HERE / 'derive_textures.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene = root / 'scene.owe';scene.write_text('unchanged')
            src = root / 'source.png'; dst = root / 'target.png'
            Image.new('RGB',(3,2),(255,128,64)).save(src)
            module.derive(src,dst,'pigment',[.5,.5,.5])
            with Image.open(dst) as im: self.assertEqual(im.size,(3,2));self.assertLess(im.getpixel((0,0))[0],255)
            before = dst.read_bytes();module.derive(src,dst,'steel');self.assertEqual(dst.read_bytes(),before)
            self.assertEqual(scene.read_text(),'unchanged')

    def test_classification_writes_assets_only(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'assets/sketchfab/armor_man_and_horse';(source/'textures').mkdir(parents=True)
            indexes=np.array([0,1,2],dtype='<u4').tobytes();uv=np.array([[0,0],[0,0],[0,0]],dtype='<f4').tobytes()
            (source/'scene.bin').write_bytes(indexes+uv)
            data={'buffers':[{'uri':'scene.bin','byteLength':len(indexes+uv)}],'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':12},{'buffer':0,'byteOffset':12,'byteLength':24}], 'accessors':[{'bufferView':0,'componentType':5125,'count':3,'type':'SCALAR'},{'bufferView':1,'componentType':5126,'count':3,'type':'VEC2'}], 'meshes':[{'primitives':[{'indices':0,'attributes':{'TEXCOORD_0':1}}]}]}
            (source/'scene.gltf').write_text(json.dumps(data));Image.new('RGB',(1,1),(255,0,0)).save(source/'textures/Armor_Mat_baseColor.jpeg')
            (root/'scenes').mkdir();scene=root/'scenes/the_kunstkammer.owe';scene.write_text('authoritative')
            result=subprocess.run([sys.executable,str(HERE/'classify_armour.py')],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            out=json.loads((source/'curated/scene.gltf').read_text());self.assertEqual(out['meshes'][0]['primitives'][1]['material'],1);self.assertEqual(scene.read_text(),'authoritative')
            original=(source/'curated/classification.bin').read_bytes()
            result=subprocess.run([sys.executable,str(HERE/'classify_armour.py')],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((source/'curated/classification.bin').read_bytes(),original)


if __name__=='__main__': unittest.main()
