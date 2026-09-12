import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import yaml
from src.depth_target import runs


class ReleaseTests(unittest.TestCase):
    def test_legacy_reports_and_rejects_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'a.csv'
            original = b'a,b\r\n1,2\r\n'
            p.write_bytes(original.replace(b'\r\n', b'\n'))
            runs.write_json(root / 'release_manifest.json', {'files': [
                {'path': 'a.csv', 'sha256': hashlib.sha256(original).hexdigest()}]})
            self.assertEqual(runs.seal_report(root, root, True)[1]['files'][0]['status'], 'legacy_crlf_reconstructed')
            for content, status in [(b'changed\n', 'mismatch'), (None, 'missing')]:
                if content is None:
                    p.unlink()
                else:
                    p.write_bytes(content)
                with self.assertRaises(runs.SealError) as ctx:
                    runs.seal_report(root, root, True)
                self.assertEqual(ctx.exception.report['files'][0]['status'], status)

    def test_new_seal_checkout_and_strictness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT)
            git('init')
            (repo / '.gitattributes').write_text('* text=auto eol=lf\n', newline='\n')
            out = repo / 'run'
            out.mkdir()
            source = root / 'source.py'
            source.write_bytes(b'x=1\r\n')
            (out / 'source.py').write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, 'must use LF'):
                runs.seal(out)
            runs.copy_source(source, out / 'source.py')
            runs.write_json(out / 'result.json', {'value': 1})
            runs.seal(out)
            git('add', '.')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
            checkout = root / 'checkout'
            subprocess.check_output(['git', 'clone', '--quiet', str(repo), str(checkout)], stderr=subprocess.STDOUT)
            report = runs.seal_report(checkout / 'run', checkout, True)[1]
            self.assertEqual(report['exact'], 2)
            (checkout / 'run/source.py').write_bytes(b'x=1\r\n')
            with self.assertRaises(runs.SealError):
                runs.seal_report(checkout / 'run', checkout, True)

    def test_binding_rejects_config_input_and_feature_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = root / 'contract'
            contract.mkdir()
            audit = root / 'audit'
            audit.mkdir()
            config = {'inputs': {'audit_run': 'audit', 'height_package': 'pack.npz'}}
            paths = [('height_package', root / 'pack.npz'), ('frozen_targets_E00', audit / 'frozen_targets.csv'),
                     ('frozen_outer_split_E00', audit / 'split_manifest.csv'), ('frozen_inner_split_E00', audit / 'inner_split_manifest.csv')]
            for role, path in paths:
                path.write_bytes(b'fixed\n')
            (audit / 'release_manifest.json').write_bytes(b'{}')
            snapshot = dict(config, audit_release_sha256=runs.sha256(audit / 'release_manifest.json'))
            (contract / 'protocol_snapshot.yaml').write_text(yaml.safe_dump(snapshot))
            (contract / 'input_provenance.csv').write_text('path,sha256,role\n'+''.join(f'{p.relative_to(root).as_posix()},{runs.sha256(p)},{r}\n' for r,p in paths))
            runs.write_json(contract / 'release_manifest.json', {'format': 'depth_target_lf_v1'})
            for name in ['features.py','splits.py']:
                for base in [root, contract / 'source_snapshot']:
                    p = base / 'src/depth_target' / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b'fixed\n')
            with patch.object(runs, 'ROOT', root):
                self.assertEqual(len(runs.verify_contract_binding(contract, config)), 4)
                changed = copy.deepcopy(config)
                changed['pixel_um'] = 2
                with self.assertRaisesRegex(ValueError, 'Configuration drift'):
                    runs.verify_contract_binding(contract, changed)
                for _, path in paths:
                    path.write_bytes(b'drift')
                    with self.assertRaisesRegex(ValueError, 'Input drift'):
                        runs.verify_contract_binding(contract, config)
                    path.write_bytes(b'fixed\n')
                (root / 'src/depth_target/features.py').write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError, 'implementation drift'):
                    runs.verify_contract_binding(contract, config)
