"""Verify historical depth releases from a clean Git checkout without staging the user workspace."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml
from src.depth_target import runs


def main():
    root = runs.ROOT
    reports = []
    with tempfile.TemporaryDirectory(prefix='depth-release-') as tmp:
        repo = Path(tmp) / 'repo'
        repo.mkdir()
        tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', 'outputs/depth_target_v1']).decode().split('\0')
        paths = [p for p in tracked if p] + ['.gitattributes']
        paths += [p.relative_to(root).as_posix() for p in runs.RUNS_DIR.glob('*/tests.log')]
        for relative in sorted(set(paths)):
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT)
        git('init')
        git('add', '.')
        git('-c', 'user.name=Release audit', '-c', 'user.email=audit@example.invalid', 'commit', '-m', 'Release candidate')
        checkout = Path(tmp) / 'checkout'
        subprocess.check_output(['git', 'clone', '--quiet', str(repo), str(checkout)], stderr=subprocess.STDOUT)
        for run in sorted((checkout / 'outputs/depth_target_v1').iterdir()):
            if run.is_dir() and (run / 'release_manifest.json').is_file():
                _, report = runs.verify_upstream_seal_tolerant(run, checkout)
                report['run'] = run.name
                reports.append(report)
    contract = next(runs.RUNS_DIR.glob('*D0*e7407e'))
    config = yaml.safe_load((root / 'config/depth_target_v1/protocol.yaml').read_text(encoding='utf-8'))
    binding = runs.verify_contract_binding(contract, config)
    output = runs.RUNS_DIR / 'release_repair_checkout_report.json'
    runs.write_json(output, {'status': 'PASS', 'scope': 'temporary Git commit and clean clone of release candidate; original repository index untouched', 'historical_runs': reports, 'D0_input_binding': binding})
    print(json.dumps({'status': 'PASS', 'runs': len(reports), 'output': str(output)}))


if __name__ == '__main__':
    main()
