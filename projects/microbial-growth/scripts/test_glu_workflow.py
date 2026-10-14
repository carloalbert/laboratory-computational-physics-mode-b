"""Regression checks for the consolidated workflow and retained data/split."""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from fit_empirical_detrended_hazard import cohort
from glu_workflow import PROJECT_DIR, load_data, load_config, validation_ids


class WorkflowTests(unittest.TestCase):
    def test_data_and_split_are_unchanged(self):
        config=load_config();data,groups=load_data();balanced=cohort(groups)
        self.assertEqual((len(data),len(groups)),(6504,303))
        expected=config['expected_cohort'];ids=balanced['ids'];valid=validation_ids()
        self.assertEqual(len(ids),expected['lineages'])
        self.assertEqual(balanced['g'].size,expected['transitions'])
        self.assertEqual(sum(v in valid for v in ids),expected['validation_lineages'])
        self.assertEqual(sum(v not in valid for v in ids),expected['training_lineages'])
        digest=hashlib.sha256((PROJECT_DIR/'data/processed/processed_input_data.csv').read_bytes()).hexdigest()
        self.assertEqual(digest,config['processed_data_sha256'])

    def test_current_entrypoints_do_not_import_retired_experiments(self):
        # Fresh interpreter avoids modules loaded by historical test discovery.
        code="""import json,sys
import fit_static_offset_detrended_hazard
import plot_static_offset_detrended_hazard
import plot_empirical_detrended_hazard
print(json.dumps(sorted(sys.modules)))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=PROJECT_DIR/'scripts',check=True,capture_output=True,text=True)
        loaded=json.loads(result.stdout)
        for name in ['screen_parsimonious_joint_hazards','fit_joint_growth_timing','run_stationary_memory_time_hazard',
                     'fit_empirical_birth_transitions','plot_joint_growth_timing','optimize_birth_size_hazard_acf']:
            self.assertNotIn(name,loaded)

    def test_retained_output_dependencies_are_explicit(self):
        for name in ['fit_empirical_detrended_hazard.py','fit_static_offset_detrended_hazard.py']:
            source=(PROJECT_DIR/'scripts'/name).read_text()
            self.assertNotIn('empirical_birth_transitions/fit_results.json',source)
        self.assertEqual(len(validation_ids()),len(load_config()['validation_ids']))


if __name__=='__main__':unittest.main()
