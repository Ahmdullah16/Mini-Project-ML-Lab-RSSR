from pathlib import Path
import unittest
import statistics
import torch
from benchmark_tools import comparison
from rssr.models import SRModel


class BenchmarkChecks(unittest.TestCase):
    def test_source_comparison(self):
        report=comparison(Path(__file__).resolve().parents[2]);self.assertEqual(len(report['rows']),6)
        for summary in report['summary']:
            rows=[r for r in report['rows'] if r['model']==summary['model']]
            self.assertEqual(summary['mean_delta_db'],statistics.mean(r['delta_psnr'] for r in rows))
            self.assertEqual(summary['seed_sd_delta_db'],statistics.stdev(r['delta_psnr'] for r in rows))
            for r in rows:self.assertAlmostEqual(r['model_psnr']-r['bicubic_psnr'],r['delta_psnr'],places=10)
        self.assertLess(report['summary'][1]['mean_delta_db'],report['summary'][0]['mean_delta_db'])

    def test_model_contract_and_parameter_counts(self):
        report=comparison(Path(__file__).resolve().parents[2]);torch.manual_seed(31)
        for row in report['summary']:
            model=SRModel(bands=4,width=16,kind=row['model'],scale=2).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),row['parameters'])
            with torch.no_grad():y=model(torch.zeros(1,4,7,9))
            self.assertEqual(tuple(y.shape),(1,4,14,18));self.assertTrue(torch.isfinite(y).all())

    def test_zero_residual_is_bicubic(self):
        model=SRModel(bands=4,width=16,kind='transformer',zero_residual=True).eval();x=torch.rand(1,4,7,9)
        with torch.no_grad():y=model(x)
        torch.testing.assert_close(y,torch.nn.functional.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False))


if __name__=='__main__':unittest.main()
