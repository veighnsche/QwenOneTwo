"""Model-free binding/accounting for the finite larger-block experiment."""
from collections import Counter
import unittest
from scripts.benchmark_attention_block import benchmark_plan,run


class AttentionBlockPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=benchmark_plan('/metadata-only/no-model-read')

    def test_existing_coverage_and_all_presentations_disclosed(self):
        p=self.prepared
        self.assertEqual(len(p['cases']),8);self.assertEqual(len(p['anchors']),3)
        self.assertEqual(p['presentations'],{'main':144,'startup_anchors':3,'distinct_existing_cases':11})
        counts=Counter()
        for job in p['jobs']:
            counts[job['phase']]+=len(job['ids'])
            if job['panel']=='X':self.assertEqual(len(job['ids']),2)
        self.assertEqual(counts,{'reference_prewarm':8,'reference_parity':8,'warmup':48,'timed':48,
                                 'observation_warmup':16,'observe_warm':16})
        self.assertEqual(len(p['jobs']),126)
        refresh=[j for j in p['jobs'] if j['refresh_reference_cache']]
        self.assertEqual(len(refresh),1);self.assertEqual(refresh[0]['phase'],'reference_parity')

    def test_timed_pairs_have_no_profiler_or_counters(self):
        jobs=self.prepared['jobs'];timed=[j for j in jobs if j['phase']=='timed']
        self.assertTrue(all(not j['count_cache'] and not j['trace_attention_block'] for j in timed))
        self.assertEqual([next(j['attention_block_backend'] for j in timed if j['cache_trial']==i) for i in range(6)],
                         ['eager','graph-v1','graph-v1','eager','eager','graph-v1'])
        self.assertEqual([j['attention_block_backend'] for j in jobs if j['trace_attention_block']],['eager','graph-v1'])
        self.assertLess(max(i for i,j in enumerate(jobs) if j['phase']=='timed'),
                        min(i for i,j in enumerate(jobs) if j['trace_attention_block']))
        self.assertTrue(all(j['cache_limit']==32 for j in jobs))

    def test_sources_bind_both_execution_and_measurement(self):
        for name in ('qwen_one_two/codec/attention_block.py','qwen_one_two/codec/backbone.py',
                     'scripts/qualify_attention_block.py','scripts/benchmark_attention_block.py'):
            self.assertEqual(len(self.prepared['sources'][name]),64)

    def test_execution_opt_in_precedes_any_model_import_or_output(self):
        with self.assertRaisesRegex(ValueError,'explicit --execute-gpu'):run(None,None)


if __name__=='__main__':unittest.main()
