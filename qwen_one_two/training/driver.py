"""A4 window orchestration for future reviewed-data runs; no CLI/auto-launch.

The producer executes a complete episode exactly once and returns normalized
component losses plus its Session. It owns task-specific reviewed supervision;
this driver cannot relabel a divergent packet or supply gold joint returns.
"""
import json
from .inputs import check, validate_record_manifest, canonical
from .episodes import collect_session


class Driver:
    def __init__(self, window, sampler, records, parents, prerequisites):
        check(window.phase == sampler.phase and window.phase != 'reference', 'native driver phase')
        check(window.scheduled_windows == sampler.windows, 'sampler/window cursor mismatch')
        check(prerequisites.get('g1_core') is True and bool(prerequisites.get('pilot_review_hash')), 'reviewed pilot/G1 prerequisite')
        if window.phase in {'guided_joint','transition','free_joint'}:
            numeric = prerequisites.get('numeric_1024', {})
            check(numeric.get('protocol') == 'v3-1024-v1' and numeric.get('protocol_complete') is True
                  and numeric.get('status') == 'PASSED' and bool(numeric.get('report_hash')), '1024 qualification prerequisite')
        self.window,self.sampler = window,sampler
        self.records = json.loads(canonical(records))
        for key in sampler.rows:
            check(key in records, 'sampler record missing from release')
            check(self.records[key]['id'] == key, 'record identity mismatch')
            check(self.records[key].get('training_category') == sampler.rows[key]['category'], 'sampler/category mismatch')
            validate_record_manifest(self.records[key],parents,for_training=True)
        self.reports = []
        self.failed = False

    def run_window(self, producer):
        check(not self.failed, 'failed window requires explicit checkpoint resume')
        check(self.window.examples == 0 and self.sampler.windows == self.window.scheduled_windows, 'window boundary')
        ids = self.sampler.next_window()
        exposures = []
        try:
            for key in ids:
                # No count-first replay: every scheduled exposure executes once.
                session,losses,exposure = producer(self.records[key],self.window.phase,self.window.scheduled_windows+1)
                exposures.append(dict(record=key,**collect_session(self.window,session,losses,exposure)))
                del session,losses
            result = self.window.step()
            report = {'samples':ids,'exposures':exposures,'window':result,'sampler':self.sampler.state_dict()}
            self.reports.append(report)
            return report
        except BaseException as exc:
            self.window.abort()
            self.failed = True
            self.reports.append({'samples':ids,'exposures':exposures,'error':str(exc),'status':'ABORTED',
                                 'sampler':self.sampler.state_dict()})
            # Keep failed exposure IDs/cursor. Never silently replace examples,
            # retry a truncated graph, or claim a partial window update.
            raise
