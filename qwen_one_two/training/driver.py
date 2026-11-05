"""A4 window orchestration for future reviewed-data runs; no CLI/auto-launch.

The producer executes a complete episode exactly once and returns normalized
component losses plus its Session. It owns task-specific reviewed supervision;
this driver cannot relabel a divergent packet or supply gold joint returns.
"""
from .inputs import check, canonical
from .release import bind_pilot, decode
from .qualification import NumericEvidence


def collect_session(*args):
    # Release/schedule preflight imports no model or CUDA runtime.
    from .episodes import collect_session as collect
    return collect(*args)


class Driver:
    def __init__(self, window, sampler, release, pilot, prerequisites, *, numeric=None):
        check(window.phase == sampler.phase and window.phase != 'reference', 'native driver phase')
        check(window.scheduled_windows == sampler.windows, 'sampler/window cursor mismatch')
        bind_pilot(release, pilot)
        sampler.assert_release(release)
        check(prerequisites.get('g1_core') is True, 'G1 prerequisite')
        if window.phase in {'guided_joint','transition','free_joint'}:
            check(isinstance(numeric, NumericEvidence), 'loaded 1024 qualification evidence required')
        check('numeric_1024' not in prerequisites, 'replace numeric attestation dictionary with loaded evidence')
        self.window,self.sampler = window,sampler
        self.release, self.pilot = release, pilot
        self.numeric = numeric
        self._prerequisites = canonical(prerequisites)
        self.reports = []
        self.failed = False

    @property
    def records(self):
        return self.release.records

    def run_window(self, producer):
        check(not self.failed, 'failed window requires explicit checkpoint resume')
        check(self.window.examples == 0 and self.sampler.windows == self.window.scheduled_windows, 'window boundary')
        self.sampler.assert_release(self.release)
        records = self.records
        ids = self.sampler.next_window()
        exposures = []
        try:
            for key in ids:
                # No count-first replay: every scheduled exposure executes once.
                # A producer may mutate its own copy; repeated exposures always
                # begin from the immutable reviewed bytes, without label drift.
                session,losses,exposure = producer(decode(canonical(records[key])),self.window.phase,self.window.scheduled_windows+1)
                exposures.append(dict(record=key,**collect_session(self.window,session,losses,exposure)))
                del session,losses
            result = self.window.step()
            report = {'samples':ids,'exposures':exposures,'window':result,'sampler':self.sampler.state_dict(),
                      'release_sha256':self.release.identity,'pilot_release_sha256':self.pilot.identity,
                      'prerequisites':decode(self._prerequisites),
                      'numeric_1024':None if self.numeric is None else self.numeric.summary}
            self.reports.append(report)
            return report
        except BaseException as exc:
            self.window.abort()
            self.failed = True
            self.reports.append({'samples':ids,'exposures':exposures,'error':str(exc),'status':'ABORTED',
                                 'sampler':self.sampler.state_dict(),'release_sha256':self.release.identity,
                                 'pilot_release_sha256':self.pilot.identity})
            # Keep failed exposure IDs/cursor. Never silently replace examples,
            # retry a truncated graph, or claim a partial window update.
            raise
