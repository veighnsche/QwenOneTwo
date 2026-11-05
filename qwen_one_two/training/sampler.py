"""Deterministic A4 exposure scheduling; no data generation or model execution."""
from collections import defaultdict
import random
from types import MappingProxyType
from .profile import PHASES
from .inputs import canonical, digest, check

RECIPES = {'reference':('reference',)*8,
           'components':('writer','writer','receiver','receiver','reader','reader','language_replay','decision_replay')}
JOINT = ('interleaved',)*4+('direct',)*2+('language_replay','decision_replay')
NONASSESSED = {'Noul':('Insufficient','Conflicting','Unresolved'),
               'Choice':('Insufficient','Conflicting','Unresolved','NoFeasibleChoice'),
               'Score':('Insufficient','Conflicting','Unresolved')}


class Sampler:
    """IDs must come from an independently validated named training release.

    Component decision pools balance operation/status/domain across cycles.
    This is a training exposure recipe, never a runtime invocation quota.
    """
    def __init__(self, rows, phase, *, release_hash, seed=17):
        from .release import sha256
        sha256(release_hash)
        check(phase in PHASES and seed == 17, 'sampler profile')
        self.phase, self.release_hash, self.seed = phase,release_hash,seed
        catalogue = {row['id']:dict(row) for row in rows}
        check(len(catalogue) == len(rows), 'duplicate sample ID')
        self.catalogue_hash = digest(canonical(catalogue))
        pools, domains = defaultdict(list), defaultdict(set)
        for row in catalogue.values():
            check(row['split'] == 'train', 'nontraining sampler record')
            category = row['category']
            if category in {'writer','receiver'}:
                operation,status,domain = row['operation'],row['status'],row['domain']
                check(operation in NONASSESSED and status in ('Assessed',)+NONASSESSED[operation], 'sampler operation/status')
                prefix = (category,operation,status)
                domains[prefix].add(domain)
                key = (*prefix,domain)
            else:
                key = (category,)
            pools[key].append(row['id'])
        self._rows = MappingProxyType({key:MappingProxyType(row) for key,row in catalogue.items()})
        self._pools = MappingProxyType({key:tuple(ids) for key,ids in pools.items()})
        self._domains = MappingProxyType({key:frozenset(values) for key,values in domains.items()})
        self.cursors = defaultdict(int)
        self.windows = 0

    @property
    def rows(self):
        return self._rows

    @property
    def pools(self):
        return self._pools

    @property
    def domains(self):
        return self._domains

    @classmethod
    def from_release(cls, release, phase):
        from .release import ReviewedRelease
        check(isinstance(release, ReviewedRelease), 'load a reviewed training release first')
        return cls(release.sampler_rows(), phase, release_hash=release.identity)

    def assert_release(self, release):
        """Verify every ID and operation/status/domain, not just category names."""
        from .release import ReviewedRelease
        check(isinstance(release, ReviewedRelease), 'load a reviewed training release first')
        expected = {row['id']:row for row in release.sampler_rows()}
        check(self.release_hash == release.identity, 'sampler/release identity mismatch')
        check(self.rows == expected and self.catalogue_hash == digest(canonical(expected)),
              'sampler/release catalogue mismatch')

    def _counter(self, key):
        value = self.cursors[key]; self.cursors[key] += 1
        return value

    def _take(self, category):
        if category in {'writer','receiver'}:
            op = ('Noul','Choice','Score')[self._counter(('operation',category)) % 3]
            index = self._counter(('status',category,op))
            status = 'Assessed' if index % 2 == 0 else NONASSESSED[op][(index//2)%len(NONASSESSED[op])]
            prefix = (category,op,status)
            domains = sorted(self.domains.get(prefix, ()))
            check(bool(domains), 'missing operation/status coverage: '+str(prefix))
            domain = domains[self._counter(('domain',*prefix)) % len(domains)]
            key = (*prefix,domain)
        else:
            key = (category,)
        pool = sorted(self.pools.get(key, ()))
        check(bool(pool), 'missing scheduled category: '+category)
        index = self._counter(('pool',*key))
        epoch,offset = divmod(index,len(pool))
        random.Random(f'{self.seed}:{key}:{epoch}').shuffle(pool)
        return pool[offset]

    def next_window(self):
        check(self.windows < PHASES[self.phase], 'phase sampler exhausted')
        previous = dict(self.cursors)
        try:
            ids = tuple(self._take(category) for category in RECIPES.get(self.phase,JOINT))
        except BaseException:
            self.cursors = defaultdict(int,previous)
            raise
        self.windows += 1
        return ids

    def state_dict(self):
        return {'phase':self.phase,'release_hash':self.release_hash,'catalogue_hash':self.catalogue_hash,
                'seed':self.seed,'windows':self.windows,
                'cursors':[{'key':list(key),'value':value} for key,value in sorted(self.cursors.items())]}

    def load_state_dict(self,state):
        for field in ('phase','release_hash','catalogue_hash','seed'):
            check(state[field] == getattr(self,field), 'sampler identity mismatch: '+field)
        check(type(state['windows']) is int and 0 <= state['windows'] <= PHASES[self.phase], 'sampler window cursor')
        cursors = {tuple(item['key']):item['value'] for item in state['cursors']}
        check(len(cursors) == len(state['cursors']) and all(type(v) is int and v >= 0 for v in cursors.values()), 'sampler cursor')
        # Window number alone must not allow skipping/replacing exposures. This
        # is identifier-only schedule replay, never a model or rollout replay.
        expected = Sampler(list(self.rows.values()), self.phase, release_hash=self.release_hash, seed=self.seed)
        for _ in range(state['windows']):
            expected.next_window()
        check(cursors == dict(expected.cursors), 'sampler cursors do not match scheduled exposures')
        self.windows = state['windows']; self.cursors = defaultdict(int,cursors)
