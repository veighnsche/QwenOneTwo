"""Existing reviewed role views for reference fitting; never authors annotations."""
from dataclasses import dataclass
import random
from .inputs import SLOTS,canonical,digest,check
from .binding import annotation
from .component_producer import reference_record
from .release import ReviewedRelease


@dataclass(frozen=True)
class View:
    identity:str
    role:str
    operation:str
    content:bytes
    sources:tuple

    @property
    def sha256(self):
        return digest(self.content)


def catalogue(release):
    check(isinstance(release,ReviewedRelease) and release.manifest['purpose']=='training',
          'reference fitting requires a reviewed training release')
    unique={}
    def add(item,operation,row,path):
        key=item.sha256
        source=(row['id'],digest(canonical(row)),path)
        if key not in unique:unique[key]=[item,operation,[]]
        unique[key][2].append(source)
    for entry in release.manifest['records']:
        if not entry['selected']:continue
        row=release.record(entry['id']);check(row['split']=='train','foreign reference split')
        operation=row['operation']
        if row.get('training_category')=='reference':
            item,_=reference_record(release,row)
            add(item,operation,row,('model_inputs','meaning'))
            continue
        def request(fields,op,path):
            for role in ('question','target'):
                if role in fields:add(annotation(role,fields[role],op),op,row,path+(role,))
            for index,value in enumerate(fields.get('alternatives',())):
                add(annotation('alternative',value,op),op,row,path+('alternatives',index))
            for role in ('dimension','conditions','zero','one','midpoint'):
                if role in fields.get('rubric',{}):
                    add(annotation(role,fields['rubric'][role],op),op,row,path+('rubric',role))
        supervision=row.get('supervision',{})
        if 'request_semantics' in supervision:
            request(supervision['request_semantics'],operation,('supervision','request_semantics'))
        for index,call in enumerate(row.get('trajectory',{}).get('calls',())):
            request(call['request_target'],call['operation'],('trajectory','calls',index,'request_target'))
        # The outcome-bearing exception is confined to the explicit Reason view.
        # Verdict, winner, score, native answers and effort are never traversed.
        if supervision.get('reason') is not None:
            add(annotation('reason',supervision['reason'],operation),operation,row,('supervision','reason'))
    views=tuple(View(key,item.role,op,item.content,tuple(sources))
                for key,(item,op,sources) in sorted(unique.items()))
    check({v.role for v in views}==set(SLOTS),'reference role coverage')
    return views


class ViewSampler:
    """Four diversity exposures and four role-balanced exposures per window.

    Each diversity cycle covers every distinct view. Metadata-only deterministic
    shuffling is not a CPU model workload. Resume pins the entire source catalogue.
    """
    def __init__(self,views,release_hash):
        self.views=tuple(views);self.release_hash=release_hash;self.windows=0
        self.catalogue_hash=digest(canonical([dict(id=v.identity,role=v.role,operation=v.operation,
            content=digest(v.content),sources=v.sources) for v in self.views]))
        self.pools={'diversity':tuple(v.identity for v in views)}
        self.pools.update({role:tuple(v.identity for v in views if v.role==role) for role in SLOTS})
        check(all(self.pools.values()),'empty reference view pool')
        self.cursors={key:0 for key in self.pools};self.role_cursor=0
        self.orders={}

    def take(self,key):
        epoch,offset=divmod(self.cursors[key],len(self.pools[key]))
        cached=self.orders.get(key)
        if cached is None or cached[0]!=epoch:
            order=list(self.pools[key]);random.Random(f'17:reference-views-v1:{key}:{epoch}').shuffle(order)
            self.orders[key]=(epoch,order)
        self.cursors[key]+=1
        return self.orders[key][1][offset]

    def next_window(self):
        check(self.windows<1000,'reference revision block exhausted')
        ids=[self.take('diversity') for _ in range(4)]
        for _ in range(4):
            role=tuple(SLOTS)[self.role_cursor%len(SLOTS)];self.role_cursor+=1
            ids.append(self.take(role))
        self.windows+=1
        return tuple(ids)

    def state_dict(self):
        return dict(protocol='reference-views-v1',release_hash=self.release_hash,catalogue_hash=self.catalogue_hash,
                    windows=self.windows,cursors=dict(self.cursors),role_cursor=self.role_cursor,seed=17)

    def load_state_dict(self,state):
        check(type(state['windows']) is int and 0<=state['windows']<=1000,'reference sampler window')
        expected=ViewSampler(self.views,self.release_hash)
        for _ in range(state['windows']):expected.next_window()
        check(state==expected.state_dict(),'reference sampler identity/cursor drift')
        self.windows=expected.windows;self.cursors=expected.cursors
        self.role_cursor=expected.role_cursor;self.orders=expected.orders


class CompleteViewSampler:
    """One exposure per unique view per shuffled pass, with exact resume."""
    def __init__(self, views, release_hash):
        self.views = tuple(views)
        self.release_hash = release_hash
        self.catalogue_hash = ViewSampler(self.views, release_hash).catalogue_hash
        self.ids = tuple(v.identity for v in self.views)
        check(len(self.ids) == len(set(self.ids)) and len(self.ids) % 8 == 0,
              'complete coverage needs distinct views and whole eight-example windows')
        self.windows = 0
        self.order = None

    @property
    def windows_per_pass(self):
        return len(self.ids)//8

    def next_window(self):
        epoch, offset = divmod(self.windows*8, len(self.ids))
        if offset == 0 or self.order is None:
            self.order = list(self.ids)
            random.Random(f'17:reference-complete-v1:{epoch}').shuffle(self.order)
        result = tuple(self.order[offset:offset+8])
        self.windows += 1
        return result

    def state_dict(self):
        return dict(protocol='reference-complete-v1', release_hash=self.release_hash,
                    catalogue_hash=self.catalogue_hash, windows=self.windows, seed=17)

    def load_state_dict(self, state):
        check(type(state.get('windows')) is int and state['windows'] >= 0, 'coverage sampler cursor')
        expected = dict(self.state_dict(), windows=state['windows'])
        check(state == expected, 'coverage sampler identity/cursor drift')
        self.windows = state['windows']; self.order = None
        # next_window reconstructs the current pass without replaying old passes.


def coverage_panel(views, required):
    """Fixed outcome-independent role coverage, retaining prior diagnostics."""
    selected = set(required)
    check(selected <= {v.identity for v in views}, 'foreign coverage diagnostic')
    for role in SLOTS:
        candidates = sorted((v for v in views if v.role == role),
                            key=lambda v: digest(('17:coverage-panel:'+v.identity).encode()))
        count = sum(v.identity in selected for v in candidates)
        for view in candidates:
            if count >= min(32, len(candidates)):break
            if view.identity not in selected:
                selected.add(view.identity); count += 1
    return tuple(v.identity for v in views if v.identity in selected)


def coverage_plateau(curves):
    """Three complete shuffled passes without a best recovery or loss."""
    stale = 0; best_exact = -1; best_loss = float('inf')
    for curve in curves:
        loss = curve.get('mean_reconstruction')
        improved = curve['panel_exact'] > best_exact or (loss is not None and loss < best_loss)
        best_exact = max(best_exact, curve['panel_exact'])
        if loss is not None:best_loss = min(best_loss, loss)
        stale = 0 if improved else stale+1
    return stale >= 3
