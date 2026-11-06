"""Model-free A4 phase/schedule arithmetic."""
import math
from .inputs import check

PHASES = {'reference':1000, 'components':2000, 'guided_joint':1000,
          'transition':2000, 'free_joint':1000}


def schedule(phase, window):
    total = PHASES[phase]
    check(1 <= window <= total, 'phase window')
    warmup = math.ceil(.05*total)
    return window/warmup if window <= warmup else .1+.45*(1+math.cos(math.pi*(window-warmup)/(total-warmup)))


def scheduled_rate(phase, window, profile='a4-v1'):
    """A4 remains bounded; explicit reference continuation has constant peak LR."""
    check(type(window) is int and window >= 1, 'schedule window')
    if profile == 'reference-coverage-v1':
        check(phase == 'reference', 'coverage schedule is reference-only')
        return 1.
    check(profile == 'a4-v1', 'unknown schedule profile')
    return schedule(phase, window)


def guided_events(phase, window, events, *, compatible=True):
    check(phase in PHASES and 1 <= window <= PHASES[phase] and events >= 0, 'guidance schedule')
    if not compatible:
        return 0
    if phase == 'guided_joint':
        return events
    if phase == 'transition':
        return math.floor((1-(window-1)/1999)*events)
    return 0



def guidance_prefix(phase, window, reviewed_events, *, first_incompatible=None):
    """Actual contiguous guidance stops at divergence; never paste gold later."""
    count = guided_events(phase,window,len(reviewed_events))
    if first_incompatible is not None:
        check(type(first_incompatible) is int and 0 <= first_incompatible <= len(reviewed_events), 'divergence boundary')
        count = min(count,first_incompatible)
    return tuple(reviewed_events[:count])
