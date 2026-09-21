#!/usr/bin/env python
"""Checked AE at 1-2-5 benchmarks; adaptive local resolution, no dense long-T scan.

Each benchmark is its own resumable NPZ. Separate N=1,3,5 short-T scans are
oscillation demonstrations. Exact matching points from compatible checked
scans can be reused; no interpolation or legacy unnormalized data is reused.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('AFMQEP_X64', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
AFM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AFM / 'code'))
import numpy as np
from scipy.signal import find_peaks

BENCHMARKS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
SCHEDULES = ('smooth', 'stiff')


def benchmark_grid(center, level=0):
    """Centered windows: full width min(20, 0.8*T), spacing <= .25 / 2**level."""
    if center <= 0 or level not in (0, 1):
        raise ValueError('Positive benchmark and resolution level 0 or 1 required')
    half = min(10., .4 * center)
    intervals = max(8, 4 * int(np.ceil((2 * half / .25) / 4))) * 2 ** level
    return np.linspace(center - half, center + half, intervals + 1)


def amplitude_statistics(T, gradients):
    mean = gradients.mean(axis=0)
    rms = np.sqrt(np.mean((gradients - mean) ** 2, axis=0))
    coarse = gradients[::2]
    coarse_rms = np.sqrt(np.mean((coarse - coarse.mean(axis=0)) ** 2, axis=0))
    change = float(np.linalg.norm(rms - coarse_rms) / max(np.linalg.norm(rms), 1e-14))
    peaks, troughs = [], []
    for j in (0, 1):
        prominence = max(.05 * np.ptp(gradients[:, j]), 1e-12)
        peaks.append(len(find_peaks(gradients[:, j], prominence=prominence)[0]))
        troughs.append(len(find_peaks(-gradients[:, j], prominence=prominence)[0]))
    return dict(mean=mean.tolist(), rms=rms.tolist(),
                half_peak_to_peak=(.5 * np.ptp(gradients, axis=0)).tolist(),
                half_grid_relative_change=change, peaks=peaks, troughs=troughs,
                resolved_cycles=bool(min(peaks + troughs) >= 3),
                window=[float(T[0]), float(T[-1])], points=len(T))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--centers', default=','.join(map(str, BENCHMARKS)))
    ap.add_argument('--example', action='store_true', help='Dense short-T demonstration instead of the 1-2-5 benchmark windows.')
    ap.add_argument('--example-max', type=float, default=60., help='Largest T in the dense example (default 60).')
    ap.add_argument('--example-min', type=float, default=.25,
                    help='Smallest T in the dense example (default .25). Cost per sample grows '
                         'with T, so a long scan can be split into equal-cost [min,max] chunks '
                         'run as separate parallel tasks and concatenated when plotting.')
    ap.add_argument('--example-step', type=float, default=.125,
                    help='Spacing of the dense example grid (default .125). The gradient '
                         'oscillation period is ~4-7 for every N here, so .5 still gives '
                         '8-13 samples per period at a quarter of the cost.')
    ap.add_argument('--Omega', type=float, default=1., help='Benchmark point (default: the study base point).')
    ap.add_argument('--Delta', type=float, default=2.6, help='Benchmark point (default: the study base point).')
    ap.add_argument('--beta', type=float, default=.1, help='Nudge strength (default: the study base value).')
    ap.add_argument('--cost', choices=('AM', 'AM2'), default='AM',
                    help='Cost operator (default: AM, as in the original bias study).')
    ap.add_argument('--allow-degenerate', action='store_true',
                    help='Permit N=1 with --cost AM2, whose gradient is identically zero.')
    ap.add_argument('--outdir', type=Path, default=AFM / 'paper_figs/data/bias_benchmarks')
    ap.add_argument('--reuse-dir', type=Path, default=AFM / 'paper_figs/data/bias_refined')
    args = ap.parse_args()
    if args.N not in (1, 3, 5, 7, 9):
        ap.error('Use the existing odd chains.')
    if not np.isfinite(args.example_max) or args.example_max < .25:
        ap.error('--example-max must be finite and at least .25')
    if args.cost == 'AM2' and args.N == 1 and not args.allow_degenerate:
        # AM = sigma^z for a single atom, so AM^2 = identity: <AM^2> = 1 in every
        # state, the loss is constant, and the gradient is identically zero.
        ap.error('N=1 with --cost AM2 has an identically zero gradient (AM^2 = 1). '
                 'Pass --allow-degenerate to record it anyway (round-off only).')
    import afmqep as A
    import dynamiqs as dq
    from afmqep.normalized_ae import NormalizedAE
    assert A.X64
    physics = dict(N=args.N, Omega=args.Omega, Delta=args.Delta, beta=args.beta, target=1., cost=args.cost,
                   rtol=1e-10, atol=1e-12, solver='Dopri8', normalized=True,
                   norm_tolerance=1e-5)
    qm = (A.Ising_Chain if args.cost == 'AM' else A.Ising_Chain_AM2)(args.N, T=1., dt=.1)
    params = qm.set_params(args.Omega, args.Delta)
    ged = np.array(A.make_grad_methods(args.beta)['ED'](params, qm, 1.)[1][:2], float)
    method = dq.method.Dopri8(rtol=1e-10, atol=1e-12, max_steps=4_000_000)
    estimators = {s: NormalizedAE(args.beta, s, method) for s in SCHEDULES}
    tight = {s: NormalizedAE(args.beta, s, dq.method.Dopri8(
        rtol=1e-12, atol=1e-14, max_steps=4_000_000)) for s in SCHEDULES}
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache = {s: {} for s in SCHEDULES}
    # Never round T: only bitwise-identical abscissae can be reused.
    candidates = sorted(args.reuse_dir.glob(f'bias_chain_N{args.N}_AM_refined*.npz'))
    candidates += sorted(args.outdir.glob(f'benchmark_chain_N{args.N}_*.npz'))
    for path in candidates:
        if path.name.endswith('.partial.npz'):
            continue  # Another window group may be writing an atomic checkpoint.
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['meta']))
            if any(meta.get(k) != v for k, v in physics.items()):
                continue
            np.testing.assert_allclose(d['g_ED'], ged, rtol=1e-8, atol=1e-10)
            for s in SCHEDULES:
                for i, T in enumerate(d['TL']):
                    g, norms = d[f'g_{s}_AE'][i], d[f'norm2_{s}'][i]
                    if np.isfinite(g).all() and np.isfinite(norms).all() and np.max(np.abs(norms - 1)) <= 1e-5:
                        cache[s][float(T)] = (g.copy(), float(d[f'y_{s}_AE'][i]), norms.copy(), str(path))
    centers = [None] if args.example else [float(t) for t in args.centers.split(',')]
    for center in centers:
        tag = 'example' if center is None else f'T{center:g}'
        if center is None and args.example_max != 60.:
            tag += f'_T{args.example_max:g}'
            if args.example_step != .125:
                tag += f'_s{args.example_step:g}'
            if args.example_min != .25:
                tag += f'_from{args.example_min:g}'
        path = args.outdir / f'benchmark_chain_N{args.N}_{tag}.npz'
        config = dict(physics, benchmark=center, example=center is None,
                      grid='centered_min20_0.8T', sampling_threshold=.05, max_level=1)
        if center is None and args.example_max != 60.:
            config.update(grid='dense_example', example_max=args.example_max,
                          example_step=args.example_step, example_min=args.example_min)
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                assert json.loads(str(previous['meta'])) == config, 'Different configuration; use a new directory.'
                if bool(previous.get('complete', False)):
                    print('Already complete:', path, flush=True)
                    continue
        result = None
        for level in (0, 1):
            TL = (np.arange(args.example_min, args.example_max + 1e-9, args.example_step)
                  if center is None else benchmark_grid(center, level))
            result = dict(TL=TL, g_ED=ged, meta=json.dumps(config), level=level, complete=False)
            for s in SCHEDULES:
                result[f'g_{s}_AE'] = np.full((len(TL), 2), np.nan)
                result[f'y_{s}_AE'] = np.full(len(TL), np.nan)
                result[f'norm2_{s}'] = np.full((len(TL), 2), np.nan)
                result[f'seconds_{s}'] = np.zeros(len(TL))
                result[f'reused_{s}'] = np.zeros(len(TL), bool)
            sources = set()

            def checkpoint():
                result['reuse_sources'] = json.dumps(sorted(sources))
                temporary = path.with_suffix('.partial.npz')
                np.savez_compressed(temporary, **result)
                temporary.replace(path)

            for s in SCHEDULES:
                for i, T in enumerate(TL):
                    key = float(T)
                    if key in cache[s]:
                        g, y, norms, source = cache[s][key]
                        result[f'reused_{s}'][i] = True
                        sources.add(source)
                    else:
                        start = time.monotonic()
                        y, values = estimators[s](params, qm, 1., T=key)
                        g = np.array(values[:2], float)
                        y = float(y)
                        diag = estimators[s].last_diagnostics
                        norms = np.array([diag['norm2_free'], diag['norm2_nudge']])
                        result[f'seconds_{s}'][i] = time.monotonic() - start
                        cache[s][key] = (g, y, norms, str(path))
                    result[f'g_{s}_AE'][i] = g
                    result[f'y_{s}_AE'][i] = y
                    result[f'norm2_{s}'][i] = norms
                    if (i + 1) % 10 == 0 or i + 1 == len(TL):
                        checkpoint()
                print(f'N={args.N} {tag} {s} level={level} points={len(TL)}', flush=True)
            stats = {s: amplitude_statistics(TL, result[f'g_{s}_AE']) for s in SCHEDULES}
            result['statistics'] = json.dumps(stats)
            # Early-T windows without three resolved cycles are retained as
            # gradient data, but refining them cannot establish a local amplitude.
            needs_refinement = any(st['resolved_cycles'] and st['half_grid_relative_change'] > .05
                                   for st in stats.values())
            if center is None or not needs_refinement or level == 1:
                break
        checks = []
        for s in SCHEDULES:
            _, g = tight[s](params, qm, 1., T=float(TL[-1]))
            difference = float(np.linalg.norm(np.array(g[:2], float) - result[f'g_{s}_AE'][-1]))
            amplitude = float(np.linalg.norm(stats[s]['rms']))
            stats[s]['sampling_resolved'] = stats[s]['half_grid_relative_change'] <= .05
            stats[s]['solver_spotcheck_resolved'] = difference <= .01 * max(amplitude, 1e-12)
            stats[s]['usable_amplitude'] = bool(stats[s]['resolved_cycles'] and stats[s]['sampling_resolved']
                                                and stats[s]['solver_spotcheck_resolved'])
            checks.append(dict(schedule=s, T=float(TL[-1]), absolute_gradient_difference=difference,
                               relative_to_ED=difference / np.linalg.norm(ged)))
        result['statistics'] = json.dumps(stats)
        result['convergence_checks'] = json.dumps(checks)
        result['complete'] = True
        checkpoint()
        print('COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
