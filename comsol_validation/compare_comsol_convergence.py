#!/usr/bin/env python3
"""Compare two independently solved COMSOL cases on a common harmonic basis."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from compare_comsol_python import fit_harmonics, phase_difference_deg


def last_cycle(time, value, f0):
    lo, hi = 3/f0-1e-10, 4/f0-1e-10
    mask=(time>=lo)&(time<hi)
    return time[mask], value[mask]


def metrics(name,t0,y0,t1,y1,f0):
    a_t,a=last_cycle(t0,y0,f0); b_t,b=last_cycle(t1,y1,f0)
    _,ha,_=fit_harmonics(a_t,a,f0); _,hb,_=fit_harmonics(b_t,b,f0)
    thda=np.linalg.norm(ha[1:])/max(abs(ha[0]),1e-300)
    thdb=np.linalg.norm(hb[1:])/max(abs(hb[0]),1e-300)
    return {
      "signal":name,
      "baseline_H1_peak":float(abs(ha[0])),"variant_H1_peak":float(abs(hb[0])),
      "H1_relative_change":float(abs(abs(hb[0])-abs(ha[0]))/max(abs(ha[0]),1e-300)),
      "H1_phase_change_deg":phase_difference_deg(hb[0],ha[0]),
      "baseline_THD":float(thda),"variant_THD":float(thdb),
      "THD_absolute_percentage_point_change":float(100*abs(thdb-thda)),
      "THD_relative_change":float(abs(thdb-thda)/max(thda,1e-300)),
    }


def load_case(path):
    g=pd.read_csv(path/'global_timeseries.csv').drop_duplicates('time_s',keep='last').sort_values('time_s')
    p=pd.read_csv(path/'pressure_points_timeseries.csv').pivot_table(index='time_s',columns='probe_name',values='p_Pa',aggfunc='last',dropna=False).sort_index()
    return g,p


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--variant',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--f0',type=float,default=70.0)
    ap.add_argument('--amplitude-tolerance',type=float,default=.01)
    ap.add_argument('--phase-tolerance-deg',type=float,default=1.0)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); bg,bp=load_case(a.baseline); vg,vp=load_case(a.variant)
    rows=[]
    for name in ['coil_current_A','coil_displacement_m','dynamic_BL_N_A']:
      if name in bg and name in vg and bg[name].notna().all() and vg[name].notna().all(): rows.append(metrics(name,bg.time_s.to_numpy(),bg[name].to_numpy(),vg.time_s.to_numpy(),vg[name].to_numpy(),a.f0))
    unavailable={}
    diagnostic_only = {'python_axis_rear_actual'}
    for name in [
      'python_axis_near_actual','python_axis_boundary_actual',
      'python_axis_rear_actual','python_offaxis_actual',
      'common_rear_physical_m0p10'
    ]:
      if name not in bp or name not in vp:
        unavailable[name]="probe column absent"
      elif not (bp[name].notna().all() and vp[name].notna().all()):
        unavailable[name]="outside one or both COMSOL acoustic domains"
      else:
        rows.append(metrics('pressure_'+name,bp.index.to_numpy(),bp[name].to_numpy(),vp.index.to_numpy(),vp[name].to_numpy(),a.f0))
    frame=pd.DataFrame(rows); frame.to_csv(a.out/'convergence_metrics.csv',index=False,float_format='%.12e')
    primary=[
      r for r in rows
      if r['signal']!='dynamic_BL_N_A'
      and r['signal'] not in {'pressure_'+name for name in diagnostic_only}
    ]
    acceptance={r['signal']:bool(r['H1_relative_change']<=a.amplitude_tolerance and abs(r['H1_phase_change_deg'])<=a.phase_tolerance_deg) for r in primary}
    summary={"baseline":str(a.baseline),"variant":str(a.variant),
             "criteria":f"H1 amplitude <={100*a.amplitude_tolerance:g}% and phase <={a.phase_tolerance_deg:g} degree",
             "amplitude_tolerance":a.amplitude_tolerance,
             "phase_tolerance_deg":a.phase_tolerance_deg,
             "unavailable_probes":unavailable,"acceptance":acceptance,
             "diagnostic_only_signals":['pressure_'+name for name in sorted(diagnostic_only)],
             "all_primary_converged":all(acceptance.values()),"metrics":rows}
    (a.out/'convergence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
