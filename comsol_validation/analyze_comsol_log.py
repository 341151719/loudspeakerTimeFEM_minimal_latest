#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter


STEP = re.compile(r"^\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(?:out\s+)?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)")


def main():
    ap=argparse.ArgumentParser();ap.add_argument('log',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    lines=a.log.read_text(encoding='utf-8',errors='ignore').splitlines()
    intervals=[];current=None;all_steps=[]
    for line in lines:
      q=re.search(r'时间间隔\s+(\d+)',line)
      if q:
        if current: intervals.append(current)
        current={'interval':int(q.group(1)),'last_step':None}
      m=STEP.match(line)
      if m:
        row={'step':int(m[1]),'time_s':float(m[2]),'step_size_s':float(m[3]),'res':int(m[4]),'jac':int(m[5]),'sol':int(m[6]),'order':int(m[7]),'Tfail':int(m[8]),'NLfail':int(m[9]),'linear_error':float(m[10]),'linear_residual':float(m[11])}
        all_steps.append(row)
        if current is not None:current['last_step']=row
    if current:intervals.append(current)
    dofs=[int(x) for line in lines for x in re.findall(r'自由度数：(\d+)',line)]
    elems=[int(m.group(1)) for line in lines if (m:=re.match(r'^\s*单元数：(\d+)',line))]
    quality=[float(x) for line in lines for x in re.findall(r'最小单元质量：([0-9.eE+-]+)',line)]
    memory=[]
    for line in lines:
      m=re.search(r'Memory:\s*(\d+)/(\d+)\s+(\d+)/(\d+)',line)
      if m:memory.append(tuple(map(int,m.groups())))
    errors=[x.strip() for x in lines if '错误' in x or 'Exception' in x or 'Error' in x]
    warnings=[x.strip() for x in lines if '警告' in x or 'Warning' in x]
    result={
      'log':str(a.log.resolve()),'completed':any('100 % - 完成' in x for x in lines),
      'time_intervals':len(intervals),'automatic_remesh_count':max(0,len(intervals)-1),
      'all_solver_dofs_min_max':([min(dofs),max(dofs)] if dofs else None),
      'transient_dofs_min_max':([min(x for x in dofs if x>50000),max(x for x in dofs if x>50000)] if any(x>50000 for x in dofs) else None),
      'triangle_elements_min_max':([min(elems),max(elems)] if elems else None),
      'mesh_min_quality_min_max':([min(quality),max(quality)] if quality else None),
      'accepted_steps_last':(all_steps[-1]['step'] if all_steps else None),
      'final_time_s':(all_steps[-1]['time_s'] if all_steps else None),
      'Tfail_sum_interval_final':sum(x['last_step']['Tfail'] for x in intervals if x['last_step']),
      'NLfail_sum_interval_final':sum(x['last_step']['NLfail'] for x in intervals if x['last_step']),
      'peak_memory_reported_MB':([max(x[1] for x in memory),max(x[3] for x in memory)] if memory else None),
      'interval_final_rows':[x['last_step'] for x in intervals if x['last_step']],
      'error_count':len(errors),
      'error_unique_counts':dict(Counter(errors)),
      'warning_count':len(warnings),
      'warning_unique_counts':dict(Counter(warnings)),
    }
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.out:a.out.write_text(text,encoding='utf-8')
    print(text);return 0
if __name__=='__main__':raise SystemExit(main())
