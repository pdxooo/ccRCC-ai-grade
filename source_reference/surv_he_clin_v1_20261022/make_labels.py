"""Builds the label/covariate inputs from the public database tables (ground truth by the user's decision of 2026-10-21;
discrepancies with the source paper are disclosed only). Run locally (needs openpyxl); every source is sha256-checked
against the download record of 2026-10-20.
  inputs/korea_surv.csv     patient_id (korea:KR<n>), recurrence, dfs_months, leibovich_fuhrman, leibovich_isup, isup, fuhrman,
                            t_stage, size_cm, necrosis
  inputs/scotland_surv.csv  the same for patient_id scotland:<n> (iCAIRD ID ICAIRD_<n>)
  inputs/korea_if.csv / scotland_if.csv   patient_id and the per-case multiplex-IF summaries (exploratory analysis only)
The tables' 'Response' column is the recurrence status and 'Disease free month' the disease-free survival time
(BioImage Archive study descriptions S-BIAD2452 / S-BIAD2454)."""
import csv,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;IN=HERE/'inputs'
SHA={'korea_clinical.xlsx':'0c600a8798a44604d18225edcf7eb6706103081694ffb7375a8fea3fa673c530',
     'scotland_clinical.xlsx':'f7ae8885cfe26b38f4fd52e7f47b7a53d0b2f49c94f4e993f13827637b1338fd',
     'korea_analysis.xlsx':'565159d2550fb75ac80ec7deffb7b1a525bf5bbf87c9b1cae32ca215ad50707c',
     'scotland_analysis.xlsx':'f485d9331777e38a98e5060558876636119da8449b13be806093217c2838236a'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(name):
    import openpyxl
    assert sha(IN/name)==SHA[name],f'{name} is not the downloaded database file'
    rows=list(openpyxl.load_workbook(IN/name,data_only=True).worksheets[0].iter_rows(values_only=True))
    return rows[0],[r for r in rows[1:] if r and r[0] is not None]
def write(name,rows,cols):
    with (IN/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
SURV=['patient_id','recurrence','dfs_months','leibovich_fuhrman','leibovich_isup','isup','fuhrman','t_stage','size_cm','necrosis']
def surv(name,pid):
    h,rows=table(name);out=[]
    for r in rows:
        d=dict(zip(h,r))
        out.append(dict(patient_id=pid(d[h[0]]),recurrence=int(d['Response']),dfs_months=float(d['Disease free month']),
                        leibovich_fuhrman=int(d['L score_Fuhrman']),leibovich_isup=int(d['Lscore_ISUP']),isup=int(d['ISUP grade']),
                        fuhrman=int(d['fuhrman Nuclear grade']),t_stage=str(d['Tumour Status']).strip(),size_cm=float(d['tumour size']),
                        necrosis=int(d['Necrosis'])))
    assert len({r['patient_id'] for r in out})==len(out) and all(r['recurrence'] in (0,1) and r['dfs_months']>=0 for r in out)
    return out
def ifrows(name,idcol,pid):
    h,rows=table(name);i=h.index(idcol);out=[]
    keep=[c for c in h if c and c!=idcol]
    for r in rows:
        d=dict(zip(h,r));o=dict(patient_id=pid(str(r[i])))
        for c in keep:o[re.sub(r'[^A-Za-z0-9]+','_',c).strip('_').lower()]=d[c]
        out.append(o)
    assert len({r['patient_id'] for r in out})==len(out);return out
def main():
    k=surv('korea_clinical.xlsx',lambda v:'korea:'+str(v).strip());s=surv('scotland_clinical.xlsx',lambda v:'scotland:'+str(v).strip().split('_')[-1])
    assert len(k)==300 and len(s)==120
    write('korea_surv.csv',sorted(k,key=lambda r:int(r['patient_id'].split('KR')[1])),SURV)
    write('scotland_surv.csv',sorted(s,key=lambda r:int(r['patient_id'].split(':')[1])),SURV)
    # Korean IF image tags look like 4IF_K26_HAZEM_... (K<n> = KR<n>, the file-name variant noted in the takeover documents)
    ki=ifrows('korea_analysis.xlsx','Image tag',lambda t:'korea:KR'+re.search(r'_KR?(\d+)_',t).group(1))
    si=ifrows('scotland_analysis.xlsx','iCAIRD ID',lambda t:'scotland:'+t.strip().split('_')[-1])
    kids={r['patient_id'] for r in k};sids={r['patient_id'] for r in s}
    assert all(r['patient_id'] in kids for r in ki) and all(r['patient_id'] in sids for r in si)
    write('korea_if.csv',ki,list(ki[0]));write('scotland_if.csv',si,list(si[0]))
    prov=dict(sources={n:dict(sha256=h) for n,h in SHA.items()},outputs={n:sha(IN/n) for n in ('korea_surv.csv','scotland_surv.csv','korea_if.csv','scotland_if.csv')},
              decision='database tables are the ground truth (user, 2026-10-21); paper discrepancies disclosed only')
    (IN/'labels_provenance.json').write_text(json.dumps(prov,indent=1)+'\n')
    for n,rows in (('korea',k),('scotland',s)):
        print(n,len(rows),'cases',sum(r['recurrence'] for r in rows),'events',sum(1 for r in rows if r['dfs_months']==0),'with DFS 0')
    print('IF korea',len(ki),'columns',list(ki[0])[1:],'| scotland',len(si),'columns',list(si[0])[1:])
if __name__=='__main__':main()
