"""Builds inputs/korea_labels.csv from the public database clinical table (the label ground truth by the user's decision of
2026-10-21: BioImage Archive S-BIAD2452 'Korean clinical Data.xlsx'; discrepancies with the source paper are disclosed
only). Columns: patient_id (korea:KR<n>), ISUP_grade, Fuhrman_grade. Run locally (needs openpyxl); the CSV is what the
training code reads, and common.load_cases() cross-checks its ISUP grades against the analysis-pool manifest."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
XLSX=HERE/'inputs/korea_clinical.xlsx'
OFFICIAL_SHA256='0c600a8798a44604d18225edcf7eb6706103081694ffb7375a8fea3fa673c530'  # download_provenance.json, 2026-10-20
URL='https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/452/S-BIAD2452/Files/Korean%20Cohort/Korean%20clinical%20Data.xlsx'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import openpyxl
    assert sha(XLSX)==OFFICIAL_SHA256,'korea_clinical.xlsx is not the downloaded database file'
    rows=list(openpyxl.load_workbook(XLSX,data_only=True).worksheets[0].iter_rows(values_only=True))
    h=rows[0];d=[dict(zip(h,r)) for r in rows[1:] if r and r[0]]
    out=[dict(patient_id='korea:'+str(r['KR']).strip(),ISUP_grade=int(r['ISUP grade']),Fuhrman_grade=int(r['fuhrman Nuclear grade'])) for r in d]
    assert len(out)==300 and len({r['patient_id'] for r in out})==300
    assert all(r['ISUP_grade'] in (1,2,3,4) and r['Fuhrman_grade'] in (1,2,3,4) for r in out)
    out.sort(key=lambda r:int(r['patient_id'].split('KR')[1]))
    with (HERE/'inputs/korea_labels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['patient_id','ISUP_grade','Fuhrman_grade']);w.writeheader();w.writerows(out)
    prov=dict(source=URL,xlsx_sha256=OFFICIAL_SHA256,csv_sha256=sha(HERE/'inputs/korea_labels.csv'),rows=len(out),
              decision='database labels are the ground truth (user, 2026-10-21); paper Table 1 discrepancies disclosed only')
    (HERE/'inputs/labels_provenance.json').write_text(json.dumps(prov,indent=1)+'\n')
    from collections import Counter
    print('ISUP',dict(sorted(Counter(r['ISUP_grade'] for r in out).items())),'Fuhrman',dict(sorted(Counter(r['Fuhrman_grade'] for r in out).items())))
if __name__=='__main__':main()
