"""서울 생활인구(집계구 단위, OA-14979) 월별 ZIP을 받아 여의동 행만 추출해 저장한다.
원본 ZIP은 개당 0.5~1.2GB라 추출이 끝나면 지운다(이미 있던 ZIP은 그대로 둔다). 표준 라이브러리만 사용.

사용법:
    python3 scripts/fetch_yeouido_pop.py 202301 202607     # 시작월~종료월
    python3 scripts/fetch_yeouido_pop.py 202410            # 한 달
"""
import csv, gzip, io, subprocess, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/living_pop"
OUT_DIR = ROOT / "data/processed/yeouido_pop"
ADM_CD = "11560540"          # 생활인구 파일의 여의동 행정동코드
PAGE = "https://data.seoul.go.kr/dataList/OA-14979/F/1/datasetView.do"
DOWNLOAD = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"

def months(start, end):
    y, m = int(start[:4]), int(start[4:])
    while f"{y}{m:02d}" <= end:
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

def download(ym, dest):
    tmp = dest.with_suffix(".part")
    cmd = ["curl", "-sL", "--fail", "--retry", "3", "-A", "Mozilla/5.0", "-e", PAGE, "-X", "POST", DOWNLOAD,
           "-d", f"infId=OA-14979&seqNo=&seq={ym[2:]}&infSeq=1", "-o", str(tmp)]
    if subprocess.run(cmd).returncode != 0 or not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    return True

def extract(zip_path, out_path):
    """ZIP 안의 일별 CSV에서 여의동 행만 골라 하나의 csv.gz로 저장. (행 수, 일 수, 집계구 수) 반환."""
    needle, rows, days, codes, header = ADM_CD.encode(), 0, 0, set(), None
    tmp = out_path.with_suffix(".part")
    with zipfile.ZipFile(zip_path) as z, gzip.open(tmp, "wt", encoding="utf-8", newline="") as g:
        w = csv.writer(g)
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".csv")):
            days += 1
            with z.open(name) as f:
                raw = f.readline()
                try:                                  # 2019년 이전 파일은 UTF-8(BOM), 이후는 CP949
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw.decode("cp949")
                head = next(csv.reader([text.lstrip("\ufeff?")]))
                if header is None:
                    header = head
                    w.writerow(header)
                for line in f:
                    if needle not in line:
                        continue
                    rec = next(csv.reader([line.decode("cp949", "replace")]))
                    if rec[2] != ADM_CD:
                        continue
                    w.writerow(rec[:len(header)])
                    codes.add(rec[3]); rows += 1
    tmp.rename(out_path)
    return rows, days, len(codes)

def main():
    start = sys.argv[1]
    end = sys.argv[2] if len(sys.argv) > 2 else start
    RAW_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ym in months(start, end):
        out = OUT_DIR / f"yeouido_{ym}.csv.gz"
        if out.exists():
            print(f"{ym} 건너뜀 (이미 있음)", flush=True)
            continue
        zp = RAW_DIR / f"LOCAL_PEOPLE_{ym}.zip"
        had_zip, t0 = zp.exists(), time.time()
        if not had_zip and not download(ym, zp):
            print(f"{ym} 다운로드 실패", flush=True)
            continue
        size = zp.stat().st_size / 1e6
        rows, days, ncodes = extract(zp, out)
        if not had_zip:
            zp.unlink()
        print(f"{ym} 완료: 원본 {size:.0f}MB, {days}일, 집계구 {ncodes}개, {rows}행, {time.time()-t0:.0f}초", flush=True)

if __name__ == "__main__":
    main()
