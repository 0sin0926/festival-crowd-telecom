"""여의동 집계구 경계를 WGS84로 변환하고, 불꽃축제일/일반 토요일 19시 생활인구를 붙여
GeoJSON과 Leaflet 지도(HTML)를 만든다. 표준 라이브러리만 사용."""
import base64, csv, io, json, math, struct, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHP_ZIP = ROOT / "data/raw/boundary/jipgyegu_shp_2016.zip"
POP_ZIP = ROOT / "data/raw/living_pop/LOCAL_PEOPLE_202410.zip"
OUT_GEOJSON = ROOT / "data/processed/yeouido_jipgyegu.geojson"
OUT_HTML = ROOT / "여의도_집계구_지도.html"
PREFIX = "11190540"          # 집계구코드 기준 여의동
ADM_CD = "11560540"          # 생활인구 파일의 여의동 행정동코드
FEST, NORMAL, HOUR = "20241005", "20241012", "19"

# --- UTM-K(EPSG:5179) -> WGS84, Krüger 급수 ---
A_, F_ = 6378137.0, 1 / 298.257222101
LAT0, LON0, K0, FE, FN = math.radians(38), math.radians(127.5), 0.9996, 1_000_000.0, 2_000_000.0
N_ = F_ / (2 - F_)
E_ = math.sqrt(F_ * (2 - F_))
AA = A_ / (1 + N_) * (1 + N_**2 / 4 + N_**4 / 64)
ALPHA = (N_/2 - 2*N_**2/3 + 5*N_**3/16, 13*N_**2/48 - 3*N_**3/5, 61*N_**3/240)
BETA = (N_/2 - 2*N_**2/3 + 37*N_**3/96, N_**2/48 + N_**3/15, 17*N_**3/480)
DELTA = (2*N_ - 2*N_**2/3 - 2*N_**3, 7*N_**2/3 - 8*N_**3/5, 56*N_**3/15)
_chi0 = math.atan(math.sinh(math.asinh(math.tan(LAT0)) - E_ * math.atanh(E_ * math.sin(LAT0))))
XI0 = _chi0 + sum(a * math.sin(2 * (j + 1) * _chi0) for j, a in enumerate(ALPHA))

def to_wgs84(x, y):
    xi, eta = XI0 + (y - FN) / (K0 * AA), (x - FE) / (K0 * AA)
    xi_p = xi - sum(b * math.sin(2*(j+1)*xi) * math.cosh(2*(j+1)*eta) for j, b in enumerate(BETA))
    eta_p = eta - sum(b * math.cos(2*(j+1)*xi) * math.sinh(2*(j+1)*eta) for j, b in enumerate(BETA))
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    lat = chi + sum(d * math.sin(2*(j+1)*chi) for j, d in enumerate(DELTA))
    lon = LON0 + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return round(math.degrees(lon), 6), round(math.degrees(lat), 6)

def read_shapes():
    z = zipfile.ZipFile(SHP_ZIP)
    shp = max((i for i in z.infolist() if i.filename.lower().endswith(".shp")), key=lambda i: i.file_size)
    shpb, dbf = z.read(shp), z.read(shp.filename[:-4] + ".dbf")
    nrec, hlen, rlen = struct.unpack("<IHH", dbf[4:12])
    codes = [dbf[hlen + r*rlen + 1: hlen + r*rlen + 21].decode("ascii", "replace").strip() for r in range(nrec)]
    pos, out, idx = 100, [], 0
    while pos < len(shpb):
        _, clen = struct.unpack(">ii", shpb[pos:pos+8]); c = shpb[pos+8:pos+8+clen*2]; pos += 8 + clen*2
        code = codes[idx]; idx += 1
        if struct.unpack("<i", c[:4])[0] == 0 or not code.startswith(PREFIX):
            continue
        nparts, npts = struct.unpack("<ii", c[36:44])
        parts = list(struct.unpack("<%di" % nparts, c[44:44+4*nparts])) + [npts]
        pts = struct.unpack("<%dd" % (2*npts), c[44+4*nparts:44+4*nparts+16*npts])
        rings, area = [], 0.0
        for k in range(nparts):
            ring = [(pts[2*j], pts[2*j+1]) for j in range(parts[k], parts[k+1])]
            area += sum(ring[j][0]*ring[j+1][1] - ring[j+1][0]*ring[j][1] for j in range(len(ring)-1)) / 2
            rings.append([to_wgs84(*p) for p in ring])
        out.append((code, abs(area), rings))
    return out

def read_pop():
    z, pop = zipfile.ZipFile(POP_ZIP), {}
    for d in (FEST, NORMAL):
        with z.open(f"LOCAL_PEOPLE_{d}.csv") as f:
            r = csv.reader(io.TextIOWrapper(f, encoding="cp949", errors="replace")); next(r)
            for row in r:
                if row[2] == ADM_CD and row[1] == HOUR:
                    pop[(row[3], d)] = float(row[4])
    return pop

LEAFLET = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/"
TILE_Z, BBOX = 16, (126.900, 37.512, 126.956, 37.544)   # 배경 지도를 HTML에 내장할 범위 (lon0, lat0, lon1, lat1)
TILE_CACHE = ROOT / "data/raw/osm_tiles"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "teamChiangmai-student-project/1.0", "Referer": "https://www.openstreetmap.org/"})
    return urllib.request.urlopen(req, timeout=30).read()

def fetch_tiles():
    """파일을 직접 열면(file://) 타일 서버가 막히므로, 필요한 범위의 OSM 타일을 한 번 받아 HTML에 내장한다."""
    n = 2 ** TILE_Z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for x in range(tx(BBOX[0]), tx(BBOX[2]) + 1):
        for y in range(ty(BBOX[3]), ty(BBOX[1]) + 1):
            f = TILE_CACHE / f"{TILE_Z}_{x}_{y}.png"
            if not f.exists():
                f.write_bytes(fetch(f"https://tile.openstreetmap.org/{TILE_Z}/{x}/{y}.png")); time.sleep(0.15)
            tiles[f"{x}/{y}"] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
    return tiles

def main():
    pop, feats = read_pop(), []
    for code, area, rings in read_shapes():
        f, n = pop.get((code, FEST), 0), pop.get((code, NORMAL), 0)
        feats.append({"type": "Feature",
                      "properties": {"code": code, "short": code[-6:], "area_km2": round(area/1e6, 3),
                                     "fest": round(f), "normal": round(n), "ratio": round(f/n, 1) if n else None},
                      "geometry": {"type": "Polygon", "coordinates": rings}})
    gj = {"type": "FeatureCollection", "features": feats}
    OUT_GEOJSON.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    html = (HTML.replace("__LCSS__", fetch(LEAFLET + "leaflet.min.css").decode()).replace("__LJS__", fetch(LEAFLET + "leaflet.min.js").decode())
            .replace("__TZ__", str(TILE_Z)).replace("__TILES__", json.dumps(fetch_tiles())).replace("__DATA__", json.dumps(gj, ensure_ascii=False)))
    OUT_HTML.write_text(html, encoding="utf-8")
    print(len(feats), "features ->", OUT_GEOJSON.name, OUT_HTML.name)

HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>여의동 집계구 지도</title>
<style>__LCSS__</style>
<script>__LJS__</script>
<style>html,body,#map{height:100%;margin:0}.lbl{background:#fff;border:1px solid #333;border-radius:3px;padding:1px 4px;font:bold 12px sans-serif;white-space:nowrap}
.info{background:#fff;padding:6px 10px;font:13px sans-serif;border-radius:4px;box-shadow:0 0 4px #0005}</style></head>
<body><div id="map"></div><script>
const data=__DATA__;const TZ=__TZ__;const TILES=__TILES__;
const map=L.map('map',{minZoom:13,maxZoom:18}).setView([37.5265,126.928],14);
map.attributionControl.addAttribution('&copy; OpenStreetMap contributors');
const t2lon=(x,z)=>x/2**z*360-180, t2lat=(y,z)=>{const n=Math.PI-2*Math.PI*y/2**z;return 180/Math.PI*Math.atan(Math.sinh(n))};
for(const [k,src] of Object.entries(TILES)){const [x,y]=k.split('/').map(Number);
 L.imageOverlay(src,[[t2lat(y+1,TZ),t2lon(x,TZ)],[t2lat(y,TZ),t2lon(x+1,TZ)]]).addTo(map);}
const col=v=>v>80000?'#7f0000':v>40000?'#d7301f':v>5000?'#fc8d59':v>2000?'#fdcc8a':'#fef0d9';
L.geoJSON(data,{style:f=>({color:'#222',weight:f.properties.area_km2>0.1?2.5:1,fillColor:col(f.properties.fest),fillOpacity:.45}),
 onEachFeature:(f,l)=>{const p=f.properties;
  l.bindPopup(`<b>${p.code}</b><br>면적 ${p.area_km2}km²<br>축제일 19시 ${p.fest.toLocaleString()}명<br>일반 토요일 19시 ${p.normal.toLocaleString()}명<br>배율 ${p.ratio}`);
  if(p.area_km2>0.1)l.bindTooltip(`${p.short}<br>${p.fest.toLocaleString()}`,{permanent:true,direction:'center',className:'lbl'});}}).addTo(map);
const info=L.control({position:'topright'});info.onAdd=()=>{const d=L.DomUtil.create('div','info');
 d.innerHTML='여의동 집계구 (2016 경계)<br>색·숫자: 2024-10-05 19시 생활인구<br>라벨: 집계구코드 끝 6자리 (면적 0.1km² 이상)';return d};info.addTo(map);
</script></body></html>"""

if __name__ == "__main__":
    main()
