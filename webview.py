"""
webview - MJPEG server + browser zone editor for catcam.

Usage from catcam.py:

    from webview import WebView
    web = WebView(port=8080, zones=ZONES)
    ...
    web.publish(view)            # each loop, after draw_overlay

Then open http://<mac-mini-ip>:8080 from any machine on the LAN.
Click to add points, name and save a zone, and the ZONES dict updates live.
"""

import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

BOUNDARY = "catcamframe"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>catcam</title>
<style>
 body{background:#15140f;color:#d8d6cc;font:14px system-ui,sans-serif;margin:0;padding:16px}
 #wrap{position:relative;display:inline-block;line-height:0}
 #vid{display:block;max-width:100%;height:auto}
 #cv{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
 #bar{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 button,input{font:14px system-ui;padding:6px 10px;border-radius:6px;
   border:1px solid #45443d;background:#22211b;color:#d8d6cc}
 button:hover{background:#2e2d25}
 #zones{margin-top:12px;white-space:pre;font:12px ui-monospace,monospace;color:#9c9a92}
 #hud{margin-top:14px;display:flex;gap:18px;align-items:center;font:13px ui-monospace,monospace;color:#9c9a92}
 #hud b{color:#d8d6cc;font-weight:500}
 .lamp{width:9px;height:9px;border-radius:50%;background:#45443d;display:inline-block;margin-right:6px}
 .lamp.on{background:#5dcaa5}
 table{border-collapse:collapse;margin-top:10px;font:13px ui-monospace,monospace;min-width:520px}
 th{text-align:left;color:#73726c;font-weight:400;padding:4px 10px 4px 0;border-bottom:1px solid #33322b}
 td{padding:5px 10px 5px 0;border-bottom:1px solid #26251f}
 .bar{display:inline-block;height:8px;background:#5dcaa5;border-radius:2px;vertical-align:middle}
 .bar.lo{background:#8a8880}
 .tag{padding:1px 6px;border-radius:4px;font-size:12px}
 .tag.hit{background:#0f6e56;color:#c9f2e3}
 .tag.miss{background:#3a3931;color:#9c9a92}
 .empty{color:#73726c;padding:8px 0}
</style></head><body>
<div id="wrap">
  <img id="vid" src="/stream">
  <canvas id="cv"></canvas>
</div>
<div id="bar">
  <input id="name" placeholder="zone name">
  <button onclick="save()">Save zone</button>
  <button onclick="undo()">Undo point</button>
  <button onclick="clr()">Clear</button>
  <button onclick="delLast()">Delete last zone</button>
  <button onclick="dump()">Print block</button>
</div>
<div id="bar" style="margin-top:8px">
  <span style="color:#73726c">capture for training:</span>
  <button onclick="cap('miss')">Missed detection</button>
  <button onclick="cap('good')">Good detection</button>
  <button onclick="cap('empty')">Empty scene</button>
  <span id="capmsg" style="color:#5dcaa5"></span>
</div>
<div id="hud">
  <span><span id="lamp" class="lamp"></span><b id="trig">idle</b></span>
  <span>infer <b id="ms">-</b> ms</span>
  <span>streak <b id="streak">-</b></span>
  <span>thresh <b id="th">-</b></span>
</div>
<div id="dets"></div>
<div id="zones"></div>
<script>
const img=document.getElementById('vid'),cv=document.getElementById('cv'),
      ctx=cv.getContext('2d');
let pts=[],nat=[640,360];

function fit(){cv.width=img.clientWidth;cv.height=img.clientHeight;
  if(img.naturalWidth)nat=[img.naturalWidth,img.naturalHeight];draw();}
img.onload=fit;addEventListener('resize',fit);

cv.onclick=e=>{const r=cv.getBoundingClientRect();
  const x=Math.round((e.clientX-r.left)/r.width*nat[0]);
  const y=Math.round((e.clientY-r.top)/r.height*nat[1]);
  pts.push([x,y]);draw();};

function draw(){ctx.clearRect(0,0,cv.width,cv.height);
  if(!pts.length)return;
  const sx=cv.width/nat[0],sy=cv.height/nat[1];
  ctx.strokeStyle='#ff4fd8';ctx.fillStyle='#ff4fd8';ctx.lineWidth=2;
  ctx.beginPath();pts.forEach((p,i)=>{const x=p[0]*sx,y=p[1]*sy;
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  if(pts.length>2)ctx.closePath();ctx.stroke();
  pts.forEach(p=>{ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,4,0,7);ctx.fill();});}

function undo(){pts.pop();draw();}
function clr(){pts=[];draw();}
async function save(){const n=document.getElementById('name').value.trim();
  if(pts.length<3){alert('Need at least 3 points');return;}
  if(!n){alert('Enter a zone name');return;}
  await fetch('/zone',{method:'POST',body:JSON.stringify({name:n,points:pts})});
  pts=[];document.getElementById('name').value='';draw();refresh();}
async function delLast(){await fetch('/zone/dellast',{method:'POST'});refresh();}
async function refresh(){const r=await fetch('/zones');const z=await r.json();
  document.getElementById('zones').textContent=
    'ZONES = {\\n'+Object.entries(z).map(([k,v])=>
      '    '+JSON.stringify(k)+': '+JSON.stringify(v).replace(/\\[(\\d+),(\\d+)\\]/g,'($1, $2)')+',')
      .join('\\n')+'\\n}';}
function dump(){refresh();}
async function cap(tag){
  const r=await(await fetch('/capture',{method:'POST',
    body:JSON.stringify({tag:tag})})).json();
  const m=document.getElementById('capmsg');
  m.textContent=r.ok?('saved '+r.name):'capture failed';
  setTimeout(()=>m.textContent='',2500);
}
refresh();setInterval(refresh,5000);

function bar(v){const w=Math.round(v*90);
  return '<span class="bar'+(v<0.3?' lo':'')+'" style="width:'+Math.max(w,2)+'px"></span> '+v.toFixed(2);}

async function status(){
  let s;try{s=await(await fetch('/status')).json()}catch(e){return}
  if(!s||s.infer_ms===undefined)return;
  document.getElementById('ms').textContent=s.infer_ms;
  document.getElementById('streak').textContent=s.streak+'/'+s.need;
  document.getElementById('th').textContent=(s.conf_thresh||0).toFixed(2);
  document.getElementById('trig').textContent=s.triggered?'TRIGGERED':'idle';
  document.getElementById('lamp').className='lamp'+(s.triggered?' on':'');
  const d=s.detections||[];
  const el=document.getElementById('dets');
  if(!d.length){el.innerHTML='<div class="empty">no detections above threshold</div>';return;}
  el.innerHTML='<table><tr><th>class</th><th>class conf</th><th>animal score</th>'
    +'<th>zone</th><th>box</th></tr>'
    +d.map(r=>'<tr><td>'+r.cls+'</td><td>'+bar(r.conf)+'</td><td>'
      +(r.target?bar(r.animal):'<span style="color:#5f5e5a">-</span>')+'</td><td>'
      +(r.zone?'<span class="tag hit">'+r.zone+'</span>'
             :'<span class="tag miss">outside</span>')+'</td><td style="color:#73726c">'
      +r.box.join(', ')+'</td></tr>').join('')+'</table>';
}
setInterval(status,300);status();
</script></body></html>"""


class WebView:
    def __init__(self, port=8080, zones=None, quality=30, host="0.0.0.0",
                 on_capture=None, order_points=None):
        self.port = port
        self.zones = zones if zones is not None else {}
        self.quality = quality
        self.on_capture = on_capture
        self.order_points = order_points
        self._jpeg = None
        self._status = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0

        handler = self._make_handler()
        self.server = ThreadingHTTPServer((host, port), handler)
        self.server.daemon_threads = True

        def _quiet_error(request, client_address, _srv=self.server):
            exc = sys.exc_info()[1]
            # Broadened to OSError to catch all socket drops cleanly
            if isinstance(exc, (OSError, TimeoutError)):
                return
            traceback.print_exc()

        self.server.handle_error = _quiet_error
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[web] serving on http://0.0.0.0:{port}")

    def publish(self, frame):
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def set_status(self, **kw):
        with self._lock:
            self._status = kw

    def _wait_frame(self, last_seq, timeout=5.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def stop(self):
        self.server.shutdown()

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 5.0  # <--- CRITICAL FIX: Drops dead sockets automatically

            def log_message(self, *a):
                pass

            def handle_one_request(self):
                try:
                    super().handle_one_request()
                except (OSError, TimeoutError):
                    self.close_connection = True

            def handle(self):
                try:
                    super().handle()
                except (OSError, TimeoutError):
                    pass

            def _send(self, code, body, ctype="application/json"):
                if isinstance(body, str):
                    body = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self._send(200, PAGE, "text/html; charset=utf-8")
                elif path == "/zones":
                    self._send(200, json.dumps(
                        {k: [list(p) for p in v] for k, v in outer.zones.items()}))
                elif path == "/status":
                    with outer._lock:
                        st = dict(outer._status)
                    self._send(200, json.dumps(st))
                elif path == "/snapshot":
                    with outer._lock:
                        jpg = outer._jpeg
                    if jpg is None:
                        self._send(503, "{}")
                    else:
                        self._send(200, jpg, "image/jpeg")
                elif path == "/stream":
                    self._stream()
                else:
                    self._send(404, "{}")

            def _stream(self):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        jpg, seq = outer._wait_frame(seq)
                        if jpg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                except (OSError, TimeoutError):
                    # Fails cleanly if the TCP window is full or client disconnected
                    pass

            def do_POST(self):
                path = urlparse(self.path).path
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b"{}"
                if path == "/zone":
                    d = json.loads(raw)
                    name = str(d.get("name", "")).strip()
                    pts = [(int(x), int(y)) for x, y in d.get("points", [])]
                    if name and len(pts) >= 3:
                        if outer.order_points is not None:
                            pts = outer.order_points(pts)
                        outer.zones[name] = pts
                        print(f"[web] zone '{name}' set ({len(pts)} pts)")
                        self._send(200, '{"ok":true}')
                    else:
                        self._send(400, '{"ok":false}')
                elif path == "/capture":
                    if outer.on_capture is None:
                        self._send(503, '{"ok":false}')
                    else:
                        tag = json.loads(raw or b"{}").get("tag", "")
                        name = outer.on_capture(tag)
                        print(f"[web] captured {name}")
                        self._send(200, json.dumps({"ok": True, "name": name}))
                elif path == "/zone/dellast":
                    if outer.zones:
                        last = list(outer.zones)[-1]
                        del outer.zones[last]
                        print(f"[web] removed zone '{last}'")
                    self._send(200, '{"ok":true}')
                else:
                    self._send(404, "{}")

        return Handler