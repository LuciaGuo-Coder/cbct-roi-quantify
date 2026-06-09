"""
口腔 CBCT 图像 ROI 量化分析服务

基于 DCTA-DilUnet 牙齿分割结果，对 mask 区域进行膨胀、CT 值量化及可视化。
- POST /quantify：接收 Base64 mask + CT 图 + expand_pixels，返回平均 CT 值与红圈 ROI 图
- GET  /health：健康检查
"""

from __future__ import annotations

import base64
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="CBCT ROI Quantify Service",
    description="口腔 CBCT 图像 ROI 量化分析接口 — 基于 DCTA-DilUnet 分割模型",
    version="1.0.0",
)


# ── Error codes ──────────────────────────────────────────────────────────────

class ErrCode(IntEnum):
    INVALID_IMAGE = 4001
    EMPTY_MASK = 4002
    EXPAND_OUT_OF_BOUNDS = 4003
    DECODE_FAILED = 4004


_ERR_MSG = {
    ErrCode.INVALID_IMAGE: "图片数据无效，无法解码为图像",
    ErrCode.EMPTY_MASK: "掩膜全为背景，不包含任何前景区域",
    ErrCode.EXPAND_OUT_OF_BOUNDS: "膨胀后 ROI 超出图像边界或覆盖全图",
    ErrCode.DECODE_FAILED: "Base64 解码失败，请确认编码是否正确",
}


def _raise(code: ErrCode, detail: Optional[str] = None) -> None:
    raise HTTPException(
        status_code=422,
        detail={"code": int(code), "message": detail or _ERR_MSG[code]},
    )


# ── Request / Response models ────────────────────────────────────────────────

class QuantifyRequest(BaseModel):
    mask_b64: str = Field(..., description="Base64 编码的 mask 图（灰度/二值，前景像素值 > 0）")
    ct_b64: str = Field(..., description="Base64 编码的 CT 图（灰度）")
    expand_pixels: int = Field(default=0, ge=0, description="mask 膨胀像素数（椭圆核，≥0）")


class QuantifyResponse(BaseModel):
    mean_ct_value: float = Field(..., description="ROI 区域的平均 CT 值（灰度 0-255）")
    roi_image_b64: str = Field(..., description="标注红圈 ROI 后的 Base64 编码 PNG 图片")
    roi_pixel_count: int = Field(..., description="ROI 区域有效像素数")


# ── Image encode / decode helpers ────────────────────────────────────────────

def _b64_to_image(b64_str: str, label: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64_str)
    except Exception:
        _raise(ErrCode.DECODE_FAILED, f"{label} Base64 解码失败")

    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        _raise(ErrCode.INVALID_IMAGE, f"{label} 无法解码为有效图像")
    return img


def _image_to_b64(img: np.ndarray, fmt: str = ".png") -> str:
    ok, encoded = cv2.imencode(fmt, img)
    if not ok:
        _raise(ErrCode.INVALID_IMAGE, "结果图像编码失败")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ── Core quantify logic ─────────────────────────────────────────────────────

def quantify_roi(
    mask: np.ndarray,
    ct: np.ndarray,
    expand_pixels: int,
) -> tuple[float, np.ndarray, int]:
    """
    对 mask 进行膨胀，计算 ROI 区域在 CT 图上的平均值，
    并在 CT 图上绘制红色轮廓标注 ROI。

    Returns: (mean_ct_value, visualization_bgr, roi_pixel_count)
    """
    mask_gray = _ensure_gray(mask)
    ct_gray = _ensure_gray(ct)

    if mask_gray.shape != ct_gray.shape:
        _raise(
            ErrCode.INVALID_IMAGE,
            f"mask 尺寸 {mask_gray.shape} 与 CT 尺寸 {ct_gray.shape} 不一致",
        )

    # 二值化（与项目中 dataset.py 的标签处理一致，前景 > 0）
    binary = (mask_gray > 0).astype(np.uint8) * 255

    if cv2.countNonZero(binary) == 0:
        _raise(ErrCode.EMPTY_MASK)

    # 椭圆核膨胀
    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * expand_pixels + 1, 2 * expand_pixels + 1),
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    roi_count = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]

    if roi_count == 0:
        _raise(ErrCode.EMPTY_MASK, "膨胀后 mask 仍为空")
    if roi_count >= total_pixels:
        _raise(ErrCode.EXPAND_OUT_OF_BOUNDS)

    # 计算 ROI 平均 CT 值
    mean_val = float(cv2.mean(ct_gray, mask=binary)[0])

    # 提取轮廓并绘制红色标注
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    return mean_val, vis, roi_count


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/quantify", response_model=QuantifyResponse, summary="ROI 量化分析")
async def quantify(req: QuantifyRequest):
    """
    接收 Base64 编码的 mask 图与 CT 图，对 mask 进行膨胀，
    计算 ROI 区域平均 CT 值，并返回标注红圈 ROI 的可视化图片。

    mask 来源：DCTA-DilUnet 分割网络的输出（20 类标签，像素值 0-19），
    前景（像素值 > 0）即为牙齿 ROI。
    """
    mask_img = _b64_to_image(req.mask_b64, "mask")
    ct_img = _b64_to_image(req.ct_b64, "ct")

    mean_val, vis, count = quantify_roi(mask_img, ct_img, req.expand_pixels)

    return QuantifyResponse(
        mean_ct_value=round(mean_val, 4),
        roi_image_b64=_image_to_b64(vis),
        roi_pixel_count=count,
    )


@app.get("/health", summary="健康检查")
async def health():
    """返回服务运行状态。"""
    return {"status": "ok"}


# ── Custom UI ────────────────────────────────────────────────────────────────

_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CBCT ROI Quantify</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--primary:#4f46e5;--primary-light:#818cf8;--bg:#f0f2f5;--card:#fff;--text:#1e293b;--text-sec:#64748b;--border:#e2e8f0;--green:#10b981;--red:#ef4444;--radius:12px;--shadow:0 4px 24px rgba(0,0,0,.08)}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.header{background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#a855f7 100%);padding:48px 24px 56px;text-align:center;position:relative;overflow:hidden}
.header::after{content:'';position:absolute;bottom:-30px;left:0;right:0;height:60px;background:var(--bg);border-radius:50% 50% 0 0}
.header h1{color:#fff;font-size:28px;font-weight:700;letter-spacing:.5px}
.header p{color:rgba(255,255,255,.85);margin-top:8px;font-size:15px}
.badge{display:inline-block;background:rgba(255,255,255,.2);backdrop-filter:blur(8px);color:#fff;padding:4px 14px;border-radius:20px;font-size:12px;margin-top:12px;letter-spacing:.3px}
.container{max-width:960px;margin:-20px auto 40px;padding:0 20px;position:relative;z-index:1}
.card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:28px 32px;margin-bottom:24px;border:1px solid var(--border);transition:box-shadow .2s}
.card:hover{box-shadow:0 8px 32px rgba(0,0,0,.12)}
.card-title{font-size:17px;font-weight:600;margin-bottom:18px;display:flex;align-items:center;gap:10px}
.card-title .icon{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.icon-purple{background:rgba(79,70,229,.1);color:var(--primary)}
.icon-green{background:rgba(16,185,129,.1);color:var(--green)}
.icon-blue{background:rgba(59,130,246,.1);color:#3b82f6}
.row{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:640px){.row{grid-template-columns:1fr}}
.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:32px 20px;text-align:center;cursor:pointer;transition:all .25s;position:relative;background:#fafbfc}
.upload-zone:hover,.upload-zone.drag{border-color:var(--primary);background:rgba(79,70,229,.04)}
.upload-zone input{display:none}
.upload-zone .label{font-size:14px;color:var(--text-sec);margin-top:8px}
.upload-zone .name{font-size:13px;color:var(--primary);margin-top:6px;word-break:break-all;font-weight:500}
.upload-icon{font-size:28px;opacity:.5}
.preview-thumb{max-width:100%;max-height:120px;margin-top:10px;border-radius:8px;border:1px solid var(--border)}
.form-group{margin-bottom:18px}
.form-group label{display:block;font-size:13px;font-weight:600;color:var(--text-sec);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.form-group input[type=number]{width:100%;padding:10px 14px;border:1.5px solid var(--border);border-radius:8px;font-size:15px;outline:none;transition:border .2s}
.form-group input[type=number]:focus{border-color:var(--primary)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 32px;border:none;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;transition:all .2s;width:100%}
.btn-primary{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;box-shadow:0 4px 14px rgba(79,70,229,.35)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(79,70,229,.45)}
.btn-primary:active{transform:translateY(0)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-sm{padding:8px 18px;font-size:13px;width:auto;border-radius:6px}
.btn-outline{background:transparent;border:1.5px solid var(--border);color:var(--text-sec)}
.btn-outline:hover{border-color:var(--primary);color:var(--primary)}
.result-panel{display:none}
.result-panel.show{display:block}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.stat-card{background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-radius:10px;padding:20px;text-align:center;border:1px solid var(--border)}
.stat-value{font-size:28px;font-weight:700;color:var(--primary);font-variant-numeric:tabular-nums}
.stat-label{font-size:12px;color:var(--text-sec);margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
.result-img{width:100%;border-radius:10px;border:1px solid var(--border);margin-top:8px}
.status{padding:10px 16px;border-radius:8px;font-size:13px;margin-bottom:16px;display:none;align-items:center;gap:8px}
.status.ok{display:flex;background:rgba(16,185,129,.08);color:#047857;border:1px solid rgba(16,185,129,.2)}
.status.err{display:flex;background:rgba(239,68,68,.08);color:#b91c1c;border:1px solid rgba(239,68,68,.2)}
.health-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.health-dot.on{background:var(--green);box-shadow:0 0 8px rgba(16,185,129,.5)}
.health-dot.off{background:var(--red);box-shadow:0 0 8px rgba(239,68,68,.5)}
.spinner{width:18px;height:18px;border:2.5px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;display:none}
.btn-primary.loading .spinner{display:inline-block}
.btn-primary.loading .btn-text{display:none}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;padding:24px;color:var(--text-sec);font-size:13px}
</style>
</head>
<body>

<div class="header">
  <h1>CBCT ROI Quantify Service</h1>
  <p>口腔 CBCT 图像 ROI 量化分析平台 &mdash; 基于 DCTA-DilUnet 分割模型</p>
  <span class="badge">v1.0.0 &bull; FastAPI</span>
</div>

<div class="container">

  <!-- Health -->
  <div class="card" id="healthCard">
    <div class="card-title"><span class="icon icon-green">&#9889;</span> 服务状态
      <span style="margin-left:auto"><span class="health-dot off" id="healthDot"></span></span>
    </div>
    <button class="btn btn-sm btn-outline" onclick="checkHealth()">检测连接</button>
    <div class="status" id="healthMsg"></div>
  </div>

  <!-- Upload -->
  <div class="card">
    <div class="card-title"><span class="icon icon-purple">&#128200;</span> ROI 量化分析</div>
    <div class="row">
      <div>
        <div class="upload-zone" id="maskZone" onclick="document.getElementById('maskFile').click()">
          <input type="file" id="maskFile" accept="image/*">
          <div class="upload-icon">&#128206;</div>
          <div class="label">上传 Mask 分割图</div>
          <div class="name" id="maskName"></div>
          <img class="preview-thumb" id="maskPreview" style="display:none">
        </div>
      </div>
      <div>
        <div class="upload-zone" id="ctZone" onclick="document.getElementById('ctFile').click()">
          <input type="file" id="ctFile" accept="image/*">
          <div class="upload-icon">&#128444;</div>
          <div class="label">上传 CT 原始切片</div>
          <div class="name" id="ctName"></div>
          <img class="preview-thumb" id="ctPreview" style="display:none">
        </div>
      </div>
    </div>
    <div class="form-group" style="margin-top:20px">
      <label>膨胀像素 (expand_pixels)</label>
      <input type="number" id="expandInput" value="5" min="0" max="100">
    </div>
    <button class="btn btn-primary" id="runBtn" onclick="runQuantify()" disabled>
      <span class="spinner"></span>
      <span class="btn-text">&#9654; 开始量化分析</span>
    </button>
    <div class="status" id="apiMsg" style="margin-top:14px"></div>
  </div>

  <!-- Result -->
  <div class="card result-panel" id="resultPanel">
    <div class="card-title"><span class="icon icon-blue">&#128202;</span> 分析结果</div>
    <div class="stats">
      <div class="stat-card">
        <div class="stat-value" id="meanVal">-</div>
        <div class="stat-label">平均 CT 值</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="roiCount">-</div>
        <div class="stat-label">ROI 像素数</div>
      </div>
    </div>
    <div style="font-size:13px;color:var(--text-sec);margin-bottom:6px;font-weight:600">ROI 红圈标注图</div>
    <img class="result-img" id="roiImg">
  </div>


</div>

<div class="footer">CBCT ROI Quantify Service &copy; 2025</div>

<script>
let maskB64='', ctB64='';

function fileToB64(file, cb){
  const r=new FileReader();
  r.onload=()=>cb(r.result.split(',')[1]);
  r.readAsDataURL(file);
}

document.getElementById('maskFile').addEventListener('change',function(e){
  const f=e.target.files[0]; if(!f)return;
  document.getElementById('maskName').textContent=f.name;
  fileToB64(f,b=>{maskB64=b});
  const img=document.getElementById('maskPreview');
  img.src=URL.createObjectURL(f); img.style.display='block';
  checkReady();
});
document.getElementById('ctFile').addEventListener('change',function(e){
  const f=e.target.files[0]; if(!f)return;
  document.getElementById('ctName').textContent=f.name;
  fileToB64(f,b=>{ctB64=b});
  const img=document.getElementById('ctPreview');
  img.src=URL.createObjectURL(f); img.style.display='block';
  checkReady();
});

['maskZone','ctZone'].forEach(id=>{
  const z=document.getElementById(id);
  z.addEventListener('dragover',e=>{e.preventDefault();z.classList.add('drag')});
  z.addEventListener('dragleave',()=>z.classList.remove('drag'));
  z.addEventListener('drop',e=>{
    e.preventDefault();z.classList.remove('drag');
    const input=z.querySelector('input');
    input.files=e.dataTransfer.files;
    input.dispatchEvent(new Event('change'));
  });
});

function checkReady(){document.getElementById('runBtn').disabled=!(maskB64&&ctB64)}

function showMsg(id,ok,msg){
  const el=document.getElementById(id);
  el.className='status '+(ok?'ok':'err');
  el.innerHTML=(ok?'&#10003; ':'&#10007; ')+msg;
}

async function checkHealth(){
  try{
    const r=await fetch('/health');
    const d=await r.json();
    document.getElementById('healthDot').className='health-dot on';
    showMsg('healthMsg',true,'服务运行正常');
  }catch(e){
    document.getElementById('healthDot').className='health-dot off';
    showMsg('healthMsg',false,'无法连接服务');
  }
}

async function runQuantify(){
  const btn=document.getElementById('runBtn');
  btn.classList.add('loading'); btn.disabled=true;
  document.getElementById('resultPanel').classList.remove('show');
  document.getElementById('apiMsg').className='status';

  try{
    const r=await fetch('/quantify',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        mask_b64:maskB64,
        ct_b64:ctB64,
        expand_pixels:parseInt(document.getElementById('expandInput').value)||0
      })
    });
    const d=await r.json();
    if(!r.ok){
      const msg=d.error?(d.error.code+' - '+d.error.message):JSON.stringify(d);
      showMsg('apiMsg',false,msg);
      return;
    }
    document.getElementById('meanVal').textContent=d.mean_ct_value.toFixed(2);
    document.getElementById('roiCount').textContent=d.roi_pixel_count.toLocaleString();
    document.getElementById('roiImg').src='data:image/png;base64,'+d.roi_image_b64;
    document.getElementById('resultPanel').classList.add('show');
    showMsg('apiMsg',true,'分析完成');
  }catch(e){
    showMsg('apiMsg',false,'请求失败：'+e.message);
  }finally{
    btn.classList.remove('loading'); checkReady();
  }
}

checkHealth();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return _INDEX_HTML


# ── Global exception handler ────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )
