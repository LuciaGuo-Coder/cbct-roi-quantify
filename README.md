# CBCT ROI Quantify Service

口腔 CBCT 图像 ROI 量化分析微服务。

基于 **DCTA-DilUnet** 牙齿分割模型，对分割 mask 进行膨胀、计算 ROI 区域平均 CT 值，并生成红圈标注的可视化图片。

## 项目结构

```
├── main.py             # FastAPI 服务入口（含 Web 交互页面）
├── ring_roi_core.py    # 任务一算法占位文件，后续替换为正式实现
├── requirements.txt    # Python 依赖
├── Dockerfile          # 容器化部署
└── README.md           # 本文档
```

## 快速启动

### 本地运行

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 运行

```bash
docker build -t cbct-quantify .
docker run -p 8000:8000 cbct-quantify
```

启动后访问 http://localhost:8000 打开可视化操作页面（Swagger API 文档可通过 http://localhost:8000/docs 访问）。

---

## API 接口

### 1. `GET /health` — 健康检查

```bash
curl http://localhost:8000/health
```

**响应：**

```json
{"status": "ok"}
```

---

### 2. `POST /quantify` — ROI 量化分析

接收 Base64 编码的 mask 图和 CT 图，调用 `ring_roi_core.get_ring_roi()` 计算环形 ROI 内平均 CT 值，返回带红圈 ROI 标注的可视化图片。

#### 请求体 (JSON)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `mask_b64` | string | 是 | Base64 编码的 mask 图（DCTA-DilUnet 输出的分割结果，前景 > 0） |
| `ct_b64` | string | 是 | Base64 编码的 CT 图（灰度，308×308） |
| `expand_pixels` | int | 否 | mask 膨胀像素数，默认 15，范围 1-50 |

#### 响应体

| 字段 | 类型 | 说明 |
|------|------|------|
| `mean_ct_value` | float | ROI 区域的平均 CT 值（灰度 0-255） |
| `roi_image_b64` | string | 带红圈 ROI 标注的 PNG 图片（Base64 编码） |
| `roi_pixel_count` | int | ROI 区域有效像素数 |

#### Python 调用示例

```python
import base64
import requests

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# mask 来自 DCTA-DilUnet 分割输出，ct 为原始 CBCT 切片
resp = requests.post("http://localhost:8000/quantify", json={
    "mask_b64": encode_image("predict_mask.png"),
    "ct_b64": encode_image("ct_slice.png"),
    "expand_pixels": 5,
})

data = resp.json()
print(f"平均 CT 值: {data['mean_ct_value']}")
print(f"ROI 像素数: {data['roi_pixel_count']}")

# 保存红圈标注的可视化结果
with open("roi_result.png", "wb") as f:
    f.write(base64.b64decode(data["roi_image_b64"]))
```

#### cURL 调用示例

```bash
MASK_B64=$(base64 -i predict_mask.png | tr -d '\n')
CT_B64=$(base64 -i ct_slice.png | tr -d '\n')

curl -X POST http://localhost:8000/quantify \
  -H "Content-Type: application/json" \
  -d "{\"mask_b64\": \"$MASK_B64\", \"ct_b64\": \"$CT_B64\", \"expand_pixels\": 5}"
```

---

## 工作流程

```
原始 CBCT 切片 (灰度 PNG)
        │
        ├──► 分割模型 ──► mask (分割标签图)
        │                      │
        └──────────────────────┤
                               ▼
                      POST /quantify
                 (mask + CT + expand_pixels)
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ 1. Base64 解码与参数校验  │
                 │ 2. 调用 get_ring_roi      │
                 │ 3. 生成红圈 ROI 可视化图  │
                 │ 4. 返回平均 CT 值和图片   │
                 └──────────────────────────┘
                               │
                               ▼
                 { mean_ct_value, roi_image_b64 }
```
