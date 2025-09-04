# 1) ボディをファイルに保存しつつHTTPコードを表示
curl -sS -H "Content-Type: application/json" -H "Accept: application/json" \
  -o /tmp/vllm_resp.json -w '\nHTTP=%{http_code}\n' \
  -d '{
    "model":"qwen3-8b",
    "messages":[{"role":"user","content":"テスト: 1行で自己紹介して。"}],
    "max_tokens":64,
    "temperature":0,
    "stream":false
  }' \
  http://192.168.1.70:8010/v1/chat/completions

# 2) JSONを安全に読み出す（空/非JSONなら中身をそのまま表示）
python - <<'PY'
import json,sys,os,io
p="/tmp/vllm_resp.json"
if not os.path.exists(p) or os.path.getsize(p)==0:
    print("[ERR] empty response"); sys.exit(1)
raw=open(p,"rb").read()
try:
    r=json.loads(raw)
    print(r["choices"][0]["message"]["content"])
except Exception as e:
    print("[NON-JSON RESPONSE]")
    print(raw[:1000].decode("utf-8","ignore"))
    raise
PY
