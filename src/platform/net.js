/* ============================================================
 * platform/net.js — 网络抽象层（双形态核心）
 * ------------------------------------------------------------
 * 设计目标：业务核心（engine / vram / tree）不依赖具体运行环境。
 *  - Web 形态：直接调用浏览器全局 fetch。
 *  - 扩展形态（Manifest V3）：所有请求路由到 Background Service Worker，
 *    由其在 manifest 声明的 host_permissions 下发起，从而突破网页端 CORS。
 *
 * 暴露两个能力：
 *   net.text(url, headers)            -> 完整文本
 *   net.range(url, start, end, headers) -> 指定字节区间的 Uint8Array
 * ============================================================ */

const HF_UA = { 'User-Agent': 'hf-vram-estimator/1.0' };

function isExtensionContext() {
  return (
    typeof chrome !== 'undefined' &&
    chrome.runtime &&
    chrome.runtime.id &&
    typeof chrome.runtime.sendMessage === 'function'
  );
}

/**
 * 创建网络实例。扩展态下通过 chrome.runtime 与 background 通信；
 * 返回结构统一为 { ok, status, body: number[] }（body 为字节数组）。
 */
function rawFetch(url, { headers = {}, method = 'GET', range } = {}) {
  const h = { ...HF_UA, ...headers };
  if (range) h['Range'] = `bytes=${range[0]}-${range[1]}`;

  if (isExtensionContext()) {
    return chrome.runtime.sendMessage({ __hfNet: true, url, method, headers: h });
  }

  return fetch(url, { method, headers: h }).then(async (res) => {
    const buf = await res.arrayBuffer();
    return { ok: res.ok, status: res.status, body: Array.from(new Uint8Array(buf)) };
  });
}

export function makeNet() {
  return {
    async text(url, headers = {}) {
      const r = await rawFetch(url, { headers });
      if (!r.ok) throw new Error(`HTTP ${r.status} 拉取失败: ${url}`);
      return new TextDecoder().decode(new Uint8Array(r.body));
    },

    async range(url, start, end, headers = {}) {
      const r = await rawFetch(url, { headers, range: [start, end] });
      if (!r.ok) throw new Error(`HTTP ${r.status} Range 请求失败: ${url} (${start}-${end})`);
      return new Uint8Array(r.body);
    },
  };
}

/* 扩展 Background Service Worker 入口（仅在 ext 构建中被引用） */
export function installBackgroundNetHandler() {
  if (typeof chrome === 'undefined' || !chrome.runtime) return;
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg && msg.__hfNet) {
      (async () => {
        try {
          const res = await fetch(msg.url, { method: msg.method || 'GET', headers: msg.headers || {} });
          const buf = await res.arrayBuffer();
          sendResponse({ ok: res.ok, status: res.status, body: Array.from(new Uint8Array(buf)) });
        } catch (e) {
          sendResponse({ ok: false, status: 0, error: String(e) });
        }
      })();
      return true; // 保持消息通道开放以支持异步响应
    }
    return false;
  });
}
