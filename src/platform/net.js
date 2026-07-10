const HF_UA = { 'User-Agent': 'hf-vram-estimator/1.0' };

async function rawFetch(url, { headers = {}, method = 'GET', range } = {}) {
  const h = { ...HF_UA, ...headers };
  if (range) h['Range'] = `bytes=${range[0]}-${range[1]}`;
  const res = await fetch(url, { method, headers: h });
  const buf = await res.arrayBuffer();
  return { ok: res.ok, status: res.status, body: Array.from(new Uint8Array(buf)) };
}

export function makeNet() {
  return {
    async text(url, headers = {}) {
      const r = await rawFetch(url, { headers });
      if (!r.ok) throw new Error(`HTTP ${r.status} fetch failed: ${url}`);
      return new TextDecoder().decode(new Uint8Array(r.body));
    },

    async range(url, start, end, headers = {}) {
      const r = await rawFetch(url, { headers, range: [start, end] });
      if (!r.ok) throw new Error(`HTTP ${r.status} Range request failed: ${url} (${start}-${end})`);
      return new Uint8Array(r.body);
    },
  };
}
