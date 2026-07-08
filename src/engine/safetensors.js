/* engine/safetensors.js — 零下载 Safetensors 头部解析
 * ------------------------------------------------------------
 * Safetensors 格式：
 *   [ 8 字节小端 uint64 = header 长度 N ][ N 字节 UTF-8 JSON ][ 权重数据 ]
 * JSON 结构：{ "<tensor_name>": { dtype, shape, data_offsets:[s,e] }, "__metadata__": {...} }
 *
 * 本模块仅通过 HTTP Range 读取前 8 字节 + 头部 JSON，不触碰权重数据。
 * ------------------------------------------------------------ */

const HEADER_LEN_BYTES = 8;

export async function readSafetensorsHeader(net, baseUrl, fileName, headers = {}) {
  const url = `${baseUrl}/${fileName}`;

  // 1) 读取前 8 字节，解析 header 长度
  const lenBuf = await net.range(url, 0, HEADER_LEN_BYTES - 1, headers);
  if (lenBuf.byteLength < HEADER_LEN_BYTES) {
    throw new Error(`文件 ${fileName} 过短，不是合法的 safetensors`);
  }
  const dv = new DataView(lenBuf.buffer, lenBuf.byteOffset, lenBuf.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true)); // little-endian

  // 2) 读取 header JSON 片段
  const headerBuf = await net.range(
    url,
    HEADER_LEN_BYTES,
    HEADER_LEN_BYTES + headerLen - 1,
    headers,
  );
  const text = new TextDecoder().decode(headerBuf);
  const json = JSON.parse(text);
  return json; // { tensorName: {dtype, shape, data_offsets}, __metadata__ }
}
