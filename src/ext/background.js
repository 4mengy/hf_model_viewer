/* 扩展形态入口：Manifest V3 Background Service Worker
 * 作为唯一网络出口，代理对 Hugging Face CDN 的 fetch，突破网页端 CORS 限制。 */
import { installBackgroundNetHandler } from '../platform/net.js';

installBackgroundNetHandler();
