/* 扩展形态入口：浏览器扩展 Popup
 * 复用与 Web 完全相同的业务核心与 UI 编排；网络请求经 background service worker 代理。 */
import { mountApp } from '../ui/app.js';
import '../styles.css';

const root = document.getElementById('app');
mountApp(root);
