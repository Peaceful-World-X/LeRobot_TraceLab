// 根据已应用类别映射的双臂轨迹确定屏幕左右，仅改变相机，不改变 XYZ。
function trajectoryInitialCamera(data) {
    const fallback = { eye: { x: 1.25, y: 1.25, z: .9 }, up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0 } };
    if (data.single_arm || !data.left.length || data.left.length !== data.right.length) return fallback;
    const delta = [0, 0, 0];
    let widest = [0, 0, 0], widestLength = 0;
    for (let i = 0; i < data.left.length; i++) {
        const difference = data.right[i].map((v, axis) => v - data.left[i][axis]);
        for (let axis = 0; axis < 3; axis++) delta[axis] += difference[axis] / data.left.length;
        const length = Math.hypot(difference[0], difference[1]);
        if (length > widestLength) { widestLength = length; widest = difference; }
    }
    // 轨迹中心几乎重合时，使用左右水平距离最明显的一帧消除朝向歧义。
    const epsilon = Math.max(1e-9, widestLength * 1e-6);
    let horizontal = Math.hypot(delta[0], delta[1]);
    if (horizontal <= epsilon && widestLength > epsilon) {
        delta.splice(0, 3, ...widest);
        horizontal = widestLength;
    }
    if (horizontal <= epsilon) return fallback; // 单臂、重合或仅高度不同，没有可靠水平左右方向。
    // 屏幕右方向为 up × eye，因此 eye=(dy,-dx,z) 将左→右方向投影到屏幕右侧。
    return {
        eye: { x: 1.8 * delta[1] / horizontal, y: -1.8 * delta[0] / horizontal, z: .9 },
        up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0 },
    };
}

// 服务器版与公开版共用的显示与交互；数据读取由各自入口提供。
window.createTrajectoryViewer = function () {
const $ = id => document.getElementById(id);
const video = $('video'), scrub = $('scrub');
const ARMS = { right: { name: '右臂', color: '#78a6d8', light: '#b8cff0', base: 0 }, left: { name: '左臂', color: '#c88768', light: '#e4ae92', base: 3 } }; // 本体左右语义。
let d = null, cache = {}, ready = false, generation = 0, controller = null;
let work = Promise.resolve(), pendingFrame = null, rendering = false, lastFrame = -1, clock = null, armView = 'both';
let rotating = false; // 鼠标拖动期间保留最新帧，避免重建 WebGL 数据打断相机交互。

let mediaRelease = null;
// 仅使用大于 0 的有限速度；首帧和全静止序列没有有效统计值。
function speedStats(values) {
    const a = values.filter(v => Number.isFinite(v) && v > 0).sort((x, y) => x - y);
    const mid = Math.floor(a.length / 2);
    return a.length ? { mean: a.reduce((s, v) => s + v, 0) / a.length, median: a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2 } : { mean: null, median: null };
}

// 统一消息出口，保留原页面的单行状态位置。
function message(text, error = false) {
    $('status').textContent = text;
    $('status').title = text;
    $('status').className = error ? 'error' : '';
}

// 只在图表和视频都准备好时开放控制。
function enable(value) {
    ready = value;
    video.controls = value;
    for (const el of [scrub, $('rateSelect'), ...document.querySelectorAll('[data-arm]')]) el.disabled = !value;
}

// Plotly 更新串行执行，换数据后旧任务自动失效。
function schedule(task, id = generation) {
    const result = work.then(() => id === generation ? task() : undefined);
    work = result.catch(error => { if (id === generation) { enable(false); video.pause(); message(error.message, true); } });
    return result;
}

// 等待视频 metadata，切换文件会取消旧等待。
function loadVideo(url, signal) {
    return new Promise((resolve, reject) => {
        const finish = error => {
            clearTimeout(timer);
            video.removeEventListener('loadedmetadata', loaded);
            video.removeEventListener('error', failed);
            signal.removeEventListener('abort', aborted);
            error ? reject(error) : resolve();
        };
        const loaded = () => finish();
        const failed = () => finish(new Error('视频无法加载或编码不受支持'));
        const aborted = () => finish(new DOMException('加载已取消', 'AbortError'));
        const timer = setTimeout(() => finish(new Error('视频加载超时')), 30000);
        video.addEventListener('loadedmetadata', loaded);
        video.addEventListener('error', failed);
        signal.addEventListener('abort', aborted, { once: true });
        if (signal.aborted) { aborted(); return; }
        video.src = url;
        video.load();
    });
}

// 两种数据源进入同一加载流程；取消旧加载并统一释放本地视频 URL。
async function load(provider) {
    const id = ++generation, rate = Number($('rateSelect').value);
    controller?.abort();
    const request = controller = new AbortController();
    enable(false); video.pause(); stopClock(); pendingFrame = null;
    video.removeAttribute('src'); video.load();
    if (mediaRelease) { mediaRelease(); mediaRelease = null; }
    message('加载中…');
    const timer = setTimeout(() => request.abort(), 60000);
    let candidate;
    try {
        candidate = await provider(request.signal);
        if (id !== generation || request.signal.aborted) throw new DOMException('加载已取消', 'AbortError');
        mediaRelease = candidate.release || null;
        await loadVideo(candidate.video_url, request.signal);
        if (id !== generation) return;
        if (!Number.isFinite(video.duration) || Math.abs(video.duration - candidate.duration) > Math.max(.02, .5 / candidate.fps)) throw new Error('视频时长与轨迹不一致，无法同步播放');
        await schedule(async () => { d = candidate; prepare(); await draw(); }, id);
        if (id !== generation) return;
        lastFrame = -1; scrub.max = d.frames - 1; scrub.value = 0;
        video.playbackRate = rate; $('rateSelect').value = String(rate); enable(true);
        $('fps').textContent = `${d.fps} Hz`;
        render(0);
        message(`episode ${d.episode} 已加载${d.warnings.length ? ' · ' + d.warnings.join('；') : ''}`);
    } catch (error) {
        if (id === generation) {
            enable(false);
            video.removeAttribute('src'); video.load();
            if (mediaRelease) { mediaRelease(); mediaRelease = null; }
            message(error.name === 'AbortError' ? '加载超时，请重试' : error.message, true);
        } else candidate?.release?.();
    } finally { clearTimeout(timer); }
}
// 完整坐标和 hover 信息只计算一次，播放时复用。
function prepare() {
    cache = {};
    document.querySelector('.plot-toolbar > b').textContent = d.point_label || 'EE';
    const single = Boolean(d.single_arm);
    for (const id of ['right-x','right-y','right-z','right-v']) $(id).style.display = single ? 'none' : '';
    document.querySelectorAll('[data-arm="right"]').forEach(el => el.style.display = single ? 'none' : '');
    for (const arm of Object.keys(ARMS)) {
        const speeds = d[`${arm}_speed`];
        cache[arm] = { x: d[arm].map(p => p[0]), y: d[arm].map(p => p[1]), z: d[arm].map(p => p[2]), speeds, stats: speedStats(speeds),
            frames: Array.from({ length: d.frames }, (_, i) => i),
            custom: speeds.map((v, i) => [i, d.source_frames?.[i] ?? i, i / d.fps, v == null ? '—' : v.toFixed(4)]) };
    }
    const colors = cache.left.frames.map(i => {
        const l = cache.left.speeds[i] > Math.max((cache.left.stats.median || 0) * .35, 1e-4);
        const r = !single && cache.right.speeds[i] > Math.max((cache.right.stats.median || 0) * .35, 1e-4);
        return l && r ? '#c78fa7' : l ? '#d39a7b' : r ? '#9ab9df' : '#6b625d';
    });
    const stops = []; let start = 0;
    for (let i = 1; i <= colors.length; i++) if (i === colors.length || colors[i] !== colors[start]) {
        stops.push(`${colors[start]} ${100 * start / colors.length}% ${100 * i / colors.length}%`); start = i;
    }
    $('timeline-track').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
}

// 保留完整轨迹上的每个采样点，已播放轨迹加深、当前点放大。
function traces(arm) {
    const a = ARMS[arm], c = cache[arm];
    const hovertemplate = `frame %{customdata[0]} · source %{customdata[1]}<br>t=%{customdata[2]:.3f} s<br>x=%{x:.4f} m · y=%{y:.4f} m · z=%{z:.4f} m<br>v=%{customdata[3]} m/s<extra>${a.name}</extra>`;
    const common = { type: 'scatter3d', mode: 'lines+markers', hovertemplate, showlegend: false };
    return [
        { ...common, x: c.x, y: c.y, z: c.z, customdata: c.custom, name: `${a.name}完整`, opacity: .28, line: { color: a.color, width: 3 }, marker: { color: a.color, size: 2.8 } },
        { ...common, x: c.x.slice(0, 1), y: c.y.slice(0, 1), z: c.z.slice(0, 1), customdata: c.custom.slice(0, 1), name: `${a.name}已播放`, line: { color: a.color, width: 7 }, marker: { color: a.color, size: 2.8 } },
        { ...common, x: [c.x[0]], y: [c.y[0]], z: [c.z[0]], customdata: [c.custom[0]], name: `${a.name}当前`, mode: 'markers', marker: { color: a.light, size: 8 } },
    ];
}

// 鼠标操作只更新了 WebGL 相机时，将事件快照同步给下一次 restyle 使用的布局。
function rememberCamera(event) {
    const camera = event['scene.camera'];
    if (camera) $('plot').layout.scene.camera = structuredClone(camera);
}

// 数据加载时初始化，重载只绑定一份相机和点选监听器。
async function draw() {
    const plot = $('plot'), camera = trajectoryInitialCamera(d);
    plot.removeAllListeners?.('plotly_click');
    plot.removeListener?.('plotly_relayouting', rememberCamera);
    plot.removeListener?.('plotly_relayout', rememberCamera);
    await Plotly.react(plot, Object.keys(ARMS).flatMap(traces), {
        uirevision: `ee-trajectory-${generation}`, paper_bgcolor: '#fffaf5', font: { color: '#262421' }, margin: { l: 0, r: 0, t: 30, b: 0 },
        scene: { aspectmode: 'data', camera, xaxis: { title: { text: 'X (m)' } }, yaxis: { title: { text: 'Y (m)' } }, zaxis: { title: { text: 'Z (m)' } } },
    }, { responsive: true, displaylogo: false });
    plot.on('plotly_click', clicked);
    // 拖动过程中也同步，播放与旋转同时发生时不会使用上一次松手的视角。
    plot.on('plotly_relayouting', rememberCamera);
    plot.on('plotly_relayout', rememberCamera);
    for (const arm of Object.keys(ARMS)) {
        const a = ARMS[arm], c = cache[arm], chart = $(`speed-chart-${arm}`);
        const series = [
            { x: c.frames, y: c.speeds, customdata: c.custom, mode: 'lines', name: '全部', line: { color: '#cfc4bb', width: 1.2 } },
            { x: [0], y: [null], customdata: [c.custom[0]], mode: 'lines', name: '已播放', line: { color: a.color, width: 2 } },
        ];
        for (const [key, label, dash] of [['median', '中位数', 'dash'], ['mean', '平均数', 'dot']]) if (c.stats[key] !== null) {
            series.push({ x: [0, Math.max(1, d.frames - 1)], y: [c.stats[key], c.stats[key]], mode: 'lines', name: label, line: { color: a.light, width: 1.4, dash }, hovertemplate: `${label}（非零）%{y:.4f} m/s<extra></extra>` });
        }
        chart.removeAllListeners?.('plotly_click');
        await Plotly.react(chart, series, {
            margin: { l: 38, r: 8, t: 35, b: 20 }, autosize: true, paper_bgcolor: '#fffaf5', plot_bgcolor: '#fffaf5',
            font: { color: '#62584f', size: 9 }, title: { text: `${a.name}速度`, font: { size: 11 }, x: .02 },
            legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 9 } },
            xaxis: { range: [0, Math.max(1, d.frames - 1)], gridcolor: '#eadfd5' }, yaxis: { title: { text: 'm/s' }, gridcolor: '#eadfd5' }, uirevision: `${generation}-${arm}`,
        }, { responsive: true, displaylogo: false });
        chart.on('plotly_click', clicked);
    }
    await applyArmView();
}

// 点选只接受真正带有采样帧号的轨迹点。
function clicked(event) {
    const frame = event.points?.[0]?.customdata?.[0];
    if (Number.isInteger(frame)) seek(frame);
}

// 同一帧不重复渲染；绘图积压时仅保留最新目标帧。
function render(frame) {
    if (!ready || !Number.isInteger(frame) || frame === lastFrame) return;
    pendingFrame = frame;
    if (rendering || rotating) return;
    rendering = true;
    schedule(async () => {
        if (!ready || rotating || pendingFrame === null) return;
        const f = pendingFrame; pendingFrame = null;
        // 只更新轨迹数据，保留用户最新相机，不写回异步绘图前的旧视角。
        const update = { x: [], y: [], z: [], customdata: [] }, indices = [], speeds = [];
        for (const arm of Object.keys(ARMS)) {
            const c = cache[arm], a = ARMS[arm];
            for (const axis of ['x', 'y', 'z']) update[axis].push(c[axis].slice(0, f + 1), [c[axis][f]]);
            update.customdata.push(c.custom.slice(0, f + 1), [c.custom[f]]); indices.push(a.base + 1, a.base + 2);
            speeds.push(Plotly.restyle(`speed-chart-${arm}`, { x: [c.frames.slice(0, f + 1)], y: [c.speeds.slice(0, f + 1)], customdata: [c.custom.slice(0, f + 1)] }, [1]));
            for (const axis of ['x', 'y', 'z']) $(`${arm}-${axis}`).textContent = c[axis][f].toFixed(4);
            $(`${arm}-v`).textContent = c.speeds[f] == null ? '—' : c.speeds[f].toFixed(4);
        }
        await Promise.all([Plotly.restyle('plot', update, indices), ...speeds]);
        lastFrame = f; scrub.value = f;
        $('frame').textContent = d.source_frames ? `输出帧 ${f}/${d.frames - 1} · 原始帧 ${d.source_frames[f]}` : `frame ${f}/${d.frames - 1}`;
        $('time').textContent = `${(f / d.fps).toFixed(3)} s`;
    }).catch(() => {}).finally(() => { rendering = false; if (ready && !rotating && pendingFrame !== null) render(pendingFrame); });
}

// 视频继续播放，松开鼠标后轨迹直接追上最新帧，不逐帧补画。
$('plot').addEventListener('pointerdown', () => { rotating = true; }, { capture: true });
for (const name of ['pointerup', 'pointercancel', 'blur']) window.addEventListener(name, () => {
    rotating = false;
    if (ready && pendingFrame !== null) render(pendingFrame);
});

// 坐标轴和相机保持原样，仅切换图层可见性。
async function applyArmView() {
    for (const [arm, a] of Object.entries(ARMS)) await Plotly.restyle('plot', { visible: !d.single_arm || arm !== 'right' ? (armView === 'both' || armView === arm) : false }, [a.base, a.base + 1, a.base + 2]);
}
function setArmView(mode) {
    if (ready) { armView = mode; schedule(applyArmView).catch(() => {}); }
}
function setRate(rate) { if (ready) video.playbackRate = rate; }

// 按视频帧区间取整；微小容差抵消 MP4 时间戳舍入。
function currentFrame(time = video.currentTime) { return Math.max(0, Math.min(d.frames - 1, Math.floor(time * d.fps + 1e-4))); }
function seek(frame) { if (ready) video.currentTime = (Math.max(0, Math.min(d.frames - 1, frame)) + .001) / d.fps; }
function stopClock() {
    if (clock === null) return;
    video.cancelVideoFrameCallback ? video.cancelVideoFrameCallback(clock) : cancelAnimationFrame(clock);
    clock = null;
}

// 优先用实际呈现帧驱动轨迹，旧浏览器回退到动画回调。
function startClock() {
    stopClock(); const id = generation;
    const tick = (_now, metadata) => {
        clock = null;
        if (!ready || id !== generation) return;
        if (!video.seeking) render(currentFrame(metadata?.mediaTime ?? video.currentTime));
        if (!video.paused && !video.ended) next();
    };
    const next = () => { clock = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick); };
    if (ready && !video.paused) next();
}

// 两个分隔条共用拖动逻辑，保留原来的横向/纵向布局调整。
function resizer(id, parent, axis, initial, min, max, update) {
    const handle = $(id); let ratio = initial, pointer = null, origin = 0, start = initial;
    const set = value => { ratio = Math.max(min, Math.min(max, value)); update(ratio); handle.setAttribute('aria-valuenow', Math.round(ratio * 100)); };
    handle.addEventListener('pointerdown', e => {
        if (e.button !== 0) return;
        pointer = e.pointerId; origin = axis === 'x' ? e.clientX : e.clientY; start = ratio;
        handle.setPointerCapture(pointer); e.preventDefault();
    });
    handle.addEventListener('pointermove', e => {
        if (pointer === e.pointerId) set(start + ((axis === 'x' ? e.clientX : e.clientY) - origin) / Math.max(1, axis === 'x' ? parent.clientWidth : parent.clientHeight));
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) handle.addEventListener(name, () => { pointer = null; });
    handle.addEventListener('dblclick', () => set(initial));
    handle.addEventListener('keydown', e => {
        const minus = axis === 'x' ? 'ArrowLeft' : 'ArrowUp', plus = axis === 'x' ? 'ArrowRight' : 'ArrowDown';
        if ([minus, plus, 'Home'].includes(e.key)) { e.preventDefault(); set(e.key === 'Home' ? initial : ratio + (e.key === plus ? .05 : -.05)); }
    });
}


enable(false);
    video.addEventListener('play', startClock);
    for (const name of ['pause', 'ended']) video.addEventListener(name, () => { stopClock(); if (ready && !video.seeking) render(currentFrame()); });
    video.addEventListener('seeked', () => { if (ready) { render(currentFrame()); startClock(); } });
    video.addEventListener('timeupdate', () => { if (ready && !video.seeking && (video.paused || !video.requestVideoFrameCallback)) render(currentFrame()); });
    video.addEventListener('error', () => { if (ready) { enable(false); video.pause(); stopClock(); message('视频播放失败，请重新加载 episode', true); } });
    video.addEventListener('ratechange', () => { if (ready) $('rateSelect').value = String(video.playbackRate); });
    scrub.addEventListener('input', () => seek(Number(scrub.value)));
    document.addEventListener('keydown', async e => {
        if (document.querySelector('dialog[open]') || !ready || e.code !== 'Space' || e.repeat || e.target.closest('input, select, textarea, button, [contenteditable="true"], [role="separator"]')) return;
        e.preventDefault();
        try { video.paused ? await video.play() : video.pause(); } catch { message('无法播放视频，请重试', true); }
    });
    const layout = document.querySelector('.layout'), panel = document.querySelector('.video-panel');
    resizer('column-resizer', layout, 'x', .62, .3, .7, r => layout.style.gridTemplateColumns = `minmax(0, ${r}fr) 8px minmax(0, ${1-r}fr)`);
    resizer('row-resizer', panel, 'y', .5, .2, .8, r => { panel.style.setProperty('--video-share', `${r}fr`); panel.style.setProperty('--charts-share', `${1-r}fr`); });
    let timer;
    const observer = new ResizeObserver(() => {
        clearTimeout(timer); timer = setTimeout(() => { if (ready) schedule(() => Promise.all(['plot', 'speed-chart-left', 'speed-chart-right'].map(id => Plotly.Plots.resize(id)))).catch(() => {}); }, 100);
    });
    for (const id of ['plot', 'speed-chart-left', 'speed-chart-right']) observer.observe($(id));
    window.addEventListener('pagehide', () => { if (mediaRelease) mediaRelease(); controller?.abort(); stopClock(); observer.disconnect(); clearTimeout(timer); });

for (const button of document.querySelectorAll('[data-arm]')) button.addEventListener('click', () => setArmView(button.dataset.arm));
$('rateSelect').addEventListener('change', () => setRate(Number($('rateSelect').value)));
return { load, message };
};
