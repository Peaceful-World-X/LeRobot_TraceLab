# H01 TraceLab 公共查看器

这是 GitHub Pages 用的纯浏览器版本。它不依赖 FastAPI、Python 或服务器文件路径，访客选择本地 parquet 和对应的 `cam_fisheye_front` MP4；单独文件入口通过弹窗打开后，浏览器直接解析并同步显示轨迹。

## 使用

1. 使用 Chrome / Edge，在 HTTPS 网站或 localhost 打开页面，点击“选择目录”。页面只保存只读目录句柄，不枚举数据集中的文件。
2. 输入 Episode 并点击“加载”，才读取 `meta/info.json`、该 Episode 的 parquet、类别对应 MP4，以及可选 `source_frame_map.csv`；FPS 自动读取。
3. `chunks_size` 用于直接定位 chunk；旧元数据没有该字段时，只扫描 `data` 下一级 chunk 目录，不递归列举其中文件或全部相机。
4. 来源映射 CSV 每次最多读取 64 KiB，并只保留当前 Episode 的映射；若使用全数据集 CSV，仍需顺序扫描该文件，但不会把全部内容装入内存。视频继续用浏览器 Blob 按需播放。
5. 不支持目录句柄的浏览器使用“单独导入”弹窗，选择 parquet、MP4 和可选映射；不会使用会枚举整个目录的 `webkitdirectory`。文件不上传服务器。

目录句柄只在当前页面保留，刷新后重新选择。状态栏只显示“目录已选择”或“episode N 已加载”；长错误单行省略，悬停可查看完整内容。

## 本地测试

```bash
python -m http.server 19100 --directory public
```

`assets/h01-viewer-public.js` 内置 hyparquet 和 Snappy 解码器，`assets/plotly.min.js` 为本地 Plotly 资源，因此页面不依赖 CDN。`ee_video_viewer.py` 仍是读取服务器路径的后端版本，两者互不影响。

## 重新生成 bundle

```bash
cd public-src
npm install
npx esbuild viewer.js --bundle --format=iife --platform=browser --minify --outfile=../public/assets/h01-viewer-public.js
```

依赖版本固定在 `public-src/package.json`，许可证文件随静态资源一起发布。

## 类别与 FPS

页面类别下拉框与服务器版 `ee_video_viewer.yaml` 对齐：H01、EBench、Robocasa365。类别会选择对应的位置字段、左右 XYZ 索引、方向变换和视频目录；H01 使用 `observation.state_endpose_quat` 与 `(-X,-Y,+Z)`。选择完整数据集目录时自动读取 `meta/info.json` 的 `fps`，FPS 输入框仅作为单独选择 parquet/MP4 且没有元数据时的兜底。

导入入口分为两种：默认优先选择“导入数据集目录”，页面会自动匹配 parquet、类别对应视频和 `meta/info.json`；“单独导入 parquet + 视频”入口默认收起，展开后可分别选择文件，FPS 仅作为无元数据时的兜底。


“自定义”类别会打开弹窗，可填写视频键、位置字段、6 个左右臂 XYZ 索引和 6 个方向系数；保存后目录自动识别与 parquet 解析均使用该配置，配置仅保存在浏览器 localStorage。

目录导入后，在顶部 `Episode` 输入非负整数（默认 0），点击“加载”或回车，即可匹配同一数据集、同一 chunk 中对应的 parquet 和当前类别视频；成功后显示 `episode N 已加载`。指定序号不存在时明确报错。单独文件弹窗确认后使用所选文件，Episode 从 parquet 文件名识别。

项目仓库：https://github.com/Peaceful-World-X/LeRobot_TraceLab 。右上角访问徽章使用与 Action-Chunking-Survey 相同的第三方服务，统计页面访问而非独立访客人数；徽章加载会请求该服务，所选 parquet/视频仍在浏览器本地处理。计数服务不可用时显示文字提示。

## 与服务器版一致的工作区

两版页面共用 `public/assets/viewer-core.js` 和 `public/assets/viewer.css`，抬头下方的轨迹、速度图、彩色时间轴、文本、双臂切换、点选跳帧、倍速、空格播放及分隔条操作一致。完整轨迹始终保留，播放过的部分加深；速度均值和中位数仅使用非零速度。

时间轴颜色表示末端活动：左臂橙色、右臂蓝色、同时活动紫色、均未超过活动阈值为灰色。目录导入自动读取同一数据集的 `source_frame_map.csv`，显示输出帧和原始帧；单文件弹窗可额外选择该 CSV。没有映射时只显示当前帧，无效映射显示提示，不伪造原始帧号。

启动服务器版时，需要保留上述两个共享文件。部署 GitHub Pages 时上传完整 `public/`，修改共享核心或 CSS 无需重建 bundle；修改本地读取入口 `public-src/viewer.js` 才需要重新打包。

浏览器对照测试：分别在 19112 启动服务器版、在 19113 启动 public 静态服务，设置 `VIEWER_TEST_DATASET` 为包含 episode 0、6 和来源映射的派生数据集，执行 `node tests/test_viewer_parity.cjs`。测试需要 Playwright；可通过 `PLAYWRIGHT_MODULE` 指定模块路径、`BROWSER_PATH` 指定 Chromium。
