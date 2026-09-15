# H01 TraceLab 公共查看器

这是 GitHub Pages 用的纯浏览器版本。它不依赖 FastAPI、Python 或服务器文件路径，访客选择本地 parquet 和对应的 `cam_fisheye_front` MP4；单独文件入口通过弹窗打开后，浏览器直接解析并同步显示轨迹。

## 支持的评测数据集演示

公开页面内置 H01、RoboDojo、Ebench、Robocasa365 和 RoboTwin2 的 episode 0 演示；选择类别后点击“查看示例”即可加载。类别映射统一来自根目录 `ee_video_viewer.yaml`。

## 使用

首次打开自动展示 YAML 中的第一个类别（当前为 H01）的 episode 0。在示例模式下切换类别，会自动加载该类别的 episode 0。服务器版和公开版共用根目录 `ee_video_viewer.yaml`：服务器运行时直接读取，公开版执行 `python public-src/build.py` 时注入同一份配置，源码中不再维护第二份类别表。

| 页面类别 | 仓库目录 | 当前示例 |
| --- | --- | --- |
| H01 | `public/example/H01` | episode 0，783 帧，30 Hz |
| Ebench | `public/example/Ebench` | episode 0，3324 帧，15 Hz |
| Robocasa365 | `public/example/RoboCasa365` | episode 0，1272 帧，20 Hz；注意目录大小写 |
| RoboDojo | `public/example/RoboDojo` | episode 0，579 帧，25 Hz |
| RoboTwin2 | `public/example/RoboTwin2` | episode 0，462 帧，50 Hz |

每个目录使用以下布局；视频键与类别的 `videoKey` 一致：

```text
example/<类别目录>/
  meta/info.json
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<视频键>/episode_000000.mp4
source_frame_map.csv  # 可选
```

`source_frame_map.csv` 只在数据经过抽帧、裁剪或重构后使用，用来把当前输出帧映射回源数据帧。格式至少包含 `output_frame,source_frame` 两列；多个 episode 时还应包含 `episode_index`。没有该文件时仍可正常查看，页面只显示当前 episode 的帧号。

示例读取器按 `info.json` 的 `fps` 和 `chunks_size` 读取频率及定位 chunk（未提供 chunk 大小时采用 1000）；缺失文件会显示具体路径。已有类别补齐同路径数据无需改 JS；添加新类别时在 YAML 增加字段映射，放入对应示例目录，再运行 `python public-src/build.py`，下拉框自动生成。所有默认路径使用站点相对地址，不使用服务器绝对路径。

用户导入目录或文件后，切换类别只切换字段映射，不自动覆盖导入的数据；点击“查看示例”可切回当前类别的默认示例。自定义配置在示例模式中使用最近选择的示例目录。

示例模式不会把整段 MP4 下载成浏览器 Blob：只读取元数据和 Parquet，视频直接交给浏览器按需请求和缓冲，因此切换类别更快；本地单独导入仍使用浏览器本地文件对象。

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
python build.py
```

依赖版本固定在 `public-src/package.json`，许可证文件随静态资源一起发布。构建脚本从根目录 `ee_video_viewer.yaml` 注入类别配置，按实际目录匹配 `example/<类别名>`（例如 `Robocasa365` 对应 `RoboCasa365`），并刷新 JS 资源版本号。修改 YAML 后运行此构建命令，再刷新页面；静态服务器不会直接执行 YAML。

## 类别与 FPS

页面类别下拉框与服务器版 `ee_video_viewer.yaml` 对齐：H01、EBench、Robocasa365。类别会选择对应的位置字段、左右 XYZ 索引、方向变换和视频目录；H01 使用 `observation.state_endpose_quat` 与 `(-X,-Y,+Z)`。FPS 在自定义配置弹窗中填写，填写后优先使用该频率。留空时目录模式读取 `meta/info.json` 的 `fps`，单独文件模式从 Parquet 时间戳推算；所有帧的时间戳仍须与采样频率一致。

导入入口分为两种：选择“数据集目录”自动匹配 parquet、类别对应视频和 `meta/info.json`；“单独导入 parquet + 视频”弹窗只选择文件，不再填写 FPS。


“自定义”类别会打开弹窗，可填写视频键、位置字段、6 个左右臂 XYZ 索引、6 个方向系数和 FPS；保存后可随时点击类别旁的“编辑配置”再次修改，取消编辑不会覆盖已保存的值。配置仅保存在浏览器 localStorage，不写入 YAML。

目录导入后，在顶部 `Episode` 输入非负整数（默认 0），点击“加载”或回车，即可匹配同一数据集、同一 chunk 中对应的 parquet 和当前类别视频；成功后显示 `episode N 已加载`。指定序号不存在时明确报错。单独文件弹窗确认后使用所选文件，Episode 从 parquet 文件名识别。

项目仓库：https://github.com/Peaceful-World-X/LeRobot_TraceLab 。右上角访问徽章使用与 Action-Chunking-Survey 相同的第三方服务，统计页面访问而非独立访客人数；徽章加载会请求该服务，所选 parquet/视频仍在浏览器本地处理。计数服务不可用时显示文字提示。

## 与服务器版一致的工作区

两版页面共用 `public/assets/viewer-core.js` 和 `public/assets/viewer.css`，抬头下方的轨迹、速度图、彩色时间轴、文本、双臂切换、点选跳帧、倍速、空格播放及分隔条操作一致。完整轨迹始终保留，播放过的部分加深；速度均值和中位数仅使用非零速度。

时间轴颜色表示末端活动：左臂橙色、右臂蓝色、同时活动紫色、均未超过活动阈值为灰色。目录导入自动读取同一数据集的 `source_frame_map.csv`，显示输出帧和原始帧；单文件弹窗可额外选择该 CSV。没有映射时只显示当前帧，无效映射显示提示，不伪造原始帧号。

启动服务器版时，需要保留上述两个共享文件。部署 GitHub Pages 时上传完整 `public/`，修改共享核心或 CSS 无需重建 bundle；修改本地读取入口 `public-src/viewer.js` 才需要重新打包。

浏览器对照测试：分别在 19112 启动服务器版、在 19113 启动 public 静态服务，设置 `VIEWER_TEST_DATASET` 为包含 episode 0、6 和来源映射的派生数据集，执行 `node tests/test_viewer_parity.cjs`。测试需要 Playwright；可通过 `PLAYWRIGHT_MODULE` 指定模块路径、`BROWSER_PATH` 指定 Chromium。
