import { parquetReadObjects, parquetMetadataAsync } from 'hyparquet'
import { compressors } from 'hyparquet-compressors'
import { openEpisode, readSourceMap } from './local-dataset.js'
import { openExample } from './example-dataset.js'

const $ = (id) => document.getElementById(id)
// build.py 从 ee_video_viewer.yaml 注入公开类别配置。
const PROFILES = PUBLIC_PROFILES
const DEFAULT_PROFILE = PUBLIC_DEFAULT_PROFILE

let viewer;
let profile = PROFILES[DEFAULT_PROFILE]
let previousProfileId = DEFAULT_PROFILE
let directoryHandle = null
let sourceMode = 'example' // 明确区分内置示例、用户目录和单独文件，避免旧文件选择覆盖新示例。
let exampleRoot = PROFILES.H01.exampleRoot

// 显示当前示例路径；自定义类别沿用最近选择的示例数据集。
function showExampleSource() {
  exampleRoot = profile.exampleRoot || exampleRoot
  $('directory-name').textContent = `示例 · ${exampleRoot.split('/').at(-1)}`
  $('directory-input').title = `${exampleRoot}；点击选择本地数据集`
}

// 用户可从本地导入显式切回当前类别的默认示例。
function useExample() {
  sourceMode = 'example'
  showExampleSource()
  $('episode-input').value = 0
  loadEpisode()
}

// 将浏览器 File 包装成 hyparquet 所需的异步字节缓冲区。
function fileBuffer(file, signal) {
  return { byteLength: file.size, slice: (start, end) => { signal?.throwIfAborted(); return file.slice(start, end).arrayBuffer() } }
}

// 所有用户提示沿用共享查看器。
function message(text, error=false) { viewer.message(text, error) }

// 选择目录只保存句柄，不枚举文件，也不读取 meta 或 parquet。
async function chooseDirectory() {
  if (!window.showDirectoryPicker) {
    message('请使用 Chrome / Edge 选择目录', true)
    $('manual-import-dialog').showModal()
    return
  }
  try {
    const handle = await window.showDirectoryPicker({ mode: 'read', id: 'lerobot-dataset' })
    directoryHandle = handle
    sourceMode = 'directory'
    $('directory-name').textContent = handle.name
    $('directory-input').title = handle.name
    message('目录已选择')
  } catch (error) {
    if (error.name === 'AbortError') return
    message('目录选择失败，请重试', true)
  }
}

// 读取并校验自定义类别配置，避免把错误索引带入 parquet 解析。
function customProfile() {
  const indices = $('custom-indices').value.split(',').map((value) => value.trim() ? Number(value.trim()) : NaN)
  const direction = $('custom-direction').value.split(',').map((value) => value.trim() ? Number(value.trim()) : NaN)
  if (!$('custom-video-key').value.trim() || !$('custom-arms-key').value.trim() || indices.length !== 6 || direction.length !== 6 || indices.some((value) => !Number.isInteger(value) || value < 0) || direction.some((value) => value !== 1 && value !== -1)) throw new Error('自定义配置需要视频键、位置字段，以及 6 个非负索引和 6 个 ±1 方向系数')
  const fps = $('custom-fps').value.trim() ? $('custom-fps').valueAsNumber : null
  if ($('custom-fps').validity.badInput || fps !== null && (!Number.isFinite(fps) || fps <= 0)) throw new Error('FPS 必须为正数，或留空自动读取')
  const value = { label: '自定义', videoKey: $('custom-video-key').value.trim(), armsKey: $('custom-arms-key').value.trim(), indices, direction, fps }
  localStorage.setItem('h01-public-custom-profile', JSON.stringify(value))
  return value
}

// 恢复上次保存的自定义类别配置。
function loadCustomProfile() {
  try {
    const value = JSON.parse(localStorage.getItem('h01-public-custom-profile') || 'null')
    if (value) {
      $('custom-video-key').value = value.videoKey || ''
      $('custom-arms-key').value = value.armsKey || ''
      $('custom-indices').value = (value.indices || []).join(',')
      $('custom-direction').value = (value.direction || []).join(',')
      $('custom-fps').value = value.fps ?? ''
      return value
    }
  } catch { /* 无效配置将在弹窗中重新填写。 */ }
  $('custom-video-key').value = ''
  $('custom-arms-key').value = ''
  $('custom-indices').value = ''
  $('custom-direction').value = ''
  $('custom-fps').value = ''
  return null
}

// 自定义已选中时仍能通过独立按钮打开编辑，不依赖 select 的 change 事件。
function editCustomProfile() {
  loadCustomProfile()
  $('custom-profile-error').textContent = ''
  $('profile-dialog').showModal()
}

// 从 parquet 记录中读取向量字段，兼容 hyparquet 返回的数组和 TypedArray。
function vector(row, key) {
  const value = row[key]
  if (!value || typeof value.length !== 'number') throw new Error(`Parquet 字段 ${key} 不是向量`)
  return Array.from(value, item => item == null ? NaN : Number(item))
}

// 将本地 parquet 转换为现有查看器使用的数据结构。
async function readParquet(file, profile, fps, signal) {
  const buffer = fileBuffer(file, signal)
  const metadata = await parquetMetadataAsync(buffer)
  if (Number(metadata.num_rows) < 1 || Number(metadata.num_rows) > 100000) throw new Error('帧数必须在 1 到 100000 之间')
  const rows = await parquetReadObjects({
    file: buffer,
    metadata,
    columns: [profile.armsKey, 'timestamp', 'frame_index'],
    compressors,
  })
  if (!rows.length || rows.length > 100000) throw new Error('帧数必须在 1 到 100000 之间')
  // 单独文件没有 info.json 时从时间戳推算；后续仍校验所有帧时间戳。
  if (fps == null && rows.length > 1) fps = (rows.length - 1) / (Number(rows.at(-1).timestamp) - Number(rows[0].timestamp))
  if (!Number.isFinite(fps) || fps <= 0) throw new Error('无法读取 FPS，请在自定义配置中填写实际采样频率')
  const right = [], left = [], timestamps = []
  let dimension
  rows.forEach((row, index) => {
    const ee = vector(row, profile.armsKey)
    dimension ??= ee.length
    if (ee.length !== dimension) throw new Error('位置向量必须等长')
    if (profile.indices.some((axis) => axis >= ee.length || !Number.isFinite(ee[axis]))) throw new Error('末端位姿向量维度不足以支持当前类别索引')
    left.push(profile.indices.slice(0, 3).map((axis, i) => ee[axis] * profile.direction[i]))
    right.push(profile.indices.slice(3, 6).map((axis, i) => ee[axis] * profile.direction[i + 3]))
    timestamps.push(Number(row.timestamp))
    if (Number(row.frame_index) !== index) throw new Error('frame_index 必须从 0 连续编号')
  })
  if (!timestamps.every((value, index) => Math.abs(value - index / fps) <= 1e-5)) throw new Error('timestamp 与 FPS 不一致，请在自定义配置中修改 FPS')
  const speed = (points) => [null, ...points.slice(1).map((point, index) => Math.hypot(...point.map((value, axis) => (value - points[index][axis]) * fps)))]
  return { warnings: [], single_arm: profile.indices.slice(0,3).every((v,i) => v === profile.indices[i+3] && profile.direction[i] === profile.direction[i+3]), episode: '本地文件', fps, frames: rows.length, duration: rows.length / fps, right, left, right_speed: speed(right), left_speed: speed(left) }
}

// 点击加载时快照目录和 Episode，仅定位这一条；单文件模式保留原导入方式。
async function filesFromSelection(profile, signal) {
  const root = directoryHandle
  if (sourceMode === 'directory') return openEpisode(root, $('episode-input').valueAsNumber, profile.videoKey, signal)
  if (sourceMode === 'example') return openExample(profile.exampleRoot || exampleRoot, $('episode-input').valueAsNumber, profile.videoKey, signal)
  return { parquet: $('parquet-input').files[0], video: $('video-input').files[0], mapping: $('mapping-input').files[0], multiEpisode: false, fps: null }
}

// 公开入口只负责本地数据读取，显示和视频时钟由共享核心维护。
async function loadEpisode() {
  const chosenProfile = structuredClone(profile)
  await viewer.load(async signal => {
    const { parquet, video: videoFile, video_url: directVideoUrl, mapping, multiEpisode, fps } = await filesFromSelection(chosenProfile, signal)
    if (!parquet || (!videoFile && !directVideoUrl)) throw new Error('请选择 parquet 和对应的 MP4 视频')
    const data = await readParquet(parquet, chosenProfile, chosenProfile.fps ?? fps, signal)
    data.episode = Number(parquet.name.match(/^episode_(\d+)\.parquet$/)?.[1] ?? 0)
    signal.throwIfAborted()
    $('episode-input').value = data.episode
    data.source_frames = mapping ? await readSourceMap(mapping, data.episode, data.frames, multiEpisode, data.warnings, signal) : null
    if (signal.aborted) throw new DOMException('加载已取消', 'AbortError')
    const url = directVideoUrl || URL.createObjectURL(videoFile)
    return { ...data, video_url: url, release: directVideoUrl ? undefined : () => URL.revokeObjectURL(url) }
  })
}

// 初始化文件选择、播放事件和 Plotly 交互。
document.addEventListener('DOMContentLoaded', () => {
  viewer = createTrajectoryViewer()
  $('profile-input').replaceChildren(...Object.entries(PROFILES).map(([id, item]) => new Option(item.label, id)), new Option('自定义', '__custom__'))
  $('profile-input').addEventListener('change', (event) => {
    const id = event.target.value
    if (id === '__custom__') {
      editCustomProfile()
      return
    }
    previousProfileId = id
    profile = PROFILES[id] || PROFILES.H01
    $('edit-custom-profile').hidden = true
    if (sourceMode === 'example') useExample()
  })
  $('edit-custom-profile').addEventListener('click', editCustomProfile)
  $('custom-profile-confirm').addEventListener('click', () => {
    try {
      profile = customProfile()
      previousProfileId = '__custom__'
      $('edit-custom-profile').hidden = false
      $('profile-dialog').close()
      message('自定义类别已保存，请选择数据集目录或单独文件后加载')
    } catch (error) { $('custom-profile-error').textContent = error.message }
  })
  $('profile-dialog').addEventListener('close', () => {
    if ($('profile-input').value === '__custom__' && previousProfileId !== '__custom__') {
      $('profile-input').value = previousProfileId
      profile = PROFILES[previousProfileId] || PROFILES[DEFAULT_PROFILE]
    }
    $('edit-custom-profile').hidden = previousProfileId !== '__custom__'
  })
  $('manual-import-button').addEventListener('click', () => $('manual-import-dialog').showModal())
  $('manual-confirm-button').addEventListener('click', () => {
    if (!$('parquet-input').files.length || !$('video-input').files.length) return
    directoryHandle = null
    sourceMode = 'files'
    $('directory-name').textContent = '单独文件'
    $('directory-input').title = '当前使用单独文件；点击选择本地数据集'
    $('manual-import-dialog').close()
    loadEpisode()
  })
  $('episode-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') loadEpisode() })
  $('load-button').addEventListener('click', loadEpisode)
  $('directory-input').addEventListener('click', chooseDirectory)
  $('example-button').addEventListener('click', useExample)
  $('parquet-input').addEventListener('change', () => { if ($('parquet-input').files.length) message('Parquet 已选择') })
  $('video-input').addEventListener('change', () => { if ($('video-input').files.length) message('视频已选择') })
  useExample()
})
