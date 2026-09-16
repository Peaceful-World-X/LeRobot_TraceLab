// 通过站点相对地址读取示例，兼容 GitHub Pages 的项目子路径。
export async function openExample(root, number, videoKey, signal) {
  if (!Number.isSafeInteger(number) || number < 0) throw new Error('Episode 必须是非负整数')
  if (!videoKey || videoKey === '.' || videoKey === '..' || /[/\\]/.test(videoKey)) throw new Error('视频键必须是单个目录名')
  const base = new URL(`${root}/`, document.baseURI)
  // HTTP 404 直接指出缺失文件，网络和权限错误不能伪装成不存在。
  const fetchFile = async (relative, optional = false) => {
    const response = await fetch(new URL(relative, base), { signal })
    if (optional && response.status === 404) return null
    if (!response.ok) throw new Error(`示例文件无法读取（${response.status}）：${root}/${relative}；可选择本地数据集`)
    return response
  }
  const info = await (await fetchFile('meta/info.json')).json()
  const fps = Number(info.fps), chunkSize = Number(info.chunks_size ?? 1000)
  if (!Number.isFinite(fps) || fps <= 0) throw new Error('示例 meta/info.json 的 FPS 无效')
  if (!Number.isSafeInteger(chunkSize) || chunkSize <= 0) throw new Error('示例 meta/info.json 的 chunks_size 无效')
  const chunk = `chunk-${String(Math.floor(number / chunkSize)).padStart(3, '0')}`
  const stem = `episode_${String(number).padStart(6, '0')}`
  const [parquet] = await Promise.all([
    fetchFile(`data/${chunk}/${stem}.parquet`).then(async r => new File([await r.blob()], `${stem}.parquet`)),
  ])
  signal?.throwIfAborted()
  const videoUrl = new URL(`videos/${chunk}/${encodeURIComponent(videoKey)}/${stem}.mp4`, base).href
  return { parquet, video: null, video_url: videoUrl, mapping: null, fps, features: info.features, multiEpisode: Number(info.total_episodes) !== 1 }
}
