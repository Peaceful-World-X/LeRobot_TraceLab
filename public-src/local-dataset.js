// 只沿相对路径打开句柄；不会递归枚举目录或创建任何文件。
async function directoryAt(root, parts) {
  let current = root
  for (const part of parts) current = await current.getDirectoryHandle(part)
  return current
}

// 类别视频键只能表示一个目录名，防止错误配置越过数据集边界。
function validSegment(value) {
  if (!value || value === '.' || value === '..' || /[/\\]/.test(value)) throw new Error('视频键必须是单个目录名')
  return value
}

// 缺少可选映射可忽略，读取权限等错误必须保留。
async function optionalFile(root, name) {
  try { return await (await root.getFileHandle(name)).getFile() }
  catch (error) { if (error.name === 'NotFoundError') return undefined; throw error }
}

// 按 meta 的 chunk 大小直接定位；旧数据缺少该字段时只枚举 data 的一级 chunk 目录。
async function episodeChunk(data, number, info, signal) {
  const filename = `episode_${String(number).padStart(6, '0')}.parquet`
  if (info.chunks_size !== undefined) {
    const size = Number(info.chunks_size)
    if (!Number.isSafeInteger(size) || size <= 0) throw new Error('meta/info.json 的 chunks_size 无效')
    const chunk = `chunk-${String(Math.floor(number / size)).padStart(3, '0')}`
    return { chunk, handle: await (await data.getDirectoryHandle(chunk)).getFileHandle(filename) }
  }
  let result
  for await (const [chunk, handle] of data.entries()) {
    signal?.throwIfAborted()
    if (handle.kind !== 'directory' || !/^chunk-\d+$/.test(chunk)) continue
    try {
      const file = await handle.getFileHandle(filename)
      if (result) throw new Error('存在同号 episode，请检查 chunk 目录')
      result = { chunk, handle: file }
    } catch (error) { if (error.name !== 'NotFoundError') throw error }
  }
  if (!result) throw new Error(`找不到 episode ${number}`)
  return result
}

// 点击加载时才读取四个相关文件：info、目标 parquet、目标相机视频和可选映射。
export async function openEpisode(root, number, videoKey, signal) {
  if (!Number.isSafeInteger(number) || number < 0) throw new Error('Episode 必须是非负整数')
  validSegment(videoKey)
  signal?.throwIfAborted()
  try {
    const meta = await root.getDirectoryHandle('meta')
    const info = JSON.parse(await (await (await meta.getFileHandle('info.json')).getFile()).text())
    const fps = Number(info.fps)
    if (!Number.isFinite(fps) || fps <= 0) throw new Error('FPS 必须是有限正数')
    signal?.throwIfAborted()
    const { chunk, handle } = await episodeChunk(await root.getDirectoryHandle('data'), number, info, signal)
    const parquet = await handle.getFile()
    const camera = await directoryAt(root, ['videos', chunk, videoKey])
    const video = await (await camera.getFileHandle(parquet.name.replace(/\.parquet$/, '.mp4'))).getFile()
    signal?.throwIfAborted()
    const mapping = await optionalFile(root, 'source_frame_map.csv')
    signal?.throwIfAborted()
    return { parquet, video, mapping, fps, features: info.features, multiEpisode: Number(info.total_episodes) !== 1 }
  } catch (error) {
    if (error.name === 'NotFoundError') throw new Error(`episode ${number} 文件不完整，请检查目录和相机类别`)
    if (error.name === 'NotAllowedError') throw new Error('目录读取权限已失效，请重新选择目录')
    throw error
  }
}

// 按 64 KiB 读取 CSV，跨块保留引号状态；不将整份映射装进内存。
export async function* csvRows(file, signal) {
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let row = [], cell = '', quoted = false, afterQuote = false, rowLength = 0
  for (let offset = 0; offset < file.size; offset += 65536) {
    signal?.throwIfAborted()
    const end = Math.min(offset + 65536, file.size)
    const text = decoder.decode(await file.slice(offset, end).arrayBuffer(), { stream: end < file.size })
    for (const c of text) {
      if (++rowLength > 1048576) throw new Error('CSV 单行过长')
      if (quoted && !afterQuote) {
        if (c === '"') afterQuote = true
        else cell += c
        continue
      }
      if (afterQuote) {
        afterQuote = false
        if (c === '"') { cell += '"'; continue }
        quoted = false
        if (![',', '\n', '\r'].includes(c)) throw new Error('CSV 引号后字符无效')
      }
      if (c === '"') { if (cell) throw new Error('CSV 引号位置无效'); quoted = true }
      else if (c === ',' || c === '\n' || c === '\r') {
        row.push(cell); cell = ''
        if (c !== ',') { if (row.some(value => value.length)) yield row; row = []; rowLength = 0 }
      } else cell += c
    }
    // 让主线程在大型映射扫描中仍可响应取消和界面操作。
    await new Promise(resolve => setTimeout(resolve, 0))
  }
  signal?.throwIfAborted()
  if (quoted && !afterQuote) throw new Error('CSV 引号未闭合')
  if (cell.length || row.length) { row.push(cell); yield row }
}

// 只保留当前 episode 的来源帧数组，错误映射仍沿用两版一致的提示。
export async function readSourceMap(file, episode, frames, multiEpisode, warnings, signal) {
  try {
    let header, ep, output, source
    const result = []
    const integer = text => { if (!/^\d+$/.test(text || '')) throw new Error('帧号无效'); const n = Number(text); if (!Number.isSafeInteger(n)) throw new Error('帧号过大'); return n }
    for await (const row of csvRows(file, signal)) {
      if (!header) {
        header = row; ep = header.indexOf('episode_index'); output = header.indexOf('output_frame'); source = header.indexOf('source_frame')
        if (output < 0 || source < 0 || (multiEpisode && ep < 0)) throw new Error('映射缺少列')
        continue
      }
      if (ep >= 0 && integer(row[ep]) !== episode) continue
      const frame = integer(row[source])
      if (integer(row[output]) !== result.length || result.length >= frames || (result.length && frame <= result[result.length-1])) throw new Error('映射顺序无效')
      result.push(frame)
    }
    if (!header || result.length !== frames) throw new Error('映射长度不符')
    return result
  } catch (error) {
    if (error.name === 'AbortError') throw error
    warnings.push('来源映射无效，仅显示当前文件帧号')
    return null
  }
}
