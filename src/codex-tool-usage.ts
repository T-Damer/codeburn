import { existsSync } from 'fs'
import { readdir, stat } from 'fs/promises'
import { basename, join } from 'path'
import { homedir } from 'os'

import { readSessionLines, type SessionLine } from './fs-utils.js'
import { estimateTokensFromChars } from './token-estimate.js'

export type ToolUsageMetric = 'context-exposure' | 'next-request' | 'result-tokens' | 'subagent' | 'total'

export type TokenUsage = {
  inputTokens: number
  cachedInputTokens: number
  outputTokens: number
  reasoningTokens: number
}

export type CodexToolInvocation = {
  sessionId: string
  parentSessionId?: string
  project?: string
  callId: string
  tool: string
  rawTool: string
  timestamp: string
  argumentTokens: number
  resultTokens: number
  contextExposureTokens: number
  nextRequest: TokenUsage
  allocatedNextRequest: TokenUsage
  nextRequestBatchSize: number
  nextRequestAttribution: 'exact-single-tool' | 'estimated-proportional' | 'unattributed'
  durationMs?: number
  failed: boolean
  resultSizeEstimated: boolean
  childSessionIds: string[]
  subagentUsage: TokenUsage
  /** Internal linkage hints; stripped from the final report. */
  resultIdentifiers?: string[]
}

export type CodexToolUsageRow = {
  tool: string
  calls: number
  failedCalls: number
  argumentTokens: number
  resultTokens: number
  contextExposureTokens: number
  nextRequest: TokenUsage
  subagentUsage: TokenUsage
  estimatedBatchCalls: number
  estimatedResultCalls: number
}

export type CodexToolUsageReport = {
  generatedAt: string
  codexHome: string
  sessionsAnalyzed: number
  rootSessionIds: string[]
  descendantsIncluded: boolean
  parseErrors: number
  oversizedLinesEstimated: number
  rows: CodexToolUsageRow[]
  invocations: CodexToolInvocation[]
}

type RolloutRef = {
  filePath: string
  sessionId: string
  mtimeMs: number
}

type SessionMeta = {
  sessionId: string
  parentSessionId?: string
  project?: string
  startedAt?: string
}

type ParsedSession = {
  meta: SessionMeta
  invocations: CodexToolInvocation[]
  usage: TokenUsage
  parseErrors: number
  oversizedLinesEstimated: number
}

type PendingInvocation = CodexToolInvocation & {
  startedAtMs?: number
  resultIdentifiers: string[]
}

type CodexEntry = {
  type?: string
  timestamp?: string
  payload?: Record<string, unknown> & {
    type?: string
    role?: string
    name?: string
    call_id?: string
    id?: string
    cwd?: string
    session_id?: string
    forked_from_id?: string
    arguments?: unknown
    input?: unknown
    action?: unknown
    output?: unknown
    result?: unknown
    info?: {
      last_token_usage?: Partial<Record<'input_tokens' | 'cached_input_tokens' | 'output_tokens' | 'reasoning_output_tokens' | 'total_tokens', number>>
      total_token_usage?: Partial<Record<'input_tokens' | 'cached_input_tokens' | 'output_tokens' | 'reasoning_output_tokens' | 'total_tokens', number>>
    }
  }
}

const ZERO_USAGE: TokenUsage = {
  inputTokens: 0,
  cachedInputTokens: 0,
  outputTokens: 0,
  reasoningTokens: 0,
}

const ROLLOUT_RE = /^rollout-.{19}-(.+)\.jsonl$/
const UUID_RE = /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/gi
const DEFAULT_LARGE_JSON_CAP = 64 * 1024 * 1024

function cloneUsage(usage: TokenUsage = ZERO_USAGE): TokenUsage {
  return { ...usage }
}

function addUsage(target: TokenUsage, source: TokenUsage): void {
  target.inputTokens += source.inputTokens
  target.cachedInputTokens += source.cachedInputTokens
  target.outputTokens += source.outputTokens
  target.reasoningTokens += source.reasoningTokens
}

export function totalUsageTokens(usage: TokenUsage): number {
  return usage.inputTokens + usage.outputTokens + usage.reasoningTokens
}

function usageFromRecord(record: Record<string, unknown> | undefined): TokenUsage {
  const value = (key: string): number => {
    const raw = record?.[key]
    return typeof raw === 'number' && Number.isFinite(raw) && raw > 0 ? raw : 0
  }
  return {
    inputTokens: value('input_tokens'),
    cachedInputTokens: value('cached_input_tokens'),
    outputTokens: value('output_tokens'),
    reasoningTokens: value('reasoning_output_tokens'),
  }
}

function usageDelta(current: TokenUsage, previous: TokenUsage): TokenUsage {
  return {
    inputTokens: Math.max(0, current.inputTokens - previous.inputTokens),
    cachedInputTokens: Math.max(0, current.cachedInputTokens - previous.cachedInputTokens),
    outputTokens: Math.max(0, current.outputTokens - previous.outputTokens),
    reasoningTokens: Math.max(0, current.reasoningTokens - previous.reasoningTokens),
  }
}

function scaleUsage(usage: TokenUsage, share: number): TokenUsage {
  return {
    inputTokens: usage.inputTokens * share,
    cachedInputTokens: usage.cachedInputTokens * share,
    outputTokens: usage.outputTokens * share,
    reasoningTokens: usage.reasoningTokens * share,
  }
}

function estimateValueTokens(value: unknown): number {
  if (typeof value === 'string') return estimateTokensFromChars(value.length)
  if (value == null) return 0
  try {
    return estimateTokensFromChars(JSON.stringify(value).length)
  } catch {
    return 0
  }
}

function normalizeToolName(raw: string, payloadType?: string): string {
  if (raw === 'exec_command' || raw === 'exec' || payloadType === 'local_shell_call') return 'Bash'
  if (raw === 'read_file') return 'Read'
  if (raw === 'write_file' || raw === 'apply_diff' || raw === 'apply_patch' || raw === 'patch_apply_end') return 'Edit'
  if (raw === 'read_dir') return 'Glob'
  if (payloadType === 'web_search_call' || raw === 'web_search') return 'WebSearch'
  return raw || payloadType || 'unknown'
}

function invocationId(payload: CodexEntry['payload'], fallback: string): string {
  const raw = payload?.call_id ?? payload?.id
  return typeof raw === 'string' && raw ? raw : fallback
}

function timestampMs(timestamp: string | undefined): number | undefined {
  if (!timestamp) return undefined
  const parsed = Date.parse(timestamp)
  return Number.isFinite(parsed) ? parsed : undefined
}

function stringifyForIdentifiers(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value ?? '')
  } catch {
    return ''
  }
}

function extractIdentifiers(value: unknown): string[] {
  const text = stringifyForIdentifiers(value)
  return [...new Set(text.match(UUID_RE)?.map((id) => id.toLowerCase()) ?? [])]
}

function failedResult(value: unknown): boolean {
  if (value == null) return false
  if (typeof value === 'string') return /^\s*(error|failed)\b/i.test(value.slice(0, 2000))
  if (typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  if (record['success'] === false || record['is_error'] === true) return true
  const exitCode = record['exit_code'] ?? record['exitCode']
  if (typeof exitCode === 'number' && exitCode !== 0) return true
  return typeof record['error'] === 'string' && record['error'].length > 0
}

function outputValue(payload: CodexEntry['payload']): unknown {
  return payload?.output ?? payload?.result ?? payload?.content ?? ''
}

function newInvocation(
  meta: SessionMeta,
  entry: CodexEntry,
  rawTool: string,
  argumentValue: unknown,
  sequence: number,
): PendingInvocation {
  const payloadType = entry.payload?.type
  const tool = normalizeToolName(rawTool, payloadType)
  return {
    sessionId: meta.sessionId,
    ...(meta.parentSessionId ? { parentSessionId: meta.parentSessionId } : {}),
    ...(meta.project ? { project: meta.project } : {}),
    callId: invocationId(entry.payload, `${meta.sessionId}:tool-${sequence}`),
    tool,
    rawTool: rawTool || payloadType || 'unknown',
    timestamp: entry.timestamp ?? '',
    argumentTokens: estimateValueTokens(argumentValue),
    resultTokens: 0,
    contextExposureTokens: 0,
    nextRequest: cloneUsage(),
    allocatedNextRequest: cloneUsage(),
    nextRequestBatchSize: 0,
    nextRequestAttribution: 'unattributed',
    failed: false,
    resultSizeEstimated: false,
    childSessionIds: [],
    subagentUsage: cloneUsage(),
    startedAtMs: timestampMs(entry.timestamp),
    resultIdentifiers: [],
  }
}

function toPublicInvocation(pending: PendingInvocation): CodexToolInvocation {
  const { startedAtMs: _startedAtMs, ...invocation } = pending
  return invocation
}

function rawStringField(head: string, field: string): string | undefined {
  const match = new RegExp(`"${field}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"`).exec(head)
  if (!match) return undefined
  try {
    return JSON.parse(`"${match[1]}"`) as string
  } catch {
    return match[1]
  }
}

function parseLargeLine(line: Buffer): { entry?: CodexEntry; estimatedOutputTokens?: number } {
  const head = line.subarray(0, Math.min(line.length, 128 * 1024)).toString('utf-8')
  const type = rawStringField(head, 'type')
  if (!type) return {}
  const payloadIndex = head.indexOf('"payload"')
  const payloadHead = payloadIndex >= 0 ? head.slice(payloadIndex) : head
  const payloadType = rawStringField(payloadHead, 'type')
  const entry: CodexEntry = {
    type,
    timestamp: rawStringField(head, 'timestamp'),
    payload: {
      type: payloadType,
      call_id: rawStringField(payloadHead, 'call_id'),
      id: rawStringField(payloadHead, 'id'),
      name: rawStringField(payloadHead, 'name'),
      session_id: rawStringField(payloadHead, 'session_id'),
      forked_from_id: rawStringField(payloadHead, 'forked_from_id'),
      cwd: rawStringField(payloadHead, 'cwd'),
    },
  }
  const isOutput = payloadType === 'function_call_output' || payloadType === 'custom_tool_call_output' || payloadType === 'local_shell_call_output'
  return { entry, ...(isOutput ? { estimatedOutputTokens: estimateTokensFromChars(line.length) } : {}) }
}

function parseLine(line: SessionLine, maxJsonLineBytes: number): { entry?: CodexEntry; estimatedOutputTokens?: number; parseError?: boolean } {
  if (Buffer.isBuffer(line) && line.length > maxJsonLineBytes) {
    const large = parseLargeLine(line)
    return { ...large, parseError: !large.entry }
  }
  const text = Buffer.isBuffer(line) ? line.toString('utf-8') : line
  if (!text || text.charCodeAt(0) !== 123) return {}
  try {
    return { entry: JSON.parse(text) as CodexEntry }
  } catch {
    return { parseError: true }
  }
}

function codexHome(override?: string): string {
  return override ?? process.env['CODEX_HOME'] ?? join(homedir(), '.codex')
}

async function listRollouts(root: string): Promise<RolloutRef[]> {
  const directories = [join(root, 'sessions'), join(root, 'archived_sessions')]
  const seen = new Set<string>()
  const refs: RolloutRef[] = []
  for (const directory of directories) {
    if (!existsSync(directory)) continue
    let entries: string[]
    try {
      entries = await readdir(directory, { recursive: true })
    } catch {
      continue
    }
    for (const relative of entries) {
      const name = basename(relative)
      const match = ROLLOUT_RE.exec(name)
      if (!match || seen.has(name)) continue
      const filePath = join(directory, relative)
      try {
        const info = await stat(filePath)
        if (!info.isFile() || info.size === 0) continue
        seen.add(name)
        refs.push({ filePath, sessionId: match[1]!, mtimeMs: info.mtimeMs })
      } catch {
        // A session can move into archived_sessions while discovery is running.
      }
    }
  }
  return refs.sort((a, b) => b.mtimeMs - a.mtimeMs)
}

async function readSessionMeta(ref: RolloutRef, maxJsonLineBytes: number): Promise<SessionMeta> {
  const meta: SessionMeta = { sessionId: ref.sessionId }
  for await (const line of readSessionLines(ref.filePath, undefined, { largeLineAsBuffer: true })) {
    const parsed = parseLine(line, maxJsonLineBytes)
    const entry = parsed.entry
    if (!entry) continue
    if (entry.type === 'session_meta') {
      const sessionId = entry.payload?.session_id
      if (typeof sessionId === 'string' && sessionId) meta.sessionId = sessionId
      const parent = entry.payload?.forked_from_id
      if (typeof parent === 'string' && parent) meta.parentSessionId = parent
      const cwd = entry.payload?.cwd
      if (typeof cwd === 'string' && cwd) meta.project = cwd
      meta.startedAt = entry.timestamp
    }
    break
  }
  return meta
}

function allocateNextRequest(batch: PendingInvocation[], usage: TokenUsage): void {
  if (batch.length === 0) return
  const totalWeight = batch.reduce((sum, invocation) => sum + Math.max(1, invocation.argumentTokens + invocation.resultTokens), 0)
  for (const invocation of batch) {
    invocation.nextRequest = cloneUsage(usage)
    invocation.nextRequestBatchSize = batch.length
    if (batch.length === 1) {
      invocation.allocatedNextRequest = cloneUsage(usage)
      invocation.nextRequestAttribution = 'exact-single-tool'
    } else {
      const weight = Math.max(1, invocation.argumentTokens + invocation.resultTokens)
      invocation.allocatedNextRequest = scaleUsage(usage, weight / totalWeight)
      invocation.nextRequestAttribution = 'estimated-proportional'
    }
  }
}

async function parseSession(ref: RolloutRef, meta: SessionMeta, maxJsonLineBytes: number): Promise<ParsedSession> {
  const openCalls = new Map<string, PendingInvocation>()
  const pendingBatch: PendingInvocation[] = []
  const liveContext = new Set<PendingInvocation>()
  const invocations: PendingInvocation[] = []
  const usage = cloneUsage()
  let cumulative = cloneUsage()
  let sawCumulative = false
  let previousCumulativeSignature = ''
  let sequence = 0
  let parseErrors = 0
  let oversizedLinesEstimated = 0

  const finishInvocation = (invocation: PendingInvocation, entry: CodexEntry, value: unknown, estimatedTokens?: number): void => {
    invocation.resultTokens = estimatedTokens ?? estimateValueTokens(value)
    invocation.resultSizeEstimated = estimatedTokens !== undefined
    invocation.failed = failedResult(value)
    invocation.resultIdentifiers = extractIdentifiers(value)
    const endedAt = timestampMs(entry.timestamp)
    if (invocation.startedAtMs !== undefined && endedAt !== undefined && endedAt >= invocation.startedAtMs) {
      invocation.durationMs = endedAt - invocation.startedAtMs
    }
    pendingBatch.push(invocation)
    liveContext.add(invocation)
    invocations.push(invocation)
  }

  for await (const line of readSessionLines(ref.filePath, undefined, { largeLineAsBuffer: true })) {
    const parsed = parseLine(line, maxJsonLineBytes)
    if (parsed.parseError) parseErrors++
    if (parsed.estimatedOutputTokens !== undefined) oversizedLinesEstimated++
    const entry = parsed.entry
    const payload = entry?.payload
    if (!entry || !payload) continue
    const payloadType = payload.type

    if (entry.type === 'compacted') {
      liveContext.clear()
      continue
    }

    if (entry.type === 'response_item' && (
      payloadType === 'function_call' ||
      payloadType === 'custom_tool_call' ||
      payloadType === 'local_shell_call' ||
      payloadType === 'web_search_call'
    )) {
      const rawTool = typeof payload.name === 'string' && payload.name
        ? payload.name
        : payloadType === 'local_shell_call'
          ? 'exec_command'
          : payloadType === 'web_search_call'
            ? 'web_search'
            : 'unknown'
      const argumentValue = payload.arguments ?? payload.input ?? payload.action ?? ''
      const invocation = newInvocation(meta, entry, rawTool, argumentValue, ++sequence)
      openCalls.set(invocation.callId, invocation)
      continue
    }

    if (entry.type === 'response_item' && (
      payloadType === 'function_call_output' ||
      payloadType === 'custom_tool_call_output' ||
      payloadType === 'local_shell_call_output'
    )) {
      const id = invocationId(payload, '')
      const invocation = openCalls.get(id) ?? newInvocation(meta, entry, 'unknown', '', ++sequence)
      openCalls.delete(id)
      finishInvocation(invocation, entry, outputValue(payload), parsed.estimatedOutputTokens)
      continue
    }

    if (entry.type === 'event_msg' && payloadType === 'patch_apply_end') {
      const invocation = newInvocation(meta, entry, 'patch_apply_end', '', ++sequence)
      finishInvocation(invocation, entry, payload, undefined)
      continue
    }

    if (entry.type === 'event_msg' && payloadType === 'mcp_tool_call_end') {
      const invocationRecord = typeof payload['invocation'] === 'object' && payload['invocation']
        ? payload['invocation'] as Record<string, unknown>
        : undefined
      const server = typeof invocationRecord?.['server'] === 'string' ? invocationRecord['server'] : 'unknown'
      const tool = typeof invocationRecord?.['tool'] === 'string' ? invocationRecord['tool'] : 'unknown'
      const rawTool = `mcp__${server}__${tool}`
      const invocation = newInvocation(meta, entry, rawTool, invocationRecord?.['arguments'] ?? '', ++sequence)
      const duration = payload['duration_ms']
      if (typeof duration === 'number' && Number.isFinite(duration) && duration >= 0) invocation.durationMs = duration
      finishInvocation(invocation, entry, payload['result'] ?? payload['output'] ?? payload, undefined)
      continue
    }

    if (entry.type === 'event_msg' && payloadType === 'token_count') {
      const info = payload.info
      if (!info) continue
      const last = usageFromRecord(info.last_token_usage as Record<string, unknown> | undefined)
      const total = usageFromRecord(info.total_token_usage as Record<string, unknown> | undefined)
      const cumulativeSignature = totalUsageTokens(total) > 0
        ? `${total.inputTokens}:${total.cachedInputTokens}:${total.outputTokens}:${total.reasoningTokens}`
        : ''
      if (cumulativeSignature && cumulativeSignature === previousCumulativeSignature) continue
      if (cumulativeSignature) previousCumulativeSignature = cumulativeSignature
      let eventUsage: TokenUsage
      if (totalUsageTokens(last) > 0) {
        eventUsage = last
      } else if (totalUsageTokens(total) > 0) {
        eventUsage = sawCumulative ? usageDelta(total, cumulative) : total
      } else {
        continue
      }
      if (totalUsageTokens(total) > 0) {
        cumulative = total
        sawCumulative = true
      }
      addUsage(usage, eventUsage)
      for (const invocation of liveContext) invocation.contextExposureTokens += invocation.resultTokens
      allocateNextRequest(pendingBatch, eventUsage)
      pendingBatch.length = 0
    }
  }

  for (const invocation of openCalls.values()) {
    invocation.failed = true
    invocations.push(invocation)
  }

  return {
    meta,
    invocations: invocations.map(toPublicInvocation),
    usage,
    parseErrors,
    oversizedLinesEstimated,
  }
}

function descendantsOf(rootIds: Set<string>, metas: Map<string, SessionMeta>): Set<string> {
  const selected = new Set(rootIds)
  let changed = true
  while (changed) {
    changed = false
    for (const meta of metas.values()) {
      if (meta.parentSessionId && selected.has(meta.parentSessionId) && !selected.has(meta.sessionId)) {
        selected.add(meta.sessionId)
        changed = true
      }
    }
  }
  return selected
}

function nearestSpawnInvocation(
  child: ParsedSession,
  parent: ParsedSession,
  assigned: Set<CodexToolInvocation>,
): CodexToolInvocation | undefined {
  const childStarted = timestampMs(child.meta.startedAt)
  const candidates = parent.invocations.filter((invocation) => invocation.rawTool === 'spawn_agent' && !assigned.has(invocation))
  if (candidates.length === 0) return undefined
  const exact = candidates.find((invocation) => invocation.childSessionIds.includes(child.meta.sessionId))
  if (exact) return exact
  const identifierMatch = candidates.find((invocation) =>
    invocation.resultIdentifiers?.includes(child.meta.sessionId.toLowerCase()),
  )
  if (identifierMatch) return identifierMatch
  if (childStarted === undefined) return candidates[0]
  return candidates
    .map((invocation) => ({ invocation, distance: childStarted - (timestampMs(invocation.timestamp) ?? childStarted) }))
    .filter(({ distance }) => distance >= 0)
    .sort((a, b) => a.distance - b.distance)[0]?.invocation ?? candidates[0]
}

function linkSubagents(sessions: ParsedSession[]): void {
  const byId = new Map(sessions.map((session) => [session.meta.sessionId, session]))
  const assigned = new Set<CodexToolInvocation>()
  for (const child of sessions) {
    if (!child.meta.parentSessionId) continue
    const parent = byId.get(child.meta.parentSessionId)
    if (!parent) continue
    let target = parent.invocations.find((invocation) =>
      invocation.rawTool === 'spawn_agent' && invocation.callId.toLowerCase().includes(child.meta.sessionId.toLowerCase()),
    )
    if (!target) {
      target = parent.invocations.find((invocation) =>
        invocation.rawTool === 'spawn_agent' && invocation.resultIdentifiers?.includes(child.meta.sessionId.toLowerCase()),
      )
    }
    target ??= nearestSpawnInvocation(child, parent, assigned)
    if (!target) {
      target = {
        sessionId: parent.meta.sessionId,
        ...(parent.meta.parentSessionId ? { parentSessionId: parent.meta.parentSessionId } : {}),
        ...(parent.meta.project ? { project: parent.meta.project } : {}),
        callId: `${parent.meta.sessionId}:spawn:${child.meta.sessionId}`,
        tool: 'spawn_agent',
        rawTool: 'spawn_agent',
        timestamp: child.meta.startedAt ?? '',
        argumentTokens: 0,
        resultTokens: 0,
        contextExposureTokens: 0,
        nextRequest: cloneUsage(),
        allocatedNextRequest: cloneUsage(),
        nextRequestBatchSize: 0,
        nextRequestAttribution: 'unattributed',
        failed: false,
        resultSizeEstimated: false,
        childSessionIds: [],
        subagentUsage: cloneUsage(),
      }
      parent.invocations.push(target)
    }
    assigned.add(target)
    target.childSessionIds.push(child.meta.sessionId)
    addUsage(target.subagentUsage, child.usage)
  }
}

function aggregateRows(invocations: CodexToolInvocation[]): CodexToolUsageRow[] {
  const rows = new Map<string, CodexToolUsageRow>()
  for (const invocation of invocations) {
    const row = rows.get(invocation.tool) ?? {
      tool: invocation.tool,
      calls: 0,
      failedCalls: 0,
      argumentTokens: 0,
      resultTokens: 0,
      contextExposureTokens: 0,
      nextRequest: cloneUsage(),
      subagentUsage: cloneUsage(),
      estimatedBatchCalls: 0,
      estimatedResultCalls: 0,
    }
    row.calls++
    if (invocation.failed) row.failedCalls++
    row.argumentTokens += invocation.argumentTokens
    row.resultTokens += invocation.resultTokens
    row.contextExposureTokens += invocation.contextExposureTokens
    addUsage(row.nextRequest, invocation.allocatedNextRequest)
    addUsage(row.subagentUsage, invocation.subagentUsage)
    if (invocation.nextRequestAttribution === 'estimated-proportional') row.estimatedBatchCalls++
    if (invocation.resultSizeEstimated) row.estimatedResultCalls++
    rows.set(invocation.tool, row)
  }
  return [...rows.values()]
}

export function metricValue(row: CodexToolUsageRow, metric: ToolUsageMetric): number {
  if (metric === 'context-exposure') return row.contextExposureTokens
  if (metric === 'next-request') return totalUsageTokens(row.nextRequest)
  if (metric === 'result-tokens') return row.resultTokens
  if (metric === 'subagent') return totalUsageTokens(row.subagentUsage)
  return row.resultTokens + row.contextExposureTokens + totalUsageTokens(row.nextRequest) + totalUsageTokens(row.subagentUsage)
}

export async function analyzeCodexToolUsage(options: {
  codexHome?: string
  session?: string
  includeDescendants?: boolean
  maxJsonLineBytes?: number
} = {}): Promise<CodexToolUsageReport> {
  const root = codexHome(options.codexHome)
  const refs = await listRollouts(root)
  const configuredCap = options.maxJsonLineBytes ?? Number(process.env['CODEBURN_TOOL_MAX_JSON_LINE_BYTES'] ?? DEFAULT_LARGE_JSON_CAP)
  const maxJsonLineBytes = Number.isFinite(configuredCap) && configuredCap > 0 ? configuredCap : DEFAULT_LARGE_JSON_CAP
  const metas = new Map<string, SessionMeta>()
  const refsById = new Map<string, RolloutRef>()
  for (const ref of refs) {
    const meta = await readSessionMeta(ref, maxJsonLineBytes)
    metas.set(meta.sessionId, meta)
    refsById.set(meta.sessionId, ref)
  }

  let rootSessionIds: string[] = []
  let selectedIds = new Set(metas.keys())
  if (options.session) {
    const matches = [...metas.keys()].filter((id) => id.startsWith(options.session!))
    if (matches.length === 0) throw new Error(`No Codex session matches ${options.session}`)
    if (matches.length > 1) throw new Error(`Session prefix ${options.session} is ambiguous (${matches.length} matches)`)
    rootSessionIds = matches
    selectedIds = new Set(matches)
    if (options.includeDescendants !== false) selectedIds = descendantsOf(selectedIds, metas)
  }

  const sessions: ParsedSession[] = []
  for (const id of selectedIds) {
    const ref = refsById.get(id)
    const meta = metas.get(id)
    if (ref && meta) sessions.push(await parseSession(ref, meta, maxJsonLineBytes))
  }
  linkSubagents(sessions)

  const invocations = sessions.flatMap((session) => session.invocations)
  const rows = aggregateRows(invocations)
  const publicInvocations = invocations.map((invocation) => {
    const { resultIdentifiers: _resultIdentifiers, ...publicInvocation } = invocation
    return publicInvocation
  })
  return {
    generatedAt: new Date().toISOString(),
    codexHome: root,
    sessionsAnalyzed: sessions.length,
    rootSessionIds,
    descendantsIncluded: options.includeDescendants !== false,
    parseErrors: sessions.reduce((sum, session) => sum + session.parseErrors, 0),
    oversizedLinesEstimated: sessions.reduce((sum, session) => sum + session.oversizedLinesEstimated, 0),
    rows,
    invocations: publicInvocations,
  }
}
