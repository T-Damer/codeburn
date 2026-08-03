import { analyzeCodexToolUsage, metricValue, totalUsageTokens, type CodexToolInvocation, type CodexToolUsageRow, type ToolUsageMetric } from './codex-tool-usage.js'

const METRICS = new Set<ToolUsageMetric>(['context-exposure', 'next-request', 'result-tokens', 'subagent', 'total'])

type CliOptions = {
  provider: string
  metric: ToolUsageMetric
  session?: string
  details: boolean
  format: 'table' | 'json' | 'csv'
  top: number
  includeDescendants: boolean
}

function parseArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    provider: 'codex',
    metric: 'context-exposure',
    details: false,
    format: 'table',
    top: 30,
    includeDescendants: true,
  }
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!
    const next = (): string => {
      const value = args[++i]
      if (!value) throw new Error(`${arg} requires a value`)
      return value
    }
    if (arg === '--provider') options.provider = next()
    else if (arg === '--metric') {
      const metric = next() as ToolUsageMetric
      if (!METRICS.has(metric)) throw new Error(`Unknown metric ${metric}`)
      options.metric = metric
    } else if (arg === '--session') options.session = next()
    else if (arg === '--details') options.details = true
    else if (arg === '--format') {
      const format = next()
      if (format !== 'table' && format !== 'json' && format !== 'csv') throw new Error(`Unknown format ${format}`)
      options.format = format
    } else if (arg === '--top') {
      const top = Number(next())
      if (!Number.isInteger(top) || top < 1 || top > 10000) throw new Error('--top must be an integer from 1 to 10000')
      options.top = top
    } else if (arg === '--no-descendants') options.includeDescendants = false
    else if (arg === '--help' || arg === '-h') printHelp(0)
    else throw new Error(`Unknown option ${arg}`)
  }
  if (options.provider !== 'codex') throw new Error('Detailed tool-token attribution currently supports only --provider codex')
  return options
}

function printHelp(exitCode: number): never {
  console.log(`\nUsage: codeburn tools [options]\n\nDetailed local Codex tool-token attribution.\n\nOptions:\n  --provider codex                 provider (currently Codex only)\n  --metric <name>                  context-exposure | next-request | result-tokens | subagent | total\n  --session <id-prefix>            one session plus descendants\n  --no-descendants                 exclude child sessions\n  --details                        print individual tool calls\n  --format <table|json|csv>        output format\n  --top <n>                        maximum rows (default 30)\n  -h, --help                       show help\n\nNotes:\n  Result and context-exposure tokens are estimated from recorded text size.\n  Next-request usage is exact for one tool; batches are divided proportionally\n  by recorded argument + result size and marked with ~.\n`)
  process.exit(exitCode)
}

function formatNumber(value: number): string {
  const rounded = Math.round(value)
  if (rounded >= 1_000_000_000) return `${(rounded / 1_000_000_000).toFixed(1)}B`
  if (rounded >= 1_000_000) return `${(rounded / 1_000_000).toFixed(1)}M`
  if (rounded >= 1_000) return `${(rounded / 1_000).toFixed(1)}K`
  return String(rounded)
}

function pad(value: string, width: number, right = false): string {
  return right ? value.padStart(width) : value.padEnd(width)
}

function renderRows(rows: CodexToolUsageRow[], metric: ToolUsageMetric, top: number): void {
  const sorted = [...rows].sort((a, b) => metricValue(b, metric) - metricValue(a, metric)).slice(0, top)
  const columns = [
    { title: 'Tool', width: Math.min(42, Math.max(12, ...sorted.map((row) => row.tool.length))), value: (row: CodexToolUsageRow) => row.tool, right: false },
    { title: 'Calls', width: 7, value: (row: CodexToolUsageRow) => formatNumber(row.calls), right: true },
    { title: 'Args', width: 9, value: (row: CodexToolUsageRow) => formatNumber(row.argumentTokens), right: true },
    { title: 'Results', width: 9, value: (row: CodexToolUsageRow) => formatNumber(row.resultTokens), right: true },
    { title: 'Exposure', width: 10, value: (row: CodexToolUsageRow) => formatNumber(row.contextExposureTokens), right: true },
    { title: 'Next input', width: 11, value: (row: CodexToolUsageRow) => `${row.estimatedBatchCalls ? '~' : ''}${formatNumber(row.nextRequest.inputTokens)}`, right: true },
    { title: 'Next out', width: 9, value: (row: CodexToolUsageRow) => formatNumber(row.nextRequest.outputTokens + row.nextRequest.reasoningTokens), right: true },
    { title: 'Subagents', width: 10, value: (row: CodexToolUsageRow) => formatNumber(totalUsageTokens(row.subagentUsage)), right: true },
    { title: 'Failed', width: 7, value: (row: CodexToolUsageRow) => String(row.failedCalls), right: true },
  ]
  console.log(columns.map((column) => pad(column.title, column.width, column.right)).join('  '))
  console.log(columns.map((column) => '-'.repeat(column.width)).join('  '))
  for (const row of sorted) {
    console.log(columns.map((column) => pad(column.value(row).slice(0, column.width), column.width, column.right)).join('  '))
  }
}

function invocationMetric(invocation: CodexToolInvocation, metric: ToolUsageMetric): number {
  if (metric === 'context-exposure') return invocation.contextExposureTokens
  if (metric === 'next-request') return totalUsageTokens(invocation.allocatedNextRequest)
  if (metric === 'result-tokens') return invocation.resultTokens
  if (metric === 'subagent') return totalUsageTokens(invocation.subagentUsage)
  return invocation.resultTokens + invocation.contextExposureTokens + totalUsageTokens(invocation.allocatedNextRequest) + totalUsageTokens(invocation.subagentUsage)
}

function renderDetails(invocations: CodexToolInvocation[], metric: ToolUsageMetric, top: number): void {
  console.log('\nTool calls')
  const sorted = [...invocations].sort((a, b) => invocationMetric(b, metric) - invocationMetric(a, metric)).slice(0, top)
  for (const invocation of sorted) {
    const estimate = invocation.nextRequestAttribution === 'estimated-proportional' || invocation.resultSizeEstimated ? ' ~' : ''
    const children = invocation.childSessionIds.length ? ` children=${invocation.childSessionIds.map((id) => id.slice(0, 8)).join(',')}` : ''
    const duration = invocation.durationMs === undefined ? '' : ` ${Math.round(invocation.durationMs)}ms`
    console.log(
      `  ${invocation.sessionId.slice(0, 8)}  ${invocation.tool}  result=${formatNumber(invocation.resultTokens)}` +
      ` exposure=${formatNumber(invocation.contextExposureTokens)} next=${formatNumber(totalUsageTokens(invocation.allocatedNextRequest))}` +
      ` subagent=${formatNumber(totalUsageTokens(invocation.subagentUsage))}${duration}${children}${estimate}`,
    )
  }
}

function csvEscape(value: unknown): string {
  const text = String(value)
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

function renderCsv(rows: CodexToolUsageRow[], metric: ToolUsageMetric, top: number): void {
  const sorted = [...rows].sort((a, b) => metricValue(b, metric) - metricValue(a, metric)).slice(0, top)
  console.log('tool,calls,failed_calls,argument_tokens,result_tokens,context_exposure_tokens,next_input_tokens,next_cached_input_tokens,next_output_tokens,next_reasoning_tokens,subagent_tokens,estimated_batch_calls,estimated_result_calls')
  for (const row of sorted) {
    console.log([
      row.tool,
      row.calls,
      row.failedCalls,
      row.argumentTokens,
      row.resultTokens,
      row.contextExposureTokens,
      row.nextRequest.inputTokens,
      row.nextRequest.cachedInputTokens,
      row.nextRequest.outputTokens,
      row.nextRequest.reasoningTokens,
      totalUsageTokens(row.subagentUsage),
      row.estimatedBatchCalls,
      row.estimatedResultCalls,
    ].map(csvEscape).join(','))
  }
}

async function main(): Promise<void> {
  let options: CliOptions
  try {
    options = parseArgs(process.argv.slice(3))
  } catch (error) {
    console.error(`codeburn tools: ${(error as Error).message}`)
    console.error('Run codeburn tools --help for usage.')
    process.exitCode = 1
    return
  }

  const report = await analyzeCodexToolUsage({
    session: options.session,
    includeDescendants: options.includeDescendants,
  })
  if (options.format === 'json') {
    console.log(JSON.stringify({ ...report, metric: options.metric }, null, 2))
    return
  }
  if (options.format === 'csv') {
    renderCsv(report.rows, options.metric, options.top)
    return
  }

  console.log(`\nCodeBurn Codex tool attribution  metric=${options.metric}`)
  console.log(`Sessions ${report.sessionsAnalyzed}  calls ${report.invocations.length}  parse errors ${report.parseErrors}`)
  if (report.oversizedLinesEstimated > 0) console.log(`Oversized result lines estimated: ${report.oversizedLinesEstimated}`)
  console.log('~ marks proportional batch attribution; tool result/exposure sizes are token estimates.\n')
  renderRows(report.rows, options.metric, options.top)
  if (options.details) renderDetails(report.invocations, options.metric, options.top)
  console.log('')
}

void main().catch((error) => {
  console.error(`codeburn tools: ${(error as Error).message}`)
  process.exitCode = 1
})
