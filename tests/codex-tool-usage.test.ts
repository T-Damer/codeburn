import { mkdtemp, mkdir, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { describe, expect, it } from 'vitest'

import { analyzeCodexToolUsage, totalUsageTokens } from '../src/codex-tool-usage.js'

const PARENT_ID = '11111111-1111-4111-8111-111111111111'
const CHILD_ID = '22222222-2222-4222-8222-222222222222'

async function fixtureHome(): Promise<string> {
  const home = await mkdtemp(join(tmpdir(), 'codeburn-tools-'))
  await mkdir(join(home, 'sessions', '2026', '08', '03'), { recursive: true })
  return home
}

async function writeRollout(home: string, sessionId: string, lines: unknown[], suffix = '00'): Promise<void> {
  const file = join(home, 'sessions', '2026', '08', '03', `rollout-2026-08-03T19-00-${suffix}-${sessionId}.jsonl`)
  await writeFile(file, lines.map((line) => JSON.stringify(line)).join('\n') + '\n')
}

function meta(sessionId: string, parentSessionId?: string, subagent = Boolean(parentSessionId)): unknown {
  return {
    type: 'session_meta',
    timestamp: '2026-08-03T19:00:00.000Z',
    payload: {
      session_id: sessionId,
      ...(parentSessionId ? { forked_from_id: parentSessionId } : {}),
      ...(subagent && parentSessionId ? { parent_thread_id: parentSessionId, thread_source: 'subagent', agent_path: '/root/exploration' } : {}),
      cwd: '/tmp/project',
      originator: 'codex_cli_rs',
    },
  }
}

function tokenCount(timestamp: string, input: number, cached: number, output: number, reasoning: number, cumulative: number): unknown {
  return {
    type: 'event_msg',
    timestamp,
    payload: {
      type: 'token_count',
      info: {
        last_token_usage: {
          input_tokens: input,
          cached_input_tokens: cached,
          output_tokens: output,
          reasoning_output_tokens: reasoning,
          total_tokens: input + output + reasoning,
        },
        total_token_usage: {
          input_tokens: cumulative,
          cached_input_tokens: cached,
          output_tokens: output,
          reasoning_output_tokens: reasoning,
          total_tokens: cumulative + output + reasoning,
        },
      },
    },
  }
}

describe('Codex tool attribution', () => {
  it('attributes a single tool to the next request and tracks repeated context exposure', async () => {
    const home = await fixtureHome()
    await writeRollout(home, PARENT_ID, [
      meta(PARENT_ID),
      {
        type: 'response_item',
        timestamp: '2026-08-03T19:00:01.000Z',
        payload: { type: 'function_call', name: 'exec_command', call_id: 'call-1', arguments: JSON.stringify({ command: 'printf 1234567890' }) },
      },
      {
        type: 'response_item',
        timestamp: '2026-08-03T19:00:02.000Z',
        payload: { type: 'function_call_output', call_id: 'call-1', output: 'x'.repeat(80) },
      },
      tokenCount('2026-08-03T19:00:03.000Z', 100, 60, 20, 5, 100),
      tokenCount('2026-08-03T19:00:04.000Z', 120, 80, 10, 2, 220),
    ])

    const report = await analyzeCodexToolUsage({ codexHome: home })
    const bash = report.rows.find((row) => row.tool === 'Bash')
    expect(bash).toBeDefined()
    expect(bash!.calls).toBe(1)
    expect(bash!.resultTokens).toBe(20)
    expect(bash!.contextExposureTokens).toBe(40)
    expect(bash!.nextRequest.inputTokens).toBe(100)
    expect(bash!.nextRequest.cachedInputTokens).toBe(60)
    expect(bash!.nextRequest.outputTokens).toBe(20)
    expect(bash!.nextRequest.reasoningTokens).toBe(5)
    expect(bash!.estimatedBatchCalls).toBe(0)
  })

  it('splits a multi-tool next request proportionally and marks it estimated', async () => {
    const home = await fixtureHome()
    await writeRollout(home, PARENT_ID, [
      meta(PARENT_ID),
      { type: 'response_item', timestamp: '2026-08-03T19:00:01.000Z', payload: { type: 'function_call', name: 'read_file', call_id: 'read', arguments: '{}' } },
      { type: 'response_item', timestamp: '2026-08-03T19:00:02.000Z', payload: { type: 'function_call_output', call_id: 'read', output: 'a'.repeat(40) } },
      { type: 'response_item', timestamp: '2026-08-03T19:00:03.000Z', payload: { type: 'function_call', name: 'exec_command', call_id: 'bash', arguments: '{}' } },
      { type: 'response_item', timestamp: '2026-08-03T19:00:04.000Z', payload: { type: 'function_call_output', call_id: 'bash', output: 'b'.repeat(120) } },
      tokenCount('2026-08-03T19:00:05.000Z', 1000, 800, 100, 0, 1000),
    ])

    const report = await analyzeCodexToolUsage({ codexHome: home })
    const read = report.rows.find((row) => row.tool === 'Read')!
    const bash = report.rows.find((row) => row.tool === 'Bash')!
    expect(read.estimatedBatchCalls).toBe(1)
    expect(bash.estimatedBatchCalls).toBe(1)
    expect(Math.round(read.nextRequest.inputTokens + bash.nextRequest.inputTokens)).toBe(1000)
    expect(bash.nextRequest.inputTokens).toBeGreaterThan(read.nextRequest.inputTokens)
  })

  it('stops context exposure at compaction', async () => {
    const home = await fixtureHome()
    await writeRollout(home, PARENT_ID, [
      meta(PARENT_ID),
      { type: 'response_item', timestamp: '2026-08-03T19:00:01.000Z', payload: { type: 'function_call', name: 'read_file', call_id: 'read', arguments: '{}' } },
      { type: 'response_item', timestamp: '2026-08-03T19:00:02.000Z', payload: { type: 'function_call_output', call_id: 'read', output: 'a'.repeat(40) } },
      tokenCount('2026-08-03T19:00:03.000Z', 100, 50, 10, 0, 100),
      { type: 'compacted', timestamp: '2026-08-03T19:00:04.000Z', payload: { type: 'compaction' } },
      tokenCount('2026-08-03T19:00:05.000Z', 80, 60, 5, 0, 180),
    ])

    const report = await analyzeCodexToolUsage({ codexHome: home })
    expect(report.rows.find((row) => row.tool === 'Read')!.contextExposureTokens).toBe(10)
  })

  it('counts completed provider-native tool calls without a separate output item', async () => {
    const home = await fixtureHome()
    await writeRollout(home, PARENT_ID, [
      meta(PARENT_ID),
      {
        type: 'response_item',
        timestamp: '2026-08-03T19:00:01.000Z',
        payload: { type: 'web_search_call', id: 'ws-1', status: 'completed', action: { query: 'Codex rollout format' } },
      },
      tokenCount('2026-08-03T19:00:02.000Z', 200, 100, 30, 5, 200),
    ])

    const report = await analyzeCodexToolUsage({ codexHome: home })
    const search = report.rows.find((row) => row.tool === 'WebSearch')
    expect(search?.calls).toBe(1)
    expect(search?.failedCalls).toBe(0)
    expect(search?.nextRequest.inputTokens).toBe(200)
  })

  it('does not treat an ordinary fork as a spawned subagent', async () => {
    const home = await fixtureHome()
    await writeRollout(home, PARENT_ID, [
      meta(PARENT_ID),
      tokenCount('2026-08-03T19:00:03.000Z', 100, 50, 10, 0, 100),
    ], '00')
    await writeRollout(home, CHILD_ID, [
      meta(CHILD_ID, PARENT_ID, false),
      tokenCount('2026-08-03T19:00:05.000Z', 400, 300, 80, 20, 400),
    ], '01')

    const report = await analyzeCodexToolUsage({ codexHome: home, session: PARENT_ID.slice(0, 8) })
    expect(report.sessionsAnalyzed).toBe(1)
    expect(report.rows.find((row) => row.tool === 'spawn_agent')).toBeUndefined()
  })

  it('links a child rollout to spawn_agent and includes its direct token usage', async () => {
    const home = await fixtureHome()
    await writeRollout(home, PARENT_ID, [
      meta(PARENT_ID),
      { type: 'response_item', timestamp: '2026-08-03T19:00:01.000Z', payload: { type: 'function_call', name: 'spawn_agent', call_id: 'spawn', arguments: JSON.stringify({ task: 'inspect parser' }) } },
      { type: 'response_item', timestamp: '2026-08-03T19:00:02.000Z', payload: { type: 'function_call_output', call_id: 'spawn', output: JSON.stringify({ agent_id: CHILD_ID }) } },
      tokenCount('2026-08-03T19:00:03.000Z', 100, 50, 10, 0, 100),
    ], '00')
    await writeRollout(home, CHILD_ID, [
      meta(CHILD_ID, PARENT_ID),
      // Replayed parent usage must not be charged to the child.
      tokenCount('2026-08-03T19:00:01.000Z', 100, 50, 10, 0, 100),
      tokenCount('2026-08-03T19:00:05.000Z', 400, 300, 80, 20, 500),
    ], '01')

    const report = await analyzeCodexToolUsage({ codexHome: home, session: PARENT_ID.slice(0, 8) })
    const spawn = report.rows.find((row) => row.tool === 'spawn_agent')!
    expect(report.sessionsAnalyzed).toBe(2)
    expect(totalUsageTokens(spawn.subagentUsage)).toBe(500)
    const call = report.invocations.find((invocation) => invocation.rawTool === 'spawn_agent')!
    expect(call.childSessionIds).toEqual([CHILD_ID])
  })
})
