# Detailed Codex tool-token attribution

`codeburn tools` reads Codex rollout files directly from `CODEX_HOME` (default:
`~/.codex`) and reports how individual shell, file, MCP, web-search, and agent
calls affect token usage.

```bash
codeburn tools
codeburn tools --metric next-request
codeburn tools --metric result-tokens --details
codeburn tools --session 019c1234 --details
codeburn tools --format json > codex-tools.json
codeburn tools --format csv > codex-tools.csv
```

## Metrics

- **Result tokens**: estimated tokens in tool output recorded by Codex.
- **Context exposure**: result tokens multiplied by the number of subsequent
  model requests that retained the result before compaction. This is the most
  useful signal for oversized shell output and repeated context pollution.
- **Next request**: exact `last_token_usage` reported by Codex for the first
  generation after a completed tool call. When several tools complete before
  one generation, CodeBurn divides the usage proportionally by each call's
  recorded argument and result size and marks the attribution as estimated.
- **Subagent**: direct model usage of child rollouts linked through
  `forked_from_id` and `spawn_agent` results. Nested agents appear on their own
  `spawn_agent` rows instead of being counted repeatedly in every ancestor.
- **Total**: a diagnostic ranking that combines the preceding signals. It is not
  a billing total because context exposure intentionally counts repeated
  appearances of the same result.

Codex does not emit an authoritative `tokens_consumed` field on a tool call.
Model token counts are exact; result size, context exposure, and multi-tool
allocation are estimates and are labelled accordingly. No prompt, command
output, or file content is retained by the report.

## Oversized rollout lines

Lines up to 64 MiB are parsed normally. Larger tool-result lines are counted
from their encoded byte size without retaining their content, and the report's
`oversizedLinesEstimated` counter is incremented. Override the parsing limit:

```bash
CODEBURN_TOOL_MAX_JSON_LINE_BYTES=$((128 * 1024 * 1024)) codeburn tools
```
