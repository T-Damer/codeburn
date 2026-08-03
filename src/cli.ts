#!/usr/bin/env node
// This launcher must stay parseable by Node 18. Do NOT add static imports.
const [major, minor] = process.versions.node.split('.').map(Number)
if (major < 22 || (major === 22 && minor < 13)) {
  process.stderr.write(
    `codeburn requires Node.js >= 22.13.0 (current: ${process.version})\n` +
    'Upgrade at https://nodejs.org/\n',
  )
  process.exit(1)
}

// Keep detailed Codex attribution isolated from the main dashboard bundle. This
// also lets `codeburn tools --help` start without scanning every provider.
const entry = process.argv[2] === 'tools' ? './codex-tools-cli.js' : './main.js'
import(entry).catch((err) => {
  process.stderr.write(String(err?.message ?? err) + '\n')
  process.exit(1)
})
