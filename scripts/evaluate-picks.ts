// AlphaHelix 选股评估入口
// 调用本地 Python 脚本计算实际收益指标，并将结果追加到 memory/stock/YYYYMMDD.md。

import { existsSync, readFileSync, appendFileSync } from "fs"
import { join } from "path"
import { execSync } from "child_process"

const PICK_DATE = process.argv[2]
const HORIZON = process.argv[3] ?? "20"

if (!PICK_DATE) {
  console.error("Usage: bun run scripts/evaluate-picks.ts <YYYYMMDD> [horizon_days]")
  process.exit(1)
}

async function main() {
  const jsonPath = join(process.cwd(), "memory/stock", `${PICK_DATE}.json`)
  const mdPath = join(process.cwd(), "memory/stock", `${PICK_DATE}.md`)

  if (!existsSync(jsonPath)) {
    console.error(`[AlphaHelix] Pick snapshot not found: ${jsonPath}`)
    process.exit(1)
  }

  const result = execSync(
    `python scripts/evaluate.py ${PICK_DATE} ${HORIZON}`,
    { cwd: process.cwd(), encoding: "utf-8", timeout: 300000, env: { ...process.env } }
  )

  const evaluation = JSON.parse(result)
  console.log(`[AlphaHelix] Evaluation result:`, evaluation)

  if (existsSync(mdPath)) {
    appendFileSync(mdPath, `\n\n## Evaluation (${new Date().toISOString().slice(0, 10)})\n\n\`\`\`json\n${JSON.stringify(evaluation, null, 2)}\n\`\`\`\n`)
  }
}

main().catch((err) => {
  console.error("[AlphaHelix] Failed:", err)
  process.exit(1)
})
