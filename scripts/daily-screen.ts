// AlphaHelix 每日选股调度入口
// 通过 HelixAgent CLI headless 模式（--format json）执行选股，避免 server 模式工具执行异常。
// 配置来源（优先级从高到低）：
// 1. 环境变量 ALPHAHELIX_HELIX_AGENT_PATH / MIMOCODE_HOME / TUSHARE_TOKEN
// 2. .env 文件（dotenv 自动加载）

import { config } from "dotenv"
import { spawn } from "node:child_process"
import path from "node:path"
import fs from "node:fs"

config({ path: new URL("../.env", import.meta.url) })

function getHelixAgentPath(): string {
  const raw = process.env.ALPHAHELIX_HELIX_AGENT_PATH?.trim()
  if (raw) return path.resolve(raw)
  // 默认与 AlphaHelix 同级的 HelixAgent 目录
  return path.resolve(path.join(process.cwd(), "..", "HelixAgent"))
}

function getMimoHome(helixPath: string): string {
  const raw = process.env.MIMOCODE_HOME?.trim()
  if (raw) return path.resolve(raw)
  return path.join(helixPath, ".mimo")
}

function getOpencodeBin(helixPath: string): string {
  return path.join(helixPath, "packages", "opencode", "src", "index.ts")
}

async function runStockPicking(): Promise<void> {
  const today = new Date().toISOString().slice(0, 10).replace(/-/g, "")
  const cwd = process.cwd()
  const mdPath = path.join(cwd, "memory", "stock", `${today}.md`)
  const jsonPath = path.join(cwd, "memory", "stock", `${today}.json`)
  const logDir = path.join(cwd, "memory", "log")
  fs.mkdirSync(logDir, { recursive: true })
  const logFile = path.join(logDir, `daily-screen-${today}-${Date.now()}.log`)
  const logFd = fs.openSync(logFile, "a")

  const helixPath = getHelixAgentPath()
  const mimoHome = getMimoHome(helixPath)
  const bin = getOpencodeBin(helixPath)

  if (!fs.existsSync(helixPath)) {
    throw new Error(`HelixAgent not found at ${helixPath}. Set ALPHAHELIX_HELIX_AGENT_PATH or place HelixAgent next to AlphaHelix.`)
  }
  if (!fs.existsSync(bin)) {
    throw new Error(`HelixAgent opencode entry not found at ${bin}`)
  }

  const prompt = `执行 AlphaHelix 每日选股：
1. 先读取 ${cwd}/memory/prompt_adaptations/latest.md 获取最新反馈层提示。
2. 调用 screen_candidates(strategy=regime, trade_date=${today}, top_n=10) 获取候选股。regime 策略会自动按市场状态选择 momentum_value_hybrid / quality_growth / contrarian，并使用 memory/weights/ 中的最新优化权重。
3. 基于候选股数据生成 Markdown 报告和 JSON 快照。
4. 用 write 工具将结果写入以下两个绝对路径：
   - ${jsonPath}
   - ${mdPath}
5. 报告包含：日期、市场概览、当前 regime（若有）、Top 10 股票（代码、名称、行业、得分、排名、逻辑、止损价、置信度）、行业分布、风险提示与责任声明。
6. 候选股已包含动量、估值、质量、资金等多维度因子，请在 rationale 中简要引用关键因子。
7. 检查 Top 10 的行业分布：若单一行业过于集中（如超过 40%），请在风险说明中提示并建议分散。
8. 如 memory 工具调用失败（Unexpected server error），直接跳过，不要阻塞主流程。`

  const args = [
    bin,
    "run",
    "--agent", "alpha-analyst",
    "--format", "json",
    "--title", `AlphaHelix ${today}`,
    prompt,
  ]

  console.log(`[AlphaHelix] Starting HelixAgent CLI in ${cwd}`)
  console.log(`[AlphaHelix] MIMOCODE_HOME=${mimoHome}`)
  console.log(`[AlphaHelix] Output: ${mdPath}, ${jsonPath}`)
  console.log(`[AlphaHelix] Log: ${logFile}`)

  // HelixAgent CLI 在 stdout 为 pipe 时偶发 Unexpected server error；重定向到文件可稳定运行。
  fs.writeSync(logFd, `[${new Date().toISOString()}] spawn: bun ${args.join(" ")}\n`)
  fs.writeSync(logFd, `cwd: ${cwd}\nMIMOCODE_HOME: ${mimoHome}\n---\n`)

  return new Promise((resolve, reject) => {
    const proc = spawn("bun", args, {
      cwd,
      env: {
        ...process.env,
        MIMOCODE_HOME: mimoHome,
      },
      stdio: ["ignore", logFd, logFd],
    })

    proc.on("close", (code) => {
      fs.writeSync(logFd, `\n---\n[${new Date().toISOString()}] exit code: ${code}\n`)
      fs.closeSync(logFd)

      const log = fs.existsSync(logFile) ? fs.readFileSync(logFile, "utf-8") : ""
      const lastEventMatch = log.match(/"type":"([^"]+)"/g)
      const lastEvent = lastEventMatch ? lastEventMatch[lastEventMatch.length - 1]!.replace(/"/g, "") : ""

      if (code !== 0) {
        console.error("[AlphaHelix] HelixAgent log tail:\n", log.slice(-4000))
        reject(new Error(`HelixAgent CLI exited with ${code}`))
        return
      }
      if (!fs.existsSync(mdPath) || !fs.existsSync(jsonPath)) {
        console.error("[AlphaHelix] HelixAgent log tail:\n", log.slice(-4000))
        reject(new Error(`Output files missing. log length=${log.length}`))
        return
      }
      console.log(`[AlphaHelix] Completed. Last event: ${lastEvent || "(none)"}`)
      console.log(`[AlphaHelix] Report: ${mdPath}`)
      console.log(`[AlphaHelix] Snapshot: ${jsonPath}`)
      console.log(`[AlphaHelix] Log: ${logFile}`)
      resolve()
    })

    proc.on("error", (err) => {
      fs.writeSync(logFd, `\n---\n[${new Date().toISOString()}] spawn error: ${err}\n`)
      fs.closeSync(logFd)
      reject(err)
    })
  })
}

runStockPicking().catch((err) => {
  console.error("[AlphaHelix] Failed:", err.message || err)
  process.exit(1)
})
