import { tool } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import path from "node:path"

export default tool({
  description: "对 A 股全市场进行量化因子筛选，返回候选股票池。内部调用 scripts/screen.py。支持 momentum_value_hybrid / quality_growth / contrarian / event_driven / regime。",
  args: {
    strategy: tool.schema.string().optional().describe("策略名称：momentum_value_hybrid | quality_growth | contrarian | event_driven | regime（默认，按市场状态自动切换）"),
    trade_date: tool.schema.string().optional().describe("筛选基准日 YYYYMMDD，默认最近交易日"),
    top_k: tool.schema.number().optional().describe("返回前 K 只，默认 50"),
  },
  async execute(args, ctx) {
    const strategy = args.strategy || "regime"
    const top_k = args.top_k ?? 50
    const trade_date = args.trade_date || ""

    const scriptPath = path.join(ctx.directory, "scripts", "screen.py")
    const python = process.platform === "win32" ? "python" : "python3"

    const output = await new Promise<string>((resolve, reject) => {
      const pyArgs = [scriptPath, strategy]
      if (trade_date) pyArgs.push(trade_date)
      pyArgs.push(String(top_k))
      const proc = spawn(python, pyArgs, {
        cwd: ctx.directory,
        env: { ...process.env, TUSHARE_TOKEN: process.env.TUSHARE_TOKEN },
      })

      let stdout = ""
      let stderr = ""
      proc.stdout.on("data", (data) => { stdout += data.toString() })
      proc.stderr.on("data", (data) => { stderr += data.toString() })
      proc.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`screen.py exited with ${code}: ${stderr || stdout}`))
        } else {
          resolve(stdout)
        }
      })
      proc.on("error", reject)
    })

    return { output, title: `screen_candidates ${strategy}` }
  },
})
