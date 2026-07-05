import { tool } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import path from "node:path"

export default tool({
  description: "对 A 股全市场进行量化因子筛选，返回候选股票池。内部调用 scripts/screen.py。支持 momentum_value_hybrid / quality_growth / contrarian / event_driven / regime，以及 GBDT 模型打分模式。",
  args: {
    strategy: tool.schema.string().optional().describe("策略名称：momentum_value_hybrid | quality_growth | contrarian | event_driven | regime（默认，按市场状态自动切换）"),
    trade_date: tool.schema.string().optional().describe("筛选基准日 YYYYMMDD，默认最近交易日"),
    top_k: tool.schema.number().optional().describe("返回前 K 只，默认 50"),
    use_gbdt: tool.schema.boolean().optional().describe("是否使用 GBDT 模型打分（默认 false）"),
    gbdt_model_path: tool.schema.string().optional().describe("GBDT 模型文件路径，默认自动查找 memory/models/ 下最新模型"),
    gbdt_threshold: tool.schema.number().optional().describe("GBDT 得分阈值，低于该值的股票被过滤"),
    gbdt_max_positions: tool.schema.number().optional().describe("GBDT 模式最大持仓数量"),
  },
  async execute(args, ctx) {
    const strategy = args.strategy || "regime"
    const top_k = args.top_k ?? 50
    const trade_date = args.trade_date || ""
    const use_gbdt = args.use_gbdt ?? false
    const gbdt_model_path = args.gbdt_model_path || ""
    const gbdt_threshold = args.gbdt_threshold
    const gbdt_max_positions = args.gbdt_max_positions

    const scriptPath = path.join(ctx.directory, "scripts", "screen.py")
    const python = process.platform === "win32" ? "python" : "python3"

    const output = await new Promise<string>((resolve, reject) => {
      const pyArgs = [scriptPath, strategy]
      if (trade_date) pyArgs.push(trade_date)
      pyArgs.push(String(top_k))
      if (use_gbdt) pyArgs.push("--use-gbdt")
      if (gbdt_model_path) {
        pyArgs.push("--gbdt-model-path")
        pyArgs.push(gbdt_model_path)
      }
      if (gbdt_threshold !== undefined) {
        pyArgs.push("--gbdt-threshold")
        pyArgs.push(String(gbdt_threshold))
      }
      if (gbdt_max_positions !== undefined) {
        pyArgs.push("--gbdt-max-positions")
        pyArgs.push(String(gbdt_max_positions))
      }
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
