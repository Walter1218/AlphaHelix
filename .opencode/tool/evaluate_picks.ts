import { tool } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import path from "node:path"

export default tool({
  description: "评估已生成的选股快照在未来 N 个交易日后的表现。内部调用 scripts/evaluate.py。",
  args: {
    date: tool.schema.string().describe("选股日期 YYYYMMDD，如 20260702"),
    horizon: tool.schema.number().optional().describe("预测 horizon 交易日数，默认 20"),
  },
  async execute(args, ctx) {
    const horizon = args.horizon ?? 20
    const scriptPath = path.join(ctx.directory, "scripts", "evaluate.py")
    const python = process.platform === "win32" ? "python" : "python3"

    const output = await new Promise<string>((resolve, reject) => {
      const proc = spawn(python, [scriptPath, args.date, String(horizon)], {
        cwd: ctx.directory,
        env: { ...process.env, TUSHARE_TOKEN: process.env.TUSHARE_TOKEN },
      })

      let stdout = ""
      let stderr = ""
      proc.stdout.on("data", (data) => { stdout += data.toString() })
      proc.stderr.on("data", (data) => { stderr += data.toString() })
      proc.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`evaluate.py exited with ${code}: ${stderr || stdout}`))
        } else {
          resolve(stdout)
        }
      })
      proc.on("error", reject)
    })

    return { output, title: `evaluate_picks ${args.date}` }
  },
})
