import { tool } from "@opencode-ai/plugin"
import fs from "node:fs"
import path from "node:path"

export default tool({
  description: "将 AlphaHelix agent 的推理或决策节点追加写入 trace 文件 memory/trace/YYYYMMDD.jsonl，用于 case 分析与 DPO 数据集构建。",
  args: {
    step: tool.schema.string().describe("trace 节点类型，例如 reasoning.screen_request, reasoning.qualitative, reasoning.final_decision, reasoning.risk_check"),
    date: tool.schema.string().describe("交易日期 YYYYMMDD，用于决定 trace 文件名"),
    strategy: tool.schema.string().optional().describe("当前策略名，如 regime / event_driven / momentum_value_hybrid"),
    payload: tool.schema.object().describe("任意可 JSON 序列化的对象，通常包含 { reasoning: string, inputs?: object, outputs?: object, metadata?: object }"),
  },
  async execute(args, ctx) {
    const traceDir = path.join(ctx.directory, "memory", "trace")
    if (!fs.existsSync(traceDir)) {
      fs.mkdirSync(traceDir, { recursive: true })
    }

    const traceFile = path.join(traceDir, `${args.date}.jsonl`)
    const event = {
      timestamp: new Date().toISOString(),
      run_id: "agent",
      step: args.step,
      date: args.date,
      strategy: args.strategy || null,
      payload: args.payload,
    }

    const line = JSON.stringify(event, ensureAsciiReplacer) + "\n"
    fs.appendFileSync(traceFile, line, { encoding: "utf-8" })

    return {
      output: `Appended trace to ${traceFile}`,
      title: `append_trace ${args.step}`,
    }
  },
})

function ensureAsciiReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "string") {
    return value
  }
  return value
}
