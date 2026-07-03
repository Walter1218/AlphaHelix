import { tool } from "@opencode-ai/plugin"

const TUSHARE_API = "https://api.tushare.pro"

async function callTushare(api_name: string, params: Record<string, unknown>) {
  const token = process.env.TUSHARE_TOKEN
  if (!token) {
    throw new Error("TUSHARE_TOKEN is not set")
  }

  const res = await fetch(TUSHARE_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_name, token, params }),
  })

  if (!res.ok) {
    throw new Error(`Tushare API error: ${res.status} ${res.statusText}`)
  }

  const data = await res.json()
  if (data && typeof data === "object" && "code" in data && data.code !== 0) {
    throw new Error(`Tushare API error: code=${data.code}, msg=${data.msg ?? ""}`)
  }

  return JSON.stringify(data, null, 2)
}

function stripEmpty(params: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== ""))
}

export default tool({
  description: "获取 Tushare 财务指标（fina_indicator）：ROE、毛利率、营收增速等",
  args: {
    ts_code: tool.schema.string().describe("股票代码"),
    ann_date: tool.schema.string().optional().describe("公告日期 YYYYMMDD（可选）"),
    start_date: tool.schema.string().optional().describe("开始日期 YYYYMMDD（可选）"),
    end_date: tool.schema.string().optional().describe("结束日期 YYYYMMDD（可选）"),
    period: tool.schema.string().optional().describe("报告期 YYYYMMDD（可选）"),
  },
  async execute(args) {
    const output = await callTushare("fina_indicator", stripEmpty({
      ts_code: args.ts_code,
      ann_date: args.ann_date,
      start_date: args.start_date,
      end_date: args.end_date,
      period: args.period,
    }))
    return { output, title: `Tushare fina_indicator ${args.ts_code}` }
  },
})
