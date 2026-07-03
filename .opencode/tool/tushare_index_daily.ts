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
  description: "获取 Tushare 指数日线行情（index_daily），如沪深300、上证指数",
  args: {
    ts_code: tool.schema.string().describe("指数代码，如 000300.SH / 000001.SH"),
    start_date: tool.schema.string().optional().describe("开始日期 YYYYMMDD（可选）"),
    end_date: tool.schema.string().optional().describe("结束日期 YYYYMMDD（可选）"),
  },
  async execute(args) {
    const output = await callTushare("index_daily", stripEmpty({
      ts_code: args.ts_code,
      start_date: args.start_date,
      end_date: args.end_date,
    }))
    return { output, title: `Tushare index_daily ${args.ts_code}` }
  },
})
