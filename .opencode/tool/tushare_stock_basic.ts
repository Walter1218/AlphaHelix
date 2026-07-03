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
  description: "获取 Tushare 股票基础信息列表（stock_basic 接口）",
  args: {
    exchange: tool.schema.string().optional().describe("交易所代码，如 SSE/SZSE，为空则返回全部"),
    list_status: tool.schema.string().optional().describe("上市状态 L/D/P，默认 L"),
    fields: tool.schema.string().optional().describe("返回字段，逗号分隔"),
  },
  async execute(args) {
    const output = await callTushare("stock_basic", stripEmpty({
      exchange: args.exchange,
      list_status: args.list_status || "L",
      fields: args.fields,
    }))
    return { output, title: "Tushare stock_basic" }
  },
})
