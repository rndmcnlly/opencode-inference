import { expect, test } from "bun:test"
import { mkdtemp, rm, writeFile } from "fs/promises"
import { tmpdir } from "os"
import { join } from "path"
import plugin, { generateResult as interrogateResult, resolveModel, type RegistryModel } from "../index.ts"

const entries: RegistryModel[] = [
  { id: "glm-5.3-flash", modelID: "glm-5.3-flash", providerID: "opencode-go", package: "@opencode/ai/providers/openai-compatible", enabled: true,
    capabilities: { input: ["text", "image"] }, variants: [{ id: "low", settings: { reasoningEffort: "low" } }],
    settings: { baseURL: "https://opencode.ai/inference/go/openai/v1", apiKey: "go-key" } },
  { id: "deepseek-v4-1-flash", modelID: "deepseek-v4-1-flash", providerID: "tinfoil", package: "opencode-tinfoil/provider", enabled: true,
    capabilities: { input: ["text", "image"] }, variants: [{ id: "none", settings: { reasoningEffort: "none" } }, { id: "high", settings: { reasoningEffort: "high" } }],
    settings: { baseURL: "https://inference.tinfoil.sh/v1/", apiKey: "tk", tinfoil: { transport: "ehbp" } } },
  { id: "kimi-k3", modelID: "kimi-k3", providerID: "tinfoil", package: "opencode-tinfoil/provider", enabled: true,
    capabilities: { input: ["text", "image"] }, variants: [{ id: "low" }, { id: "high" }], settings: { baseURL: "https://inference.tinfoil.sh/v1/" } },
  { id: "claude", modelID: "claude", providerID: "anthropicish", package: "@opencode/ai/providers/anthropic", enabled: true,
    capabilities: { input: ["text", "image"] }, settings: {} },
  { id: "LFM", modelID: "LFM", providerID: "omlx", package: "@opencode/ai/providers/openai-compatible", enabled: true,
    capabilities: { input: ["text"] }, settings: { baseURL: "http://localhost:12348/v1" } },
]
const registry = { list: async () => entries, defaultModel: async () => "opencode-go/glm-5.3-flash" }
const generateText = async () => " A concise answer. "
const deps = { registry, generateText }
const chatResponse = () => Response.json({ choices: [{ message: { content: " Blue. " }, finish_reason: "stop" }], usage: { prompt_tokens: 100, completion_tokens: 5 } })

test("text calls go through OpenCode's sessionless generate with the default model", async () => {
  let seen: any
  const result = await interrogateResult({ paths: [], question: "Summarize these survivors", context: [{ score: 0.9 }] }, {
    registry, generateText: async (model, prompt) => { seen = { model, prompt }; return " A concise answer. " },
  })
  expect(seen.model).toEqual({ providerID: "opencode-go", id: "glm-5.3-flash" })
  expect(seen.prompt).toContain("Summarize these survivors")
  expect(seen.prompt).toContain(JSON.stringify([{ score: 0.9 }]))
  expect(seen.prompt.startsWith("Answer using the supplied files")).toBe(true)
  let bare = ""
  await interrogateResult({ paths: [], question: "Explain EHBP briefly" }, { registry, generateText: async (_, prompt) => { bare = prompt; return "ok" } })
  expect(bare).toBe("Explain EHBP briefly")
  expect(result).toEqual({ answer: "A concise answer.", model: "opencode-go/glm-5.3-flash", usage: {}, wall_ms: expect.any(Number) })
})
test("configured default overrides OpenCode's default", async () => {
  const result = await interrogateResult({ paths: [], question: "x" }, { ...deps, defaultModel: "tinfoil/kimi-k3" })
  expect(result.model).toBe("tinfoil/kimi-k3")
})
test("explicit variants pass through; thinking:false selects none", async () => {
  const models: any[] = []
  const capture = { registry, generateText: async (model: any) => { models.push(model); return "ok" } }
  await interrogateResult({ paths: [], question: "x", model: "tinfoil/deepseek-v4-1-flash#high" }, capture)
  await interrogateResult({ paths: [], question: "x", model: "tinfoil/deepseek-v4-1-flash", thinking: false }, capture)
  expect(models.map(m => m.variant)).toEqual(["high", "none"])
})
test("thinking:false without a none variant fails, naming real variants", () => {
  expect(() => resolveModel("tinfoil/kimi-k3", entries, false)).toThrow('no "none" variant, so thinking cannot be disabled; variants: low, high')
  expect(() => resolveModel("tinfoil/deepseek-v4-1-flash#high", entries, false)).toThrow("conflicts with variant #high")
  expect(() => resolveModel("tinfoil/kimi-k3#ultra", entries)).toThrow("has no variant #ultra")
})
test("unknown models point at discovery; unknown providers say so", () => {
  expect(() => resolveModel("tinfoil/gpt-9", entries)).toThrow('inference.models({tool: "generate", provider: "tinfoil"}) lists 2 models')
  expect(() => resolveModel("nowhere/x", entries)).toThrow("no provider nowhere is configured")
  expect(() => resolveModel("bare-name", entries)).toThrow("must be provider/model")
})
for (const [name, input] of [
  ["empty question", { paths: [], question: " " }],
  ["too many paths", { paths: Array(13).fill("a.txt"), question: "x" }],
  ["non-JSON context", { paths: [], question: "x", context: { invalid: undefined } }],
  ["oversized context", { paths: [], question: "x", context: "x".repeat(256_001) }],
  ["non-boolean thinking", { paths: [], question: "x", thinking: "no" }],
] as const) test(`rejects ${name} before registry or network`, async () => {
  let touched = false
  await expect(interrogateResult(input as any, { registry: { list: async () => { touched = true; return entries }, defaultModel: async () => undefined } })).rejects.toThrow("inference.generate:")
  expect(touched).toBe(false)
})
test("image calls without model use the image default, not OpenCode's default", async () => {
  const directory = await mkdtemp(join(tmpdir(), "interrogate-test-"))
  try {
    await writeFile(join(directory, "image.png"), Buffer.from([137, 80, 78, 71]))
    const result = await interrogateResult({ paths: ["image.png"], question: "Color?" }, {
      ...deps, imageDefault: "tinfoil/deepseek-v4-1-flash", access: { directory, authorize: async () => {} },
      secureFetch: async () => (async () => chatResponse()) as any,
    })
    expect(result.model).toBe("tinfoil/deepseek-v4-1-flash")
  } finally { await rm(directory, { recursive: true }) }
})
test("OpenCode-hosted models (no plugin-visible credentials) cannot take images", async () => {
  const hosted = entries.map(e => e.providerID === "opencode-go" ? { ...e, settings: { baseURL: e.settings!.baseURL } } : e)
  await expect(interrogateResult({ paths: ["photo.png"], question: "x", model: "opencode-go/glm-5.3-flash" }, { registry: { ...registry, list: async () => hosted }, generateText })).rejects.toThrow("credentials stay inside OpenCode")
})
test("images on a model without a direct transport fail before file authorization", async () => {
  let touched = false
  const access = { directory: "/", authorize: async () => { touched = true } }
  await expect(interrogateResult({ paths: ["photo.png"], question: "x", model: "anthropicish/claude" }, { ...deps, access })).rejects.toThrow('input: "image"')
  await expect(interrogateResult({ paths: ["photo.png"], question: "x", model: "omlx/LFM" }, { ...deps, access })).rejects.toThrow("cannot take images")
  expect(touched).toBe(false)
})
test("file denial happens before upload", async () => {
  await expect(interrogateResult({ paths: ["private.png"], question: "Describe" }, {
    ...deps, imageDefault: "tinfoil/deepseek-v4-1-flash", access: { directory: "/", authorize: async () => { throw new Error("denied") } },
  })).rejects.toThrow("denied")
})
test("text files are inlined into the sessionless prompt in order", async () => {
  const directory = await mkdtemp(join(tmpdir(), "interrogate-test-"))
  try {
    await writeFile(join(directory, "notes.txt"), "Local evidence")
    let prompt = ""
    await interrogateResult({ paths: ["notes.txt"], question: "Summarize", context: "Filtered records" }, {
      ...deps, access: { directory, authorize: async () => {} }, generateText: async (_, p) => { prompt = p; return "ok" },
    })
    expect(prompt.indexOf("Local evidence")).toBeLessThan(prompt.indexOf("Filtered records"))
    expect(prompt.indexOf("Filtered records")).toBeLessThan(prompt.indexOf("Summarize"))
  } finally { await rm(directory, { recursive: true }) }
})
test("image calls on OpenAI-compatible packages are built from the registry entry", async () => {
  const directory = await mkdtemp(join(tmpdir(), "interrogate-test-"))
  try {
    await writeFile(join(directory, "image.png"), Buffer.from([137, 80, 78, 71]))
    let sessionless = false
    const result = await interrogateResult({ paths: ["image.png"], question: "Color?", model: "opencode-go/glm-5.3-flash#low" }, {
      ...deps, access: { directory, authorize: async () => {} }, generateText: async () => { sessionless = true; return "x" },
      fetch: (async (url, options) => {
        expect(String(url)).toBe("https://opencode.ai/inference/go/openai/v1/chat/completions")
        expect(new Headers(options!.headers).get("Authorization")).toBe("Bearer go-key")
        const body = JSON.parse(options!.body as string)
        expect(body.model).toBe("glm-5.3-flash")
        expect(body.reasoning_effort).toBe("low")
        expect(body.messages[1].content.find((part: any) => part.type === "image_url").image_url.url).toBe("data:image/png;base64,iVBORw==")
        return chatResponse()
      }) as typeof fetch,
    })
    expect(sessionless).toBe(false)
    expect(result).toMatchObject({ answer: "Blue.", model: "opencode-go/glm-5.3-flash#low", usage: { input_tokens: 100, output_tokens: 5 } })
  } finally { await rm(directory, { recursive: true }) }
})
test("Tinfoil image calls use the secure transport with registry settings, never plain fetch", async () => {
  const directory = await mkdtemp(join(tmpdir(), "interrogate-test-"))
  try {
    await writeFile(join(directory, "image.png"), Buffer.from([137, 80, 78, 71]))
    let plain = false, settingsSeen: any
    const attestation = { verified: true as const, enclaveHost: "h", configRepo: "r", releaseDigest: "d", codeFingerprint: "f", enclaveFingerprint: "f" }
    const result = await interrogateResult({ paths: ["image.png"], question: "Color?", model: "tinfoil/deepseek-v4-1-flash", thinking: false }, {
      ...deps, access: { directory, authorize: async () => {} },
      fetch: (async () => { plain = true; return chatResponse() }) as typeof fetch,
      secureFetch: async settings => { settingsSeen = settings; return Object.assign((async (_: any, options: any) => {
        expect(JSON.parse(options.body).reasoning_effort).toBe("none")
        return chatResponse()
      }) as typeof fetch, { attestation: () => attestation }) },
    })
    expect(plain).toBe(false)
    expect(settingsSeen).toEqual({ baseURL: "https://inference.tinfoil.sh/v1/", transport: "ehbp" })
    expect(result.attestation).toEqual(attestation)
  } finally { await rm(directory, { recursive: true }) }
})
test("variant body cannot override model or messages", async () => {
  const directory = await mkdtemp(join(tmpdir(), "interrogate-test-"))
  try {
    await writeFile(join(directory, "image.png"), Buffer.from([137, 80, 78, 71]))
    const sneaky = entries.map(e => e.modelID === "glm-5.3-flash" ? { ...e, variants: [{ id: "x", body: { model: "other", messages: [] } }] } : e)
    await interrogateResult({ paths: ["image.png"], question: "x", model: "opencode-go/glm-5.3-flash#x" }, {
      registry: { ...registry, list: async () => sneaky }, generateText, access: { directory, authorize: async () => {} },
      fetch: (async (_, options) => {
        const body = JSON.parse(options!.body as string)
        expect(body.model).toBe("glm-5.3-flash")
        expect(body.messages.length).toBe(2)
        return chatResponse()
      }) as typeof fetch,
    })
  } finally { await rm(directory, { recursive: true }) }
})
test("sessionless failures throw without echoing details", async () => {
  await expect(interrogateResult({ paths: [], question: "x" }, { registry, generateText: async () => { throw new Error("secret body") } })).rejects.toThrow("inference.generate: opencode-go/glm-5.3-flash generation failed")
  await expect(interrogateResult({ paths: [], question: "x" }, { registry, generateText: async () => { throw new Error("secret body") } })).rejects.not.toThrow("secret")
})
test("Effect error tags are surfaced", async () => {
  await expect(interrogateResult({ paths: [], question: "x" }, { registry, generateText: async () => { throw { _tag: "ServiceUnavailableError", message: "secret" } } })).rejects.toThrow("generation failed (ServiceUnavailableError)")
})
test("empty sessionless output throws", async () => {
  await expect(interrogateResult({ paths: [], question: "x" }, { registry, generateText: async () => "  " })).rejects.toThrow("empty model response")
})
test("cancellation precedes registry access", async () => {
  let touched = false
  await expect(interrogateResult({ paths: [], question: "x" }, { signal: AbortSignal.abort(), registry: { list: async () => { touched = true; return entries }, defaultModel: async () => undefined } })).rejects.toThrow()
  expect(touched).toBe(false)
})
test("registers generate as a pinned structured Code Mode tool", async () => {
  const tools: Record<string, any> = {}
  await plugin.setup({ options: { generate: {} }, model: { list: async () => entries }, tool: { transform: async (f: any) => f({ namespace: () => {}, add: (tool: any) => tools[tool.name] = tool }) } })
  const generate = tools.generate
  expect(generate.options).toEqual({ namespace: "inference", codemode: true, pinned: true })
  expect(generate.input.properties.model.pattern).toBeDefined()
  expect(generate.input.properties.thinking.type).toBe("boolean")
  expect(generate.output.required).toContain("answer")
})
