import { describe, expect, test } from "bun:test"
import plugin, { attestationEvidence, BUILTIN_CONFIG, decideCatalog, rejectionHint, contentParts, createClient, loadConfig, prepareRequest, resolveRoute, toOpenAIDecisions, validateResponse, type Request } from "../index.ts"
import { mkdtemp, rm, writeFile, symlink } from "fs/promises"
import { tmpdir } from "os"
import { join } from "path"

const request: Request = {
  state: { ticket: "Checkout is down" },
  questions: {
    outage: { type: "noul", instructions: "Is checkout down?" },
    team: { type: "choice", instructions: "Which team?", criteria: { payments: "Checkout", account: "Login" } },
    impact: { type: "score", instructions: "How disruptive?", criteria: ["None", "Minor", "Unusable"] },
  },
}
const answer = () => ({
  model: "typesafe/jev-1.13-20260917", provider: "TypeSafe", id: "test-id",
  answers: {
    outage: { type: "noul", noul: 0.97 },
    team: { type: "choice", choice: "payments", confidence: 0.8, probabilities: { payments: 0.9, account: 0.1 } },
    impact: { type: "score", score: 1.8, confidence: 0.7, probabilities: { 0: 0, 1: 0.2, 2: 0.8 }, legend: { 0: "None", 1: "Minor", 2: "Unusable" } },
  },
  usage: { input_tokens: 100, output_tokens: 20, cost: 0.0000042 },
})

describe("request validation and OpenRouter symmetry", () => {
  test("adds only model/privacy defaults without rewriting questions", () => {
    const copy = structuredClone(request)
    const { payload } = prepareRequest(request)
    expect(payload.model).toBe("typesafe/jev-1.13")
    expect(payload.provider).toEqual({ zdr: true, data_collection: "deny" })
    expect(payload.questions).toEqual(copy.questions)
    expect(request).toEqual(copy)
  })
  test("preserves structured guidance", () => {
    const input = { ...request,
      questions: { structured: { type: "choice" as const, instructions: { task: "classify" }, criteria: { a: null, b: ["guidance"] } } } }
    const { payload } = prepareRequest(input)
    expect(payload.questions).toEqual(input.questions)
  })
  test("accepts upstream single-choice/single-level score shapes", () => {
    expect(() => prepareRequest({ state: "x", questions: { c: { type: "choice", instructions: "x", criteria: { only: "x" } }, s: { type: "score", instructions: "x", criteria: ["only"] } } })).not.toThrow()
  })
  test("accepts explicit Noul criteria and arrays as state", () => {
    expect(() => prepareRequest({ state: ["x"], questions: { b: { type: "noul", instructions: ["x"], criteria: { true: { meaning: "yes" }, false: "no" } } } })).not.toThrow()
  })
  const invalid: [string, any][] = [
    ["empty questions", { ...request, questions: {} }],
    ["wrong question type", { ...request, questions: { bad: { type: "text", instructions: "x" } } }],
    ["missing instructions", { ...request, questions: { bad: { type: "noul" } } }],
    ["empty Choice", { ...request, questions: { bad: { type: "choice", instructions: "x", criteria: {} } } }],
    ["empty Score", { ...request, questions: { bad: { type: "score", instructions: "x", criteria: [] } } }],
    ["incomplete Noul criteria", { ...request, questions: { bad: { type: "noul", instructions: "x", criteria: { true: "yes" } } } }],
    ["unknown fields", { ...request, endpoint: "https://evil.example" }],
    ["model selection", { ...request, model: "other/model" }],
    ["provider selection", { ...request, provider: { only: ["typesafe"] } }],
    ["primitive state", { ...request, state: 123 }],
    ["nonfinite", { ...request, state: { number: Infinity } }],
    ["undefined", { ...request, state: { value: undefined } }],
    ["date coercion", { ...request, state: new Date() }],
    ["oversized", { ...request, state: "x".repeat(256_001) }],
    ["too many questions", { ...request, questions: Object.fromEntries(Array.from({ length: 33 }, (_, i) => [i, { type: "noul", instructions: "x" }])) }],
    ["privacy bypass", { ...request, provider: { zdr: false } }],
    ["training bypass", { ...request, provider: { data_collection: "allow" } }],
  ]
  for (const [name, input] of invalid) test(`rejects ${name} before credentials/network`, async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "key" } })
    await expect(client(input)).rejects.toThrow("inference.decide:")
    expect(touched).toBe(false)
  })
  test("rejects cycles", () => {
    const state: any = {}; state.self = state
    expect(() => prepareRequest({ ...request, state })).toThrow("cyclic")
  })
})

describe("response validation", () => {
  test("accepts complete native response", () => expect(() => validateResponse(answer(), request.questions)).not.toThrow())
  test("accepts optional distributions/metadata being omitted", () => {
    const value: any = answer()
    delete value.answers.team.probabilities; delete value.answers.team.confidence
    delete value.answers.impact.probabilities; delete value.answers.impact.legend
    delete value.provider; delete value.id; delete value.usage.cost
    expect(() => validateResponse(value, request.questions)).not.toThrow()
  })
  const mutations: [string, (v: any) => void][] = [
    ["missing question", v => delete v.answers.outage],
    ["extra question", v => v.answers.extra = { type: "noul", noul: 1 }],
    ["wrong answer type", v => v.answers.outage.type = "choice"],
    ["invalid probability", v => v.answers.outage.noul = 1.1],
    ["unknown choice", v => v.answers.team.choice = "invented"],
    ["missing distribution label", v => delete v.answers.team.probabilities.account],
    ["zero distribution", v => v.answers.team.probabilities = { payments: 0, account: 0 }],
    ["invalid confidence", v => v.answers.team.confidence = NaN],
    ["score outside scale", v => v.answers.impact.score = 3],
    ["invalid legend", v => delete v.answers.impact.legend[2]],
    ["invalid tokens", v => v.usage.input_tokens = -1],
    ["invalid cost", v => v.usage.cost = -1],
  ]
  for (const [name, mutate] of mutations) test(`rejects ${name}`, () => {
    const value = answer(); mutate(value)
    expect(() => validateResponse(value, request.questions)).toThrow("inference.decide:")
  })
  test("allows rounded distribution sums", () => {
    const value = answer(); value.answers.team.probabilities = { payments: 0.9, account: 0.09 }
    expect(() => validateResponse(value, request.questions)).not.toThrow()
  })
})

describe("transport and plugin", () => {
  test("fixed endpoint, fixed policy, native envelope, no response coercion", async () => {
    const response = answer()
    let calls = 0
    const client = createClient({ readKey: async () => "secret", fetch: (async (url, options) => {
      calls++
      expect(url).toBe("https://openrouter.ai/api/alpha/decisions")
      expect(new Headers(options!.headers).get("Authorization")).toBe("Bearer secret")
      expect(JSON.parse(options!.body as string).provider.zdr).toBe(true)
      return Response.json(response)
    }) as typeof fetch })
    expect(await client(request)).toEqual({ ...response, model: "openrouter/typesafe/jev-1.13", resolvedModel: "typesafe/jev-1.13-20260917" })
    expect(calls).toBe(1)
  })
  test("HTTP errors do not echo upstream bodies or retry", async () => {
    let calls = 0
    const client = createClient({ readKey: async () => "secret", fetch: (async () => { calls++; return new Response("state and secret", { status: 429 }) }) as typeof fetch })
    await expect(client(request)).rejects.toThrow("OpenRouter HTTP 429")
    expect(calls).toBe(1)
  })
  test("invalid JSON response", async () => {
    const client = createClient({ readKey: async () => "secret", fetch: (async () => new Response("not JSON")) as typeof fetch })
    await expect(client(request)).rejects.toThrow("invalid JSON response")
  })
  test("pre-cancelled request performs no credential access", async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "key" } })
    await expect(client(request, AbortSignal.abort())).rejects.toThrow()
    expect(touched).toBe(false)
  })
  test("in-flight cancellation reaches transport", async () => {
    const controller = new AbortController()
    const client = createClient({ readKey: async () => "secret", fetch: (async (_, options) => {
      controller.abort()
      options!.signal!.throwIfAborted()
      return Response.json(answer())
    }) as typeof fetch })
    await expect(client(request, controller.signal)).rejects.toThrow()
  })
  test("network errors are redacted", async () => {
    const client = createClient({ readKey: async () => "secret", fetch: (async () => { throw new Error("secret") }) as typeof fetch })
    await expect(client(request)).rejects.toThrow("network request failed")
  })
  test("Code Mode-only pinned registration; models() reads the live registry without credentials", async () => {
    const tools: Record<string, any> = {}
    const registryEntries = [
      { id: "kimi-k3", modelID: "kimi-k3", providerID: "tinfoil", package: "opencode-tinfoil/provider", enabled: true, capabilities: { input: ["text", "image"] },
        variants: [{ id: "low", settings: { reasoningEffort: "low" } }], settings: { baseURL: "https://inference.tinfoil.sh/v1/", apiKey: "SECRET-KEY" } },
      { id: "claude", modelID: "claude", providerID: "opencode", package: "@opencode/ai/providers/anthropic", enabled: true, capabilities: { input: ["text", "image"] }, settings: { apiKey: "SECRET-KEY" } },
    ]
    await plugin.setup({
      options: { decide: {} },
      model: { list: async () => ({ data: registryEntries }), default: async () => ({ providerID: "opencode", modelID: "claude" }) },
      tool: { transform: async (f: any) => f({ namespace: () => {}, add: (t: any) => tools[t.name] = t }) },
    })
    expect(Object.keys(tools).sort()).toEqual(["decide", "generate", "models"])
    expect(tools.decide.options).toEqual({ namespace: "inference", codemode: true, pinned: true })
    expect(tools.decide.output.required).toContain("answers")
    expect(tools.decide.input.properties.model.pattern).toBe("^(openrouter|tinfoil)/[^\\s]+$")
    expect(tools.decide.input.properties.model.enum).toBeUndefined()
    expect(tools.decide.input.properties.confidential).toBeUndefined()
    const listed = (await tools.models.execute({ tool: "generate" })).output
    expect(listed.decide).toBeUndefined()
    expect(JSON.stringify(listed)).not.toContain("SECRET-KEY")
    expect(JSON.stringify(listed)).not.toContain("baseURL")
    expect(listed.generate.models).toEqual([
      { model: "tinfoil/kimi-k3", input: ["text", "image"], variants: ["low"], images: true, imagesVerified: true },
      { model: "opencode/claude", input: ["text", "image"], variants: [], images: false, imagesVerified: false },
    ])
    const seeing = (await tools.models.execute({ tool: "generate", input: "image" })).output
    expect(seeing.decide).toBeUndefined()
    expect(seeing.generate.models.map((m: any) => m.model)).toEqual(["tinfoil/kimi-k3"])
  })
})

describe("decide model grammar", () => {
  test("omitted model uses the default", () => {
    expect(resolveRoute(BUILTIN_CONFIG, {}).name).toBe("openrouter/typesafe/jev-1.13")
  })
  test("the provider prefix selects the transport", () => {
    expect(resolveRoute(BUILTIN_CONFIG, { model: "tinfoil/decision-large" })).toMatchObject({ transport: "tinfoil-decisions", model: "decision-large" })
    expect(resolveRoute(BUILTIN_CONFIG, { model: "openrouter/typesafe/jev-1.13" })).toMatchObject({ transport: "openrouter-decisions", model: "typesafe/jev-1.13" })
  })
  test("any provider/model passes through; unknown providers point at discovery", () => {
    expect(resolveRoute(BUILTIN_CONFIG, { model: "openrouter/upstage/solar-decide" })).toEqual({ name: "openrouter/upstage/solar-decide", transport: "openrouter-decisions", model: "upstage/solar-decide" })
    expect(resolveRoute(BUILTIN_CONFIG, { model: "tinfoil/decision-xl" })).toMatchObject({ transport: "tinfoil-decisions", model: "decision-xl" })
    expect(() => resolveRoute(BUILTIN_CONFIG, { model: "gpt-whatever" })).toThrow('inference.models({tool: "decide"})')
    expect(() => resolveRoute(BUILTIN_CONFIG, { model: "acme/decider" })).toThrow("provider openrouter or tinfoil")
  })
  test("upstream rejections become recovery hints without echoing bodies", async () => {
    expect(rejectionHint(404, "No endpoints found matching your data policy", false)).toContain("no endpoint satisfies ZDR")
    expect(rejectionHint(400, "acme/x is not a valid model ID", false)).toContain("not a decision model")
    expect(rejectionHint(400, "Model acme/x does not exist", false)).toContain("not a decision model")
    expect(resolveRoute(BUILTIN_CONFIG, { model: "openrouter/~liquid/d1-ultra-latest" }).model).toBe("~liquid/d1-ultra-latest")
    expect(resolveRoute(BUILTIN_CONFIG, { model: "openrouter/baseten/mammoth-1:free" }).model).toBe("baseten/mammoth-1:free")
    expect(rejectionHint(429, "", true)).toContain("rate limited")
    const client = createClient({ readKey: async () => "k", fetch: (async () => new Response("No endpoints found matching your data policy; state: SECRET", { status: 404 })) as typeof fetch })
    const rejected = client({ ...request, model: "openrouter/acme/new-decider" })
    await expect(rejected).rejects.toThrow("openrouter/acme/new-decider rejected (OpenRouter HTTP 404); no endpoint satisfies ZDR")
    await expect(rejected).rejects.not.toThrow("SECRET")
  })
  test("live catalog merges hints, flags ZDR, and exposes retired hints", async () => {
    const catalog = await decideCatalog(BUILTIN_CONFIG, { cache: false, readKey: async () => "tk", fetch: (async (url: any) => {
      const u = String(url)
      if (u.includes("output_modalities=decisions")) return Response.json({ data: [
        { id: "typesafe/jev-1.13", name: "TypeSafe: Jev 1.13", architecture: { input_modalities: ["text", "image"], output_modalities: ["decisions"] } },
        { id: "upstage/solar-decide", name: "Upstage: Solar Decide", created: 1791000000, architecture: { input_modalities: ["text"], output_modalities: ["decisions"] } },
        { id: "acme/no-zdr", architecture: { input_modalities: ["text"], output_modalities: ["decisions"] } },
      ] })
      if (u.includes("/endpoints/zdr")) return Response.json({ data: [{ model_id: "typesafe/jev-1.13" }, { model_id: "upstage/solar-decide" }] })
      if (u.endsWith("/v1/models")) return Response.json({ data: [{ id: "decision-large", type: "decision" }, { id: "glm-5-3", type: "chat" }, { id: "decision-xl", type: "decision" }] })
      throw new Error("unexpected " + u)
    }) as typeof fetch })
    const by = Object.fromEntries(catalog.models.map(entry => [entry.model, entry]))
    expect(by["openrouter/typesafe/jev-1.13"]).toMatchObject({ live: true, probed: true, zdr: true, input: ["text", "image"] })
    expect(by["openrouter/upstage/solar-decide"]).toMatchObject({ live: true, probed: false, zdr: true, description: "Upstage: Solar Decide" })
    expect(by["openrouter/acme/no-zdr"]).toMatchObject({ live: true, zdr: false, usable: false })
    expect(by["openrouter/upstage/solar-decide"]).toMatchObject({ usable: true, created: "2026-10-03" })
    expect(by["openrouter/perplexity/pplx-decider-v1.2"].usable).toBe(false)
    expect(catalog.models[0].model).toBe("openrouter/upstage/solar-decide")
    expect(catalog.models.findIndex(m => !m.usable)).toBeGreaterThan(catalog.models.findLastIndex(m => m.usable))
    expect(by["openrouter/perplexity/pplx-decider-v1.2"]).toMatchObject({ live: false, probed: true })
    expect(by["tinfoil/decision-xl"]).toMatchObject({ live: true, probed: false, limits: expect.stringContaining("Noul criteria") })
    expect(by["tinfoil/glm-5-3"]).toBeUndefined()
    expect(catalog.warnings).toEqual([])
  })
  test("unreachable catalogs fall back to hints with warnings", async () => {
    const catalog = await decideCatalog(BUILTIN_CONFIG, { cache: false, readKey: async () => "tk", fetch: (async () => { throw new Error("offline") }) as typeof fetch })
    expect(catalog.warnings.length).toBe(2)
    expect(catalog.models.every(entry => entry.probed && !entry.live)).toBe(true)
  })
  test("confidential is no longer a request field", async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "key" } })
    await expect(client({ ...request, confidential: true } as any)).rejects.toThrow("unsupported request field")
    expect(touched).toBe(false)
  })
  test("config file adds provider/model entries and rejects transport overrides", async () => {
    const directory = await mkdtemp(join(tmpdir(), "decisions-config-"))
    try {
      const path = join(directory, "inference.json")
      await writeFile(path, JSON.stringify({ decide: { default: "tinfoil/decision-small", routes: { "openrouter/x/clef": { description: "test" } } } }))
      const config = await loadConfig(path)
      expect(config.default).toBe("tinfoil/decision-small")
      expect(config.routes["openrouter/x/clef"]).toEqual({ transport: "openrouter-decisions", model: "x/clef", description: "test" })
      expect(Object.keys(config.routes)).toContain("openrouter/typesafe/jev-1.13")
      await writeFile(path, JSON.stringify({ decide: { routes: { "tinfoil/decision-small": { transport: "openrouter-decisions" } } } }))
      await expect(loadConfig(path)).rejects.toThrow("follows the provider prefix")
      await writeFile(path, JSON.stringify({ decide: { routes: { jev: {} } } }))
      await expect(loadConfig(path)).rejects.toThrow("must be provider/model")
      await writeFile(path, JSON.stringify({ decide: { default: "openrouter/nope" } }))
      await expect(loadConfig(path)).rejects.toThrow("is not defined")
      expect(await loadConfig(join(directory, "missing.json"))).toEqual(BUILTIN_CONFIG)
    } finally { await rm(directory, { recursive: true }) }
  })
})

describe("Tinfoil OpenAI-format transport", () => {
  const tinfoilAnswer = {
    answers: [
      { type: "predicate", name: "outage", probability: 0.97 },
      { type: "choice", name: "team", choice: "payments", confidence: 0.9, probabilities: [{ value: "payments", probability: 0.9 }, { value: "account", probability: 0.1 }] },
      { type: "score", name: "impact", score: 1.8, confidence: 0.8, probabilities: [{ value: 0, label: "None", probability: 0 }, { value: 1, label: "Minor", probability: 0.2 }, { value: 2, label: "Unusable", probability: 0.8 }] },
    ],
    usage: { input_tokens: 305, output_tokens: 0 },
  }
  test("maps keyed questions to ordered named questions without rewriting", () => {
    const body = toOpenAIDecisions("decision-large", request.state, request.questions)
    expect(body.input).toBe(JSON.stringify(request.state))
    expect(body.questions).toEqual([
      { type: "predicate", name: "outage", instructions: "Is checkout down?" },
      { type: "choice", name: "team", instructions: "Which team?", choices: [{ value: "payments", description: "Checkout" }, { value: "account", description: "Login" }] },
      { type: "score", name: "impact", instructions: "How disruptive?", levels: [{ label: "None" }, { label: "Minor" }, { label: "Unusable" }] },
    ])
  })
  test("rejects shapes with no faithful equivalent", () => {
    expect(() => toOpenAIDecisions("m", "x", { b: { type: "noul", instructions: "x", criteria: { true: "y", false: "n" } } })).toThrow("no equivalent")
    expect(() => toOpenAIDecisions("m", "x", { c: { type: "choice", instructions: "x", criteria: { only: "x" } } })).toThrow("2–255")
  })
  test("uses the secure transport, never plain fetch, and returns the shared envelope", async () => {
    let plain = false, url = ""
    const client = createClient({
      readKey: async transport => { expect(transport).toBe("tinfoil-decisions"); return "tk" },
      fetch: (async () => { plain = true; return Response.json({}) }) as typeof fetch,
      secureFetch: async () => (async (u: any, options: any) => {
        url = String(u)
        expect(JSON.parse(options.body).model).toBe("decision-medium")
        return Response.json(tinfoilAnswer)
      }) as typeof fetch,
    })
    const result = await client({ ...request, model: "tinfoil/decision-medium" })
    expect(plain).toBe(false)
    expect(url).toBe("https://inference.tinfoil.sh/v1/decisions")
    expect(result.model).toBe("tinfoil/decision-medium")
    expect(result.answers.outage).toEqual({ type: "noul", noul: 0.97 })
    expect(result.answers.team).toMatchObject({ type: "choice", choice: "payments", probabilities: { payments: 0.9, account: 0.1 } })
    expect(result.answers.impact).toMatchObject({ type: "score", score: 1.8, legend: { 0: "None", 1: "Minor", 2: "Unusable" } })
  })
  test("attestation evidence is returned for attested routes", async () => {
    const doc = { securityVerified: true, enclaveHost: "router.tinfoil.sh", configRepo: "tinfoilsh/confidential-model-router", releaseDigest: "abc", codeFingerprint: "c1", enclaveFingerprint: "e1", tlsPublicKey: "k" }
    const secure = Object.assign((async () => Response.json(tinfoilAnswer)) as typeof fetch, { attestation: () => attestationEvidence(doc) })
    const result = await createClient({ readKey: async () => "tk", secureFetch: async () => secure })({ ...request, model: "tinfoil/decision-large" })
    expect(result.attestation).toEqual({ verified: true, enclaveHost: "router.tinfoil.sh", configRepo: "tinfoilsh/confidential-model-router", releaseDigest: "abc", codeFingerprint: "c1", enclaveFingerprint: "e1" })
    expect(() => attestationEvidence({ ...doc, securityVerified: false })).toThrow("not securityVerified")
  })
  test("refusals pass through per question", async () => {
    const value = structuredClone(tinfoilAnswer)
    value.answers[0] = { type: "refusal", name: "outage" } as any
    const client = createClient({ readKey: async () => "tk", secureFetch: async () => (async () => Response.json(value)) as typeof fetch })
    expect((await client({ ...request, model: "tinfoil/decision-small" })).answers.outage).toEqual({ type: "refusal" })
  })
  test("attestation failure is reported without plaintext fallback", async () => {
    const failure = Object.assign(new Error("body with secret state"), { name: "KeyConfigMismatchError" })
    const client = createClient({ readKey: async () => "tk", secureFetch: async () => (async () => { throw failure }) as typeof fetch })
    const rejected = client({ ...request, model: "tinfoil/decision-large" })
    await expect(rejected).rejects.toThrow("(KeyConfigMismatchError); nothing was sent unverified. Known first-request transient")
    await expect(rejected).rejects.not.toThrow("secret")
  })
  test("media on a Tinfoil route fails before file access", async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "tk" } })
    await expect(client({ ...request, model: "tinfoil/decision-large", state: [{ type: "image", path: "x.png" }] }, undefined, { directory: "/", authorize: async () => { touched = true } })).rejects.toThrow("Tinfoil decisions adapter is text/JSON only")
    expect(touched).toBe(false)
  })
})

describe("disk-backed multimodal state", () => {
  test("ordinary JSON arrays stay literal", () => {
    expect(contentParts([{ event: "login" }])).toBeUndefined()
  })
  test("rejects media URLs and raw upstream/base64 parts", () => {
    expect(() => prepareRequest({ ...request, state: [{ type: "image", path: "https://example.org/image.png" }] })).toThrow("URLs")
    expect(() => prepareRequest({ ...request, state: [{ type: "image_url", image_url: { url: "data:image/png;base64,eA==" } }] })).toThrow("disk-backed")
  })
  test("audio/video parts fail before catalog, file authorization, credentials, or network", async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "key" }, fetch: (async () => { touched = true; return Response.json({}) }) as typeof fetch })
    await expect(client({ ...request, state: [{ type: "audio", path: "missing.wav" }] }, undefined, {
      directory: "/", authorize: async () => { touched = true },
    })).rejects.toThrow("no tested adapter yet")
    expect(touched).toBe(false)
  })
  test("an unknown model gets images only if its live catalog advertises them", async () => {
    let authorized = false
    const client = createClient({ readKey: async () => "key", fetch: (async (url: any) => String(url).includes("/acme/new/endpoints")
      ? new Response("not found", { status: 404 }) : Response.json({})) as typeof fetch })
    await expect(client({ ...request, model: "openrouter/acme/new", state: [{ type: "image", path: "x.png" }] }, undefined, { directory: "/", authorize: async () => { authorized = true } })).rejects.toThrow("no catalog entry for acme/new (HTTP 404)")
    expect(authorized).toBe(false)
  })
  test("reads UTF-8 files, resolves relative paths, preserves content order", async () => {
    const directory = await mkdtemp(join(tmpdir(), "decisions-test-"))
    try {
      await writeFile(join(directory, "ticket.txt"), "Checkout is fixed.")
      const authorized: string[] = []
      const client = createClient({ readKey: async () => "key", fetch: (async (_, options) => {
        expect(JSON.parse(options!.body as string).state).toEqual([{ type: "text", text: "Context" }, { type: "text", text: "Checkout is fixed." }])
        return Response.json(answer())
      }) as typeof fetch })
      await client({ ...request, state: [{ type: "text", text: "Context" }, { type: "text", path: "ticket.txt" }] }, undefined, {
        directory, authorize: async path => { authorized.push(path) },
      })
      expect(authorized).toContain(join(directory, "ticket.txt"))
    } finally { await rm(directory, { recursive: true }) }
  })
  test("denial prevents credentials and upload", async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "key" } })
    await expect(client({ ...request, state: [{ type: "text", path: "private.txt" }] }, undefined, {
      directory: "/", authorize: async () => { throw new Error("denied") },
    })).rejects.toThrow("denied")
    expect(touched).toBe(false)
  })
  test("requires permission context for file reads", async () => {
    await expect(createClient()({ ...request, state: [{ type: "text", path: "ticket.txt" }] })).rejects.toThrow("permission context")
  })
  test("verified image route also checks live catalog and serializes bytes host-side", async () => {
    const directory = await mkdtemp(join(tmpdir(), "decisions-test-"))
    try {
      await writeFile(join(directory, "image.png"), Buffer.from([137, 80, 78, 71]))
      let calls = 0
      const client = createClient({ readKey: async () => "key", fetch: (async (url, options) => {
        calls++
        if (String(url).endsWith("/endpoints")) return Response.json({ data: { architecture: { input_modalities: ["text", "image"] } } })
        const body = JSON.parse(options!.body as string)
        expect(body.state).toEqual([{ type: "image_url", image_url: { url: "data:image/png;base64,iVBORw==" } }])
        expect(body.provider.zdr).toBe(true)
        return Response.json(answer())
      }) as typeof fetch })
      await client({ ...request, model: "openrouter/perplexity/pplx-decider-v1.2", state: [{ type: "image", path: "image.png" }] }, undefined, { directory, authorize: async () => {} })
      expect(calls).toBe(2)
    } finally { await rm(directory, { recursive: true }) }
  })
  test("catalog modality regression fails before permissions", async () => {
    let touched = false
    const client = createClient({ readKey: async () => { touched = true; return "key" }, fetch: (async () => Response.json({ data: { architecture: { input_modalities: ["text"] } } })) as typeof fetch })
    await expect(client({ ...request, model: "openrouter/perplexity/pplx-decider-v1.2", state: [{ type: "image", path: "image.png" }] }, undefined, { directory: "/", authorize: async () => { touched = true } })).rejects.toThrow("does not advertise image input")
    expect(touched).toBe(false)
  })
  test("symlink target gets its own permission check", async () => {
    const directory = await mkdtemp(join(tmpdir(), "decisions-test-"))
    try {
      await writeFile(join(directory, "target.txt"), "Evidence")
      await symlink(join(directory, "target.txt"), join(directory, "alias.txt"))
      const authorized: string[] = []
      const client = createClient({ readKey: async () => "key", fetch: (async () => Response.json(answer())) as typeof fetch })
      await client({ ...request, state: [{ type: "text", path: "alias.txt" }] }, undefined, { directory, authorize: async path => { authorized.push(path) } })
      expect(authorized.length).toBe(2)
      expect(authorized[0]).toEndWith("alias.txt")
      expect(authorized[1]).toEndWith("target.txt")
    } finally { await rm(directory, { recursive: true }) }
  })
})
