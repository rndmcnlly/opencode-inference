import { describe, expect, test } from "bun:test"
import { mkdtemp, rm, writeFile } from "fs/promises"
import { tmpdir } from "os"
import { join } from "path"
import { imageDefault, keyReader, loadConfig, parseKeyFile, readInferenceSection, verifiedImages, type RegistryModel } from "../index.ts"

const seeing = (providerID: string, modelID: string, settings: Record<string, any> = { baseURL: "https://example.test/v1", apiKey: "k" }): RegistryModel =>
  ({ id: modelID, modelID, providerID, package: "opencode-tinfoil/provider", enabled: true, capabilities: { input: ["text", "image"] }, settings })

describe("credentials", () => {
  test("environment wins over key files and the registry", async () => {
    const read = keyReader({ env: { OPENROUTER_API_KEY: " env-key " }, credentials: async () => { throw new Error("not consulted") } })
    expect(await read("openrouter-decisions")).toBe("env-key")
  })
  test("a configured key file accepts a bare key or a dotenv line", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-keys-"))
    try {
      await writeFile(join(directory, "bare"), "tk-123\n")
      await writeFile(join(directory, "dotenv"), "# comment\nexport OPENROUTER_API_KEY=\"or-456\"\nOTHER=x\n")
      const read = keyReader({ env: {}, credentials: async () => ({ tinfoil: join(directory, "bare"), openrouter: join(directory, "dotenv") }) })
      expect(await read("tinfoil-decisions")).toBe("tk-123")
      expect(await read("openrouter-decisions")).toBe("or-456")
      expect(() => parseKeyFile("A=1\nB=2", "OPENROUTER_API_KEY", "f")).toThrow("bare key or a OPENROUTER_API_KEY=... line")
      const missing = keyReader({ env: {}, credentials: async () => ({ tinfoil: join(directory, "nope") }) })
      await expect(missing("tinfoil-decisions")).rejects.toThrow("cannot read credentials.tinfoil")
    } finally { await rm(directory, { recursive: true }) }
  })
  test("falls back to the same-named OpenCode provider, then names every option", async () => {
    const registry = { list: async () => [seeing("tinfoil", "kimi-k3", { apiKey: "from-registry" })], defaultModel: async () => undefined }
    expect(await keyReader({ env: {}, registry })("tinfoil-decisions")).toBe("from-registry")
    await expect(keyReader({ env: {}, registry })("openrouter-decisions")).rejects.toThrow(/OPENROUTER_API_KEY.*credentials\.openrouter.*openrouter provider/)
  })
})

describe("settings sources", () => {
  test("plugin options are read like the settings file", async () => {
    const options = { decide: { default: "tinfoil/decision-large" }, generate: { imageDefault: "x/y" } }
    expect((await loadConfig(options)).default).toBe("tinfoil/decision-large")
    expect(await readInferenceSection("generate", options)).toEqual({ imageDefault: "x/y" })
    expect(await readInferenceSection("credentials", options)).toEqual({})
    await expect(readInferenceSection("decide", { decide: [] })).rejects.toThrow("plugin options: decide must be an object")
  })
})

describe("image default", () => {
  test("configured beats discovered; discovered is the first verified model this registry reaches", () => {
    const entries = [seeing("tinfoil", "kimi-k3"), seeing("local", "vision")]
    expect(imageDefault(entries, verifiedImages({}), "chosen/model")).toBe("chosen/model")
    expect(imageDefault(entries, verifiedImages({}))).toBe("tinfoil/kimi-k3")
    expect(imageDefault([seeing("local", "vision")], verifiedImages({ verifiedImages: ["local/vision"] }))).toBe("local/vision")
    expect(imageDefault([seeing("local", "vision")], verifiedImages({}))).toBeUndefined()
    expect(() => verifiedImages({ verifiedImages: "local/vision" })).toThrow("array of provider/model strings")
  })
})
