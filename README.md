# opencode-inference

**Probabilistic decisions and one-shot generation, as tools your agent can call.**

An OpenCode V2 plugin that adds one Code Mode namespace, `inference`, with three tools:

- `decide`: ask trained decision models (OpenRouter, Tinfoil) questions about some state and get native probabilities back, not prose or Booleans.
- `generate`: a one-turn, tool-free micro-subagent over any model in your OpenCode registry, for second opinions, image descriptions, or summarizing `decide` results.
- `models`: live discovery for both, including the constraints that would otherwise cost an agent a failed call.

```js
const seeing = await tools.inference.models({ tool: "generate", input: "image" })
const verdict = await tools.inference.decide({
  model: "tinfoil/decision-large",
  state: [{ type: "text", path: "ticket.txt" }],
  questions: { outage: { type: "noul", instructions: "Is checkout broken right now?" } },
})
const look = await tools.inference.generate({
  paths: ["screenshot.png"], question: "Does the error banner mention payments?",
})
```

Status: experimental, maintained for its author's own use, tested with OpenCode 2.1.4 (running on Bun). Issues and pull requests are welcome; there is no broad compatibility promise.

## Set up

Add the package to `plugins` in `~/.config/opencode/opencode.json(c)` or a project's `opencode.json(c)`:

```jsonc
{
  "plugins": [
    "github:rndmcnlly/opencode-inference#main"
  ]
}
```

Use a commit hash in place of `main` to pin a revision. For a local checkout, give its absolute path as the package, and run `bun install` in it once so the Tinfoil SDK is present.

### Credentials

`generate` needs none of its own: it uses whatever providers OpenCode already has. `decide` calls OpenRouter and Tinfoil directly and looks for each key in this order:

1. The environment: `OPENROUTER_API_KEY`, `TINFOIL_API_KEY`.
2. A key file named in `credentials` (below). The file may hold a bare key or a dotenv line such as `OPENROUTER_API_KEY=...`.
3. The OpenCode provider with the same ID (`openrouter`, `tinfoil`), if it has an API key configured.

You only need the providers you use. Without a Tinfoil key, `tinfoil/` models fail with a clear error and `models()` lists them as hints only.

### Settings (optional)

Put settings in the plugin entry's `options`. If an entry has no options, the plugin reads `~/.config/opencode/inference.json` instead (override the location with `OPENCODE_INFERENCE_CONFIG`). Changes to that file take effect on the next call. Options are read when the plugin loads and replace the file entirely.

```jsonc
{
  "plugins": [
    {
      "package": "github:rndmcnlly/opencode-inference#main",
      "options": {
        "decide": { "default": "tinfoil/decision-large" },
        "generate": {
          "default": "tinfoil/kimi-k3",
          "imageDefault": "tinfoil/gemma4-31b",
          "verifiedImages": ["my-local/qwen-vl"]
        },
        "credentials": { "openrouter": "~/.secrets/openrouter" }
      }
    }
  ]
}
```

| Key | Meaning |
| --- | --- |
| `decide.default` | Decision model used when a call names none (built-in: `openrouter/typesafe/jev-1.13`). |
| `decide.routes` | Per-model `description` and `media` (`image`, `audio`, `video`) hints, merged over the built-ins. |
| `generate.default` | Text model used when a call names none (otherwise OpenCode's default model). |
| `generate.imageDefault` | Image model used when a call names none (otherwise the first verified image model your registry can reach). |
| `generate.verifiedImages` | Extra `provider/model` IDs you have probed for image input. |
| `credentials.openrouter`, `credentials.tinfoil` | Key file paths; `~` expands. |

## Model strings follow the subagent grammar

Both tools take `model` as `provider/model`, the same strings used for subagents and sessions. Which providers suit a task is a policy decision for the caller's instructions. The tools carry no privacy classes or confidentiality flags.

| | `decide` | `generate` |
| --- | --- | --- |
| Models | Any `openrouter/…` or `tinfoil/…` decision model, discovered live | Any OpenCode registry model, checked live per call |
| Variants | none | `provider/model#variant`; `thinking: false` selects `#none` or fails naming real variants |
| Discovery | `models({tool: "decide"})`: live catalogs, `usable`, `created`, `zdr`, limits | `models({tool: "generate"})`: input, variants, `images`, `imagesVerified` |

## decide

This tool uses trained decision heads only. Chat models coerced into yes/no logprobs are a different instrument, because they are trained differently, and deliberately have no transport here.

The plugin encodes structure (transports, wire formats, safety). The catalog is whatever providers list today. `model` is `openrouter/<anything>` or `tinfoil/<anything>`: the prefix selects the transport and the rest passes through, so a model released this morning works without a plugin update. Unknown names fail upstream with recovery hints mapped from the status code and well-known phrases (no ZDR endpoint, not a decision model, rate limit, credit, credentials). Upstream response bodies are never echoed.

- `models({tool: "decide"})` reads OpenRouter (`/models?output_modalities=decisions` joined with `/endpoints/zdr`) and Tinfoil (`/v1/models`, `type: "decision"`) live. Results are cached for 10 minutes and merged with built-in hints. Each entry has:
  - `usable`: live and not ZDR-blocked. Usable models are listed first, newest first.
  - `created`, `input`, `zdr`, `probed`, and `limits`.
  - `live`: whether the provider lists it now. `false` on a hint means the model was retired upstream.

  If a catalog is unavailable, results fall back to the hints and include `warnings`.
- OpenRouter requests always enforce zero data retention (ZDR) and no data collection, so trying a new model cannot weaken that. `~alias` models have unknown ZDR status until called.
- Images go only to `openrouter/` models whose live catalog lists image input, because a silent wrong answer is worse than an error. Tinfoil's decisions adapter accepts text and JSON only. Audio and video parts have no tested adapter yet.
- Tinfoil uses an OpenAI-format adapter, which cannot represent Noul `criteria` or a one-option Choice faithfully. These are rejected, never folded into instructions. Tinfoil may also return `{type: "refusal"}` for an individual question.
- `model` echoes the requested string, and `resolvedModel` is the upstream snapshot.
- A Score is a probability-weighted position on the scale (an expected value), not a bucket. A Noul is P(yes), never a Boolean. Callers own thresholds and policy.
- **Calibration caveat:** on the same unambiguous outage ticket, Jev gave 0.51 when asked as a Noul and 0.94 when asked as a Choice. On an ambiguous ticket, only Jev hedged (~0.3), while most other models said <0.05. Ask in more than one form, or across models, before trusting a single probability.

## generate

A one-turn, tool-free micro-subagent.

- **Text-only calls** use OpenCode's sessionless generation (`ctx.generate.text` in 2.1.4, documented as `ctx.experimental.generate.text`; both are feature-detected). The model string resolves through the same registry as sessions, so your credentials and provider plugins apply (for example, `opencode-tinfoil`'s attested transport) and no session is created.
  - Text files and context are inlined into the prompt.
  - There is no system role, so evidence-handling instructions lead the prompt, and only when evidence is supplied.
  - This path reports no token usage and has no max-tokens setting.
- **Image calls** cannot use that route because it accepts text only. Instead, the plugin builds the request directly from the same registry entry: base URL, API key, variant `reasoningEffort`, and request body.
  - Providers packaged with `opencode-tinfoil` go through Tinfoil's `SecureClient` using the entry's own settings, which may include a separate attestation bundle URL, and return `attestation` evidence.
  - This requires an OpenAI-chat package and credentials visible to plugins, or a loopback endpoint.
  - OpenCode-hosted providers keep their auth inside OpenCode, so they cannot take images here.
- `images` means an image call can be built; `imagesVerified` means it has actually been probed. Built-in verified models: Tinfoil `gemma4-31b`, `deepseek-v4-1-flash`, `kimi-k3`. Add your own with `generate.verifiedImages`. Registry capability claims are not evidence: one OpenAI-compatible gateway claimed image input for a GLM model and returned HTTP 400. When probing a small local model, use two colors, since a 2B model answered "Blue" for a green image.
- Every call resolves against the live registry, which reloads when the configuration changes, so adding or removing a provider in `opencode.json` takes effect without a restart.
- Host scripts without a plugin context can reach the same registry and sessionless route through the authenticated `opencode api` CLI (`cliRegistry`).

## Shared boundaries

- Tinfoil traffic uses only the Tinfoil JS `SecureClient`, one attested client per endpoint. It verifies attestation first, then sends EHBP-encrypted bodies, and throws rather than sending plaintext. The EHBP transport is enforced, so no provider setting can downgrade it to TLS.
  - EHBP encrypts bodies, not headers.
  - `attestation` carries `configRepo`, `releaseDigest`, and code and enclave fingerprints. Equal fingerprints mean the enclave is running the published release.
- Files are read host-side under session permissions: the external-directory gate applies, and symlink targets are rechecked. Capability rejections happen before any permission prompt or file read.
- Upstream error bodies are never echoed. Error class names and Effect tags are. `models()` never returns credentials, even though registry entries contain them.
- **Known transient:** the first request on a fresh Tinfoil client often fails with `KeyConfigMismatchError`. By design there is no automatic retry, only a hint in the error message.

## Why one file

OpenCode hot-reloads a plugin's entry module only. Imported sibling modules stay cached until a restart, which once made edits silently look ineffective. Keeping everything in `index.ts` avoids that.

## Agent-experience testing, 2026-10-27

Small agents (glm-5.3-flash) ran second-opinion, confidential-summary, explicit-provider, and blind-vision tasks across four iterations. Their failures drove these features:

- per-model `limits` and `media`
- attestation evidence
- a successor for a retired model
- actionable errors
- `imageDefault`, because OpenCode's default model could not see images
- `imagesVerified`, after unverified image claims cost one vision task five calls

## Development

```sh
bun install
bun test tests
```

The integration boundaries (registry, fetch, secure fetch, credentials, permissions) are injectable, so the tests run offline.

## License

MIT
