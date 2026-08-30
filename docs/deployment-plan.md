# Deployment and integration plan

## Decision

Deploy `google/gemma-4-E4B-it` through NVIDIA's Jetson Orin vLLM image.  The
device is an Orin NX with 16 GB RAM, JetPack 6 / L4T r36.3, NVMe storage and
the NVIDIA container toolkit, all matching the supported path.  E4B, unlike
the larger 26B-A4B and 31B variants, supports text, image and audio input.
vLLM is selected because the Jetson guide identifies it as the audio-capable
small-model path; E2B with llama.cpp has a current audio limitation.

## Phase 0: unblock and prepare

1. Fix DNS so `ghcr.io` and `huggingface.co` resolve, then confirm outbound
   HTTPS works.
2. Confirm the operator can run `sudo docker ps`.  Optionally add `nvidia` to
   the `docker` group and re-login; that grants effectively root-equivalent
   Docker access, so keep it limited to a trusted local account.
3. Accept the Gemma terms on Hugging Face and place an access token only in
   `config.env` (mode `0600`).
4. Attach and test a supported camera and USB microphone.  Current inspection
   found neither a V4L2 camera node nor an ALSA recording device.

## Phase 1: model service

1. Start `scripts/start-vllm.sh` and watch `sudo docker logs -f gemma4-vllm`.
2. Run `scripts/smoke-test.sh` after the service is ready.
3. Keep the endpoint on localhost in this phase.  Do not expose an unauthenticated
   vLLM endpoint to the LAN or internet.
4. During initial load and requests, use `tegrastats` to check memory, swap,
   clocks and temperatures.  Do not co-host another large LLM/VLM.

## Phase 2: perception and speech

Use dedicated real-time components, then give Gemma compact structured events:

```text
camera -> TensorRT/DeepStream detector -> event JSON --+
microphone -> ASR / wake-word service -> transcript ---+-> Gemma E4B -> tool request
                                                        |
                                                        +-> logs / UI response
```

Gemma can analyse an image or audio clip, but should not be the continuous
object detector or wake-word engine: latency, repeatability and power use are
better with a detector and ASR designed for streaming.  Candidate components
are a TensorRT/DeepStream object detector, a local wake-word engine, and a
Jetson-optimised Whisper/FastConformer ASR service.  Choose the specific model
only after identifying the camera format, microphone and required languages.

## Phase 3: safe lighting control

Never let free-form model output control GPIO, mains relays or lights directly.
Put a small tool gateway between the model and hardware:

```text
Gemma tool call -> policy gateway -> allow-listed command -> MQTT/Home Assistant/isolated relay
                         |                    |
                         +-> audit log        +-> device acknowledgement
```

The gateway should expose narrow operations such as `set_light(room, state,
brightness)`; validate room names and value ranges; require confirmation for
unsafe/high-impact commands; set timeouts and a safe default state; and record
each request plus the device acknowledgement.  For mains-powered lights, use
certified isolated hardware rather than a Jetson GPIO pin.

## Phase 4: end-to-end acceptance

1. Text: model can produce a valid tool request but cannot invoke unknown tools.
2. Camera: a known test object becomes a timestamped event with confidence and
   no action occurs without an explicit policy.
3. Voice: wake word, local transcription, confirmation and response work under
   expected background noise.
4. Lighting: gateway rejects malformed/unauthorised commands; allowed commands
   produce an acknowledgement and audit record; a service restart fails safe.
5. Soak test: run at least an hour while checking thermal and memory stability.
