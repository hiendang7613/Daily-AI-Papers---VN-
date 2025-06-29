# QuickNotebookAI
# Daily-AI-Papers---VN-

```mermaid



flowchart LR
    %% ─── Nodes ─────────────────────────────────────────
    A([Audio Streaming])
    B[VAD<br/>Silero]
    C[[Audio Segments]]
    D[Language ID<br/>Whisper]
    E[Per-language<br/>Alignment]
    F[Whisper v3 Turbo<br/>faster-whisper fp16]
    G[Stable-Prefix<br/>LA-n + τ]
    S[Speaker Embedding<br/>ECAPA / TDNN / ResNet293]
    R[Re-align Text]
    OUT((Committed<br/>Uncommitted<br/>Text))

    %% ─── Main ASR path ────────────────────────────────
    A --> B --> C --> D --> E --> F --> G --> R --> OUT

    %% ─── Speaker-ID side path ─────────────────────────
    C -. waveform .-> S -. cosine sim > θ .-> R

    %% ─── Styling (optional) ───────────────────────────
    classDef stage fill:#eaf8ff,stroke:#333;
    class B,C,D,E,F,G,S,R stage;
    style A fill:#f5fbff,stroke:#333;
    style OUT fill:#fffadc,stroke:#333;



```mermaid


flowchart LR
    %% ─── NODES ───────────────────────────────────────
    A["Audio\nStreaming"]
    B["Silero VAD\n(Voice Activity Detection)"]
    C["Audio\nSegments"]
    D["Whisper\nLanguage ID"]
    E["Per-language\nAlignment"]
    F["Whisper v3 Turbo\nfaster-whisper fp16"]
    G["Stable-Prefix\nLA-n  +  τ"]
    S["Speaker Embedding\nECAPA / TDNN / ResNet293"]
    R["Re-align Text"]
    OUT(["Committed\n&\nUncommitted\nText"])

    %% ─── MAIN ASR PATH ───────────────────────────────
    A --> B --> C --> D --> E --> F --> G --> R --> OUT

    %% ─── SPEAKER-ID SIDE PATH ───────────────────────
    C -. waveform .-> S -. "cosine sim > θ" .-> R

    %% ─── STYLING (tuỳ chọn) ─────────────────────────
    classDef stage fill:#E8F4FF,stroke:#333,stroke-width:1px;
    class B,C,D,E,F,G,S,R stage;
    style A fill:#F0FAFF,stroke:#333,stroke-width:1px;
    style OUT fill:#FFF7C7,stroke:#333,stroke-width:2px;


