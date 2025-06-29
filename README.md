# QuickNotebookAI
# Daily-AI-Papers---VN-


flowchart LR
    %% ───── Input ──────────────────────────────────────────────────────────
    subgraph LIVE_AUDIO["Audio Stream (30 s context)"]
        A[AUDIO<br/>Streaming]
    end

    %% ───── Main path ──────────────────────────────────────────────────────
    A --> B[VAD<br/>Silero]
    B --> C[[<audio&nbsp;segments>]]

    C --> D[Language&nbsp;ID<br/>Whisper]
    D --> E[Per-language<br/>Alignment]
    E --> F[Whisper v3 Turbo<br/>(faster-whisper fp16)]
    F --> G[Stable-Prefix<br/>LA-n + τ]

    %% ───── Speaker-ID side path ──────────────────────────────────────────
    C -- waveform --> S[Speaker&nbsp;Embedding<br/>(ECAPA / TDNN / ResNet293)]
    S -- cosine&nbsp;sim > θ --> R[Re-align&nbsp;Text]

    %% ───── Merge & output ────────────────────────────────────────────────
    G --> R
    R --> OUT((COMMITTED<br/>UNCOMMITTED<br/>Text))

    %% ───── Styling ───────────────────────────────────────────────────────
    classDef stage fill:#e8f9ff,stroke:#333,stroke-width:1px;
    class B,C,D,E,F,G,S,R stage;
    style A fill:#f0f8ff,stroke:#333;
    style OUT fill:#fffec9,stroke:#333,stroke-width:2px;

    %% Optional note for latency
    linkStyle 6 stroke-width:2px,stroke-dasharray: 5 3
    class OUT end;
