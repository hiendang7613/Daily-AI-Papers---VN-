# QuickNotebookAI
# Daily-AI-Papers---VN-

```mermaid


flowchart LR
    %% ─── Input ─────────────────────────────────────────────────────
    subgraph LIVE_AUDIO["Audio Stream (30&nbsp;s context)"]
        A[AUDIO<br/>Streaming]
    end

    %% ─── Main ASR path ─────────────────────────────────────────────
    A --> B[VAD<br/>Silero]
    B --> C[[&lt;audio&nbsp;segments&gt;]]

    C --> D[Language&nbsp;ID<br/>Whisper]
    D --> E[Per-language<br/>Alignment]
    E --> F[Whisper&nbsp;v3&nbsp;Turbo<br/>(faster-whisper&nbsp;fp16)]
    F --> G[Stable-Prefix<br/>LA-n&nbsp;+&nbsp;τ]

    %% ─── Speaker-ID side path ─────────────────────────────────────
    C -- waveform --> S[Speaker&nbsp;Embedding<br/>(ECAPA&nbsp;/&nbsp;TDNN&nbsp;/&nbsp;ResNet293)]
    S -- cosine&nbsp;sim&nbsp;&gt;&nbsp;θ --> R[Re-align&nbsp;Text]

    %% ─── Merge & output ────────────────────────────────────────────
    G --> R
    R --> OUT((COMMITTED<br/>UNCOMMITTED<br/>Text))

    %% ─── Simple styling (optional) ────────────────────────────────
    classDef stage fill:#eaf8ff,stroke:#333;
    class B,C,D,E,F,G,S,R stage;
    style A fill:#f5fbff,stroke:#333;
    style OUT fill:#fffadc,stroke:#333;

