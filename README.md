# QuickNotebookAI
# Daily-AI-Papers---VN-

```mermaid

flowchart TD
    %% ===== INPUT =========================================================
    A([Audio stream\n30-s sliding window])

    %% ===== PRE-PROCESSING ================================================
    subgraph PRE["Pre-processing"]
        direction TB
        B[VAD\nSilero]
        C[[Audio segments]]
        D[Language ID\nWhisper]
        E[Alignment\n(per language)]
    end

    %% ===== ASR ENGINE =====================================================
    subgraph ASR["ASR engine"]
        F[Whisper v3 Turbo\n(faster-whisper fp16)]
    end

    %% ===== POST-PROCESSING ===============================================
    subgraph POST["Post-processing"]
        direction TB
        G[Stable-Prefix\n(Local Agreement + τ)]
        R[Re-align text]
        OUT(["Committed\n&\nUncommitted\ntext"])
    end

    %% ===== SPEAKER-ID PATH ===============================================
    subgraph SPK["Speaker ID"]
        S[Speaker embedding\n(ECAPA / TDNN / ResNet293)]
    end

    %% ===== MAIN FLOW ======================================================
    A --> B --> C --> D --> E --> F --> G --> R --> OUT

    %% ===== SPEAKER FLOW ====================================================
    C -. waveform .-> S
    S -. "cosine sim > θ" .-> R

    %% ===== STYLE ===========================================================
    classDef input       fill:#FFFFFF,stroke:#333,stroke-width:2px;
    classDef preprocessing fill:#FFF9E6,stroke:#333,stroke-width:1px;
    classDef asr         fill:#E6F3FF,stroke:#333,stroke-width:1px;
    classDef post        fill:#FDE6F2,stroke:#333,stroke-width:1px;
    classDef speaker     fill:#EDEBFF,stroke:#333,stroke-width:1px;

    class A input;
    class B,C,D,E preprocessing;
    class F asr;
    class G,R,OUT post;
    class S speaker;


```
