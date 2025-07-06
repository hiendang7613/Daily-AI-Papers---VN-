# QuickNotebookAI
# Daily-AI-Papers---VN-

```mermaid
graph TD
  %% ===== Backbone & feature pyramid =====
  subgraph Backbone
    A0["ResNet / Swin / ViT (+ FPN)"] -->|multi-scale {P3,P4,P5}| P5
  end

  subgraph PixelDecoder["MaskDINOEncoder (Conv + FPN)"]
    P5 --> PD
  end

  %% ===== Two-stage proposal =====
  PD --> RP["Two-stage Proposal<br/>(300 anchors)"]
  RP -->|top 100| Q0["Object Queries (100)"]

  %% ===== Transformer Decoder with MSDeformAttn x6 =====
  subgraph Decoder["Transformer Decoder × 6 layers"]
    Q0 -->|Self-Attn (MHA)| Q0
    Q0 -- "Cross-Attn<br/>**MSDeformAttn**" --> PD
    Q0 -->|FFN| Q0
  end

  %% ===== Heads =====
  Q0 --> CL["Class / Box"]
  Q0 --> ME["Mask Embedding"]
  PD --> ML["Mask Logits"]
  ME --> ML
  ML --> Mout["Instance / Semantic Masks"]
  CL --> Bout["Boxes & Scores"]

  class PD,Mout,Bout stroke-width:2,stroke-dasharray: 3 3     %% highlight outputs


```



