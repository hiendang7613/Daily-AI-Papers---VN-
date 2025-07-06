# QuickNotebookAI
# Daily-AI-Papers---VN-

```mermaid
graph TD
  %% Backbone & FPN
  subgraph Backbone
    A0["ResNet / Swin / ViT (+ FPN)"] -->|"multi-scale P3-P5"| P5
  end

  %% Pixel decoder (Conv + FPN)
  subgraph PixelDecoder
    P5 --> PD["Pixel Decoder"]
  end

  %% Two-stage proposals
  PD --> RP["Two-stage Proposal (300 anchors)"]
  RP -->|"top 100"| Q0["Object Queries"]

  %% Transformer decoder với MSDeformAttn
  subgraph Decoder["Transformer Decoder (6 layers)"]
    Q0 -->|"Self-Attn"| Q0
    Q0 -- "Cross-Attn MSDeformAttn" --> PD
    Q0 -->|"FFN"| Q0
  end

  %% Heads
  Q0 --> CL["Class / Box"]
  Q0 --> ME["Mask Embedding"]
  PD --> ML["Mask Logits"]
  ME --> ML
  ML --> Mout["Instance & Semantic Masks"]
  CL --> Bout["Boxes & Scores"]


