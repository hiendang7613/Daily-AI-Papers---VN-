1.  **TOPIC_TREE**

    *   Natural Language Processing (NLP)
        *   Large Language Models (LLMs)
            *   Model Architecture & Efficiency
                *   Efficient Architectures
                    *   Sparse Attention Mechanisms
                        *   2404.14219 | Đề xuất blocksparse attention xen kẽ dense attention trong phi-3-small để giảm KV cache và tăng tốc.
                    *   Layer-wise Scaling
                        *   2404.14619 | Áp dụng layer-wise scaling (phân bổ tham số không đồng đều giữa các lớp) trong OpenELM.
                    *   Hybrid Transformer-SSM Models
                        *   2403.19887 | Đề xuất Jamba, kiến trúc lai Transformer, Mamba (SSM) và MoE với cấu trúc lồng ghép có thể cấu hình.
                        *   2405.16712 | Đề xuất Zamba, kiến trúc lai Mamba với khối self-attention và MLP duy nhất được chia sẻ và tái sử dụng.
                    *   RWKV Variants
                        *   2404.05892 | Đề xuất Eagle (trạng thái ẩn ma trận đa đầu) và Finch (hồi quy động, token-shift phụ thuộc dữ liệu) cho RWKV.
                    *   Recurrent Neural Networks (Gated Linear RNNs)
                        *   2404.07904 | Đề xuất HGRN2 với mở rộng trạng thái dựa trên tích ngoài không tham số và biến thể đa đầu.
                *   Mixture-of-Experts (MoE)
                    *   Architecture and Routing
                        *   2404.15045 | Đề xuất MH-MoE, cơ chế đa đầu tách token thành sub-token, định tuyến độc lập đến expert.
                        *   2405.17976 | Đề xuất Attention Router cho MoE, xem xét tương quan expert qua cơ chế Attention.
                    *   Efficient Architectures with MoE
                        *   2404.07413 | Đề xuất JetMoE, áp dụng SMoE cho cả lớp attention (MoA) và FFN, chia sẻ K/V trong MoA.
                *   Long Context Modeling
                    *   Efficient Transformers
                        *   2404.07143 | Đề xuất Infini-attention, kết hợp bộ nhớ nén dài hạn (attention tuyến tính) và attention cục bộ trong một khối Transformer.
                    *   Memory-Augmented LLMs
                        *   2404.09173 | Đề xuất TransformerFAM, cơ chế bộ nhớ phản hồi (FAM) dạng trạng thái ẩn cập nhật block-wise trong mỗi lớp Transformer.
                *   Inference Optimization
                    *   KV Cache Optimization
                        *   2404.14469 | Đề xuất SnapKV, nén KV cache của prompt dài trước khi sinh token dựa trên mẫu chú ý từ cửa sổ quan sát.
                        *   2405.12981 | Đề xuất Layer-Condensed KV Cache, query ở các lớp chỉ tương tác với K/V lớp trên cùng.
                    *   Speculative Decoding
                        *   2404.18911 | Đề xuất Kangaroo, giải mã tự suy đoán dựa trên thoát sớm kép (adapter nhẹ tạo mô hình nháp, dừng tạo nháp theo độ tin cậy).
                        *   2403.09919 | Đề xuất ReDrafter, RNN làm mô hình nháp, dynamic tree attention loại bỏ tiền tố trùng lặp, KD cục bộ.
                        *   2405.00263 | Đề xuất Clover, giải mã suy đoán tích hợp kiến thức tuần tự (Kết nối Hồi quy, Bộ giải mã Chú ý).
                    *   Early Exit and Dynamic Compute
                        *   2404.02258 | Đề xuất MoD (Mixture-of-Depths), phân bổ tính toán động cấp token bằng định tuyến expert-choice top-k, duy trì FLOPs tĩnh.
                        *   2404.16710 | Đề xuất LayerSkip, kết hợp layer dropout và early exit loss (đầu LM chia sẻ) cho huấn luyện, và giải mã tự suy đoán.
            *   Training Strategies & Methodologies
                *   Efficient Training Strategies
                    *   2404.08634 | Đề xuất Inheritune, kế thừa lớp đầu từ mô hình lớn, huấn luyện lại và mở rộng dần để tạo SLM hiệu quả (tránh "lớp lười").
                *   Pre-training Strategies
                    *   Selective Token Training
                        *   2404.07965 | Đề xuất SLM (Selective Language Modeling), huấn luyện LLM tập trung vào token hữu ích (excess loss cao) dựa trên RM.
                    *   Scaling and Optimization
                        *   2404.06395 | Đề xuất WSD LRS (bộ lập lịch LR 3 giai đoạn) và MWTE (thử nghiệm trên SLM) cho huấn luyện LLM.
                    *   Multi-Token Prediction
                        *   2404.19737 | Đề xuất mục tiêu dự đoán đa token (n đầu ra độc lập, unembedding chung) và tối ưu bộ nhớ huấn luyện.
                *   Continual Pretraining
                    *   Multilingual Adaptation and Safety Alignment
                        *   2404.00399 | Đề xuất quy trình học liên tục CAP (đa ngôn ngữ, code) và CAT (căn chỉnh an toàn với Biden-Harris Redteam Dataset) cho AURORA-M.
                *   Long Context Handling
                    *   Instruction Tuning for Long Context
                        *   2404.16811 | Đề xuất IN2 training, tạo dữ liệu QA tổng hợp (thông tin đặt ngẫu nhiên trong ngữ cảnh dài) để chống "lost-in-the-middle".
                    *   Offline Context Learning
                        *   2404.07979 | Đề xuất LLoCO, kết hợp nén ngữ cảnh (AutoCompressor) với PEFT (LoRA) trên miền dữ liệu cụ thể để học ngữ cảnh dài offline.
            *   Alignment & Preference Learning
                *   Reinforcement Learning from Human Feedback (RLHF)
                    *   Scalable Frameworks and Systems
                        *   2405.01481 | Giới thiệu NeMo-Aligner, toolkit RLHF/DPO/SPIN hiệu năng cao, tích hợp Megatron-LM, PyTriton, TensorRT-LLM.
                    *   Online Iterative RLHF
                        *   2405.07863 | Trình bày quy trình Online Iterative RLHF (dùng proxy model, DPO) cho cộng đồng nguồn mở.
                *   Alignment for Reasoning Tasks
                    *   2404.02078 | Giới thiệu ULTRAINTERACT (dữ liệu cây ưu tiên cho suy luận) và hàm mục tiêu RM mới (L_BT + L_DR).
                    *   2404.19733 | Đề xuất Iterative RPO (DPO+NLL trên cặp sở thích từ CoT) để cải thiện suy luận LLM.
                *   Offline Preference Optimization Strategies
                    *   2404.09656 | Đề xuất TR Alignment (TR-DPO, TR-IPO, TR-KTO) cập nhật động mô hình tham chiếu để giảm quá tối ưu hóa.
                *   General Preference Optimization
                    *   2404.03715 | Đề xuất DNO (Direct Nash Optimization), tối ưu LLM dựa trên hàm ưu tiên tổng quát P(y ≻ y'|x) với mục tiêu hồi quy đối nghịch.
                *   Self-Play Methods
                    *   2405.00675 | Đề xuất SPPO (Self-Play Preference Optimization), tinh chỉnh LLM dựa trên xấp xỉ cập nhật trọng số mũ (cân bằng Nash).
                *   Factuality-Aware Alignment
                    *   2405.01525 | Đề xuất FLAME (SFT/DPO nhận biết tính xác thực) với dữ liệu tự sinh (SFT) và RM xác thực (DPO).
            *   Representation Learning
                *   Text Embeddings
                    *   Training Strategies
                        *   2405.06932 | Đề xuất Piccolo2, chiến lược multi-task hybrid loss (InfoNCE, Cosent, InfoNCE label negatives) cho embedding.
            *   Reasoning in Language Models
                *   Algorithmic Reasoning
                    *   Positional Encoding for Algorithmic Reasoning
                        *   2405.17399 | Đề xuất Abacus Embeddings (mã hóa vị trí chữ số tương đối trong số) cho bài toán số học.
                *   Self-Improving LLMs via Search and Reinforcement Learning
                    *   2404.12253 | Đề xuất ALPHA LLM, tích hợp ηMCTS (tìm kiếm mức tùy chọn, phân nhánh thích ứng) và bộ ba phê bình (value, PRM, ORM) cho tự cải thiện.
                *   Learning to Search
                    *   2404.03683 | Đề xuất SoS (Stream of Search), biểu diễn quá trình tìm kiếm (khám phá, quay lui, lỗi) thành chuỗi văn bản để huấn luyện LM.
            *   Function Calling
                *   On-Device Function Calling Optimization
                    *   2404.01744 | Đề xuất functional tokens và quy trình fine-tuning/suy luận hiệu quả cho Octopus v2.
            *   Automated Workflow Generation and Grounding
                *   2404.13050 | Đề xuất FlowMind, tạo quy trình tự động bằng LLM với "lecture recipe" (API đáng tin cậy) và phản hồi người dùng tương tác.
            *   Security & Safety
                *   Adversarial Attacks & Defenses
                    *   Automated Red Teaming / Jailbreaking
                        *   2404.16873 | Đề xuất AdvPrompter (LLM tạo hậu tố đối kháng thích ứng) và AdvPrompterTrain (huấn luyện xen kẽ không gradient).
                    *   Instruction Hierarchy Training
                        *   2404.13208 | Đề xuất phân cấp lệnh (System > User > Tool) và tạo dữ liệu (Context Synthesis/Ignorance) để chống ghi đè chỉ dẫn.
            *   Evaluation
                *   Long Context In-Context Learning
                    *   2404.02060 | Giới thiệu LongICLBench, benchmark ICL dài trên phân loại nhãn cực trị.
                *   LLM-as-a-judge Methodologies
                    *   2404.18796 | Đề xuất PoLL (Panel of LLM evaluators), sử dụng hội đồng LLM đánh giá đa dạng thay vì một mô hình lớn.
                *   Robustness and Generalization
                    *   Benchmark Contamination Analysis
                        *   2405.00332 | Xây dựng GSM1k (benchmark toán song song GSM8k) và phân tích overfitting LLM.
            *   Tokenization and Compression
                *   2404.03626 | Đề xuất Equal-Info Windows, chia văn bản theo lượng thông tin nén (AC) cố định và reset trạng thái để LLM học trên bitstream.
            *   Theoretical Analysis of Language Models
                *   Representational Capacity of Transformer Models
                    *   2404.14994 | Chứng minh Transformer (hard/sparse attention) biểu diễn chính xác n-gram LM, phân tích năng lực biểu diễn xác suất.
            *   Parameter-Efficient Fine-Tuning (PEFT)
                *   Representation Finetuning
                    *   2404.03592 | Đề xuất ReFT (Representation Finetuning) và LoReFT (can thiệp biểu diễn ẩn không gian con tuyến tính hạng thấp).
                *   High-Rank Adaptation
                    *   2405.12130 | Đề xuất MoRA, cập nhật trọng số hạng cao (ma trận vuông M) với toán tử không tham số.
                *   Adapter Composition and Routing
                    *   Zero-shot Adapter Routing
                        *   2405.11157 | Đề xuất MBC (phân cụm LoRA theo tương đồng tham số) và Arrow Routing (định tuyến zero-shot dựa trên SVD của LoRA).
                *   Empirical Evaluation & Deployment
                    *   2405.00732 | Đánh giá LoRA 4-bit trên 10 mô hình/31 tác vụ, đo lường hiệu năng LoRAX.
                *   Analysis of LoRA
                    *   2405.09673 | So sánh LoRA và FFT (CPT, IFT), phân tích quên kiến thức và hạng ma trận thay đổi trọng số.
                *   Data-Free PEFT Transfer
                    *   2405.17258 | Đề xuất Trans-LoRA, truyền PEFT giữa mô hình bằng KD (dữ liệu tổng hợp lọc bởi discriminator).
            *   Model Compression and Optimization
                *   Pruning and Regularization
                    *   2405.12250 | Đề xuất regularization Lcosine (giảm tuyến tính biểu diễn) và tỉa lớp (xấp xỉ tuyến tính + chưng cất layer-wise).
            *   Multilingual LLMs
                *   Novel Architectures
                    *   2405.06694 | Đề xuất SUTRA, tách biệt học khái niệm (lõi MoE) và ngôn ngữ (encoder/decoder NMT).
                *   Model Development and Evaluation
                    *   2405.15032 | Giới thiệu Aya 23 (8B, 35B), mô hình đa ngôn ngữ (23 ngôn ngữ) dựa trên Command, tập trung "chiều sâu".
            *   Model Training & Data
                *   Transparent and Reproducible LLMs
                    *   2405.19327 | Giới thiệu MAP-Neo (7B song ngữ Anh-Trung), NEO Scaling Law, pipeline Matrix Data Pile, huấn luyện 2 giai đoạn.
    *   Computer Vision (CV)
        *   Generative Models
            *   Image Generation
                *   Text-to-Image Synthesis
                    *   Alignment and Controllability in Diffusion Models
                        *   2404.03653 | Đề xuất CoMat, tinh chỉnh T2I bằng đối sánh khái niệm I2T (Kích hoạt Khái niệm, Bảo toàn Độ trung thực, Tập trung Thuộc tính).
                    *   Controllable and Compositional Generation
                        *   2405.08246 | Đề xuất BlobGEN, biểu diễn blob dày đặc (tham số elip + mô tả) và masked cross-attention cho kiểm soát cục bộ.
                *   Face Editing
                    *   Diffusion-based Face Reenactment and Swapping
                        *   2405.12970 | Đề xuất Face-Adapter (SCG dự đoán landmark 3D/mask, IE mã hóa ID, AC điều khiển thuộc tính) cho tái hiện/hoán đổi khuôn mặt.
            *   Video Generation
                *   Controllable Human Video Generation
                    *   360-Degree Human Video Generation with Diffusion Models
                        *   2405.17405 | Đề xuất Human4DiT, kiến trúc 4D diffusion transformer phân cấp (ảnh 2D, thời gian, góc nhìn) cho video người 360 độ.
                *   Metamorphic Time-lapse Video Generation
                    *   2404.05014 | Đề xuất MagicTime, Magic Adaptive Strategy (adapter không gian/thời gian), Dynamic Frames Extraction, Magic Text-Encoder cho video biến đổi.
            *   3D Generation
                *   Controllable and Interactive Generation
                    *   2405.16510 | Đề xuất Interactive3D, kết hợp GS (kéo thả 3D) và InstantNGP (tinh chỉnh băm tương tác) cho tạo sinh 3D có kiểm soát.
                *   Text-to-3D Generation
                    *   Score Distillation Methods
                        *   2405.11252 | Đề xuất Trajectory Score Matching (TSM) và pixel-by-pixel gradient clipping cho 3DGS với SDXL.
            *   Diffusion Models
                *   Consistency Models
                    *   Phased Consistency Models
                        *   2405.18407 | Đề xuất PCM, chia quỹ đạo ODE thành pha, thực thi nhất quán trong pha, cho phép lấy mẫu xác định đa bước.
                *   Latent Diffusion Models
                    *   Efficient Autoencoder Architectures
                        *   2405.14477 | Đề xuất LiteVAE, DWT đa cấp độ + UNet nhẹ cho encoder, SMC (tích chập tự điều biến) cho decoder.
        *   Video Understanding
            *   Efficient Video Architectures
                *   Mobile Video Recognition
                    *   2405.08344 | Đề xuất SqueezeTime, nén chiều thời gian vào kênh, Khối Học Kênh-Thời gian (CTL) với TFC và IOI.
        *   3D Reconstruction
            *   Neural Reconstruction
                *   Large Reconstruction Models
                    *   2404.12385 | Đề xuất MeshLRM, tích hợp trích xuất/render lưới khả vi (DiffMC, Nvdiffrast) vào LRM, loss độ mờ tia.
            *   Dynamic Scene Reconstruction
                *   Monocular 4D Reconstruction with Unknown Poses
                    *   2405.18426 | Đề xuất GFlow, phục dựng cảnh 4D (3DGS) và ước tính tư thế camera đồng thời từ video đơn mắt (khởi tạo/làm dày đặc Gaussian dựa trên prior, tối ưu xen kẽ).
            *   Feed-forward Reconstruction from Sparse Views
                *   2404.19702 | Đề xuất GS-LRM, Transformer dự đoán trực tiếp tham số 3DGS cho mỗi pixel từ ảnh sparse-view.
        *   Human Motion Synthesis
            *   Controllable Text-to-Motion Generation
                *   Latent Consistency Models
                    *   2404.19759 | Đề xuất MotionLCM, áp dụng LCD cho MLD, tích hợp motion ControlNet (giám sát kép) cho điều khiển chuyển động thời gian thực.
        *   Evaluation & Benchmarking
            *   Visual Code Generation from Plots
                *   2405.07990 | Giới thiệu Plot2Code benchmark và phương pháp đánh giá (GPT-4V Judgement, Text-Match Ratio).
    *   Multimodal AI
        *   Foundation Models
            *   Mixed-Modal Generation and Understanding
                *   Early-Fusion Token-Based Models
                    *   2405.09818 | Đề xuất Chameleon, MLLM early-fusion (token ảnh/văn bản xen kẽ) với QK-Norm để ổn định huấn luyện.
            *   In-Context Learning
                *   Many-Shot Learning & Evaluation
                    *   2405.09798 | Khảo sát many-shot ICL trên MLLM (GPT-4o, Gemini 1.5 Pro), phân tích hiệu quả dữ liệu và batching.
        *   Vision Language Models (MLLMs / VLMs)
            *   Efficient Vision Language Models
                *   2405.09215 | Xây dựng Xmodel-VLM (LLM 1.1B + CLIP ViT-L + projector MLP) theo kiến trúc LLaVA-1.5.
            *   Vision-Language Connection Architectures
                *   Multi-Layer Feature Integration
                    *   2405.13800 | Đề xuất Dense Connector (STI, SCI, DCI) tích hợp đặc trưng thị giác đa lớp từ encoder đông lạnh vào LLM.
            *   High-Resolution Image Understanding
                *   Dynamic Patching Strategies
                    *   2404.06512 | Đề xuất VisionLLM-H, Dynamic Image Partition (patch động theo tỷ lệ khung hình) và token newline cho ảnh 4K HD.
        *   Video Understanding
            *   Multimodal Human Behavior Understanding
                *   Joint Video-Motion Language Modeling
                    *   2405.20204 | Đề xuất MotionLLM, mô hình hóa đồng thời video và motion (bộ dịch V-L riêng, huấn luyện hợp nhất 2 giai đoạn).
        *   Image and 3D Object Captioning
            *   Fact-Checking and Hallucination Reduction
                *   2404.19752 | Đề xuất VFC (VisualFactChecker), quy trình không cần huấn luyện tạo chú thích (đề xuất, xác minh bằng công cụ, tạo) và CLIP-Image-Score.
    *   Machine Learning (ML)
        *   Deep Learning
            *   Model Interpretability
                *   Mechanistic Interpretability of Transformers
                    *   2405.15071 | Phân tích grokking trong suy luận ngầm của Transformer (ghép nối, so sánh), vai trò của phân phối dữ liệu.
            *   Network Architectures
                *   Kolmogorov-Arnold Networks
                    *   2404.19756 | Đề xuất KAN, hàm kích hoạt học được trên cạnh (B-spline), cập nhật/mở rộng lưới động.
                *   Efficient Architectures & Components
                    *   2405.11582 | Đề xuất PRepBN (BatchNorm tái tham số hóa chuyển đổi dần từ LN) và SLA (ReLU kernel + DWC).
        *   Optimization
            *   Stochastic Gradient Methods
                *   Averaging Techniques
                    *   2405.15682 | Đề xuất Schedule-Free optimization (gradient nội suy yt, cập nhật zt, xt) và định lý online-to-batch mới.
        *   Self-Supervised Learning
            *   Data Curation
                *   Clustering-based Curation
                    *   2405.15613 | Đề xuất k-means phân cấp lặp lại trên tâm cụm và lấy mẫu cân bằng phân cấp để tinh lọc dữ liệu SSL.
    *   AI Systems
        *   Model Deployment & Optimization
            *   Inference Engines
                *   (Covered under NLP > LLMs > Inference Optimization)
        *   Computer Architecture
            *   Domain-Specific Accelerators
                *   AI/ML Accelerators
                    *   Dataflow Architectures
                        *   2405.07518 | Giới thiệu SN40L RDU, bộ tăng tốc luồng dữ liệu (PCU, PMU, RDN) với bộ nhớ 3 cấp (SRAM, HBM, DDR) cho CoE.
    *   Distributed Machine Learning
        *   Parallel and Distributed Training
            *   Pipeline Parallelism
                *   Optimization Techniques for Pipeline Parallelism
                    *   2405.18047 | Đề xuất 2BP (2-Stage Backpropagation) tách backward-p1 (gradient theo input) và backward-p2 (gradient theo tham số).
    *   Robotics
        *   Robotic Learning
            *   Manipulation
                *   Generalist Robot Policies
                    *   2405.12213 | Giới thiệu Octo, chính sách robot tổng quát (Transformer, attention theo khối, token readout) cho phép tinh chỉnh hiệu quả.
    *   Reinforcement Learning
        *   Model-Based Reinforcement Learning
            *   World Models
                *   Diffusion-Based World Models
                    *   2405.12399 | Đề xuất DIAMOND, tác nhân RL huấn luyện trong mô hình thế giới dựa trên khuếch tán EDM.
    *   Other
        *   (Papers that are primarily analyses or surveys without a primary novel technical contribution fitting elsewhere)
            *   2405.08707 | Phân tích lý thuyết Transformer như mạng Hopfield, hàm năng lượng toàn cục qua MM.
            *   2405.18870 | Đánh giá khả năng ToM bậc cao (2-6) của LLM bằng bộ dữ liệu MoToMQA mới.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                          | Đột phá                                                                                                                            | Ảnh hưởng                                                                                                                                     |
    | :--- | :-------- | :------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2404.19756 | Neural Networks, KAN, Kolmogorov-Arnold, Learnable Activation | KAN: Kiến trúc mạng nơ-ron mới với hàm kích hoạt có thể học được trên cạnh, dựa trên định lý Kolmogorov-Arnold, vượt trội MLP.        | Mở ra một hướng đi hoàn toàn mới cho thiết kế mạng nơ-ron, có tiềm năng thay thế MLP với khả năng biểu diễn tốt hơn và dễ diễn giải hơn.       |
    | 2    | 2405.09818 | Multimodal, Early-Fusion, Chameleon, QK-Norm, Token-based | Chameleon: MLLM early-fusion dựa trên token, xử lý/tạo ảnh-văn bản xen kẽ tùy ý bằng một Transformer duy nhất, ổn định với QK-Norm. | Cung cấp một kiến trúc đa phương thức thống nhất, linh hoạt, có khả năng mở rộng quy mô lớn và xử lý các tác vụ đa phương thức phức tạp.      |
    | 3    | 2404.19737 | LLM, Multi-Token Prediction, Self-Speculative Decoding  | Huấn luyện LLM dự đoán song song n token tương lai, tăng tốc suy luận 3 lần qua self-speculative decoding không cần mô hình nháp.     | Cải thiện hiệu quả sử dụng dữ liệu huấn luyện và tăng tốc độ suy luận LLM đáng kể mà không tăng chi phí huấn luyện.                            |
    | 4    | 2405.10300 | Text-to-Music, Long-form, Diffusion Transformer, Latent Diffusion | Tạo nhạc dài (4m45s) chất lượng cao, có cấu trúc, từ văn bản trong một lần xử lý bằng DiT trên latent rate thấp, không cần mã ngữ nghĩa. | Đẩy xa giới hạn tạo nhạc dài có cấu trúc, mở ra khả năng ứng dụng trong sáng tác và sản xuất âm nhạc chuyên nghiệp.                             |
    | 5    | 2405.01535 | LLM Evaluation, LM-as-a-judge, Weight Merging, Prometheus 2 | Prometheus 2: Mô hình đánh giá LLM hợp nhất (DA & PR) mã nguồn mở, đạt SOTA bằng trộn trọng số, hỗ trợ tiêu chí tùy chỉnh.         | Cung cấp công cụ đánh giá LLM mạnh mẽ, linh hoạt, mã nguồn mở, thu hẹp khoảng cách với các mô hình độc quyền.                                 |
    | 6    | 2405.18407 | Diffusion Models, Consistency Models, Phased Consistency, PCM | PCM: Cho phép lấy mẫu xác định đa bước cho CMs bằng cách thực thi nhất quán theo pha, cải thiện chất lượng và kiểm soát.             | Giải quyết các hạn chế của LCMs, tăng tính nhất quán, khả năng kiểm soát và chất lượng ảnh ở chế độ ít bước.                                  |
    | 7    | 2405.15682 | Optimization, Schedule-Free, Online-to-Batch, Momentum  | Phương pháp tối ưu Schedule-Free không cần lịch trình LR, hội tụ tối ưu cho lồi Lipschitz, dựa trên định lý online-to-batch mới.   | Loại bỏ sự phụ thuộc vào tổng số bước huấn luyện, đơn giản hóa quá trình tối ưu và có thể cải thiện hiệu năng thực nghiệm.                     |
    | 8    | 2405.14906 | Code LLM, Instruction Tuning, AIEV-INSTRUCT, AutoCoder  | AIEV-INSTRUCT: Tạo dữ liệu code chất lượng cao (tương tác agent, unit test, phản hồi thực thi) và AutoCoder tự cài đặt thư viện. | Cải thiện đáng kể chất lượng và độ tin cậy của dữ liệu huấn luyện code, tạo ra các Code Interpreter mạnh mẽ hơn.                               |
    | 9    | 2405.07518 | AI Accelerator, Dataflow, CoE, SambaNova SN40L RDU      | SN40L RDU: Bộ tăng tốc luồng dữ liệu với bộ nhớ 3 cấp (SRAM, HBM, DDR) cho CoE, hợp nhất toán tử sâu, vượt trội GPU.                 | Cung cấp giải pháp phần cứng chuyên dụng hiệu năng cao cho việc triển khai các hệ thống AI dựa trên nhiều mô hình chuyên biệt.                 |
    | 10   | 2405.00675 | LLM Alignment, Self-Play, Preference Optimization, SPPO | SPPO: Tinh chỉnh LLM dựa trên xấp xỉ cập nhật trọng số mũ (cân bằng Nash), vượt trội DPO/RLHF không cần dữ liệu GPT-4.             | Cung cấp phương pháp alignment hiệu quả, không cần dữ liệu giám sát từ mô hình mạnh hơn, có nền tảng lý thuyết vững chắc.                     |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2405.12250 – Regularization Lcosine giảm độ tuyến tính biểu diễn giữa các lớp liên tiếp trong pretraining – Suy nghĩ:** Hướng tiếp cận phản trực giác nhưng thú vị, nếu cơ chế thực sự như giả thuyết (buộc mô hình tăng cường phi tuyến ở nơi khác) thì rất tiềm năng.
    *   **2405.09818 – QK-Norm (LayerNorm trên Q, K trước softmax) để ổn định huấn luyện MLLM early-fusion quy mô lớn – Suy nghĩ:** Giải pháp kiến trúc đơn giản nhưng hiệu quả cho vấn đề bất ổn định cụ thể trong MLLM early-fusion, rất thực tế.
    *   **2404.19756 – KAN: Hàm kích hoạt học được (B-spline) trên cạnh, cập nhật/mở rộng lưới động – Suy nghĩ:** Kiến trúc mạng nơ-ron hoàn toàn mới, có nền tảng lý thuyết mạnh, tiềm năng lớn cho các mô hình chính xác và dễ diễn giải hơn MLP.
    *   **2405.17399 – Abacus Embeddings: mã hóa vị trí chữ số tương đối trong số (offset ngẫu nhiên khi huấn luyện) – Suy nghĩ:** Giải pháp embedding vị trí chuyên biệt và hiệu quả cho các tác vụ số học, giải quyết vấn đề căn chỉnh chữ số một cách trực quan.
    *   **2405.12130 – MoRA: cập nhật trọng số hạng cao (ma trận vuông M) với toán tử không tham số (f_comp, f_decomp) – Suy nghĩ:** Cải tiến PEFT thông minh, tăng hạng cập nhật mà vẫn giữ số tham số và khả năng tích hợp, giải quyết hạn chế của LoRA.
    *   **2405.19327 – NEO Scaling Law: mô hình hóa loss (N, D, C) với L∞, A*N^-α, B*D^-β, G*C^-γ cho dữ liệu đa nguồn – Suy nghĩ:** Đề xuất luật co giãn mới, toàn diện hơn, hữu ích cho việc tối ưu huấn luyện LLM trên dữ liệu phức hợp.
    *   **2405.18407 – PCM: chia quỹ đạo ODE thành pha, thực thi nhất quán trong pha (tham số hóa dựa trên nghiệm PF-ODE) – Suy nghĩ:** Giải pháp mới lạ và có cơ sở lý thuyết cho consistency models, cho phép lấy mẫu xác định đa bước và cải thiện kiểm soát.
    *   **2405.15682 – Schedule-Free optimization: gradient nội suy yt = (1-β)zt + βxt, cập nhật zt, xt – Suy nghĩ:** Phương pháp tối ưu độc đáo, loại bỏ lịch trình LR, có nền tảng lý thuyết vững chắc và hiệu quả thực nghiệm.
    *   **2405.14906 – AIEV-INSTRUCT: tạo dữ liệu code (tương tác agent, unit test, phản hồi thực thi) và AutoCoder tự cài đặt thư viện – Suy nghĩ:** Quy trình tạo dữ liệu code chất lượng cao, tích hợp xác minh thực thi, rất thực tế.
    *   **2405.07518 – SN40L RDU: bộ tăng tốc luồng dữ liệu (PCU, PMU, RDN) với bộ nhớ 3 cấp (SRAM, HBM, DDR) cho CoE – Suy nghĩ:** Kiến trúc phần cứng đột phá, giải quyết "bức tường bộ nhớ" cho các hệ thống AI phức hợp.
    *   **2405.00675 – SPPO: hàm mục tiêu dựa trên xấp xỉ cập nhật trọng số mũ (tối ưu P(y ≻ πt | x)) – Suy nghĩ:** Phương pháp alignment mới lạ, có nền tảng lý thuyết trò chơi, không cần RM và hiệu quả hơn DPO/RLHF.
    *   **2405.19759 – MotionLCM: LCD cho MLD, motion ControlNet (giám sát kép latent/motion space) – Suy nghĩ:** Ứng dụng LCD cho sinh chuyển động và cơ chế giám sát kép cho ControlNet là những đóng góp kỹ thuật sáng tạo.
    *   **2405.18047 – 2BP (2-Stage Backpropagation): tách backward-p1 (gradient theo input) và backward-p2 (gradient theo tham số) – Suy nghĩ:** Kỹ thuật tối ưu hóa pipeline parallelism hợp lý, giảm thời gian chờ, dù tăng bộ nhớ.
    *   **2405.13865 – ReVideo: huấn luyện 3 giai đoạn (ưu tiên chuyển động, tách rời, khử khối) và SAFM (hợp nhất thích ứng không-thời gian) – Suy nghĩ:** Chiến lược huấn luyện và mô-đun hợp nhất thông minh để giải quyết khớp nối điều kiện trong chỉnh sửa video cục bộ.
    *   **2405.12970 – Face-Adapter: SCG (landmark 3D + mask vùng thay đổi), IE (transformer decoder + learnable queries), AC (điều khiển thuộc tính) – Suy nghĩ:** Kiến trúc adapter nhẹ, tách biệt điều khiển ID/cấu trúc/thuộc tính, SCG cung cấp hướng dẫn không gian chính xác.
    *   **2405.08054 – Coin3D: 3D adapter (2 UNet 3D xử lý proxy voxel và latent MV), proxy-bounded editing, progressive volume caching, volume-SDS – Suy nghĩ:** Khung làm việc tương tác 3D sáng tạo, 3D adapter và volume caching là những đóng góp kỹ thuật đáng chú ý.
    *   **2405.16712 – Zamba: lõi Mamba + khối GSA (self-attention + MLP chia sẻ, tái sử dụng) với đầu vào (xl + x0) và Linl riêng – Suy nghĩ:** Kiến trúc lai SSM-Transformer hiệu quả về bộ nhớ, cơ chế GSA chia sẻ là một ý tưởng mới.
    *   **2405.13800 – Dense Connector: STI, SCI, DCI tích hợp đặc trưng thị giác đa lớp từ encoder đông lạnh vào LLM – Suy nghĩ:** Ý tưởng đơn giản nhưng hiệu quả, khai thác tốt hơn thông tin từ encoder thị giác có sẵn.
    *   **2405.08748 – Hunyuan-DiT: DiT đa độ phân giải, bộ mã hóa văn bản kép (CLIP song ngữ + T5 đa ngôn ngữ), data convoy, MLLM tinh chỉnh caption – Suy nghĩ:** Hệ thống T2I hoàn chỉnh, kết hợp bộ mã hóa kép và tối ưu dữ liệu là những đóng góp thực tiễn.
    *   **2404.19752 – VFC: quy trình tạo chú thích (đề xuất đa mô hình, xác minh bằng Detector/VQA do LLM điều phối, tạo chú thích cuối) và CLIP-Image-Score – Suy nghĩ:** Quy trình không cần huấn luyện, giảm ảo giác hiệu quả bằng xác minh dựa trên công cụ.
    *   **2405.10637 – Layer-Condensed KV Cache: query ở các lớp chỉ tương tác với K/V lớp trên cùng, lớp khởi động, huấn luyện song song xấp xỉ – Suy nghĩ:** Phương pháp độc đáo giảm KV cache theo chiều sâu, giải quyết vấn đề bộ nhớ LLM.
    *   **2405.19332 – SELM: mục tiêu tối ưu hai cấp cho RM (thành phần lạc quan + KL) và mục tiêu SELM cho LLM (DPO + thành phần khám phá) – Suy nghĩ:** Tích hợp khám phá chủ động vào DPO một cách có nguyên tắc, không cần RM.
    *   **2405.18386 – Instruct-MusicGen: audio fusion (nhân bản self-attention + cross-attention sửa đổi + cổng) và text fusion (LoRA trên cross-attention T5) – Suy nghĩ:** Tích hợp hiệu quả điều kiện âm thanh/văn bản vào MusicGen bằng adapter/LoRA.
    *   **2405.09215 – Xmodel-VLM: projector XDP (MLP 2 lớp + Mish + downsampling 75%) – Suy nghĩ:** Mặc dù dựa trên LLaVA, việc thiết kế projector có downsampling là một điều chỉnh thực tế cho VLM nhẹ.
    *   **2405.01536 – Pair Customization: tối ưu hóa kết hợp 2 LoRA (content, style) với ràng buộc trực giao và Style Guidance khi suy luận – Suy nghĩ:** Phương pháp mới lạ học khác biệt phong cách từ cặp ảnh, giải quyết overfitting nội dung.
    *   **2405.00983 – Nhận dạng nhân vật dựa trên theo dõi không cần huấn luyện (embedding gốc + exemplar từ video) cho tạo AD – Suy nghĩ:** Giải pháp thực tế cho nhất quán nhân vật trong AD tự động, không cần dữ liệu huấn luyện chuyên biệt.
    *   **2405.18750 – T2V-Turbo: tích hợp RM (ảnh-văn bản, video-văn bản) vào CD, tối ưu trực tiếp trên mẫu sinh một bước – Suy nghĩ:** Cải thiện VCM hiệu quả, tránh lan truyền ngược qua chuỗi lấy mẫu dài, rất thực tế.
    *   **2405.17976 – Attention Router: tính điểm định tuyến MoE bằng self-attention (Q,K,V từ token đầu vào chiếu lên N expert) – Suy nghĩ:** Cơ chế định tuyến mới cho MoE, có tiềm năng mô hình hóa tương quan expert tốt hơn.
    *   **2405.07526 – MS MARCO Web Search: bộ dữ liệu web quy mô lớn (ClueWeb22 + 10M truy vấn Bing + click log) – Suy nghĩ:** Tài nguyên dữ liệu cực kỳ giá trị, thúc đẩy nghiên cứu IR web quy mô thực.
    *   **2405.20204 – MotionLLM: bộ dịch V-L riêng cho video và motion, huấn luyện hợp nhất 2 giai đoạn – Suy nghĩ:** Kiến trúc hợp lý để tích hợp video và motion, giải quyết khoảng cách phương thức lớn.
    *   **2405.18377 – LLAMA-NAS: one-shot NAS (InstaTune sửa đổi: cập nhật xen kẽ super/sub-network) và tìm kiếm tiến hóa LINAS cho LLM – Suy nghĩ:** Áp dụng và điều chỉnh hiệu quả one-shot NAS cho nén LLM lớn, không cần huấn luyện lại sau tìm kiếm.
    *   **2405.07990 – Plot2Code: GPT-4V Judgement (đánh giá tương đồng trực quan biểu đồ) và Text-Match Ratio (độ chính xác văn bản) – Suy nghĩ:** Phương pháp đánh giá tự động mới, chuyên biệt cho tạo mã biểu đồ, rất cần thiết.
    *   **2405.06932 – Piccolo2: multi-task hybrid loss (InfoNCE, Cosent, InfoNCE label negatives) cho embedding – Suy nghĩ:** Chiến lược huấn luyện embedding thống nhất, chọn loss phù hợp cho từng loại tác vụ, cải thiện hiệu năng.
    *   **2405.15223 – iVideoGPT: token hóa nén (VQGAN có điều kiện Ec/Ep) và dự đoán tương tác bằng Transformer (hành động nhúng vào token [S], phần thưởng từ token cuối) – Suy nghĩ:** Kiến trúc world model hiệu quả, token hóa nén là một đóng góp sáng tạo cho video dài.
    *   **2405.11157 – Arrow Routing: định tuyến LoRA zero-shot dựa trên SVD của ma trận LoRA (AB^T) – Suy nghĩ:** Cơ chế định tuyến thông minh, không cần dữ liệu gốc, rất phù hợp cho hệ thống phân tán.
    *   **2405.10300 – Grounding DINO 1.5 Edge: Efficient Feature Enhancer (chỉ xử lý P5, vanilla self-attention, cross-scale fusion P3/P4) – Suy nghĩ:** Thiết kế thực tế để giảm chi phí tính toán cho dò tìm đối tượng mở tập trên thiết bị biên.
    *   **2405.01481 – NeMo-Aligner: PPO phân tán (PyTriton điều phối actor/critic/ref/RM trên các cụm) và TensorRT-LLM cho rollout (Refitter, resharding) – Suy nghĩ:** Bộ công cụ hệ thống mạnh mẽ, giải quyết scaling RLHF cho LLM rất lớn.
    *   **2405.15319 – Gstack (G↑direct): xếp chồng lớp từ mô hình cơ sở đã huấn luyện để tăng trưởng LLM – Suy nghĩ:** Kỹ thuật tăng trưởng đơn giản nhưng hiệu quả, tiết kiệm chi phí tiền huấn luyện LLM lớn.
    *   **2405.12107 – Imp-3B: Khảo sát và tối ưu hóa các thành phần (LLM nhẹ Phi-2, SigLIP, LoRA, epochs, dữ liệu SFT) cho LMM nhẹ – Suy nghĩ:** Nghiên cứu thực nghiệm có giá trị, cung cấp hướng dẫn xây dựng LMM nhẹ hiệu quả.
    *   **2404.18212 – Paint by Inpaint (PIPE pipeline): tạo dữ liệu thêm đối tượng (lọc mask, xóa bằng inpainting, lọc kết quả, tạo hướng dẫn VLM-LLM) – Suy nghĩ:** Phương pháp luận độc đáo tạo dữ liệu chất lượng cao cho object insertion, giải quyết vấn đề nhất quán nguồn-đích.
    *   **2405.12213 – Octo: Transformer (attention theo khối, token readout) cho chính sách robot tổng quát, cho phép thêm/bớt cảm biến/hành động khi tinh chỉnh – Suy nghĩ:** Kiến trúc GRP linh hoạt, giải quyết hạn chế về khả năng thích ứng của các GRP trước.
    *   **2405.09220 – Phân tích lý thuyết Transformer học tìm đường (FFN học A_obs, Attention Value học R_obs) và hạn chế về tính bắc cầu – Suy nghĩ:** Đóng góp lý thuyết sâu sắc, giải thích cơ chế và giới hạn của Transformer trong lập kế hoạch.
    *   **2405.01525 – FLAME: SFT_F (dữ liệu tự sinh cho fact-based) và DPO_F (kết hợp RM_IF và RM_fact) – Suy nghĩ:** Phương pháp luận có hệ thống để cải thiện tính xác thực LLM, phân biệt rõ loại chỉ dẫn.
    *   **2405.15613 – K-means phân cấp lặp lại trên tâm cụm và lấy mẫu cân bằng phân cấp để tinh lọc dữ liệu SSL – Suy nghĩ:** Phương pháp tự động và có nguyên tắc để xử lý phân phối lệch trong dữ liệu SSL.
    *   **2405.11582 – PRepBN (BatchNorm tái tham số hóa chuyển đổi dần từ LN) và SLA (ReLU kernel + DWC) – Suy nghĩ:** Các cải tiến kiến trúc Transformer hiệu quả, giải quyết nút thắt tốc độ của LN và attention.
    *   **2405.08246 – BlobGEN: biểu diễn blob dày đặc (elip nghiêng + mô tả) và masked cross-attention cho kiểm soát cục bộ – Suy nghĩ:** Phương pháp mới lạ, cải thiện khả năng kiểm soát và tính hợp thành trong T2I.
    *   **2405.00233 – SemantiCodec: VQ ngữ nghĩa cố định (k-means trên AudioMAE) + VQ âm học học được + LDM decoder – Suy nghĩ:** Kiến trúc codec sáng tạo, tách biệt ngữ nghĩa/âm học, đạt bitrate và token rate cực thấp.
    *   **2405.17405 – Human4DiT: 4D diffusion transformer phân cấp (ảnh 2D, thời gian, góc nhìn) cho video người 360 độ – Suy nghĩ:** Kiến trúc DiT 4D phân cấp hợp lý, giải quyết bài toán tạo video người 360 độ phức tạp.
    *   **2405.17258 – Trans-LoRA: truyền PEFT giữa mô hình bằng KD (dữ liệu tổng hợp lọc bởi discriminator huấn luyện cùng LoRA nguồn) – Suy nghĩ:** Giải pháp sáng tạo cho việc tái sử dụng PEFT, đặc biệt hữu ích khi không có dữ liệu gốc.
    *   **2405.15223 – iVideoGPT: token hóa nén (VQGAN có điều kiện Ec/Ep) và dự đoán tương tác bằng Transformer (hành động nhúng vào token [S], phần thưởng từ token cuối) – Suy nghĩ:** Kiến trúc world model hiệu quả, token hóa nén là một đóng góp sáng tạo cho video dài.
    *   **2405.18386 – Instruct-MusicGen: audio fusion (nhân bản self-attention + cross-attention sửa đổi + cổng) và text fusion (LoRA trên cross-attention T5) – Suy nghĩ:** Tích hợp hiệu quả điều kiện âm thanh/văn bản vào MusicGen bằng adapter/LoRA.
    *   **2405.14477 – LiteVAE: DWT đa cấp độ + UNet nhẹ cho encoder, SMC (tích chập tự điều biến) cho decoder, loss tần số cao – Suy nghĩ:** Kiến trúc VAE hiệu quả, kết hợp DWT và mạng nhẹ, SMC là một đóng góp kỹ thuật thú vị.
    *   **2405.09874 – Dual3D: Khuếch tán tiềm ẩn đa chế độ kép (2D và render 3D từ tri-plane) và suy luận chuyển đổi chế độ kép – Suy nghĩ:** Khung làm việc sáng tạo cho sinh 3D, cân bằng tốc độ và nhất quán 3D.
    *   **2405.08448 – Phân tích thực nghiệm hiệu chỉnh online (RLHF) và offline (DPO/IPO) – Suy nghĩ:** Nghiên cứu thực nghiệm giá trị, làm rõ vai trò của lấy mẫu on-policy và các thách thức của phương pháp offline.
    *   **2405.08295 – SpeechVerse: kết hợp đặc trưng âm thanh đa lớp (trọng số học được), mô-đun tích chập giảm chiều, học theo chương trình 2 giai đoạn cho PEFT – Suy nghĩ:** Khung làm việc hợp lý và hiệu quả cho mô hình âm thanh-ngôn ngữ đa nhiệm.
    *   **2405.20335 – Xwin-LM: quy trình căn chỉnh LLM (SFT, RM, RSFT, DPO) và bộ dữ liệu Xwin-Pair/Set (GPT-4 gán nhãn) – Suy nghĩ:** Công trình kỹ thuật vững chắc, cung cấp quy trình và tài nguyên căn chỉnh LLM mở.
    *   **2405.14105 – DSI (Distributed Speculative Inference) với SP (Speculation Parallelism) và đồng bộ hóa dựa trên kết quả xác minh – Suy nghĩ:** Ý tưởng mạnh mẽ giải quyết hạn chế tuần tự của SI, có tiềm năng tăng tốc đáng kể.
    *   **2405.07065 – LogoMotion: Visually-Grounded Code Synthesis/Repair (VLM phân tích/tạo/sửa code hoạt ảnh) và Code-Connected AI Editing Widgets – Suy nghĩ:** Phương pháp sáng tạo kết hợp VLM, sinh mã, sửa lỗi tự động cho hoạt ảnh logo.
    *   **2405.18870 – Đánh giá ToM bậc cao (2-6) của LLM bằng bộ dữ liệu MoToMQA mới (dựa trên IMT) – Suy nghĩ:** Nghiên cứu thực nghiệm quan trọng, cung cấp công cụ và phân tích sâu về khả năng ToM của LLM.
    *   **2405.18426 – GFlow: phục dựng cảnh 4D (3DGS) và ước tính tư thế camera đồng thời từ video đơn mắt (khởi tạo/làm dày đặc Gaussian dựa trên prior, tối ưu xen kẽ tĩnh/động) – Suy nghĩ:** Phương pháp mới lạ, kết hợp 3DGS với tiên nghiệm 2D và tối ưu hóa xen kẽ cho bài toán khó.
    *   **2405.16537 – I2VEdit: Motion LoRA (Skip-Interval Attention), SARP (nhiễu vùng mịn), Spatial/Temporal Attention Matching – Suy nghĩ:** Framework chỉnh sửa video hiệu quả, các đóng góp kỹ thuật giải quyết vấn đề cụ thể trong pipeline.
    *   **2405.15613 – K-means phân cấp lặp lại trên tâm cụm và lấy mẫu cân bằng phân cấp để tinh lọc dữ liệu SSL – Suy nghĩ:** Phương pháp tự động và có nguyên tắc để xử lý phân phối lệch trong dữ liệu SSL.
    *   **2405.14867 – DMD2: loại bỏ loss hồi quy, TTUR (cập nhật µfake thường xuyên hơn G), loss GAN, mô phỏng ngược (huấn luyện student đa bước) – Suy nghĩ:** Các cải tiến đáng kể cho DMD, giải quyết chi phí và giới hạn của DMD gốc.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Khả năng Diễn giải và Độ Tin cậy của KANs (2404.19756):** Mặc dù KANs hứa hẹn khả năng diễn giải tốt hơn MLP, việc hiểu rõ các hàm B-spline đã học và đảm bảo độ tin cậy của chúng trong các ứng dụng quan trọng vẫn là một thách thức. Cơ hội: Phát triển các công cụ trực quan hóa và phân tích tự động cho KANs, nghiên cứu các phương pháp chính quy hóa để tăng tính diễn giải và độ bền của các hàm kích hoạt học được.
    *   **Mở rộng Quy mô và Ứng dụng Thực tế của Kiến trúc Lai (Jamba 2403.19887, Zamba 2405.16712, Griffin 2402.19427, MEGALODON 2404.08801):** Các kiến trúc này cho thấy hiệu quả cao, nhưng việc huấn luyện và triển khai ở quy mô rất lớn (hàng trăm tỷ/nghìn tỷ tham số) và trên các tác vụ đa dạng ngoài ngôn ngữ vẫn cần được khám phá. Cơ hội: Nghiên cứu các chiến lược huấn luyện phân tán tối ưu cho kiến trúc lai, phát triển các phiên bản đa phương thức và khám phá ứng dụng trong khoa học, robot.
    *   **Tối ưu hóa Tự động cho Suy luận Suy đoán (Speculative Decoding):** Các phương pháp như Kangaroo (2404.18911), Clover (2405.00263), ReDrafter (2403.09919), LayerSkip (2404.16710), DSI (2405.14105) và dự đoán đa token (2404.19737) đều có các siêu tham số (số token suy đoán, ngưỡng tin cậy, lớp thoát sớm) cần tinh chỉnh. Cơ hội: Phát triển các thuật toán meta-learning hoặc RL để tự động tối ưu hóa các siêu tham số này dựa trên đặc điểm của mô hình, tác vụ và tài nguyên phần cứng.
    *   **Tính Nhất quán Ngữ nghĩa và Vật lý trong Tạo Sinh Đa Phương Thức Dài hạn:** Các mô hình tạo nhạc dài (2405.10300), video dài (2405.17405, 2405.15757), và 3D tương tác (2404.09833, 2404.13026, 2404.16510) đang tiến bộ, nhưng việc duy trì tính nhất quán ngữ nghĩa, logic nhân quả, và các quy luật vật lý trong các chuỗi tạo sinh rất dài và phức tạp vẫn là một thách thức lớn. Cơ hội: Tích hợp các mô hình thế giới có cấu trúc (như iVideoGPT 2405.15223), bộ nhớ dài hạn hiệu quả hơn (Infini-attention 2404.07143, TransformerFAM 2404.09173), và các cơ chế kiểm tra/sửa lỗi tự động trong quá trình tạo sinh.
    *   **Căn chỉnh LLM với các Giá trị Phức hợp và Mâu thuẫn:** Các phương pháp như TR Alignment (2404.09656), SPPO (2405.00675), DNO (2404.03715), COOL RLHF (2403.17297), FLAME (2405.01525), và Alignment Studio (2403.09704) đang cố gắng giải quyết vấn đề căn chỉnh, nhưng việc xử lý các hệ thống giá trị phức tạp, đa chiều, và có thể mâu thuẫn từ nhiều nguồn (người dùng, xã hội, quy định) vẫn rất khó khăn. Cơ hội: Phát triển các framework lý thuyết và thuật toán mới cho việc học và dung hòa các sở thích/giá trị đa dạng, có thể bao gồm các cơ chế đàm phán hoặc bỏ phiếu có trọng số trong mô hình phần thưởng hoặc chính sách.
    *   **Hiệu quả Dữ liệu và Khả năng Khái quát hóa Zero/Few-Shot trong Điều kiện Dữ liệu Chuyên biệt Hạn chế:** Mặc dù có các phương pháp tạo dữ liệu tổng hợp (2405.14906, 2404.12803, 2404.18212) và PEFT (2404.03592, 2405.12130, 2405.11157, 2405.17258), việc làm cho các mô hình lớn khái quát hóa tốt sang các miền/tác vụ hoàn toàn mới chỉ với rất ít hoặc không có dữ liệu chuyên biệt vẫn là một mục tiêu quan trọng. Cơ hội: Nghiên cứu các kỹ thuật meta-learning hiệu quả hơn cho PEFT, các phương pháp truy xuất và tích hợp kiến thức zero-shot mạnh mẽ hơn, và các chiến lược tạo dữ liệu tổng hợp có khả năng bao phủ không gian tác vụ tốt hơn.
    *   **Tích hợp Nhận thức Thị giác Sâu vào MLLM:** Các benchmark như Blink (2404.12390) cho thấy MLLM hiện tại còn yếu về nhận thức thị giác cốt lõi. Các kiến trúc như Ferret-v2 (2404.07973), Groma (2404.13013), VisionLLM-H (2404.06512), ConvLLaVA (2405.15738), Dense Connector (2405.13800) đang cố gắng cải thiện điều này. Cơ hội: Phát triển các kiến trúc kết nối thị giác-ngôn ngữ mới, các mục tiêu tiền huấn luyện đa phương thức tập trung hơn vào các kỹ năng nhận thức cơ bản (như suy luận không gian, hiểu quan hệ đối tượng), và tích hợp các module thị giác chuyên biệt một cách hiệu quả hơn.

5.  **FUTURE_IDEAS**

    ✨ **KAN-Powered Interpretable World Models for Physical Reasoning**
    *   Motivation: Current world models (e.g., DIAMOND 2405.12399, iVideoGPT 2405.15223) often use black-box neural networks (Transformers, U-Nets), making it hard to understand their learned physics or debug failures. Kolmogorov-Arnold Networks (KANs, 2404.19756) offer potential for better interpretability and accuracy.
    *   Key novelty: Replace the core transition/dynamics model within a world model architecture with a KAN. The learnable activation functions on the KAN edges could potentially capture explicit physical laws or relationships from observational data.
    *   Approach:
        1.  Use a VQVAE (like in iVideoGPT) or a DWT-based encoder (LiteVAE 2405.14477) to get low-dimensional latent states of the environment.
        2.  Train a KAN to predict the next latent state `z_{t+1}` given the current latent state `z_t` and action `a_t`.
        3.  The KAN's learned activation functions can be visualized and potentially simplified to symbolic expressions, offering insights into the learned dynamics.
        4.  Integrate this KAN-based world model into an RL agent (e.g., Dreamer-like) for planning and control.
    *   Dataset + Metrics: Physics-based simulation environments (e.g., MuJoCo, Isaac Gym, Physion) with ground-truth physical parameters. Metrics: Prediction accuracy of future states, interpretability of learned KAN functions (e.g., can they be mapped to known physical equations?), sample efficiency of RL agent, task performance.
    *   Risk/Feasibility: Medium-high risk. KANs are new and their scalability to very complex dynamics is unproven. Training KANs can be slower than traditional NNs. Extracting symbolic equations from learned splines is non-trivial.

    ✨ **Infini-MoE: Scaling Mixture-of-Experts with Infinite Virtual Experts via Compressive Memory**
    *   Motivation: Mixture-of-Experts (MoE) models (2403.19887, 2404.07413, 2404.15045, 2405.17976) increase capacity but are limited by the physical number of experts. Infini-attention (2404.07143) allows for unbounded context.
    *   Key novelty: An MoE architecture where the "experts" are not fixed, finite neural networks, but rather dynamically retrieved and synthesized "virtual experts" from a large-scale, continuously updated compressive memory (inspired by Infini-attention's memory mechanism).
    *   Approach:
        1.  Router: A standard MoE router selects a small number of "slots" or "query types" for a given token.
        2.  Compressive Expert Memory: A large (potentially unbounded) memory stores (token_embedding, expert_computation_trace/parameters) pairs or their compressed representations. This memory is updated as the model trains or encounters new data.
        3.  Expert Synthesis/Retrieval: For each selected slot/query type, the router's output is used to query the Compressive Expert Memory. Instead of routing to a fixed expert, it retrieves relevant computation traces or parameters.
        4.  Execution: A shared, flexible computation graph executes the retrieved "virtual expert" logic on the input token. This could involve synthesizing a temporary FFN using retrieved parameters or replaying a computation trace.
    *   Dataset + Metrics: Large diverse text corpora (The Pile, C4) and specialized datasets requiring diverse expertise (e.g., multi-domain QA, code generation in many languages). Metrics: Perplexity, downstream task performance, effective number of "virtual experts" utilized, memory footprint of the compressive memory.
    *   Risk/Feasibility: Very high risk. Designing an efficient retrieval and synthesis mechanism for virtual experts from a compressive memory is extremely complex. Ensuring stability and effective learning is a major challenge. (Moon-shot)

    ✨ **Self-Correcting Code Generation via Iterative Symbolic Execution and LLM Feedback**
    *   Motivation: LLMs generate code (2402.19173, 2405.14906) but often make subtle errors. Current repair methods are limited. Symbolic execution can rigorously verify code properties.
    *   Key novelty: An iterative loop where an LLM generates code, a symbolic execution engine (e.g., KLEE, Angr) analyzes it to find potential errors or counterexamples to desired properties (derived from the prompt or unit tests from AIEV-INSTRUCT 2405.14906), and this symbolic feedback (not just pass/fail) is used to prompt the LLM to refine the code.
    *   Approach:
        1.  User provides a natural language prompt for a coding task.
        2.  LLM (e.g., AutoCoder) generates an initial code solution.
        3.  Symbolic Execution Engine:
            *   Converts the prompt into formal specifications/assertions (can be LLM-assisted).
            *   Executes the generated code symbolically, exploring different paths.
            *   If a path violates a spec or a unit test fails, it provides a concrete counterexample or a trace leading to the error.
        4.  Feedback Formulation: The symbolic error trace/counterexample is translated into natural language feedback for the LLM.
        5.  LLM Refinement: The LLM receives the original prompt, its previous code, and the symbolic feedback, then generates a revised code. This loop continues.
    *   Dataset + Metrics: Code generation benchmarks (HumanEval, MBPP), especially those with complex logic or security vulnerabilities. Metrics: Functional correctness (pass@k), code quality, number of iterations to converge, ability to fix specific bug classes identified by symbolic execution.
    *   Risk/Feasibility: Medium risk. Symbolic execution can be slow and state-space explosion is an issue for complex programs. Translating symbolic traces into effective natural language feedback for LLMs is non-trivial.

    ✨ **Adaptive Representation Finetuning (AdaReFT) for Dynamic Task Composition**
    *   Motivation: ReFT/LoReFT (2404.03592) shows promise for parameter-efficient finetuning by editing representations. However, current PEFT methods often train separate adapters for each task.
    *   Key novelty: An extension of LoReFT where multiple, small LoReFT modules (each specialized for a "primitive" skill or sub-task) are learned. At inference time, a gating LLM dynamically selects and composes a sequence or combination of these LoReFT interventions on the hidden states of a frozen base LLM to solve a novel, complex task.
    *   Approach:
        1.  Define a set of primitive skills (e.g., "summarize", "extract sentiment", "translate to SQL", "identify main entity").
        2.  Train a separate LoReFT module ϕ_i for each primitive skill i on relevant data. Each ϕ_i learns to intervene at specific layers.
        3.  Train a Gating LLM: Given a complex user instruction, the Gating LLM outputs a "plan" which is a sequence of (primitive skill_i, target layer_l) or a weighted combination of interventions.
        4.  Execution: The base LLM processes the input. At specified layers, the selected LoReFT interventions are applied according to the Gating LLM's plan.
    *   Dataset + Metrics: Multi-task instruction-following datasets (e.g., SuperNI, FLAN). Metrics: Performance on complex, compositional tasks, zero-shot generalization to new task compositions, number of active parameters per task.
    *   Risk/Feasibility: Medium-high risk. Learning truly composable representation interventions is hard. The Gating LLM needs to be very effective at task decomposition and intervention planning. Interference between composed interventions could be an issue.

6.  **READING_LIST**

    *   2404.19756 – KAN (Kolmogorov-Arnold Networks) · Kiến trúc mạng nơ-ron mới, có thể là một giải pháp thay thế mạnh mẽ và dễ diễn giải hơn cho MLP.
    *   2405.09818 – Chameleon (Early-Fusion MLLM) · Kiến trúc MLLM thống nhất, xử lý/tạo ảnh-văn bản xen kẽ bằng một Transformer duy nhất, cùng kỹ thuật ổn định QK-Norm.
    *   2404.07143 – Infini-attention · Cơ chế attention với bộ nhớ nén cho phép Transformer xử lý ngữ cảnh dài vô hạn, rất quan trọng.
    *   2404.19737 – Multi-Token Prediction LLM · Cải thiện hiệu quả sử dụng dữ liệu và tăng tốc suy luận đáng kể mà không tăng chi phí huấn luyện.
    *   2405.10300 – Jen-1 (Long-form Music Generation) · Tạo nhạc dài, có cấu trúc bằng DiT trên latent rate thấp, một bước tiến lớn cho sinh nhạc.
    *   2404.12253 – ALPHA LLM · Tích hợp MCTS và LLM cho tự cải thiện suy luận, đạt SOTA trên các benchmark toán học khó.
    *   2405.00675 – SPPO (Self-Play Preference Optimization) · Phương pháp alignment mới, hiệu quả, có nền tảng lý thuyết trò chơi, không cần dữ liệu GPT-4.
    *   2405.07518 – SN40L RDU (AI Accelerator) · Kiến trúc phần cứng đột phá cho CoE, giải quyết "bức tường bộ nhớ" với bộ nhớ 3 cấp.
    *   2404.03592 – LoReFT (Representation Finetuning) · Hướng PEFT mới lạ, can thiệp biểu diễn ẩn, hiệu quả hơn LoRA về tham số.
    *   2404.02905 – VAR (Visual AutoRegressive) · Mô hình tự hồi quy ảnh theo tỷ lệ, lần đầu vượt DiT trên ImageNet, rất hứa hẹn.

7.  **META_REFLECTION**

    *   Tháng 05/2024 tiếp tục chứng kiến những bước tiến đáng kể trong nhiều lĩnh vực AI, đặc biệt là sự xuất hiện của các kiến trúc nền tảng mới và các phương pháp tối ưu hóa hiệu quả.
        1.  **Kiến trúc Nền tảng Mới Ngoài Transformer:** Sự ra đời của KAN (2404.19756) là một điểm nhấn đặc biệt, thách thức sự thống trị của MLP và mở ra hướng đi mới cho các mô hình dễ diễn giải và có khả năng chính xác cao hơn. Các kiến trúc lai và dựa trên SSM (Jamba 2403.19887, Zamba 2405.16712, Griffin 2402.19427, MEGALODON 2404.08801, HGRN2 2404.07904) tiếp tục được cải tiến, cho thấy tiềm năng lớn trong việc xử lý chuỗi dài hiệu quả.
        2.  **Hiệu quả LLM (Training & Inference):** Tối ưu hóa LLM vẫn là một chủ đề nóng. Các phương pháp mới về KV cache (Layer-Condensed KV Cache 2405.10637, SnapKV 2404.14469), speculative decoding (Kangaroo 2404.18911, Clover 2405.00263, ReDrafter 2403.09919, Multi-Token Prediction 2404.19737), và phân bổ tính toán động (MoD 2404.02258, LayerSkip 2404.16710) đang liên tục đẩy giới hạn về tốc độ và hiệu quả tài nguyên. Các phương pháp PEFT cũng có những bước tiến mới với LoReFT (2404.03592) và MoRA (2405.12130).
        3.  **Mô hình Đa phương thức (MLLM/VLM):** Lĩnh vực này tiếp tục phát triển mạnh mẽ với các kiến trúc mới (Chameleon 2405.09818), các phương pháp cải thiện khả năng hiểu chi tiết ở độ phân giải cao (VisionLLM-H 2404.06512, ConvLLaVA 2405.15738, Ferret-v2 2404.07973), và các kỹ thuật kết nối thị giác-ngôn ngữ hiệu quả hơn (Dense Connector 2405.13800). Việc tạo sinh đa phương thức (nhạc, video, 3D) cũng có nhiều đột phá về chất lượng và khả năng kiểm soát.
        4.  **Căn chỉnh và Học từ Phản hồi:** Các phương pháp alignment ngày càng tinh vi hơn, không chỉ dựa vào RLHF/DPO truyền thống mà còn khám phá các hướng mới như tối ưu hóa dựa trên lý thuyết trò chơi (SPPO 2405.00675), tối ưu ưu tiên tổng quát (DNO 2404.03715), và tích hợp khám phá chủ động (SELM 2405.19332). Việc tập trung vào các khía cạnh cụ thể như tính xác thực (FLAME 2405.01525) và suy luận (Iterative RPO 2404.19733, ALPHA LLM 2404.12253) cũng được chú trọng.
        5.  **Tạo và Tinh lọc Dữ liệu:** Tầm quan trọng của dữ liệu tiếp tục được khẳng định. Các quy trình tạo dữ liệu tổng hợp quy mô lớn cho các tác vụ chuyên biệt (AIEV-INSTRUCT 2405.14906 cho code, Square 2404.12803 cho VQA văn bản, FRet 2403.20327 cho embedding, MovieLLM 2403.01422 cho video, PIPE 2404.18212 cho object insertion) và các phương pháp tinh lọc dữ liệu (k-means phân cấp 2405.15613) đang trở nên phổ biến.
        6.  **Hệ thống và Công cụ AI:** Sự phát triển của các framework và công cụ hỗ trợ (NeMo-Aligner 2405.01481, OSWORLD 2404.07972, LEGENT 2404.18243, LVLM-Interpret 2404.03118, SN40L RDU 2405.07518) cho thấy sự trưởng thành của lĩnh vực, hướng tới việc xây dựng các hệ thống AI phức tạp, có khả năng mở rộng và dễ tiếp cận hơn.
        7.  **An toàn và Đạo đức AI:** Các vấn đề về an toàn, chống tấn công (AdvPrompter 2404.16873, Instruction Hierarchy 2404.13208) và đánh giá mô hình một cách có trách nhiệm (MoToMQA 2405.18870, GSM1k 2405.00332, PoLL 2404.18796) tiếp tục là những lĩnh vực nghiên cứu quan trọng.
Nhìn chung, tháng 05/2024 cho thấy một sự đa dạng hóa trong các hướng tiếp cận, từ việc khám phá các kiến trúc tính toán hoàn toàn mới, tối ưu hóa sâu các thành phần hiện có, đến việc xây dựng các hệ thống AI phức tạp hơn và các công cụ hỗ trợ mạnh mẽ. Sự tập trung vào hiệu quả, khả năng mở rộng, và độ tin cậy của AI ngày càng rõ nét.
