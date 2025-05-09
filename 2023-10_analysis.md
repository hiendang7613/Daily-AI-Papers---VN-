1.  **TOPIC_TREE**

    *   NLP (Natural Language Processing)
        *   LLM Architecture & Pretraining
            *   Efficient Training & Optimization
                *   Quantization & Compression
                    *   2310.11453 | BitNet giới thiệu kiến trúc Transformer 1-bit huấn luyện từ đầu, giảm chi phí bộ nhớ và năng lượng.
                    *   2310.18313 | Đề xuất khung FP8 mixed-precision tự động cho huấn luyện LLM, tối ưu end-to-end.
                    *   2310.08659 | LoftQ đề xuất phương pháp lượng tử hóa kết hợp khởi tạo low-rank-aware cho LoRA fine-tuning.
                    *   2310.16795 | QMoE triển khai nén dữ liệu phụ thuộc quy mô cho MoE với định dạng sub-1-bit và offloading.
                    *   2310.16836 | Đề xuất kỹ thuật pre-shifted exponent bias cho post-training floating-point quantization (FPQ) 4-bit của LLM.
                    *   2310.10944 | TEQ giới thiệu biến đổi tương đương huấn luyện được (trainable equivalent transformation) cho lượng tử hóa LLM, tương thích với PTQ.
                    *   2310.17157 | DEJAVU giới thiệu "contextual sparsity" và predictor bất đồng bộ để tăng tốc LLM inference với structured dynamic sparsity.
                *   Context Window Extension & Positional Encoding
                    *   2310.16450 | CLEX đề xuất khuôn khổ liên tục mở rộng Position Embedding (PE) bằng Neural ODE.
                *   Data Ordering & Pretraining Strategies
                    *   2310.10638 | IN-CONTEXT PRETRAINING huấn luyện LLM trên chuỗi tài liệu liên quan thay vì ngẫu nhiên, sử dụng thuật toán truy vấn lân cận và giải TSP.
            *   Model Adaptation & Fine-tuning
                *   Parameter-Efficient Fine-Tuning (PEFT)
                    *   2310.11454 | VeRA đề xuất reparameterization với ma trận ngẫu nhiên cố định và vector scaling có thể huấn luyện, giảm tham số so với LoRA.
                    *   2310.18356 | LoRAShear đề xuất thuật toán pruning có cấu trúc tiến bộ LoRA Half-Space Projected Gradient (LHSPG) và dynamic knowledge recovery.
                *   Instruction Tuning & Alignment
                    *   2310.12823 | AgentTuning là phương pháp instruction-tuning lai ghép dữ liệu agent và general để tăng khả năng agent của LLM.
                    *   2310.00898 | PIT (Implicit Self-Improvement) cho phép LLM tự học mục tiêu cải tiến ngầm từ dữ liệu ưa thích, không cần rubric.
                    *   2310.13385 | Đề xuất Probabilistic Ranking và Contextual Ranking để fine-tune LLM (Tuna) ưu tiên phản hồi chất lượng cao.
                    *   2310.19019 | TeacherLM giới thiệu quy trình huấn luyện đa giai đoạn và cơ sở dữ liệu TeacherData-2M để LLM sinh giải thích có cấu trúc (fundamentals, CoT, common mistakes).
                *   Domain-Specific LLMs
                    *   2310.10631 | LLEMMA tiếp tục huấn luyện Code Llama trên Proof-Pile-2 (văn bản khoa học, toán học, mã toán học) cho lý luận toán học.
                    *   2310.09263 | Table-GPT đề xuất "table-tuning" để huấn luyện LLM trên đa dạng table-tasks, cải thiện khả năng đọc hiểu bảng.
                    *   2310.17784 | FLLM sử dụng multitask prompt-based finetuning và Abductive Augmentation Reasoning (AAR) cho dữ liệu tài chính.
                *   Self-Improvement & Self-Correction
                    *   2310.01798 | Phân tích thất bại của LLM trong "tự sửa lỗi nội tại" (intrinsic self-correction) cho bài toán suy luận.
                    *   2310.13522 | TRIPOST là quy trình huấn luyện lặp giúp mô hình nhỏ tự cải thiện qua interactive trajectory editing và data post-processing.
            *   Reasoning & Planning
                *   Prompting Strategies
                    *   2310.03051 | Thinking for Doing (T4D) benchmark và Foresee and Reflect (FaR) prompting cho LLM kết nối suy luận trạng thái tâm trí với hành động.
                    *   2310.01714 | Analogical prompting đề xuất LLM tự sinh ví dụ minh họa (self-generated exemplars) liên quan trước khi giải bài toán.
                *   Code Execution & Tool Use
                    *   2310.03731 | MathCodeInstruct giới thiệu dataset LCE (Language-Code-Execution) và pipeline inference tích hợp thực thi mã cho lý luận toán học.
                    *   2310.08992 | CodeChain là khung inference modular cho LLM tự sửa đổi mã có hướng dẫn bởi sub-module đại diện qua clustering.
                *   Agent Capabilities
                    *   2310.08740 | Đại lý zero-shot điều khiển máy tính với compact screen representation, staged plan-and-follow và structured thought management.
                    *   2310.13227 | ToolChain* đề xuất thuật toán tìm kiếm cây A* cho LLM điều hướng không gian hành động của công cụ.
            *   Evaluation & Benchmarking
                *   2310.08491 | PROMETHEUS là LLM evaluator 13B chuyên đánh giá fine-grained dựa trên user-defined rubrics, huấn luyện trên FEEDBACK COLLECTION.
                *   2310.17631 | JudgeLM là LLM judge được fine-tune để đánh giá cặp câu trả lời, giải quyết các loại bias (position, knowledge, format).
                *   2310.01557 | SmartPlay benchmark đánh giá khả năng của LLM như tác nhân thông minh qua 6 trò chơi và 9 năng lực cốt lõi.
                *   2310.08678 | Đánh giá khả năng suy luận tài chính của ChatGPT/GPT-4 trên câu hỏi thi CFA, so sánh ZS, CoT, FS prompting.
            *   Mechanistic Interpretability
                *   2310.15916 | Phân tách quá trình In-Context Learning (ICL) thành hai thành phần học thuật (A) và ứng dụng quy tắc (f) thông qua task vector.
            *   Efficient Inference
                *   2310.12962 | Emulated Fine-Tuning (EFT) phân tách quy mô pretrain và finetune, cho phép up-scaling và speculative decoding.
            *   Privacy & Security
                *   2310.16789 | WIKIMIA benchmark động và MIN-K% PROB phương pháp MIA tham chiếu tự do để phát hiện dữ liệu tiền huấn luyện.
            *   Data Synthesis & Augmentation
                *   2310.13671 | S3 (Synthesis Step by Step) là khung tổng quát kết hợp sinh dữ liệu hạt giống và tối ưu bằng phản hồi lỗi từ mô hình nhỏ (Error Extrapolation-based Synthesis).
                *   2310.13127 | Auto-Instruct tự động sinh và xếp hạng instruction cho black-box LLM bằng nhiều meta-prompt và ranker model (QT5).
            *   Controlled Generation & Decoding
                *   2310.17022 | Controlled Decoding (CD) phân tách base LM và prefix scorer, đề xuất CD-Q (off-policy Bellman update) và blockwise CD.
                *   2310.09520 | Reward-Augmented Decoding (RAD) sử dụng mô hình reward unidirectional và caching để điều khiển sinh văn bản.
                *   2310.09139 | CONSENSUS GAME và EQUILIBRIUM-RANKING cho giải mã ngôn ngữ kết hợp generative và discriminative LM.
        *   Retrieval-Augmented Generation (RAG)
            *   2310.11511 | SELF-RAG là khung LLM chủ động truy vấn, sinh văn bản và tự đánh giá bằng reflection tokens.
            *   2310.03214 | FRESH PROMPT là phương pháp few-shot ICL tích hợp chứng cứ cập nhật từ search engine vào prompt.
    *   Computer Vision (CV)
        *   Image Generation & Synthesis
            *   Diffusion Models
                *   Efficient Training & Architectures
                    *   2310.00426 | PIXART-α đề xuất chiến lược huấn luyện 3 giai đoạn và T2I Transformer hiệu quả (AdaLN-single, re-parameterization).
                    *   2310.15111 | MDM (Multi-Resolution Diffusion Model) với NestedUNet và progressive training cho ảnh/video độ phân giải cao pixel-space.
                *   Data-Centric Methods
                    *   2310.16656 | RECAP sử dụng mô hình caption tự động (PaLI tinh chỉnh) để tái chú thích dữ liệu T2I, cải thiện chất lượng sinh ảnh.
                    *   2310.16825 | CommonCanvas giới thiệu telephoning (BLIP-2 tạo chú thích tổng hợp) và CommonCatalog dataset (ảnh CC) cho T2I.
                *   Fast Sampling & Distillation
                    *   2310.01407 | CoDi chưng cất trực tiếp mô hình diffusion có điều kiện từ LDM tiền huấn luyện không cần dữ liệu gốc, sử dụng tính nhất quán khuếch tán.
                    *   2310.13268 | DPM-Solver-v3 là bộ giải ODE đa bậc với empirical model statistics (EMS) và pseudo-order solver cho sampling nhanh DPMs.
                *   Controllable Generation
                    *   2310.08579 | HyperHuman giới thiệu Latent Structural Diffusion Model (RGB, depth, normal) và Structure-Guided Refiner cho sinh ảnh người có kiểm soát.
                    *   2310.19784 | CustomNet tích hợp Zero-1-to-3 cho tùy biến đối tượng zero-shot với điều khiển vị trí và nền (Dual Cross-Attention).
            *   Text-to-Image (T2I) General
                *   2310.03502 | Kandinsky giới thiệu kiến trúc latent diffusion với image prior (diffusion transformer-encoder) và MoVQ autoencoder tùy biến cho T2I.
            *   Evaluation & Benchmarking
                *   2310.01596 | ImagenHub là framework chuẩn hóa dataset, inference và human evaluation cho 7 tác vụ conditional image generation.
                *   2310.15144 | DEsignBench là benchmark T2I tập trung vào thiết kế thị giác, sử dụng GPT-4V để đánh giá tự động.
            *   Iterative Prompt Refinement
                *   2310.08541 | Idea2Img sử dụng GPT-4V đa vai trò (sinh/sửa prompt, chọn ảnh, phản hồi) để tự tinh chỉnh lặp lại prompt cho mô hình T2I.
        *   3D Content Creation & Understanding
            *   Text-to-3D Generation
                *   Procedural Generation
                    *   2310.12945 | 3D-GPT là framework không cần huấn luyện, dùng LLM đa tác nhân (task dispatch, conceptualization, modeling) sinh mã Python điều khiển Blender.
                *   Diffusion-based & Neural Fields
                    *   2310.16818 | DreamCraft3D kết hợp view-conditioned 3D diffusion prior (Zero-1-to-3) với SDS lai, progressive view training và bootstrapped score distillation.
                    *   2310.08529 | GaussianDreamer kết hợp diffusion 3D và 2D qua 3D Gaussian Splatting, với noisy point growing và color perturbation.
                    *   2310.11784 | Progressive3D phân tách tạo 3D từ prompt phức tạp thành chuỗi chỉnh sửa cục bộ có điều hướng với editable regions và OSCS.
                    *   2310.17075 | HyperFields dự đoán trọng số NeRF theo trình tự lớp bằng dynamic hypernetwork có điều kiện văn bản và activation, dùng NeRF distillation.
            *   Single-Image 3D Reconstruction
                *   2310.15008 | Wonder3D là mô hình khuếch tán chéo miền đa quan điểm tạo đồng thời bản đồ pháp tuyến và ảnh màu từ ảnh đơn, với domain switcher và geometry-aware normal fusion.
            *   Dynamic View Synthesis & Rendering
                *   2310.11448 | 4K4D giới thiệu đại diện điểm đám mây 4D, mô hình ngoại thất hybrid và differentiable depth peeling cho render real-time độ phân giải cao.
        *   Video Generation & Synthesis
            *   Diffusion Models
                *   2310.19512 | VideoCrafter mở rộng SD UNet thành spatio-temporal 3D U-Net (Spatial & Temporal Transformers, FPS embedder) và nhánh I2V với Text-Aligned Rich Image Embedding.
                *   2310.08465 | MotionDirector đề xuất Dual-Path LoRAs và appearance-debiased temporal loss để tách biệt học appearance và motion trong video diffusion.
                *   2310.15169 | FreeNoise đề xuất Local Noise Shuffle Unit và Window-based Attention Fusion cho inference video diffusion dài hơn Ntrain khung hình, không cần tuning.
            *   Evaluation & Benchmarking
                *   2310.11440 | EvalCrafter là pipeline benchmark T2V với quy trình sinh prompt, 17 metrics đa khía cạnh (VQ, text-video consistency, motion, temporal consistency) và human-alignment.
        *   Meta-Learning & Few-Shot Classification
            *   2310.10971 | CAML tái định nghĩa n-way k-shot image classification thành non-causal sequence modeling với ELMES class encoder, học khái niệm "in-context".
        *   Foundation Model Fusion & Continual Learning
            *   2310.15308 | SAM-CLIP trộn SAM và CLIP qua multi-task distillation và continual learning, phát sinh khả năng zero-shot semantic segmentation.
            *   2310.16226 | TIC-CLIP giới thiệu benchmark Time-Continual (TIC-DataComp) và giao thức streaming continual training cho CLIP với replay buffer.
        *   Architecture Comparison
            *   2310.16764 | Đánh giá khả năng mở rộng của NFNet (ConvNet) trên JFT-4B, cho thấy hiệu suất tương đương ViT với cùng budget tính toán.
    *   Multimodal Learning
        *   Vision-Language Models (VLM) / Multimodal LLMs (MLLM)
            *   Instruction Tuning & Architectures
                *   2310.03744 | Đề xuất Response Format Prompting, MLP Cross-Modal Connector và Pipeline Xử Lý Ảnh Độ Phân Giải Cao cho VLM.
                *   2310.09199 | PaLI-3 tích hợp SigLIP contrastively pretrained ViT-G/14, huấn luyện đa giai đoạn với curriculum resolution và mở rộng segmentation bằng VQ-VAE.
                *   2310.09478 | MiniGPT-v2 sử dụng token định danh tác vụ, ghép token hình ảnh lân cận và huấn luyện 3 giai đoạn cho MLLM.
            *   Visual Grounding & Referring Tasks
                *   2310.11441 | Set-of-Mark (SoM) Prompting kích hoạt khả năng grounding của GPT-4V bằng phân đoạn ảnh và gán dấu "speakable".
            *   Hallucination Mitigation
                *   2310.16045 | Woodpecker là framework hậu xử lý khử ảo giác training-free cho MLLM, sử dụng pipeline 5 bước và visual knowledge base.
            *   Evaluation & Benchmarking
                *   2310.14566 | HALLUSION BENCH là benchmark chẩn đoán lỗi hallucination ngôn ngữ và illusion thị giác trong LVLM với control pairs và GPT-4-assisted evaluation.
                *   2310.16534 | Đánh giá định lượng và phân tích hành vi của GPT-4V trên nhiều tác vụ thị giác-ngôn ngữ, bao gồm cả đặc điểm nhạy cảm.
                *   2310.19061 | Đánh giá toàn diện khả năng của GPT-4V trong VQA y tế, phân tích hạn chế và khả năng qua 7 khía cạnh.
            *   Video Understanding
                *   2310.19773 | MM-VID là hệ thống pipeline 4 mô-đun (pre-processing, knowledge collection, clip description, script generation) dùng GPT-4V/GPT-4 cho hiểu video dài.
        *   Audio-Language Integration
            *   2310.13289 | SALMONN là MLLM tích hợp Whisper và BEATs qua window-level Q-Former và few-shot activation tuning cho xử lý âm thanh tổng quát.
            *   2310.00704 | UniAudio là foundation model LLM thống nhất 11 tác vụ tạo âm thanh (speech, sounds, music) qua tokenization chung và multi-scale Transformer.
        *   Tool-Augmented LLMs for Multimodal Tasks
            *   2310.17796 | ControlLLM sử dụng Tool Graph (ToG) với DFS search và tool assessment module để điều phối thực thi chuỗi công cụ đa phương tiện.
            *   2310.11954 | MusicAgent là hệ thống tác nhân tự động dùng LLM phân tích yêu cầu và phối hợp công cụ âm nhạc đa nguồn với chuẩn I/O thống nhất.
    *   Reinforcement Learning (RL)
        *   RL from Human Feedback (RLHF) & Alignment
            *   2310.12773 | Safe RLHF tích hợp Safe RL (CMDP, Lagrangian dual) vào RLHF với annotation hai chiều (hữu ích, vô hại) và Cost Model.
            *   2310.03716 | Phân tích thiên vị độ dài (length bias) trong RLHF, cho thấy phần lớn cải thiện từ PPO là do tăng độ dài, đề xuất NRG metric.
        *   Reward Engineering & Modeling
            *   2310.12931 | EUREKA sử dụng LLM (GPT-4) với environment-as-context, evolutionary search và reward reflection để zero-shot sinh hàm thưởng cho RL.
            *   2310.12921 | VLM-RM sử dụng VLM tiền huấn luyện (CLIP) làm reward model zero-shot cho RL thị giác, với Goal-Baseline Regularization.
        *   Preference-based Policy Learning
            *   2310.13639 | Contrastive Preference Learning (CPL) học chính sách trực tiếp từ phản hồi ưu tiên dựa trên regret, không cần học reward model.
    *   Embodied AI & Robotics
        *   Vision-Language Modeling for Embodiment
            *   2310.08588 | Octopus là mô hình lập trình viên thị giác-ngôn ngữ (VLM) sinh mã Python thực thi tác vụ trong môi trường OctoVerse, huấn luyện bằng RLEF.
        *   LLM-Driven Planning & Control
            *   2310.10645 | ITP là khung hai cấp (high-level planner, low-level executor) dùng LLM với vision module và replanning động cho robot.
            *   2310.10625 | Video Language Planning (VLP) kết hợp VLM (policy, heuristic) và T2V model (dynamics) cho lập kế hoạch video dài hạn trong robot.
    *   Knowledge Distillation
        *   2310.13332 | Đề xuất học tương tác đa vòng giữa LM nhỏ và LLM đen, kết hợp self-reflection (triplet loss) để chưng cất khả năng suy luận.
    *   ML Systems
        *   Efficient On-Device Fine-tuning
            *   2310.17752 | PockEngine là framework huấn luyện dựa trên biên dịch (compilation-based) cho fine-tuning hiệu quả và thưa (sparse backpropagation) trên thiết bị biên.
        *   Low-bit LLM Serving
            *   2310.19102 | Atom là thuật toán lượng tử hóa 4-bit (trọng số, kích hoạt) cho LLM serving, kết hợp mixed-precision, channel reordering và custom CUDA kernels.
    *   Other
        *   2310.10837 | Khung thống nhất cho xấp xỉ MLP hai lớp trong Transformer (Top-K activation, PKM cải tiến, σ-MoE), với regularization và khởi tạo mới.
        *   2310.09983 | FARZI đề xuất data distillation cho dữ liệu tự hồi quy bằng low-rank latent summary, reverse-mode differentiation cho Adam.
        *   2310.12274 | Multi-Concept Prompt Learning (MCPL) học đồng thời nhiều prompt concept từ câu-ảnh không cần annotation, dùng Attention Masking và Prompts Contrastive Loss.
        *   2310.03734 | ITIT là khung huấn luyện dựa trên cycle consistency (T2I2T, I2T2I) cho mô hình sinh vision-language, tận dụng dữ liệu không cặp đôi.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                      | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                                           |
    | :--- | :-------- | :---------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
    | 1    | 2310.11453 | 1-bit LLM, BitNet, BitLinear, Training from scratch   | Giới thiệu BitNet, kiến trúc Transformer 1-bit đầu tiên cho LLM được huấn luyện từ đầu, thay vì hậu lượng tử hóa.                        | Tiềm năng giảm đáng kể chi phí bộ nhớ, năng lượng và tăng tốc độ cho LLM, mở đường cho LLM hiệu quả hơn trên thiết bị hạn chế tài nguyên.                               |
    | 2    | 2310.11511 | SELF-RAG, Retrieval-Augmented, Self-Reflection      | LLM chủ động quyết định khi nào cần truy xuất, tự sinh truy vấn và tự đánh giá chất lượng đầu ra bằng reflection tokens.                 | Cải thiện tính xác thực và độ tin cậy của LLM trong các tác vụ đòi hỏi kiến thức, giảm hallucination, tăng khả năng kiểm soát.                                       |
    | 3    | 2310.12931 | EUREKA, Reward Shaping, LLM for RL, Zero-shot       | Tự động hóa việc thiết kế hàm thưởng phức tạp cho RL bằng cách sử dụng LLM (GPT-4) zero-shot với environment-as-context và evolutionary search. | Cách mạng hóa quy trình reward engineering, cho phép giải quyết các bài toán RL phức tạp mà trước đây đòi hỏi chuyên môn sâu và tinh chỉnh thủ công.                     |
    | 4    | 2310.00426 | PIXART-α, Efficient T2I, Diffusion Transformer, SAM-LLaVA | Kiến trúc T2I Transformer hiệu quả (PIXART-α) với chiến lược huấn luyện 3 giai đoạn và quy trình tạo dữ liệu auto-labeling giàu thông tin. | Giảm đáng kể thời gian huấn luyện (88%) và tài nguyên tính toán cho các mô hình T2I chất lượng cao, dân chủ hóa việc phát triển mô hình T2I.                         |
    | 5    | 2310.17157 | DEJAVU, Contextual Sparsity, LLM Inference, Lookahead Predictor | Giới thiệu "contextual sparsity" động và predictor bất đồng bộ, cho phép tăng tốc LLM inference đáng kể trên GPU mà không giảm chất lượng. | Giải quyết một trong những thách thức lớn của LLM deployment là độ trễ và chi phí tính toán, giúp LLM dễ tiếp cận và sử dụng trong ứng dụng thời gian thực hơn. |
    | 6    | 2310.15008 | Wonder3D, Single-Image 3D, Multi-view Diffusion, Normal Fusion | Tạo đồng thời bản đồ pháp tuyến và ảnh màu đa quan điểm từ ảnh đơn bằng diffusion model, sau đó hợp nhất thành lưới 3D có texture.        | Đơn giản hóa và tăng tốc đáng kể quá trình tạo nội dung 3D chất lượng cao từ một ảnh duy nhất, có tiềm năng lớn cho VR/AR và game.                               |
    | 7    | 2310.12773 | Safe RLHF, Constrained MDP, Lagrangian Dual, Alignment | Tích hợp Safe RL vào RLHF bằng CMDP và Lagrangian dual, tách biệt annotation hữu ích/vô hại để tối ưu LLM an toàn và hữu ích.             | Cung cấp một phương pháp có nguyên tắc để cân bằng giữa tính hữu ích và tính an toàn của LLM, một vấn đề quan trọng trong AI alignment.                               |
    | 8    | 2310.03714 | DSPy, LM Programming, Pipeline Optimization, Teleprompters | Framework lập trình LM trừu tượng hóa pipeline thành graph, tự động tối ưu prompt/fine-tuning bằng "teleprompters".                   | Thay đổi cách xây dựng và tối ưu hóa các ứng dụng dựa trên LM, giúp quá trình này trở nên có hệ thống, hiệu quả và ít phụ thuộc vào prompt engineering thủ công. |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2310.11453 – BitLinear & Group Quantization – Suy nghĩ:** Thành phần 1-bit thay thế nn.Linear và kỹ thuật quantization theo nhóm là cốt lõi cho việc huấn luyện LLM 1-bit từ đầu, rất hứa hẹn cho LLM siêu hiệu quả.
    *   **2310.03502 – Image Prior (Diffusion Transformer-Encoder) & Sber-MoVQGAN – Suy nghĩ:** Image prior học ánh xạ text embedding sang image embedding giúp định hướng latent diffusion tốt hơn; MoVQ tùy biến cải thiện giải mã. Kết hợp này có tiềm năng cho T2I chất lượng cao.
    *   **2310.11511 – Reflection Tokens (Retrieve, ISREL, ISSUP, ISUSE) & Beam Search cấp đoạn – Suy nghĩ:** Token đặc biệt cho phép LLM học khi nào truy vấn và tự đánh giá, beam search cấp đoạn tối ưu hóa quá trình sinh có truy vấn. Rất sáng tạo và giải quyết vấn đề "khi nào nên RAG".
    *   **2310.00426 – AdaLN-single & SAM-LLaVA auto-labeling – Suy nghĩ:** AdaLN-single giảm tham số đáng kể trong DiT; quy trình auto-labeling tạo caption dày đặc khái niệm giúp huấn luyện T2I hiệu quả hơn. Hướng tiếp cận data-centric mạnh mẽ.
    *   **2310.12945 – Hệ thống đa tác nhân LLM cho 3D (Task Dispatch, Conceptualization, Modeling) & Prompt Engineering với tài liệu hàm – Suy nghĩ:** Phân chia vai trò LLM để xử lý các khía cạnh khác nhau của việc tạo 3D từ text là một cách tiếp cận thông minh, không cần huấn luyện mạng neural.
    *   **2310.10631 – Proof-Pile-2 & AlgebraicStack datasets – Suy nghĩ:** Các bộ dữ liệu chuyên biệt này là đóng góp quan trọng cho cộng đồng nghiên cứu lý luận toán học bằng LLM, thúc đẩy phát triển trong lĩnh vực này.
    *   **2310.08491 – FEEDBACK COLLECTION dataset & Chuỗi sinh feedback rồi sinh score với marker [RESULT] – Suy nghĩ:** Dataset quy mô lớn cho evaluator LLM và chiến lược fine-tuning giải quyết degeneration là những đóng góp thực tiễn cho việc đánh giá LLM.
    *   **2310.16656 – RECAP (PaLI tinh chỉnh tái chú thích LAION) & Prefix conditioning (RECAP Short/Long) – Suy nghĩ:** Cải thiện chất lượng caption đầu vào cho T2I là một hướng data-centric hiệu quả, prefix conditioning giúp kiểm soát độ chi tiết.
    *   **2310.15916 – Phân tách ICL thành A (học) và f (ứng dụng) qua Task Vector patching – Suy nghĩ:** Cung cấp một khung giả thuyết rõ ràng và cơ chế thực nghiệm để hiểu rõ hơn về cách ICL hoạt động trong LLM.
    *   **2310.15111 – NestedUNet & Progressive Training cho Multi-Resolution Diffusion – Suy nghĩ:** Kiến trúc lồng nhau và lịch trình huấn luyện tiến triển cho phép tạo ảnh/video độ phân giải cao trong không gian pixel mà không cần cascaded models hay latent diffusion.
    *   **2310.09263 – Table-tuning paradigm & Augment dữ liệu 4 cấp độ cho table tasks – Suy nghĩ:** Mở rộng instruction-tuning cho dữ liệu bảng hai chiều và chiến lược augment dữ liệu đa dạng là hướng đi mới để LLM hiểu bảng tốt hơn.
    *   **2310.11448 – 4K4D representation (điểm đám mây 4D + lưới đặc trưng 4D) & Differentiable Depth Peeling – Suy nghĩ:** Kết hợp rasterization phần cứng với representation 4D và depth peeling khả vi cho phép render real-time cảnh động chất lượng cao.
    *   **2310.03744 – MLP Cross-Modal Connector & Pipeline Xử Lý Ảnh Độ Phân Giải Cao (chia lưới, hợp nhất feature) – Suy nghĩ:** Connector MLP tăng khả năng biểu diễn liên modal; pipeline xử lý ảnh HD linh hoạt và hiệu quả dữ liệu cho VLM.
    *   **2310.12823 – AgentInstruct dataset & AgentTuning (lai ghép D_agent với D_general) – Suy nghĩ:** Dataset tương tác agent đa tác vụ và chiến lược fine-tuning lai ghép là đóng góp quan trọng để LLM có năng lực agent mà vẫn giữ khả năng chung.
    *   **2310.08588 – OctoVerse environments & Reinforcement Learning with Environmental Feedback (RLEF) – Suy nghĩ:** Bộ môi trường đa dạng và RLEF cung cấp nền tảng tốt để huấn luyện và đánh giá VLM sinh mã thực thi trong môi trường tương tác.
    *   **2310.17631 – Swap Augmentation & Reference Support/Drop cho JudgeLM – Suy nghĩ:** Các kỹ thuật này giúp giảm các loại bias (position, knowledge, format) khi huấn luyện LLM làm giám khảo, tăng tính tổng quát.
    *   **2310.16825 – Telephoning (BLIP-2 tạo chú thích tổng hợp) & CommonCatalog dataset – Suy nghĩ:** Sử dụng VLM để tạo chú thích tổng hợp cho dữ liệu không nhãn là một cách thông minh để xây dựng dataset T2I chất lượng cao, có bản quyền rõ ràng.
    *   **2310.03051 – Foresee and Reflect (FaR) prompting framework – Suy nghĩ:** Cấu trúc hai bước (dự đoán diễn biến, đánh giá và chọn hành động) giúp LLM định hướng suy luận tốt hơn cho các bài toán kết nối Theory of Mind với hành động.
    *   **2310.03714 – DSPy programming model (Signatures, Modules, Teleprompters) – Suy nghĩ:** Trừu tượng hóa pipeline LM thành graph với các module có thể học và tối ưu tự động là một bước tiến lớn so với prompt engineering thủ công.
    *   **2310.18313 – Precision decoupling & Automatic scaling cho FP8 training – Suy nghĩ:** Các kỹ thuật này giúp ngăn underflow/overflow và áp dụng FP8 end-to-end (tính toán, lưu trữ, giao tiếp) cho huấn luyện LLM, rất quan trọng cho hiệu quả.
    *   **2310.16818 – Bootstrapped Score Distillation (LBSD) & Progressive View Training cho 3D – Suy nghĩ:** LBSD cho phép mô hình 3D tự cải thiện qua việc fine-tune DreamBooth trên render đa góc nhìn; progressive view training giúp tối ưu geometry.
    *   **2310.11454 – VeRA (Vector-based Random Matrix Adaptation) – Suy nghĩ:** Chỉ học vector scaling và chia sẻ ma trận ngẫu nhiên cố định giúp giảm đáng kể tham số trainable so với LoRA, rất hiệu quả về bộ nhớ.
    *   **2310.10638 – IN-CONTEXT PRETRAINING với thuật toán Maximum Traveling Salesman trên đồ thị tài liệu – Suy nghĩ:** Sắp xếp tài liệu theoความ liên quan ngữ nghĩa để huấn luyện LLM là một ý tưởng độc đáo, có thể cải thiện khả năng học ngữ cảnh dài.
    *   **2310.03731 – MathCodeInstruct dataset (LCE: Language-Code-Execution) & Problem interpolation prompting – Suy nghĩ:** Dataset tích hợp thực thi mã và chiến lược sinh bài toán mới giúp LLM học suy luận toán học kết hợp công cụ hiệu quả.
    *   **2310.12773 – Safe RLHF với Cost Model và Lagrangian dual – Suy nghĩ:** Chính thức hóa RLHF an toàn dưới dạng CMDP và tách biệt annotation hữu ích/vô hại là một đóng góp quan trọng cho AI alignment.
    *   **2310.11441 – Set-of-Mark (SoM) Prompting & Mark Allocation algorithm – Suy nghĩ:** Sử dụng dấu "speakable" và thuật toán đặt dấu thông minh để GPT-4V thực hiện visual grounding zero-shot là một cách tiếp cận rất thực tế.
    *   **2310.08659 – LoftQ (Low-rank-aware initialization cho LoRA quantization) – Suy nghĩ:** Đồng thời tối ưu trọng số lượng tử hóa và adapters ngay từ đầu giúp cải thiện hiệu suất fine-tuning ở bit thấp.
    *   **2310.16795 – QMoE (Quantized MoE) với expert grouping và LZW-like sub-1-bit compression – Suy nghĩ:** Các kỹ thuật này cho phép nén mô hình MoE khổng lồ xuống sub-1-bit với overhead thấp, rất quan trọng cho deployment.
    *   **2310.14566 – HALLUSION BENCH với control pairs & GPT-4-assisted evaluation – Suy nghĩ:** Benchmark chẩn đoán lỗi hallucination/illusion với cấu trúc cặp câu hỏi so sánh và đánh giá tự động bằng GPT-4 là công cụ hữu ích.
    *   **2310.09199 – SigLIP contrastive pretraining cho VLM & Segmentation bằng VQ-VAE mask tokens – Suy nghĩ:** SigLIP cải thiện localization; VQ-VAE mask tokens cho phép VLM xuất segmentation mask như text.
    *   **2310.12931 – Environment-as-context & Reward reflection cho EUREKA – Suy nghĩ:** Đưa mã nguồn môi trường vào LLM và tổng hợp thống kê huấn luyện thành feedback text giúp LLM tự động sinh hàm thưởng RL hiệu quả.
    *   **2310.13639 – Contrastive Preference Learning (CPL) & Bias regularizer – Suy nghĩ:** Học chính sách trực tiếp từ preference không cần reward model, bias regularizer cải thiện trên dữ liệu offline hữu hạn.
    *   **2310.11954 – MusicAgent với chuẩn I/O thống nhất cho công cụ âm nhạc & Task Planner/Tool Selector/Executor/Response Generator – Suy nghĩ:** Kiến trúc agent phối hợp nhiều công cụ âm nhạc chuyên biệt thông qua LLM và chuẩn I/O chung mở ra hướng mới cho sáng tạo âm nhạc AI.
    *   **2310.18356 – LoRA Half-Space Projected Gradient (LHSPG) & Dynamic knowledge recovery – Suy nghĩ:** LHSPG cho phép pruning có cấu trúc LLM gắn LoRA và chuyển giao kiến thức; dynamic recovery giúp phục hồi hiệu năng.
    *   **2310.15308 – SAM-CLIP multi-head architecture & Two-stage multi-task distillation – Suy nghĩ:** Kiến trúc backbone chung với head riêng và quy trình học liên tục, chưng cất đa tác vụ giúp hợp nhất hiệu quả hai VFM.
    *   **2310.00898 – PIT (Implicit Self-Improvement) với Reward model RPIT (reward gap) & Curriculum Reinforcement Learning – Suy nghĩ:** Cho phép LLM tự học cải tiến từ preference data mà không cần rubric, reward gap và curriculum RL là những ý tưởng mới.
    *   **2310.15008 – Domain switcher & Multi-view self-attention / Cross-domain attention cho Wonder3D – Suy nghĩ:** Domain switcher cho phép UNet xử lý đa miền (normal, color) mà không quên kiến thức; các cơ chế attention đảm bảo nhất quán hình học và vật liệu.
    *   **2310.09478 – Token định danh tác vụ & Ghép token hình ảnh lân cận cho MiniGPT-v2 – Suy nghĩ:** Token tác vụ giúp MLLM phân biệt lệnh; ghép token hình ảnh giảm tính toán mà vẫn giữ độ phân giải.
    *   **2310.00704 – UniAudio multi-scale Transformer (global & local) & Tokenization thống nhất cho âm thanh – Suy nghĩ:** Kiến trúc Transformer đa tỉ lệ xử lý chuỗi âm thanh dài hiệu quả; tokenization chung cho nhiều loại âm thanh và điều kiện là bước tiến tới foundation model cho audio.
    *   **2310.19773 – MM-VID pipeline (Pre-Processing, Knowledge Collection, Clip Description, Script Generation) & Visual Prompting cho GPT-4V – Suy nghĩ:** Pipeline module hóa sử dụng GPT-4V/GPT-4 để hiểu video dài và tạo kịch bản chi tiết.
    *   **2310.12921 – VLM-RM (VLM as Reward Model) & Goal-Baseline Regularization – Suy nghĩ:** Sử dụng VLM (CLIP) zero-shot làm reward model cho RL thị giác và kỹ thuật regularization mới giúp định hướng reward.
    *   **2310.01407 – CoDi (Conditional Diffusion Distillation) & Tính nhất quán khuếch tán có điều kiện – Suy nghĩ:** Chưng cất LDM có điều kiện không cần dữ liệu gốc, dựa trên tính nhất quán PF-ODE, rất thực tiễn.
    *   **2310.13671 – S3 (Synthesis Step by Step) với Error Extrapolation-based Synthesis (EES) – Suy nghĩ:** Sử dụng lỗi của mô hình nhỏ để hướng dẫn LLM sinh dữ liệu tổng hợp chất lượng cao hơn là một cách tiếp cận data synthesis lặp thông minh.
    *   **2310.03214 – FRESH PROMPT với sắp xếp chứng cứ theo thời gian – Suy nghĩ:** Tích hợp chứng cứ từ search engine vào prompt, ưu tiên thông tin mới, giúp LLM trả lời câu hỏi cần kiến thức cập nhật.
    *   **2310.17796 – ControlLLM với Tool Graph (ToG) và DFS search – Suy nghĩ:** Xây dựng đồ thị công cụ và dùng thuật toán tìm kiếm để LLM điều phối các tool đa phương tiện là một cách tiếp cận có cấu trúc.
    *   **2310.13268 – DPM-Solver-v3 với Empirical Model Statistics (EMS) & Pseudo-order solver – Suy nghĩ:** EMS tự động tối ưu parameterization cho ODE solver; pseudo-order solver ổn định ở NFE thấp.
    *   **2310.08541 – Idea2Img với GPT-4V 3 vai trò (sinh/sửa prompt, chọn ảnh, phản hồi) & Bộ nhớ lịch sử thử nghiệm – Suy nghĩ:** Framework tự động khám phá và tối ưu prompt cho T2I model bằng LMM, không cần biết trước về T2I model.
    *   **2310.08529 – GaussianDreamer với Noisy point growing & Color perturbation – Suy nghĩ:** Kết hợp diffusion 3D (khởi tạo) và 2D (tối ưu SDS) trên Gaussian Splatting, các phép perturbation tăng chi tiết.
    *   **2310.13289 – SALMONN với Window-level Q-Former & Few-shot activation tuning – Suy nghĩ:** Q-Former theo cửa sổ xử lý audio dài; activation tuning kích hoạt khả năng cross-modal của MLLM âm thanh.
    *   **2310.11440 – EvalCrafter với SD-Score & Human-alignment bằng hồi quy tuyến tính – Suy nghĩ:** SD-Score phát hiện concept forgetting trong T2V; human-alignment giúp objective metrics khớp đánh giá người dùng.
    *   **2310.10971 – CAML với ELMES Class Encoder & Pre-training đa miền – Suy nghĩ:** Mã hóa nhãn bằng ELMES và pre-training trên nhiều bộ dữ liệu cho phép universal few-shot classification "in-context".
    *   **2310.08579 – Latent Structural Diffusion Model (RGB, depth, normal) & Structure-Guided Refiner – Suy nghĩ:** Đồng thời khử nhiễu đa cấu trúc trong latent space và refiner đa điều kiện cho sinh ảnh người chất lượng cao.
    *   **2310.19512 – VideoCrafter với Spatial & Temporal Transformers & Text-Aligned Rich Image Embedding – Suy nghĩ:** Kiến trúc U-Net 3D tách biệt ST/TT và nhánh I2V với embedding ảnh phong phú giúp tạo video chất lượng từ text/image.
    *   **2310.16045 – Woodpecker pipeline (trích xuất concept, tạo câu hỏi, xác thực visual knowledge, tạo visual claims, sửa lỗi) – Suy nghĩ:** Framework hậu xử lý training-free dùng LLM và vision models để tự động phát hiện và sửa hallucination trong MLLM.
    *   **2310.13332 – Interactive multi-round distillation với student feedback & Self-reflection triplet loss – Suy nghĩ:** Vòng lặp tương tác giữa student LM và teacher LLM, kết hợp học từ lỗi của chính mình, giúp chưng cất khả năng suy luận hiệu quả.
    *   **2310.08740 – Compact screen representation & Staged plan-and-follow / Structured thought management – Suy nghĩ:** Các kỹ thuật này cho phép LLM zero-shot điều khiển UI máy tính hiệu quả hơn, giảm số lần gọi LLM và quản lý "suy nghĩ" tốt hơn.
    *   **2310.08465 – MotionDirector với Dual-Path LoRAs & Appearance-debiased temporal loss – Suy nghĩ:** Tách biệt học appearance và motion trong video diffusion, cho phép tùy biến chuyển động và áp dụng lên nhiều appearance khác nhau.
    *   **2310.01714 – Analogical prompting với Self-generated exemplars & tutorials – Suy nghĩ:** LLM tự sinh ví dụ và kiến thức liên quan "on-the-fly" để giải quyết vấn đề, không cần few-shot examples thủ công.
    *   **2310.17075 – HyperFields (dynamic hypernetwork dự đoán trọng số NeRF) & NeRF distillation loss – Suy nghĩ:** Sinh trọng số NeRF động theo văn bản và activation, cho phép zero-shot/few-shot text-to-3D nhanh.
    *   **2310.17022 – Controlled Decoding (CD) với CD-Q (Bellman update) & Blockwise CD – Suy nghĩ:** Framework RL tokenwise giữ base LM cố định, điều khiển sinh văn bản qua prefix scorer, hỗ trợ multi-objective.
    *   **2310.12404 – Loop Copilot với Global Attribute Table – Suy nghĩ:** LLM điều phối nhiều AI âm nhạc chuyên biệt, GAT duy trì trạng thái nhạc qua nhiều vòng chỉnh sửa, rất thực tiễn cho sáng tạo tương tác.
    *   **2310.03734 – ITIT (Image-Text-Image-Text cycle consistency) & Kiến trúc encoder chung, decoder riêng – Suy nghĩ:** Huấn luyện VLM sinh trên dữ liệu không cặp đôi bằng cycle consistency, giảm đáng kể nhu cầu dữ liệu cặp đôi.
    *   **2310.17752 – PockEngine với sparse backpropagation (graph pruning, dead code elimination) – Suy nghĩ:** Framework biên dịch cho fine-tuning thưa hiệu quả trên thiết bị biên, hiện thực hóa lợi ích của sparse BP.
    *   **2310.16836 – Pre-shifted exponent bias cho FPQ & Grid search tham số định dạng/clip – Suy nghĩ:** Kỹ thuật reparameterize exponent bias giúp lượng tử hóa FP4 hiệu quả cho cả trọng số và kích hoạt LLM.
    *   **2310.12963 – Self-verification dạng entailment & Router POMDP cho LLM cascade – Suy nghĩ:** Ước tính độ tin cậy của SLM bằng entailment và dùng POMDP để định tuyến truy vấn giữa các LLM black-box.
    *   **2310.09139 – CONSENSUS GAME & EQUILIBRIUM-RANKING (no-regret learning) – Suy nghĩ:** Framework lý thuyết trò chơi cho giải mã ngôn ngữ, kết hợp generative và discriminative LM mà không cần fine-tuning.
    *   **2310.17784 – FLLM với Abductive Augmentation Reasoning (AAR: FAP, FAE, FADOM) – Suy nghĩ:** FLLM tiền xử lý dữ liệu tài chính; AAR tự động sinh, kiểm tra, hiệu chỉnh nhãn giả để tăng cường dữ liệu huấn luyện.
    *   **2310.13227 – ToolChain* (A* search cho LLM tool use) & Hàm chi phí/heuristic kết hợp long-term memory, self-consistency, imagination score – Suy nghĩ:** Thuật toán A* hiệu quả để LLM tìm kiếm trong không gian hành động của công cụ.
    *   **2310.13119 – DreamSpace (top-down panoramic texturing) & Dual texture alignment / Implicit texture imitating – Suy nghĩ:** Pipeline toàn diện cho texturing lưới cảnh 360° từ điểm nhìn trung tâm, xử lý tốt vùng che khuất.
    *   **2310.12962 – Emulated Fine-Tuning (EFT) & Up-Scaling / Test-time blending – Suy nghĩ:** Phân tách quy mô pretrain và finetune, cho phép giả lập fine-tuning mô hình lớn bằng mô hình nhỏ và điều chỉnh hành vi runtime.
    *   **2310.12274 – Multi-Concept Prompt Learning (MCPL) với Attention Masking & Prompts Contrastive Loss – Suy nghĩ:** Học đồng thời nhiều concept prompt từ một câu-ảnh không cần annotation, rất hiệu quả về bộ nhớ.
    *   **2310.10645 – ITP (Iterative Task Planning) với high-level planner & low-level executor (LLM) – Suy nghĩ:** Khung LLM hai cấp cho robot, tự động chuyển skill Python thành API và replanning động.
    *   **2310.10837 – σ-MoE (sigmoid gating, Top-K experts) & PKM cải tiến (ReLU, Top-K) – Suy nghĩ:** Các cải tiến này cho MoE và PKM giúp xấp xỉ MLP hiệu quả hơn trong Transformer.
    *   **2310.09983 – FARZI (Factorized Autoregressive Data Distillation) & Reverse-mode differentiation cho Adam – Suy nghĩ:** Chưng cất dữ liệu tự hồi quy thành latent summary, thuật toán đạo hàm ngược hiệu quả cho Adam giúp tối ưu meta-learning.
    *   **2310.08715 – SUTLM (Speech-Unit and Text Language Model) & CRA/PELM metrics / CST/AST mixing – Suy nghĩ:** Mô hình LM chung cho cả speech unit và text, cùng các kỹ thuật trộn và metric đánh giá cross-modal mới.
    *   **2310.19102 – Atom (4-bit quantization) với Mixed-precision channel reordering & Fused group quantization – Suy nghĩ:** Giải pháp lượng tử hóa 4-bit toàn diện cho LLM serving, tối ưu phần cứng và xử lý outlier hiệu quả.

4.  **GAPS_AND_OPPORTUNITIES**
    *   **Hiệu quả và Khả năng mở rộng của LLM:**
        *   *Gaps:* Nhiều phương pháp PEFT (VeRA, LoftQ) dù giảm tham số trainable nhưng vẫn cần đánh giá kỹ về hiệu năng trên các tác vụ phức tạp và khả năng kết hợp với các kỹ thuật nén khác. Chi phí huấn luyện và inference cho các mô hình cực lớn (ví dụ MoE trong 2310.16795) vẫn là rào cản. Các phương pháp lượng tử hóa (BitNet, FP8, FPQ, TEQ, Atom) cần khám phá sâu hơn về ảnh hưởng đến các khả năng phức tạp như reasoning và in-context learning ở các bit-width cực thấp.
        *   *Opportunities:* Phát triển các kỹ thuật PEFT/quantization mới ít ảnh hưởng đến downstream performance hơn nữa. Nghiên cứu kiến trúc LLM mới vốn đã hiệu quả (ví dụ: non-Transformer). Tối ưu hóa thuật toán và phần cứng cho sparse/quantized LLM. Khám phá giới hạn của việc scaling down (ví dụ: sub-1-bit) mà vẫn giữ được năng lực.
    *   **Multimodality (Đa phương thức):**
        *   *Gaps:* Hầu hết các MLLM (PaLI-3, MiniGPT-v2, SALMONN) vẫn dựa trên việc kết nối các encoder chuyên biệt với một LLM text-based. Sự tích hợp sâu và học từ đầu (end-to-end) cho nhiều modal (đặc biệt là video, audio phức tạp, 3D) còn hạn chế. Vấn đề hallucination trong MLLM (2310.16045, 2310.14566) vẫn là thách thức lớn. Đánh giá MLLM (2310.11440, 2310.14566, 2310.16534, 2310.19061) cần các benchmark toàn diện và phương pháp tự động đáng tin cậy hơn.
        *   *Opportunities:* Kiến trúc MLLM thực sự "multimodal-native". Phương pháp hiệu quả để pretrain MLLM trên dữ liệu đa phương thức quy mô lớn, có thể tận dụng dữ liệu không cặp đôi (như 2310.03734). Kỹ thuật grounding tốt hơn giữa các modal. Giải pháp mạnh mẽ hơn cho hallucination mitigation trong MLLM. Phát triển benchmark động và tương tác cho MLLM.
    *   **Generative AI (Đặc biệt là 3D và Video):**
        *   *Gaps:* Tạo 3D (Wonder3D, DreamCraft3D, GaussianDreamer, Progressive3D, HyperFields) và video (VideoCrafter, MotionDirector, FreeNoise) chất lượng cao, nhất quán, có thể điều khiển và dài vẫn là bài toán khó. Các phương pháp hiện tại thường tốn kém tài nguyên, giới hạn độ phân giải/thời lượng, hoặc gặp vấn đề về tính nhất quán (temporal, view). Điều khiển chi tiết (ví dụ: tương tác vật lý, biểu cảm phức tạp) còn hạn chế.
        *   *Opportunities:* Diffusion model hiệu quả hơn cho 3D/video. Phương pháp kết hợp các representation khác nhau (explicit meshes, implicit fields, Gaussian splatting). Kỹ thuật điều khiển (control signals) tinh vi hơn. Tận dụng LLM/VLM để lập kế hoạch và điều khiển quá trình sinh 3D/video (như 2310.10625).
    *   **Reasoning, Planning, và Agent Capabilities:**
        *   *Gaps:* Khả năng suy luận phức tạp, đa bước, và lập kế hoạch dài hạn của LLM (đặc biệt trong môi trường tương tác như 2310.08588, 2310.10645, 2310.08740, 2310.13227) vẫn cần cải thiện. Việc tích hợp tool use một cách linh hoạt và hiệu quả (ControlLLM, ToolChain*) còn nhiều thách thức. Khả năng tự sửa lỗi nội tại (2310.01798) của LLM còn yếu.
        *   *Opportunities:* Phát triển kiến trúc và phương pháp huấn luyện LLM tăng cường khả năng reasoning và planning. Framework agent tổng quát hơn, có khả năng học hỏi từ tương tác và tự cải thiện (như EUREKA, TRIPOST, PIT). Nghiên cứu sâu hơn về mechanistic interpretability của ICL (2310.15916) để cải thiện reasoning.
    *   **Alignment, Safety, và Evaluation:**
        *   *Gaps:* RLHF có thể bị ảnh hưởng bởi length bias (2310.03716). Việc đảm bảo an toàn và tránh các hành vi không mong muốn một cách có hệ thống (Safe RLHF) vẫn là một lĩnh vực đang phát triển. Đánh giá LLM (PROMETHEUS, JudgeLM, SmartPlay, DEsignBench, HALLUSION BENCH, WIKIMIA) cần toàn diện hơn, ít bị "gaming" và phản ánh đúng năng lực thực tế trong các ứng dụng cụ thể (tài chính 2310.08678, y tế 2310.19061).
        *   *Opportunities:* Phương pháp alignment mới hiệu quả hơn, ít tốn kém hơn RLHF và ít bias hơn. Kỹ thuật "red teaming" tự động và benchmark an toàn mạnh mẽ hơn. Phát triển các metric đánh giá LLM/MLLM có khả năng diễn giải và chống lại các chiến lược đối phó.
    *   **Data-Centric AI:**
        *   *Gaps:* Chất lượng và sự đa dạng của dữ liệu huấn luyện vẫn là yếu tố then chốt. Các phương pháp tạo dữ liệu tổng hợp (RECAP, CommonCanvas, S3, MathCodeInstruct, TeacherData-2M, Auto-Instruct, FARZI) cần cải thiện về độ tin cậy và khả năng bao phủ các trường hợp hiếm.
        *   *Opportunities:* Kỹ thuật data augmentation và synthesis thông minh hơn, có thể tự động điều chỉnh theo nhu cầu của mô hình. Phương pháp tận dụng dữ liệu không nhãn/yếu nhãn hiệu quả hơn (như ITIT). Nghiên cứu về "data pruning" và "data selection" để huấn luyện hiệu quả hơn.
    *   **On-Device AI và Efficiency:**
        *   *Gaps:* Triển khai các mô hình lớn trên thiết bị biên với tài nguyên hạn chế (PockEngine, Atom) vẫn là một thách thức lớn, đòi hỏi sự đồng thiết kế giữa thuật toán và phần cứng.
        *   *Opportunities:* Các kỹ thuật nén (quantization, pruning, distillation) mới, đặc biệt cho các kiến trúc mới. Framework biên dịch và tối ưu hóa chuyên biệt cho on-device training/inference.
    *   **Interpretability và Trustworthiness:**
        *   *Gaps:* Hiểu rõ "tại sao" mô hình đưa ra một quyết định cụ thể (đặc biệt với LLM và MLLM) vẫn còn là một "hộp đen". Các phương pháp như của 2310.15916 (task vector) là bước đầu nhưng cần nhiều hơn nữa.
        *   *Opportunities:* Phát triển các công cụ và kỹ thuật mới để diễn giải hành vi của mô hình. Xây dựng các mô hình vốn đã có tính diễn giải (interpretable by design).

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Adaptive Contextual Sparsity for Multimodal LLMs**
    *   **Motivation:** MLLMs xử lý nhiều loại dữ liệu (text, image, audio, video) dẫn đến chi phí tính toán rất lớn. DEJAVU (2310.17157) cho thấy contextual sparsity hiệu quả cho LLM text. Mở rộng ý tưởng này cho MLLM có thể mang lại lợi ích lớn.
    *   **Key Novelty:** Phát triển predictor không chỉ dựa trên text context mà còn cả đặc trưng từ các modal khác để dự đoán dynamic structured sparsity cho cả các thành phần xử lý cross-modal attention và các encoder/decoder chuyên biệt của MLLM.
    *   **Approach:**
        1.  Thiết kế các lightweight predictors cho từng modality-specific component và cross-modal fusion layers.
        2.  Input của predictor có thể là embedding từ các modal khác nhau và trạng thái ẩn của LLM.
        3.  Huấn luyện predictors để dự đoán head/neuron nào có thể bỏ qua mà ít ảnh hưởng đến output cuối cùng trên các tác vụ MLLM.
        4.  Tích hợp lookahead prediction và sparse kernels chuyên dụng cho MLLM.
    *   **Dataset + Metrics:** Sử dụng các benchmark MLLM như VQA (VQAv2), Image Captioning (COCO), Video QA (MSRVTT-QA). Metrics: Accuracy, BLEU, ROUGE, CIDEr, và đo lường speedup/FLOPs reduction.
    *   **Risk/Feasibility:** Cao. Khó khăn trong việc thiết kế predictor hiệu quả cho nhiều modal và sự phức tạp của việc triển khai sparse kernels cho kiến trúc MLLM đa dạng. Tính toán offline để huấn luyện predictor có thể tốn kém.

    ✨ **Idea 2: Self-Correcting Generative Agents for Complex 3D Scene Creation (Interdisciplinary: LLM Agents + 3D Generation + Robotics Principles)**
    *   **Motivation:** Tạo cảnh 3D phức tạp, có ý nghĩa từ text (như 2310.12945, 2310.16818, 2310.08529) vẫn cần nhiều cải tiến về tính nhất quán, khả năng tương tác và tuân thủ các ràng buộc vật lý/logic. LLM agents (2310.12823, 2310.08588) có khả năng lập kế hoạch và sử dụng công cụ.
    *   **Key Novelty:** Một hệ thống LLM agent đa vai trò (Planner, Modeler, Physicist, Critic) sử dụng các công cụ tạo 3D (diffusion, procedural), công cụ mô phỏng vật lý, và khả năng tự phản ánh/sửa lỗi (như SELF-RAG 2310.11511, Woodpecker 2310.16045, PIT 2310.00898) để lặp đi lặp lại việc tạo và tinh chỉnh cảnh 3D cho đến khi đạt yêu cầu.
    *   **Approach:**
        1.  **Planner Agent:** Phân rã yêu cầu text phức tạp thành các sub-tasks và đối tượng 3D cần tạo.
        2.  **Modeler Agent:** Sử dụng các T23D models (GaussianDreamer, DreamCraft3D) hoặc procedural generation (3D-GPT) để tạo các thành phần.
        3.  **Physicist Agent:** (Có thể là một simulator tích hợp) Kiểm tra tính hợp lý vật lý, tương tác giữa các đối tượng.
        4.  **Critic Agent:** Sử dụng VLM (như GPT-4V) để đánh giá sự phù hợp của cảnh 3D với prompt ban đầu, tính nhất quán, và các lỗi tiềm ẩn. Sinh feedback cho Planner và Modeler.
        5.  Vòng lặp refine dựa trên feedback, có thể sử dụng các kỹ thuật như EUREKA (2310.12931) để tinh chỉnh các tham số của công cụ tạo hình.
    *   **Dataset + Metrics:** Không có dataset chuẩn. Đánh giá dựa trên human evaluation về chất lượng, tính nhất quán, sự tuân thủ prompt, và tính hợp lý vật lý. Có thể phát triển benchmark mới với các yêu cầu phức tạp.
    *   **Risk/Feasibility:** Rất cao (Moon-shot). Đòi hỏi tích hợp nhiều hệ thống phức tạp, khả năng giao tiếp hiệu quả giữa các agent, và VLM đủ mạnh để làm Critic. Chi phí tính toán cực lớn.

    ✨ **Idea 3: Zero-Shot Cross-Lingual Table Understanding via Table-Tuned LLM with In-Context Pretraining**
    *   **Motivation:** Table-GPT (2310.09263) cho thấy tiềm năng của việc "table-tuning" LLM cho các tác vụ hiểu bảng. IN-CONTEXT PRETRAINING (2310.10638) cải thiện khả năng học ngữ cảnh dài bằng cách sắp xếp tài liệu liên quan. Kết hợp hai ý tưởng này có thể tạo ra LLM hiểu bảng tốt hơn, đặc biệt là cho các ngôn ngữ ít tài nguyên.
    *   **Key Novelty:** Huấn luyện một LLM trên dữ liệu bảng đa ngôn ngữ, sử dụng IN-CONTEXT PRETRAINING trong đó các "tài liệu" là các bảng và mô tả/câu hỏi liên quan đến bảng đó, được nhóm theo ngôn ngữ hoặc chủ đề. Mục tiêu là LLM có thể zero-shot thực hiện các table tasks trên ngôn ngữ mới chỉ bằng cách cung cấp một vài ví dụ bảng và tác vụ bằng ngôn ngữ đó trong context.
    *   **Approach:**
        1.  Thu thập dataset bảng đa ngôn ngữ (ví dụ: từ Wikipedia, các báo cáo tài chính công khai đa ngôn ngữ).
        2.  Áp dụng IN-CONTEXT PRETRAINING: Nhóm các bảng và các cặp (instruction, completion) liên quan đến bảng theo ngôn ngữ hoặc chủ đề. Sử dụng thuật toán tương tự TSP để sắp xếp chuỗi các "table documents" này.
        3.  Tiếp tục huấn luyện một LLM đa ngôn ngữ trên chuỗi dữ liệu bảng đã sắp xếp này.
        4.  Fine-tune LLM bằng phương pháp table-tuning (như 2310.09263) trên một tập hợp con các ngôn ngữ có nhiều tài nguyên.
        5.  Đánh giá khả năng zero-shot/few-shot trên các ngôn ngữ chưa từng thấy trong giai đoạn fine-tuning.
    *   **Dataset + Metrics:** Sử dụng các benchmark hiểu bảng hiện có và mở rộng sang phiên bản đa ngôn ngữ (ví dụ: WikiTableQuestions, Spider dịch sang nhiều ngôn ngữ). Metrics: Exact Match, F1 score cho các tác vụ như table QA, NL-to-SQL trên bảng.
    *   **Risk/Feasibility:** Trung bình đến Cao. Thu thập và chuẩn hóa dataset bảng đa ngôn ngữ chất lượng cao là một thách thức. Hiệu quả của IN-CONTEXT PRETRAINING cho dữ liệu bảng cần được kiểm chứng. Khả năng zero-shot cross-lingual có thể bị hạn chế bởi sự khác biệt cấu trúc giữa các ngôn ngữ.

    ✨ **Idea 4: Continual Learning for LLM-based Evaluators with Dynamic Rubric Adaptation**
    *   **Motivation:** PROMETHEUS (2310.08491) và JudgeLM (2310.17631) là các LLM evaluator mạnh mẽ nhưng dựa trên rubric cố định hoặc dữ liệu huấn luyện tĩnh. Trong thực tế, tiêu chí đánh giá và các loại lỗi mới của LLM liên tục xuất hiện.
    *   **Key Novelty:** Phát triển một LLM evaluator có khả năng học liên tục (continual learning) từ các feedback mới và tự động điều chỉnh/mở rộng rubric đánh giá của nó. Sử dụng các kỹ thuật từ TIC-CLIP (2310.16226) cho replay buffer và các phương pháp phát hiện OOD để xác định khi nào cần cập nhật rubric.
    *   **Approach:**
        1.  Huấn luyện một LLM evaluator ban đầu (như PROMETHEUS) trên FEEDBACK COLLECTION.
        2.  Thiết lập một streaming protocol cho dữ liệu đánh giá mới (ví dụ: từ người dùng, từ các mô hình mới được kiểm tra).
        3.  Sử dụng một replay buffer (chiến lược All hoặc Exp từ TIC-CLIP) để lưu trữ các mẫu đánh giá cũ và mới.
        4.  Khi có dữ liệu mới, nếu mô hình evaluator không chắc chắn hoặc phát hiện mẫu OOD (ví dụ: loại lỗi mới, yêu cầu đánh giá mới), kích hoạt một module "Rubric Adaptation" (có thể là một LLM khác hoặc tương tác người dùng) để cập nhật/mở rộng rubric.
        5.  Tiếp tục huấn luyện LLM evaluator trên dữ liệu trộn từ replay buffer và rubric đã cập nhật, chú trọng vào việc không quên các tiêu chí cũ.
    *   **Dataset + Metrics:** Bắt đầu với FEEDBACK COLLECTION. Dữ liệu mới sẽ được thu thập liên tục. Metrics: Pearson correlation với đánh giá của con người, tính nhất quán của evaluator theo thời gian, khả năng phát hiện lỗi mới.
    *   **Risk/Feasibility:** Cao. Đảm bảo tính ổn định và tránh catastrophic forgetting trong continual learning là khó. Tự động cập nhật rubric một cách đáng tin cậy là thách thức lớn, có thể cần human-in-the-loop.

6.  **READING_LIST**

    *   2310.11453 – BitNet · Giới thiệu LLM 1-bit huấn luyện từ đầu, đột phá về hiệu quả.
    *   2310.11511 – SELF-RAG · Framework RAG thông minh với khả năng tự truy vấn và tự đánh giá.
    *   2310.12931 – EUREKA · Phương pháp dùng LLM zero-shot để tự động sinh hàm thưởng cho RL, rất sáng tạo.
    *   2310.00426 – PIXART-α · Kiến trúc T2I Transformer hiệu quả và chiến lược huấn luyện data-centric.
    *   2310.03714 – DSPy · Framework lập trình LM mới, tự động tối ưu pipeline, thay đổi cách tiếp cận ứng dụng LM.
    *   2310.17157 – DEJAVU · Kỹ thuật contextual sparsity động giúp tăng tốc LLM inference đáng kể.
    *   2310.12773 – Safe RLHF · Phương pháp có nguyên tắc để xây dựng LLM vừa hữu ích vừa an toàn.
    *   2310.15008 – Wonder3D · Tạo 3D chất lượng cao từ ảnh đơn với kiến trúc diffusion đa miền sáng tạo.
    *   2310.03734 – ITIT · Huấn luyện VLM sinh trên dữ liệu không cặp đôi, tiềm năng lớn cho việc giảm chi phí dữ liệu.
    *   2310.18313 – FP8 Training · Khung FP8 mixed-precision toàn diện cho huấn luyện LLM hiệu quả.

7.  **META_REFLECTION**
    Tập hợp các bài báo tháng 10 năm 2023 cho thấy một số xu hướng phát triển AI nổi bật:
    *   **Tối ưu hóa hiệu quả LLM:** Một lượng lớn nghiên cứu tập trung vào việc làm cho LLM trở nên nhẹ hơn, nhanh hơn và tiết kiệm tài nguyên hơn thông qua các kỹ thuật lượng tử hóa (BitNet, FP8, LoftQ, QMoE, FPQ, TEQ, Atom), PEFT (VeRA, LoRAShear), và inference hiệu quả (DEJAVU, EFT). Điều này phản ánh nhu cầu cấp thiết về việc triển khai LLM trên các thiết bị hạn chế và giảm chi phí vận hành.
    *   **Nâng cao khả năng Reasoning, Planning và Agent của LLM:** Nhiều công trình khám phá cách cải thiện khả năng suy luận (LLEMMA, MathCodeInstruct, Analogical Prompting), lập kế hoạch (ITP, VLP, ToolChain*), và xây dựng các agent tự trị hơn (AgentTuning, Octopus, EUREKA, ControlLLM, MusicAgent, Idea2Img). Xu hướng này cho thấy tham vọng biến LLM thành những thực thể có khả năng hành động và giải quyết vấn đề phức tạp.
    *   **Bùng nổ Generative AI đa phương thức, đặc biệt là 3D và Video:** Các mô hình Diffusion tiếp tục thống trị lĩnh vực sinh ảnh, video và 3D, với nhiều cải tiến về chất lượng, khả năng điều khiển, và hiệu quả (PIXART-α, Kandinsky, MDM, RECAP, CommonCanvas, Wonder3D, DreamCraft3D, GaussianDreamer, VideoCrafter, MotionDirector). Sự quan tâm đến việc tạo nội dung 3D từ text hoặc ảnh đơn là rất lớn.
    *   **Tăng cường tính tin cậy và an toàn của AI:** Các vấn đề về hallucination (SELF-RAG, Woodpecker, HALLUSION BENCH), alignment (Safe RLHF, PIT, Tuna, length bias analysis), và phát hiện dữ liệu (WIKIMIA) được chú trọng, cho thấy sự trưởng thành của lĩnh vực khi tập trung vào việc xây dựng AI đáng tin cậy và có trách nhiệm.
    *   **Data-Centric AI và tự cải thiện:** Tầm quan trọng của dữ liệu chất lượng cao được nhấn mạnh qua các phương pháp tái chú thích (RECAP), tạo dữ liệu tổng hợp (CommonCanvas, S3, TeacherData-2M), và các framework tự cải thiện dựa trên dữ liệu (PIT, TRIPOST, CodeChain). DSPy (2310.03714) cũng cho thấy hướng tự động hóa việc tối ưu pipeline dựa trên dữ liệu.
    *   **Hệ thống và Benchmark:** Sự phát triển của các hệ thống phức tạp tích hợp nhiều module AI (MM-VID, ControlLLM, Loop Copilot) và các benchmark chuyên sâu (SmartPlay, ImagenHub, DEsignBench, HALLUSION BENCH, EvalCrafter, WIKIMIA, TIC-CLIP) cho thấy lĩnh vực đang hướng tới việc xây dựng các ứng dụng thực tế hơn và đánh giá AI một cách toàn diện hơn.
    *   **Học hỏi từ dữ liệu không hoàn hảo/ít ỏi:** Các kỹ thuật như tận dụng dữ liệu không cặp đôi (ITIT), zero-shot/few-shot learning (EUREKA, VLM-RM, CAML, SoM Prompting) và distillation (2310.13332) tiếp tục được khám phá để giảm sự phụ thuộc vào lượng lớn dữ liệu có nhãn hoàn hảo.
