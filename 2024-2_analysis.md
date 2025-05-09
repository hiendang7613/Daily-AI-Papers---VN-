1.  **TOPIC_TREE**

    *   Natural Language Processing (NLP)
        *   Large Language Models (LLMs)
            *   Model Architecture & Efficiency
                *   Efficient Architectures
                    *   Linear Attention & Alternatives
                        *   2402.10644 | Đề xuất ReBased, một kernel bậc hai học được cho linear attention, cải thiện khả năng học trong ngữ cảnh dài.
                        *   2402.04248 | Giới thiệu MambaFormer, kiến trúc lai Mamba và Attention, loại bỏ positional encoding, cải thiện ICL.
                        *   2402.01771 | Đề xuất BlackMamba, kiến trúc kết hợp Mamba (SSM) và MoE MLP, cùng phương pháp khởi tạo Sinkhorn mới.
                    *   Parameter Sharing
                        *   2402.14905 | Đề xuất chia sẻ trọng số tức thời theo khối cho LLM di động, nhấn mạnh kiến trúc sâu-và-hẹp.
                        *   2402.16840 | Giới thiệu MobiLlama, chia sẻ một khối FFN duy nhất giữa các lớp transformer để tạo SLM hiệu quả.
                    *   Mixture-of-Experts (MoE)
                        *   2402.01739 | Phát hành OpenMoE và phân tích chuyên sâu về cơ chế định tuyến trong MoE.
                *   Model Compression
                    *   Quantization
                        *   Extreme Low-bit (1-bit)
                            *   2402.17764 | Đề xuất BitNet b1.58 với lượng tử hóa trọng số `absmean` thành tam phân {-1, 0, 1}.
                            *   2402.04291 | Giới thiệu BiLLM, PTQ 1-bit với xử lý riêng trọng số nổi bật (residual approximation) và không nổi bật (distribution splitting).
                            *   2402.11295 | Đề xuất kiến trúc lớp tuyến tính 1-bit mới (ma trận dấu ±1, hai vector giá trị FP16) và phương pháp khởi tạo SVID.
                        *   Vector Quantization
                            *   2402.15319 | Đề xuất GPTVQ, phương pháp VQ sau huấn luyện dựa trên Hessian cho LLM.
                        *   Delta Quantization
                            *   2402.10193 | Đề xuất BitDelta, lượng tử hóa 1-bit cho delta trọng số sau tinh chỉnh, với scale distillation.
                *   Context Window Extension
                    *   2402.13753 | Đề xuất LongRoPE, khai thác bất đối xứng RoPE và tìm kiếm tiến hóa để mở rộng ngữ cảnh, cùng chiến lược lũy tiến và phục hồi ngữ cảnh ngắn.
                    *   2402.10171 | Đề xuất chiến lược "Per-source Upsampling" và chứng minh hiệu quả huấn luyện liên tục full attention (64K-80K) với ít dữ liệu để đạt ngữ cảnh 128K.
                    *   2402.17463 | Đề xuất Dual Chunk Attention (DCA), kiến trúc chú ý không cần huấn luyện lại để mở rộng ngữ cảnh dài bằng cách xử lý chunk và vị trí RoPE có điều kiện.
                *   Inference Acceleration
                    *   Speculative Decoding
                        *   2402.11131 | Đề xuất Speculative Streaming, hợp nhất dự đoán và xác minh token trong một mô hình duy nhất qua Multi-Stream Attention.
                    *   Efficient Attention Mechanisms
                        *   2402.15220 | Đề xuất ChunkAttention (PAKV cache dạng cây tiền tố và kernel TPP) để chia sẻ KV cache và tối ưu self-attention cho tiền tố chung.
            *   Reasoning
                *   Mathematical Reasoning
                    *   2402.03300 | Đề xuất quy trình chọn lọc dữ liệu toán học lặp lại và thuật toán RL GRPO (loại bỏ critic).
                    *   2402.14830 | Đề xuất quy trình tạo dữ liệu Agent-Instruct và học lặp lại cho SLM giải toán.
                *   Automated Reasoning Structure Discovery
                    *   2402.03620 | Đề xuất SELF-DISCOVER, LLM tự khám phá cấu trúc lý luận phù hợp nhiệm vụ từ các mô-đun nguyên tử.
                *   Decoding Strategies for Reasoning
                    *   2402.10200 | Đề xuất CoT-decoding, gợi mở CoT tiềm ẩn bằng cách khám phá token thay thế top-k ở bước giải mã đầu tiên.
                *   Latent Multi-hop Reasoning Analysis
                    *   2402.16837 | Phân tích và định lượng khả năng suy luận đa bước tiềm ẩn của LLM, giới thiệu chỉ số ENTREC và CNSTSCORE.
                *   Sensitivity to Premise Order
                    *   2402.08939 | Phân tích sự nhạy cảm của LLM với thứ tự tiền đề trong suy luận logic và toán học, giới thiệu bộ dữ liệu R-GSM.
            *   Instruction Tuning & Alignment
                *   Synthetic Data Generation
                    *   2402.13064 | Đề xuất GLAN, tạo dữ liệu chỉ thị tổng quát quy mô lớn từ phân loại tri thức.
                    *   2402.10176 | Đề xuất "Masked Text Solution Prompting" và "fair sampling" để tạo dữ liệu hướng dẫn toán học OpenMathInstruct-1 từ LLM nguồn mở.
                *   Unified Generative & Representational Models
                    *   2402.09906 | Đề xuất GRIT, hợp nhất khả năng sinh văn bản và tạo embedding trong một LLM qua instruction tuning.
                *   Long Context Instruction Tuning
                    *   2401.18058 | Đề xuất quy trình tạo dữ liệu hướng dẫn dài (LongAlign) và loss weighting cho packing.
                *   Online Direct Preference Optimization
                    *   2402.04792 | Đề xuất Online AI Feedback (OAIF) tích hợp phản hồi trực tuyến từ LLM chú thích vào các phương pháp DAP.
                *   Active Exploration for RLHF
                    *   2402.00396 | Áp dụng Double Thompson Sampling (DTS) và Epistemic Neural Network (ENN) để chọn cặp phản hồi hiệu quả trong RLHF.
            *   Knowledge Integration & Continual Learning
                *   2402.12847 | Đề xuất Pre-Instruction-Tuning (PIT), đảo ngược thứ tự huấn luyện (QA trước, rồi Docs) để cải thiện hấp thụ kiến thức mới.
            *   Domain-Specific LLMs
                *   Financial Multimodal LLMs
                    *   2402.10986 | Xây dựng FinTral (đa phương thức tài chính dựa trên Mistral-7B), FinSet (bộ dữ liệu/benchmark tài chính).
                *   Music Generation and Understanding
                    *   2402.16153 | Đề xuất ChatMusician (LLaMA2 với ABC notation), MusicPile (bộ dữ liệu), MusicTheoryBench (benchmark).
                *   Chemical Language Models
                    *   2402.06852 | Xây dựng ChemLLM, ChemData (dữ liệu hướng dẫn hóa học từ template), ChemBench (benchmark trắc nghiệm hóa học).
            *   Multilingual LLMs
                *   Training Strategies
                    *   2402.00786 | Đề xuất chiến lược huấn luyện song ngữ Pháp-Anh cân bằng (1:1) và tokenizer tùy chỉnh.
                *   Instruction Finetuning
                    *   2402.07827 | Giới thiệu mô hình Aya, tỉa dữ liệu xP3x dựa trên con người và chưng cất ngữ cảnh an toàn đa ngôn ngữ.
            *   Model Fusion
                *   2402.16107 | Đề xuất FUSIONCHAT (hợp nhất-rồi-trộn LLM không đồng nhất) và VARM (trọng số trộn dựa trên tỷ lệ biến thiên tham số).
            *   Performance Enhancement
                *   Ensemble Methods and Scaling Laws
                    *   2402.05120 | Hệ thống hóa Agent Forest (lấy mẫu và bỏ phiếu đa số) cho LLM, làm rõ tính trực giao với các kỹ thuật khác.
            *   Scaling Laws for Finetuning
                *   2402.17193 | Đề xuất định luật nhân tính kết hợp (kích thước dữ liệu tinh chỉnh và yếu tố khác) và quy trình khớp tham số chung cho tinh chỉnh LLM.
        *   Code Generation
            *   Iterative Code Refinement
                *   2402.14658 | Đề xuất bộ dữ liệu Code-Feedback (tương tác đa lượt với phản hồi thực thi/người dùng) và OpenCodeInterpreter.
            *   Reinforcement Learning Optimization
                *   2402.01391 | Đề xuất StepCoder (CCCS - curriculum code completion, FGO - fine-grained optimization) cho RL trong sinh mã.
        *   Information Retrieval
            *   Retrieval-Augmented Generation (RAG)
                *   2401.18059 | Đề xuất RAPTOR, xây dựng cây phân cấp cho văn bản dài bằng phân cụm và tóm tắt đệ quy để truy xuất.
        *   Sequence Modeling (General)
            *   Long Sequence Processing
                *   2402.10790 | Đề xuất RMT-R, bổ sung self-retrieval vào Recurrent Memory Transformer, và benchmark BABILong.
            *   Comparative Analysis (Transformers vs SSMs)
                *   2402.01032 | Phân tích hạn chế của GSSM so với Transformer trong sao chép, đề xuất Hard-ALiBi.
        *   Data Resources & Curation
            *   Pre-training Corpora
                *   2402.00159 | Phát hành Dolma (3T token tiếng Anh) và Dolma Toolkit, hệ thống hóa quy trình quản lý dữ liệu.
            *   Multilingual Instruction Tuning Datasets
                *   2402.06619 | Giới thiệu Aya Dataset (chú thích thủ công), Aya Collection (tổng hợp/mở rộng), Aya Annotation Platform, Aya Evaluation Suite.
            *   Data Selection Strategies
                *   2402.09668 | Đề xuất ASK-LLM (lấy mẫu dựa trên chất lượng từ LLM proxy) và DENSITY (lấy mẫu dựa trên độ bao phủ với IPS).
        *   Software & Libraries
            *   LLM Workflow Management & Reproducibility
                *   2402.10379 | Giới thiệu DataDreamer, thư viện Python tích hợp tạo dữ liệu, huấn luyện, đánh giá LLM với trọng tâm tái lập.
        *   Statistical Language Models
            *   Large-Scale n-gram Models
                *   2401.17377 | Đề xuất mô hình ∞-gram (backoff dựa trên mẫu số) và engine Infini-gram (dựa trên suffix array) cho dữ liệu nghìn tỷ token.
        *   Evaluation and Benchmarking
            *   Contextual Understanding Evaluation
                *   2402.00858 | Tổng hợp benchmark đánh giá hiểu ngữ cảnh LLM (CR, DST, Discourse, QR) và phương pháp chuyển đổi tác vụ cho ICL.
    *   Computer Vision (CV)
        *   Generative Models
            *   Image Generation
                *   Text-to-Image Synthesis
                    *   Consistent Subject Generation
                        *   2402.03286 | Đề xuất ConsiStory (SDSA, Query Blending, Feature Injection) để sinh ảnh chủ thể nhất quán, training-free.
                    *   Model Adaptation / Fine-tuning
                        *   Parameter-Efficient Fine-Tuning (PEFT)
                            *   2402.17412 | Áp dụng DiffuseKronA (tích Kronecker) cho cá nhân hóa mô hình khuếch tán T2I.
                    *   Multi-Concept Composition
                        *   2402.16843 | Đề xuất LoRA Switch và LoRA Composite (không cần huấn luyện) để tổng hợp nhiều LoRA trong quá trình giải mã khuếch tán.
                *   Diffusion Models (General)
                    *   Model Compression & Acceleration
                        *   Diffusion Distillation
                            *   2402.13929 | Đề xuất chưng cất đối nghịch liên tục cho SDXL (SDXL-Lightning) với discriminator dựa trên UNet và mục tiêu có điều kiện.
                    *   Fine-Tuning Strategies
                        *   2402.10210 | Đề xuất SPIN-Diffusion, tinh chỉnh mô hình khuếch tán dựa trên self-play trên quỹ đạo khuếch tán, không cần dữ liệu ưu tiên.
            *   Video Generation
                *   Text-to-Video Synthesis
                    *   2402.17177 | Tổng quan về Sora, hệ thống hóa kiến trúc DiT, spacetime latent patches, và vai trò của LLM tăng cường prompt.
                    *   2402.14797 | Đề xuất khung EDM sửa đổi cho video độ phân giải cao (σ_in), chiến lược huấn luyện ảnh-video (fps thay đổi), và kiến trúc Transformer dựa trên FIT mở rộng.
                *   Image-to-Video Generation
                    *   2402.04324 | Đề xuất ConsistI2V với điều kiện hóa không-thời gian dựa trên khung hình đầu tiên và khởi tạo nhiễu FrameInit.
                *   Human-centric Video Generation
                    *   Audio-driven Talking Head Generation
                        *   2402.17485 | Đề xuất EMO, sinh video chân dung nói chuyện từ âm thanh và ảnh, dùng diffusion model, ReferenceNet, Audio Layers.
                *   Interactive Video Generation / World Models from Video
                    *   2402.15391 | Đề xuất Genie (ST-ViViT encoder, Latent Action Model, ST-transformer dynamics model) học hành động tiềm ẩn không giám sát để tạo môi trường tương tác từ video.
                *   Conditional Video Generation
                    *   Identity-Specific Video Generation
                        *   2402.09368 | Đề xuất Video Custom Diffusion (VCD) với 3D Gaussian Noise Prior, ID module cải tiến, và Face/Tiled VCD.
                    *   Motion Control with Bounding Boxes
                        *   2402.01566 | Đề xuất Boximator, mô-đun điều khiển plug-in với ràng buộc hộp cứng/mềm và kỹ thuật self-tracking.
                    *   Efficient & Stylized Video Synthesis
                        *   2402.00769 | Đề xuất AnimateLCM, tách rời học phong cách (từ ảnh) và tăng tốc sinh video (ảnh rồi chuyển động) qua hợp nhất trọng số.
            *   3D Generation
                *   Feed-forward 3D Reconstruction
                    *   Multi-view Gaussian Generation
                        *   2402.05054 | Đề xuất LGM, tạo 3D Gaussian độ phân giải cao từ ảnh đa góc nhìn bằng U-Net bất đối xứng và giải mã pixel-thành-Gaussian.
        *   Object Detection
            *   Real-time Object Detection
                *   2402.13616 | Đề xuất PGI (Thông tin Gradient Lập trình được) và kiến trúc GELAN (Generalized Efficient Layer Aggregation Network).
        *   Image Segmentation
            *   Foundation Models
                *   Segment Anything Model Acceleration
                    *   2402.05008 | Đề xuất EfficientViT-SAM, thay thế encoder SAM bằng EfficientViT và quy trình huấn luyện 2 giai đoạn (chưng cất + end-to-end).
        *   Video Understanding
            *   Video Captioning
                *   Hierarchical Video Captioning
                    *   2402.13250 | Đề xuất Video ReCap, kiến trúc video-ngôn ngữ đệ quy và học theo chương trình phân cấp để tạo phụ đề đa cấp cho video dài.
            *   Representation Learning
                *   Video Foundation Model Pre-training
                    *   2402.13217 | Đề xuất VideoPrism, tiền huấn luyện hai giai đoạn (tương phản video-văn bản, rồi MVM cải tiến với chưng cất global-local và xáo trộn token).
        *   Egocentric Vision
            *   Multimodal Egocentric Datasets
                *   2402.13349 | Công bố bộ dữ liệu AEA (Aria Everyday Activities) đa phương thức từ kính Project Aria với dữ liệu tri giác máy cải tiến.
        *   Flexible Resolution Processing
            *   Transformer-based Diffusion Models
                *   2402.12376 | Đề xuất FiT (Flexible Vision Transformer) với 2D RoPE, Masked MHSA, SwiGLU và kỹ thuật ngoại suy VisionNTK/YaRN để xử lý ảnh độ phân giải/tỷ lệ đa dạng.
    *   Multimodal AI
        *   Vision-Language Models (VLMs)
            *   UI and Infographics Understanding
                *   2402.04615 | Đề xuất ScreenAI, screen schema (biểu diễn UI/infographics), quy trình tạo dữ liệu tiền huấn luyện bằng LLM, và vá ảnh linh hoạt.
            *   Long Context Modeling
                *   2402.08268 | Huấn luyện Transformer 1M token cho văn bản và video, sử dụng Blockwise RingAttention, tạo dữ liệu QA tổng hợp, masked sequence packing.
            *   Visual Prompting & Instruction Tuning
                *   Object-level Understanding in VLMs
                    *   2402.11248 | Đề xuất CoLLaVO với Crayon Prompt (bản đồ màu panoptic + truy vấn nhúng) và Dual QLoRA (học xen kẽ hiểu đối tượng và tác vụ VL).
            *   Multilingual Vision-Language Models
                *   2402.14818 | Đề xuất PALO, quy trình bán tự động tạo dữ liệu hướng dẫn đa phương thức đa ngôn ngữ và benchmark đa ngôn ngữ mới.
            *   Controllable Generation & Adaptation
                *   2402.05140 | Đề xuất TAG-LLM, kiến trúc thẻ nhúng (miền, chức năng) và quy trình huấn luyện 3 giai đoạn để điều hướng LLM cho miền/tác vụ chuyên biệt.
        *   Multimodal Large Language Models (MLLMs)
            *   Any-to-Any Generation
                *   Discrete Representation Unification
                    *   2402.12226 | Đề xuất AnyGPT, kiến trúc MLLM any-to-any dựa trên token hóa rời rạc thống nhất nhiều phương tiện và khung tạo sinh hai giai đoạn.
        *   Text-based VQA
            *   Hybrid On-Device/Cloud Systems
                *   2402.08017 | Đề xuất Lumos, hệ thống lai (STR trên thiết bị, MM-LLM trên cloud) cho VQA giàu văn bản, với phát hiện ROI và tối ưu STR.
        *   Vision-Language Pre-training
            *   Contrastive Learning
                *   Large-Scale Model Scaling
                    *   2402.04252 | Huấn luyện EVA-CLIP-18B (CLIP 18 tỷ tham số) theo triết lý "yếu-dạy-mạnh" của EVA.
        *   Multi-Concept Personalization Data Generation
            *   2402.15504 | Đề xuất Gen4Gen, pipeline bán tự động dùng chuỗi mô hình nền tảng để tạo dữ liệu huấn luyện cá nhân hóa đa khái niệm, cùng chỉ số CP-CLIP/TI-CLIP.
    *   Machine Learning (ML)
        *   Generative Models (General)
            *   Diffusion Models
                *   Application in Neural Network Parameter Generation
                    *   2402.13144 | Đề xuất p-diff, sinh tham số mạng nơ-ron bằng latent diffusion model và autoencoder tham số có nhiễu.
            *   Audio
                *   Unsupervised Audio Editing
                    *   2402.10009 | Đề xuất ZEUS, chỉnh sửa âm thanh không giám sát zero-shot bằng nhiễu loạn khuếch tán ngược theo hướng PC của hiệp phương sai hậu nghiệm.
        *   Reinforcement Learning (RL)
            *   Deep Reinforcement Learning
                *   Network Architectures
                    *   Mixture of Experts for Parameter Scaling
                        *   2402.08609 | Tích hợp Soft MoE vào lớp avant-dernier của mạng giá trị trong Deep RL (DQN, Rainbow) để tăng hiệu năng theo số expert.
        *   Model Pre-training
            *   Data Efficiency
                *   Data Selection Strategies
                    *   (Duplicate of NLP > Data Resources & Curation > Data Selection Strategies) 2402.09668
        *   Trustworthy AI
            *   Model Watermarking
                *   Radioactivity Detection
                    *   2402.14904 | Đề xuất phương pháp phát hiện "phóng xạ" (LLM huấn luyện trên dữ liệu thủy vân) cho kịch bản mở (reading mode) và đóng (filter ϕ).
        *   Interpretability
            *   Large Language Model Interpretability
                *   Survey and Position Paper
                    *   2402.01761 | Tổng quan và định vị về diễn giải LLM, cơ hội, thách thức và hướng nghiên cứu.
    *   AI Agents & Robotics
        *   Language Agents
            *   Embodied Agents
                *   Generalist OS-Level Agents
                    *   2402.07456 | Đề xuất OS-Copilot (framework khái niệm) và FRIDAY (tác tử tự cải thiện với Tool Generator, Critic, Refiner, Self-Directed Learning).
            *   Game Playing
                *   Tactical Battle Games
                    *   2402.01118 | Đề xuất POK´ELLM ON với ICRL (học tăng cường trong ngữ cảnh bằng phản hồi văn bản), KAG (sinh tăng cường tri thức từ Pokédex), Consistent Action Generation.
            *   Planning & Decision Making
                *   Amortized Planning with Transformers
                    *   2402.04494 | Xây dựng ChessBench, ứng dụng Transformer lớn học trực tiếp action-value từ dữ liệu giám sát quy mô lớn cho cờ vua, không cần tìm kiếm.
                *   Language Agent Planning Benchmarks
                    *   2402.01622 | Giới thiệu TravelPlanner, benchmark đánh giá tác tử ngôn ngữ lập kế hoạch du lịch phức tạp (dài hạn, dùng công cụ, nhiều ràng buộc).
            *   Multimodal Agents
                *   Desktop and Web Automation
                    *   2402.17553 | Giới thiệu OmniACT, bộ dữ liệu và benchmark đánh giá tác tử đa phương thức trên desktop/web bằng sinh mã PyAutoGUI, và DetACT module.
        *   Robotics
            *   Human-Robot Interaction
                *   Learning from Human Feedback
                    *   LLM-based Interactive Teaching with Predictive Control
                        *   2402.11450 | Đề xuất LMPC (Language Model Predictive Control) và Top-User Conditioning để tinh chỉnh LLM viết mã robot từ tương tác nhiều lượt.
            *   Legged Robotics
                *   Locomotion Control
                    *   Safe and Agile Navigation
                        *   2401.17583 | Đề xuất ABS (Agile Behavior System) với chính sách linh hoạt, giá trị Reach-Avoid học được, và kiến trúc điều khiển dual-policy cho robot bốn chân.
        *   Agent Foundation Models
            *   Unified Pre-training Strategies
                *   2402.05929 | Đề xuất khung tiền huấn luyện hợp nhất cho tác tử đa phương thức (ngôn ngữ, visual MAE, dự đoán hành động) với huấn luyện đồng thời các thành phần.
        *   Long-Document Comprehension Agents
            *   2402.09727 | Đề xuất ReadAgent, agent LLM xử lý văn bản dài bằng gist memory và interactive look-up, với episode pagination.
        *   Conversational AI
            *   Task-Oriented Dialogue Systems for Web Interaction
                *   2402.05930 | Đề xuất Dense Markup Ranking (DMR) để chọn/xếp hạng phần tử HTML và bộ chỉ số đánh giá chi tiết cho điều hướng web hội thoại (WEBLINX).
    *   AI for Science
        *   (Covered under NLP > LLMs > Domain-Specific LLMs: Chemistry)
    *   Machine Learning Systems
        *   Large-Scale Deep Learning Training
            *   System Optimization for LLM Training
                *   Efficiency and Fault Tolerance at Scale
                    *   2402.15627 | Đề xuất MegaScale, tối ưu hóa hệ thống (giao tiếp TP/SP, tải dữ liệu, khởi tạo NCCL, kiểm soát tắc nghẽn, quy trình chịu lỗi) cho huấn luyện LLM >10k GPU.
    *   Structured Data Understanding
        *   Instruction Tuning for Structured Knowledge Grounding
            *   2402.16671 | Xây dựng StructLM Dataset (hợp nhất 25 bộ SKG + SlimOrca) và chiến lược tinh chỉnh hướng dẫn trên CodeLlama.
    *   Speech Processing
        *   Text-to-Speech (TTS)
            *   Large-Scale TTS Models
                *   Neural Speech Synthesis Architectures
                    *   2402.08093 | Đề xuất mã hóa giọng nói WavLM-based (tách người nói, BPE), bộ giải mã speechcode streamable, và chứng minh khả năng nổi bật về ngữ điệu khi tăng quy mô.
    *   Other
        *   (Papers that are primarily position papers or surveys without a primary novel technical contribution fitting elsewhere, or very niche applications if any remain)
            *   2402.17139 | Bài báo định vị: Video như một giao diện hợp nhất cho các nhiệm vụ trong thế giới thực.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                  |
    | :--- | :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2402.17764 | LLM, Quantization, 1-bit, BitNet b1.58, absmean      | Đạt hiệu năng tương đương FP16 với mô hình LLM 1.58-bit (trọng số tam phân {-1,0,1}) từ 3B tham số trở lên.                               | Giảm đáng kể chi phí tính toán, bộ nhớ, năng lượng cho LLM, tiềm năng cho thiết bị biên và phần cứng chuyên dụng.                             |
    | 2    | 2402.15391 | Video Generation, Interactive, World Model, Unsupervised Latent Actions | Genie: Tạo môi trường tương tác (game 2D) từ video không nhãn bằng cách học không giám sát các hành động tiềm ẩn (latent actions). | Mở ra hướng mới cho việc xây dựng world models từ dữ liệu video khổng lồ trên internet mà không cần nhãn hành động.                           |
    | 3    | 2402.08268 | Multimodal, Long Context, Video, 1M tokens, RingAttention | Huấn luyện thành công mô hình Transformer xử lý ngữ cảnh 1 triệu token cho cả văn bản và video.                                        | Đẩy xa giới hạn xử lý ngữ cảnh dài trong các mô hình đa phương thức, mở khả năng cho các ứng dụng hiểu video/văn bản sâu hơn.                 |
    | 4    | 2402.17485 | Video Generation, Talking Head, Audio-driven, Diffusion | EMO: Sinh video chân dung nói chuyện biểu cảm trực tiếp từ âm thanh và ảnh, không cần mô hình 3D trung gian.                             | Cải thiện tính biểu cảm và tự nhiên của video nói chuyện được tạo ra, tiềm năng cho avatar ảo, lồng tiếng.                                  |
    | 5    | 2402.13753 | LLM, Context Extension, RoPE, Evolutionary Search    | LongRoPE: Mở rộng ngữ cảnh LLM lên 2048k token với chi phí tinh chỉnh thấp bằng tìm kiếm tiến hóa các hệ số co giãn RoPE bất đối xứng. | Cho phép LLM xử lý các tài liệu và cuộc hội thoại cực dài, tăng cường khả năng hiểu và suy luận trên ngữ cảnh rộng.                          |
    | 6    | 2402.04494 | Planning, Chess, Transformers, Amortized Planning    | Transformer lớn (270M) đạt trình độ đại kiện tướng cờ vua (Elo 2895) chỉ bằng dự đoán action-value, không cần tìm kiếm lúc chơi.        | Chứng minh khả năng "khấu hao" hoạch định phức tạp vào mạng nơ-ron, mở ra hướng mới cho các bài toán hoạch định không cần tìm kiếm.          |
    | 7    | 2402.14830 | LLM, Math Reasoning, SLM, Agent-Instruct, Iterative Learning | Orca-Math (SLM 7B) đạt SOTA trên GSM8K (86.81% pass@1) mà không cần ensembling hay công cụ, nhờ dữ liệu Agent-Instruct và học lặp lại. | Cho thấy SLM có thể đạt hiệu năng rất cao trong suy luận toán học với dữ liệu chất lượng và chiến lược học phù hợp.                         |
    | 8    | 2402.15319 | LLM, Quantization, PTQ, Vector Quantization, Hessian | GPTVQ: Đạt SOTA về cân bằng kích thước-độ chính xác cho LLM ở mức bit thấp bằng VQ sau huấn luyện dựa trên Hessian.                      | Cung cấp phương pháp nén LLM hiệu quả cao, đặc biệt ở các mức bit thấp, giữ được độ chính xác tốt hơn các phương pháp PTQ khác.             |
    | 9    | 2402.05008 | Image Segmentation, SAM, Acceleration, EfficientViT  | EfficientViT-SAM: Tăng tốc SAM gốc đáng kể (48.9x) mà không làm giảm (thậm chí cải thiện) hiệu năng zero-shot.                         | Giải quyết hạn chế về tốc độ của SAM, cho phép ứng dụng rộng rãi hơn trong các kịch bản thời gian thực.                                    |
    | 10   | 2402.10193 | LLM, Model Compression, Delta Quantization, 1-bit    | BitDelta: Nén delta trọng số LLM sau tinh chỉnh xuống 1-bit mà gần như không giảm hiệu năng, giảm >10x kích thước delta.                | Giảm đáng kể chi phí lưu trữ và tăng hiệu quả phục vụ nhiều mô hình LLM tinh chỉnh đồng thời.                                               |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2402.17764 – Lượng tử hóa trọng số `absmean` thành {-1, 0, 1} và kiến trúc BitNet b1.58 – Suy nghĩ:** Đột phá trong việc giảm bit-width trọng số xuống cực thấp mà vẫn giữ hiệu năng, việc thêm giá trị '0' so với BitNet gốc là một cải tiến hợp lý cho feature filtering.
    *   **2402.17485 – Sinh video nói chuyện trực tiếp từ âm thanh/ảnh bằng diffusion model không qua trung gian 3D (EMO) – Suy nghĩ:** Loại bỏ các bước trung gian 3DMM/mesh giúp tăng tính biểu cảm và tự nhiên, một hướng đi thông minh.
    *   **2402.14905 – Chia sẻ trọng số tức thời theo khối (immediate block-wise weight sharing) cho LLM di động – Suy nghĩ:** Giải pháp thực tế, nhận biết ràng buộc phần cứng (cache locality) để tối ưu độ trễ suy luận trên thiết bị.
    *   **2402.03300 – Thuật toán Group Relative Policy Optimization (GRPO) loại bỏ critic trong PPO – Suy nghĩ:** Cải tiến thú vị cho PPO, giải quyết bài toán chi phí tài nguyên khi huấn luyện mô hình lớn, dù cần thêm phân tích lý thuyết.
    *   **2402.13753 – Tìm kiếm tiến hóa các hệ số co giãn RoPE bất đối xứng (LongRoPE) – Suy nghĩ:** Phương pháp sáng tạo để tối ưu hóa RoPE cho ngữ cảnh siêu dài, linh hoạt hơn các quy tắc cố định.
    *   **2402.03620 – Khung SELF-DISCOVER cho LLM tự khám phá cấu trúc lý luận (SELECT, ADAPT, IMPLEMENT) – Suy nghĩ:** Hướng đi rất tiềm năng để LLM tự động hóa việc xây dựng chiến lược giải quyết vấn đề, tăng tính linh hoạt và diễn giải.
    *   **2402.10200 – CoT-decoding: gợi mở CoT tiềm ẩn bằng khám phá token thay thế ở bước giải mã đầu tiên – Suy nghĩ:** Cách tiếp cận đơn giản nhưng hiệu quả để khai thác khả năng suy luận nội tại của LLM mà không cần prompt phức tạp.
    *   **2402.13144 – Sinh tham số mạng nơ-ron bằng latent diffusion model (p-diff) với autoencoder có nhiễu – Suy nghĩ:** Ứng dụng diffusion model vào một lĩnh vực mới (sinh tham số), ý tưởng thêm nhiễu vào autoencoder để tăng cường hiệu quả là đáng chú ý.
    *   **2402.10644 – Kernel bậc hai học được (ReBased) cho linear attention bằng biến đổi affine đầu vào – Suy nghĩ:** Cải tiến hợp lý cho Based, làm cho kernel linh hoạt hơn và có khả năng gán điểm chú ý gần zero.
    *   **2402.15391 – Latent Action Model (LAM) học không giám sát hành động tiềm ẩn từ video – Suy nghĩ:** Đóng góp kỹ thuật nổi bật, mở ra hướng đi mới cho việc xây dựng world models từ video không nhãn.
    *   **2402.03286 – Subject-Driven Self-Attention (SDSA) và Correspondence-based Feature Injection (DIFT) cho sinh ảnh chủ thể nhất quán (ConsiStory) – Suy nghĩ:** Kết hợp thông minh các cơ chế attention và DIFT để cân bằng nhất quán chủ thể và đa dạng bố cục mà không cần huấn luyện.
    *   **2402.14083 – Search dynamics bootstrapping: Transformer học cải thiện thuật toán A* bằng cách mô phỏng và rút ngắn quá trình tìm kiếm – Suy nghĩ:** Ý tưởng mới lạ, Transformer không chỉ mô phỏng mà còn "học để tìm kiếm tốt hơn" thuật toán kinh điển.
    *   **2402.13064 – GLAN: Tạo dữ liệu chỉ thị tổng quát quy mô lớn từ phân loại tri thức, không cần dữ liệu mồi – Suy nghĩ:** Giải pháp có hệ thống và khả năng mở rộng cao để giải quyết vấn đề thiếu dữ liệu chỉ thị đa dạng.
    *   **2402.12376 – 2D RoPE và các kỹ thuật ngoại suy VisionNTK/YaRN tách rời cho FiT (Flexible Vision Transformer) – Suy nghĩ:** Áp dụng và điều chỉnh thành công khái niệm RoPE và ngoại suy từ LLM sang xử lý ảnh độ phân giải linh hoạt.
    *   **2402.07827 – Tỉa dữ liệu hướng dẫn đa ngôn ngữ (xP3x) dựa trên chú thích con người và chưng cất ngữ cảnh an toàn đa ngôn ngữ – Suy nghĩ:** Các bước xử lý dữ liệu và an toàn quan trọng cho LLM đa ngônǝữ chất lượng cao, đặc biệt cho ngôn ngữ ít tài nguyên.
    *   **2402.13616 – Programmable Gradient Information (PGI) với nhánh phụ trợ khả nghịch và thông tin phụ trợ đa cấp – Suy nghĩ:** Cơ chế huấn luyện thú vị để chống mất mát thông tin và tạo gradient đáng tin cậy mà không ảnh hưởng tốc độ suy luận.
    *   **2402.01093 – Mạng Chiếu (Projected Network - PN) cho SLM chuyên biệt: chiếu tuyến tính từ mô hình lớn thành nhiều chuyên gia nhỏ – Suy nghĩ:** Kiến trúc thông minh cho phép chuyên biệt hóa nhanh chóng và tiết kiệm chi phí cho nhiều miền đích từ một lần tiền huấn luyện.
    *   **2402.07456 – Học tự định hướng (Self-Directed Learning) cho tác tử FRIDAY tự đề xuất chương trình học và tích lũy kỹ năng – Suy nghĩ:** Cơ chế sáng tạo giúp agent tự động làm chủ ứng dụng mới, tăng khả năng thích ứng.
    *   **2402.12226 – Token hóa rời rạc thống nhất nhiều phương tiện (ảnh, giọng nói, nhạc, văn bản) cho MLLM any-to-any (AnyGPT) – Suy nghĩ:** Cách tiếp cận thanh lịch để hợp nhất đa phương tiện trong LLM mà không cần thay đổi kiến trúc cốt lõi.
    *   **2402.11131 – Speculative Streaming: Chú ý Đa luồng (MSA) tích hợp dự đoán và xác minh token trong một LLM duy nhất – Suy nghĩ:** Kiến trúc hiệu quả, loại bỏ mô hình nháp phụ trợ, giảm đáng kể tham số bổ sung so với các phương pháp một mô hình khác.
    *   **2402.04615 – Screen schema (biểu diễn UI/infographics) và quy trình tạo dữ liệu tiền huấn luyện tự động bằng LLM cho ScreenAI – Suy nghĩ:** Giải pháp sáng tạo để tạo dữ liệu quy mô lớn cho hiểu UI, giải quyết nút thắt về dữ liệu.
    *   **2402.01391 – Curriculum of Code Completion Subtasks (CCCS) và Fine-Grained Optimization (FGO) cho RL trong sinh mã (StepCoder) – Suy nghĩ:** Các giải pháp kỹ thuật hợp lý giải quyết vấn đề khám phá và tối ưu hóa trong RL cho sinh mã.
    *   **2402.10790 – RMT-R: Tự truy xuất (self-retrieval) dựa trên toàn bộ lịch sử trạng thái bộ nhớ trong Recurrent Memory Transformer – Suy nghĩ:** Cải tiến kiến trúc rõ ràng, giải quyết nút cổ chai bộ nhớ của RMT gốc, cho phép xử lý ngữ cảnh siêu dài.
    *   **2402.09668 – ASK-LLM: Lấy mẫu dữ liệu dựa trên chất lượng bằng cách dùng LLM proxy đánh giá tính thông tin qua prompting – Suy nghĩ:** Phương pháp thông minh, tận dụng khả năng lý luận của LLM để chọn dữ liệu chất lượng hơn perplexity.
    *   **2402.16107 – VARM: Tự động xác định trọng số trộn tham số LLM dựa trên tỷ lệ biến thiên ma trận tham số sau tinh chỉnh – Suy nghĩ:** Cơ chế xác định trọng số tự động và chi tiết, giải quyết hạn chế của các phương pháp trộn tham số trước.
    *   **2402.05930 – Dense Markup Ranking (DMR): Lựa chọn và xếp hạng phần tử HTML bằng dual-encoder cho điều hướng web hội thoại – Suy nghĩ:** Giải pháp kỹ thuật hợp lý và hiệu quả cho việc xử lý DOM lớn trong tác tử web.
    *   **2402.15627 – Tối ưu hóa giao tiếp TP/SP bằng cách gộp all-gather/reduce-scatter vào lớp Linear và pipelining với GEMM (MegaScale) – Suy nghĩ:** Kỹ thuật tối ưu hóa hệ thống sâu sắc, che giấu hiệu quả độ trễ giao tiếp ở quy mô lớn.
    *   **2402.09727 – ReadAgent: Phân đoạn episode bằng LLM, tạo gist memory và interactive look-up để xử lý văn bản dài – Suy nghĩ:** Phương pháp trực quan, mô phỏng cách con người đọc hiểu, hiệu quả mà không cần huấn luyện lại LLM.
    *   **2402.10176 – Masked Text Solution Prompting: Che giá trị trung gian trong lời giải văn bản tham khảo để tổng hợp lời giải code-interpreter – Suy nghĩ:** Kỹ thuật thông minh để tận dụng lời giải tham khảo mà không bị mô hình "gian lận", cải thiện chất lượng sinh dữ liệu từ LLM yếu hơn.
    *   **2401.17377 – Engine Infini-gram dựa trên suffix array để truy vấn xác suất ∞-gram trên dữ liệu nghìn tỷ token – Suy nghĩ:** Giải pháp hiệu quả dựa trên cấu trúc dữ liệu kinh điển để vượt qua hạn chế của n-gram truyền thống ở quy mô lớn.
    *   **2402.08609 – Tích hợp Soft MoE vào lớp avant-dernier của mạng giá trị trong Deep RL (DQN, Rainbow) – Suy nghĩ:** Ứng dụng thành công Soft MoE để giải quyết vấn đề mở rộng tham số trong RL dựa trên giá trị.
    *   **2402.15504 – Pipeline Gen4Gen: Chuỗi mô hình nền tảng (phân tách, LLM, MLLM, inpainting) để tạo dữ liệu cá nhân hóa đa khái niệm – Suy nghĩ:** Kết hợp hiệu quả các mô hình AI cho một tác vụ cụ thể, nhấn mạnh vai trò của chất lượng dữ liệu.
    *   **2402.11450 – Language Model Predictive Control (LMPC) và Top-User Conditioning cho dạy robot tương tác – Suy nghĩ:** Phương pháp luận mới lạ kết hợp SFT trên LLM với MPC để tối ưu hóa tương tác nhiều lượt.
    *   **2402.11248 – Crayon Prompt (bản đồ màu panoptic + truy vấn nhúng) và Dual QLoRA cho hiểu đối tượng trong VLM (CoLLaVO) – Suy nghĩ:** Ý tưởng thú vị để đưa thông tin cấu trúc đối tượng vào VLM và cân bằng các mục tiêu huấn luyện.
    *   **2402.10193 – Lượng tử hóa delta trọng số 1-bit (BitDelta) bằng Sign(∆) và scale factor tinh chỉnh bằng distillation – Suy nghĩ:** Phương pháp cực kỳ đơn giản và hiệu quả để nén delta, cho thấy sự dư thừa lớn trong thông tin tinh chỉnh.
    *   **2402.10009 – ZEUS: Chỉnh sửa âm thanh không giám sát bằng nhiễu loạn khuếch tán ngược theo PC của hiệp phương sai hậu nghiệm – Suy nghĩ:** Phương pháp sáng tạo, tạo biến thể ngữ nghĩa âm thanh mà không cần mô tả văn bản hay tối ưu hóa test-time.
    *   **2402.05008 – EfficientViT-SAM: Thay thế encoder SAM bằng EfficientViT và quy trình huấn luyện 2 giai đoạn (chưng cất + end-to-end) – Suy nghĩ:** Giải pháp hiệu quả để tăng tốc SAM mà không hy sinh độ chính xác, một cải tiến thực tế quan trọng.
    *   **2402.00769 – AnimateLCM: Tách rời học phong cách (từ ảnh) và tăng tốc sinh video (ảnh rồi chuyển động) qua hợp nhất trọng số – Suy nghĩ:** Chiến lược thông minh, giảm thời gian suy luận và chi phí thu thập dữ liệu video phong cách hóa.
    *   **2402.15000 – Chưng cất có chọn lọc bộ phân rã câu hỏi (decomposition) từ teacher LLM sang student SLM (SD-T, SD-R) – Suy nghĩ:** Ý tưởng thực tế, chưng cất phần "dễ" (phân rã) và giữ lại phần "khó" (giải quyết) của LLM lớn, tối ưu chi phí/hiệu năng.
    *   **2402.10524 – Phân cụm giải thích (rationale) tự động dựa trên LLM để tóm tắt lý do đánh giá AutoSxS – Suy nghĩ:** Đóng góp kỹ thuật cụ thể, giúp làm rõ lý do đánh giá LLM, hữu ích cho việc phân tích kết quả.
    *   **2402.05140 – TAG-LLM: Kiến trúc thẻ nhúng (miền, chức năng) và quy trình huấn luyện 3 giai đoạn phân cấp – Suy nghĩ:** Cách tiếp cận module hóa và hiệu quả tham số để thích ứng LLM cho các miền chuyên biệt, đặc biệt là phi ngôn ngữ.
    *   **2402.13250 – Video ReCap: Kiến trúc video-ngôn ngữ đệ quy và học theo chương trình phân cấp cho phụ đề video dài – Suy nghĩ:** Hướng tiếp cận có hệ thống, giải quyết bài toán khó là tạo phụ đề đa cấp cho video rất dài.
    *   **2402.17463 – Dual Chunk Attention (DCA): Cơ chế chú ý trong chunk (RoPE lặp lại) và giữa các chunk (chỉ số vị trí lớn không đổi/cửa sổ cục bộ) – Suy nghĩ:** Đóng góp kỹ thuật đáng chú ý, mở rộng ngữ cảnh LLM hiệu quả mà không cần huấn luyện lại, thiết kế lại cách tính ma trận vị trí tương đối.
    *   **2402.17412 – DiffuseKronA: Hiệu chỉnh tham số mô hình khuếch tán T2I bằng tích Kronecker (A ⊗ B) cho ∆W – Suy nghĩ:** Ứng dụng mới của KronA vào cá nhân hóa T2I, tiềm năng giảm tham số và tăng ổn định so với LoRA.
    *   **2402.14904 – "Chế độ đọc" (reading mode) phát hiện phóng xạ LLM bằng cách phân tích dự đoán token tiếp theo trên văn bản thủy vân – Suy nghĩ:** Kỹ thuật độc đáo và hiệu quả để thăm dò trực tiếp phản ứng nội bộ của mô hình mở với tín hiệu thủy vân.
    *   **2402.13217 – VideoPrism: Chưng cất global-local và xáo trộn token trong mô hình hóa video bị che dấu – Suy nghĩ:** Cải tiến kỹ thuật rõ ràng cho tiền huấn luyện video, tận dụng hiệu quả dữ liệu video-văn bản đa dạng.
    *   **2402.01566 – Boximator: Mô-đun điều khiển plug-in với ràng buộc hộp cứng/mềm và kỹ thuật "self-tracking" cho sinh video – Suy nghĩ:** Kỹ thuật "self-tracking" là đóng góp đáng chú ý, giải quyết hiệu quả thách thức học liên kết hộp-đối tượng.
    *   **2402.14797 – Khung khuếch tán EDM sửa đổi cho video độ phân giải cao (σ_in) và kiến trúc Transformer dựa trên FIT mở rộng – Suy nghĩ:** Giải quyết vấn đề SNR trong EDM cho video và hạn chế của U-Net một cách có cơ sở.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Lý thuyết về Lượng tử hóa Cực thấp:** Mặc dù các mô hình 1-bit (2402.17764, 2402.04291, 2402.11295, 2402.10193) cho thấy tiềm năng lớn, vẫn còn thiếu hiểu biết lý thuyết sâu sắc về tại sao chúng hoạt động tốt ở một số quy mô nhất định và giới hạn của chúng. Cơ hội: Phát triển khung lý thuyết giải thích hiệu quả của lượng tử hóa cực thấp và các định luật scaling mới liên quan.
    *   **Khả năng Khái quát hóa của World Models từ Video:** Genie (2402.15391) là một bước tiến, nhưng khả năng khái quát hóa sang các loại video phức tạp hơn (3D, đời thực) và tính nhất quán logic trong tương tác dài hạn vẫn là thách thức. Cơ hội: Nghiên cứu các phương pháp học hành động tiềm ẩn mạnh mẽ hơn và các kiến trúc bộ nhớ/động học có khả năng duy trì sự nhất quán lâu dài.
    *   **Đánh giá và Kiểm soát Tính Biểu Cảm/Chất lượng trong Sinh Video/Âm thanh:** Các mô hình như EMO (2402.17485), BASE TTS (2402.08093) đạt được tính biểu cảm cao, nhưng việc đánh giá định lượng và kiểm soát tinh vi các khía cạnh này vẫn khó khăn. Cơ hội: Phát triển các chỉ số đánh giá khách quan hơn cho tính biểu cảm, ngữ điệu, và các phương pháp điều khiển chi tiết hơn các yếu tố này trong quá trình tạo sinh.
    *   **Chiến lược Huấn luyện Hiệu quả cho Ngữ cảnh Siêu dài:** Các phương pháp như LongRoPE (2402.13753), DCA (2402.17463), RMT-R (2402.10790), LongAlign (2401.18058) và (2402.10171) đang đẩy giới hạn, nhưng chi phí huấn luyện/tinh chỉnh vẫn là rào cản. Cơ hội: Nghiên cứu các kiến trúc và thuật toán tối ưu hóa hiệu quả hơn nữa cho việc học các phụ thuộc tầm xa mà không cần full attention hoặc lượng dữ liệu khổng lồ.
    *   **Tích hợp Suy luận Logic và Kiến thức Nền tảng vào LLM một cách Đáng Tin cậy:** Các phương pháp như SELF-DISCOVER (2402.03620), CoT-decoding (2402.10200), và phân tích về suy luận tiềm ẩn (2402.16837, 2402.08939) cho thấy LLM có khả năng suy luận nhưng vẫn còn hạn chế. Cơ hội: Phát triển các cơ chế để LLM tích hợp kiến thức bên ngoài một cách linh hoạt và thực hiện suy luận đa bước đáng tin cậy hơn, có thể kiểm chứng được.
    *   **Tạo Dữ liệu Tổng hợp Chất lượng Cao và Đa dạng:** Các công trình như GLAN (2402.13064), Agent-Instruct (2402.14830), OpenMathInstruct-1 (2402.10176), Gen4Gen (2402.15504) cho thấy tầm quan trọng của dữ liệu tổng hợp. Cơ hội: Phát triển các phương pháp tạo dữ liệu tự động hoặc bán tự động tiên tiến hơn, có khả năng kiểm soát chất lượng, độ đa dạng, và giảm thiểu thiên kiến.
    *   **An toàn và Căn chỉnh Đa Ngôn ngữ/Đa Phương thức:** Aya (2402.07827), PALO (2402.14818) giải quyết vấn đề đa ngôn ngữ, nhưng an toàn và căn chỉnh trong không gian đa phương thức và nhiều ngôn ngữ vẫn là thách thức lớn. Cơ hội: Phát triển các kỹ thuật căn chỉnh mạnh mẽ, công bằng và an toàn cho các mô hình xử lý nhiều loại dữ liệu và ngôn ngữ.
    *   **Khả năng Khái quát hóa của Tác tử AI:** Các tác tử như FRIDAY (2402.07456), POK´ELLM ON (2402.01118), và các benchmark (TravelPlanner 2402.01622, OmniACT 2402.17553) cho thấy tiềm năng nhưng cũng bộc lộ những khó khăn trong việc khái quát hóa sang các nhiệm vụ/môi trường mới. Cơ hội: Nghiên cứu các phương pháp học liên tục, học chuyển giao và tự cải thiện mạnh mẽ hơn cho tác tử AI.
    *   **Hiệu quả Tham số và Tính toán cho Mô hình Nền tảng:** Các kiến trúc như MambaFormer (2402.04248), BlackMamba (2402.01771), MobiLlama (2402.16840) và các kỹ thuật như Projected Networks (2402.01093) đang tìm cách tối ưu hóa. Cơ hội: Khám phá các kiến trúc và kỹ thuật chia sẻ/nén tham số mới lạ hơn nữa để đạt được sự cân bằng tối ưu giữa hiệu năng và chi phí.
    *   **Diễn giải và Độ tin cậy của các Mô hình Phức hợp:** Với các hệ thống kết hợp nhiều mô hình (Gen4Gen 2402.15504, LAVE 2402.10294) hoặc các cơ chế phức tạp (PGI 2402.13616), việc hiểu và tin tưởng vào quyết định của chúng trở nên khó khăn hơn. Cơ hội: Phát triển các công cụ và phương pháp diễn giải mới cho các hệ thống AI phức hợp, đa thành phần.

5.  **FUTURE_IDEAS**

    ✨ **Neuro-Symbolic Dynamic Reasoning Kernels for LLMs**
    *   Motivation: LLMs struggle with complex, multi-step reasoning that requires dynamic planning and grounding in symbolic knowledge (gaps identified in 2402.03620, 2402.16837, 2402.08939). Current methods like CoT are often brittle.
    *   Key novelty: Integrate a differentiable symbolic reasoning engine (e.g., based on logic programming or graph neural networks operating on knowledge graphs) as a specialized "reasoning kernel" that the LLM can learn to invoke and query. This kernel would handle explicit reasoning steps, while the LLM manages natural language understanding and generation.
    *   Approach:
        1.  Develop a library of differentiable symbolic operations (e.g., unification, graph traversal, logical deduction).
        2.  Design an interface where LLM hidden states can be mapped to symbolic queries for the kernel, and kernel outputs can be mapped back to LLM embeddings.
        3.  Train the LLM end-to-end (or via RL using the kernel's success as reward) to learn *when* and *how* to use the reasoning kernel for specific sub-problems identified within a larger query.
    *   Dataset + Metrics: Mathematical reasoning (GSM8K, MATH), complex QA (HotpotQA, StrategyQA), planning tasks (TravelPlanner 2402.01622). Metrics: Accuracy, logical consistency of reasoning steps, interpretability of kernel calls.
    *   Risk/Feasibility: High risk due to complexity of differentiable symbolic reasoning and LLM-kernel integration. Feasibility depends on advances in neuro-symbolic AI.

    ✨ **Self-Evolving Data Curation Pipelines using Meta-Learned Quality Assessors**
    *   Motivation: Data quality and diversity are paramount (e.g., 2402.03300, 2402.00159, 2402.06619, 2402.09668, 2401.18058, 2402.10176, 2402.15504), but manual curation is unscalable, and current automated methods (like ASK-LLM 2402.09668) can be costly or biased by the proxy LLM.
    *   Key novelty: A meta-learning framework where a "curator agent" (itself an LLM) learns to dynamically adjust and combine various data filtering, generation, and augmentation techniques based on the downstream performance of a target model being trained. The curator agent would learn a "data quality policy."
    *   Approach:
        1.  Define a space of data processing operations (filters, synthetic data generators like GLAN 2402.13064, augmentation techniques).
        2.  The curator agent, observing the target model's training loss/validation performance, selects and parameterizes a sequence of these operations to apply to a raw data pool.
        3.  Use reinforcement learning or evolutionary strategies to train the curator agent, where the "reward" is the improvement in the target model's performance or a combined metric of performance and data efficiency.
    *   Dataset + Metrics: Large raw text/code corpora (Common Crawl, GitHub). Metrics: Downstream task performance of models trained on curated data, diversity of curated data, cost of curation.
    *   Risk/Feasibility: Medium-high risk. Defining the action space for the curator and the reward signal can be challenging. Computational cost of meta-learning loop.

    ✨ **Cross-Modal Latent Action Spaces for Generalist Embodied Agents**
    *   Motivation: Current embodied agents (2402.07456, 2402.05929, 2401.17583) often have action spaces tied to specific modalities or environments. Genie (2402.15391) learns latent actions from video, but extending this to diverse modalities (text, audio commands, physical interactions) is key for generalist agents.
    *   Key novelty: Learn a unified, low-dimensional latent action space that is shared across multiple input modalities (video, audio, language instructions) and output modalities (robotic arm movements, navigation commands, generated language/code). This space would capture the *intent* and *effect* of actions abstractly.
    *   Approach:
        1.  Use a multimodal autoencoder (inspired by AnyGPT 2402.12226 for discrete tokens, but for continuous actions) to learn to reconstruct actions in different modalities from a shared latent representation.
        2.  Train a dynamics model in this latent action space, predicting future world states (also multimodally represented) given the current state and a latent action.
        3.  The agent plans in this latent action space and then decodes the latent action into a modality-specific executable action.
    *   Dataset + Metrics: Egocentric datasets with diverse interactions (AEA 2402.13349, Ego4D), robotics simulation environments (e.g., Habitat, RoboDesk). Metrics: Task completion rates across diverse tasks, zero-shot transfer to new (but related) actions/modalities, efficiency of planning in latent space.
    *   Risk/Feasibility: High risk. Learning such a disentangled and generalizable latent action space is extremely challenging. Requires significant advances in multimodal representation learning and world modeling. (Moon-shot)

    ✨ **Adaptive Quantization Networks for Dynamic Resource Allocation**
    *   Motivation: While 1-bit LLMs (2402.17764, 2402.04291, 2402.11295) are promising for static efficiency, real-world scenarios involve fluctuating computational budgets (e.g., on-device vs. cloud, battery levels).
    *   Key novelty: Develop LLMs whose quantization levels (bit-widths for weights/activations) can be dynamically adjusted *at inference time* on a per-layer or per-module basis, without retraining, to trade off accuracy for speed/energy.
    *   Approach:
        1.  Train a "super-quantized" network that implicitly learns representations suitable for multiple bit-widths (e.g., using techniques from once-for-all networks or dynamic channel pruning, adapted for quantization).
        2.  A small, efficient controller network (or a rule-based system) determines the optimal quantization configuration for different parts of the LLM based on the current query complexity, available resources, and target latency/accuracy.
        3.  This could build upon ideas from GPTVQ (2402.15319) by having multiple codebooks or BitDelta (2402.10193) by having switchable delta precision.
    *   Dataset + Metrics: Standard LLM benchmarks (perplexity, downstream tasks) evaluated under various resource constraints (FLOPs, memory, latency). Metrics: Pareto frontier of accuracy vs. resource usage.
    *   Risk/Feasibility: Medium risk. Training a stable super-quantized network is hard. The controller needs to be very lightweight.

6.  **READING_LIST**

    *   2402.17764 – BitNet b1.58 · Đột phá về LLM 1.58-bit với hiệu năng cao, mở ra hướng mới cho LLM siêu hiệu quả.
    *   2402.15391 – Genie · Phương pháp học không giám sát hành động tiềm ẩn từ video để tạo môi trường tương tác, rất sáng tạo.
    *   2402.03620 – SELF-DISCOVER · LLM tự khám phá cấu trúc lý luận, một bước tiến quan trọng cho khả năng tự chủ của AI.
    *   2402.13753 – LongRoPE · Kỹ thuật mở rộng ngữ cảnh LLM lên mức rất dài (2048k) với chi phí thấp, giải quyết một hạn chế lớn.
    *   2402.08268 – 1M Token Multimodal LLM · Thành tựu kỹ thuật ấn tượng về huấn luyện mô hình Transformer với ngữ cảnh siêu dài cho cả văn bản và video.
    *   2402.14830 – Orca-Math · Cho thấy SLM có thể đạt hiệu năng suy luận toán học SOTA mà không cần công cụ phức tạp, nhấn mạnh vai trò của dữ liệu và chiến lược học.
    *   2401.18059 – RAPTOR · Phương pháp truy xuất thông tin mới lạ cho RAG bằng cây phân cấp từ tóm tắt và phân cụm đệ quy.
    *   2402.15319 – GPTVQ · Kỹ thuật lượng tử hóa vector sau huấn luyện dựa trên Hessian, đạt SOTA về nén LLM.
    *   2402.11131 – Speculative Streaming · Giải pháp một mô hình hiệu quả cho speculative decoding, giảm độ phức tạp hệ thống.
    *   2402.00159 & 2402.00838 – Dolma & OLMo · Nỗ lực quan trọng về tính mở trong LLM, cung cấp dữ liệu và mô hình nền tảng cho cộng đồng.

7.  **META_REFLECTION**

    *   Tập hợp các bài báo tháng 02/2024 cho thấy một bức tranh sôi động của nghiên cứu AI, với các xu hướng nổi bật:
        1.  **Hiệu quả LLM (Efficiency):** Tiếp tục là một chủ đề nóng với nhiều đột phá trong lượng tử hóa cực thấp (BitNet b1.58, BiLLM, OneBit, BitDelta), kiến trúc hiệu quả (MambaFormer, BlackMamba, MobiLlama), và tăng tốc suy luận (Speculative Streaming, ChunkAttention). Điều này phản ánh nhu cầu cấp thiết triển khai LLM trên các thiết bị hạn chế tài nguyên và giảm chi phí vận hành.
        2.  **Mở rộng Ngữ cảnh (Long Context):** Nhiều phương pháp sáng tạo (LongRoPE, DCA, RMT-R, LongAlign, 1M Token LLM) được đề xuất để LLM có thể xử lý lượng thông tin ngày càng lớn, mở ra khả năng ứng dụng trong các tác vụ đòi hỏi hiểu biết sâu rộng.
        3.  **Suy luận và Lập kế hoạch (Reasoning & Planning):** Có sự tập trung vào việc cải thiện khả năng suy luận toán học (GRPO, Orca-Math), khám phá cấu trúc lý luận tự động (SELF-DISCOVER), và phân tích các cơ chế suy luận tiềm ẩn của LLM. Các tác tử AI có khả năng lập kế hoạch phức tạp (TravelPlanner, POK´ELLM ON, FRIDAY) cũng đang được khám phá.
        4.  **Đa phương thức (Multimodality):** Hướng phát triển mạnh mẽ với các mô hình có khả năng xử lý và tạo sinh kết hợp nhiều loại dữ liệu (EMO, Genie, Sora review, FinTral, AnyGPT, VideoPrism, ScreenAI, CoLLaVO, LAVE, Boximator, AnimateLCM, ConsistI2V). Đặc biệt, sinh video và hiểu video dài, có điều khiển đang có những bước tiến lớn.
        5.  **Dữ liệu là Trung tâm (Data-centric AI):** Tầm quan trọng của dữ liệu tiếp tục được khẳng định qua các nỗ lực xây dựng bộ dữ liệu quy mô lớn, chất lượng cao (Dolma, Aya, Code-Feedback, FinSet, MusicPile, ChessBench, StructLM, OmniACT, AEA), các kỹ thuật tạo dữ liệu tổng hợp (GLAN, Agent-Instruct, OpenMathInstruct-1, Gen4Gen), và các chiến lược chọn lọc/tỉa dữ liệu thông minh (ASK-LLM, DENSITY, Aya pruning).
        6.  **Chuyên biệt hóa và Cá nhân hóa (Specialization & Personalization):** Xu hướng phát triển các LLM/VLM cho các miền cụ thể (tài chính, âm nhạc, hóa học) và các kỹ thuật cá nhân hóa hiệu quả (ConsiStory, DiffuseKronA, Gen4Gen, VCD) đang gia tăng.
        7.  **Tính mở và Tái lập (Openness & Reproducibility):** Nhiều công trình nhấn mạnh việc công bố mã nguồn, dữ liệu và mô hình (OLMo, Dolma, OpenMoE, DataDreamer, mE5, Orca-Math, LongAlign), thúc đẩy sự phát triển của cộng đồng.
        8.  **Học tăng cường và Căn chỉnh (Reinforcement Learning & Alignment):** Các phương pháp mới cho RLHF (DTS+ENN, OAIF) và tối ưu hóa RL cho các tác vụ cụ thể (StepCoder, LMPC) tiếp tục được nghiên cứu để cải thiện hành vi và sự hữu ích của mô hình.
        Nhìn chung, lĩnh vực AI đang hướng tới các mô hình ngày càng có năng lực, hiệu quả, linh hoạt và dễ tiếp cận hơn, đồng thời giải quyết các thách thức về suy luận, hiểu biết ngữ cảnh sâu và tương tác đa phương thức phức tạp.
