
1.  **TOPIC_TREE**

    *   Natural Language Processing (NLP)
        *   Large Language Models (LLMs)
            *   Efficient Training & Fine-tuning
                *   Memory-Efficient Training
                    *   2403.03507 | Đề xuất GaLore, chiếu gradient vào không gian hạng thấp để giảm bộ nhớ optimizer, cho phép học toàn bộ tham số với cơ chế chuyển không gian con.
                *   Efficient Fine-tuning Frameworks
                    *   2403.13372 | Giới thiệu LLaMA Factory, framework hợp nhất tinh chỉnh LLM, và chiến lược Model-Sharing RLHF.
                *   Continual Pre-training
                    *   2403.08763 | Chứng minh chiến lược kết hợp đơn giản (LR re-warming/re-decaying, replay) đủ cho huấn luyện liên tục LLM lớn, đề xuất lịch trình LR "vô hạn".
                *   Mixture-of-Experts (MoE) Integration
                    *   2403.07816 | Đề xuất BTX, quy trình Branch-Train-MiX để huấn luyện chuyên gia song song và hợp nhất thành MoE.
            *   Model Compression
                *   Pruning
                    *   Structured Pruning (Layer Pruning)
                        *   2403.17887 | Đề xuất ShortGPT, tỉa lớp dựa trên thước đo Block Influence (thay đổi trạng thái ẩn).
            *   Alignment & Preference Tuning
                *   Reinforcement Learning from Human Feedback (RLHF)
                    *   Parameter-Efficient RLHF
                        *   2403.10704 | Áp dụng LoRA cho RM và RL trong RLHF (PE-RLHF), đánh giá hiệu suất và hiệu quả tài nguyên.
                    *   Conditional & Online RLHF
                        *   2403.17297 | Đề xuất COOL RLHF với Conditional Reward Model và Online PPO để xử lý sở thích đa dạng và chống reward hacking.
                *   Direct Preference Optimization (DPO) Methods
                    *   2403.19270 | Đề xuất stepwise DPO (sDPO), áp dụng DPO lặp lại trên phân vùng dữ liệu tuần tự với cập nhật mô hình tham chiếu.
                *   Contextual Alignment / Policy Alignment
                    *   2403.09704 | Đề xuất Alignment Studio (Framers, Instructors, Auditors) để điều chỉnh LLM theo quy định, giá trị cụ thể.
            *   Position Embeddings
                *   RoPE Length Generalization
                    *   2403.00071 | Đề xuất Resonance RoPE (làm tròn bước sóng RoPE thành số nguyên) và benchmark POSGEN cho TSTL.
            *   Domain Adaptation
                *   Instruction Tuning Data Generation
                    *   2403.03883 | Đề xuất tạo dữ liệu hướng dẫn pháp lý tổng hợp bằng LLM mô phỏng hội thoại đa lượt.
                *   Biomedical Language Models
                    *   2403.18421 | Xây dựng và công bố BioMedLM (2.7B), huấn luyện trên PubMed với tokenizer tùy chỉnh.
            *   Model Editing and Updating
                *   Memory-Augmented LLMs
                    *   2403.11901 | Đề xuất Larimar, LLM với bộ nhớ nhiều tập phân tán, cập nhật one-shot bằng pseudo-inverse, hỗ trợ xóa kiến thức.
            *   Tool Learning / Tool Augmentation
                *   Interactive Learning / Learning from Feedback
                    *   2403.04746 | Đề xuất STE (Simulated Trial and Error) với bộ nhớ ngắn/dài hạn, tự tinh chỉnh lặp lại và tưởng tượng truy vấn để học sử dụng API.
            *   Pretraining Data Engineering
                *   2403.04652 | Xây dựng quy trình xử lý dữ liệu tiền huấn luyện đa tầng (cascaded) cho Yi LLM, kết hợp nhiều bộ lọc và khử trùng lặp.
            *   Language Model Interaction & Combination
                *   Collaborative Decoding
                    *   2403.03870 | Đề xuất Co-LLM, khung học biến ẩn để phối hợp sinh token xen kẽ giữa nhiều LLM, tối ưu hóa hợp lý biên.
            *   Prompt Engineering
                *   Prompt Compression
                    *   Task-Agnostic Prompt Compression
                        *   2403.12968 | Đề xuất LLMLingua-2, nén prompt trích xuất bằng phân loại token (huấn luyện trên dữ liệu chưng cất từ GPT-4).
            *   Data Augmentation
                *   LLM-based Iterative Data Enhancement
                    *   2403.15042 | Đề xuất LLM2LLM, tăng cường dữ liệu lặp lại có mục tiêu (dựa trên lỗi sai của học viên) bằng LLM giáo viên cho ít dữ liệu.
            *   Analysis of Algorithmic Progress
                *   2403.05812 | Định lượng tiến bộ thuật toán trong tiền huấn luyện LM bằng luật tỷ lệ tăng cường (thời gian, tính toán, thuật toán).
            *   Evaluation and Benchmarking
                *   Reward Model Benchmarking
                    *   2403.13787 | Giới thiệu RewardBench, benchmark và framework đánh giá mô hình phần thưởng trên các truy vấn thử thách.
                *   Long-form Factuality Assessment
                    *   2403.18802 | Giới thiệu LongFact (bộ dữ liệu câu hỏi dài) và SAFE (đánh giá xác thực dài bằng LLM agent + Google Search).
                *   Reinforcement Learning for Reasoning Evaluation
                    *   2403.04642 | So sánh hiệu suất và độ phức tạp mẫu của các thuật toán RL (EI, PPO, RCRL) cho suy luận toán học của LLM.
        *   Code Generation
            *   Large Language Models for Code
                *   2402.19173 | Xây dựng The Stack v2 (dữ liệu code lớn) và huấn luyện StarCoder2 (3B, 7B, 15B) với chiến lược 2 giai đoạn.
        *   Text Mining
            *   Taxonomy Generation & Text Classification
                *   LLM-based Methods
                    *   2403.12173 | Đề xuất TnT-LLM, tạo taxonomy và phân loại văn bản quy mô lớn bằng LLM (suy luận đa bước zero-shot, gán nhãn giả).
    *   Computer Vision (CV)
        *   Generative Models
            *   Image Generation
                *   Text-to-Image Synthesis
                    *   Cascaded Diffusion Models
                        *   Relay Diffusion
                            *   2403.05121 | Đề xuất CogView3, áp dụng relay diffusion trong không gian tiềm ẩn với lịch trình làm mờ tuyến tính đơn giản hóa.
                    *   Efficient Diffusion Models / Knowledge Distillation
                        *   2403.16627 | Đề xuất SDXS, chưng cất ControlNet, LFM (mất mát khớp nối đặc trưng SSIM) và Chưng cất điểm số phân đoạn.
                    *   Enhancing Prompt Following with LLMs
                        *   2403.05135 | Đề xuất ELLA, tích hợp LLM vào diffusion model qua Timestep-Aware Semantic Connector (TSC) huấn luyện trên dữ liệu chú thích lại bằng MLLM.
                    *   Personalized Text-to-Image Synthesis
                        *   2403.13535 | Đề xuất IDAdapter, sử dụng Mixed Facial Features (MFF) từ nhiều ảnh tham chiếu và adapter chuyên biệt để cá nhân hóa ảnh người.
                        *   2403.11781 | Đề xuất Infinite-ID, tách biệt định danh-ngữ nghĩa, huấn luyện tăng cường ID (vô hiệu hóa cross-attention văn bản) và tương tác đặc trưng (mixed attention, AdaIN-mean) khi suy luận.
                    *   Text-to-Image Synthesis with Regional Control
                        *   2403.09055 | Đề xuất SemanticDraw, 3 chiến lược ổn định hóa (tiền trung bình hóa latent, bootstrapping tập trung-mặt nạ, mặt nạ lượng tử hóa) và pipeline luồng đa-văn bản cho MultiDiffusion với bộ lập lịch tăng tốc.
                *   Image Editing
                    *   Photorealistic Object Removal and Insertion
                        *   2403.18818 | Đề xuất ObjectDrop, huấn luyện có giám sát phản thực tế (ảnh trước/sau loại bỏ vật thể) và giám sát bootstrap cho chèn vật thể.
                    *   Point-based Manipulation
                        *   2403.04437 | Đề xuất StableDrag, theo dõi điểm phân biệt (học bộ lọc) và tăng cường latent dựa trên độ tin cậy cho kéo thả điểm ảnh.
                *   Diffusion Models (General)
                    *   Model Distillation
                        *   2403.12015 | Đề xuất LADD, chưng cất khuếch tán đối nghịch tiềm ẩn sử dụng đặc trưng sinh từ teacher và hoạt động hoàn toàn trong không gian ẩn.
                    *   Parallel Inference Acceleration
                        *   2402.19481 | Đề xuất DistriFusion, song song hóa patch phân tán với giao tiếp bất đồng bộ, phép toán thưa và GN bất đồng bộ để tăng tốc suy luận khuếch tán.
            *   Video Generation
                *   Text-to-Video Generation
                    *   Multi-Agent Frameworks
                        *   2403.13248 | Đề xuất Mora, khung đa tác tử (tích hợp module nguồn mở) với tinh chỉnh tự điều biến và huấn luyện data-free cho tạo video.
                    *   Diffusion-based Video Generation Enhancement
                        *   2403.05438 | Đề xuất VideoElevator, plug-and-play nâng cao chất lượng T2V bằng T2I (tinh chỉnh chuyển động T2V, nâng cao chất lượng không gian T2I).
                *   Image-to-Video Synthesis
                    *   2403.01800 | Đề xuất AtomoVideo, bơm thông tin ảnh đa mức độ, huấn luyện zero terminal SNR & v-prediction, kiến trúc adapter cho I2V.
                *   Video Editing
                    *   Diffusion-based Video Editing
                        *   2403.14468 | Đề xuất AnyV2V, khung không cần tinh chỉnh, tách chỉnh sửa ảnh khung đầu và sinh video I2V có bơm đặc trưng từ video gốc.
                        *   2403.09334 | Đề xuất EVE/FDD, chưng cất khuếch tán nhân tố hóa không giám sát để căn chỉnh mô hình chỉnh sửa video từ nhiều giáo viên chuyên biệt.
                *   Audio-driven Avatars/Humans
                    *   2403.08764 | Đề xuất VLOGGER, mô hình khuếch tán стохастик (M) dự đoán chuyển động 3D (khuôn mặt, cơ thể) từ âm thanh, và kiến trúc khuếch tán video dùng render 3D làm điều khiển.
                *   Latent Video Diffusion Models
                    *   2403.14148 | Đề xuất CMD, phân tách video thành khung nội dung (sinh bởi LDM ảnh tinh chỉnh) và biểu diễn chuyển động tiềm ẩn (sinh bởi DiT nhẹ).
        *   3D Computer Vision
            *   Novel View Synthesis
                *   Explicit Scene Representations
                    *   Gaussian Splatting Variants
                        *   2403.17888 | Đề xuất 2DGS, biểu diễn cảnh bằng đĩa Gaussian 2D, splatting 2D vi phân chính xác và các hàm mục tiêu (méo độ sâu, nhất quán pháp tuyến).
                *   Multi-view Consistent Generation using Video Diffusion
                    *   2403.12008 | Đề xuất SV3D, adapt mô hình SVD cho NVS có điều khiển tư thế camera, quỹ đạo động và Triangular CFG Scaling.
            *   Single-View 3D Reconstruction
                *   Feed-forward Mesh Generation
                    *   2403.05034 | Đề xuất CRM, U-Net tích chập tạo triplane từ 6 ảnh trực giao + CCM, và biểu diễn Flexicubes cho tối ưu lưới end-to-end.
                *   Amortized Reconstruction with Gaussian Splatting
                    *   2403.18795 | Đề xuất GambaFormer (Mamba cho 3DGS), ràng buộc mặt nạ xuyên tâm và huấn luyện tăng tiến số góc nhìn.
            *   Text-to-3D Generation
                *   Alignment with Human Preferences
                    *   2403.14613 | Đề xuất Reward3D (RM 3D-aware) và DreamFL (tinh chỉnh MVD bằng Reward3D) cho text-to-3D.
            *   3D Representation
                *   Structured Explicit Radiance Fields
                    *   2403.19655 | Đề xuất GaussianCube, khớp Gaussian ràng buộc tăng cường và Vận chuyển Tối ưu để cấu trúc hóa 3DGS vào lưới voxel.
            *   Digital Clothing
                *   Image-based 3D Garment Generation and Stylization
                    *   2403.18816 | Đề xuất Garment3DGen, biến dạng lưới bảo toàn topology (giám sát bằng pseudo-GT 3D, loss 2D/embedding) và tạo texture UV chi tiết.
        *   Representation Learning
            *   Self-Supervised Learning
                *   Joint-Embedding Predictive Architectures (JEPA) / World Models
                    *   2403.00504 | Đề xuất Image World Models (IWM), JEPA học dự đoán tác động biến đổi quang học toàn cục trong không gian ẩn, và predictor finetuning.
        *   Vision Transformer Architectures
            *   Unified Architectures / Positional Encoding Techniques
                *   2403.00522 | Đề xuất VisionLLaMA (áp dụng LLaMA components cho ViT) và AS2DRoPE (RoPE 2D tự co giãn cho độ phân giải tùy ý).
            *   Resolution Adaptation / Scalability
                *   2403.18361 | Đề xuất ViTAR với Adaptive Token Merger (ATM) và Fuzzy Positional Encoding (FPE) để xử lý ảnh đa độ phân giải hiệu quả.
                *   2403.13043 | Hệ thống hóa "Scaling on Scales" (S2), áp dụng mô hình thị giác nhỏ cố định trên nhiều độ phân giải ảnh đầu vào.
        *   Unified Vision Models
            *   Vision Foundation Models
                *   Autoregressive Vision Modeling
                    *   2403.09394 | Đề xuất GiT, Transformer đa lớp với giao diện ngôn ngữ phổ quát (OOV) và giải mã song song cho nhiều tác vụ thị giác.
        *   Video Understanding
            *   Action Recognition / Long Video Understanding
                *   State Space Models for Video
                    *   2403.06977 | Đề xuất VideoMamba, SSM thuần cho video với khối B-Mamba hai chiều và quét không-thời gian "Spatial-First".
            *   Long-form Video Understanding
                *   Agent-based Video Question Answering
                    *   2403.10517 | Đề xuất VideoAgent, hệ thống agent LLM với chọn lọc khung hình lặp lại, tạo truy vấn và tự đánh giá để hiểu video dài.
            *   Long Video Understanding (General)
                *   Synthetic Data Generation for Video Instruction Tuning
                    *   2403.01422 | Đề xuất MovieLLM, pipeline tạo dữ liệu hướng dẫn phim dài (cốt truyện, khung hình, hội thoại, QA) bằng GPT-4 và textual inversion.
    *   Multimodal AI
        *   Large Language and Vision Models (LLVMs / MLLMs)
            *   Training Strategies and Component Analysis
                *   2403.09611 | Phân tích hệ thống ảnh hưởng của kiến trúc (encoder ảnh, connector) và dữ liệu tiền huấn luyện lên hiệu năng MLLM (MM1).
            *   Efficient Architectures and Training Strategies
                *   2403.05525 | Đề xuất DeepSeek-VL với bộ mã hóa thị giác lai (SigLIP+SAM), SFT data từ use case taxonomy, và modality warm-up.
                *   State-Space Model based MLLMs
                    *   2403.14520 | Đề xuất Cobra, MLLM dựa trên Mamba, kết hợp DINOv2+SigLIP, và huấn luyện trực tiếp không tiền-căn chỉnh.
            *   Inference Optimization
                *   Token Pruning
                    *   2403.06764 | Đề xuất FastV, tỉa token hình ảnh plug-and-play trong lớp sâu LVLM dựa trên điểm chú ý trung bình.
            *   Knowledge Integration and Fusion
                *   Leveraging External Vision Models via Mixture of Experts
                    *   2403.07508 | Đề xuất MoAI, tích hợp thông tin từ nhiều mô hình CV chuyên biệt vào MLLM qua MoAI-Compressor và MoAI-Mixer (MoE attention).
            *   Visual Document Understanding
                *   OCR-free Document Understanding
                    *   Unified Structure Learning for MLLMs
                        *   2403.12895 | Đề xuất DocOwl 1.5 với Unified Structure Learning (structure-aware parsing, multi-grained text localization) và H-Reducer.
            *   Code Generation
                *   Image-to-Code Generation
                    *   Front-End Web Development
                        *   2403.03163 | Xây dựng benchmark Design2Code (web thực tế) và bộ chỉ số đánh giá tự động (CLIP, Block-Match) cho image-to-code.
                        *   2403.09029 | Đề xuất WebSight, quy trình tổng hợp dữ liệu ảnh-HTML (Tailwind CSS) quy mô lớn bằng LLM.
            *   Efficient High-Resolution VLMs
                *   2403.18814 | Đề xuất Mini-Gemini với Dual Vision Encoders (ViT LR, ConvNet HR), Patch Info Mining, và SFT tạo prompt cho sinh ảnh.
            *   Video Large Language Models (Video LLMs)
                *   Temporal Localization and Reasoning
                    *   2403.19046 | Đề xuất LITA với token thời gian, token SlowFast, và tác vụ/bộ dữ liệu RTL cho định vị thời gian kết hợp suy luận.
                *   Large-Scale Pretraining for Video-Audio-Text
                    *   2403.15377 | Đề xuất InternVideo2, huấn luyện lũy tiến 3 giai đoạn (tái tạo token không che, học tương phản đa phương tiện, dự đoán token tiếp theo) và hệ thống chú thích VidCap.
        *   Multimodal Document Understanding
            *   Scientific Literature Analysis
                *   2403.10301 | Đề xuất Uni-SMART, tích hợp phân tích đa phương thức và LLM cho tài liệu khoa học, với RAG đa phương thức và huấn luyện lặp lại.
    *   Machine Learning (ML)
        *   Optimization Methods
            *   Memory-Efficient Training
                *   (Duplicate of NLP > LLMs > Efficient Training > Memory-Efficient Training) 2403.03507
        *   Time Series Analysis
            *   Forecasting
                *   Pretrained Foundation Models
                    *   2403.07815 | Đề xuất Chronos, framework huấn luyện trước mô hình chuỗi thời gian xác suất dựa trên LM tiêu chuẩn, token hóa (mean scaling, quantization) và KernelSynth.
        *   Evaluation and Benchmarking
            *   Visual Mathematical Reasoning Evaluation
                *   2403.14624 | Đề xuất MATHVERSE, quy trình biến đổi bài toán toán trực quan (6 phiên bản) và chiến lược đánh giá CoT bằng GPT-4.
            *   Large Language Model Evaluation
                *   Human Preference Based Evaluation
                    *   Crowdsourced Pairwise Comparison Ranking
                        *   2403.04132 | Đề xuất thuật toán lấy mẫu chủ động (active sampling) và quy trình phát hiện bất thường cho Chatbot Arena.
    *   AI Security & Privacy
        *   Model Stealing Attacks
            *   Language Model Parameter Extraction
                *   2403.06634 | Đề xuất tấn công top-down trích xuất lớp nhúng cuối và chiều ẩn của LLM hộp đen qua API (SVD trên logit, tái tạo logit).
    *   AI Systems
        *   Systems for Large Models
            *   Training Efficiency
                *   Memory Offloading and Scheduling
                    *   2403.06504 | Đề xuất LoHan, active gradient offloading (optimizer trên CPU chồng lấp backward GPU) và holistic traffic-aware activation swapping (swap/recompute) cho huấn luyện LLM trên GPU đơn lẻ.
        *   Agent Architectures and Systems
            *   Agent Operating Systems
                *   2403.16971 | Đề xuất AIOS, kiến trúc kernel (scheduler, context/memory/storage/tool manager, LLM core) cho agent LLM.
    *   Robotics
        *   Robot Learning
            *   Locomotion Control
                *   Generative Modeling for Humanoid Locomotion
                    *   2402.19469 | Định hình điều khiển robot hình người thành dự đoán token tiếp theo trên quỹ đạo cảm biến-vận động, xử lý thiếu phương thức bằng token mặt nạ.
        *   Human-Robot Interaction
            *   Collaborative Robotics
                *   Modular Systems with Foundation Models
                    *   2402.18796 | Đề xuất MOSAIC, kiến trúc mô-đun (LLM, VLM, dự báo chuyển động) với LLM nhúng trong Cây Hành Vi cho nấu ăn cộng tác.
    *   Deep Learning (General)
        *   Sequence Modeling
            *   State Space Models
                *   Architectural Enhancements
                    *   2403.00818 | Đề xuất DenseSSM, kết nối dày đặc trạng thái ẩn (hoặc K/V cho RetNet) với mô-đun chuyển tiếp chọn lọc và hợp nhất.
            *   Recurrent Neural Networks
                *   Gated Recurrent Units
                    *   2402.19427 | Đề xuất RG-LRU (gating kép số thực) và kiến trúc lai Griffin (RG-LRU + local attention), Hawk (thuần RG-LRU).
    *   Distributed Computing
        *   Large Scale Machine Learning
            *   Distributed Training
                *   Sequence Parallelism for Attention
                    *   2403.09347 | Đề xuất BurstAttention, attention phân tán (phân chia chuỗi, giao tiếp vòng) với GAO (online softmax) và LAO (tiling cục bộ).
    *   Audio Processing
        *   Generative Audio Synthesis
            *   High-Fidelity Vocoding and Enhancement
                *   2403.10493 | Đề xuất MusicHiFi, GAN thống nhất cho vocoding, BWE (kết nối tắt), M2S (mã hóa mid-side).
        *   Speech Processing
            *   Speech Synthesis
                *   Text-to-Speech (TTS)
                    *   Zero-Shot TTS
                        *   2403.03100 | Đề xuất NaturalSpeech 3 với FACodec (FVQ tách rời thuộc tính âm thanh) và Factorized Diffusion Model (sinh thuộc tính tuần tự).
    *   Other
        *   (Papers that are primarily position papers, surveys, or very specific analyses without a primary novel technical contribution fitting elsewhere)
            *   2403.04732 | Đánh giá và phân tích khả năng suy luận diễn dịch dựa trên thị giác của VLM bằng RPMs.
            *   2403.15371 | Đánh giá khả năng khám phá tự nhiên của LLM trong MAB bằng học trong ngữ cảnh.
            *   2403.04706 | Phân tích và chứng minh năng lực giải toán tiềm ẩn của LLaMA-2 7B, khai thác bằng SFT quy mô lớn với dữ liệu tổng hợp.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                          | Đột phá                                                                                                                                  | Ảnh hưởng                                                                                                                                         |
    | :--- | :-------- | :------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
    | 1    | 2403.05530 | Gemini 1.5, Multimodal, Long Context, 10M tokens, MoE   | Xử lý ngữ cảnh dài tới 10 triệu token đa phương thức (văn bản, ảnh, video, âm thanh) với khả năng truy xuất gần như hoàn hảo.             | Đặt ra tiêu chuẩn mới cho khả năng xử lý ngữ cảnh siêu dài, mở ra nhiều ứng dụng mới cho MLLM trong việc hiểu và tương tác với lượng lớn thông tin. |
    | 2    | 2403.03507 | LLM Training, Memory Efficiency, GaLore, Low-Rank Gradient | GaLore: Giảm 65% bộ nhớ optimizer khi tiền huấn luyện LLaMA 7B mà vẫn giữ hiệu năng, cho phép huấn luyện mô hình lớn hơn trên phần cứng hạn chế. | Giải quyết một nút thắt lớn trong huấn luyện LLM, dân chủ hóa việc huấn luyện các mô hình lớn.                                                      |
    | 3    | 2402.19469 | Humanoid Robot, Locomotion, Generative Model, Missing Modality | Điều khiển robot hình người Digit (zero-shot) bằng cách mô hình hóa quỹ đạo cảm biến-vận động như dự đoán token, xử lý dữ liệu thiếu phương thức. | Mở ra hướng tiếp cận mới cho điều khiển robot phức tạp bằng cách tận dụng sức mạnh của mô hình sinh và dữ liệu đa dạng (kể cả video người).        |
    | 4    | 2403.03206 | Text-to-Image, Rectified Flow, Diffusion Transformer, Weighted Timestep Sampling | MM-DiT với lấy mẫu bước thời gian có trọng số cho Rectified Flow đạt hiệu năng SOTA, vượt trội các mô hình khuếch tán hàng đầu.                 | Cải thiện đáng kể chất lượng và hiệu quả của các mô hình sinh ảnh dựa trên Rectified Flow, có thể thay thế các mô hình khuếch tán phức tạp hơn.   |
    | 5    | 2402.19427 | Efficient LLM, Griffin, RG-LRU, Hybrid RNN-Attention    | Griffin (RG-LRU + Local Attention) đạt hiệu năng tương đương Llama-2 với dữ liệu ít hơn và suy luận nhanh hơn đáng kể.                     | Cung cấp một kiến trúc thay thế Transformer hiệu quả cao, đặc biệt cho các ứng dụng yêu cầu độ trễ thấp và thông lượng cao.                       |
    | 6    | 2403.06738 | 3D Generation, Video Diffusion, Multi-view, Camera Control | V3D: Sử dụng mô hình khuếch tán video (SVD) để tạo đa góc nhìn 3D chất lượng cao từ ảnh đơn với khả năng điều khiển camera rõ ràng.         | Cung cấp phương pháp tạo 3D nhanh, chất lượng cao, có kiểm soát, tận dụng các mô hình video mạnh mẽ.                                             |
    | 7    | 2403.18802 | LLM Evaluation, Long-form Factuality, Search-Augmented, SAFE | SAFE: Phương pháp đánh giá tự động tính xác thực dạng dài bằng LLM agent và Google Search, chính xác hơn con người và rẻ hơn 20 lần.      | Giải quyết thách thức lớn trong việc đánh giá độ tin cậy của LLM khi tạo nội dung dài, cung cấp công cụ hiệu quả và có thể mở rộng.             |
    | 8    | 2403.04692 | Text-to-Image, Diffusion Transformer, High-Resolution, KV Compression | PixArt-Σ: Sinh ảnh 4K trực tiếp từ DiT 0.6B tham số với mô-đun nén KV, đạt chất lượng SOTA.                                            | Cho phép tạo ảnh độ phân giải siêu cao với mô hình tương đối nhỏ, mở rộng khả năng ứng dụng của DiT.                                               |
    | 9    | 2403.13187 | Model Merging, Evolutionary Algorithm, Parameter Space, Data Flow Space | Evolutionary Model Merge: Tự động khám phá công thức trộn mô hình (tham số và luồng dữ liệu) bằng thuật toán tiến hóa, tạo mô hình SOTA. | Cung cấp phương pháp tự động, không cần huấn luyện để tạo ra các mô hình nền tảng mạnh mẽ từ các mô hình nguồn mở có sẵn.                          |
    | 10   | 2403.03100 | Text-to-Speech, Zero-Shot, Disentangled Attributes, Factorized Diffusion | NaturalSpeech 3: FACodec tách rời thuộc tính âm thanh và Factorized Diffusion Model sinh giọng nói chất lượng cao, tự nhiên, có kiểm soát. | Đạt được chất lượng giọng nói tổng hợp SOTA, cải thiện đáng kể tính tự nhiên, tương đồng và khả năng kiểm soát trong TTS zero-shot.              |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2403.03507 – Chiếu gradient vào không gian hạng thấp (GaLore) và cơ chế chuyển không gian con định kỳ – Suy nghĩ:** Giải pháp thông minh để giảm bộ nhớ optimizer mà vẫn học toàn bộ tham số, rất thực tế cho huấn luyện LLM lớn.
    *   **2402.19173 – Quy trình xử lý dữ liệu The Stack v2 đa bước chi tiết, bao gồm phát hiện giấy phép/ngôn ngữ tiên tiến và lọc dựa trên cộng đồng – Suy nghĩ:** Mặc dù không phải thuật toán mới, đây là một nỗ lực kỹ thuật đáng kể, tạo ra tài nguyên chất lượng cao và minh bạch cho cộng đồng Code LLM.
    *   **2403.03163 – Bộ chỉ số đánh giá tự động Design2Code dựa trên tương đồng thị giác (CLIP) và khớp nối chi tiết thành phần (Block-Match, Text, Position, Color) – Suy nghĩ:** Cung cấp cách đánh giá image-to-code khách quan hơn so với so sánh mã nguồn, tập trung vào kết quả thị giác.
    *   **2403.06634 – Kỹ thuật tái tạo vector logit đầy đủ từ API hạn chế (top-K, logit bias) bằng token tham chiếu hoặc ép tạo chuỗi – Suy nghĩ:** Phương pháp tấn công sáng tạo, cho thấy lỗ hổng tiềm ẩn ngay cả với các API LLM có vẻ an toàn.
    *   **2403.13372 – Chiến lược Model-Sharing RLHF sử dụng adapters và value heads riêng biệt trên một mô hình gốc duy nhất – Suy nghĩ:** Giải pháp thực tiễn và hiệu quả để giảm đáng kể yêu cầu bộ nhớ cho RLHF, giúp dân chủ hóa phương pháp này.
    *   **2403.03883 – Tạo dữ liệu hướng dẫn pháp lý tổng hợp bằng LLM mô phỏng hội thoại đa lượt (người dùng-trợ lý) – Suy nghĩ:** Hướng tiếp cận sáng tạo để tạo dữ liệu tinh chỉnh chuyên biệt, phù hợp cho các tác vụ đòi hỏi lý luận sâu.
    *   **2403.17887 – Thước đo Block Influence (BI) dựa trên thay đổi trạng thái ẩn đầu vào-đầu ra của lớp để tỉa lớp (ShortGPT) – Suy nghĩ:** Tiêu chí tỉa lớp trực quan, dựa trên chức năng, có tiềm năng hiệu quả hơn các phương pháp dựa trên độ lớn.
    *   **2403.13248 – Tinh chỉnh đa tác tử tự điều biến (self-modulated) với embedding điều biến học được cho phối hợp tác tử (Mora) – Suy nghĩ:** Cơ chế thú vị để cải thiện sự phối hợp giữa các mô-đun trong hệ thống đa tác tử.
    *   **2403.09629 – Thuật toán lấy mẫu song song tạo suy nghĩ tiềm ẩn từ mọi vị trí token bằng attention mask đặc biệt (Quiet-STaR) – Suy nghĩ:** Giải pháp kỹ thuật quan trọng giúp việc học suy luận ngầm từ dữ liệu lớn trở nên khả thi về mặt tính toán.
    *   **2403.07508 – MoAI-Mixer: MoE attention hòa trộn đặc trưng thị giác gốc, phụ trợ (từ CV models) và ngôn ngữ (MoAI) – Suy nghĩ:** Thiết kế MoE tường minh, sử dụng expert attention để hòa trộn các nguồn thông tin khác nhau một cách có chủ đích, rất sáng tạo.
    *   **2403.10131 – Xây dựng dữ liệu huấn luyện RAFT với tài liệu "vàng" và "gây nhiễu", yêu cầu CoT có trích dẫn – Suy nghĩ:** Cách chuẩn bị dữ liệu thông minh để LLM học cách phớt lờ nhiễu và căn cứ vào bằng chứng.
    *   **2403.16971 – Context Manager (AIOS) với snapshot/restore ngữ cảnh LLM (dựa trên logits/text) cho time-slicing – Suy nghĩ:** Giải pháp quan trọng cho phép lập lịch và quản lý tài nguyên hiệu quả cho các tác vụ LLM dài hạn trong môi trường đa agent.
    *   **2403.12015 – Đặc trưng sinh từ teacher LDM làm đầu vào cho discriminator, hoạt động hoàn toàn trong không gian ẩn (LADD) – Suy nghĩ:** Đơn giản hóa và cải tiến đáng kể ADD, đặc biệt hiệu quả cho chưng cất LDM lớn, đa tỷ lệ.
    *   **2403.03206 – Lấy mẫu bước thời gian có trọng số (Logit-Normal, Mode, CosMap) cho Rectified Flow và kiến trúc MM-DiT (luồng xử lý riêng, attention chung) – Suy nghĩ:** Các đóng góp kỹ thuật rõ ràng, cải thiện hiệu quả RF và tăng cường tương tác đa phương thức trong DiT.
    *   **2403.18361 – Adaptive Token Merger (ATM) gộp token lặp lại bằng GridAttention và Fuzzy Positional Encoding (FPE) cho ViTAR – Suy nghĩ:** Giải pháp hiệu quả để ViT xử lý ảnh đa độ phân giải mà vẫn giữ hiệu năng và tương thích MAE.
    *   **2403.09029 – Quy trình tổng hợp dữ liệu WebSight (HTML + Tailwind CSS) bằng LLM hai giai đoạn và chèn ảnh động từ URL – Suy nghĩ:** Giải pháp sáng tạo để tạo dữ liệu image-to-HTML quy mô lớn, chất lượng cao.
    *   **2403.06504 – Active gradient offloading (optimizer CPU chồng lấp backward GPU) và holistic traffic-aware activation swapping (LoHan) – Suy nghĩ:** Các kỹ thuật tối ưu hóa hệ thống sâu sắc, giải quyết nút thắt cổ chai trong huấn luyện LLM trên GPU đơn lẻ.
    *   **2403.13187 – Trộn mô hình trong không gian luồng dữ liệu (DFS merging) bằng tối ưu hóa tiến hóa trình tự lớp và hệ số tỷ lệ (Evolutionary Model Merge) – Suy nghĩ:** Phương pháp rất sáng tạo, cho phép tạo kiến trúc lai ghép mạnh mẽ mà không cần huấn luyện lại.
    *   **2403.10301 – Quy trình tiền xử lý chuyển đổi yếu tố đa phương thức (bảng, biểu đồ, phân tử, phản ứng) thành định dạng văn bản tùy chỉnh cho LLM (Uni-SMART) – Suy nghĩ:** Hướng đi thực tế để LLM xử lý tài liệu khoa học phức tạp, dù cần chi tiết hơn về định dạng.
    *   **2402.19155 – Mô hình bGPT (Transformer phân cấp patch-byte) mô phỏng thuật toán và phần cứng từ chuỗi byte – Suy nghĩ:** Ý tưởng thú vị về "Byte Models" như bộ mô phỏng thế giới số, mở rộng khả năng của mô hình học sâu.
    *   **2403.14624 – Quy trình biến đổi bài toán toán trực quan thành 6 phiên bản (cân bằng thông tin văn bản-hình ảnh) và chiến lược đánh giá CoT bằng GPT-4 (MATHVERSE) – Suy nghĩ:** Phương pháp luận đánh giá sáng tạo, cho phép đo lường sâu hơn khả năng hiểu hình ảnh của MLLM.
    *   **2402.19427 – Lớp RG-LRU (gating kép số thực, không phụ thuộc trạng thái ẩn trước) và kiến trúc lai Griffin (RG-LRU + local attention) – Suy nghĩ:** Đóng góp kỹ thuật đáng chú ý cho mô hình ngôn ngữ hiệu quả, cạnh tranh với Transformer/Mamba.
    *   **2403.18814 – Patch Info Mining: tổng hợp thông tin từ ảnh phân giải cao vào embedding thị giác phân giải thấp qua cross-attention cục bộ (Mini-Gemini) – Suy nghĩ:** Cơ chế hiệu quả để VLM xử lý ảnh HR mà không tăng số token đầu vào LLM.
    *   **2403.07815 – Token hóa chuỗi thời gian (mean scaling, uniform quantization) cho LM tiêu chuẩn và KernelSynth (sinh dữ liệu GP với kernel kết hợp) (Chronos) – Suy nghĩ:** Cách tiếp cận tối giản nhưng hiệu quả đáng ngạc nhiên cho dự báo chuỗi thời gian zero-shot.
    *   **2403.00522 – AS2DRoPE: RoPE 2D tự động co giãn cho độ phân giải tùy ý trong VisionLLaMA – Suy nghĩ:** Mở rộng RoPE hợp lý cho thị giác, giải quyết vấn đề độ phân giải thay đổi mà không cần huấn luyện lại.
    *   **2403.05135 – Timestep-Aware Semantic Connector (TSC) với AdaLN tích hợp LLM vào diffusion model (ELLA) – Suy nghĩ:** Kỹ thuật thông minh, cho phép LLM cung cấp ngữ nghĩa phù hợp với từng giai đoạn khử nhiễu.
    *   **2403.05525 – Bộ mã hóa thị giác lai (SigLIP LR + SAM HR) và chiến lược "modality warm-up" cho DeepSeek-VL – Suy nghĩ:** Giải pháp thực tế để xử lý ảnh HR hiệu quả và bảo tồn năng lực ngôn ngữ khi huấn luyện LMM.
    *   **2403.19270 – Stepwise DPO (sDPO): DPO lặp lại trên phân vùng dữ liệu tuần tự với cập nhật mô hình tham chiếu – Suy nghĩ:** Mở rộng DPO sáng tạo, giải quyết hạn chế về mô hình tham chiếu cố định, tạo hiệu ứng curriculum learning.
    *   **2403.04746 – Bộ nhớ ngắn hạn (quỹ đạo chi tiết) và dài hạn (kinh nghiệm cô đọng) trong STE để LLM học sử dụng API – Suy nghĩ:** Cơ chế bộ nhớ lấy cảm hứng từ sinh học, giúp LLM học tương tác và khám phá API hiệu quả.
    *   **2402.18796 – LLM nhúng trong Cây Hành Vi (Behavior Tree) cho lập kế hoạch nhiệm vụ robot cộng tác (MOSAIC) – Suy nghĩ:** Giải pháp mô-đun hóa, chia nhỏ bài toán lý luận phức tạp cho LLM, giảm lỗi và tăng kiểm soát.
    *   **2403.03100 – FACodec: FVQ tách rời thuộc tính âm thanh (nội dung, ngữ điệu, âm sắc, chi tiết) bằng information bottleneck, supervision, GRL, detail dropout (NaturalSpeech 3) – Suy nghĩ:** Kiến trúc tách rời thuộc tính âm thanh rất tinh vi và hiệu quả.
    *   **2403.08764 – Mô hình khuếch tán стохастик (M) dự đoán chuyển động 3D (khuôn mặt, cơ thể) từ âm thanh và kiến trúc khuếch tán video dùng render 3D làm điều khiển (VLOGGER) – Suy nghĩ:** Hướng đi mới lạ, tạo video người nói chuyện toàn thân biểu cảm từ âm thanh và ảnh đơn.
    *   **2403.01779 – Cơ chế hợp nhất trang phục (outfitting fusion) trong self-attention và dropout trang phục cho thử đồ ảo (OOTDiffusion) – Suy nghĩ:** Giải pháp VTON không cần warping, bảo toàn chi tiết tốt và có khả năng kiểm soát.
    *   **2403.17888 – Biểu diễn cảnh bằng đĩa Gaussian 2D, splatting 2D vi phân chính xác và các hàm mục tiêu (méo độ sâu, nhất quán pháp tuyến) (2DGS) – Suy nghĩ:** Cải tiến đáng kể so với 3DGS, nâng cao độ chính xác hình học và chất lượng bề mặt.
    *   **2403.04437 – Theo dõi điểm phân biệt (học bộ lọc) và tăng cường latent dựa trên độ tin cậy cho kéo thả điểm ảnh (StableDrag) – Suy nghĩ:** Giải quyết hạn chế của DragGAN/DragDiffusion, tăng ổn định và độ chính xác theo dõi.
    *   **2403.01422 – "Story planner" (LLM phân cấp) tạo cốt truyện/khung hình/hội thoại/QA và textual inversion cố định phong cách cho tạo dữ liệu phim dài (MovieLLM) – Suy nghĩ:** Pipeline sáng tạo để giải quyết vấn đề thiếu dữ liệu huấn luyện video dài có hướng dẫn.
    *   **2403.06764 – FastV: Tỉa token hình ảnh plug-and-play trong lớp sâu LVLM dựa trên điểm chú ý trung bình – Suy nghĩ:** Giải pháp đơn giản, trực quan, không cần huấn luyện lại để giảm chi phí suy luận LVLM.
    *   **2403.12895 – H-Reducer: mã hóa thông tin bố cục và giảm đặc trưng thị giác cho ảnh HR bằng tích chập ngang (DocOwl 1.5) – Suy nghĩ:** Giải pháp kỹ thuật đơn giản nhưng hiệu quả để xử lý ảnh tài liệu độ phân giải cao.
    *   **2403.16990 – Bounded Attention (mặt nạ trong softmax attention) và Bounded Guidance (loss dựa trên attention che mặt nạ) cho sinh ảnh đa chủ thể – Suy nghĩ:** Kỹ thuật không cần huấn luyện, giải quyết rò rỉ ngữ nghĩa/hình ảnh khi tạo nhiều chủ thể.
    *   **2403.15377 – Tái tạo token video không che đậy (hướng dẫn bởi 2 chuyên gia) và hệ thống chú thích VidCap (tổng hợp từ nhiều nguồn bằng LLM) (InternVideo2) – Suy nghĩ:** Các kỹ thuật tiền huấn luyện và tạo dữ liệu mạnh mẽ cho mô hình video nền tảng.
    *   **2403.00818 – DenseSSM: Kết nối dày đặc trạng thái ẩn (hoặc K/V) liên lớp trong SSM với mô-đun chuyển tiếp chọn lọc và hợp nhất – Suy nghĩ:** Cải tiến kiến trúc hợp lý cho SSM, chống suy giảm thông tin ở lớp sâu.
    *   **2403.09347 – GAO (online softmax) và LAO (tiling cục bộ) trong BurstAttention để xử lý chuỗi dài phân tán – Suy nghĩ:** Kết hợp thông minh các kỹ thuật để tối ưu bộ nhớ và tính toán cho attention phân tán.
    *   **2403.05034 – CRM: U-Net tích chập tạo triplane từ 6 ảnh trực giao + CCM, khai thác tương ứng không gian – Suy nghĩ:** Hướng tiếp cận hợp lý, tận dụng tiên nghiệm hình học cho tái tạo 3D feed-forward.
    *   **2403.18795 – Ràng buộc mặt nạ xuyên tâm (radial mask constraint) dựa trên mặt nạ đa góc nhìn cho GambaFormer (Mamba cho 3DGS) – Suy nghĩ:** Loại bỏ sự cần thiết của giám sát đám mây điểm 3D, đơn giản hóa huấn luyện.
    *   **2403.19655 – Khớp Gaussian ràng buộc tăng cường và Vận chuyển Tối ưu để cấu trúc hóa 3DGS vào lưới voxel (GaussianCube) – Suy nghĩ:** Phương pháp sáng tạo để tạo biểu diễn 3DGS có cấu trúc, tường minh, phù hợp cho mô hình sinh.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Khả năng Khái quát hóa và Độ Tin cậy của LLM trong Các Miền Chuyên Biệt:** Mặc dù có các mô hình chuyên biệt (2403.18421, 2403.03883), việc đảm bảo LLM không "ảo giác" và đưa ra thông tin chính xác, đáng tin cậy trong các lĩnh vực đòi hỏi kiến thức sâu (y học, pháp lý) vẫn là thách thức. Cơ hội: Phát triển các phương pháp tích hợp kiến thức có cấu trúc một cách linh hoạt hơn, cơ chế tự kiểm tra và xác minh thông tin, và các kỹ thuật giải thích quyết định tốt hơn.
    *   **Hiệu quả Tính toán và Bộ nhớ cho Huấn luyện/Suy luận Mô hình Cực Lớn:** Các kỹ thuật như GaLore (2403.03507), LoHan (2403.06504), BurstAttention (2403.09347) đang giải quyết vấn đề này, nhưng vẫn còn không gian cho các giải pháp đột phá hơn, đặc biệt cho các mô hình với hàng nghìn tỷ tham số hoặc ngữ cảnh hàng chục triệu token (như Gemini 1.5 Pro 2403.05530). Cơ hội: Nghiên cứu kiến trúc phần cứng/phần mềm đồng thiết kế, thuật toán tối ưu hóa phân tán mới, và các phương pháp nén/lượng tử hóa triệt để hơn nữa.
    *   **Học Tương tác và Thích ứng Liên tục cho Tác tử AI:** Các tác tử như STE (2403.04746), SOTOPIA-π (2403.08715), MOSAIC (2402.18796) đang hướng tới khả năng học từ tương tác, nhưng việc học hiệu quả, an toàn và liên tục trong môi trường động, phức tạp vẫn còn nhiều khó khăn. Cơ hội: Phát triển các thuật toán RL hiệu quả hơn trong việc sử dụng phản hồi (ngôn ngữ, đa phương thức), cơ chế bộ nhớ và học hỏi kinh nghiệm tốt hơn, và khả năng tự khám phá/tò mò một cách an toàn.
    *   **Đánh giá Toàn diện và Đáng Tin cậy cho các Khả năng AI Nâng cao:** Các benchmark như Design2Code (2403.03163), MATHVERSE (2403.14624), RewardBench (2403.13787), LongFact (2403.18802) đang cố gắng giải quyết vấn đề này, nhưng việc đánh giá các khả năng phức tạp (suy luận đa bước, hiểu biết xã hội, sáng tạo, tính xác thực dạng dài) vẫn rất khó khăn và thường phụ thuộc vào đánh giá của con người hoặc LLM khác (có thể thiên vị). Cơ hội: Xây dựng các phương pháp đánh giá tự động, khách quan, có khả năng diễn giải và bao phủ rộng hơn các khía cạnh của trí tuệ nhân tạo.
    *   **Tạo Sinh 3D/Video Chất lượng Cao, Có Điều Khiển và Nhất quán:** Các mô hình như VLOGGER (2403.08764), CMD (2403.14148), V3D (2403.06738), Garment3DGen (2403.18816), 2DGS (2403.17888), GambaFormer (2403.18795), GaussianCube (2403.19655) cho thấy tiến bộ lớn, nhưng việc kiểm soát chi tiết, đảm bảo tính nhất quán vật lý và ngữ nghĩa trong thời gian dài, và tạo ra nội dung đa dạng, chất lượng cao vẫn là mục tiêu. Cơ hội: Nghiên cứu các biểu diễn không-thời gian hiệu quả hơn, mô hình động học thế giới tốt hơn, và các cơ chế điều khiển đa phương thức tinh vi hơn.
    *   **Sự Hợp nhất Kiến trúc và Nguyên lý Học giữa các Miền:** Các ý tưởng như VisionLLaMA (2403.00522), Byte Models (2402.19155), GiT (2403.09394) đang khám phá các kiến trúc thống nhất. Cơ hội: Tìm kiếm các nguyên lý học và biểu diễn chung có thể áp dụng hiệu quả cho nhiều loại dữ liệu và tác vụ, tiến tới các mô hình nền tảng thực sự tổng quát.
    *   **An toàn, Đạo đức và Quản trị AI:** Với sự phát triển của các mô hình ngày càng mạnh mẽ (Gemini 1.5 Pro 2403.05530, StarCoder2 2402.19173) và các kỹ thuật tấn công (2403.06634), việc đảm bảo AI được sử dụng một cách có trách nhiệm, công bằng và an toàn trở nên cấp thiết hơn bao giờ hết. Alignment Studio (2403.09704) là một bước đi. Cơ hội: Phát triển các kỹ thuật căn chỉnh mạnh mẽ hơn, cơ chế phát hiện và giảm thiểu thiên kiến, các phương pháp đảm bảo tính riêng tư và bảo mật dữ liệu, và các khung quản trị AI hiệu quả.

5.  **FUTURE_IDEAS**

    ✨ **Meta-Cognitive Agents with Self-Generated Curricula for Open-Ended Skill Acquisition**
    *   Motivation: Current agents (2403.04746, 2403.08715) learn specific skills or improve on predefined tasks. True general intelligence requires agents to identify their own knowledge gaps and autonomously design learning experiences.
    *   Key novelty: An agent that not only learns from interaction (like STE or SOTOPIA-π) but also meta-learns to generate a curriculum of increasingly complex, self-proposed tasks and environments (inspired by automated task generation in 2403.08715 but more open-ended) to continuously expand its capabilities without human intervention.
    *   Approach:
        1.  The agent maintains an internal "epistemic state" representing its current knowledge and uncertainty about the world/APIs/social norms.
        2.  A "curiosity module" (e.g., a separate LLM or a component of the main agent) analyzes this epistemic state to identify areas of high uncertainty or low competence.
        3.  This module then generates new "hypothetical tasks" or "exploratory scenarios" designed to reduce uncertainty or acquire new skills, potentially by combining known concepts in novel ways (similar to how KernelSynth 2403.07815 combines kernels).
        4.  The agent attempts these self-generated tasks, updating its epistemic state and core policy (e.g., using a refined STE-like process).
    *   Dataset + Metrics: Start with simulated environments (e.g., extended ToolBench, SOTOPIA, or even game environments like Minecraft). Metrics: Number of novel skills acquired over time, complexity of self-generated tasks, transferability of learned skills to unseen human-defined tasks, rate of epistemic uncertainty reduction.
    *   Risk/Feasibility: High risk. Defining and measuring "skills" and "knowledge gaps" autonomously is very challenging. The curiosity module could lead to unproductive exploration. Requires robust self-evaluation. (Moon-shot)

    ✨ **Differentiable Structured World Models for Compositional Video Generation and Understanding**
    *   Motivation: Current video generation (2403.13248, 2403.05438, 2403.01800, 2403.14148) and understanding (2403.10517, 2403.15377, 2403.06977) models often struggle with long-term consistency, object permanence, and complex causal interactions. SceneScript (2403.13064) offers a structured language representation for 3D scenes.
    *   Key novelty: Extend the idea of structured scene representation (like SceneScript) to dynamic 4D (3D + time) world models that are fully differentiable. This model would represent a scene not just as objects and layout, but as a graph of interacting entities with learnable physical/semantic properties and transition rules, all expressed in a structured, differentiable format.
    *   Approach:
        1.  Define a "dynamic scene script" language that includes commands for object instantiation, property assignment (physics, appearance), relationship definition, and action/event specification over time.
        2.  Develop a neural interpreter that can execute this script to produce (or condition) video frames (e.g., by controlling a differentiable renderer or a video diffusion model like V3D 2403.06738).
        3.  Train an encoder (e.g., VideoMamba 2403.06977 based) to parse an input video into this dynamic scene script.
        4.  The entire system (encoder, script representation, interpreter/renderer) is trained end-to-end, potentially with objectives like video reconstruction, future frame prediction, and answering causal questions about the video.
    *   Dataset + Metrics: Synthetic datasets with known ground-truth scene graphs and dynamics (e.g., CATER, Physion++), and real-world video datasets (ActivityNet, Something-Something). Metrics: Video prediction quality (PSNR, SSIM, FVD), accuracy of inferred scene scripts, compositional generalization to new object/action combinations, causal reasoning accuracy.
    *   Risk/Feasibility: Very high risk. Designing a sufficiently expressive yet learnable dynamic scene script language and a differentiable interpreter/renderer is a massive challenge. Scalability to complex real-world scenes is a major hurdle. (Moon-shot)

    ✨ **Federated Continual Pre-training with Adaptive Low-Rank Gradients**
    *   Motivation: Continual pre-training of LLMs (2403.08763) is crucial but centralized training is costly and raises data privacy concerns. GaLore (2403.03507) reduces optimizer memory using low-rank gradients.
    *   Key novelty: Combine federated learning with continual pre-training, where clients (e.g., different organizations or devices) continually pre-train a shared global model on their local, private data. To manage communication costs and client heterogeneity, clients only compute and transmit low-rank projections of their gradients (inspired by GaLore), and the rank `r` can be adapted based on client resources or data novelty.
    *   Approach:
        1.  A central server maintains the global LLM.
        2.  Clients download the current global model.
        3.  Each client performs continual pre-training on its local data stream, computing full gradients.
        4.  Instead of sending full gradients, each client computes a low-rank projection (e.g., using SVD or a learned projector) of its accumulated gradients. The rank `r` could be dynamically adjusted.
        5.  These low-rank gradient updates are sent to the server, which aggregates them (e.g., by reconstructing and averaging, or averaging in the low-rank space if compatible) to update the global model.
        6.  Incorporate techniques from 2403.08763 (LR re-warming, replay of a small global dataset subset) at the client or server level to manage catastrophic forgetting.
    *   Dataset + Metrics: Decentralized text corpora (e.g., web data crawled by different entities, domain-specific private datasets). Metrics: Perplexity on global and local test sets, downstream task performance after federated continual pre-training, communication efficiency, privacy preservation (if differential privacy is added to low-rank updates).
    *   Risk/Feasibility: Medium-high risk. Aggregating low-rank gradient updates effectively can be complex. Ensuring stability and convergence in a federated, continual, and low-rank setting is challenging. Adaptive rank selection adds another layer of complexity.

    ✨ **AIOS-Powered Personalized Education Agents**
    *   Motivation: AIOS (2403.16971) provides a kernel for managing LLM agent resources. Personalized education requires agents that can adapt to individual student needs, track progress over long periods, and utilize diverse learning tools/materials.
    *   Key novelty: Build a personalized education agent on top of an AIOS-like kernel. The agent would leverage AIOS's memory management for long-term student profiles, context management for handling diverse learning sessions (e.g., math problems, history lessons), and tool management for accessing educational APIs (e.g., Wolfram Alpha, Khan Academy).
    *   Approach:
        1.  Develop specialized "education managers" within the AIOS kernel (or as high-level services):
            *   Student Profile Manager: Stores learning history, strengths, weaknesses, preferred learning styles (long-term memory via AIOS storage).
            *   Curriculum Manager: Dynamically generates or adapts learning paths based on the student profile and educational goals (could use ideas from 2403.08715 for task generation).
            *   Instructional Tool Manager: Integrates various educational APIs and content sources.
        2.  The core LLM agent interacts with the student, plans lessons, retrieves information, and invokes tools via AIOS system calls.
        3.  Use RLHF (potentially Model-Sharing RLHF from 2403.13372 for efficiency) with feedback from students/teachers to fine-tune the agent's teaching strategies.
    *   Dataset + Metrics: Educational dialogues, textbook content, Q&A datasets (e.g., SQuAD, GSM8K adapted for tutoring). Metrics: Student learning gains (pre/post tests), engagement levels, task completion rates, student/teacher satisfaction scores.
    *   Risk/Feasibility: Medium risk. Integrating diverse educational tools and creating truly adaptive curricula is complex. Ensuring pedagogical soundness and avoiding harmful biases are critical. AIOS itself is still a research concept.

6.  **READING_LIST**

    *   2403.05530 – Gemini 1.5 Pro · Đột phá về xử lý ngữ cảnh siêu dài (10M token) đa phương thức, mở ra nhiều khả năng mới.
    *   2403.03507 – GaLore · Kỹ thuật giảm bộ nhớ optimizer rất quan trọng cho việc huấn luyện LLM lớn hiệu quả.
    *   2402.19469 – Humanoid Robot Control via Token Prediction · Hướng tiếp cận mới lạ và hiệu quả cho điều khiển robot phức tạp bằng mô hình sinh.
    *   2402.19427 – Griffin (RG-LRU) · Kiến trúc hồi quy hiệu quả, cạnh tranh mạnh mẽ với Transformer/Mamba về hiệu năng và tốc độ.
    *   2403.03206 – MM-DiT for Rectified Flow · Kết hợp kiến trúc Transformer đa phương thức mới và lấy mẫu bước thời gian có trọng số, đạt SOTA trong sinh ảnh.
    *   2403.13187 – Evolutionary Model Merge · Phương pháp tự động hóa việc trộn mô hình rất sáng tạo, không cần huấn luyện lại.
    *   2403.18802 – SAFE & LongFact · Giải pháp đánh giá tính xác thực dạng dài tự động, hiệu quả và cần thiết cho LLM.
    *   2403.06738 – V3D (Video Diffusion for 3D) · Ứng dụng thông minh mô hình khuếch tán video cho tạo 3D đa góc nhìn có kiểm soát.
    *   2403.16971 – AIOS (Agent Operating System) · Kiến trúc kernel tham vọng và cần thiết cho việc quản lý và phục vụ agent LLM hiệu quả.
    *   2403.04746 – STE (Simulated Trial and Error) · Phương pháp học sử dụng công cụ cho LLM rất trực quan và hiệu quả, dựa trên thử và sai có bộ nhớ.

7.  **META_REFLECTION**

    *   Tập hợp các bài báo tháng 03/2024 tiếp tục cho thấy sự phát triển mạnh mẽ của AI, đặc biệt trong lĩnh vực Mô hình Ngôn ngữ Lớn (LLM) và các ứng dụng đa phương thức. Các xu hướng chính bao gồm:
        1.  **Đẩy Xa Giới Hạn Năng Lực Mô Hình:** Các mô hình như Gemini 1.5 Pro (2403.05530) với khả năng xử lý ngữ cảnh 10 triệu token đa phương thức cho thấy một bước nhảy vọt về quy mô và khả năng. Đồng thời, các kiến trúc mới như Griffin (2402.19427) và các cải tiến cho SSM (DenseSSM 2403.00818, VideoMamba 2403.06977) đang thách thức sự thống trị của Transformer về hiệu quả và hiệu năng.
        2.  **Hiệu Quả Huấn Luyện và Suy Luận:** Nhu cầu về các mô hình hiệu quả hơn vẫn rất lớn. GaLore (2403.03507) giải quyết vấn đề bộ nhớ optimizer, trong khi các kỹ thuật như Model-Sharing RLHF (2403.13372), tỉa token (FastV 2403.06764), và tăng tốc suy luận khuếch tán (DistriFusion 2402.19481, LADD 2403.12015, SDXS 2403.16627) đang được tích cực nghiên cứu.
        3.  **Tạo Sinh Đa Phương Thức Nâng Cao:** Lĩnh vực tạo sinh (ảnh, video, 3D, âm thanh) tiếp tục có những đột phá ấn tượng. Các mô hình như CogView3 (2403.05121), VLOGGER (2403.08764), CMD (2403.14148), V3D (2403.06738), 2DGS (2403.17888), Garment3DGen (2403.18816), NaturalSpeech 3 (2403.03100) và MusicHiFi (2403.10493) cho thấy khả năng tạo ra nội dung ngày càng thực tế, có kiểm soát và chất lượng cao.
        4.  **Tác tử AI và Học Tương Tác:** Phát triển các tác tử AI có khả năng học hỏi từ tương tác, sử dụng công cụ và hoạt động trong các môi trường phức tạp là một hướng đi quan trọng (STE 2403.04746, SOTOPIA-π 2403.08715, MOSAIC 2402.18796, VideoAgent 2403.10517). Kiến trúc hệ thống cho agent (AIOS 2403.16971) cũng bắt đầu được hình thành.
        5.  **Chất lượng Dữ liệu và Quy trình Huấn luyện:** Tầm quan trọng của dữ liệu chất lượng cao và các chiến lược huấn luyện thông minh được nhấn mạnh qua các công trình về tạo dữ liệu tổng hợp (WebSight 2403.09029, MovieLLM 2403.01422, LLM2LLM 2403.15042, Panda-70M 2402.19479, Synth2 2403.07750), xử lý dữ liệu tiền huấn luyện (Yi 2403.04652), và các phương pháp huấn luyện/căn chỉnh mới (sDPO 2403.19270, RAFT 2403.10131, COOL RLHF 2403.17297).
        6.  **Đánh giá và Phân tích Mô hình:** Nhu cầu về các phương pháp đánh giá sâu sắc và đáng tin cậy ngày càng tăng, thể hiện qua các benchmark mới cho các khả năng cụ thể (Design2Code 2403.03163, MATHVERSE 2403.14624, RewardBench 2403.13787, LongFact 2403.18802) và các phân tích chuyên sâu về hành vi mô hình (2403.04732, 2403.15371, 2403.05812).
        7.  **Tính Mở và Tái Lập:** Việc công bố mô hình, dữ liệu và framework (StarCoder2 2402.19173, Gemma 2403.08295, LLaMA Factory 2403.13372, BioMedLM 2403.18421) tiếp tục là một xu hướng tích cực, thúc đẩy sự phát triển của cộng đồng.
Nhìn chung, tháng 03/2024 cho thấy sự trưởng thành hơn trong việc giải quyết các vấn đề cốt lõi của AI, từ hiệu quả tính toán, chất lượng dữ liệu, đến khả năng tương tác và độ tin cậy của mô hình, đồng thời mở ra những hướng đi mới đầy tham vọng cho tương lai.
