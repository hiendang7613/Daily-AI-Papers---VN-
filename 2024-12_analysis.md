1.  **TOPIC_TREE**

    ```markdown
    *   **Large Language Models (LLMs)**
        *   **Core Model Development & Architectures**
            *   **Foundation Model Engineering & Scaling:** 2412.15115 | Hệ thống hóa kiến thức xây dựng và huấn luyện dòng LLM Qwen2.5, tập trung vào quy mô dữ liệu và các giai đoạn huấn luyện.
            *   **Mixture-of-Experts (MoE):** 2412.01253 | Cải tiến kiến trúc MoE với cơ chế cân bằng tải mới (EP, PEP) và tối ưu hóa bộ nhớ KV cache cho ngữ cảnh dài.
            *   **Byte-level & Latent Space Modeling:**
                *   2412.09871 | Giới thiệu Byte Latent Transformer (BLT) với patching động dựa trên entropy, xử lý byte hiệu quả.
                *   2412.08635 | Đề xuất LatentLM thống nhất mô hình hóa dữ liệu rời rạc và liên tục bằng Causal Transformer và next-token diffusion.
            *   **Position Embeddings:** 2412.17739 | Đề xuất Fourier Position Embedding (FoPE) để cải thiện tổng quát hóa độ dài bằng cách mô hình hóa đa tần số và xử lý tần số huấn luyện không đủ.
        *   **Training & Fine-tuning Strategies**
            *   **Data-Efficient Pre-training & Continued Pre-training:**
                *   2412.17743 | Trình bày quy trình huấn luyện YuLan-Mini 2.42B hiệu quả về dữ liệu với lập lịch dữ liệu chi tiết và tối ưu hóa ổn định.
                *   2412.04862 | Mô tả các chiến lược huấn luyện trước, khử nhiễm dữ liệu, tạo dữ liệu SFT và tối ưu hóa ưu tiên cho dòng EXAONE 3.5.
            *   **Synthetic Data Generation & Augmentation:**
                *   2412.08905 | Phát triển các kỹ thuật tạo dữ liệu tổng hợp mới (đảo ngược hướng dẫn, Q&A từ chuỗi suy luận) và hậu huấn luyện cho phi-4.
                *   2412.14689 | Đề xuất ToEdit, phương pháp chỉnh sửa cấp độ token để tạo dữ liệu bán tổng hợp, ngăn chặn suy sụp mô hình.
                *   2412.03679 | Giới thiệu AGORA BENCH để đánh giá khả năng sinh dữ liệu của LLM và metric PGR.
                *   2412.16926 | Đề xuất quy trình tăng cường dữ liệu hai bước (sinh và lọc) cho ICL với LCLM khi dữ liệu hạn chế.
            *   **Alignment & Preference Optimization:**
                *   2411.19943 | Đề xuất cDPO, cải tiến DPO bằng cách tận dụng "critical tokens" trong suy luận toán học.
                *   2412.01981 | Đề xuất phương pháp thu được PRM ngầm định từ ORM chỉ với nhãn cấp độ phản hồi, dựa trên tham số hóa phần thưởng.
                *   2412.16145 | Đề xuất OREO, thuật toán RL ngoại tuyến để cải thiện suy luận đa bước bằng tối ưu hóa Soft Bellman Equation.
            *   **Instruction Following & Robustness:**
                *   2412.14922 | Giới thiệu ROBUST FT, framework SFT mạnh mẽ với dữ liệu phản hồi nhiễu, bao gồm phát hiện và khử nhiễu.
                *   2412.11231 | Đề xuất IC-IFD, thước đo đánh giá hiệu quả dữ liệu hướng dẫn tích hợp độ phức tạp của hướng dẫn.
            *   **Multi-Agent LLM Training:** 2412.01928 | Đề xuất MALT, chiến lược hậu huấn luyện hợp tác ba LLM chuyên biệt (Generator, Verifier, Refiner) để cải thiện suy luận.
        *   **Reasoning in LLMs**
            *   **Mathematical & Code Reasoning:**
                *   2412.06559 | Giới thiệu PROCESS BENCH để xác định lỗi trong suy luận toán học và phân tích xu hướng lỗi của LLM.
                *   2412.00154 | Mô tả nỗ lực tái tạo o1 (O1-CODER) với TCG, MCTS cho dữ liệu suy luận mã giả, và self-play+RL.
            *   **Latent & Compressed Reasoning:**
                *   2412.06769 | Đề xuất Coconut (Chain of Continuous Thought) cho phép LLM suy luận trong không gian tiềm ẩn liên tục.
                *   2412.13171 | Đề xuất Compressed Chain-of-Thought (CCOT) tạo token chiêm nghiệm liên tục, nén chuỗi lý luận.
            *   **Self-Improvement & Automated Reasoning Patterns:**
                *   2412.17256 | Đề xuất B-STaR, khung lý luận tự học tự động điều chỉnh cấu hình để cân bằng khám phá-khai thác.
                *   2411.18478 | Đề xuất HiAR-ICL, tự động khám phá và sử dụng "thought cards" (mẫu suy luận cấp cao) cho ICL.
            *   **Efficient Reasoning:** 2412.18547 | Đề xuất TALE, framework điều chỉnh động số token suy luận dựa trên độ phức tạp bài toán.
        *   **LLM Efficiency & Serving**
            *   **Optimizers:**
                *   2412.11768 | Đề xuất SGD-SaI, tối ưu hóa SGD với điều chỉnh tỷ lệ học theo g-SNR tại khởi tạo.
                *   2412.05270 | Đề xuất APOLLO, optimizer tiết kiệm bộ nhớ xấp xỉ tỷ lệ điều chỉnh gradient qua không gian phụ trợ hạng thấp.
            *   **Context Compression & KV Cache:**
                *   2412.17483 | Đề xuất chiến lược mã hóa tự động chi tiết và ước lượng tầm quan trọng token theo đoạn để cải thiện nén ngữ cảnh Fine KV.
                *   2412.17747 | Đề xuất coprocessor hoạt động trên KV-cache của LLM đóng băng, tạo embedding tiềm ẩn để tăng cường cache.
            *   **Inference Serving:** 2412.20993 | Đề xuất `certaindex` (proxy đo tiến độ suy luận) và `Dynasor` (hệ thống phục vụ LLM thích ứng) để phân bổ tài nguyên động.
        *   **LLM Safety & Evaluation**
            *   **Safety Alignment:** 2412.16720 | Mô tả dòng mô hình o1, tập trung vào an toàn qua deliberative alignment và Instruction Hierarchy.
            *   **Evaluation Metrics & Benchmarks (General):**
                *   2412.13147 | Đề xuất G-Pass@k, độ đo đánh giá đồng thời năng lực giải quyết vấn đề và tính ổn định của LLM.
                *   2412.21187 | Đề xuất metric hiệu quả đầu ra (ξO) và quy trình (ξP) để đánh giá "overthinking" trong LLM.
    *   **Multimodal Models (LMMs/VLMs/Video/Audio/3D)**
        *   **Vision-Language Models (VLMs) - General**
            *   **Scaling & Architecture:**
                *   2412.05271 | InternVL 2.5: Nhân rộng bộ mã hóa thị giác, LLM và cải tiến xử lý dữ liệu động độ phân giải cao.
                *   2412.03555 | PaliGemma 2: Nâng cấp lên Gemma 2, tinh chỉnh chiến lược huấn luyện đa giai đoạn và nghiên cứu ảnh hưởng của độ phân giải/kích thước mô hình.
                *   2412.04424 | Florence-VL: Khai thác Florence-2 để trích xuất đặc trưng thị giác đa dạng (chiều sâu/rộng) và hợp nhất bằng DBFusion.
                *   2412.04468 | NVILA: Kiến trúc "scale-then-compress" với Dynamic-S2, VEP, temporal averaging và tỉa tót dữ liệu DeltaLoss.
                *   2412.09604 | SynerGen-VL: Cơ chế gập/bung token và tiền huấn luyện căn chỉnh lũy tiến dựa trên chuyên gia thị giác cho MLLM hợp nhất.
            *   **Data Strategies & Augmentation (VLMs):**
                *   2412.18525 | Đề xuất "Explanatory Instructions" và bộ dữ liệu DECVT để VLM khái quát hóa zero-shot tốt hơn.
                *   2412.05237 | MAmmoTH-VL-Instruct: Quy trình tạo dữ liệu instruction-tuning đa phương thức CoT quy mô lớn bằng mô hình nguồn mở.
                *   2411.19930 | Quy trình sinh-sau-đó-lọc dữ liệu hướng dẫn trực quan chuyên biệt theo miền bằng mô hình nguồn mở.
                *   2412.08443 | POINTS1.5: Cải tiến lọc dữ liệu và khuôn mẫu trò chuyện cho huấn luyện hướng dẫn trực quan.
            *   **Efficiency & Token Reduction (VLMs):** 2412.04467 | VisionZip: Giảm token thị giác bằng lựa chọn token chủ đạo và hợp nhất token ngữ cảnh, có thể training-free.
            *   **Geometric Perception:** 2412.08737 | Giới thiệu benchmark Geoperception và họ mô hình Euclid tối ưu cho nhận thức hình học cấp thấp bằng dữ liệu tổng hợp và data curriculum.
            *   **Multilingual VLMs:** 2412.07112 | Xây dựng mô hình VLM đa ngôn ngữ Maya và bộ dữ liệu tiền huấn luyện đa ngôn ngữ (8 ngôn ngữ) đã lọc độc tính.
        *   **Video-LMMs & Video Understanding/Generation**
            *   **Video-LMM Design & Training:** 2412.10360 | Nghiên cứu Scaling Consistency, ApolloBench, và ưu thế của FPS sampling cho video-LMM.
            *   **Text-to-Video (T2V) & Image-to-Video (I2V) Generation:**
                *   2412.07730 | STIV: Hợp nhất T2V và TI2V trong một DiT với frame replacement, JIT-CFG và image condition dropout.
                *   2412.04814 | LIFT: Framework tinh chỉnh T2V theo phản hồi con người, gồm LIFT-HRA (dataset) và LIFT-CRITIC (reward model).
                *   2412.00131 | Kling: Mô hình sinh video Kling với WF-VAE, Skiparse Denoiser, 3D RoPE và các chiến lược huấn luyện/dữ liệu tiên tiến.
            *   **Specialized Video Generation & Editing:**
                *   2412.02259 | VGoT: Tạo video đa cảnh tự động từ một câu lệnh, với mô hình hóa cốt truyện động và duy trì nhất quán nhân vật/chuyển cảnh.
                *   2412.14173 | AniDoc: Tô màu video phác thảo hoạt hình với dẫn dắt bởi tương ứng, xử lý phác thảo nhị phân và huấn luyện phác thảo thưa.
                *   2412.03552 | Imagine360: Tạo video 360 từ video phối cảnh với kiến trúc hai nhánh, mặt nạ đối cực và thiết kế nhận biết độ cao.
            *   **Video Depth Estimation:** 2411.19189 | RollingDepth: Chuyển đổi LDM ảnh đơn (Marigold) thành mô hình ước tính độ sâu video đa khung với suy luận "rolling" và căn chỉnh toàn cục.
        *   **Omni-modal & Streaming Multimodal Systems**
            *   2412.09596 | IXC2.5-OmniLive: Hệ thống tương tác đa phương thức luồng dài hạn với kiến trúc ba mô-đun (nhận thức, bộ nhớ dài hạn, lý luận).
            *   2412.09501 | Lyra: MLLM tập trung giọng nói với Latent Cross-Modality Regularizer, Multi-Modality LoRA, Latent Multi-Modality Extractor và xử lý giọng nói dài.
        *   **Image Generation & Editing (Non-video context)**
            *   **Efficient & Controllable Image Generation:**
                *   2412.01819 | SWITTI: Transformer non-causal sinh ảnh theo scale, vô hiệu hóa CFG ở scale cao, giải quyết bất ổn định huấn luyện.
                *   2412.09619 | SnapGen: Kiến trúc UNet và Decoder hiệu quả, chưng cất kiến thức đa mức độ với timestep-aware scaling và chưng cất bước đối kháng cho sinh ảnh di động.
                *   2412.03895 | NoiseRefine: Học ánh xạ nhiễu sang "không gian nhiễu không cần hướng dẫn" bằng Multistep Score Distillation.
                *   2412.08580 | SDXL-SG: Mô hình nền tảng tích hợp bộ mã hóa đồ thị cảnh (GNN) để sinh ảnh phức tạp từ SG, cùng bộ dữ liệu LAION-SG.
                *   2412.07589 | DiffSensei: Tạo truyện tranh manga tùy chỉnh đa nhân vật với MLLM làm bộ điều hợp đặc trưng và kiểm soát bố cục, cùng bộ dữ liệu MangaZero.
                *   2412.08486 | Leffa: Hàm mất mát regularisation học trường dòng trong self-attention để giảm méo mó chi tiết trong sinh ảnh người.
            *   **In-Context Image Generation/Editing:** 2412.01824 | X-Prompt: Nén ví dụ trong ngữ cảnh bằng X-Prompt Tokens, tăng cường tác vụ và Chỉnh sửa Ảnh Tăng cường Truy xuất (RAIE).
            *   **Interactive Image Editing:**
                *   2412.10316 | BrushEdit: Chỉnh sửa ảnh dựa trên inpainting và hướng dẫn ngôn ngữ, tương tác đa vòng với agent hợp tác và mô hình inpainting "all-in-one".
                *   2412.04301 | SwiftEdit: Framework nghịch đảo một bước cho mô hình khuếch tán một bước, và ARaM để chỉnh sửa cục bộ, cho phép chỉnh sửa ảnh "tức thời".
            *   **Unified Image Generation & Editing Frameworks:** 2412.07774 | UniReal: Hợp nhất tạo và chỉnh sửa ảnh như tạo video không liên tục, với embedding chỉ mục ảnh và prompting phân cấp.
            *   **Image Tokenization for Unified Models:** 2412.03069 | TokenFlow: Kiến trúc dual-codebook (ngữ nghĩa, pixel) và ánh xạ chia sẻ để hợp nhất học biểu diễn cho hiểu và sinh đa phương thức.
            *   **Image Watermarking:** 2412.04653 | WIND: Sử dụng nhiễu khởi tạo làm thủy vân, nhúng thông tin nhóm bằng mẫu Fourier để tăng tốc phát hiện.
            *   **Model Quantization (Image Gen):** 2412.18653 | Lượng tử hóa FLUX.1-dev xuống 1.58-bit (PTQ) không cần dữ liệu ảnh, với kernel tùy chỉnh.
        *   **3D Generation & Understanding**
            *   **3D Asset Generation:** 2412.01506 | TRELLIS: Biểu diễn SLAT thống nhất (lưới 3D thưa + đặc trưng đa khung nhìn) và quy trình sinh hai giai đoạn cho nhiều định dạng 3D.
            *   **3D Scene Graph Representation:** 2412.18450 | 3DGraphLLM: Tạo biểu diễn đồ thị cảnh 3D học được cho LLM, ánh xạ quan hệ ngữ nghĩa vào không gian embedding token.
            *   **3D to 4D Animation:** 2412.20422 | 3to4D: Tạo hoạt ảnh 4D cho đối tượng 3D đầu vào bằng I2V diffusion và SDS, với chiến lược lấy mẫu điểm nhìn tăng dần và SDS có mặt nạ.
            *   **Interactive 3D Environment Generation:** 2412.09624 | GenEx: Sinh và khám phá thế giới 3D nhất quán từ một ảnh RGB, với khởi tạo panorama và chuyển đổi thế giới động.
            *   **3D LiDAR Scene Completion:** 2412.03515 | ScoreLiDAR: Chưng cất mô hình khuếch tán hoàn thiện cảnh LiDAR 3D, với Structural Loss mới.
        *   **Audio-driven Generation:** 2412.01064 | FLOAT: Tạo chân dung nói chuyện dựa trên âm thanh sử dụng flow matching trong không gian ẩn chuyển động, với Flow Matching Transformer (FMT) và điều khiển cảm xúc.
        *   **Multimodal Reasoning & Alignment**
            *   **General Multimodal Reasoning:**
                *   2412.14835 | AR-MCTS: Kết hợp truy vấn chủ động trong MCTS và PRM đa phương thức tự động căn chỉnh để cải thiện suy luận MLLM.
                *   2412.17451 | M-STAR: Huấn luyện tự tiến hóa liên tục, PRM đa phương thức, và điều chỉnh nhiệt độ động cho suy luận đa phương thức.
                *   2412.18319 | CoMCTS: Tìm kiếm cây Monte Carlo tập thể cho MLLM, tích hợp học tập tập thể để tìm đường suy luận hiệu quả và phản ánh.
            *   **Medical Multimodal Reasoning:**
                *   2412.18925 | Phát triển khả năng suy luận y khoa phức tạp cho LLM qua huấn luyện hai giai đoạn (SFT với quỹ đạo tìm kiếm, RL với trình xác minh y khoa).
                *   2412.20070 | Nghiên cứu Compositional Generalization (CG) trong MLLM y tế sử dụng bộ dữ liệu Med-MAT và MAT-Triplet.
                *   2412.07769 | BiMediX2: LMM y sinh song ngữ (Anh-Ả Rập) với kiến trúc hợp nhất và bộ dữ liệu hướng dẫn đa phương thức BiMed-V.
            *   **Efficient Multimodal Inference:** 2412.03248 | AIM: Suy luận thích ứng training-free cho MLLM bằng gộp và cắt tỉa visual token lặp đi lặp lại.
        *   **Multimodal Evaluation & Benchmarking**
            *   **General Generative Model Evaluation:** 2412.09645 | Evaluation Agent: Framework đánh giá động và tương tác cho mô hình sinh trực quan sử dụng agent dựa trên LLM.
            *   **Long-Context Multimodal Evaluation:** 2412.15204 | LongBench v2: Benchmark đánh giá hiểu và suy luận sâu trên văn bản dài đa tác vụ (8k-2M từ).
            *   **Personalized LMM Evaluation:** 2412.12606 | MDI-Benchmark: Đánh giá khả năng cá nhân hóa của LMM theo độ phức tạp câu hỏi và nhóm tuổi người dùng.
            *   **RAG Evaluation (Domain-Specific):** 2412.13018 | OmniEval: Benchmark RAG tự động và đa chiều cho lĩnh vực tài chính, với LLM evaluator được tinh chỉnh.
            *   **Multimodal Retrieval Data:** 2412.14475 | MegaPairs: Quy trình tổng hợp dữ liệu (cặp ảnh tương quan, hướng dẫn truy vấn) cho truy xuất đa phương thức.
    *   **AI Agents**
        *   **GUI Agents:**
            *   2412.04454 | AGUVIS: Framework thuần thị giác cho tác tử GUI tự hành, với quy trình xử lý dữ liệu THE AGUVIS COLLECTION và huấn luyện hai giai đoạn.
            *   2412.09605 | AgentTrek: Quy trình tự động tổng hợp quỹ đạo tác nhân GUI chất lượng cao từ hướng dẫn web.
            *   2412.13501 | Khảo sát về tác nhân GUI, phân loại kiến trúc, benchmark và thách thức.
        *   **Vision-Language-Action (VLA) Models:** 2411.19309 | GRAPE: Tối ưu hóa Sở thích theo Quỹ đạo (TPO) và Tạo Sở thích Hướng dẫn bằng Chi phí (GCPG) cho VLA.
        *   **Agent Benchmarking (Workplace):** 2412.14161 | TheAgentCompany: Benchmark đánh giá AI agent trên tác vụ công việc thực tế trong môi trường công ty phần mềm mô phỏng.
    *   **Robotics & Embodied AI**
        *   **Robot Error Detection & Monitoring:** 2412.04455 | Code-as-Monitor (CaM): Framework phát hiện lỗi robot bằng cách diễn giải thành bài toán thỏa mãn ràng buộc không gian-thời gian và giám sát bằng mã do VLM tạo.
    *   **Foundational AI Concepts & Surveys**
        *   **Reinforcement Learning Memory:** 2412.06531 | Chính thức hóa định nghĩa các loại bộ nhớ của agent trong RL và phương pháp luận đánh giá.
        *   **Next Token Prediction in Multimodal Learning (Survey):** 2412.18619 | Hệ thống hóa kiến thức học đa phương tiện qua lăng kính Dự đoán Token Tiếp theo (NTP).
    *   **Other (Specific Model Components/Techniques not easily fitting above)**
        *   **Autoregressive Model Distillation/Acceleration:**
            *   2412.15119 | PAR: Sinh song song phi cục bộ cho mô hình tự hồi quy trực quan, dựa trên sinh tuần tự token khởi tạo và sinh song song token liên vùng.
            *   2412.17153 | Distilled Decoding (DD): Chưng cất mô hình AR bằng Flow Matching để lấy mẫu trong một hoặc vài bước.
            *   2412.07720 | ACD IT: Transformer khuếch tán có điều kiện tự hồi quy theo khối, kết hợp mô hình AR và khuếch tán, với SCAM và RoPE-ND.
        *   **Code LLM Evaluation & Data:** 2412.05210 | CodeArena (benchmark đánh giá CodeLLM theo sở thích con người) và SynCode-Instruct (kho ngữ liệu hướng dẫn mã hóa tổng hợp).
    ```

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                                           |
    | :--- | :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
    | 1    | 2412.00131 | Video Generation, Skiparse Attention, WF-VAE, Kling  | Kiến trúc sinh video Kling với WF-VAE hiệu quả và Skiparse Denoiser, đạt chất lượng SOTA trên nhiều độ phân giải và thời lượng.             | Đặt ra tiêu chuẩn mới cho sinh video chất lượng cao, mã nguồn mở, thúc đẩy nghiên cứu về VAE hiệu quả và cơ chế attention thưa cho video.                               |
    | 2    | 2412.09871 | Byte-level LLM, Entropy Patching, BLT                | Byte Latent Transformer (BLT) với patching động dựa trên entropy, xử lý byte thô hiệu quả, cạnh tranh với Llama 3 8B với ít FLOPs hơn. | Mở ra hướng mới cho LLM không phụ thuộc tokenizer, tiềm năng xử lý ngôn ngữ đa dạng và dữ liệu nhiễu tốt hơn.                                                            |
    | 3    | 2412.01506 | 3D Generation, SLAT, Sparse VAE, Rectified Flow      | Biểu diễn SLAT thống nhất và quy trình sinh 3D hai giai đoạn (TRELLIS) cho nhiều định dạng 3D chất lượng cao, không cần pre-fitting.      | Cải thiện đáng kể chất lượng và tính linh hoạt của việc tạo tài sản 3D, có thể ứng dụng rộng rãi trong game, VR/AR.                                                      |
    | 4    | 2412.08635 | Unified Modeling, LatentLM, Next-token Diffusion     | LatentLM thống nhất mô hình hóa dữ liệu rời rạc và liên tục bằng Causal Transformer, σ-VAE và next-token diffusion.                     | Cung cấp một framework mạnh mẽ và linh hoạt cho các mô hình đa phương thức thế hệ tiếp theo, có khả năng xử lý nhiều loại dữ liệu.                                      |
    | 5    | 2412.01819 | Efficient Image Gen, Non-Causal Transformer, SWITTI  | SWITTI: Transformer non-causal sinh ảnh theo scale, vô hiệu hóa CFG ở scale cao, tăng tốc và giảm bộ nhớ đáng kể.                       | Đẩy mạnh hướng phát triển mô hình sinh ảnh tự hồi quy hiệu quả, cạnh tranh với diffusion model về tốc độ.                                                              |
    | 6    | 2412.03069 | Unified Tokenization, Dual Codebook, TokenFlow       | TokenFlow: Kiến trúc dual-codebook (ngữ nghĩa, pixel) và ánh xạ chia sẻ để hợp nhất học biểu diễn cho hiểu và sinh đa phương thức.       | Giải quyết vấn đề đánh đổi giữa hiểu ngữ nghĩa và tái tạo chi tiết trong VLM, tạo ra token hình ảnh linh hoạt hơn.                                                       |
    | 7    | 2412.11919 | Unified RAG, Hierarchical FM-Index, RetroLLM         | RetroLLM: Tích hợp truy xuất (sinh bằng chứng có ràng buộc FM-Index) và sinh văn bản vào một LLM duy nhất, giải quyết false pruning. | Thay đổi cách tiếp cận RAG truyền thống, cho phép tối ưu hóa chung và truy xuất chi tiết hơn.                                                                          |
    | 8    | 2412.07720 | Hybrid AR-Diffusion, ACD IT, SCAM, RoPE-ND           | ACD IT: Kết hợp tự hồi quy theo khối và khuếch tán có điều kiện để sinh thông tin hình ảnh liên tục, không cần token hóa rời rạc.      | Mở ra hướng mới cho mô hình sinh đa phương thức, kết hợp ưu điểm của AR (hiệu quả, KV-cache) và Diffusion (chất lượng).                                                  |
    | 9    | 2412.04455 | Robot Error Detection, Code-as-Monitor, CaM          | CaM: Diễn giải phát hiện lỗi robot thành bài toán thỏa mãn ràng buộc không gian-thời gian, giám sát bằng mã do VLM tạo.                 | Cung cấp một framework phát hiện lỗi robot chính xác, hiệu quả và có khả năng giải thích, kết hợp VLM và hình học.                                                        |
    | 10   | 2412.16145 | Offline RL for LLMs, Reasoning, OREO                 | OREO: Thuật toán RL ngoại tuyến tối ưu hóa Soft Bellman Equation để cải thiện suy luận đa bước của LLM từ dữ liệu không cặp ưu tiên.    | Giải quyết hạn chế của DPO (cần dữ liệu cặp) và SFT (chỉ học từ quỹ đạo đúng) cho các tác vụ suy luận phức tạp.                                                            |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2412.13663 – Quy trình unpadding tích hợp Flash Attention, kiến trúc "Deep & Narrow" tối ưu GPU, quy trình mở rộng ngữ cảnh đa giai đoạn cho encoder, alternating attention tùy chỉnh cho encoder – Suy nghĩ:** Các cải tiến thực tế và hiệu quả cho encoder-only models, đặc biệt là unpadding và tối ưu kiến trúc cho phần cứng cụ thể.
    *   **2412.10360 – Scaling Consistency (hiện tượng chuyển giao quyết định thiết kế từ mô hình nhỏ sang lớn cho video-LMM), ApolloBench (quy trình giám tuyển benchmark video hiệu quả), FPS Sampling (ưu thế so với uniform sampling khi huấn luyện video-LMM) – Suy nghĩ:** Scaling Consistency là một phát hiện quan trọng, có thể tiết kiệm tài nguyên đáng kể. ApolloBench và FPS sampling là những đóng góp thực tiễn cho nghiên cứu video-LMM.
    *   **2412.08905 – Kỹ thuật tạo cặp DPO dựa trên pivotal token search, phương pháp tạo dữ liệu tổng hợp bằng instruction reversal, quy trình tạo Q&A từ chuzości suy luận, kỹ thuật lọc câu hỏi plurality-based – Suy nghĩ:** Các phương pháp tạo dữ liệu tổng hợp sáng tạo, đặc biệt là instruction reversal và Q&A từ chuỗi suy luận, có tiềm năng lớn.
    *   **2412.02687 – Proper Guidance - SwiftBrush (PG-SB) (thang đo hướng dẫn ngẫu nhiên cho VSD), Negative-Away Steer Attention (NASA) (tích hợp prompt phủ định cho mô hình khuếch tán một bước) – Suy nghĩ:** PG-SB giải quyết vấn đề ổn định VSD một cách thông minh. NASA là một đóng góp mới lạ và hữu ích cho mô hình một bước.
    *   **2412.04467 – VisionZip (lựa chọn token chủ đạo dựa trên attention của vision encoder và hợp nhất token ngữ cảnh dựa trên tương đồng key) – Suy nghĩ:** Phương pháp giảm token thị giác trực quan, dễ triển khai, có chế độ training-free là một lợi thế.
    *   **2412.18925 – Quy trình tạo "vấn đề y khoa có thể kiểm chứng", "trình xác minh y khoa" dựa trên LLM, phương pháp huấn luyện hai giai đoạn (SFT với quỹ đạo tìm kiếm chiến lược, RL với trình xác minh) cho suy luận y khoa – Suy nghĩ:** Phương pháp luận có hệ thống để cải thiện suy luận LLM trong miền y khoa, đặc biệt là việc tạo dữ liệu SFT từ quỹ đạo tìm kiếm.
    *   **2412.09871 – Byte Latent Transformer (BLT) (kiến trúc LLM byte-level với Local Encoder/Decoder và Latent Global Transformer), Entropy Patching (patching động dựa trên entropy), Encoder Hash n-gram Embeddings, Encoder Multi-Headed Cross-Attention – Suy nghĩ:** Kiến trúc byte-level rất sáng tạo, entropy patching là một ý tưởng mạnh mẽ để xử lý byte hiệu quả.
    *   **2412.09596 – Kiến trúc hệ thống ba mô-đun (Streaming Perception, Multi-modal Long Memory, Reasoning) tách rời cho tương tác luồng dài hạn, cơ chế nén và truy xuất bộ nhớ trong Multi-modal Long Memory Module, quy trình Instruction Prediction – Suy nghĩ:** Kiến trúc hệ thống tham vọng, giải quyết thách thức tương tác AI dài hạn với luồng video/audio.
    *   **2412.09624 – Quy trình chuyển đổi thế giới (world transition) dạng sinh video toàn cảnh động có điều kiện và Spherical-Consistency Learning (SCL) cho khám phá thế giới 3D từ ảnh đơn – Suy nghĩ:** Hướng đi thú vị cho việc tạo môi trường 3D tương tác từ đầu vào tối thiểu.
    *   **2412.13147 – Độ đo G-Pass@k, G-Pass@kτ, mG-Pass@k (đánh giá đồng thời năng lực giải quyết vấn đề và tính ổn định của LLM) – Suy nghĩ:** Các độ đo cần thiết để đánh giá sâu hơn về tính nhất quán và độ "thành thạo" của LLM.
    *   **2412.14922 – ROBUST FT (hệ thống đa chuyên gia cộng tác phát hiện nhiễu, chiến lược khử nhiễu tăng cường ngữ cảnh, cơ chế lựa chọn dữ liệu dựa trên entropy) – Suy nghĩ:** Framework toàn diện và thực tế để xử lý nhiễu trong SFT.
    *   **2412.06769 – Coconut (Chain of Continuous Thought) (trạng thái ẩn cuối làm embedding đầu vào cho bước suy luận tiềm ẩn tiếp theo, token <bot>/<eot> để chuyển chế độ) – Suy nghĩ:** Phương pháp độc đáo cho phép LLM suy luận trong không gian tiềm ẩn, có tiềm năng vượt qua hạn chế của ngôn ngữ rời rạc.
    *   **2412.18653 – Lượng tử hóa FLUX.1-dev xuống 1.58-bit (PTQ) không cần dữ liệu ảnh, chỉ dựa vào tự giám sát từ mô hình gốc, kernel tính toán linear tùy chỉnh cho trọng số 1.58-bit – Suy nghĩ:** Đạt mức lượng tử hóa rất thấp cho T2I SOTA mà không cần dữ liệu ảnh là một đột phá về hiệu quả.
    *   **2412.18525 – "Explanatory Instructions" (định nghĩa mục tiêu nhiệm vụ thị giác qua mô tả ngôn ngữ chi tiết về phép biến đổi), Dataset of Explanatory CV Tasks (DECVT) – Suy nghĩ:** Ý tưởng đột phá để cải thiện khả năng hiểu và khái quát hóa nhiệm vụ của VLM.
    *   **2412.01506 – Biểu diễn SLAT (Structured LATent) thống nhất (lưới 3D thưa + đặc trưng đa khung nhìn dày đặc), Sparse VAE với 3D shifted window attention (3D-SW-MHA), quy trình sinh SLAT hai giai đoạn với rectified flow transformers và sparse convolution packing – Suy nghĩ:** Phương pháp tạo 3D toàn diện, SLAT là một biểu diễn thông minh.
    *   **2412.07730 – Thay thế frame điều kiện (Frame Replacement) trong DiT, Joint Image-Text Classifier-Free Guidance (JIT-CFG), Dropout điều kiện hình ảnh – Suy nghĩ:** Các kỹ thuật đơn giản nhưng hiệu quả để hợp nhất T2V và TI2V trong một mô hình.
    *   **2412.14835 – Truy vấn chủ động (Active Retrieval) tích hợp vào MCTS expansion, PRM đa phương thức tự động căn chỉnh qua chương trình học hai giai đoạn (step-wise DPO, point-wise fine-tuning) – Suy nghĩ:** Tích hợp truy vấn chủ động vào MCTS là một ý tưởng sáng tạo cho suy luận MLLM.
    *   **2412.01824 – Cơ chế nén ví dụ trong ngữ cảnh X-Prompt Tokens và attention masking chuyên biệt, quy trình tăng cường tác vụ (chuyển đổi tạo ảnh thành dự đoán text, đảo ngược tác vụ), Chỉnh sửa Ảnh Tăng cường Truy xuất (RAIE) – Suy nghĩ:** Các giải pháp kỹ thuật đáng chú ý để cải thiện học trong ngữ cảnh cho tạo ảnh VLM.
    *   **2411.19943 – "Critical tokens" trong suy luận toán học, phương pháp "contrastive estimation" để xác định critical tokens, cDPO (tận dụng critical tokens để tăng cường DPO) – Suy nghĩ:** Khái niệm "critical tokens" và cách xác định/sử dụng chúng là một hướng mới để cải thiện alignment trong suy luận.
    *   **2412.04424 – Khai thác Florence-2 để trích xuất đặc trưng thị giác đa dạng (chiều sâu/rộng), kiến trúc hợp nhất đặc trưng Depth-Breadth Fusion (DBFusion) bằng tích hợp theo kênh – Suy nghĩ:** Cách tiếp cận hợp lý để làm giàu biểu diễn thị giác cho MLLM.
    *   **2412.02259 – Mô hình hóa cốt truyện động với tự xác thực, Lan truyền nhận dạng xuyên cảnh nhận biết danh tính (IPP tokens), Cơ chế chuyển tiếp ẩn liền kề với đặt lại nhiễu nhận biết ranh giới – Suy nghĩ:** Framework sáng tạo giải quyết tạo video đa cảnh tự động.
    *   **2412.14173 – Tô màu dẫn dắt bởi tương ứng (Correspondence-guided Colorization) sử dụng bản đồ điểm, Nhị phân hóa phác thảo và Tăng cường dữ liệu nền, Huấn luyện phác thảo thưa hai giai đoạn – Suy nghĩ:** Giải pháp toàn diện cho tô màu video phác thảo, giải quyết các thách thức thực tế.
    *   **2412.07760 – Module Multi-View Synchronization (MVS) plug-and-play với cross-view self-attention, quy trình thu thập dữ liệu và chiến lược huấn luyện hỗn hợp cho video đa camera – Suy nghĩ:** Module MVS là ý tưởng hợp lý để đồng bộ hóa video đa góc nhìn.
    *   **2412.14475 – Quy trình tổng hợp dữ liệu MegaPairs (khai thác cặp ảnh tương quan bằng ba loại mô hình tương tự, tạo hướng dẫn truy vấn hai giai đoạn bằng MLLM và LLM) – Suy nghĩ:** Phương pháp tổng hợp dữ liệu chất lượng cao và đa dạng cho truy xuất đa phương thức.
    *   **2412.15119 – Sinh song song phi cục bộ (non-local parallel generation) cho mô hình AR trực quan (sinh tuần tự token khởi tạo vùng, sinh song song token liên vùng), cấu trúc chuỗi đầu vào đặc biệt, chú ý hai chiều theo nhóm với tự hồi quy toàn cục – Suy nghĩ:** Phương pháp hiệu quả để tăng tốc sinh AR trực quan mà không cần thay đổi lớn kiến trúc.
    *   **2412.14689 – ToEdit (Chỉnh sửa Cấp độ Token dựa trên xác suất từ mô hình tiên nghiệm để tạo dữ liệu bán tổng hợp) – Suy nghĩ:** Đóng góp kỹ thuật mới, có chứng minh lý thuyết, giải quyết suy sụp mô hình không lặp lại.
    *   **2412.05210 – CodeArena (benchmark đánh giá CodeLLM theo sở thích con người), SynCode-Instruct (quy trình tạo kho ngữ liệu hướng dẫn mã hóa tổng hợp quy mô lớn bằng LLM tạo câu hỏi mới và kiểm soát chất lượng tự động) – Suy nghĩ:** CodeArena giải quyết khoảng trống đánh giá CodeLLM. Quy trình tạo SynCode-Instruct rất tiềm năng.
    *   **2412.09501 – Latent Cross-Modality Regularizer (sử dụng DTW căn chỉnh token giọng nói-văn bản), Multi-Modality LoRA Pipeline, Latent Multi-Modality Extractor (chọn lọc token đa phương tiện theo khối dựa trên truy vấn văn bản), Quy trình tích hợp khả năng xử lý giọng nói dài – Suy nghĩ:** Các cơ chế mới và hợp lý cho MLLM tập trung giọng nói, đặc biệt là xử lý giọng nói dài.
    *   **2412.07589 – DiffSensei (MLLM làm bộ điều hợp đặc điểm nhân vật tương thích văn bản cho tạo manga), Cơ chế chèn chú ý chéo có mặt nạ (kiểm soát bố cục nhân vật), Kỹ thuật nhúng hội thoại (kiểm soát vị trí bong bóng thoại), Bộ dữ liệu MangaZero – Suy nghĩ:** Hướng tiếp cận mới cho tạo manga tùy chỉnh, MLLM adapter là ý tưởng sáng tạo.
    *   **2412.04814 – LIFT (quy trình tinh chỉnh T2V ba giai đoạn theo phản hồi con người), LIFT-HRA (bộ dữ liệu phản hồi người gồm điểm và lý do), LIFT-CRITIC (mô hình phần thưởng VLM dự đoán điểm và tạo giải thích), Reward-Weighted Learning (RWL) – Suy nghĩ:** Phương pháp luận tiên phong và toàn diện để tích hợp phản hồi chi tiết của con người vào tinh chỉnh T2V.
    *   **2412.17256 – Định lượng và theo dõi động lực khám phá/khai thác (Pass@K-S, Reward@K-S), "điểm cân bằng" (balance score) đánh giá hiệu quả dữ liệu, B-STaR (tự động điều chỉnh cấu hình để tối đa hóa điểm cân bằng) – Suy nghĩ:** Các chỉ số và cơ chế mới để làm sáng tỏ và tối ưu hóa quá trình tự học lặp đi lặp lại.
    *   **2411.19309 – Tối ưu hóa Sở thích theo Quỹ đạo (TPO), Tạo Sở thích Hướng dẫn bằng Chi phí (GCPG) (phân rã nhiệm vụ, đề xuất điểm khóa và hàm chi phí bằng LLM/VLM) cho VLA – Suy nghĩ:** Framework sáng tạo, GCPG tự động hóa việc tạo hàm chi phí là một điểm mạnh.
    *   **2412.18319 – Collective Monte Carlo Tree Search (CoMCTS) (mở rộng tập thể, mô phỏng và định vị lỗi tập thể), Tìm kiếm đường suy luận phản ánh, Collective Supervised Fine-Tuning (CoSFT) – Suy nghĩ:** Kết hợp MCTS với học tập tập thể là một ý tưởng tiềm năng cho suy luận MLLM.
    *   **2412.16145 – OREO (Offline REasoning Optimization) (tối ưu hóa Soft Bellman Equation để học chính sách và hàm giá trị cho suy luận LLM), các biến thể hàm mục tiêu OREO (token, step, response-level) – Suy nghĩ:** Phương pháp RL ngoại tuyến có cơ sở lý thuyết vững chắc, giải quyết hạn chế của DPO cho suy luận đa bước.
    *   **2412.18547 – TALE (Token-Budget-Aware LLM Reasoning), TALE-EP (ước tính ngân sách token bằng LLM zero-shot prompt), TALE-PT (nội bộ hóa nhận biết ngân sách qua hậu huấn luyện), Hiện tượng "Token Elasticity" – Suy nghĩ:** Giải quyết chi phí token của CoT một cách hợp lý, "Token Elasticity" là một phát hiện thú vị.
    *   **2412.08580 – Bộ mã hóa đồ thị cảnh phụ trợ (GNN) tích hợp vào SDXL (SDXL-SG), Bộ dữ liệu LAION-SG (chú thích SG tự động bằng GPT-4o) – Suy nghĩ:** Tích hợp GNN vào SDXL là một cách tiếp cận hợp lý để sinh ảnh phức tạp từ SG.
    *   **2412.01928 – MALT (Multi-Agent LLM Training) (huấn luyện hợp tác ba LLM chuyên biệt: Generator, Verifier, Refiner), Quy trình tạo dữ liệu dựa trên mở rộng cây tìm kiếm đa tác nhân và lặp giá trị để gán tín chỉ tự động – Suy nghĩ:** Phương pháp hậu huấn luyện đa tác nhân trực quan, cơ chế gán tín chỉ tự động là một đóng góp đáng chú ý.
    *   **2412.00154 – Máy Tạo Ca Kiểm Thử (TCG) huấn luyện hai giai đoạn (SFT, DPO), Tổng hợp dữ liệu mã nguồn tăng cường suy luận bằng MCTS (với hành động định nghĩa trước) – Suy nghĩ:** Khung thử nghiệm chi tiết để tái tạo khả năng suy luận System-2 cho lập trình.
    *   **2412.17451 – Huấn luyện tự tiến hóa liên tục (kế thừa trạng thái optimizer, scheduler), PRM đa phương thức đầu tiên, Điều chỉnh động nhiệt độ lấy mẫu – Suy nghĩ:** Các cải tiến có hệ thống cho huấn luyện tự tiến hóa trong suy luận đa phương thức.
    *   **2412.11768 – SGD-SaI (SGD với Momentum và điều chỉnh tỷ lệ học theo g-SNR tính tại khởi tạo) – Suy nghĩ:** Ý tưởng đơn giản, trực quan và có tính mới, thách thức các optimizer phức tạp.
    *   **2412.21187 – Metric hiệu quả đầu ra (ξO), Metric hiệu quả quy trình (ξP) (đánh giá đa dạng chiến lược suy luận bằng LLM phân cụm) – Suy nghĩ:** Các metric định lượng mới và cần thiết để đánh giá "overthinking" trong LLM.
    *   **2412.20422 – Chiến lược lấy mẫu điểm nhìn tăng dần, Hàm mất mát Score Distillation Sampling có mặt nạ (attention-masked SDS), Chiến lược lấy mẫu thời gian mới cho NeRF 4D (cố định frame đầu, lấy mẫu đều với nhiễu nhỏ) cho 3D to 4D animation – Suy nghĩ:** Các đóng góp kỹ thuật hợp lý để cải thiện tạo hoạt ảnh 4D từ đối tượng 3D.
    *   **2412.17739 – Fourier Position Embedding (FoPE) (mô hình hóa mỗi chiều embedding như Chuỗi Fourier, cắt bỏ tần số huấn luyện không đủ) – Suy nghĩ:** Đóng góp kỹ thuật đáng chú ý dựa trên phân tích lý thuyết sâu sắc về hạn chế của RoPE.
    *   **2412.13018 – Quy trình tạo dữ liệu đánh giá RAG tự động đa tác nhân (sử dụng GPT-4), Hệ thống chỉ số đánh giá RAG đa chiều với LLM evaluator được tinh chỉnh – Suy nghĩ:** Phương pháp luận rõ ràng để xây dựng benchmark RAG tài chính, LLM evaluator tinh chỉnh là cải tiến quan trọng.
    *   **2412.04455 – "Constraint elements" (biểu diễn hình học nhỏ gọn cho ràng buộc robot), ConSeg (mô hình phân đoạn đa mức độ chi tiết nhận biết ràng buộc), Quy trình lập trình trực quan nhận biết ràng buộc (VLM tạo "monitor code" từ visual prompts) – Suy nghĩ:** Hướng tiếp cận sáng tạo cho phát hiện lỗi robot, "constraint elements" và "monitor code" là những ý tưởng mạnh mẽ.
    *   **2412.04301 – Framework nghịch đảo một bước cho mô hình khuếch tán một bước (huấn luyện hai giai đoạn), Kỹ thuật điều chỉnh lại trọng số chú ý dựa trên mặt nạ (ARaM) – Suy nghĩ:** Giải pháp đột phá về tốc độ cho chỉnh sửa ảnh theo văn bản.
    *   **2411.19189 – LDM đa khung với cross-frame self-attention, Lược đồ suy luận "rolling" với kernel giãn nở, Quy trình căn chỉnh toàn cục dựa trên tối ưu hóa L1 mạnh mẽ, Tinh chỉnh tùy chọn dựa trên diffusion với làm mịn trung bình và chiến lược coarse-to-fine, Chiến lược huấn luyện với chuẩn hóa độ sâu nghịch đảo theo đoạn – Suy nghĩ:** Phương pháp tiếp cận sáng tạo và thực tế cho ước tính độ sâu video.
    *   **2412.20993 – `certaindex` (biến proxy đo tiến độ suy luận LLM dựa trên độ chắc chắn), `Dynasor` (hệ thống phục vụ suy luận LLM thích ứng sử dụng `certaindex`) – Suy nghĩ:** Ý tưởng trực quan và có tính ứng dụng cao để phân bổ tài nguyên động cho suy luận LLM.
    *   **2412.09604 – Cơ chế gập token (token folding) và đầu giải mã tự hồi quy nông để bung token (token unfolding) cho MLLM, Chiến lược tiền huấn luyện căn chỉnh lũy tiến dựa trên chuyên gia thị giác – Suy nghĩ:** Giải pháp kỹ thuật thông minh để MLLM xử lý ảnh độ phân giải cao hiệu quả.
    *   **2411.18478 – Năm hành động suy luận nguyên tử, Xây dựng "thought cards" qua MCTS với hàm thưởng tự nhất quán, Thước đo lựa chọn đường suy luận VOC-inspired, Khung đánh giá độ phức tạp nhận thức, Suy luận có hướng dẫn bằng "thought cards" – Suy nghĩ:** Hướng tiếp cận mới và có giá trị cho suy luận phức tạp của LLM bằng mẫu tư duy trừu tượng.
    *   **2412.18450 – Biểu diễn đồ thị cảnh 3D học được cho LLM (3DGraphLLM), Thuật toán tạo chuỗi token embedding đồ thị phẳng từ đồ thị con k-láng giềng – Suy nghĩ:** Phương pháp luận có tính mới, mã hóa tường minh quan hệ ngữ nghĩa 3D cho LLM.
    *   **2412.18153 – DepthLab (kiến trúc hai nhánh U-Net cho depth inpainting), Cơ chế hợp nhất đặc trưng theo từng lớp bằng self-attention, Chuẩn hóa độ sâu ngẫu nhiên, Mã hóa mặt nạ bằng VAE, Điều kiện bằng CLIP image encoder – Suy nghĩ:** Phương pháp mới và tiềm năng cho depth inpainting, kiến trúc hai nhánh và hợp nhất đặc trưng là đóng góp đáng chú ý.
    *   **2412.17153 – Distilled Decoding (DD) (chưng cất mô hình AR bằng Flow Matching cho từng bước sinh token, huấn luyện mạng học trò học ánh xạ từ trạng thái trung gian đến chuỗi cuối) – Suy nghĩ:** Phương pháp mới lạ để tăng tốc sinh mô hình AR, không cần dữ liệu huấn luyện gốc của thầy.
    *   **2412.11919 – Ràng buộc FM-Index phân cấp (sinh "clues" rồi sinh bằng chứng), Chiến lược giải mã có ràng buộc nhìn về phía trước (điều chỉnh logits dựa trên "future windows") cho RAG hợp nhất – Suy nghĩ:** Các đóng góp kỹ thuật cốt lõi giải quyết hiệu quả vấn đề "false pruning" trong sinh bằng chứng có ràng buộc.
    *   **2412.08486 – Hàm mất mát regularisation Leffa (học trường dòng trong self-attention để hướng truy vấn đích vào khóa tham chiếu chính xác) – Suy nghĩ:** Phương pháp trực quan và hiệu quả để giảm méo mó chi tiết trong sinh ảnh người có điều kiện.
    *   **2412.13171 – Compressed Chain-of-Thought (CCOT) (tạo token chiêm nghiệm liên tục, có nội dung, độ dài thay đổi), Cơ chế tạo token chiêm nghiệm tự hồi quy từ tầng trung gian, Quy trình huấn luyện CCOTφ và DECODEψ – Suy nghĩ:** Phương pháp có tính mới để cải thiện hiệu quả suy luận nhiều bước, đặc biệt là tạo token từ tầng trung gian.
    *   **2412.07720 – ACD IT (Transformer khuếch tán có điều kiện tự hồi quy theo khối), Skip-Causal Attention Mask (SCAM), RoPE-ND (mã hóa vị trí tương đối đa chiều) – Suy nghĩ:** Phương pháp lai ghép độc đáo giữa AR và khuếch tán, SCAM và RoPE-ND là những thiết kế mới.
    *   **2412.07774 – Embedding chỉ mục ảnh (liên kết token ảnh với thuật ngữ tham chiếu), Lược đồ prompting phân cấp (context prompts, image prompts) cho framework tạo/chỉnh sửa ảnh hợp nhất – Suy nghĩ:** Các cơ chế giúp tăng cường khả năng kiểm soát và giảm mơ hồ trong mô hình hợp nhất.
    *   **2412.04653 – WIND (Watermarking In Noise with a Directory) (nhúng thông tin nhóm của nhiễu khởi tạo vào nhiễu bằng mẫu Fourier để tăng tốc phát hiện thủy vân) – Suy nghĩ:** Giải pháp thông minh cho bài toán đánh đổi giữa bền vững và hiệu quả phát hiện thủy vân.
    *   **2412.01064 – Flow Matching Transformer (FMT) (điều kiện theo từng khung hình, self-attention có mặt nạ cho flow matching trong tạo chân dung nói), Incremental Classifier-Free Vector field (iCFV) – Suy nghĩ:** Áp dụng flow matching và FMT là hướng đi thông minh cho tạo video chân dung nói.
    *   **2412.09605 – AgentTrek (quy trình tự động tổng hợp quỹ đạo tác nhân GUI từ hướng dẫn web, gồm thu thập/lọc văn bản, chuyển đổi thành đặc tả nhiệm vụ, thực thi bằng VLM agent, đánh giá bằng VLM evaluator) – Suy nghĩ:** Giải pháp sáng tạo và thực tiễn cho khan hiếm dữ liệu huấn luyện tác nhân GUI.
    *   **2412.15213 – CrossFlow (học ánh xạ trực tiếp giữa các modality bằng flow matching, loại bỏ nhiễu đầu vào và điều kiện truyền thống), Variational Encoder (VE) chuẩn hóa latent nguồn, CFG với indicator, Contrastive loss cho Text VE – Suy nghĩ:** Hướng tiếp cận mới lạ cho sinh đa phương thức, loại bỏ nhiễu đầu vào và cross-attention là một đột phá.
    *   **2412.01253 – Cơ chế cân bằng tải EP và PEP cho MoE, Khối chú ý lai (SWA + full attention), Tái sử dụng bộ đệm KV chéo lớp, Phân đoạn expert chi tiết cân bằng – Suy nghĩ:** Các cải tiến kiến trúc và tối ưu hóa thực tế cho mô hình MoE ngữ cảnh dài.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Hiệu quả và Khả năng mở rộng của LLM/LMM:** Nhiều paper tập trung vào tối ưu hóa (2412.11768, 2412.05270, 2412.01253), nén ngữ cảnh (2412.17483), lượng tử hóa (2412.18653), và suy luận hiệu quả (2412.18547, 2412.20993, 2412.04467, 2412.03248).
        *   *Gap:* Vẫn còn nhu cầu lớn về các mô hình nhỏ gọn hơn, nhanh hơn, đặc biệt cho thiết bị di động và các ứng dụng thời gian thực, mà không hy sinh quá nhiều hiệu năng. Cân bằng giữa nén và duy trì thông tin chi tiết vẫn là thách thức.
        *   *Opportunity:* Nghiên cứu kiến trúc mới tối ưu cho phần cứng cụ thể, các kỹ thuật lượng tử hóa/tỉa thưa (pruning) sâu hơn, các phương pháp nén KV-cache linh hoạt hơn, và các thuật toán suy luận thích ứng thông minh hơn. Khám phá các mô hình "phi Transformer" hiệu quả.
    *   **Chất lượng và Sự đa dạng của Dữ liệu Huấn luyện:** Nhiều nỗ lực tạo dữ liệu tổng hợp (2412.08905, 2412.14689, 2412.03679, 2412.05237, 2411.19930, 2412.14475, 2412.05210, 2412.00927, 2412.09605), lọc dữ liệu (2412.14922, 2412.04862, 2412.08443, 2412.07112), và lập lịch dữ liệu (2412.17743).
        *   *Gap:* Nguy cơ suy sụp mô hình từ dữ liệu tổng hợp lặp đi lặp lại, đảm bảo tính đa dạng và tránh thiên kiến trong dữ liệu tổng hợp. Chi phí tạo và kiểm duyệt dữ liệu chất lượng cao vẫn lớn.
        *   *Opportunity:* Phát triển các phương pháp sinh dữ liệu tổng hợp tiên tiến hơn có khả năng duy trì "đuôi dài" của phân phối dữ liệu thực. Các kỹ thuật tự giám sát hoặc bán giám sát để lọc và cải thiện dữ liệu. Nghiên cứu về "data curriculum" tự động.
    *   **Suy luận Phức tạp và Đa bước:** Nhiều nghiên cứu về CoT, suy luận tiềm ẩn, đa tác nhân, RL cho suy luận (2411.19943, 2412.01981, 2412.16145, 2412.06769, 2412.13171, 2412.17256, 2411.18478, 2412.01928, 2412.00154, 2412.17451, 2412.14835, 2412.18319).
        *   *Gap:* Khả năng suy luận logic sâu, suy luận toán học và khoa học vẫn còn hạn chế. Khó khăn trong việc gán tín chỉ cho các bước suy luận dài. Tính nhất quán và khả năng kiểm chứng của các bước suy luận.
        *   *Opportunity:* Các framework suy luận mới kết hợp logic hình thức và học máy. Phát triển các mô hình phần thưởng quy trình (PRM) hiệu quả hơn và ít tốn kém hơn. Nghiên cứu về khả năng tự sửa lỗi và tự phản tư sâu hơn.
    *   **Hợp nhất Đa phương thức (Multimodal Unification):** Các mô hình như LatentLM (2412.08635), TokenFlow (2412.03069), ACD IT (2412.07720), UniReal (2412.07774), CrossFlow (2412.15213) đang hướng tới việc xử lý nhiều modality trong một framework.
        *   *Gap:* Căn chỉnh (alignment) sâu và có ý nghĩa giữa các modality khác nhau. Xử lý hiệu quả các luồng dữ liệu đa phương thức không đồng bộ và có độ dài thay đổi lớn. Khả năng khái quát hóa thành phần thực sự.
        *   *Opportunity:* Nghiên cứu các không gian biểu diễn chung (common embedding spaces) mạnh mẽ hơn. Các kiến trúc mới cho phép tương tác chéo phương tiện linh hoạt và hiệu quả. Các phương pháp học tự giám sát đa phương thức quy mô lớn.
    *   **Tác nhân AI (AI Agents):** Lĩnh vực đang phát triển nhanh với các benchmark và phương pháp mới (2412.04454, 2412.09605, 2411.19309, 2412.14161, 2412.04455).
        *   *Gap:* Khả năng lập kế hoạch dài hạn, thích ứng với môi trường động và bất ngờ, tương tác an toàn và hiệu quả với con người và các agent khác. Thiếu dữ liệu huấn luyện chất lượng cao cho các tác vụ phức tạp.
        *   *Opportunity:* Phát triển các mô hình thế giới (world models) tốt hơn cho agent. Các phương pháp RL hiệu quả hơn cho học chính sách trong môi trường phức tạp. Tự động hóa việc tạo nhiệm vụ và kịch bản huấn luyện cho agent.
    *   **Đánh giá (Evaluation):** Nhiều benchmark và metric mới được đề xuất (2412.10360, 2412.13147, 2412.06559, 2412.03679, 2412.21187, 2412.15204, 2412.12606, 2412.13018, 2412.09645, 2412.05210, 2412.08580).
        *   *Gap:* Đánh giá các khả năng "mềm" như tính sáng tạo, độ tin cậy, khả năng giải thích, và sự phù hợp với sở thích con người vẫn còn nhiều thách thức. Các benchmark hiện tại có thể nhanh chóng bị bão hòa.
        *   *Opportunity:* Phát triển các phương pháp đánh giá tự động, tương tác và có khả năng thích ứng. Các metric mới tập trung vào quy trình suy luận và các khía cạnh chất lượng khó định lượng.
    *   **An toàn và Độ tin cậy (Safety & Robustness):** Các bài báo như 2412.16720 (o1), 2412.14922 (ROBUST FT), 2412.14689 (ToEdit), 2412.04653 (WIND) chạm đến các khía cạnh này.
        *   *Gap:* Đảm bảo các mô hình lớn hành xử an toàn, không tạo ra thông tin sai lệch (hallucination), và mạnh mẽ trước các tấn công đối nghịch hoặc dữ liệu nhiễu.
        *   *Opportunity:* Các kỹ thuật alignment an toàn mới, phương pháp phát hiện và giảm thiểu ảo giác, cơ chế tự giám sát và tự sửa lỗi liên quan đến an toàn.

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Neuro-Symbolic Multimodal Agents with Dynamic Knowledge Graph Integration**
    *   **Motivation:** Current agents struggle with complex reasoning, long-term planning, and adapting to novel situations due to limitations in symbolic reasoning and knowledge grounding. Multimodal inputs add another layer of complexity.
    *   **Key novelty:** Combine a neural LMM agent with a dynamically updated symbolic knowledge graph (KG). The LMM handles perception and low-level actions, while the KG stores and reasons over structured knowledge extracted from observations and interactions. The novelty lies in the tight, bidirectional coupling and co-evolution of the neural and symbolic components.
    *   **Approach:**
        1.  LMM (e.g., based on Kling 2412.00131 for perception, and a reasoning LLM like o1 2412.16720) processes multimodal input (video, audio, text).
        2.  A "Symbol Grounding Module" (inspired by 3DGraphLLM 2412.18450 but for general concepts) extracts entities, relations, and states, populating/updating a dynamic KG.
        3.  A "Symbolic Planner/Reasoner" (potentially a smaller, specialized LLM or a classic planner) operates on the KG to generate high-level plans and verify LMM-generated sub-goals.
        4.  The LMM executes plan steps, with its reasoning augmented by relevant KG subgraphs (retrieved using techniques similar to RetroLLM 2412.11919).
        5.  Feedback from execution (success/failure, new observations) updates the KG.
    *   **Dataset + Metrics:** Environments like TheAgentCompany (2412.14161) or more complex simulated/real-world robotics tasks (building on CaM 2412.04455 for error detection). Metrics: task completion rate, planning efficiency, KG accuracy, reasoning consistency, adaptability to changes.
    *   **Risk/Feasibility:** High risk. Symbol grounding is a long-standing AI challenge. Maintaining KG consistency and efficient reasoning over a large, dynamic KG is difficult. Scalability of the symbolic planner. Feasibility: Medium-to-High for restricted domains, Low for general-purpose open-world agents in the near term. (Liên lĩnh vực, Khả thi cao cho domain hẹp)

    ✨ **Idea 2: Self-Correcting Byte/Latent Transformers for Universal Data Compression and Generation**
    *   **Motivation:** Models like BLT (2412.09871) and LatentLM (2412.08635) show promise in handling raw data without modality-specific tokenizers. However, ensuring fidelity and learning complex, long-range dependencies in this raw/latent space is challenging.
    *   **Key novelty:** A byte/latent-level Transformer that incorporates a continuous self-correction mechanism during both training and generation, inspired by MALT (2412.01928) or M-STAR (2412.17451) but operating at the level of byte/latent sequence construction.
    *   **Approach:**
        1.  **Core Model:** A large Byte Latent Transformer (BLT) or LatentLM as the generator.
        2.  **Verifier Module:** A smaller, efficient model (or a distilled version of the generator) that predicts the "correctness" or "self-consistency" of generated byte/latent sequences at different granularities (local patches, global structure). This could use ideas from `certaindex` (2412.20993).
        3.  **Refiner Module:** Takes the generator's output and verifier's feedback to iteratively refine the byte/latent sequence. This could involve re-sampling problematic segments or applying learned correction transformations in the latent space.
        4.  **Training:** End-to-end training with a primary reconstruction/generation loss, plus an auxiliary loss from the verifier, and potentially an RL component where the refiner is rewarded for improvements. Data could be any raw data stream (text, images, audio serialized as bytes).
    *   **Dataset + Metrics:** Large diverse datasets of raw byte sequences (e.g., The Stack for code, LAION for images serialized, LibriSpeech for audio). Metrics: Compression ratio vs. reconstruction quality (PSNR, SSIM for images; perplexity for text after decoding), generation quality (FID, IS), and ability to model long-range dependencies (e.g., using LongBench v2 2412.15204 adapted for byte-level tasks).
    *   **Risk/Feasibility:** High risk. Defining "correctness" for arbitrary byte sequences is hard. The verifier and refiner might add significant overhead. Training stability for such a system could be an issue. Feasibility: Medium. (Khả thi cao, có yếu tố moon-shot nếu hướng tới "universal")

    ✨ **Idea 3: Meta-Learned Modular Diffusion Models for Ultra-Personalized Content Generation**
    *   **Motivation:** Current generative models can be fine-tuned for style or subject but struggle with deep, nuanced personalization across many users and rapidly changing preferences. Models like DiffSensei (2412.07589) show a step towards character customization.
    *   **Key novelty:** A framework where a vast library of small, specialized diffusion "skill" modules (e.g., for specific art styles, object types, emotional expressions, animation primitives from FLOAT 2412.01064) are meta-learned. A routing/composition network, conditioned on user history and explicit requests, dynamically selects and combines these modules to generate highly personalized content. This goes beyond simple LoRA mixing.
    *   **Approach:**
        1.  **Module Library:** Train thousands of small, expert diffusion U-Nets (or components like attention blocks from Leffa 2412.08486) on specific, narrow tasks/styles/concepts.
        2.  **Meta-Learning Router:** A meta-learner (e.g., a Transformer or GNN) learns to predict the optimal combination and weighting of skill modules based on a user's profile (implicit preferences from interaction history, explicit preferences) and the current generation task. This could use ideas from CrossFlow (2412.15213) for mapping user state to module configurations.
        3.  **Dynamic Composition:** At inference time, the router assembles a custom diffusion pipeline by selecting and wiring together the chosen modules. The generation process might involve iterative refinement, with different modules contributing at different stages or scales (inspired by SWITTI 2412.01819).
        4.  **Continuous Learning:** User feedback (explicit ratings, implicit signals like engagement) is used to update both the skill modules (fine-tuning) and the meta-learner (RL or preference optimization like LIFT 2412.04814 or OREO 2412.16145).
    *   **Dataset + Metrics:** Large-scale user interaction data with generated content (e.g., from a social platform or creative tool). Datasets like MangaZero (2412.07589) for specific domains. Metrics: Personalization accuracy (how well generated content matches user preferences, possibly evaluated by a separate preference model or human evaluation like MDI-Benchmark 2412.12606), diversity of generated content for a user, speed of adaptation to new preferences, module utilization efficiency.
    *   **Risk/Feasibility:** Very High risk. Managing and training a massive library of modules is complex. The meta-learner for routing/composition is a significant research challenge. Ensuring coherent and high-quality output from dynamically combined modules is difficult. Feasibility: Low in the short-term, a true moon-shot. (Moon-shot)

    ✨ **Idea 4: Explainable and Verifiable Medical LLM Reasoning via Hybrid Inference**
    *   **Motivation:** Medical LLMs (2412.18925, 2412.20070, 2412.07769) need to be highly accurate and their reasoning process transparent and verifiable, which current black-box models struggle with. PROCESS BENCH (2412.06559) highlights error detection needs.
    *   **Key novelty:** A hybrid LLM that combines neural generation of hypotheses and reasoning steps with a symbolic verification layer that checks against medical knowledge bases and logical constraints. The output would not only be an answer but also a verifiable reasoning trace with confidence scores for each step, referencing specific medical facts.
    *   **Approach:**
        1.  **Neural Hypothesis Generation:** An LLM (e.g., a fine-tuned o1-like model or BiMediX2) generates potential diagnostic paths, treatment options, or explanations as a CoT.
        2.  **Symbolic Knowledge Base:** Curate/build a comprehensive medical KG (ontologies, clinical guidelines, drug interactions).
        3.  **Step-wise Verification Module:** Each step in the LLM's CoT is translated into a query or a set of logical statements. This module checks consistency against the KG, identifies unsupported claims, or flags potential contradictions. This could use ideas from Implicit PRM (2412.01981) to assign confidence.
        4.  **Iterative Refinement:** If a step is flagged, the LLM is prompted to revise its reasoning, potentially exploring alternative paths (inspired by CoMCTS 2412.18319 or HiAR-ICL 2411.18478).
        5.  **Explainable Output:** The final output includes the answer, the validated reasoning chain, links to supporting evidence in the KG, and any unresolved uncertainties.
    *   **Dataset + Metrics:** Med-MAT (2412.20070), medical case reports, clinical trial data. Metrics: Accuracy of final answer, verifiability of reasoning steps (human expert + automated KG checks), percentage of unsupported steps, interpretability scores.
    *   **Risk/Feasibility:** High risk. Building and maintaining a comprehensive, accurate medical KG is a massive undertaking. Translating LLM reasoning into verifiable symbolic queries is non-trivial. Ensuring the LLM can effectively use feedback from the verifier. Feasibility: Medium. (Liên lĩnh vực, Khả thi cao cho các chuyên khoa hẹp)

6.  **READING_LIST**

    *   2412.00131 – Kling · SOTA mã nguồn mở cho sinh video, kiến trúc WF-VAE và Skiparse Denoiser rất đáng chú ý.
    *   2412.09871 – BLT · Hướng tiếp cận byte-level LLM với entropy patching rất mới mẻ và hiệu quả.
    *   2412.01506 – TRELLIS/SLAT · Giải pháp toàn diện và chất lượng cao cho tạo tài sản 3D đa định dạng.
    *   2412.08635 – LatentLM · Framework thống nhất tiềm năng cho mô hình hóa đa phương thức rời rạc và liên tục.
    *   2412.01819 – SWITTI · Kiến trúc Transformer non-causal hiệu quả cho sinh ảnh theo scale.
    *   2412.11919 – RetroLLM · Hợp nhất truy xuất và sinh trong RAG một cách thông minh.
    *   2412.07720 – ACD IT · Kết hợp AR và Diffusion một cách sáng tạo cho sinh ảnh/video.
    *   2412.04455 – CaM · Framework phát hiện lỗi robot dựa trên mã VLM tạo ra rất độc đáo.
    *   2412.16145 – OREO · Phương pháp RL ngoại tuyến hiệu quả cho suy luận LLM, giải quyết hạn chế của DPO.
    *   2412.15213 – CrossFlow · Ý tưởng học ánh xạ trực tiếp giữa các modality bằng flow matching rất mới.
    *   2412.17739 – FoPE · Phân tích sâu sắc về RoPE và đề xuất FoPE cải thiện tổng quát hóa độ dài.
    *   2412.03895 – NoiseRefine · Hướng đi thú vị để loại bỏ CFG trong diffusion models bằng cách tinh chỉnh nhiễu.

7.  **META_REFLECTION**

    *   Tập hợp các bài báo tháng 12/2024 cho thấy một sự trưởng thành và phân hóa rõ rệt trong nhiều lĩnh vực AI.
        *   **LLMs và LMMs:** Xu hướng không chỉ dừng lại ở việc tăng quy mô mà còn tập trung mạnh vào hiệu quả (tối ưu hóa, lượng tử hóa, kiến trúc mới như MoE, Byte-level, Latent Space), chất lượng dữ liệu (tổng hợp, lọc, tăng cường), và khả năng suy luận phức tạp (đa bước, tự cải thiện, kết hợp RL). Các mô hình đa phương thức đang tiến tới sự hợp nhất thực sự trong cách xử lý và biểu diễn các modality khác nhau, với nhiều nỗ lực đáng chú ý trong việc tạo video, âm thanh, và nội dung 3D chất lượng cao và có kiểm soát.
        *   **AI Agents:** Lĩnh vực tác nhân (đặc biệt là GUI agents và VLA) đang nhận được sự quan tâm lớn, với việc xây dựng các benchmark thực tế hơn và các phương pháp huấn luyện mới nhằm cải thiện khả năng lập kế hoạch, hành động và tương tác trong môi trường phức tạp.
        *   **Đánh giá và An toàn:** Nhận thức về tầm quan trọng của việc đánh giá toàn diện và đảm bảo an toàn cho các mô hình AI ngày càng tăng. Điều này thể hiện qua sự xuất hiện của nhiều benchmark chuyên biệt hơn (cho văn bản dài, suy luận toán học, sở thích người dùng, tác nhân, RAG) và các phương pháp tập trung vào alignment an toàn, giảm thiểu ảo giác, và xây dựng mô hình phần thưởng quy trình.
        *   **Sự hội tụ của các kỹ thuật:** Nhiều bài báo cho thấy sự giao thoa và kết hợp các kỹ thuật từ các nhánh khác nhau (ví dụ: RL cho LLM reasoning, flow matching cho AR models, diffusion cho 3D/video, LLM cho data generation/evaluation).
        *   **Mở và Tái lập:** Có một xu hướng đáng mừng về việc công bố mã nguồn, dữ liệu và mô hình, thúc đẩy tính tái lập và hợp tác trong cộng đồng (ví dụ: Kling, BLT, các benchmark mới).
    Nhìn chung, lĩnh vực AI đang chuyển dịch từ việc chỉ tập trung vào hiệu năng trên các tác vụ hẹp sang việc xây dựng các hệ thống thông minh hơn, hiệu quả hơn, an toàn hơn, và có khả năng tương tác, suy luận phức tạp trong các kịch bản đa dạng và thực tế hơn. Thách thức về dữ liệu, hiệu quả tính toán và khả năng kiểm chứng vẫn là những động lực chính cho các nghiên cứu tiếp theo.
