1.  **TOPIC_TREE**

    *   **Mô hình Ngôn ngữ Lớn (Large Language Models - LLMs)**
        *   **Kiến trúc và Cơ chế Cốt lõi (Architectures & Core Mechanisms)**
            *   Cải tiến Kiến trúc Transformer (Transformer Innovations)
                *   `2502.05795` | LayerNorm Scaling giải quyết "Curse of Depth" do Pre-LN gây ra bằng cách kiểm soát bùng nổ phương sai ở các tầng sâu.
                *   `2502.09245` | LIMe (Layer-Integrated Memory) cho phép các đầu chú ý truy cập biểu diễn từ tất cả các tầng trước đó, cải thiện dòng thông tin theo chiều sâu.
                *   `2502.19587` | NeoBERT hiện đại hóa kiến trúc BERT với RoPE, Pre-RMSNorm, SwiGLU và chiến lược huấn luyện hai giai đoạn cho ngữ cảnh dài.
            *   Mô hình Chuỗi Tuyến tính & Không gian Trạng thái (Linear Sequence & State Space Models)
                *   `2502.11089` (cũng là `2502.13685`) | Mixture-of-Memories (MoM) tăng cường dung lượng và giảm nhiễu bộ nhớ cho mô hình chuỗi tuyến tính bằng nhiều bộ nhớ độc lập và router.
                *   `2502.13145` | mmMamba phát triển VLM dựa trên Mamba-2 thông qua chưng cất lũy tiến từ Transformer VLM, đạt độ phức tạp tuyến tính.
            *   Mô hình Khuếch tán cho Ngôn ngữ (Diffusion Models for Language)
                *   `2502.09992` | LLaDA là LLM 8B đầu tiên dựa trên khuếch tán mặt nạ, cho thấy khả năng học trong ngữ cảnh và tuân thủ chỉ dẫn tương tự ARM.
                *   `2502.11564` | RDLM (Riemannian Diffusion Language Model) tích hợp hình học đa tạp thống kê vào khuếch tán liên tục cho dữ liệu rời rạc, tổng quát hóa các mô hình khuếch tán rời rạc.
        *   **Huấn luyện và Tinh chỉnh (Training & Fine-tuning)**
            *   Chiến lược Dữ liệu và Tiền huấn luyện (Data Strategies & Pre-training)
                *   `2502.02737` | Phát triển tập dữ liệu chuyên biệt (FineMath, Stack-Edu) và huấn luyện đa giai đoạn với tái cân bằng thủ công cho mô hình nhỏ.
                *   `2502.18934` | Kanana (Anh-Hàn) sử dụng tiền huấn luyện theo giai đoạn, depth up-scaling, và iterative pruning/distillation để đạt hiệu quả cao với chi phí thấp.
            *   Học Suy luận và Kiến thức (Reasoning & Knowledge Learning)
                *   `2501.19393` | Quy trình giám tuyển dữ liệu s1K (1000 mẫu) hiệu quả và kỹ thuật "budget forcing" để kiểm soát thời gian suy nghĩ của LLM.
                *   `2502.03387` | Giả thuyết LIMO cho thấy LLM có thể học suy luận phức tạp từ lượng rất nhỏ dữ liệu chất lượng cao (817 mẫu) làm "khuôn mẫu nhận thức".
                *   `2502.07374` | Phân tích chưng cất Long CoT, chỉ ra cấu trúc logic quan trọng hơn nội dung bước suy luận.
                *   `2502.12143` | Mix Distillation giải quyết "Small Model Learnability Gap" bằng cách trộn dữ liệu CoT dài/ngắn hoặc từ teacher lớn/nhỏ.
                *   `2502.07316` | CODEI/O huấn luyện LLM dự đoán đầu vào/đầu ra của code và sinh CoT, CODEI/O++ thêm phản hồi thực thi để cải thiện.
            *   Chỉnh sửa và Căn chỉnh Mô hình (Model Editing & Alignment)
                *   `2502.14502` | Nghiên cứu tinh chỉnh LoRA bằng diễn giải dữ kiện mới và bổ sung dữ kiện đã biết để giảm tác động tiêu cực lên kiến thức cũ.
                *   `2502.01237` | Cải tiến ORPO và ASFT thành phương pháp hai giai đoạn và thêm tham số β để kiểm soát tối ưu hóa sở thích.
                *   `2502.09604` | SelfCite sử dụng tín hiệu phần thưởng tự giám sát (context ablation) để cải thiện tạo trích dẫn mà không cần dữ liệu chú giải.
                *   `2502.13131` | DRMs (Decomposed Reward Models) dùng PCA để trích xuất các thành phần sở thích trực giao từ dữ liệu so sánh nhị phân, hỗ trợ hiệu chỉnh tại thời điểm kiểm thử.
        *   **Hiệu quả và Khả năng mở rộng (Efficiency & Scalability)**
            *   Xử lý Ngữ cảnh Dài (Long Context Handling)
                *   `2502.08910` | InfiniteHiP tích hợp tỉa token phân cấp, điều chỉnh RoPE động và offload KV cache hiệu quả cho suy luận ngữ cảnh dài.
                *   `2502.17129` | Khảo sát toàn diện về các kỹ thuật cho LLM ngữ cảnh dài (kiến trúc, cơ sở hạ tầng, huấn luyện, đánh giá).
                *   `2502.20082` | LongRoPE2 đề xuất thuật toán điều chỉnh tỷ lệ RoPE dựa trên tìm kiếm tiến hóa và huấn luyện cửa sổ ngữ cảnh hỗn hợp để mở rộng ngữ cảnh hiệu quả.
            *   Nén Mô hình và Suy luận Hiệu quả (Model Compression & Efficient Inference)
                *   `2502.13063` | Nghiên cứu nén chuỗi token thành vector [mem] bằng tối ưu hóa per-sample, khám phá dung lượng thông tin của LLM.
                *   `2501.19324` | RSD (Reward-Guided Speculative Decoding) sử dụng PRM để tăng tốc suy luận LLM cho tác vụ suy luận.
                *   `2502.05003` | QuEST áp dụng biến đổi Hadamard và trust gradient estimator cho QAT, cho phép lượng tử hóa LLM xuống tới 1-bit.
                *   `2502.07864` | TransMLA chuyển đổi mô hình GQA sang MLA tương đương toán học, tăng khả năng biểu diễn với cùng KV cache.
                *   `2502.18137` | SpargeAttn là phương pháp attention thưa trực tuyến hai giai đoạn, không cần huấn luyện, tăng tốc suy luận cho DiT.
            *   Scaling Laws
                *   `2502.08606` | Đề xuất định luật co giãn chưng cất (distillation scaling law) đầu tiên, mô hình hóa ảnh hưởng của thầy và trò.
        *   **Đánh giá, Phân tích và Diễn giải (Evaluation, Analysis & Interpretability)**
            *   Đánh giá Năng lực và Độ tin cậy (Capability & Trustworthiness Evaluation)
                *   `2502.08946` | PHYSICO benchmark đánh giá mức độ hiểu khái niệm vật lý của LLM, định lượng hiện tượng "vẹt ngẫu nhiên".
                *   `2502.06329` | FailSafeQA benchmark đánh giá khả năng phục hồi và nhận thức ngữ cảnh của LLM trong QA tài chính ngữ cảnh dài.
                *   `2502.14296` | TrustGen, nền tảng đánh giá động tính đáng tin cậy của GenFM (T2I, LLM, VLM).
                *   `2502.01534` | Xác định và định lượng "rò rỉ ưu tiên" (preference leakage) trong LLM-làm-giám-khảo.
                *   `2502.17955` | Benchmark đánh giá khả năng truy xuất và chuyển giao kiến thức thực tế xuyên ngôn ngữ của LM (X-FaKT).
                *   `2502.04313` | CAPA, thước đo tương đồng mô hình xác suất điều chỉnh theo cơ hội.
            *   Diễn giải và Phân tích Cơ chế Bên trong (Interpretability & Internal Mechanisms)
                *   `2502.15007` | LLM-Microscope, framework phân tích phi tuyến tính và bộ nhớ ngữ cảnh ở cấp độ token.
                *   `2502.03032` | Theo dõi luồng đặc trưng xuyên lớp dựa trên cosine similarity của trọng số bộ giải mã SAE, xây dựng đồ thị luồng.
        *   **Ứng dụng LLM (LLM Applications)**
            *   Suy luận Toán học và Logic (Mathematical & Logical Reasoning)
                *   `2502.06703` | Nghiên cứu Test-Time Scaling (TTS) tối ưu tính toán và có nhận thức phần thưởng cho LLM, cho thấy mô hình nhỏ có thể vượt mô hình lớn.
                *   `2502.19613` | Framework hai giai đoạn (IFT + RL) cho LLM tự thưởng và tự sửa lỗi trong suy luận toán học.
                *   `2502.06781` | OREAL, framework RL dựa trên phần thưởng kết quả cho suy luận toán học, với lý thuyết về học từ mẫu dương BoN.
                *   `2502.01456` | PRIME, framework RL trực tuyến với Implicit Process Reward Model cho suy luận toán học và lập trình.
                *   `2502.03373` | Tối ưu hóa CoT dài bằng Cosine Reward, hình phạt lặp N-gram, và Action Prompting.
                *   `2502.14768` | Logic-RL, framework RL dựa trên luật để huấn luyện suy luận logic trên câu đố K&K.
                *   `2502.03544` | AlphaGeometry2 mở rộng ngôn ngữ miền, tạo sơ đồ tự động, và thuật toán tìm kiếm SKEST cho chứng minh định lý hình học.
            *   Lập trình và Kỹ thuật Phần mềm (Coding & Software Engineering)
                *   `2502.18449` | SWE-RL, phương pháp RL đầu tiên dùng dữ liệu tiến hóa phần mềm và phần thưởng dựa trên luật để cải thiện LLM cho tác vụ SE.
                *   `2502.14382` | S*, framework scaling thời gian kiểm thử lai (song song + tuần tự gỡ lỗi) cho sinh mã.
                *   `2502.06807` | So sánh LRM mục đích chung (o1, o3) với hệ thống chuyên biệt (o1-ioi) cho lập trình thi đấu, cho thấy RL quy mô lớn có thể vượt trội.
                *   `2502.12115` | SWE-Lancer, benchmark dựa trên công việc freelance thực tế, đánh giá LLM trên tác vụ IC SWE và SWE Manager bằng E2E test.
            *   Tài chính (Finance)
                *   `2502.08127` | Fino1, LLM chuyên biệt cho tài chính, tinh chỉnh bằng SFT (CoT chưng cất) và RL (PPO với verifier-based reward).
                *   `2502.18772` | Plutus-ben, benchmark tài chính tiếng Hy Lạp và Plutus-8B, LLM tài chính tiếng Hy Lạp.
                *   `2502.05878` | Framework RAG cho dự báo chuỗi thời gian tài chính (StockLLM, FinSeer).
                *   `2502.11433` | FLAG-TRADER, LLM tích hợp RL dựa trên gradient cho giao dịch tài chính.
            *   Tạo Khảo sát Học thuật (Academic Survey Generation)
                *   `2502.14776` | SurveyX, hệ thống tự động tạo khảo sát học thuật bằng LLM với quy trình hai giai đoạn và AttributeTree.
            *   Tạo Dữ liệu Giải độc Văn bản (Text Detoxification Data Generation)
                *   `2502.06394` | SynthDetoxM, bộ dữ liệu song ngữ tổng hợp đa ngôn ngữ cho giải độc văn bản, tạo bằng LLM với few-shot prompting.
    *   **Mô hình Đa phương thức (Multimodal Models - MLLMs / VLMs)**
        *   Kiến trúc và Cơ chế Nền tảng (Architectures & Foundational Mechanisms)
            *   `2502.13923` | Qwen2.5-VL cải tiến ViT với Window Attention, MRoPE căn chỉnh thời gian tuyệt đối, và xử lý độ phân giải/FPS động.
            *   `2502.14786` | SigLIP 2, họ bộ mã hóa thị giác-ngôn ngữ đa ngôn ngữ, kết hợp SigLIP, LocCa, SILC/TIPS, NaFlex và chắt lọc ACID.
            *   `2502.01341` | Module ALIGN ánh xạ đặc trưng thị giác vào tổ hợp lồi của embedding văn bản trong LLM.
            *   `2502.05173` | VideoRoPE, phương pháp mã hóa vị trí quay 3D cho video (LTA, DL, ATS) để hiểu video dài.
        *   Huấn luyện và Dữ liệu (Training & Data)
            *   `2502.18411` | OmniAlign-V, bộ dữ liệu SFT và DPO đa phương thức để cải thiện liên kết MLLM với sở thích người dùng; MM-AlignBench.
        *   Đánh giá và Phân tích (Evaluation & Analysis)
            *   `2502.10391` | Mô hình Phần thưởng dựa trên Phê bình và Điều chỉnh Tỷ lệ Phần thưởng Động trong MM-DPO cho MLLM.
            *   `2502.04320` | ConceptAttention, phương pháp diễn giải DiT đa phương thức, tạo bản đồ saliency cho khái niệm văn bản tùy ý.
        *   Ứng dụng Cụ thể (Specific Applications)
            *   `2502.19634` | MedVLM-R1, Medical VLM tạo giải thích suy luận tường minh cho VQA X quang bằng RL (GRPO).
            *   `2502.13449` | Mol-LLaMA, mô hình ngôn ngữ phân tử lớn, tinh chỉnh hướng dẫn đa phương thức để hiểu biết tổng quát về phân tử.
    *   **Sinh Ảnh và Video (Image & Video Generation/Editing)**
        *   Sinh Video (Video Generation)
            *   `2502.01061` | OmniHuman tạo video người với điều kiện omni-conditions, appearance conditioning tái sử dụng DiT, và CFG annealing.
            *   `2502.04896` | Goku Transformer sinh ảnh và video đồng thời bằng Rectified Flow và full attention, huấn luyện đa giai đoạn.
            *   `2502.02492` | VideoJAM học biểu diễn kết hợp ngoại hình-chuyển động và Inner-Guidance để tăng tính nhất quán chuyển động video.
            *   `2502.11079` | Phantom sinh video nhất quán chủ thể với data pipeline cross-pair và cơ chế nhúng động.
            *   `2502.10248` | Step-Video-T2V, mô hình T2V 30B với Video-VAE nén sâu, DiT 3D full attention, và Video-DPO.
            *   `2502.08639` | CineMaster, framework T2V hai giai đoạn với điều khiển 3D (bounding box, camera) và Semantic Layout ControlNet.
            *   `2502.04507` | Sliding Tile Attention (STA) tối ưu hóa Video DiT bằng cách loại bỏ mixed blocks, có kernel tối ưu và STA Mask Search.
        *   Sinh Ảnh (Image Generation)
            *   `2502.07870` | TextAtlas5M, bộ dữ liệu quy mô lớn cho sinh ảnh chứa văn bản dày đặc/dạng dài; TextAtlasEval benchmark.
            *   `2502.18364` | ART (Anonymous Region Transformer) sinh đồng thời ảnh tham chiếu, nền và nhiều lớp ảnh trong suốt dựa trên bố cục vùng ẩn danh.
            *   `2502.17157` | DICEPTION, mô hình khuếch tán tổng quát cho nhận thức thị giác, thống nhất đầu ra tác vụ vào không gian RGB.
        *   Chỉnh sửa Ảnh và Video (Image & Video Editing)
            *   `2502.17258` | VideoGrain, chỉnh sửa video đa mức độ chi tiết (class, instance, part) bằng ST-Layout Attention điều chỉnh cross- và self-attention.
            *   `2502.17363` | KV-Edit, chỉnh sửa ảnh dựa trên văn bản không cần huấn luyện, dùng KV cache trong DiT để bảo toàn nền.
            *   `2502.14397` | PhotoDoodle, framework chỉnh sửa ảnh tạo hiệu ứng "doodle" từ few-shot, với OmniEditor và EditLoRA.
            *   `2502.08590` | Light-A-Video, framework chiếu sáng lại video không cần huấn luyện, dùng Consistent Light Attention và Progressive Light Fusion.
            *   `2501.14677` | MatAnyone, video matting với Consistent Memory Propagation và chiến lược huấn luyện dùng dữ liệu phân đoạn.
            *   `2502.18417` | GHOST 2.0, cải tiến head reenactment với Aligner dùng bộ mã hóa Emotion duy nhất và Blender cải tiến.
            *   `2502.05176` | AuraFusion360, lấp đầy vùng trống trong cảnh 360° dựa trên ảnh tham chiếu, sử dụng AGDD và SDEdit với nhiễu cấu trúc.
        *   Nén và Biểu diễn 3D (3D Compression & Representation)
            *   `2502.06608` | TripoSG, mô hình transformer sinh hình dạng 3D (SDF) từ ảnh dựa trên rectified flow, với VAE 3D mới và MoE.
    *   **Hệ thống Tác tử AI (AI Agent Systems)**
        *   Nền tảng và Framework (Platforms & Frameworks)
            *   `2502.14499` | MLGym, framework môi trường Gym cho đánh giá và phát triển AI Research Agents bằng RL; MLGym-Bench.
            *   `2502.13130` | Magma, mô hình nền tảng đa phương thức cho AI agent, huấn luyện bằng SoM (Set-of-Mark) và ToM (Trace-of-Mark).
            *   `2502.18864` | AI co-scientist, hệ thống đa tác nhân (Gemini 2.0) hỗ trợ khám phá khoa học, tạo giả thuyết và lên kế hoạch thí nghiệm.
            *   `2502.01506` | TwinMarket, mô phỏng thị trường tài chính đa tác nhân với tác tử LLM-BDI và mạng xã hội động.
        *   Phân tích Hành vi Tác tử (Agent Behavior Analysis)
            *   `2502.08235` | Phân tích hiện tượng "overthinking" trong LRM agentic, đề xuất khung đánh giá và chiến lược giảm thiểu.
        *   Giao tiếp Đa Tác tử (Multi-Agent Communication)
            *   `2502.06060` | Huấn luyện LLM giao tiếp đa tác tử trong game suy luận xã hội (Among Us) không cần dữ liệu người, tách biệt nghe-nói và dùng phần thưởng dày đặc.
    *   **Xử lý Âm thanh và Tiếng nói (Audio & Speech Processing)**
        *   Mô hình Ngôn ngữ Nói (Speech Language Models - SLMs)
            *   `2502.12900` | Soundwave, framework huấn luyện SLM hiệu quả dữ liệu với Alignment Adapter, Shrinking Adapter và Dynamic Data Mixture.
            *   `2502.15814` | Slam, quy trình huấn luyện SLM chất lượng cao trên 1 GPU/24h, tối ưu các thành phần và dùng DPO với dữ liệu tổng hợp.
        *   Sinh Nhạc (Music Generation)
            *   `2502.13128` | SongGen, kiến trúc transformer tự hồi quy đơn giai đoạn cho sinh nhạc từ văn bản, hỗ trợ mixed/dual-track mode và auxiliary vocal loss.
        *   Bộ dữ liệu Âm thanh (Audio Datasets)
            *   `2502.16584` | Audio-FLAN, bộ dữ liệu instruction-tuning quy mô lớn hợp nhất tác vụ hiểu và sinh audio (tiếng nói, âm nhạc, âm thanh).
    *   **Học tăng cường (Reinforcement Learning - RL)**
        *   Phương pháp RL cho LLM (RL Methods for LLMs)
            *   `2502.06781` | OREAL (Outcome REwArd-based RL) cho suy luận toán học.
            *   `2502.01456` | PRIME (Process RL through IMplicit rEwards) cho suy luận toán học/lập trình.
            *   `2502.03373` | RL với Cosine Reward và Action Prompting cho Long CoT.
        *   Ứng dụng RL (RL Applications)
            *   `2502.14372` | RL cho giảm trọng số mã sửa lỗi lượng tử.
            *   `2502.12152` | HUMANUP, RL hai giai đoạn cho robot hình người đứng dậy.
            *   `2502.13144` | RAD, RL đầu-cuối dựa trên 3DGS cho lái xe tự hành.
            *   `2502.11433` | FLAG-TRADER, RL cho giao dịch tài chính bằng LLM.
    *   **Xây dựng và Đánh giá Benchmark (Benchmark Creation & Evaluation)**
        *   `2502.14739` | SuperGPQA, benchmark kiến thức cấp sau đại học, xây dựng bằng cộng tác Người-LLM.
        *   `2502.09560` | EMBODIEDBENCH, benchmark toàn diện cho MLLM-based embodied agents.
        *   `2502.09696` | ZeroBench, benchmark suy luận hình ảnh cực khó cho LMM.
        *   `2502.09619` | ProbeLog, phương pháp tìm kiếm mô hình zero-shot dựa trên probing và collaborative probing.
    *   **Nén Mô hình và Tối ưu hóa Bộ mã hóa (Model Compression & Encoder Optimization)**
        *   `2502.08690` | Skrr, tỉa khối hai pha (Skip & Re-use) cho bộ mã hóa văn bản trong T2I.
    *   **Other**
        *   (Không có paper nào thuộc nhóm này, đạt mục tiêu < 5%)

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                      |
    | :--- | :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | `2502.06807` | LRM, Competitive Programming, RL, o3, Test-time Strategy | Mô hình LRM (o3) được huấn luyện bằng RL quy mô lớn, tự học chiến lược suy luận, vượt trội hệ thống chuyên biệt với chiến lược thủ công. | Mở ra hướng phát triển LRM mục đích chung mạnh mẽ, có khả năng tự tối ưu hóa suy luận, giảm sự phụ thuộc vào thiết kế thủ công.                 |
    | 2    | `2502.03544` | Automated Theorem Proving, Geometry, Symbolic Reasoning, Neuro-Symbolic, SKEST | AlphaGeometry2 giải quyết các bài toán hình học Olympiad phức tạp, bao gồm cả bài toán phi kiến thiết, tự động hóa hình thức hóa bằng Gemini. | Đẩy mạnh giới hạn của AI trong suy luận toán học hình thức, tiệm cận khả năng của con người ở các cuộc thi danh giá.                             |
    | 3    | `2502.10248` | Text-to-Video, Diffusion Transformer, Video-VAE, Video-DPO, Bilingual | Step-Video-T2V, mô hình T2V mã nguồn mở 30B tham số, đạt chất lượng SOTA với Video-VAE nén sâu và tối ưu hóa bằng Video-DPO.             | Cung cấp một mô hình T2V mã nguồn mở cực mạnh, thúc đẩy nghiên cứu và ứng dụng trong lĩnh vực sinh video chất lượng cao, có kiểm soát.          |
    | 4    | `2502.06608` | Image-to-3D, Flow Transformer, 3D VAE, SDF, MoE      | TripoSG, mô hình flow transformer 4B tham số sinh hình dạng 3D (SDF) từ ảnh đơn, đạt độ trung thực SOTA nhờ VAE 3D mới và dữ liệu lớn. | Một bước tiến lớn trong sinh 3D từ ảnh, mở ra khả năng tạo tài sản 3D chất lượng cao nhanh chóng, có ứng dụng trong game, VR/AR.             |
    | 5    | `2502.05003` | LLM Compression, QAT, Hadamard Transform, 1-bit Quantization | QuEST cho phép QAT ổn định cho LLM xuống tới 1-bit (W1A1) với hiệu năng cạnh tranh BF16, sử dụng biến đổi Hadamard và trust gradient. | Cách mạng hóa việc triển khai LLM trên thiết bị tài nguyên hạn chế, giảm đáng kể chi phí bộ nhớ và năng lượng.                               |
    | 6    | `2502.18864` | AI for Science, Multi-Agent System, Hypothesis Generation, Drug Discovery | AI co-scientist (Gemini 2.0) hỗ trợ khám phá khoa học y sinh, tạo giả thuyết và lên kế hoạch thí nghiệm, đã có xác thực wet-lab.        | Cho thấy tiềm năng to lớn của AI trong việc tăng tốc và tự động hóa các khâu quan trọng của nghiên cứu khoa học, đặc biệt trong y sinh.        |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **`2502.02737` – Quy trình huấn luyện đa giai đoạn với tái cân bằng thủ công (on-the-fly rebalancing) cho LLM nhỏ** – Điều chỉnh tỷ lệ trộn dữ liệu dựa trên hiệu suất giai đoạn trước là một cách tiếp cận thực tế và thông minh để tối ưu mô hình nhỏ khi không thể chạy nhiều thử nghiệm toàn diện. *Suy nghĩ: Rất hữu ích cho các nhóm nghiên cứu có nguồn lực hạn chế, nhấn mạnh data-centric AI.*
    *   **`2502.01061` – Chiến lược huấn luyện omni-conditions và Appearance Conditioning tái sử dụng DiT cho tạo video người** – Cho phép học từ dữ liệu đa dạng và bảo toàn danh tính hiệu quả mà không tăng tham số. *Suy nghĩ: Giải quyết tốt vấn đề giới hạn dữ liệu và chi phí tham số trong sinh video người.*
    *   **`2502.08946` – PHYSICO benchmark với cặp nhiệm vụ low-level (ghi nhớ) và high-level (suy luận trừu tượng bằng lưới) để đo lường hiểu biết vật lý của LLM** – Một cách tiếp cận sáng tạo để định lượng "vẹt ngẫu nhiên". *Suy nghĩ: Cung cấp một công cụ đánh giá sâu sắc hơn về khả năng hiểu thực sự của LLM.*
    *   **`2502.14499` – MLGym, framework môi trường Gym cho AI Research Agents, cho phép huấn luyện bằng RL** – Đặt nền móng cho việc phát triển tác tử AI nghiên cứu một cách có hệ thống. *Suy nghĩ: Mở ra hướng mới cho việc tự động hóa nghiên cứu AI bằng RL.*
    *   **`2502.13923` – MRoPE căn chỉnh Thời gian Tuyệt đối và sử dụng tọa độ tuyệt đối trong MLLM (Qwen2.5-VL)** – Cải thiện hiểu biết không gian-thời gian, đặc biệt cho video và định vị. *Suy nghĩ: Giải quyết các vấn đề tinh tế nhưng quan trọng trong xử lý đa phương thức động.*
    *   **`2502.15007` – Phương pháp đánh giá bộ nhớ ngữ cảnh ở cấp độ token bằng tái tạo tiền tố và phát hiện vai trò token "phụ" (LLM-Microscope)** – Cung cấp công cụ định lượng mới và phát hiện thú vị về vai trò của các token không mang nhiều ngữ nghĩa. *Suy nghĩ: Giúp hiểu sâu hơn về cách LLM lưu trữ và xử lý thông tin ngữ cảnh.*
    *   **`2502.11089` – Kiến trúc Mixture-of-Memories (MoM) với router cho mô hình chuỗi tuyến tính** – Lấy cảm hứng từ khoa học thần kinh, giải quyết vấn đề dung lượng và nhiễu bộ nhớ. *Suy nghĩ: Một hướng đi mới cho các mô hình thay thế Transformer hiệu quả hơn.*
    *   **`2502.08910` – Modular Hierarchical Context Pruning và Dynamic RoPE Adjustment trong InfiniteHiP** – Giải quyết đồng thời hiệu năng, bộ nhớ và OOL generalization cho ngữ cảnh dài. *Suy nghĩ: Một giải pháp toàn diện và thực tế cho LLM ngữ cảnh siêu dài.*
    *   **`2502.05171` – Kiến trúc LLM hồi quy theo chiều sâu (Prelude-Core-Coda) cho suy luận ngầm** – Cho phép tăng độ sâu tính toán linh hoạt tại thời điểm kiểm tra mà không tăng tham số. *Suy nghĩ: Một cách tiếp cận thú vị để mô hình "suy nghĩ sâu hơn" mà không cần sinh token trung gian.*
    *   **`2501.19393` – Kỹ thuật "budget forcing" can thiệp tại thời điểm giải mã để kiểm soát thời gian suy nghĩ của LLM** – Một phương pháp đơn giản nhưng hiệu quả để cải thiện suy luận. *Suy nghĩ: Rất thực tế và dễ áp dụng để tăng hiệu suất LLM trong các tác vụ khó.*
    *   **`2502.09992` – LLaDA, LLM 8B dựa trên khuếch tán mặt nạ (masked diffusion)** – Chứng minh mô hình khuếch tán có thể đạt được các khả năng của LLM tự hồi quy. *Suy nghĩ: Mở ra một hướng kiến trúc mới tiềm năng cho LLM, có thể có những ưu điểm riêng.*
    *   **`2502.04896` – Goku Transformer với full attention và rectified flow cho sinh ảnh và video đồng thời** – Cho phép mô hình hóa tốt hơn các chuyển động phức tạp và học biểu diễn thống nhất. *Suy nghĩ: Kiến trúc mạnh mẽ cho sinh đa phương thức không gian-thời gian.*
    *   **`2502.14776` – AttributeTree để tiền xử lý tài liệu tham khảo thành cấu trúc cây cho SurveyX** – Trích xuất thông tin khóa hiệu quả và tối ưu hóa sử dụng ngữ cảnh LLM. *Suy nghĩ: Một kỹ thuật tiền xử lý thông minh cho các tác vụ RAG hoặc tổng hợp tài liệu.*
    *   **`2502.12900` – Soundwave với Alignment Adapter (CTC loss) và Shrinking Adapter (CTC-based + cross-attention) cho Speech LLM** – Căn chỉnh và rút gọn hiệu quả, cho phép huấn luyện SLM chất lượng cao với ít dữ liệu. *Suy nghĩ: Giải quyết các thách thức cốt lõi trong việc xây dựng SLM hiệu quả.*
    *   **`2502.17258` – ST-Layout Attention trong VideoGrain điều chỉnh đồng thời cross- và self-attention dựa trên layout để chỉnh sửa video đa mức độ chi tiết** – Giải quyết vấn đề feature coupling. *Suy nghĩ: Một cơ chế attention tinh vi cho phép kiểm soát chỉnh sửa video ở mức độ cao.*
    *   **`2502.18449` – Hàm phần thưởng dựa trên luật (difflib.SequenceMatcher) cho SWE-RL** – Tránh sự phức tạp của phần thưởng dựa trên thực thi trong RL cho kỹ thuật phần mềm. *Suy nghĩ: Một giải pháp thực tế và hiệu quả cho việc áp dụng RL vào các miền phức tạp.*
    *   **`2501.19324` – Reward-Guided Speculative Decoding (RSD) sử dụng PRM để chấp nhận/từ chối token từ mô hình nháp** – Giới thiệu "controlled bias" để tăng tốc suy luận, đặc biệt cho tác vụ suy luận. *Suy nghĩ: Một cải tiến thông minh cho speculative decoding, cân bằng giữa tốc độ và chất lượng.*
    *   **`2502.13063` – Tối ưu hóa vector [mem] trên từng mẫu để nén chuỗi token vào LLM đóng băng** – Khám phá giới hạn dung lượng biểu diễn của không gian nhúng LLM. *Suy nghĩ: Một phương pháp thú vị để nghiên cứu lý thuyết thông tin trong LLM, dù chưa thực tế cho nén thông thường.*
    *   **`2502.15814` – Slam, quy trình huấn luyện SLM trên 1 GPU/24h, kết hợp khởi tạo TWIST, dữ liệu tổng hợp và DPO với dữ liệu tổng hợp** – Chứng minh khả năng đạt hiệu năng cao với tài nguyên cực thấp. *Suy nghĩ: Dân chủ hóa nghiên cứu SLM.*
    *   **`2502.05173` – VideoRoPE với Phân bổ Tần số Thấp cho Chiều Thời gian (LTA) và Bố cục Đường chéo (DL)** – Cải thiện mã hóa vị trí 3D cho video dài, giảm nhiễu và tăng tính đối xứng. *Suy nghĩ: Giải quyết các vấn đề nền tảng trong việc mở rộng RoPE cho video.*
    *   **`2502.02492` – Inner-Guidance trong VideoJAM, sử dụng dự đoán chuyển động nội tại làm tín hiệu hướng dẫn động** – Tăng cường tính nhất quán chuyển động video. *Suy nghĩ: Một cơ chế self-guidance mới lạ và hiệu quả.*
    *   **`2502.19634` – MedVLM-R1 sử dụng RL (GRPO) với phần thưởng dựa trên luật để tạo giải thích suy luận tường minh cho VQA X quang** – Tiên phong trong việc tạo VLM y tế có khả năng diễn giải. *Suy nghĩ: Rất quan trọng cho việc xây dựng AI y tế đáng tin cậy.*
    *   **`2502.14382` – S* với tổng hợp đầu vào thích ứng (adaptive input synthesis) và đối chiếu theo cặp dựa trên thực thi để lựa chọn mẫu mã** – Cơ chế lựa chọn mạnh mẽ cho scaling lai trong sinh mã. *Suy nghĩ: Một cách tiếp cận thông minh để chọn giải pháp tốt nhất khi có nhiều ứng viên.*
    *   **`2502.06781` – OREAL với mô hình phần thưởng cấp token để gán tín dụng trong RL cho suy luận toán học** – Giải quyết vấn đề phần thưởng thưa một cách hiệu quả. *Suy nghĩ: Cải thiện đáng kể việc huấn luyện RL cho các tác vụ suy luận dài.*
    *   **`2502.01456` – PRIME cập nhật Implicit PRM trực tuyến chỉ với nhãn kết quả đầu ra** – Giảm chi phí thu thập dữ liệu và nguy cơ reward hacking. *Suy nghĩ: Một bước tiến quan trọng trong việc làm cho RL với PRM trở nên thực tế hơn.*
    *   **`2502.11079` – Data pipeline tạo cặp dữ liệu chéo video (cross-pair) cho Phantom để chống copy-paste trong sinh video nhất quán chủ thể** – Giải quyết một vấn đề khó trong huấn luyện mô hình sinh có điều kiện từ ảnh. *Suy nghĩ: Rất quan trọng để mô hình học được sự biến dạng và khái quát hóa chủ thể.*
    *   **`2502.03032` – Theo dõi luồng đặc trưng SAE xuyên lớp bằng cosine similarity trọng số bộ giải mã, không cần dữ liệu** – Cung cấp công cụ diễn giải mạnh mẽ và điều khiển mô hình đa lớp. *Suy nghĩ: Mở ra khả năng hiểu và kiểm soát LLM ở mức độ chi tiết hơn.*
    *   **`2502.13130` – SoM (Set-of-Mark) và ToM (Trace-of-Mark) làm nhiệm vụ tiền huấn luyện thay thế cho AI agent (Magma)** – Học action grounding và planning từ dữ liệu đa dạng (UI, robot, video). *Suy nghĩ: Phương pháp sáng tạo để hợp nhất các nguồn dữ liệu không đồng nhất cho agent.*
    *   **`2502.18137` – SpargeAttn với nén token có chọn lọc theo độ tự tương đồng và softmax trực tuyến thưa tại cấp độ warp** – Tăng tốc attention hiệu quả mà không cần huấn luyện. *Suy nghĩ: Một kỹ thuật tối ưu hóa attention rất thực tế và mạnh mẽ.*
    *   **`2502.08690` – Skrr với pha Tái sử dụng (Re-use) khối liền kề để bù đắp khối bị tỉa trong bộ mã hóa văn bản T2I** – Phục hồi hiệu năng hiệu quả sau khi tỉa. *Suy nghĩ: Một cải tiến thông minh cho các phương pháp tỉa mô hình.*
    *   **`2502.05003` – QuEST với biến đổi Hadamard và trust gradient estimator cho QAT** – Cho phép lượng tử hóa LLM xuống mức bit rất thấp. *Suy nghĩ: Đột phá trong nén LLM, mở đường cho triển khai trên thiết bị yếu.*
    *   **`2502.03544` – Thuật toán tạo sơ đồ tự động cho bài toán hình học phi kiến thiết và SKEST (Shared Knowledge Ensemble of Search Trees) trong AlphaGeometry2** – Mở rộng đáng kể khả năng giải toán hình học của AI. *Suy nghĩ: Các thuật toán mạnh mẽ cho suy luận tượng trưng phức tạp.*
    *   **`2502.08639` – Semantic Layout ControlNet và Camera Adapter trong CineMaster để điều khiển sinh video 3D** – Tích hợp hiệu quả tín hiệu điều khiển 3D vào mô hình khuếch tán T2V. *Suy nghĩ: Cho phép kiểm soát chi tiết hơn trong quá trình sáng tạo video.*
    *   **`2502.18364` – Bố cục vùng ẩn danh (anonymous region layout) và ART (Anonymous Region Transformer) cho sinh ảnh đa lớp** – Cho phép sinh đồng thời nhiều lớp ảnh trong suốt với số lượng và độ phân giải thay đổi. *Suy nghĩ: Một cách tiếp cận mới cho compositional generation, giảm gánh nặng chú thích.*
    *   **`2502.14372` – Hàm phần thưởng cân bằng giảm bậc nút, bảo toàn khoảng cách mã và action masking cho RL trong giảm trọng số mã QEC** – Thiết kế RL chuyên biệt cho một bài toán tối ưu hóa phức tạp trong vật lý lượng tử. *Suy nghĩ: Minh chứng cho khả năng ứng dụng của RL trong các lĩnh vực khoa học chuyên sâu.*
    *   **`2502.11433` – FLAG-TRADER tích hợp LLM vào policy network của RL cho giao dịch tài chính, tinh chỉnh một phần LLM bằng policy gradient** – Kết hợp suy luận ngôn ngữ và tối ưu hóa RL trực tiếp. *Suy nghĩ: Một hướng đi hứa hẹn cho việc áp dụng LLM vào các quyết định động, có rủi ro.*
    *   **`2501.14677` – Consistent Memory Propagation (CMP) với hợp nhất bộ nhớ thích ứng theo vùng và Scaled DDC Loss cho video matting (MatAnyone)** – Cải thiện tính ổn định và chi tiết ở biên. *Suy nghĩ: Các kỹ thuật tinh vi để xử lý tác vụ video matting chất lượng cao.*
    *   **`2502.10391` – Critique-Based Reward Model và Dynamic Reward Scaling trong MM-DPO cho MLLM alignment** – Tăng cường khả năng diễn giải và hiệu quả của RLHF cho MLLM. *Suy nghĩ: Cải tiến quan trọng cho việc căn chỉnh MLLM với sở thích người dùng phức tạp.*
    *   **`2502.09619` – Collaborative Probing dùng phân tách ma trận để giảm chi phí tạo ProbeLog descriptor cho tìm kiếm mô hình** – Giải pháp thực tế cho việc lập chỉ mục kho mô hình lớn. *Suy nghĩ: Giúp việc tìm kiếm và tái sử dụng mô hình trở nên khả thi hơn.*
    *   **`2502.01534` – Định nghĩa và thước đo "Preference Leakage Score" (PLS) cho LLM-làm-giám-khảo** – Xác định một dạng nhiễm bẩn mới và quan trọng trong đánh giá LLM. *Suy nghĩ: Nâng cao nhận thức về độ tin cậy của các phương pháp đánh giá dựa trên LLM.*
    *   **`2502.13144` – Mục tiêu phụ trợ có cấu trúc và dày đặc dựa trên xác suất hành động không được chọn và ước lượng lợi thế cho RL trong lái xe tự hành (RAD)** – Giải quyết vấn đề phần thưởng thưa hiệu quả. *Suy nghĩ: Một cách thông minh để cung cấp tín hiệu học phong phú hơn cho RL trong các tác vụ phức tạp.*
    *   **`2502.11663` – Token mặt nạ liên quan đến diffusion và tái tạo mặt nạ hai nhánh không-thời gian trong MaskGWM cho world model lái xe** – Tăng cường sức mạnh tổng hợp giữa tái tạo và sinh khuếch tán. *Suy nghĩ: Một cách tiếp cận mới để học world model hiệu quả hơn.*
    *   **`2502.05795` – LayerNorm Scaling (nhân với 1/√ℓ) để giải quyết "Curse of Depth" trong LLM Pre-LN** – Một sửa đổi đơn giản nhưng hiệu quả để cải thiện huấn luyện LLM sâu. *Suy nghĩ: Có thể trở thành một tiêu chuẩn mới trong kiến trúc Transformer.*
    *   **`2502.13128` – Mục tiêu dự đoán token vocal phụ trợ trong chế độ hỗn hợp của SongGen để cải thiện độ rõ ràng vocal** – Một giải pháp đơn giản nhưng hiệu quả cho bài toán khó trong sinh nhạc có lời. *Suy nghĩ: Cải thiện đáng kể chất lượng của một thành phần quan trọng trong bài hát.*

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Hiểu biết sâu và Suy luận Thực sự:**
        *   Nhiều LLM vẫn gặp khó khăn với suy luận phức tạp, dễ mắc lỗi logic hoặc "vẹt ngẫu nhiên" (PHYSICO `2502.08946`). Cần các phương pháp đánh giá sâu hơn và các kiến trúc/kỹ thuật huấn luyện cải thiện khả năng hiểu thực sự thay vì chỉ khớp mẫu.
        *   Khả năng khái quát hóa của các kỹ năng suy luận (ví dụ: từ logic K&K sang toán OOD trong Logic-RL `2502.14768`) còn hạn chế và cần được nghiên cứu thêm.
    *   **Hiệu quả Dữ liệu và Tính toán:**
        *   Mặc dù có các phương pháp như LIMO (`2502.03387`) hay s1K (`2501.19393`) cho thấy tiềm năng học từ ít dữ liệu, việc tạo ra dữ liệu chất lượng cao này vẫn tốn công sức. Cần các phương pháp tự động hơn để tạo "cognitive templates".
        *   Chi phí huấn luyện và suy luận của các mô hình lớn (đặc biệt là đa phương thức và video) vẫn là rào cản lớn. Cần thêm nghiên cứu về nén mô hình (QuEST `2502.05003`), kiến trúc hiệu quả (MoM `2502.11089`, mmMamba `2502.13145`), và suy luận hiệu quả (RSD `2501.19324`, SpargeAttn `2502.18137`, RAS `2502.10389`).
    *   **Độ tin cậy và An toàn của AI:**
        *   "Overthinking" trong LRM agentic (`2502.08235`) và "preference leakage" trong LLM-as-a-judge (`2502.01534`) là những vấn đề mới nổi cần giải pháp.
        *   Việc tạo trích dẫn đáng tin cậy (SelfCite `2502.09604`) và đảm bảo tính trung thực nói chung vẫn là thách thức.
        *   Cần các benchmark động và toàn diện hơn cho tính đáng tin cậy (TrustGen `2502.14296`).
    *   **Khả năng Đa phương thức và Tích hợp:**
        *   Căn chỉnh (alignment) giữa các phương thức (ví dụ: thị giác-ngôn ngữ trong ALIGN `2502.01341`, tiếng nói-văn bản trong Soundwave `2502.12900`) vẫn là một lĩnh vực nghiên cứu tích cực.
        *   Việc tạo ra các mô hình đa phương thức thực sự "bản địa" (native) như mmMamba (`2502.13145`) thay vì ghép nối các bộ mã hóa riêng biệt là một hướng đi hứa hẹn.
        *   Hiểu và sinh các tương tác không-thời gian phức tạp trong video (CineMaster `2502.08639`, VideoGrain `2502.17258`, Phantom `2502.11079`) và 3D (TripoSG `2502.06608`, AuraFusion360 `2502.05176`) còn nhiều dư địa cải thiện.
    *   **AI cho Khoa học và các Miền Chuyên biệt:**
        *   AI co-scientist (`2502.18864`), AlphaGeometry2 (`2502.03544`), Mol-LLaMA (`2502.13449`), Fino1 (`2502.08127`) cho thấy tiềm năng lớn, nhưng việc áp dụng AI vào các lĩnh vực khoa học và chuyên ngành sâu hơn đòi hỏi dữ liệu chuyên biệt chất lượng cao, khả năng suy luận chuyên sâu và các phương pháp đánh giá phù hợp.
        *   Cần thêm nghiên cứu về RL cho các bài toán khoa học phức tạp như QEC (`2502.14372`).
    *   **Tác tử AI (AI Agents):**
        *   Phát triển các tác tử có khả năng lập kế hoạch dài hạn, tương tác hiệu quả với môi trường và học hỏi từ kinh nghiệm (MLGym `2502.14499`, Magma `2502.13130`, EMBODIEDBENCH `2502.09560`).
        *   Giao tiếp đa tác tử hiệu quả và có ý nghĩa (Among Us paper `2502.06060`).
    *   **Lý thuyết và Nền tảng AI:**
        *   Hiểu rõ hơn về các cơ chế bên trong của LLM (LLM-Microscope `2502.15007`, SAE flow graphs `2502.03032`).
        *   Phát triển các định luật co giãn tổng quát hơn (Distillation scaling law `2502.08606`).
        *   Nghiên cứu các kiến trúc thay thế Transformer (RDLM `2502.11564`, LLaDA `2502.09992`).
    *   **Ngôn ngữ ít tài nguyên và Đa dạng văn hóa:**
        *   Phát triển mô hình và benchmark cho các ngôn ngữ ít tài nguyên như tiếng Hy Lạp (Plutus-ben `2502.18772`) và cải thiện khả năng chuyển giao kiến thức xuyên ngôn ngữ (X-FaKT `2502.17955`).

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Neuro-Symbolic Cognitive Architecture for LLMs (NS-Cog)**
    *   **Motivation:** LLMs hiện tại mạnh về khớp mẫu nhưng yếu về suy luận logic trừu tượng và hiểu biết nhân quả sâu sắc (như PHYSICO `2502.08946` chỉ ra). AlphaGeometry2 (`2502.03544`) cho thấy sức mạnh của việc kết hợp neuro-symbolic.
    *   **Key novelty:** Tích hợp một module suy luận tượng trưng (symbolic reasoning engine) có thể học được (differentiable) vào kiến trúc LLM. LLM đóng vai trò chuyển đổi ngôn ngữ tự nhiên thành biểu diễn tượng trưng và ngược lại, cũng như đề xuất các heuristic cho module tượng trưng. Module tượng trưng thực hiện suy luận logic chặt chẽ.
    *   **Approach:**
        1.  Thiết kế một ngôn ngữ tượng trưng trung gian (intermediate symbolic language - ISL) đủ mạnh để biểu diễn kiến thức và các bước suy luận.
        2.  Huấn luyện LLM (ví dụ, một NeoBERT `2502.19587` được tinh chỉnh) để dịch vấn đề sang ISL và diễn giải kết quả từ ISL sang ngôn ngữ tự nhiên.
        3.  Phát triển một bộ giải (solver) tượng trưng có thể học được, có thể là một Graph Neural Network hoạt động trên đồ thị kiến thức được biểu diễn bằng ISL, hoặc một hệ thống dựa trên logic lập trình quy nạp (Inductive Logic Programming).
        4.  Huấn luyện end-to-end hoặc xen kẽ, sử dụng phần thưởng từ việc giải đúng vấn đề (như trong OREAL `2502.06781` hoặc Logic-RL `2502.14768`).
    *   **Dataset + Metrics:** Các benchmark suy luận phức tạp như MATH, GPQA, các bài toán logic (K&K), và các bộ dữ liệu yêu cầu hiểu biết sâu như PHYSICO. Metrics: Accuracy, Logical Consistency, Explainability (chất lượng của các bước suy luận tượng trưng).
    *   **Risk/Feasibility:** Cao. Thiết kế ISL và solver tượng trưng có thể học được là rất thách thức. Việc tích hợp và huấn luyện end-to-end cũng phức tạp.

    ✨ **Idea 2: Self-Improving AI Research Agents via MLGym and Meta-Learning (SIARA)**
    *   **Motivation:** MLGym (`2502.14499`) cung cấp môi trường để huấn luyện AI research agents. AI co-scientist (`2502.18864`) cho thấy tiềm năng của AI trong nghiên cứu. Cần cơ chế để agent tự cải thiện khả năng nghiên cứu.
    *   **Key novelty:** Một meta-learning framework nơi các AI research agent (được huấn luyện trong MLGym) tự đề xuất các nhiệm vụ nghiên cứu mới (trong MLGym-Bench hoặc tự tạo), thực hiện chúng, đánh giá kết quả, và sử dụng kinh nghiệm đó để cập nhật chiến lược nghiên cứu (meta-policy) của chính chúng.
    *   **Approach:**
        1.  Sử dụng một LLM (hoặc MLLM như Magma `2502.13130` nếu có tương tác đa phương thức) làm "meta-agent".
        2.  Meta-agent đề xuất các cấu hình nhiệm vụ mới trong MLGym (ví dụ: thay đổi dataset, kiến trúc mô hình cơ sở, hàm loss) hoặc các hướng nghiên cứu mới dựa trên phân tích các "gaps" từ các nghiên cứu trước (có thể được cung cấp dưới dạng văn bản).
        3.  Một tập hợp các "worker agents" (được huấn luyện bằng RL trong MLGym) cố gắng giải quyết các nhiệm vụ này.
        4.  Kết quả (thành công, thất bại, hiệu quả, tài nguyên sử dụng) được dùng để cập nhật meta-policy của meta-agent (ví dụ, bằng RL hoặc evolutionary strategies) để nó đề xuất các nhiệm vụ tốt hơn trong tương lai.
        5.  Sử dụng các kỹ thuật từ `2502.18864` (generate, debate, evolve) để meta-agent tự cải thiện các đề xuất nghiên cứu.
    *   **Dataset + Metrics:** MLGym-Bench, các bài báo khoa học (để trích xuất gaps). Metrics: Số lượng nhiệm vụ mới được giải quyết thành công, tính mới lạ của các giải pháp được tìm thấy, hiệu quả khám phá của meta-agent.
    *   **Risk/Feasibility:** Cao. Việc định nghĩa không gian nhiệm vụ và hàm phần thưởng cho meta-agent là rất khó. Cần lượng lớn tài nguyên tính toán.

    ✨ **Idea 3: Universal Video Codec using Masked Diffusion Models (UniVidCodec)**
    *   **Motivation:** Các mô hình sinh video (Step-Video-T2V `2502.10248`, Goku `2502.04896`) đang rất mạnh mẽ. Video-VAE trong Step-Video-T2V đạt tỷ lệ nén cao. LLaDA (`2502.09992`) cho thấy tiềm năng của masked diffusion cho ngôn ngữ. MaskGWM (`2502.11663`) sử dụng masked reconstruction cho world model.
    *   **Key novelty:** Một mô hình khuếch tán mặt nạ (masked diffusion model) được huấn luyện để hoạt động như một codec video phổ quát, có khả năng nén và giải nén video với chất lượng cao và tỷ lệ nén linh hoạt, đồng thời có thể học các prior về chuyển động và nội dung.
    *   **Approach:**
        1.  Kiến trúc: Dựa trên DiT hoặc một biến thể của MaskGWM.
        2.  Huấn luyện:
            *   Nhiệm vụ chính: Tái tạo các phần video bị che (masked video reconstruction) ở các tỷ lệ che khác nhau.
            *   Sử dụng token mặt nạ liên quan đến diffusion như trong MaskGWM.
            *   Mục tiêu là học một biểu diễn tiềm ẩn (latent representation) nhỏ gọn cho các phần không bị che, từ đó có thể tái tạo lại toàn bộ video.
            *   Có thể kết hợp với các kỹ thuật từ Video-VAE (`2502.10248`) để học không gian tiềm ẩn hiệu quả.
        3.  Nén: Mã hóa video bằng cách đưa qua encoder (phần của DiT) để lấy biểu diễn tiềm ẩn của các token không bị che (hoặc một tập token đại diện).
        4.  Giải nén: Sử dụng decoder (phần còn lại của DiT) và các token tiềm ẩn đã nén để tái tạo video gốc thông qua quá trình khử nhiễu/khuếch tán ngược.
    *   **Dataset + Metrics:** Các bộ dữ liệu video đa dạng (ví dụ: WebVid, InternVid). Metrics: Tỷ lệ nén, PSNR, SSIM, VMAF, FVD. So sánh với các codec truyền thống (H.265, AV1) và các codec dựa trên neural network khác.
    *   **Risk/Feasibility:** Trung bình đến Cao. Đạt được tỷ lệ nén và chất lượng cạnh tranh với codec truyền thống là rất khó. Chi phí huấn luyện mô hình khuếch tán trên video lớn.

    ✨ **Idea 4: Trust-Calibrated LLM-as-a-Judge using Preference Leakage Score (PLS) and Decomposed Reward Models (DRMs) (TC-Judge)**
    *   **Motivation:** "Preference leakage" (`2502.01534`) là một vấn đề nghiêm trọng khi dùng LLM làm giám khảo. DRMs (`2502.13131`) cung cấp cách hiểu sở thích đa dạng. Cần một phương pháp đánh giá LLM đáng tin cậy hơn.
    *   **Key novelty:** Một hệ thống LLM-as-a-judge được hiệu chỉnh để giảm thiểu preference leakage. Hệ thống này sử dụng PLS để phát hiện và định lượng thiên vị, và DRMs để hiểu các khía cạnh khác nhau của "chất lượng" phản hồi, từ đó đưa ra đánh giá khách quan hơn.
    *   **Approach:**
        1.  Xây dựng một tập hợp các LLM giám khảo đa dạng.
        2.  Với mỗi cặp (LLM giám khảo, LLM được đánh giá), tính toán PLS để ước tính mức độ thiên vị tiềm ẩn.
        3.  Huấn luyện một DRM trên dữ liệu so sánh từ nhiều nguồn (bao gồm cả người và các LLM giám khảo khác nhau) để phân rã khái niệm "chất lượng" thành các thành phần trực giao (ví dụ: tính hữu ích, tính trung thực, tính mạch lạc, tính an toàn).
        4.  Khi đánh giá một phản hồi mới, sử dụng tổ hợp các đầu ra từ DRM, có trọng số được điều chỉnh dựa trên PLS của LLM giám khảo (ví dụ: giảm trọng số của các thành phần mà LLM giám khảo có thiên vị cao).
        5.  Có thể sử dụng CAPA (`2502.04313`) để đo lường sự đồng thuận giữa các giám khảo đã được hiệu chỉnh.
    *   **Dataset + Metrics:** Các benchmark đánh giá LLM hiện có (Arena-Hard, AlpacaEval). Metrics: Độ tương quan với đánh giá của con người, tính nhất quán giữa các giám khảo đã hiệu chỉnh, khả năng phát hiện các lỗi tinh vi.
    *   **Risk/Feasibility:** Trung bình. Việc thu thập đủ dữ liệu để huấn luyện DRM và tính toán PLS một cách đáng tin cậy có thể tốn kém. Việc diễn giải các thành phần của DRM vẫn là một thách thức.

    ✨ **Idea 5 (Moon-shot): Emergent General Intelligence from Multi-Agent Social Deduction and Creation (EGISoDAC)**
    *   **Motivation:** Giao tiếp và hợp tác/cạnh tranh trong môi trường phức tạp có thể là chìa khóa cho AGI. Paper `2502.06060` cho thấy LLM có thể học giao tiếp trong game suy luận xã hội. AI co-scientist (`2502.18864`) và các agent sáng tạo (ART `2502.18364`, SongGen `2502.13128`) cho thấy khả năng tạo ra nội dung phức tạp.
    *   **Key novelty:** Một hệ sinh thái ảo quy mô lớn nơi hàng ngàn AI agent (dựa trên các LLM/MLLM/SLM tiên tiến) tương tác, giao tiếp, hợp tác và cạnh tranh trong các trò chơi suy luận xã hội (social deduction) và các nhiệm vụ sáng tạo nội dung (creative construction) mở. Mục tiêu là để trí thông minh tổng quát hơn tự nổi lên (emerge) từ các tương tác này.
    *   **Approach:**
        1.  Thiết kế một thế giới ảo (virtual world) với các quy tắc linh hoạt, cho phép nhiều loại tương tác và mục tiêu.
        2.  Các agent có thể là chuyên gia trong các lĩnh vực khác nhau (suy luận, ngôn ngữ, thị giác, âm thanh, lập trình, khoa học) dựa trên các mô hình từ bộ papers này.
        3.  Nhiệm vụ bao gồm:
            *   Các game suy luận xã hội phức tạp hơn Among Us, yêu cầu hiểu biết sâu về tâm lý, lừa dối, thuyết phục.
            *   Các dự án sáng tạo chung: cùng nhau viết truyện, làm phim (sử dụng các kỹ thuật từ CineMaster, Phantom, ART), sáng tác nhạc (SongGen), phát triển phần mềm (SWE-RL, S*), hoặc thậm chí là đề xuất và thực hiện các thí nghiệm khoa học ảo (AI co-scientist).
        4.  Sử dụng cơ chế phần thưởng nội tại (intrinsic motivation) và phần thưởng dựa trên thành công của nhóm/cá nhân.
        5.  Các agent có thể tự cải thiện thông qua RL, học từ quan sát hành vi của agent khác, và thậm chí là "tiến hóa" kiến trúc/tham số của chúng.
    *   **Dataset + Metrics:** Không có dataset cố định. Môi trường tự tạo ra dữ liệu. Metrics: Sự phức tạp của các hành vi nổi lên, khả năng giải quyết các vấn đề ngày càng khó, sự đa dạng của các "nền văn hóa" agent, khả năng chuyển giao kỹ năng sang các nhiệm vụ mới không lường trước.
    *   **Risk/Feasibility:** Cực kỳ cao (Moon-shot). Yêu cầu tài nguyên tính toán khổng lồ. Việc thiết kế môi trường và cơ chế phần thưởng đủ phong phú và cân bằng là cực kỳ khó. Nguy cơ xuất hiện các hành vi không mong muốn hoặc không thể kiểm soát. Tuy nhiên, tiềm năng đột phá về hiểu biết AGI là rất lớn.

6.  **READING_LIST**

    *   `2502.06807` – Competitive Programming with Large Reasoning Models · *Cho thấy sức mạnh của RL quy mô lớn trong việc tạo ra các LRM có khả năng suy luận vượt trội, tự học chiến lược.*
    *   `2502.03544` – AlphaGeometry2: Carl Sagan Update · *Một minh chứng ấn tượng về khả năng giải toán hình học phức tạp của AI, kết hợp neuro-symbolic hiệu quả.*
    *   `2502.10248` – Step-Video-T2V: A 30B Parameter Bilingual Text-to-Video Foundation Model · *Một mô hình T2V mã nguồn mở SOTA, với nhiều đóng góp kỹ thuật giá trị (Video-VAE, Video-DPO).*
    *   `2502.05003` – QuEST: Quantized Embedding Space Training for LLMs · *Đột phá trong lượng tử hóa LLM xuống mức bit rất thấp, quan trọng cho việc triển khai hiệu quả.*
    *   `2502.18864` – An AI co-scientist for accelerating materials discovery · *Một hệ thống AI agent đầy tham vọng cho khám phá khoa học, cho thấy tiềm năng thực tế của AI trong nghiên cứu.*
    *   `2502.03387` – The LIMO Hypothesis: Less-Is-More for Reasoning in Foundation Models · *Một giả thuyết thú vị và kết quả bất ngờ về việc học suy luận hiệu quả từ lượng rất nhỏ dữ liệu chất lượng cao.*
    *   `2502.05795` – Curse of Depth in Pre-LayerNorm Transformers and the LayerNorm Scaling Solution · *Xác định và giải quyết một vấn đề cơ bản trong kiến trúc Transformer, có thể ảnh hưởng đến nhiều mô hình trong tương lai.*
    *   `2502.17129` – Thus Spake Long-Context Large Language Model: A Survey on the Architectures, Infrastructures, Training, and Evaluation of LLMs with Long Context Capability · *Một bài khảo sát toàn diện và cần thiết về một lĩnh vực đang rất nóng: LLM ngữ cảnh dài.*
    *   `2502.08235` – Overthinking in Large Reasoning Model Agents · *Xác định một vấn đề hành vi quan trọng của LRM agentic và đề xuất cách đánh giá, mở ra hướng nghiên cứu mới về AI agent đáng tin cậy.*
    *   `2502.13130` – Magma: A Multimodal Foundation Model for AI Agents · *Đề xuất các kỹ thuật tiền huấn luyện SoM và ToM sáng tạo để xây dựng AI agent đa phương thức tổng quát.*

7.  **META_REFLECTION**

    Tập hợp các bài báo tháng 02/2025 cho thấy một bức tranh sôi động và đa dạng của nghiên cứu AI, với một số xu hướng nổi bật:

    1.  **Sự trỗi dậy của Suy luận Nâng cao và Tác tử AI:** Một lượng lớn các bài báo tập trung vào việc cải thiện khả năng suy luận của LLM/MLLM, từ suy luận toán học (AlphaGeometry2, OREAL, LIMO), logic (Logic-RL), lập trình (SWE-RL, S*, o1/o3), đến các tác vụ agentic phức tạp hơn (AI co-scientist, Magma, MLGym, EMBODIEDBENCH). Điều này cho thấy cộng đồng đang hướng tới việc xây dựng các AI không chỉ "biết" mà còn "hiểu" và "hành động" một cách thông minh. Tuy nhiên, các vấn đề như "overthinking" (`2502.08235`) và sự cần thiết của dữ liệu chất lượng cao cho suy luận (LIMO `2502.03387`, s1K `2501.19393`) vẫn là thách thức.

    2.  **Đa phương thức ngày càng phức tạp và tích hợp sâu:** Các mô hình đa phương thức không còn dừng lại ở việc hiểu ảnh-văn bản đơn giản. Chúng đang tiến tới xử lý video với nhận thức 3D và điều khiển chi tiết (CineMaster `2502.08639`, VideoGrain `2502.17258`), sinh 3D từ ảnh (TripoSG `2502.06608`), hiểu và sinh âm thanh/nhạc/tiếng nói (Soundwave `2502.12900`, Slam `2502.15814`, SongGen `2502.13128`, Audio-FLAN `2502.16584`). Việc căn chỉnh và tích hợp các phương thức ở mức độ sâu hơn (ALIGN `2502.01341`, mmMamba `2502.13145`) là một hướng đi quan trọng.

    3.  **Hiệu quả Tính toán và Dữ liệu là Ưu tiên hàng đầu:** Với sự gia tăng về quy mô và độ phức tạp của mô hình, các nghiên cứu về hiệu quả trở nên cực kỳ quan trọng. Điều này thể hiện qua các công trình về LLM ngữ cảnh dài (InfiniteHiP `2502.08910`, LongRoPE2 `2502.20082`), nén mô hình (QuEST `2502.05003`, Skrr `2502.08690`), suy luận hiệu quả (RSD `2501.19324`, SpargeAttn `2502.18137`), kiến trúc tuyến tính (MoM `2502.11089`), và các chiến lược huấn luyện hiệu quả dữ liệu (FineMath `2502.02737`, LIMO `2502.03387`, Slam `2502.15814`). Định luật co giãn chưng cất (`2502.08606`) cũng góp phần vào việc tối ưu hóa quy trình huấn luyện.

    4.  **Học tăng cường (RL) tiếp tục mở rộng phạm vi ứng dụng:** RL không chỉ giới hạn trong game hay robot mà còn được áp dụng mạnh mẽ để cải thiện LLM (OREAL `2502.06781`, PRIME `2502.01456`, Logic-RL `2502.14768`, o1/o3 `2502.06807`), kỹ thuật phần mềm (SWE-RL `2502.18449`), tài chính (FLAG-TRADER `2502.11433`), và thậm chí cả tối ưu hóa mã QEC (`2502.14372`). Các phương pháp RL đang trở nên tinh vi hơn với các thiết kế phần thưởng và cơ chế cập nhật chuyên biệt.

    5.  **Đánh giá và Độ tin cậy ngày càng được chú trọng:** Sự phát triển nhanh chóng của các mô hình đòi hỏi các phương pháp đánh giá nghiêm ngặt và đáng tin cậy hơn. Nhiều benchmark mới được đề xuất cho các năng lực chuyên biệt (PHYSICO `2502.08946`, FailSafeQA `2502.06329`, SuperGPQA `2502.14739`, SWE-Lancer `2502.12115`, TextAtlasEval `2502.07870`, ZeroBench `2502.09696`, EMBODIEDBENCH `2502.09560`, Plutus-ben `2502.18772`, X-FaKT `2502.17955`). Các vấn đề về độ tin cậy như "preference leakage" (`2502.01534`) và tính diễn giải (LLM-Microscope `2502.15007`, SAE flow graphs `2502.03032`, ConceptAttention `2502.04320`) cũng được quan tâm. Nền tảng đánh giá động như TrustGen (`2502.14296`) là một hướng đi cần thiết.

    6.  **Sự giao thoa giữa các Kiến trúc và Kỹ thuật:** Các ý tưởng từ các loại mô hình khác nhau đang được vay mượn và kết hợp. Ví dụ, kiến trúc Transformer được áp dụng cho khuếch tán (DiT), các kỹ thuật từ LLM được dùng cho mô hình nói/âm thanh, và các mô hình khuếch tán được dùng cho nhận thức thị giác (DICEPTION `2502.17157`). Các mô hình lai (ví dụ: mmMamba-hybrid `2502.13145`) cũng xuất hiện.

Nhìn chung, tháng 02/2025 cho thấy một sự trưởng thành nhất định trong nghiên cứu AI, với việc giải quyết các vấn đề ngày càng phức tạp, tập trung vào hiệu quả, độ tin cậy và các ứng dụng thực tiễn có ý nghĩa, đồng thời không ngừng khám phá các giới hạn mới của trí tuệ nhân tạo.
