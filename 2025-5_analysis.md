1.  **TOPIC_TREE**

    *   **Large Language Models (LLMs) & Reasoning Language Models (RLMs)**
        *   **Reasoning Enhancement & Optimization**
            *   **Chain-of-Thought (CoT) & Reasoning Processes**
                *   2505.03318 | UNIFIED REWARD-THINK: Mô hình thưởng đa phương thức dựa trên CoT dài để cải thiện suy luận hiểu và tạo sinh hình ảnh/video.
                *   2505.02387 | RM-R1 (REASRMS): Mô hình hóa phần thưởng như tác vụ suy luận với Chain-of-Rubrics để tăng diễn giải và hiệu suất.
                *   2505.00703 | T2I-R1 (BiCoT-GRPO): Sinh ảnh với CoT hai cấp độ (ngữ nghĩa và token) được tối ưu đồng thời bằng RL để cải thiện hiểu prompt phức tạp.
                *   2504.20752 | Grokking for Multi-hop Reasoning: Tăng cường dữ liệu có chủ đích để kích hoạt grokking cho suy luận đa bước trên dữ liệu văn bản thực tế.
                *   2504.20708 | Subthought Analysis: Phân tích các bước suy luận trung gian ("subthoughts") và tổng hợp câu trả lời bằng "mode" để tăng độ tin cậy.
                *   2504.21659 | AdaR1: Framework tối ưu hóa suy luận thích ứng, kết hợp hợp nhất mô hình và huấn luyện ưu tiên hai cấp độ để chọn giữa Long-CoT và Short-CoT.
                *   2505.02391 | GVM-RAFT: Phân bổ ngân sách lấy mẫu động cho từng prompt trong huấn luyện CoT để tối thiểu hóa phương sai gradient, cải thiện hội tụ.
            *   **Mathematical & Formal Reasoning**
                *   2504.21233 | SLM Math Reasoning Enhancement: Quy trình huấn luyện 4 giai đoạn (chưng cất CoT, Rollout DPO, RLVR) với các cải tiến ổn định RL cho SLM giải toán.
                *   2504.21318 | Phi-4-reasoning/plus: Huấn luyện SFT với "teachable prompts" và RL dựa trên kết quả để tăng cường suy luận toán học, khoa học, coding cho SLM.
                *   2505.02735 | Semantic Verification for Formal Math: Quy trình xác minh ngữ nghĩa đa LLM và lọc bác bỏ dựa trên phủ định để xây dựng benchmark toán học hình thức (FormalMATH).
                *   2505.04528 | FPS/D-FPS: Framework giải quyết vấn đề có thể xác minh ở cấp độ quy trình bằng cách sử dụng môi trường chứng minh định lý hình thức và đánh giá RPE.
            *   **Knowledge Augmentation & Retrieval**
                *   2505.00023 | CORG: Framework tổ chức ngữ cảnh phức tạp (phân biệt, mơ hồ, phản thực tế, trùng lặp) thành các nhóm để cải thiện QA đa tài liệu.
                *   2505.04253 | External Features for Adaptive RAG: Hệ thống truy xuất thích ứng độc lập với LLM, sử dụng bộ phân loại dựa trên 27 đặc trưng thông tin bên ngoài (bao gồm "Knowledgability" được tính trước) để quyết định truy xuất.
        *   **Agentic AI & Tool Integration**
            *   2504.21776 | WebThinker: Agent nghiên cứu sâu cho LRM tự chủ tìm kiếm, khám phá web và soạn thảo báo cáo trong một quy trình suy luận liên tục.
            *   2505.01441 | ARTIST: Framework RL tác tử hợp nhất lý luận, tích hợp công cụ động và RL (GRPO) cho LLM, với chiến lược che mất mát cho đầu ra công cụ.
            *   2505.04588 | ZEROSEARCH: Framework RL huấn luyện khả năng tìm kiếm của LLM không cần công cụ tìm kiếm thực tế, sử dụng LLM mô phỏng và curriculum rollout.
            *   2505.00234 | Traj-Bootstrap (+DB-Selection, +Exemplar-Selection): Các phương pháp tự khởi động và chọn lọc để xây dựng cơ sở dữ liệu ví dụ trong ngữ cảnh từ quỹ đạo thành công của agent.
            *   2504.20406 | Offline Skill Framework for Software Automation: Khám phá và tạo bộ kỹ năng phần mềm (script) đã xác minh ngoại tuyến, sử dụng GNN để dự đoán tương hợp API và LLM để tạo/tinh chỉnh script.
            *   2504.19394 | RocketBench & RL for Rocket Design: Nền tảng đánh giá LLM cho thiết kế tên lửa và tinh chỉnh LLM bằng RL (GRPO) để vượt trội chuyên gia.
        *   **Model Efficiency & Optimization**
            *   2505.02819 | ReplaceMe: Phương pháp tỉa thưa độ sâu không cần huấn luyện, thay thế khối transformer bằng phép biến đổi tuyến tính ước tính.
            *   2505.02922 | RetroInfer: Hệ thống suy luận tối ưu hóa truy xuất cho LLM ngữ cảnh dài, coi KV cache như hệ thống lưu trữ vector với wave index và xấp xỉ attention ba phần.
            *   2505.03005 | RADLADS (RAD-RWKV6/7): Quy trình chuyển đổi nhanh chóng transformer (softmax attention) sang linear attention (RAD-RWKV) với lượng dữ liệu rất nhỏ.
            *   2504.20966 | Softpick Attention: Hàm chuẩn hóa chỉnh lưu thay thế softmax trong attention, loại bỏ attention sink và massive activations, tương thích FlashAttention.
            *   2505.00358 | R&B (Regroup & Balance): Framework tối ưu hỗn hợp dữ liệu huấn luyện bằng cách tái phân cụm ngữ nghĩa và cập nhật động tỷ lệ trộn miền dựa trên ma trận Gram gradient.
            *   2505.02222 | Telescoping for muP: Thuật toán "dịch chuyển kính thiên văn" để tinh chỉnh lưới tìm kiếm siêu tham số cho muP, giải quyết sai số ước lượng.
        *   **Pre-training, Fine-tuning & Alignment**
            *   2505.03052 | SLUNG (Masked/Unlikelihood): Phương pháp tiền huấn luyện cho phép mô hình hiểu dữ liệu rủi ro cao mà không tạo ra nó, bằng cách điều chỉnh hàm mất mát có chọn lọc.
            *   2504.21039 | Foundation-Sec-8B: Huấn luyện tiếp tục Llama 3.1-8B trên kho dữ liệu an ninh mạng mới (5.1 tỷ token) được tạo bằng quy trình lọc relevancy chuyên biệt.
            *   2504.21635 | Sadeed: Tinh chỉnh SLM decoder-only (Kuwain 1.5B) cho phục hồi dấu phụ tiếng Ả Rập, với quy trình tiền xử lý dữ liệu nghiêm ngặt.
        *   **Evaluation & Interpretability**
            *   2505.00662 | DeepCritic: Framework hai giai đoạn (SFT + RL) huấn luyện LLM thực hiện phê bình sâu sắc cho từng bước giải toán.
            *   2504.21117 | Inversion Learning for Evaluation Prompts: Tự động tạo prompt đánh giá hiệu quả, đặc thù cho LLM bằng học nghịch đảo và chưng cất dữ liệu nghịch đảo.
            *   2505.02311 | AttenHScore & Dynamic Invocation: Thước đo phát hiện ảo giác dựa trên xác suất token và attention, cùng ngưỡng động để invocation LLM và tái tổ chức kiến thức RAG.
            *   2505.03368 | Geospatial Mechanistic Interpretability: Phân tích không gian (tự tương quan) trên biểu diễn LLM của tên địa danh để hiểu cách LLM xử lý thông tin địa lý.
            *   2505.02130 | LLM Attention on Graphs Analysis: Phân tích thực nghiệm cơ chế chú ý của LLM trên dữ liệu đồ thị, phát hiện hạn chế và hiện tượng "Skewed Line Sink".
    *   **Multimodal AI**
        *   **Vision-Language Models (VLMs) & Multimodal Language Models (MLLMs)**
            *   **Unified Understanding & Generation**
                *   2505.03318 | UNIFIED REWARD-THINK: Mô hình thưởng đa phương thức dựa trên CoT dài cho hiểu và tạo sinh hình ảnh/video.
                *   2505.02471 | Ming-Lite-Uni: Bộ sinh ảnh hợp nhất MLLM cố định với mô hình khuếch tán tinh chỉnh, sử dụng token học được đa tỷ lệ và căn chỉnh biểu diễn đa tỷ lệ.
            *   **Grounded Interpretation & Compositionality**
                *   2504.21336 | UniBiomed: Kiến trúc MLLM+SAM cho diễn giải hình ảnh y sinh có cơ sở, tự động tạo prompt phân đoạn và học tương tác.
                *   2504.21850 | COMPACT: Quy trình tạo dữ liệu tinh chỉnh hợp thành thị giác có kiểm soát độ phức tạp để cải thiện khả năng kết hợp kỹ năng của MLLM.
                *   2505.03821 | LEGO VPT Benchmark: Bộ tác vụ LEGO và câu hỏi chẩn đoán để đánh giá visual perspective taking của VLM.
            *   **Specialized Applications**
                *   2505.02872 | DalEye-LLaVA/Llama: Mô hình sinh đa phương thức/LLM sử dụng chuyển động mắt (mã hóa chuyên biệt hoặc biểu diễn văn bản) để giải mã mục tiêu đọc.
                *   2505.03735 | SoccerAgent: Hệ thống đa tác tử hiểu bóng đá, tích hợp công cụ chuyên biệt và cơ sở tri thức SoccerWiki.
        *   **Reward Modeling & Alignment for Multimodal**
            *   2505.02835 | StableReinforce (R1-Reward): Thuật toán RL ổn định (Pre-CLIP, Advantage Filter, Consistency Reward) cho huấn luyện mô hình phần thưởng đa phương thức.
        *   **Speech & Voice AI**
            *   2505.02707 | Voila: Voice-Language Foundation Model với kiến trúc Transformer phân cấp, Voila-Tokenizer và phiên bản tự chủ full-duplex.
            *   2505.02625 | LLaMA-Omni 2: SpeechLM mô-đun (Qwen2.5 + Whisper + AR decoder) với Gate Fusion và streaming "Read-R-Write-W" cho tương tác giọng nói thời gian thực.
            *   2505.03739 | VITA-Audio: Mô hình lời nói lớn đầu cuối tạo token âm thanh ngay trong lượt LLM đầu tiên nhờ mô-đun Multiple Cross-modal Token Prediction (MCTP).
            *   2504.18715 | Spatial Speech Translation: Hệ thống dịch nói không gian, đồng thời, biểu cảm, thời gian thực trên chip di động, với tách nguồn, tinh chỉnh chống nhiễu và tái tạo âm thanh hai tai.
    *   **Computer Vision (CV)**
        *   **Image/Video Generation & Editing**
            *   **Text-to-Image/Video Synthesis**
                *   2505.00703 | T2I-R1 (BiCoT-GRPO): Sinh ảnh từ văn bản với CoT hai cấp độ và RL để cải thiện hiểu prompt phức tạp.
                *   2504.21650 | HoloTime (Panoramic Animator): Mô hình khuếch tán hai giai đoạn chuyển ảnh toàn cảnh tĩnh thành video động, với kỹ thuật tuần hoàn toàn cảnh.
            *   **Controllable Generation & Editing**
                *   2504.20438 | PixelHacker (LCG): Image inpainting dựa trên diffusion với Latent Categories Guidance (tiền cảnh/hậu cảnh) để cải thiện nhất quán.
                *   2505.01079 | Iterative Multi-Object Editing: Framework không cần huấn luyện với Layer-wise Memory, Background Consistency Guidance và Multi-Query Disentangled Cross-Attention.
                *   2505.02370 | SuperEdit: Chỉnh sửa hướng dẫn chỉnh sửa ảnh bằng VLM dựa trên diffusion priors và học tương phản với triplet loss để cải thiện độ chính xác.
                *   2505.02823 | MUSAR: Tùy biến đa chủ thể từ dữ liệu đơn chủ thể bằng Debiased Diptych Learning và Dynamic/Static Attention Routing.
                *   2505.03730 | FlexiAct: Truyền tải hành động linh hoạt từ video tham chiếu sang chủ thể tùy ý với RefAdapter và Frequency-aware Action Extraction.
                *   2504.21855 | ReVision (PPPM): Framework cải thiện sinh video bằng cách tích hợp và tối ưu hóa kiến thức vật lý 3D từ video được sinh ra, sử dụng Parameterized Physical Prior Model.
                *   2505.04512 | HunyuanCustom: Framework sinh video tùy chỉnh đa phương thức (ảnh, âm thanh, video, văn bản) với các mô-đun hợp nhất, tăng cường ID và tích hợp điều kiện chuyên biệt.
                *   2505.00497 | KeySync: Đồng bộ khẩu hình độ phân giải cao hai giai đoạn (keyframe, nội suy) với chiến lược mặt nạ chống rò rỉ và xử lý che khuất.
        *   **3D Vision & Scene Understanding**
            *   2505.02005 | Switch-NeRF++ (HMoHE): NeRF quy mô lớn với mạng gating dựa trên hash và các chuyên gia hash không đồng nhất, tối ưu hóa điều phối điểm.
            *   2504.21650 | Panoramic Space-Time Reconstruction: Tái tạo cảnh 4D từ video toàn cảnh với ước tính độ sâu không-thời gian mới lạ.
            *   2505.04622 | PrimitiveAnything: Trừu tượng hóa hình dạng 3D thành sinh lắp ráp khối cơ bản tự hồi quy với tham số hóa không mơ hồ và transformer có điều kiện hình dạng.
            *   2505.02836 | Scenethesis: Framework agentic không cần huấn luyện tạo cảnh 3D tương tác thực tế từ văn bản, tích hợp LLM, VFM, tối ưu hóa vật lý (SDF) và đánh giá cảnh.
        *   **Video Understanding**
            *   2505.01583 | TEMPURA & VER Dataset: Framework huấn luyện hai giai đoạn (Masked Event Prediction Reasoning, Video Segmentation/Dense Captioning) và bộ dữ liệu VER cho hiểu video theo thời gian.
        *   **Medical Image Analysis**
            *   2504.21336 | UniBiomed: Kiến trúc MLLM+SAM cho diễn giải hình ảnh y sinh có cơ sở, tự động tạo prompt phân đoạn và học tương tác.
            *   2505.03538 | RAIL (DFS, CAL): Framework học bán giám sát dual-group dual-student Mean Teacher cho phân đoạn răng 3D, với Disagreement-Focused Supervision và Confidence-Aware Learning.
        *   **RGB-Event Fusion**
            *   2505.01548 | BRENet (MET, BFAM, TFM): Kiến trúc phân đoạn ngữ nghĩa RGB-Event với Motion-enhanced Event Tensor, Bidirectional Flow Aggregation Module và Temporal Fusion Module.
        *   **Multi-Object Tracking**
            *   2505.00534 | Urban Traffic Monitoring Framework: Framework 4 bước cho giám sát giao thông đô thị, với adaptive aggregation loss và inter-class NMS.
    *   **Reinforcement Learning (RL)**
        *   **RL Algorithms & Enhancements**
            *   2505.03335 | Absolute Zero (AZR): Mô hình RLVR mới, LLM tự đề xuất nhiệm vụ suy luận mã nguồn (Deduction, Abduction, Induction) và tự cải thiện không cần dữ liệu ngoài.
            *   2505.02156 | AML (AMPO): Framework Adaptive Mode Learning cho tác nhân xã hội thực hiện Long-CoT thích ứng qua các chế độ tư duy, tối ưu bằng Adaptive Mode Policy Optimization.
            *   2505.02835 | StableReinforce (R1-Reward): Thuật toán RL ổn định (Pre-CLIP, Advantage Filter, Consistency Reward) cho huấn luyện mô hình phần thưởng đa phương thức.
        *   **RL Applications**
            *   2505.01441 | ARTIST: Framework RL tác tử cho LLM sử dụng công cụ động.
            *   2504.19394 | RocketBench & RL for Rocket Design: Tinh chỉnh LLM bằng RL (GRPO) cho thiết kế tên lửa.
            *   2505.02094 | SkillMimic-V2 (STG, STF, ATS, HE): Học kỹ năng tương tác từ minh họa thưa thớt/nhiễu bằng RLID với Đồ thị Quỹ đạo Khâu nối, Trường Chuyển tiếp Trạng thái, Lấy mẫu Thích ứng và Bộ mã hóa Lịch sử.
    *   **Robotics**
        *   2505.00562 | TeLoGraF: Kiến trúc GNN + flow matching để sinh quỹ đạo thỏa mãn đặc tả Signal Temporal Logic (STL) tổng quát.
        *   2504.18904 | ROBOVERSE (METASIM): Hạ tầng mô phỏng robot ba lớp thống nhất các trình mô phỏng, quy trình tạo tác vụ AI và quy trình Real-to-Sim.
        *   2505.03912 | OpenHelix: Phân tích thực nghiệm VLA hệ thống kép (MLLM + policy cấp thấp) cho điều khiển robot.
    *   **Software Engineering & Automation**
        *   2504.21798 | SWE-smith: Quy trình tạo dữ liệu huấn luyện quy mô lớn cho tác nhân kỹ thuật phần mềm, tập trung tạo môi trường thực thi trước rồi tổng hợp lỗi.
        *   2504.20406 | Offline Skill Framework for Software Automation: Khám phá và tạo bộ kỹ năng phần mềm (script) đã xác minh ngoại tuyến, sử dụng GNN để dự đoán tương hợp API.
    *   **Benchmarking, Datasets & Evaluation**
        *   **LLM & Reasoning Evaluation**
            *   2505.02735 | FormalMATH Benchmark Construction: Xây dựng benchmark toán học hình thức với quy trình xác minh ngữ nghĩa đa LLM và lọc bác bỏ.
            *   2505.04110 | Alpha Excel Benchmark: Bộ tiêu chuẩn đánh giá LLM dựa trên thử thách Financial Modeling World Cup, với phương pháp chuyển đổi và cơ sở hạ tầng đánh giá.
        *   **Multimodal & Vision Evaluation**
            *   2505.01490 | WorldGenBench & Knowledge Checklist Score: Benchmark đánh giá tích hợp kiến thức thế giới và suy luận ngầm của mô hình T2I, với phương pháp đánh giá có cấu trúc.
            *   2505.03821 | LEGO VPT Benchmark: Bộ tác vụ LEGO và câu hỏi chẩn đoán để đánh giá visual perspective taking của VLM.
            *   2505.01456 | UnLOK-VQA & Attack-Defense Framework: Benchmark xóa kiến thức đa phương thức khỏi MLLM và framework đánh giá tính mạnh mẽ của phương pháp xóa.
        *   **Agent & Task Automation Evaluation**
            *   2504.19394 | RocketBench: Nền tảng đánh giá LLM cho thiết kế tên lửa.
            *   2505.04364 | SwarmBench: Benchmark đánh giá khả năng phối hợp phi tập trung của LLM agent trong bầy đàn với ràng buộc cục bộ.
            *   2505.03570 | OSUniverse & COTGeminiValidator: Benchmark điều hướng GUI desktop cho AI agent, với cơ chế xác thực tự động dựa trên LLM.
            *   2505.04606 | OmniGIRL: Benchmark đa ngôn ngữ, đa phương thức, đa lĩnh vực cho giải quyết issue GitHub.
            *   2504.18373 | Auto-SLURP: Benchmark đánh giá framework đa tác tử cho trợ lý cá nhân, mở rộng SLURP với gán nhãn lại và máy chủ mô phỏng.
        *   **Specialized Datasets**
            *   2504.21650 | 360World Dataset: Bộ dữ liệu video toàn cảnh (camera cố định) kèm mô tả văn bản cho tái tạo cảnh 4D.
            *   2505.01583 | VER Dataset: Bộ dữ liệu video với chú thích sự kiện dày đặc, căn chỉnh thời gian và dữ liệu suy luận có cấu trúc cho hiểu video.
            *   2504.20605 | TF1-EN-3M: Bộ dữ liệu truyện ngụ ngôn quy mô lớn tạo bằng mở rộng prompt tổ hợp.
            *   2505.00212 | Who&When Dataset: Bộ dữ liệu bản ghi lỗi từ hệ thống đa tác tử LLM với chú thích quy lỗi.
        *   **Medical Imaging Benchmarking**
            *   2504.18983 | MediAug: Framework benchmark đánh giá phương pháp tăng cường dữ liệu mix-based cho ảnh y tế.
    *   **Statistics & Policy Learning**
        *   2504.19043 | Optimal Randomized Interventions for Conjoint Analysis: Tìm kiếm can thiệp ngẫu nhiên tối ưu (phân phối xác suất trên thuộc tính) cho lựa chọn hồ sơ ứng viên, có xét môi trường đối nghịch.
    *   **Information Systems**
        *   2504.20859 | X-Cross: Kiến trúc tích hợp động các LM chuyên biệt theo miền (LoRA) cho khuyến nghị tuần tự xuyên miền, với cơ chế tích hợp theo lớp và trọng số động.
    *   **Human-Computer Interaction (HCI)**
        *   2505.03164 | InfoVids Evaluation: Khảo sát trải nghiệm người xem với video thông tin tích hợp người thuyết trình và trực quan hóa 3D trong không gian chung.
        *   2505.03105 | Cognitio Emergens (CE) Framework: Framework lý thuyết phân tích và hướng dẫn đồng kiến tạo tri thức giữa người và AI (Cấu hình Agency, Chiều kích Tri thức, Động lực Hợp tác).
    *   **Surveys & Overviews**
        *   2505.02567 | Survey: Multimodal Unified Models (Understanding & Generation) - Phân loại kiến trúc (diffusion, autoregressive, hybrid) và chiến lược mã hóa.
        *   2504.21853 | Survey: Interactive Generative Video (IGV) - Khung hệ thống 5 mô-đun (Generation, Control, Memory, Dynamics, Intelligence) và ứng dụng.
        *   2505.01658 | Survey: LLM Inference Engines & Optimization - Tổng hợp 25 công cụ suy luận LLM và các kỹ thuật tối ưu hóa liên quan.
        *   2505.00551 | Survey: DeepSeek-R1 Replication Studies & RLM Future Directions - Tổng hợp các nỗ lực tái tạo DeepSeek-R1 và hướng phát triển RLM.
        *   2504.19056 | Survey: Generative AI for Character Animation - Khảo sát kỹ thuật AI tạo sinh cho các thành phần thiết kế nhân vật hoạt hình.
        *   2505.00174 | Meta-Analysis: AI Governance Research Trends - Phân tích xu hướng và khoảng trống nghiên cứu quản trị AI tạo sinh.
        *   2505.01043 | Survey: Low-Precision Training for LLMs - Tổng hợp và phân loại các phương pháp huấn luyện LLM độ chính xác thấp.
        *   2504.19720 | Survey: Efficient LLM Inference Serving - Hệ thống hóa các phương pháp phục vụ suy luận LLM hiệu quả.
        *   2505.03418 | Survey: LLMs for Complex Problem Solving & Knowledge Augmentation - Tổng quan phương pháp LLM giải quyết vấn đề phức tạp (suy luận đa bước, tăng cường tri thức, xác minh).
    *   **Other**
        *   2505.02214 | Qwen3 PTQ Analysis: Đánh giá có hệ thống hiệu năng Qwen3 dưới 5 phương pháp lượng tử hóa sau huấn luyện (PTQ) cổ điển.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                        | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                    |
    | :--- | :-------- | :---------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2505.03335 | RLVR, Self-Improvement, Code Reasoning, Autonomous    | Mô hình "Absolute Zero" cho LLM tự học suy luận mã nguồn hoàn toàn không cần dữ liệu ngoài, tự đề xuất và giải quyết nhiệm vụ.          | Mở ra hướng mới cho việc xây dựng các LLM có khả năng tự cải thiện liên tục và đạt được năng lực suy luận siêu phàm mà không cần giám sát.        |
    | 2    | 2505.00949 | LLM, Efficient Reasoning, NAS, RL, Dynamic Toggle     | Llama-Nemotron: Kết hợp NAS, FFN Fusion và RL quy mô lớn (FP8 generation) để tạo LLM suy luận hiệu quả với khả năng chuyển đổi chế độ linh hoạt. | Cung cấp LLM mạnh mẽ, hiệu quả cho suy luận, có thể tùy chỉnh độ chi tiết, thúc đẩy ứng dụng LLM trong các tác vụ đòi hỏi suy luận sâu.          |
    | 3    | 2504.21776 | LRM, Tool-Augmented Reasoning, Autonomous Web         | WebThinker: LRM tự chủ khám phá web sâu (tìm kiếm, điều hướng) và soạn thảo báo cáo nghiên cứu trong một luồng suy luận liên tục.         | Nâng cao đáng kể khả năng nghiên cứu và thu thập thông tin động của LRM, tiến gần hơn đến các trợ lý nghiên cứu AI tự trị thực sự.             |
    | 4    | 2505.00703 | Text-to-Image, Reasoning-enhanced, RL, Bi-level CoT   | T2I-R1: Sinh ảnh với CoT hai cấp độ (ngữ nghĩa & token) được tối ưu đồng thời bằng BiCoT-GRPO, cải thiện hiểu prompt phức tạp.          | Cải thiện đáng kể khả năng của mô hình T2I trong việc diễn giải các prompt phức tạp, tạo ra hình ảnh chính xác và có cấu trúc hơn.             |
    | 5    | 2505.02836 | Text-to-3D Scene, Agentic, Physics-aware, SDF       | Scenethesis: Framework agentic không cần huấn luyện, kết hợp LLM, VFM và tối ưu hóa vật lý dựa trên SDF để tạo cảnh 3D tương tác thực tế. | Cho phép tạo ra các cảnh 3D đa dạng, thực tế và tuân thủ vật lý từ văn bản mà không cần dữ liệu 3D, mở rộng ứng dụng trong game, VR, mô phỏng. |
    | 6    | 2504.20752 | Multi-hop Reasoning, Grokking, Data Augmentation      | Lần đầu kích hoạt grokking cho suy luận đa bước trên dữ liệu văn bản thực tế quy mô lớn bằng tăng cường dữ liệu có chủ đích (tăng ϕr). | Mở ra hướng mới để LLM học các mạch suy luận tổng quát hóa cao từ dữ liệu thưa thớt, cải thiện khả năng suy luận ẩn.                          |
    | 7    | 2504.18715 | Spatial Speech Translation, Real-time, Expressive     | Hệ thống dịch nói không gian, đồng thời, biểu cảm đầu tiên chạy thời gian thực trên chip di động, xử lý đa người nói và nhiễu.        | Đột phá trong giao tiếp đa ngôn ngữ tự nhiên, liền mạch trong môi trường thực tế, có tiềm năng ứng dụng lớn trong thiết bị đeo.                 |
    | 8    | 2505.03005 | Model Compression, Transformer to Linear Attention    | RADLADS: Chuyển đổi transformer sang linear attention (RAD-RWKV) chỉ với ~0.005% dữ liệu gốc, giữ hiệu năng cao.                       | Giúp việc triển khai LLM lớn với linear attention trở nên khả thi và hiệu quả hơn nhiều, giảm chi phí huấn luyện và suy luận.                 |
    | 9    | 2505.02922 | LLM Inference, Long-Context, KV Cache Optimization    | RetroInfer: Tối ưu KV cache cho LLM ngữ cảnh dài bằng wave index, xấp xỉ attention ba phần và ước tính có giới hạn độ chính xác.        | Cải thiện đáng kể hiệu quả suy luận cho LLM với ngữ cảnh rất dài, mở rộng khả năng ứng dụng của chúng.                                       |
    | 10   | 2505.02823 | Multi-Subject Customization, Single-Subject Data, DiT | MUSAR: Tùy biến đa chủ thể từ dữ liệu đơn chủ thể bằng Debiased Diptych Learning và Dynamic/Static Attention Routing trên DiT.      | Giải quyết vấn đề khan hiếm dữ liệu đa chủ thể, cho phép tạo ảnh tùy chỉnh nhiều đối tượng chất lượng cao một cách linh hoạt hơn.             |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2505.03318 – UNIFIED REWARD-THINK & Quy trình huấn luyện 3 giai đoạn (Cold Start, Rejection Sampling, GRPO) – Suy nghĩ:** Phương pháp huấn luyện CoT dài cho mô hình thưởng đa phương thức là mới, đặc biệt việc dùng GRPO với verifiable rewards (Format, Accuracy) để tinh chỉnh CoT là một hướng đi thông minh, có tiềm năng cải thiện độ tin cậy của reward signal.
    *   **2505.03335 – Mô hình học "Absolute Zero" & AZR (Proposer-Solver, 3 chế độ Deduction/Abduction/Induction, learnability reward) – Suy nghĩ:** Ý tưởng LLM tự đề xuất nhiệm vụ, tự giải và tự học trong môi trường xác minh (code executor) là đột phá. Hàm thưởng "learnability" rất sáng tạo để điều chỉnh độ khó. Có tiềm năng tạo ra AI suy luận cực mạnh.
    *   **2505.02707 – Voila-Tokenizer (semantic & acoustic tokens) & Kiến trúc Transformer phân cấp đa tỷ lệ & Voila-autonomous (Streaming Auto Encoder) – Suy nghĩ:** Tách token ngữ nghĩa và âm thanh là hướng đi hay để cân bằng nội dung và sắc thái. Kiến trúc streaming full-duplex rất tham vọng và cần thiết cho tương tác tự nhiên.
    *   **2504.20752 – Quy trình tăng cường dữ liệu có chủ đích tăng tỷ lệ ϕr để kích hoạt grokking cho suy luận đa bước – Suy nghĩ:** Việc chứng minh dữ liệu tổng hợp không chính xác vẫn giúp học cấu trúc quan hệ là một phát hiện quan trọng. Mở rộng grokking ra dữ liệu thực tế là bước tiến lớn.
    *   **2505.02387 – RM-R1 (REASRMS) & Chain-of-Rubrics (CoR) & Quy trình huấn luyện 2 giai đoạn (chưng cất CoT + RLVR) – Suy nghĩ:** Coi RM là tác vụ suy luận và dùng CoR để tạo giải thích/rubrics là hướng đi tốt cho diễn giải. Quy trình huấn luyện kết hợp chưng cất và RLVR có vẻ mạnh mẽ.
    *   **2504.21635 – Quy trình tiền xử lý và chuẩn hóa dữ liệu tiếng Ả Rập có dấu phụ nghiêm ngặt cho SLM – Suy nghĩ:** Đóng góp về xử lý dữ liệu rất thực tế và chi tiết, giải quyết các vấn đề đặc thù của tiếng Ả Rập, quan trọng cho các ngôn ngữ ít tài nguyên.
    *   **2505.00662 – DeepCritic & Quy trình tạo dữ liệu SFT (Initial, In-depth, Final Critique Synthesis) & RL với Monte Carlo error identification – Suy nghĩ:** Quy trình tạo phê bình đa bước, tự phản biện là mới lạ để LLM học phê bình sâu. Dùng Monte Carlo để gán nhãn RL cho phê bình cũng là một giải pháp thực tế.
    *   **2504.21776 – Deep Web Explorer (điều hướng web đa bước) & Autonomous Think-Search-and-Draft & RL DPO trực tuyến lặp lại – Suy nghĩ:** Khả năng tự chủ điều hướng web sâu và soạn thảo đồng thời là bước tiến lớn cho agent. DPO trực tuyến lặp lại để tối ưu sử dụng công cụ là một chiến lược RL hiệu quả.
    *   **2505.00703 – BiCoT-GRPO (Tối ưu đồng thời semantic-level CoT và token-level CoT) & Hệ thống phần thưởng từ tập hợp chuyên gia thị giác – Suy nghĩ:** Tách biệt và tối ưu đồng thời hai cấp độ CoT trong sinh ảnh là một ý tưởng mạnh mẽ. Hệ thống phần thưởng đa chuyên gia cần thiết cho bài toán chủ quan như sinh ảnh.
    *   **2504.20438 – Latent Categories Guidance (LCG) với embedding tiền cảnh/hậu cảnh & Quy trình tạo và gán nhãn dữ liệu LCG – Suy nghĩ:** Hướng dẫn ngữ nghĩa bằng chỉ hai embedding học được (tiền cảnh/hậu cảnh) là một cách đơn giản hóa thông minh cho inpainting, tránh phụ thuộc text prompt.
    *   **2504.21233 – Quy trình huấn luyện SLM 4 giai đoạn cho toán học & Tái sử dụng mẫu bị loại bỏ cho Rollout DPO & Cải tiến RL (tối ưu prompt, tái cân bằng thưởng, ủ nhiệt độ) – Suy nghĩ:** Quy trình toàn diện và các cải tiến RL cụ thể giải quyết vấn đề ổn định khi huấn luyện SLM giải toán là những đóng góp thực tiễn.
    *   **2505.02222 – Thuật toán "dịch chuyển kính thiên văn" (telescoping) cho muP – Suy nghĩ:** Ý tưởng điều chỉnh lưới tìm kiếm siêu tham số qua các quy mô để bù sai số ước lượng của muP là một giải pháp kỹ thuật tiềm năng, cần thêm chi tiết để đánh giá đầy đủ.
    *   **2504.21318 – Quy trình giám tuyển dữ liệu SFT với "teachable prompts" và phản hồi từ o3-mini & RL dựa trên kết quả với tập toán nhỏ chất lượng cao – Suy nghĩ:** Chiến lược chọn "teachable prompts" và dùng mô hình mạnh tạo dữ liệu suy luận chất lượng cao là cốt lõi. Kết hợp SFT chuyên biệt và RL tập trung là hướng đi hiệu quả.
    *   **2505.01441 – ARTIST framework & Xen kẽ truy vấn/đầu ra công cụ trong chuỗi lý luận & Che mất mát cho đầu ra công cụ trong GRPO & Phần thưởng tổng hợp – Suy nghĩ:** Coi sử dụng công cụ là hoạt động cốt lõi trong RL và che mất mát đầu ra công cụ là những đóng góp kỹ thuật thông minh, giúp LLM học chiến lược hiệu quả.
    *   **2505.04588 – ZEROSEARCH & Mô-đun Truy xuất Mô phỏng (Simulation LLM) & Curriculum Rollout & Che mất mát token truy xuất – Suy nghĩ:** Loại bỏ phụ thuộc công cụ tìm kiếm thực bằng LLM mô phỏng và curriculum là giải pháp sáng tạo và thực tiễn, giảm chi phí và tăng kiểm soát.
    *   **2505.00949 – NAS (Puzzle) + FFN Fusion cho kiến trúc LLM không đồng nhất & Quy trình huấn luyện 5 giai đoạn (bao gồm RL với FP8 generation) & Dynamic Reasoning Toggle – Suy nghĩ:** Kết hợp NAS và FFN fusion để tối ưu suy luận là hướng đi mạnh. RL với FP8 generation và SFT cho dynamic toggle là những cải tiến kỹ thuật đáng giá.
    *   **2505.03730 – FlexiAct & RefAdapter (LoRA cho điều kiện ảnh bất kỳ) & Frequency-aware Action Extraction (FAE) – Suy nghĩ:** RefAdapter linh hoạt và FAE trích xuất hành động dựa trên tần số trong quá trình khử nhiễu là những ý tưởng mới lạ, giải quyết tốt bài toán truyền tải hành động không đồng nhất.
    *   **2505.03005 – RADLADS & Kiến trúc RAD-RWKV6/7 (đơn giản hóa RWKV cho chuyển đổi) & Quy trình 3 bước (truyền trọng số, đồng chỉnh trạng thái ẩn, chưng cất) – Suy nghĩ:** Quy trình chuyển đổi transformer sang RNN hiệu quả với lượng dữ liệu cực nhỏ là một đột phá, giúp tạo RNN mạnh dễ dàng hơn.
    *   **2505.01079 – Layer-wise Memory & Background Consistency Guidance (BCG) & Multi-Query Disentangled Cross-Attention (MQD) – Suy nghĩ:** Các cơ chế này giải quyết tốt tính nhất quán và tích hợp đối tượng trong chỉnh sửa ảnh lặp lại nhiều đối tượng mà không cần huấn luyện lại.
    *   **2504.20966 – Hàm softpick (ReLU(exp(x)-1) / sum(|exp(x)-1|)) & Thuật toán FlashAttention-2 đã điều chỉnh – Suy nghĩ:** Softpick là một giải pháp thanh lịch và hiệu quả cho vấn đề attention sink, có tiềm năng thay thế softmax trong nhiều ứng dụng transformer.
    *   **2505.02735 – Quy trình xác minh ngữ nghĩa đa LLM (dịch ngược và so sánh) & Chiến lược lọc bác bỏ dựa trên phủ định (chứng minh ¬T) – Suy nghĩ:** Các phương pháp tự động hóa và tăng độ tin cậy cho việc xây dựng benchmark toán học hình thức, giảm thiểu nỗ lực thủ công.
    *   **2504.21850 – COMPACT & Quy trình tạo dữ liệu 4 bước (Lấy mẫu khả năng, Tạo hội thoại, Xác minh chất lượng, Tập hợp) & Kiểm soát độ phức tạp thành phần (k) – Suy nghĩ:** Chủ động kiểm soát độ phức tạp thành phần trong dữ liệu huấn luyện là một cách tiếp cận hệ thống để cải thiện khả năng hợp thành của MLLM.
    *   **2505.02819 – ReplaceMe & Ước tính biến đổi tuyến tính T (nghiệm giải tích L2 hoặc tối ưu số cosine) & Hợp nhất T vào FFN2 – Suy nghĩ:** Phương pháp tỉa thưa độ sâu không cần huấn luyện, trực quan và hiệu quả, đặc biệt giá trị khi tài nguyên hạn chế.
    *   **2504.21117 – Học nghịch đảo cho tạo prompt đánh giá & Chưng cất dữ liệu nghịch đảo (black-box) & Tạo prompt nghịch đảo one-shot – Suy nghĩ:** Ý tưởng dùng học nghịch đảo để tự động tạo prompt đánh giá đặc thù cho mô hình là mới mẻ và hiệu quả về dữ liệu.
    *   **2505.02922 – RetroInfer & Wave index (chỉ mục vector nhận biết attention) & Tripartite attention approximation & Accuracy-bounded attention estimation & Segmented clustering & Wave buffer – Suy nghĩ:** Hệ thống toàn diện và nhiều thành phần sáng tạo để tối ưu KV cache cho ngữ cảnh dài, giải quyết vấn đề bộ nhớ và tốc độ.
    *   **2505.02391 – GVM (Gradient Variance Minimization) & Phân bổ ngân sách lấy mẫu động dựa trên tỷ lệ chấp nhận và norm gradient – Suy nghĩ:** Chiến lược phân bổ ngân sách động dựa trên lý thuyết để giảm phương sai gradient là một cải tiến thông minh cho huấn luyện CoT.
    *   **2505.00234 – Traj-Bootstrap & +DB-Selection (population-based training cho DB) & +Exemplar-Selection (dựa trên empirical utility) – Suy nghĩ:** Các chiến lược tự động xây dựng và tinh chỉnh kho ví dụ trong ngữ cảnh từ kinh nghiệm của agent, đặc biệt +DB-Selection là một ý tưởng mạnh.
    *   **2504.20708 – Phân đoạn "subthoughts" dựa trên dấu hiệu chuyển tiếp & Tạo dòng suy luận mới từ subthoughts & Tổng hợp câu trả lời bằng mode – Suy nghĩ:** Khai thác các điểm dừng trung gian của một suy luận để khám phá các nhánh tiềm năng là một cách trực quan để tăng độ tin cậy.
    *   **2505.02835 – StableReinforce & Pre-CLIP & Advantage Filter & Consistency Reward & Huấn luyện theo độ khó tăng dần – Suy nghĩ:** Các kỹ thuật Pre-CLIP và Advantage Filter giải quyết trực tiếp các vấn đề bất ổn định trong PPO/Reinforce++. Consistency Reward là một bổ sung hợp lý.
    *   **2505.02156 – AML & AMPO (Adaptive Mode Policy Optimization với lợi thế mode-level và sample-level) & 4 chế độ tư duy phân cấp & Phần thưởng đa thành phần – Suy nghĩ:** Khả năng lựa chọn chế độ tư duy linh hoạt và thuật toán AMPO cải tiến từ GRPO là những đóng góp kỹ thuật rõ ràng cho tác nhân xã hội.
    *   **2505.02625 – LLaMA-Omni 2 & Gate Fusion module & Streaming "Read-R-Write-W" & Huấn luyện 2 giai đoạn – Suy nghĩ:** Gate Fusion và chiến lược streaming "Read-R-Write-W" là những cải tiến quan trọng cho SpeechLM mô-đun, đạt độ trễ thấp và giọng nói tự nhiên.
    *   **2505.02094 – STG (Stitched Trajectory Graph) & STF (State Transition Field) & ATS (Adaptive Trajectory Sampling) & HE (History Encoder) – Suy nghĩ:** Các kỹ thuật tăng cường dữ liệu và học từ minh họa nhiễu/thưa thớt này rất sáng tạo, đặc biệt STG và STF giúp tạo quỹ đạo khả thi.
    *   **2505.03735 – SoccerAgent & Kiến trúc đa tác tử (Aplan, Aexec) & Hộp công cụ 18 công cụ chuyên biệt & Lập kế hoạch chuỗi công cụ và thực thi lặp lại – Suy nghĩ:** Kiến trúc agentic chuyên biệt cho một miền phức tạp như bóng đá, với khả năng lập kế hoạch và sử dụng công cụ đa dạng là một hướng đi mạnh mẽ.
    *   **2505.03821 – Bộ tác vụ LEGO VPT & Bộ 7 câu hỏi chẩn đoán 3 cấp độ nhận thức – Suy nghĩ:** Phương pháp luận đánh giá VPT bằng LEGO có kiểm soát và bộ câu hỏi phân tầng là một đóng góp chất lượng cao, giúp hiểu rõ hơn hạn chế của VLM.
    *   **2504.21039 – Quy trình thu thập và tiền xử lý dữ liệu an ninh mạng & Bộ lọc relevancy transformer – Suy nghĩ:** Xây dựng bộ lọc relevancy tùy chỉnh dựa trên transformer cho một miền chuyên biệt là một đóng góp thực tiễn quan trọng cho chất lượng dữ liệu.
    *   **2505.02872 – DalEye-LLaVA (bộ mã hóa chuyển động mắt chuyên biệt) & DalEye-Llama (biểu diễn chuyển động mắt dạng văn bản) – Suy nghĩ:** Hai cách tiếp cận mới để tích hợp thông tin chuyển động mắt vào LLM/MLLM, mở ra hướng khai thác tín hiệu hành vi đọc.
    *   **2505.04512 – HunyuanCustom & Mô-đun hợp nhất text-image (LLaVA based) & Tăng cường ID hình ảnh (temporal concatenation) & AudioNet (căn chỉnh phân cấp audio-video) & Cơ chế tích hợp điều kiện video (patchify alignment, identity-disentangled conditioning) – Suy nghĩ:** Framework toàn diện với nhiều mô-đun chuyên biệt để tùy chỉnh video đa phương thức, đặc biệt là các cơ chế duy trì ID và tích hợp điều kiện.
    *   **2504.21650 – Panoramic Animator (2 giai đoạn, HDF, PCT) & Panoramic Space-Time Reconstruction (ước tính độ sâu không-thời gian với luồng quang và giám sát đa khung hình) & 360World Dataset – Suy nghĩ:** Các kỹ thuật tạo video toàn cảnh động và tái tạo 4D nhất quán là những đóng góp kỹ thuật đáng kể, cùng với bộ dữ liệu chuyên biệt.
    *   **2505.04364 – SwarmBench & Hệ thống vật lý tùy chỉnh & 5 tác vụ phối hợp & Ràng buộc nhận thức/giao tiếp cục bộ – Suy nghĩ:** Benchmark tập trung vào trí tuệ bầy đàn nổi bật với các ràng buộc chặt chẽ, một hướng đánh giá quan trọng cho LLM agent.
    *   **2505.01583 – TEMPURA & Masked Event Prediction Reasoning (FIM for video, LLM pseudo-data) & Video Segmentation/Dense Captioning (neo trực tiếp vào timestamp) & VER Dataset – Suy nghĩ:** Masked Event Prediction Reasoning là một đóng góp nổi bật, sử dụng LLM để tạo dữ liệu suy luận nhân quả cho video.
    *   **2504.18904 – METASIM (MetaConfig, Aligned Simulator Backends, Gym Wrapper) & Quy trình tạo tác vụ AI-assisted & Quy trình Real-to-Sim cho tài sản – Suy nghĩ:** METASIM là một hạ tầng trừu tượng hóa mạnh mẽ, có tiềm năng thống nhất các trình mô phỏng robot. Các quy trình tạo tác vụ/tài sản bằng AI rất sáng tạo.
    *   **2505.00023 – CORG (Graph Constructor, Reranker, Aggregator) & Thuật toán lặp xây dựng đồ thị quan hệ ngữ cảnh – Suy nghĩ:** Phân tách và tổ chức ngữ cảnh dựa trên mối quan hệ để xử lý QA đa tài liệu phức tạp là một cách tiếp cận có cấu trúc và hiệu quả.
    *   **2505.04528 – FPS/D-FPS framework & Định hình giải quyết vấn đề như MDP & Đánh giá RPE (Restricted Propositional Equivalence) – Suy nghĩ:** Nhúng toàn bộ quá trình giải quyết vấn đề vào môi trường chứng minh định lý hình thức là một đóng góp lý thuyết và kỹ thuật quan trọng cho AI đáng tin cậy.
    *   **2504.18715 – Joint Localization and Speech Separation (tìm kiếm dựa trên góc, TF-GridNet streaming, phân cụm loại bỏ nguồn giả) & Tinh chỉnh mô hình dịch chống chịu nhiễu & Tái tạo âm thanh hai tai với bù trễ dịch – Suy nghĩ:** Các giải pháp kỹ thuật cho tách nguồn, dịch và tái tạo không gian trong môi trường thực tế rất phức tạp và sáng tạo, đặc biệt là xử lý bù trễ.
    *   **2505.03912 – OpenHelix (Phân tích thực nghiệm VLA hệ thống kép) – Suy nghĩ:** Mặc dù là phân tích, việc cung cấp nền tảng mã nguồn mở và các khuyến nghị dựa trên thực nghiệm có hệ thống là đóng góp giá trị cho cộng đồng robot học.
    *   **2505.03739 – VITA-Audio & Multiple Cross-modal Token Prediction (MCTP) & Kiến trúc dự đoán phân tầng cho MCTP – Suy nghĩ:** MCTP là một giải pháp thông minh để đạt "zero audio token delay" trong SpeechLM, cải thiện đáng kể độ trễ tương tác.
    *   **2504.21798 – SWE-smith & 4 chiến lược tổng hợp lỗi tự động (LM Modify/Rewrite, AST, Combine Bugs, PR Mirror) & Xác thực dựa trên thực thi & Môi trường thực thi dùng chung – Suy nghĩ:** Quy trình tạo dữ liệu SWE quy mô lớn với môi trường chung và các chiến lược tổng hợp lỗi đa dạng là một cải tiến lớn về khả năng mở rộng.
    *   **2504.18983 – MediAug framework & Quy trình đánh giá thống nhất 6 phương pháp mix-based augmentation trên CNN/ViT cho ảnh y tế – Suy nghĩ:** Cung cấp một nền tảng chuẩn hóa cần thiết để so sánh công bằng các phương pháp tăng cường dữ liệu trong lĩnh vực y tế.
    *   **2505.04606 – OmniGIRL benchmark & Quy trình thu thập/xử lý dữ liệu 5 giai đoạn & Tích hợp thành phần đa phương thức (ảnh, link) – Suy nghĩ:** Benchmark đa ngôn ngữ, đa phương thức, đa lĩnh vực đầu tiên cho giải quyết issue GitHub, giải quyết hạn chế lớn của các benchmark trước.
    *   **2505.03164 – InfoVids (Body Object Model - BOM) & Phân tích trải nghiệm người xem – Suy nghĩ:** Mặc dù BOM là công cụ, việc đề xuất và phân tích trải nghiệm của một định dạng trình bày dữ liệu mới (InfoVids) là một đóng góp HCI.
    *   **2505.02836 – Scenethesis & Lập kế hoạch cảnh thô (LLM) + Tinh chỉnh thị giác (VFM) + Tối ưu hóa vật lý (SDF) + Đánh giá cảnh – Suy nghĩ:** Sự kết hợp của LLM, VFM và tối ưu hóa dựa trên SDF là một hướng đi mạnh mẽ để tạo cảnh 3D thực tế mà không cần huấn luyện.
    *   **2505.03570 – OSUniverse & COTGeminiValidator (checklist-based thinking) & Kiến trúc benchmark module hóa – Suy nghĩ:** COTGeminiValidator sử dụng LLM để xác thực tự động là một đóng góp kỹ thuật đáng chú ý cho benchmarking agent GUI.
    *   **2505.02823 – MUSAR & Debiased Diptych Learning (Static Attention Routing, Dual-branch LoRA) & Dynamic Attention Routing – Suy nghĩ:** Các cơ chế routing attention để học tùy biến đa chủ thể từ dữ liệu đơn chủ thể là rất sáng tạo và giải quyết vấn đề khan hiếm dữ liệu.
    *   **2504.21336 – UniBiomed & Tích hợp MLLM-SAM & Tự động tạo prompt phân đoạn từ MLLM & Học tương tác SAM-MLLM & Chuẩn hóa VQA y sinh – Suy nghĩ:** Tích hợp MLLM và SAM với cơ chế học tương hỗ và tự động tạo prompt là một bước tiến quan trọng cho diễn giải hình ảnh y sinh.
    *   **2504.19043 – Tìm kiếm can thiệp ngẫu nhiên tối ưu (phân phối xác suất) & Mở rộng sang môi trường đối nghịch & Điều chuẩn phân phối hồ sơ – Suy nghĩ:** Chuyển từ tìm điểm tối ưu sang phân phối tối ưu và tích hợp lý thuyết trò chơi là hướng đi mới cho phân tích conjoint.
    *   **2505.02393 – IEF-VAD & Student's-t likelihood + xấp xỉ Laplace cho trọng số hợp nhất & Cập nhật tuần tự kiểu Kalman & Tinh chỉnh lặp khử lỗi – Suy nghĩ:** Framework hợp nhất đa phương thức (ảnh-sự kiện) có cơ sở lý thuyết vững chắc, xử lý nhiễu và độ không chắc chắn hiệu quả.
    *   **2505.01548 – Motion-enhanced Event Tensor (MET) & BRENet (BFAM, TFM) – Suy nghĩ:** MET là một biểu diễn sự kiện mới, dày đặc, kết hợp luồng quang và xử lý tần số. BFAM và TFM là các mô-đun hợp nhất RGB-Event hiệu quả.
    *   **2504.20605 – Quy trình tạo dữ liệu truyện ngụ ngôn TF1-EN-3M (mở rộng prompt tổ hợp 6 yếu tố) & Quy trình đánh giá lai ghép (LLM critic + metrics) – Suy nghĩ:** Tạo dữ liệu truyện có cấu trúc quy mô lớn bằng mô hình nhỏ và quy trình đánh giá đa chiều là những đóng góp thực tiễn.
    *   **2505.00534 – Framework giám sát giao thông 4 bước & Adaptive aggregation loss & Inter-class NMS & Tạo đặc trưng Deep SORT riêng – Suy nghĩ:** Các cải tiến cụ thể cho theo dõi đa camera trong giao thông đô thị, đặc biệt là adaptive aggregation loss và inter-class NMS.
    *   **2505.01456 – UnLOK-VQA benchmark & Pipeline tạo dữ liệu với proximity levels & Attack-and-defense framework & Whitebox attack dựa trên hidden states – Suy nghĩ:** Benchmark và framework đánh giá unlearning đa phương thức rất toàn diện, đặc biệt là các mức độ gần và attack mới.
    *   **2505.00212 – Chính thức hóa quy lỗi tự động cho MAS LLM & Bộ dữ liệu Who&When & 3 phương pháp quy lỗi (All-at-once, Step-by-step, Binary search) – Suy nghĩ:** Tiên phong định nghĩa và cung cấp dữ liệu cho một lĩnh vực nghiên cứu mới và quan trọng.
    *   **2504.18373 – Auto-SLURP benchmark & Gán nhãn lại slot chi tiết & Tích hợp máy chủ mô phỏng/API & Kiến trúc workflow đa tác tử mẫu – Suy nghĩ:** Mở rộng SLURP cho đánh giá end-to-end framework đa tác tử là một đóng góp cần thiết.
    *   **2504.20859 – X-Cross & Tích hợp động LoRA adapters theo lớp & Trọng số động (dương/âm) & Tinh chỉnh/tích hợp biểu diễn có tương tác cặp miền – Suy nghĩ:** Phương pháp mới lạ cho khuyến nghị xuyên miền, tận dụng LoRA và tích hợp thông tin đa lớp một cách linh hoạt.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Reasoning & Reliability in LLMs/MLLMs:**
        *   Cải thiện khả năng suy luận đa bước phức tạp, đặc biệt là suy luận ẩn và suy luận nhân quả trong cả ngôn ngữ và đa phương thức. (Từ 2504.20752, 2505.01583, 2505.00703)
        *   Giảm thiểu "ảo giác" (hallucinations) trong LLM/MLLM, phát triển các cơ chế phát hiện và tự sửa lỗi hiệu quả hơn. (Từ 2505.02311)
        *   Nâng cao độ tin cậy và khả năng xác minh của các chuỗi suy luận, đặc biệt trong các miền đòi hỏi độ chính xác cao như toán học, khoa học, y tế. (Từ 2505.04528, 2505.02735)
        *   Phát triển các phương pháp hiệu quả để LLM học và tích hợp kiến thức thế giới sâu rộng, kiến thức miền chuyên biệt, và xử lý thông tin động, thay đổi nhanh. (Từ 2505.01490, 2504.21039, 2505.03418)
    *   **Efficiency & Scalability:**
        *   Phát triển các kiến trúc LLM/MLLM hiệu quả hơn về mặt tính toán và bộ nhớ, đặc biệt cho suy luận trên thiết bị biên hoặc với ngữ cảnh dài. (Từ 2505.02922, 2505.03005, 2504.20966, 2505.02819, 2505.01043)
        *   Tối ưu hóa quy trình huấn luyện (ví dụ: data mixing, hyperparameter tuning, low-precision training) để giảm chi phí và thời gian. (Từ 2505.00358, 2505.02222)
        *   Nghiên cứu các phương pháp tỉa thưa, lượng tử hóa mạnh mẽ hơn mà ít ảnh hưởng đến hiệu năng, đặc biệt cho các mô hình tiên tiến như Qwen3. (Từ 2505.02214, 2505.02819)
    *   **Multimodal Integration & Generation:**
        *   Cải thiện sự nhất quán, khả năng kiểm soát và chất lượng trong sinh ảnh/video đa phương thức, đặc biệt là với nhiều đối tượng, tương tác phức tạp và yêu cầu duy trì ID. (Từ 2505.02823, 2505.04512, 2504.21855, 2505.00497)
        *   Phát triển các phương pháp hợp nhất đa phương thức (ví dụ: RGB-Event, ảnh-sự kiện-âm thanh) mạnh mẽ hơn, có khả năng xử lý nhiễu và sự không đồng nhất giữa các nguồn. (Từ 2505.01548, 2505.02393)
        *   Nâng cao khả năng hiểu và tạo sinh giọng nói tự nhiên, biểu cảm, độ trễ thấp, và có nhận biết không gian trong các hệ thống tương tác. (Từ 2505.02707, 2505.02625, 2505.03739, 2504.18715)
        *   Tạo ra các biểu diễn 3D/4D từ đa phương thức một cách chính xác và hiệu quả hơn, đặc biệt cho các cảnh động và quy mô lớn. (Từ 2505.02005, 2504.21650, 2505.04622, 2505.02836)
    *   **Agentic AI & Autonomous Systems:**
        *   Nâng cao khả năng lập kế hoạch, sử dụng công cụ động, và tương tác môi trường của các agent AI. (Từ 2504.21776, 2505.01441, 2505.04588)
        *   Phát triển các agent có khả năng tự cải thiện, tự học từ kinh nghiệm và tự đề xuất nhiệm vụ. (Từ 2505.03335, 2505.00234)
        *   Nghiên cứu sâu hơn về trí tuệ bầy đàn và phối hợp phi tập trung trong các hệ thống đa tác tử LLM. (Từ 2505.04364, 2505.00212)
        *   Tạo ra các agent chuyên biệt cho các miền phức tạp (ví dụ: kỹ thuật phần mềm, thiết kế kỹ thuật, khoa học) với khả năng tự động hóa cao. (Từ 2504.21798, 2504.20406, 2504.19394)
    *   **Data-centric AI:**
        *   Phát triển các phương pháp tạo, giám tuyển và tăng cường dữ liệu huấn luyện chất lượng cao, đa dạng và có kiểm soát cho các tác vụ cụ thể (ví dụ: suy luận hợp thành, truyện ngụ ngôn, kỹ thuật phần mềm). (Từ 2504.21850, 2504.20605, 2504.21798)
        *   Nghiên cứu các kỹ thuật học từ dữ liệu nhiễu, thưa thớt hoặc không hoàn hảo, đặc biệt trong học từ minh họa. (Từ 2505.02094)
        *   Xây dựng các bộ benchmark toàn diện hơn, bao gồm các khía cạnh đa ngôn ngữ, đa phương thức, đa lĩnh vực và các tình huống thực tế phức tạp. (Từ 2505.04606, 2505.01490, 2505.03570, 2504.18373)
    *   **Trustworthy & Responsible AI:**
        *   Phát triển các phương pháp hiệu quả để kiểm soát việc tạo ra nội dung không mong muốn (độc hại, có bản quyền) trong khi vẫn duy trì khả năng hiểu của mô hình. (Từ 2505.03052)
        *   Nghiên cứu và phát triển các kỹ thuật xóa kiến thức (unlearning) mạnh mẽ và có thể kiểm chứng cho MLLM. (Từ 2505.01456)
        *   Cải thiện khả năng diễn giải và hiểu cơ chế hoạt động bên trong của LLM/MLLM, đặc biệt khi xử lý các loại dữ liệu có cấu trúc (đồ thị, không gian địa lý). (Từ 2505.03368, 2505.02130)
        *   Nghiên cứu các vấn đề quản trị AI, đặc biệt là các rủi ro trong các lĩnh vực ứng dụng nhạy cảm và tác động sau triển khai. (Từ 2505.00174)
    *   **Human-AI Collaboration:**
        *   Thiết kế các framework và mô hình tương tác hiệu quả hơn cho sự đồng kiến tạo tri thức giữa người và AI. (Từ 2505.03105)
        *   Cải thiện trải nghiệm người dùng trong các ứng dụng AI tương tác (ví dụ: trình bày dữ liệu, dịch nói). (Từ 2505.03164, 2504.18715)

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Neuro-Symbolic Absolute Zero Reasoner for Scientific Discovery**
    *   **Motivation:** Paper 2505.03335 (Absolute Zero) cho thấy tiềm năng của LLM tự học suy luận mã nguồn. Mở rộng điều này sang khám phá khoa học, nơi các định luật vật lý/hóa học có thể được biểu diễn hình thức.
    *   **Key novelty:** Kết hợp khả năng tự đề xuất nhiệm vụ của Absolute Zero với các công cụ suy luận ký hiệu (symbolic reasoners) và môi trường mô phỏng khoa học (ví dụ: mô phỏng phân tử, vật lý). LLM đề xuất giả thuyết (dưới dạng mã hoặc biểu thức hình thức), tự kiểm chứng qua mô phỏng/suy luận ký hiệu, và tự cải thiện.
    *   **Approach:**
        1.  LLM (Proposer) tạo ra các giả thuyết/thí nghiệm tiềm năng dưới dạng mã có thể thực thi trong môi trường mô phỏng khoa học hoặc các định lý/biểu thức hình thức.
        2.  Môi trường (Simulator/Symbolic Solver) thực thi/chứng minh, cung cấp kết quả có thể xác minh.
        3.  LLM (Solver) cố gắng giải thích/dự đoán kết quả.
        4.  Phần thưởng dựa trên "khả năng học" của giả thuyết (như AZR) và tính đúng đắn của giải thích/dự đoán.
        5.  Ba chế độ học: Deduction (dự đoán kết quả thí nghiệm), Abduction (tìm điều kiện ban đầu/tham số mô hình từ kết quả), Induction (tổng hợp quy luật/phương trình từ dữ liệu thí nghiệm).
    *   **Dataset + Metrics:** Bắt đầu với các bộ dữ liệu vật lý/hóa học cơ bản có thể mô phỏng (ví dụ: tương tác hạt, phản ứng đơn giản). Metrics: Tỷ lệ khám phá thành công các quy luật đã biết, khả năng dự đoán hiện tượng mới, tính hiệu quả của quá trình tự học.
    *   **Risk/Feasibility:** Cao. Rủi ro: Không gian tìm kiếm giả thuyết cực lớn, khó khăn trong việc thiết kế môi trường mô phỏng/ký hiệu đủ mạnh và linh hoạt. Khả thi: Có thể bắt đầu với các miền khoa học đơn giản hóa.

    ✨ **Idea 2: Grokking Compositional Understanding in Multimodal Models via Controlled Data Synthesis**
    *   **Motivation:** Paper 2504.20752 kích hoạt grokking cho suy luận đa bước NLP. Paper 2504.21850 (COMPACT) tạo dữ liệu có kiểm soát độ phức tạp cho MLLM. Kết hợp hai ý tưởng này để MLLM "grok" được khả năng hợp thành đa phương thức.
    *   **Key novelty:** Thiết kế một quy trình tăng cường dữ liệu đa phương thức (hình ảnh + văn bản mô tả/câu hỏi) có chủ đích để tăng tỷ lệ ϕr cho các "khả năng hợp thành" cụ thể (ví dụ: hiểu đồng thời màu sắc + vị trí + hành động). Mục tiêu là làm MLLM chuyển từ ghi nhớ các cặp (ảnh, mô tả) sang học các "mạch hợp thành" tổng quát.
    *   **Approach:**
        1.  Định nghĩa các khả năng đa phương thức nguyên tử (như COMPACT).
        2.  Tạo các sự kiện nguyên tử (ví dụ: ảnh có "quả táo đỏ", câu hỏi "màu gì?").
        3.  Tạo các sự kiện suy luận hợp thành (ví dụ: ảnh có "quả táo đỏ trên bàn", câu hỏi "vật màu đỏ ở đâu?").
        4.  Sử dụng mô hình sinh ảnh/VQA để tạo ra lượng lớn các biến thể của sự kiện nguyên tử và suy luận, đảm bảo tỷ lệ ϕr cao cho các tổ hợp khả năng.
        5.  Huấn luyện MLLM trên dữ liệu này trong thời gian dài, theo dõi hiện tượng grokking trên các tác vụ hợp thành OOD.
    *   **Dataset + Metrics:** Tạo bộ dữ liệu tổng hợp dựa trên COCO hoặc các bộ VQA khác, nhưng có kiểm soát ϕr. Metrics: Độ chính xác trên các câu hỏi hợp thành OOD, phân tích sự hình thành mạch bên trong MLLM (nếu có thể).
    *   **Risk/Feasibility:** Trung bình-Cao. Rủi ro: Khó khăn trong việc định nghĩa và kiểm soát chính xác ϕr cho các khả năng đa phương thức phức tạp. Việc tạo dữ liệu tổng hợp chất lượng cao và đa dạng là thách thức. Khả thi: Có thể bắt đầu với các tổ hợp 2-3 khả năng.

    ✨ **Idea 3: Real-time Adaptive Reasoning Toggle for Resource-Constrained Embodied Agents**
    *   **Motivation:** Paper 2505.00949 (Llama-Nemotron) có "dynamic reasoning toggle". Paper 2505.02156 (AML) cho tác nhân xã hội chọn chế độ tư duy. Áp dụng ý tưởng này cho robot/agent hiện thân hoạt động trong môi trường động với tài nguyên tính toán hạn chế.
    *   **Key novelty:** Một meta-controller học cách tự động chuyển đổi giữa các "chế độ suy luận" (ví dụ: phản ứng nhanh dựa trên policy cấp thấp, lập kế hoạch ngắn hạn bằng SLM, lập kế hoạch dài hạn bằng LLM/VLM mạnh hơn nhưng chậm hơn) cho agent hiện thân, dựa trên độ phức tạp của tình huống, mức độ không chắc chắn, và tài nguyên sẵn có (pin, băng thông).
    *   **Approach:**
        1.  Định nghĩa các chế độ suy luận với chi phí/lợi ích khác nhau.
        2.  Sử dụng RL để huấn luyện meta-controller. Trạng thái bao gồm thông tin về nhiệm vụ, môi trường, tài nguyên agent. Hành động là chọn chế độ suy luận.
        3.  Phần thưởng kết hợp hiệu suất hoàn thành nhiệm vụ, hiệu quả sử dụng tài nguyên, và độ trễ quyết định.
        4.  Có thể sử dụng các kỹ thuật từ AML (AMPO) hoặc AttenHScore (2505.02311) để meta-controller đánh giá khi nào cần "suy nghĩ sâu hơn".
    *   **Dataset + Metrics:** Các môi trường mô phỏng robot (ví dụ: ROBOVERSE 2504.18904, Isaac Gym) với các tác vụ đa dạng về độ phức tạp. Metrics: Tỷ lệ hoàn thành nhiệm vụ, thời gian hoàn thành, năng lượng tiêu thụ, số lần gọi LLM/VLM tốn kém.
    *   **Risk/Feasibility:** Trung bình. Rủi ro: Thiết kế không gian trạng thái và hàm thưởng cho meta-controller phức tạp. Đảm bảo chuyển đổi chế độ mượt mà. Khả thi: Các thành phần riêng lẻ đã có, thách thức là tích hợp và huấn luyện meta-controller.

    ✨ **Idea 4: Zero-Shot Transfer of Action Primitives via Denoising-based Action Extraction (Moon-shot)**
    *   **Motivation:** Paper 2505.03730 (FlexiAct) giới thiệu Frequency-aware Action Extraction (FAE) để trích xuất hành động từ video tham chiếu trong quá trình khử nhiễu. Ý tưởng này có thể được đẩy xa hơn.
    *   **Key novelty:** Huấn luyện một mô hình khuếch tán video cực lớn trên đa dạng các loại hành động và chủ thể. Sau đó, sử dụng một cơ chế tương tự FAE nhưng tổng quát hơn để "trích xuất" các embedding hành động nguyên tử (action primitives) từ video tham chiếu *bất kỳ* (kể cả các hành động chưa từng thấy rõ ràng trong huấn luyện) và áp dụng chúng cho một chủ thể mục tiêu mới trong một frame ảnh, mà không cần huấn luyện lại FAE cho từng video tham chiếu.
    *   **Approach:**
        1.  Huấn luyện một mô hình I2V nền tảng (như CogVideoX-I2V) trên một bộ dữ liệu video hành động cực lớn và đa dạng.
        2.  Trong quá trình khử nhiễu, thay vì FAE học được cho từng video, thiết kế một module (có thể là một mạng nơ-ron nhỏ) học cách "đọc" các đặc trưng tần số từ các lớp MMDiT của mô hình I2V nền tảng (đã được điều kiện hóa trên video tham chiếu) để tạo ra một embedding hành động tổng quát.
        3.  Embedding hành động này sau đó được sử dụng để điều khiển việc sinh video cho ảnh mục tiêu.
        4.  Tập trung vào việc học một không gian embedding hành động có tính tổng quát cao, tách biệt khỏi ngoại hình và cấu trúc cụ thể của chủ thể tham chiếu.
    *   **Dataset + Metrics:** Các bộ dữ liệu video hành động lớn (ví dụ: Kinetics, Something-Something, Ego4D). Metrics: Tính nhất quán hành động, tính tự nhiên của chuyển động được truyền tải, khả năng tổng quát hóa cho các hành động và chủ thể mới. Đánh giá định tính và định lượng (ví dụ: so sánh embedding hành động).
    *   **Risk/Feasibility:** Rất cao (Moon-shot). Rủi ro: Khó khăn trong việc học một không gian embedding hành động thực sự tổng quát và tách rời. Việc "đọc" đặc trưng tần số một cách linh hoạt là thách thức lớn. Khả thi: Có thể bắt đầu với các loại hành động giới hạn hơn.

    ✨ **Idea 5: Self-Correcting Formal Problem Solvers with Mechanistic Interpretability Feedback**
    *   **Motivation:** Paper 2505.04528 (FPS/D-FPS) giải quyết vấn đề hình thức trong FTP. Paper 2505.03368 và 2505.02130 nghiên cứu diễn giải LLM. Kết hợp chúng để LLM tự sửa lỗi trong quá trình chứng minh hình thức.
    *   **Key novelty:** Một LLM agent giải quyết bài toán trong môi trường FTP (như Lean). Khi agent gặp bế tắc hoặc tạo ra một bước chứng minh không hiệu quả, một module diễn giải (ví dụ: phân tích attention, probing) được kích hoạt để xác định "điểm yếu" trong suy luận của LLM (ví dụ: token/khái niệm nào bị hiểu sai, mối quan hệ nào bị bỏ qua). Thông tin này được dùng làm phản hồi để LLM tự điều chỉnh chiến lược chứng minh hoặc prompt của nó.
    *   **Approach:**
        1.  LLM agent tương tác với FTP để tạo các bước chứng minh.
        2.  Nếu không có tiến triển hoặc gặp lỗi, kích hoạt module diễn giải trên các trạng thái ẩn/attention của LLM liên quan đến bước chứng minh thất bại.
        3.  Module diễn giải cung cấp phản hồi có cấu trúc (ví dụ: "LLM đang chú ý quá nhiều vào giả thuyết X thay vì Y", "Khái niệm Z chưa được áp dụng đúng").
        4.  LLM sử dụng phản hồi này để điều chỉnh lại kế hoạch, thử một chiến thuật khác, hoặc yêu cầu làm rõ khái niệm.
        5.  Có thể huấn luyện LLM bằng RL để học cách sử dụng phản hồi từ module diễn giải một cách hiệu quả.
    *   **Dataset + Metrics:** Các bộ dữ liệu chứng minh định lý (ví dụ: miniF2F, ProofNet). Metrics: Tỷ lệ giải quyết thành công, số bước chứng minh, khả năng tự sửa lỗi khi gặp bế tắc.
    *   **Risk/Feasibility:** Cao. Rủi ro: Phát triển module diễn giải đủ mạnh và cung cấp phản hồi hữu ích là rất khó. LLM có thể không hiểu hoặc không sử dụng hiệu quả phản hồi đó. Khả thi: Bắt đầu với các loại lỗi chứng minh cụ thể và các kỹ thuật diễn giải đơn giản hơn.

6.  **READING_LIST**

    *   2505.03335 – Absolute Zero · Đột phá về LLM tự học suy luận mã nguồn không cần dữ liệu ngoài.
    *   2505.00949 – Llama-Nemotron · Kiến trúc LLM suy luận hiệu quả kết hợp NAS, FFN Fusion và RL, có dynamic reasoning toggle.
    *   2504.21776 – WebThinker · Agent LRM tự chủ khám phá web sâu và soạn thảo báo cáo, một bước tiến cho AI nghiên cứu.
    *   2505.00703 – T2I-R1 (BiCoT-GRPO) · Phương pháp mới cho T2I với CoT hai cấp độ, cải thiện hiểu prompt phức tạp.
    *   2504.20752 – Grokking for Multi-hop Reasoning · Lần đầu kích hoạt grokking cho suy luận đa bước trên dữ liệu văn bản thực tế.
    *   2504.18715 – Spatial Speech Translation · Hệ thống dịch nói không gian, đồng thời, biểu cảm đầu tiên chạy thời gian thực trên thiết bị di động.
    *   2505.02836 – Scenethesis · Framework agentic không cần huấn luyện tạo cảnh 3D tương tác thực tế từ văn bản, tích hợp LLM, VFM, SDF.
    *   2505.03005 – RADLADS · Quy trình chuyển đổi transformer sang linear attention cực kỳ hiệu quả về dữ liệu.
    *   2505.04528 – FPS/D-FPS · Framework giải quyết vấn đề hình thức có thể xác minh trong môi trường chứng minh định lý.
    *   2505.03052 – SLUNG · Phương pháp tiền huấn luyện độc đáo cho phép LLM hiểu mà không tạo ra dữ liệu rủi ro.

7.  **META_REFLECTION**

    *   Tập hợp các bài báo tháng 05/2025 cho thấy một số xu hướng phát triển AI nổi bật. **Thứ nhất, khả năng suy luận (reasoning)** tiếp tục là một trọng tâm lớn, không chỉ trong LLM (với các phương pháp CoT nâng cao, tự học, tối ưu hóa thích ứng, và ứng dụng vào các miền phức tạp như toán học, code, thiết kế kỹ thuật) mà còn được mở rộng sang các mô hình đa phương thức (ví dụ: mô hình thưởng CoT, sinh ảnh có CoT). Có sự chuyển dịch từ việc chỉ tạo ra kết quả sang việc tạo ra quy trình suy luận có thể diễn giải và đáng tin cậy.
    *   **Thứ hai, tính tự chủ và khả năng tác tử (agentic capabilities)** của AI đang được đẩy mạnh. Nhiều công trình tập trung vào việc xây dựng các agent có khả năng tự lập kế hoạch, sử dụng công cụ động, tương tác với môi trường (web, phần mềm, mô phỏng vật lý), tự cải thiện từ kinh nghiệm, và thậm chí tự đề xuất nhiệm vụ học tập. Điều này thể hiện tham vọng tạo ra các hệ thống AI có khả năng giải quyết vấn đề phức tạp một cách độc lập hơn.
    *   **Thứ ba, hiệu quả (efficiency)** và **khả năng mở rộng (scalability)** vẫn là những ưu tiên hàng đầu. Điều này được thể hiện qua các nghiên cứu về tối ưu hóa kiến trúc (NAS, linear attention, softpick), nén mô hình (tỉa thưa, lượng tử hóa), tối ưu hóa suy luận (KV cache), và các chiến lược huấn luyện hiệu quả hơn về dữ liệu và tính toán (data mixing, adaptive sampling, RL ổn định).
    *   **Thứ tư, tích hợp đa phương thức (multimodal integration)** ngày càng trở nên tinh vi. Các mô hình không chỉ xử lý nhiều loại dữ liệu (văn bản, ảnh, video, giọng nói, chuyển động mắt, dữ liệu 3D) mà còn tập trung vào sự tương tác sâu và có ý nghĩa giữa các phương thức, ví dụ như diễn giải hình ảnh có cơ sở, sinh video/giọng nói có kiểm soát và đồng bộ, và hiểu biết các tương tác phức tạp trong các miền như thể thao hay y tế.
    *   **Thứ năm, tầm quan trọng của dữ liệu và benchmark chất lượng cao** tiếp tục được nhấn mạnh. Nhiều công trình giới thiệu các bộ dữ liệu mới, quy trình tạo/giám tuyển dữ liệu sáng tạo, và các benchmark chuyên biệt để đánh giá các năng lực AI ngày càng phức tạp (ví dụ: suy luận ngầm, phối hợp bầy đàn, xóa kiến thức, giải quyết issue đa phương thức).
    *   **Cuối cùng, có một sự quan tâm ngày càng tăng đối với các vấn đề về độ tin cậy, an toàn và quản trị AI**, thể hiện qua các nghiên cứu về kiểm soát tạo sinh nội dung rủi ro, xóa kiến thức, diễn giải cơ chế hoạt động, và phân tích xu hướng quản trị.
Nhìn chung, lĩnh vực AI đang hướng tới việc xây dựng các hệ thống thông minh hơn, tự chủ hơn, hiệu quả hơn, có khả năng tương tác đa phương thức phong phú và đáng tin cậy hơn, đồng thời nhận thức rõ hơn về các thách thức và trách nhiệm đi kèm.
