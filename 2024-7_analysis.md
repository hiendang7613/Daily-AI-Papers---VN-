1.  **TOPIC_TREE**

    *   **Large Language Models (LLMs)**
        *   **Foundational Models & Pretraining**
            *   **Scaling & Architecture**
                *   `2407.10671` | Xây dựng dòng mô hình Qwen2 bằng cách tích hợp và cải tiến các kỹ thuật hiện có, tập trung vào kỹ thuật hệ thống và dữ liệu quy mô lớn.
                *   `2407.12327` | Đề xuất phương pháp huấn luyện và kiến trúc cho Mô hình Ngôn ngữ Tam phân (TriLM) hiệu quả, có khả năng mở rộng.
                *   `2407.13623` | Nghiên cứu tích hợp yếu tố kích thước từ vựng (Vocabulary Size) vào luật tỷ lệ (Scaling Laws) của LLM, đề xuất loss chuẩn hóa và phương pháp xác định kích thước từ vựng tối ưu.
            *   **Data Mixture & Curation**
                *   `2407.01492` | Đề xuất REGMIX, một quy trình hiệu quả để tối ưu hóa hỗn hợp dữ liệu tiền huấn luyện LLM bằng cách sử dụng nhiều mô hình proxy nhỏ và hồi quy.
                *   `2407.12854` | Xây dựng kho dữ liệu MASSIVE DS (1.4 nghìn tỷ token) và quy trình hiệu quả để nghiên cứu mở rộng quy mô kho dữ liệu cho mô hình ngôn ngữ dựa trên truy xuất.
            *   **Multilingual LLMs**
                *   `2407.20743` | Áp dụng huấn luyện tiếp tục (continual pretraining) và ORPO để tạo Meltemi 7B, LLM mở đầu tiên cho tiếng Hy Lạp.
                *   `2407.19672` | Áp dụng kỹ thuật Nơ-ron Chuyên biệt Ngôn ngữ (LSN) và xây dựng tập SFT đa dạng để tăng cường hiệu quả LLM cho các ngôn ngữ Đông Nam Á.
                *   `2407.05975` | Nghiên cứu huấn luyện liên tục đa ngôn ngữ quy mô lớn (>100 ngôn ngữ) cho LLaMA mà không mở rộng từ vựng, tập trung vào chiến lược tăng cường dữ liệu.
        *   **Instruction Tuning & Alignment**
            *   **Data Generation & Augmentation for SFT/Alignment**
                *   `2406.20094` | Đề xuất phương pháp luận tổng hợp dữ liệu dựa trên persona quy mô lớn (Persona Hub) và kỹ thuật nhắc lệnh tăng cường persona.
                *   `2407.03502` | Đề xuất AgentInstruct, một framework agentic để tự động tạo dữ liệu tổng hợp quy mô lớn, đa dạng và chất lượng cao cho SFT.
                *   `2407.08348` | Đề xuất quy trình SFT toán học hai giai đoạn với dữ liệu tổng hợp quy mô lớn (Skywork-MathQA) được tạo bằng core-set và augmentation.
            *   **Preference Optimization & Self-Correction**
                *   `2407.18248` | Tích hợp Direct Preference Optimization (DPO) vào quy trình tự huấn luyện lặp lại để cải thiện suy luận chuỗi tư duy (CoT) trong toán học.
                *   `2407.00782` | Đề xuất Step-Controlled DPO (SCDPO) để cải thiện suy luận toán học bằng giám sát lỗi theo từng bước và tạo dữ liệu đối nghịch có kiểm soát.
                *   `2407.16637` | Đề xuất quy trình tạo dữ liệu sở thích tổng hợp C2-SYN và áp dụng DPO để dạy LLM khả năng tự sửa lỗi kịp thời (course-correction) khi tạo nội dung độc hại.
            *   **Domain Adaptation**
                *   `2407.19584` | Áp dụng quy trình thích ứng miền ba giai đoạn (pretraining, IFT, DPO) cho LLM pháp lý quy mô lớn (MoE) với dữ liệu chuyên biệt và tổng hợp.
        *   **Long Context Processing**
            *   **Architectures & Methods**
                *   `2407.09450` | Đề xuất EM-LLM, kiến trúc tích hợp phân đoạn sự kiện dựa trên Bayesian surprise, tinh chỉnh đồ thị và truy xuất bộ nhớ hai giai đoạn (tương tự + liền kề) cho ngữ cảnh dài.
                *   `2407.12077` | Giới thiệu kiến trúc lai GoldFinch (Finch-C2 RWKV-based + GOLD transformer) với pre-fill O(1) và cơ chế TokenCat nén cache key toàn cục, loại bỏ value cache.
                *   `2407.04841` | Đề xuất kiến trúc Associative Recurrent Memory Transformer (ARMT) tích hợp self-attention cục bộ, hồi quy cấp đoạn và bộ nhớ kết hợp có hiệu chỉnh để xử lý chuỗi siêu dài.
                *   `2407.04620` | Đề xuất khung Test-Time Training (TTT) cho lớp mô hình chuỗi, nơi trạng thái ẩn là một mô hình học máy được cập nhật tự giám sát, cho phép xử lý ngữ cảnh dài với độ phức tạp tuyến tính.
            *   **Efficient Inference for Long Context**
                *   `2407.14057` | Đề xuất LazyLLM, phương pháp tỉa token động và lũy tiến dựa trên điểm chú ý, cùng Aux Cache để phục hồi token, tăng tốc prefilling.
                *   `2407.02490` | Đề xuất MInference, phương pháp khai thác 3 mẫu hình thưa thớt động (A-shape, Vertical-Slash, Block-Sparse) trong attention ngữ cảnh dài, kết hợp tìm kiếm offline và xấp xỉ online để tăng tốc pre-filling.
            *   **Model Training & Adaptation for Long Context**
                *   `2407.14482` | Trình bày quy trình huấn luyện tiếp nối và tinh chỉnh hướng dẫn 3 giai đoạn để mở rộng Llama3-70B lên ngữ cảnh 128K, bao gồm tạo dữ liệu SFT ngữ cảnh dài tổng hợp.
        *   **Efficient LLMs**
            *   **Low-bitwidth Models & Quantization**
                *   `2407.12327` | Đề xuất phương pháp huấn luyện và kiến trúc cho Mô hình Ngôn ngữ Tam phân (TriLM) hiệu quả, có khả năng mở rộng. (Đã liệt kê ở Foundational Models)
                *   `2407.08296` | Đề xuất Q-GaLore, phương pháp huấn luyện LLM tiết kiệm bộ nhớ với trọng số INT8, ma trận chiếu gradient low-rank INT4 và cập nhật không gian con lazy.
            *   **Inference Optimization (General)**
                *   `2407.07304` | Đề xuất SlimAttention (phân rã 1D điểm chú ý), lượng tử hóa INT8 KV cache chi tiết và tối ưu hóa suy luận phân tán trên CPU.
                *   `2407.21018` | Đề xuất THINK, phương pháp tỉa kênh Key cache phụ thuộc truy vấn dựa trên norm Frobenius của tương tác Query-Key, giảm bộ nhớ KV cache.
            *   **Structured Pruning & Knowledge Distillation**
                *   `2407.14679` | Đề xuất chiến lược élagage có cấu trúc đa trục dựa trên kích hoạt, cộng dồn thông tin đầu chú ý, tìm kiếm kiến trúc nhẹ và chưng cất kiến thức đặc thù để nén LLM.
            *   **Parameter-Efficient Fine-Tuning (PEFT)**
                *   `2407.01906` | Đề xuất Expert-Specialized Fine-Tuning (ESFT) cho MoE LLM, chỉ tinh chỉnh các expert có độ liên quan cao với tác vụ.
        *   **Interpretability, Analysis & Safety**
            *   `2407.14507` | Khảo sát và hệ thống hóa cơ chế kiến thức trong LLM (sử dụng và tiến hóa), các giả thuyết lưu trữ kiến thức và các vấn đề mở.
            *   `2407.06946` | Đề xuất phương pháp đánh giá khả năng tự nhận thức của LLM bằng "câu hỏi bảo mật" do chính LM tạo ra, không cần truy cập nội bộ.
            *   `2407.10058` | Đề xuất bộ dữ liệu RETURN và phương pháp Name-Aware Unlearning Framework (NAUF) để unlearn thông tin cá nhân trong LLM.
            *   `2407.12784` | Đề xuất AGENT POISON, phương pháp tấn công backdoor vào agent LLM dựa trên RAG bằng cách nhiễm độc bộ nhớ/KB và tối ưu trigger để thao túng truy xuất.
        *   **LLMs for Specific Data Types**
            *   `2407.09025` | Đề xuất SHEET COMPRESSOR, framework mã hóa nén bảng tính cho LLM và quy trình Chain of Spreadsheet (CoS) cho tác vụ QA.
    *   **Multimodal AI**
        *   **Vision-Language Models (VLMs/LMMs)**
            *   **Architectures & Pretraining**
                *   `2407.03320` | Cải tiến LVLM InternLM-XComposer-2.5 với ngữ cảnh dài (96K), xử lý ảnh độ phân giải cao và khả năng soạn thảo đa dạng.
                *   `2407.07726` | Đề xuất chiến lược huấn luyện PaliGemma không đóng băng bộ mã hóa hình ảnh và dùng khởi động tốc độ học chậm để cải thiện kỹ năng nền tảng.
                *   `2406.11832` | Đề xuất EVE, VLM không bộ mã hóa với lớp Patch Aligning Layer (PAL) và quy trình huấn luyện 3 giai đoạn để tăng cường nhận dạng hình ảnh.
                *   `2407.14177` | Đề xuất kiến trúc EVLM (dựa trên Flamingo) với đặc trưng thị giác phân cấp, nhiều token học được thay `<image>`, và MoE trong lớp Gated Cross Attention.
            *   **Instruction Tuning & Data Generation for VLMs/LMMs**
                *   `2406.19280` | Đề xuất phương pháp tái định dạng dữ liệu "unblinded" dùng MLLM để tạo dữ liệu VQA y tế chất lượng cao từ cặp ảnh-văn bản PubMed.
                *   `2407.17453` | Đề xuất quy trình huấn luyện VILA2 tự cải thiện VLM thông qua tăng cường dữ liệu lặp lại (tự tăng cường và tăng cường chuyên gia).
                *   `2407.07053` | Đề xuất quy trình tự hướng dẫn đa phương thức dùng LLM sinh mã để tạo ảnh trừu tượng và cặp hướng dẫn suy luận trực quan tương ứng.
                *   `2407.04172` | Đề xuất quy trình tạo dữ liệu hướng dẫn trực tiếp từ ảnh biểu đồ bằng VLM và fine-tuning đơn giai đoạn cho ChartGemma (dựa trên PaliGemma).
                *   `2407.08739` | Đề xuất bộ máy dữ liệu tự động (MAVIS) tạo dữ liệu toán học trực quan (sơ đồ, chú thích, CoT) và quy trình huấn luyện MLLM 4 giai đoạn.
            *   **Unified Multimodal Input Processing**
                *   `2407.07895` | Đề xuất LLaVA-NeXT-Interleave, sử dụng định dạng xen kẽ ảnh-văn bản làm mẫu chung để huấn luyện đồng thời LMM trên đa ảnh, video, 3D và ảnh đơn.
            *   **Multimodal Representation Learning**
                *   `2407.12580` | Đề xuất E5-V, phương pháp dùng prompt để hợp nhất embedding đa phương thức từ MLLM và fine-tuning chỉ trên dữ liệu văn bản.
                *   `2407.20171` | Đề xuất DIVA, framework tự giám sát sau huấn luyện dùng phản hồi tạo sinh từ mô hình khuếch tán để tối ưu hóa biểu diễn hình ảnh CLIP chỉ với dữ liệu ảnh.
            *   **Multimodal Retrieval**
                *   `2407.01449` | Đề xuất ColPali, áp dụng tương tác muộn (late interaction) cho embedding đa vector từ VLM xử lý trực tiếp ảnh tài liệu để truy xuất.
            *   **Multimodal Generation (Story Generation)**
                *   `2407.08683` | Đề xuất SEED-Story, quy trình 3 giai đoạn (MLLM tuning, de-tokenizer adaptation) và multimodal attention sink để tạo truyện đa phương thức dài, nhất quán.
            *   **Specific Applications (Medical)**
                *   `2407.05131` | Đề xuất RULE, framework kiểm soát rủi ro và tinh chỉnh ưu tiên (KBPT) để tăng tính đúng đắn của Med-LVLM khi dùng RAG.
        *   **Audio-Language Models (ALMs)**
            *   `2407.10759` | Đề xuất Qwen2-Audio với quy trình tiền huấn luyện dùng prompt tự nhiên và huấn luyện đồng thời hai chế độ (Phân tích Âm thanh, Trò chuyện Thoại).
            *   `2407.04051` | Đề xuất bộ mã hóa giọng nói ngữ nghĩa có giám sát S3 (dựa trên SenseVoice) và tích hợp vào hệ thống TTS CosyVoice.
        *   **Generative Models (Vision & Video)**
            *   **Image Generation & Editing**
                *   `2406.19997` | Đề xuất phương pháp sinh ảnh tự hồi quy dựa trên mã hóa wavelet nhúng và transformer.
                *   `2407.16982` | Đề xuất Diffree, tích hợp mô-đun dự đoán mặt nạ đối tượng (OMP) vào diffusion model để thêm đối tượng vào ảnh chỉ bằng văn bản, và bộ dữ liệu OABench.
                *   `2407.03471` | Xây dựng bộ dữ liệu AURORA (thay đổi tối thiểu) và benchmark AURORA-BENCH cho chỉnh sửa ảnh hành động/suy luận, cùng metric DiscEdit.
                *   `2407.11633` | Đề xuất DiT-MoE, áp dụng Mixture-of-Experts vào Diffusion Transformer với chuyên gia chia sẻ và loss cân bằng tải để sinh ảnh thưa.
            *   **Video Generation**
                *   `2407.02371` | Xây dựng quy trình tạo bộ dữ liệu OpenVid-1M và kiến trúc Multi-modal Video Diffusion Transformer (MVDiT) với self-attention và temporal-attention đa phương thức.
                *   `2407.19918` | Đề xuất FreeLong, phương pháp training-free dùng SpectralBlend Temporal Attention để mở rộng mô hình video ngắn tạo video dài.
                *   `2407.16655` | Đề xuất MovieDreamer, framework phân cấp (AR + Diffusion) với kịch bản đa phương thức (bao gồm face embedding) để tạo video dài mạch lạc, bảo toàn định danh.
                *   `2407.17438` | Xây dựng bộ dữ liệu HumanVid (video người thực tế và tổng hợp) với chú thích chuyển động camera và người, cùng mô hình cơ sở CamAnimate.
            *   **3D Generation from Image**
                *   `2407.19548` | Đề xuất Cycle3D, quy trình lặp kết hợp diffusion 2D và tái tạo 3D (Gaussian Splatting) trong quá trình khử nhiễu để tạo 3D từ ảnh đơn.
            *   **Virtual Try-On**
                *   `2407.16224` | Đề xuất OutfitAnyone, kiến trúc khuếch tán hai luồng với ReferenceNet và Refiner để thử đồ ảo toàn bộ trang phục.
            *   **Reward-based Alignment for Generative Models**
                *   `2407.08737` | Đề xuất VADER, phương pháp hiệu chỉnh mô hình khuếch tán video bằng lan truyền ngược gradient từ các reward model đa dạng, dùng truncated backpropagation.
            *   **Text-to-Audio Synthesis**
                *   `2407.14358` | Phát triển Stable Audio Open, mô hình latent diffusion text-to-audio dựa trên Stable Audio 2.0, huấn luyện trên dữ liệu Creative Commons.
    *   **Evaluation & Benchmarking**
        *   **LLM Evaluation**
            *   `2407.01370` | Đề xuất quy trình "Haystack" tạo kho văn bản lớn có kiểm soát và nhiệm vụ SummHay để đánh giá tóm tắt ngữ cảnh dài và trích dẫn nguồn.
            *   `2407.11963` | Giới thiệu NeedleBench, khung đánh giá ngữ cảnh dài song ngữ (truy xuất, lý luận) và Thử thách Truy vết Tổ tiên (ATC) cho lý luận đa bước.
            *   `2407.10457` | Phân tích vai trò của tính không xác định (non-determinism) trong đánh giá LLM, chỉ ra sự cần thiết phải xem xét nó.
        *   **Multimodal Evaluation**
            *   `2407.06581` | Thiết kế BlindTest, benchmark gồm 7 tác vụ thị giác cấp thấp (hình học cơ bản) để đánh giá nhận biết chi tiết không gian của VLM.
            *   `2407.01284` | Đề xuất hệ mét đánh giá 4 chiều (IK, IG, CM, RM) và chiến lược tăng cường khái niệm kiến thức (KCA) cho suy luận toán học trực quan của LMM.
            *   `2407.04842` | Xây dựng benchmark MJ-BENCH và khung đánh giá mô hình thẩm định đa phương thức cho sinh ảnh (phù hợp, an toàn, chất lượng, thiên kiến).
            *   `2407.00468` | Đề xuất quy trình chú thích bộ ba (gốc, nhận thức, kiến thức) và độ đo Genuine Accuracy (GA) để đánh giá LMM đáng tin cậy hơn.
            *   `2407.12772` | Đề xuất LMMs-Eval Lite (chọn lọc dữ liệu bằng k-Center), LiveBench (dữ liệu cập nhật liên tục) và phương pháp phát hiện trùng lặp ảnh bằng n-gram token ảnh.
            *   `2407.14505` | Xây dựng benchmark T2V-CompBench và bộ phương pháp đánh giá tính kết hợp (compositionality) trong T2V.
            *   `2407.10957` | Đề xuất nhiệm vụ Referring Audio-Visual Segmentation (Ref-AVS) và bộ dữ liệu Ref-AVS Bench để phân đoạn đối tượng dựa trên gợi ý đa phương tiện.
        *   **Agent Evaluation**
            *   `2407.18961` | Xây dựng benchmark MMAU để đánh giá tách biệt các năng lực cốt lõi của agent LLM (Hiểu, Lý luận, Lập kế hoạch, Giải quyết, Tự sửa lỗi) bằng dữ liệu tĩnh.
            *   `2407.18901` | Xây dựng môi trường mô phỏng AppWorld Engine và benchmark để đánh giá agent tương tác phức tạp với ứng dụng, sử dụng đánh giá dựa trên trạng thái DB.
        *   **Tabular Data Benchmarking**
            *   `2406.19380` | Giới thiệu TabReD, benchmark dữ liệu bảng cấp công nghiệp, giàu đặc trưng, với phân chia huấn luyện/kiểm tra dựa trên thời gian.
    *   **AI for Specific Domains & Applications**
        *   **Robotics & Embodied AI**
            *   `2406.19741` | Đề xuất ROS-LLM, framework tích hợp ROS với LLM cho lập trình robot bằng ngôn ngữ tự nhiên, có cơ chế phản hồi.
            *   `2407.20179` | Đề xuất Theia, quy trình chưng cất kiến thức từ nhiều Vision Foundation Model vào một bộ mã hóa thị giác nhỏ gọn cho học robot.
            *   `2407.20798` | Đề xuất DAAG, framework dùng LLM điều phối VLM và Diffusion Model để tăng cường dữ liệu kinh nghiệm (HEA) cho tác nhân RL.
            *   `2407.10943` | Xây dựng GRScenes (bộ dữ liệu cảnh 3D tương tác quy mô lớn) và GRResidents (NPC điều khiển bởi LLM với World Knowledge Manager).
        *   **Software Engineering & Code**
            *   `2407.01489` | Đề xuất AGENTLESS, quy trình 3 pha (định vị, sửa chữa, xác thực) giải quyết vấn đề phát triển phần mềm không cần agent tự quyết định.
        *   **Healthcare & Affective Computing**
            *   `2407.19340` | Đề xuất kiến trúc tri-modal (BiLSTM, Model-Level Fusion) tích hợp đầu ra phân loại của GPT-4 để phân loại trầm cảm.
            *   `2407.13301` | Đề xuất Chain-of-Diagnosis (CoD), quy trình 5 bước cho LLM chẩn đoán y tế, và phương pháp giảm entropy để chọn triệu chứng hỏi tiếp.
        *   **Chemistry**
            *   `2407.20267` | Đề xuất kiến trúc encoder-decoder SMI-TED (và MoE) để tạo biểu diễn ẩn cho chuỗi SMILES và tiền huấn luyện 2 giai đoạn.
        *   **Content Moderation**
            *   `2407.20729` | Xây dựng quy trình thu thập, gán nhãn và huấn luyện bộ phân loại NSFW đầu tiên cho tiếng Malaysia.
    *   **Agent Systems & Frameworks**
        *   `2407.16741` | Xây dựng OpenHands, nền tảng agent mã nguồn mở với kiến trúc luồng sự kiện, môi trường thực thi sandbox (bash, IPython, browser) và thư viện AgentSkills.
        *   `2407.17535` | Đề xuất LAMBDA, hệ thống đa tác nhân (Programmer, Inspector) với Cơ chế Tích hợp Tri thức cho phân tích dữ liệu không cần code.
        *   `2407.17789` | Đề xuất AgentScope, nền tảng mô phỏng đa agent quy mô lớn với cơ chế phân tán dựa trên actor model, tương tác agent-môi trường linh hoạt và tạo hồ sơ agent tự động.
        *   `2407.07061` | Đề xuất Internet of Agents (IoA), kiến trúc client-server nhiều lớp cho cộng tác agent phân tán, với giao thức tích hợp, hình thành nhóm lồng nhau và kiểm soát luồng hội thoại tự trị.
        *   `2407.20183` | Đề xuất MindSearch, kiến trúc đa tác nhân (WebPlanner, WebSearcher) với lập kế hoạch DAG động bằng mã LLM và truy xuất phân cấp cho tìm kiếm web phức tạp.
    *   **Vision Backbones & Representation Learning**
        *   `2407.18907` | Đề xuất SHIC, phương pháp không giám sát học liên kết ảnh-với-mẫu 3D bằng cách giảm thành đối sánh ảnh-render và tổng hợp thông tin đa khung nhìn.
        *   `2407.19985` | Đề xuất MoNE (Mixture of Nested Experts), sử dụng mô hình con lồng nhau làm expert với chi phí khác nhau và định tuyến EPR cho Vision Transformer.
        *   `2407.08083` | Đề xuất MambaVision, backbone lai Mamba-Transformer phân cấp với MambaVision Mixer cải tiến và self-attention ở tầng cuối.
    *   **Time Series Analysis**
        *   `2407.07874` | Đề xuất Toto, mô hình transformer cho chuỗi thời gian với Proportional Factorized Space-Time Attention, Student-T Mixture Model Head và tiền huấn luyện trên dữ liệu quan sát quy mô lớn.
    *   **Sequence Modeling (General)**
        *   `2407.01392` | Đề xuất Diffusion Forcing (DF) và Causal Diffusion Forcing (CDF), huấn luyện mô hình chuỗi nhân quả khử nhiễu token với mức nhiễu độc lập, cho phép lấy mẫu và hướng dẫn linh hoạt.
    *   **Other**
        *   `2407.16674` | `2407.15017` | `2407.17952` | `2407.20581` | Các bài báo này chủ yếu là phân tích, khảo sát, hoặc áp dụng kỹ thuật hiện có cho miền/ngôn ngữ cụ thể mà không có đóng góp kỹ thuật thuật toán/kiến trúc mới nổi bật theo định nghĩa. (Lưu ý: Một số đã được phân loại ở trên nếu có khía cạnh phù hợp, phần "Other" này để đảm bảo không bỏ sót nếu không khớp hoàn toàn).

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                               |
    | :--- | :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | `2407.09450` | EM-LLM, Long Context, Episodic Memory, Surprise      | Tích hợp cơ chế phân đoạn sự kiện và truy xuất bộ nhớ hai giai đoạn (tương tự + liền kề) lấy cảm hứng từ nhận thức con người vào LLM. | Mở ra hướng xử lý ngữ cảnh dài vô hạn trên lý thuyết mà không cần fine-tuning, cải thiện khả năng suy luận dài hạn của LLM.                 |
    | 2    | `2407.12077` | GoldFinch, Hybrid RNN-Attention, O(1) Pre-fill, TokenCat | Kiến trúc lai Finch-C2 (RWKV-based) và GOLD transformer với pre-fill O(1), cache key toàn cục siêu nén và loại bỏ value cache.        | Tiềm năng giảm bộ nhớ cache cực lớn và tăng tốc pre-fill cho LLM ngữ cảnh dài, thúc đẩy mô hình hiệu quả hơn.                             |
    | 3    | `2407.01489` | AGENTLESS, Program Repair, SWE-bench, No Agent       | Quy trình 3 pha (định vị, sửa chữa, xác thực) giải quyết vấn đề phần mềm không cần LLM tự quyết định, vượt trội agent mã nguồn mở.     | Thách thức giả định về sự cần thiết của agent tự chủ phức tạp, mở ra hướng tiếp cận đơn giản, hiệu quả cho sửa lỗi tự động.              |
    | 4    | `2407.08348` | Skywork-MathQA, Math SFT, Two-stage SFT, Data Augmentation | Đạt hiệu năng SOTA (vượt GPT-4 đời đầu) trên MATH benchmark chỉ bằng SFT hai giai đoạn trên LLM 7B với dữ liệu tổng hợp quy mô lớn. | Chứng minh tiềm năng lớn của SFT có cấu trúc và dữ liệu chất lượng cao để nâng cao năng lực suy luận toán học cho LLM nhỏ.               |
    | 5    | `2407.18901` | AppWorld, Agent Benchmark, Complex Tool Use, State-based Eval | Môi trường mô phỏng ứng dụng thực tế phức tạp và phương pháp đánh giá agent dựa trên trạng thái DB, thách thức các LLM hàng đầu. | Đặt ra tiêu chuẩn mới cho đánh giá agent tương tác, thúc đẩy phát triển agent có khả năng lập trình và tương tác phức tạp hơn.          |
    | 6    | `2407.04172` | ChartGemma, Chart Understanding, VLM, Single-stage FT  | Tạo dữ liệu hướng dẫn trực tiếp từ ảnh biểu đồ bằng VLM và fine-tuning đơn giai đoạn trên PaliGemma, đạt SOTA trên nhiều benchmark. | Đơn giản hóa quy trình huấn luyện VLM hiểu biểu đồ, cải thiện chất lượng dữ liệu và hiệu năng với mô hình nhỏ hơn.                       |
    | 7    | `2407.16655` | MovieDreamer, Long Video Generation, Hierarchical, Face ID | Framework phân cấp (AR + Diffusion) với kịch bản đa phương thức (bao gồm face embedding) để tạo video dài mạch lạc, bảo toàn định danh. | Giải quyết thách thức tạo video dài có cốt truyện và nhân vật nhất quán, một bước tiến quan trọng trong sinh video.                     |
    | 8    | `2407.00320` | LiteSearch, Tree Search, Dynamic Budget, Value Network | Thuật toán tìm kiếm cây LiteSearch với lựa chọn nút và ngân sách khám phá động, giảm chi phí tính toán cho suy luận LLM.             | Cung cấp giải pháp hiệu quả hơn cho các bài toán suy luận đòi hỏi tìm kiếm không gian lớn, cân bằng giữa khám phá và khai thác.         |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **`2407.09025` – SHEET COMPRESSOR (Structural Anchors, Inverted-Index Translation, Data-Format-Aware Aggregation) và Chain of Spreadsheet (CoS) – Suy nghĩ:** Các mô-đun nén chuyên biệt cho bảng tính là đóng góp kỹ thuật rõ ràng, giải quyết vấn đề thực tế về giới hạn token của LLM khi xử lý bảng tính lớn. CoS là một ứng dụng hợp lý của CoT cho miền dữ liệu này.
    *   **`2406.20094` – Persona-driven data synthesis (Text-to-Persona, Persona-to-Persona, Persona-enhanced few-shot prompting) – Suy nghĩ:** Phương pháp luận mới lạ và có tiềm năng lớn để tạo dữ liệu đa dạng ở quy mô chưa từng có bằng cách khai thác khả năng nhập vai của LLM. Đơn giản nhưng hiệu quả.
    *   **`2407.01370` – Quy trình tổng hợp dữ liệu "Haystack" và nhiệm vụ "SummHay" cùng metric đánh giá Coverage/Citation – Suy nghĩ:** Phương pháp tổng hợp dữ liệu có kiểm soát để tạo benchmark đánh giá tóm tắt ngữ cảnh dài và trích dẫn là rất sáng tạo, cho phép đánh giá tự động đáng tin cậy hơn.
    *   **`2407.06581` – Benchmark BlindTest cho thị giác cấp thấp của VLM – Suy nghĩ:** Thiết kế benchmark tập trung vào "điểm mù" chi tiết không gian của VLM là một đóng góp quan trọng, giúp xác định và phân tích một hạn chế cơ bản của các mô hình hiện tại.
    *   **`2407.01284` – Hệ mét đánh giá 4 chiều (IK, IG, CM, RM) và chiến lược tăng cường khái niệm kiến thức (KCA) cho suy luận toán học trực quan – Suy nghĩ:** Phương pháp đánh giá quá trình suy luận dựa trên phân rã khái niệm kiến thức là rất mới lạ và sâu sắc, cung cấp công cụ chẩn đoán mạnh mẽ.
    *   **`2407.12327` – Phương pháp huấn luyện TriLM và biến thể model parallelism với scale value độc lập trên shard – Suy nghĩ:** Nghiên cứu quy mô lớn về mô hình tam phân, kỹ thuật tính scale value độc lập là một giải pháp thực tế cho huấn luyện song song, dù có thể tạo artifact nhỏ.
    *   **`2407.09450` – Phân đoạn sự kiện dựa trên Bayesian surprise, tinh chỉnh ranh giới bằng lý thuyết đồ thị, và truy xuất bộ nhớ hai giai đoạn (similarity + contiguity) trong EM-LLM – Suy nghĩ:** Kết hợp các khái niệm từ khoa học nhận thức vào LLM một cách sáng tạo, tạo ra một cơ chế xử lý ngữ cảnh dài hiệu quả và có cơ sở lý thuyết.
    *   **`2407.01489` – Quy trình AGENTLESS (định vị phân cấp, sửa chữa diff, xác thực bằng test tái hiện lỗi) – Suy nghĩ:** Cách tiếp cận đơn giản nhưng hiệu quả đáng ngạc nhiên, thách thức sự cần thiết của agent phức tạp cho sửa lỗi, quy trình định vị và xác thực có tính mới.
    *   **`2406.19280` – Phương pháp tái định dạng dữ liệu "unblinded" dùng MLLM cho VQA y tế – Suy nghĩ:** Giải pháp thực tế và giá trị, sử dụng khả năng của MLLM để "nhìn" ảnh khi tái định dạng văn bản giúp giảm nhiễu và tăng chất lượng dữ liệu VQA y tế.
    *   **`2407.19918` – SpectralBlend Temporal Attention (SpectralBlend-TA) với tách rời chú ý cục bộ-toàn cục và trộn phổ cho video dài – Suy nghĩ:** Giải pháp training-free sáng tạo, giải quyết vấn đề méo tần số khi mở rộng mô hình video ngắn sang video dài bằng cách cân bằng thông tin cục bộ và toàn cục trong miền tần số.
    *   **`2407.12784` – AGENT POISON: Tối ưu hóa trigger backdoor cho RAG-LLM agent với Uniqueness Loss và Compactness Loss – Suy nghĩ:** Phương pháp tấn công backdoor mới lạ, thao túng không gian nhúng RAG để kiểm soát truy xuất là đóng góp kỹ thuật cốt lõi, rất tinh vi.
    *   **`2407.08737` – VADER: Hiệu chỉnh mô hình khuếch tán video bằng lan truyền ngược gradient từ reward models, dùng truncated backpropagation – Suy nghĩ:** Phương pháp hiệu quả để hiệu chỉnh mô hình khuếch tán video, giải quyết vấn đề bộ nhớ và tận dụng reward model có sẵn.
    *   **`2407.03502` – AgentInstruct: Framework agentic tạo dữ liệu tổng hợp từ dữ liệu thô qua 3 luồng (transformation, generation, refinement) – Suy nghĩ:** Hướng đi sáng tạo, giảm phụ thuộc vào prompt hạt giống, tăng khả năng tạo tác vụ mới lạ cho huấn luyện LLM.
    *   **`2407.20179` – Theia: Chưng cất kiến thức từ nhiều VFM vào một encoder thị giác cho robot, dùng feature translators và tối ưu đồng thời đặc trưng không gian – Suy nghĩ:** Phương pháp chưng cất đa-VFM hợp lý, tập trung vào token không gian cho học robot, hiệu quả về tính toán.
    *   **`2407.01449` – ColPali: Tương tác muộn trên embedding đa vector từ VLM xử lý trực tiếp ảnh tài liệu – Suy nghĩ:** Hướng tiếp cận mới lạ cho truy xuất tài liệu giàu hình ảnh, đơn giản hóa pipeline và cho phép huấn luyện end-to-end.
    *   **`2407.07053` – Quy trình tự hướng dẫn đa phương thức dùng LLM sinh mã để tạo ảnh trừu tượng và cặp hướng dẫn suy luận trực quan – Suy nghĩ:** Sáng tạo trong việc dùng LLM sinh mã để đảm bảo độ chính xác hình ảnh trừu tượng, giải quyết vấn đề thiếu dữ liệu huấn luyện chất lượng cao cho LMM hiểu hình ảnh trừu tượng.
    *   **`2407.14057` – LazyLLM: Tỉa token động và lũy tiến dựa trên attention score, với Aux Cache để phục hồi token – Suy nghĩ:** Giải pháp training-free thông minh để tăng tốc prefilling ngữ cảnh dài, cơ chế Aux Cache xử lý phục hồi token hiệu quả.
    *   **`2407.01392` – Diffusion Forcing (DF) / Causal Diffusion Forcing (CDF): Huấn luyện mô hình chuỗi nhân quả khử nhiễu token với mức nhiễu độc lập và lịch trình linh hoạt – Suy nghĩ:** Phương pháp mới lạ kết hợp điểm mạnh của mô hình tự hồi quy và khuếch tán, cho phép tạo chuỗi dài ổn định và hướng dẫn mạnh mẽ.
    *   **`2407.02490` – MInference: Phân loại 3 mẫu hình thưa thớt động (A-shape, VS, BS), tìm kiếm offline gán mẫu tối ưu cho head, xấp xỉ online xây dựng chỉ số thưa thớt – Suy nghĩ:** Giải pháp sáng tạo kết hợp phân tích offline và xấp xỉ online hiệu quả cùng kernel GPU tối ưu để tăng tốc pre-filling LLM ngữ cảnh dài.
    *   **`2407.00782` – Step-Controlled DPO (SCDPO): Tạo dữ liệu đối nghịch có lỗi được kiểm soát theo bước và DPO nhận biết bước cho suy luận toán học – Suy nghĩ:** Phương pháp giám sát lỗi theo bước tự động, không cần chú thích thủ công, cải thiện gán tín dụng trong DPO.
    *   **`2407.20798` – DAAG/HEA: LLM điều phối VLM và Diffusion Model để tăng cường dữ liệu kinh nghiệm (biến đổi video quan sát) cho tác nhân RL – Suy nghĩ:** Phương pháp luận thú vị kết hợp các mô hình nền tảng để cải thiện hiệu quả sử dụng dữ liệu trong RL, đặc biệt là quy trình khuếch tán đảm bảo nhất quán video.
    *   **`2407.10957` – Nhiệm vụ Referring Audio-Visual Segmentation (Ref-AVS) và bộ dữ liệu Ref-AVS Bench – Suy nghĩ:** Nhiệm vụ mới thách thức, giải quyết hạn chế của các tác vụ phân đoạn tham chiếu trước, bộ dữ liệu là đóng góp quan trọng. Framework crossmodal transformer là ứng dụng hợp lý.
    *   **`2407.04363` – AriGraph: Kiến trúc bộ nhớ đồ thị tích hợp động bộ nhớ ngữ nghĩa và tình huống, với quy trình học và truy xuất hai giai đoạn cho agent LLM – Suy nghĩ:** Phương pháp sáng tạo để biểu diễn tri thức có cấu trúc và truy xuất hiệu quả trong môi trường POMDP, cơ chế học và cập nhật trực tuyến là điểm mạnh.
    *   **`2407.21018` – THINK: Tỉa kênh Key cache phụ thuộc truy vấn dựa trên norm Frobenius của tương tác Query-Key – Suy nghĩ:** Hướng tiếp cận mới để giảm bộ nhớ KV cache bằng cách khai thác dư thừa ở chiều kênh, tiêu chí tỉa kênh có cơ sở lý thuyết.
    *   **`2407.18908` – Wolf: Framework tổng hợp video caption dựa trên mixture-of-experts (VLM ảnh + VLM video) và metric CapScore dựa trên LLM – Suy nghĩ:** Kết hợp nhiều VLM để cải thiện chất lượng caption video là hướng đi mới. CapScore là metric đánh giá thú vị.
    *   **`2407.17952` – BetterDepth: Bộ tinh chỉnh khuếch tán plug-and-play cho MDE zero-shot, với tiền căn chỉnh toàn cục và che cục bộ theo patch – Suy nghĩ:** Giải pháp lai ghép hợp lý, kết hợp robustness của MDE zero-shot và detail của diffusion, các kỹ thuật huấn luyện giúp giữ kiến thức tiên nghiệm.
    *   **`2407.08083` – MambaVision: Backbone lai Mamba-Transformer phân cấp với MambaVision Mixer (tích chập thường + nhánh đối xứng) – Suy nghĩ:** Kiến trúc lai mới lạ, MambaVision Mixer được thiết kế lại phù hợp hơn cho thị giác, cân bằng tốt độ chính xác/tốc độ.
    *   **`2407.07874` – Proportional Factorized Space-Time Attention và Student-T Mixture Model Head trong Toto – Suy nghĩ:** Cơ chế chú ý không-thời gian theo tỷ lệ linh hoạt và đầu ra hỗn hợp Student-T là những đóng góp kỹ thuật giải quyết thách thức thực tế của dữ liệu chuỗi thời gian.
    *   **`2407.20267` – Kiến trúc encoder-decoder SMI-TED tạo biểu diễn ẩn cho toàn bộ chuỗi SMILES và biến thể MoE định tuyến bằng embedding phân tử – Suy nghĩ:** Kiến trúc encoder-decoder tạo biểu diễn ẩn cho SMILES là đóng góp kỹ thuật đáng chú ý cho lĩnh vực hóa học.
    *   **`2407.19548` – Cycle3D: Quy trình lặp kết hợp diffusion 2D và tái tạo 3D (GS) trong khử nhiễu, với tích hợp time embedding và tương tác đặc trưng – Suy nghĩ:** Kiến trúc lai ghép sáng tạo, cơ chế phản hồi dùng ảnh render lại để điều hướng diffusion là đóng góp kỹ thuật đáng chú ý cho tạo 3D từ ảnh đơn.
    *   **`2407.07061` – Internet of Agents (IoA): Kiến trúc client-server, giao thức tích hợp agent, hình thành nhóm lồng nhau tự trị, kiểm soát luồng hội thoại FSM – Suy nghĩ:** Framework có cấu trúc tốt, các cơ chế tự trị cho hình thành nhóm và kiểm soát luồng hội thoại là mới lạ và có tiềm năng.
    *   **`2407.16637` – Quy trình tạo dữ liệu sở thích tổng hợp C2-SYN (cắt ngắn, trigger sửa lỗi) và áp dụng DPO để dạy LLM tự sửa lỗi kịp thời – Suy nghĩ:** Quy trình tạo dữ liệu C2-SYN chuyên biệt cho "course-correction" là sáng tạo, giải quyết vấn đề chi phí thu thập dữ liệu người.
    *   **`2407.18961` – Phương pháp đánh giá `planner-shift` và `solver-shift` trong MMAU – Suy nghĩ:** Ý tưởng sáng tạo để cô lập và đánh giá tách biệt các kỹ năng lập kế hoạch và giải quyết vấn đề của agent LLM.
    *   **`2407.12580` – E5-V: Biểu diễn dựa trên prompt để hợp nhất embedding đa phương thức từ MLLM và huấn luyện chỉ trên dữ liệu văn bản – Suy nghĩ:** Hướng đi sáng tạo và hiệu quả chi phí, khai thác instruction following của MLLM để hợp nhất không gian embedding.
    *   **`2407.14679` – Cộng dồn thông tin còn dư từ đầu chú ý bị élagage, tìm kiếm kiến trúc nhẹ bằng huấn luyện lại giới hạn, và KD với trọng số động/biến đổi tuyến tính cho trạng thái ẩn – Suy nghĩ:** Các kỹ thuật này trong quy trình nén LLM đa trục là những bổ sung sáng tạo và thực tế.
    *   **`2407.06189` – Video-STaR: Quy trình tự huấn luyện theo chu kỳ cho LVLM video với Xác minh Nhãn (giám sát yếu) và Hợp lý hóa Nhãn – Suy nghĩ:** Áp dụng và điều chỉnh kỹ thuật tự huấn luyện cho LVLM video một cách sáng tạo, tận dụng các loại nhãn video đa dạng làm giám sát yếu.
    *   **`2407.19985` – MoNE: Mixture of Nested Experts với các mô hình con lồng nhau làm expert chi phí khác nhau và định tuyến EPR – Suy nghĩ:** Kết hợp sáng tạo kiến trúc lồng nhau và MoE để đạt hiệu quả tính toán trong Vision Transformer mà không tăng tham số.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Đánh giá và Benchmark:**
        *   Nhiều phương pháp mới thiếu benchmark toàn diện hoặc dựa trên các metric chưa phản ánh đầy đủ năng lực (ví dụ: chất lượng truyện đa phương thức, độ tin cậy của agent trong môi trường phức tạp, khả năng khái quát hóa của dữ liệu tổng hợp). → **Cơ hội:** Phát triển benchmark đa dạng, thực tế hơn, và các metric đánh giá tự động đáng tin cậy hơn, đặc biệt cho các tác vụ tạo sinh phức tạp và agent tương tác.
        *   Vấn đề nhiễm bẩn dữ liệu (data contamination) vẫn là một thách thức lớn cho các benchmark tĩnh. → **Cơ hội:** Phát triển thêm các phương pháp tạo benchmark động (như LiveBench `2407.12772`) hoặc các kỹ thuật phát hiện nhiễm bẩn tinh vi hơn.
        *   Đánh giá khả năng suy luận logic đa bước và chiều sâu hiểu biết của LLM/LMM vẫn còn hạn chế. → **Cơ hội:** Xây dựng các benchmark tập trung sâu hơn vào các loại suy luận phức tạp, nhân quả, và trừu tượng.
    *   **Dữ liệu:**
        *   Chất lượng và sự đa dạng của dữ liệu tổng hợp (synthetic data) phụ thuộc nhiều vào mô hình tạo dữ liệu (thường là các LLM/VLM mạnh). → **Cơ hội:** Nghiên cứu các phương pháp tạo dữ liệu ít phụ thuộc hơn, hoặc các kỹ thuật kiểm soát chất lượng và giảm thiên kiến (bias) trong dữ liệu tổng hợp.
        *   Thiếu dữ liệu chất lượng cao cho các ngôn ngữ ít tài nguyên và các miền chuyên biệt. → **Cơ hội:** Phát triển các kỹ thuật thu thập, tăng cường và tổng hợp dữ liệu hiệu quả hơn cho các kịch bản này.
    *   **Hiệu quả và Khả năng mở rộng:**
        *   Chi phí tính toán cho huấn luyện và suy luận các mô hình lớn (đặc biệt là MoE, mô hình khuếch tán, agent phức tạp) vẫn là rào cản. → **Cơ hội:** Tiếp tục nghiên cứu các kiến trúc hiệu quả hơn, thuật toán tối ưu hóa, và kỹ thuật nén/lượng tử hóa tiên tiến.
        *   Khả năng mở rộng của các hệ thống agent phức tạp (ví dụ: quản lý nhiều agent, giao tiếp hiệu quả, xử lý lỗi) cần được cải thiện. → **Cơ hội:** Phát triển các framework agent mạnh mẽ hơn, hỗ trợ tốt hơn cho phân tán và tương tác động.
    *   **An toàn, Tin cậy và Diễn giải:**
        *   Vấn đề ảo giác (hallucination), thiên kiến, và khả năng bị tấn công (backdoor, jailbreak) vẫn tồn tại. → **Cơ hội:** Nghiên cứu sâu hơn về cơ chế gây ra các vấn đề này và phát triển các phương pháp phòng thủ, căn chỉnh hiệu quả hơn, cũng như các kỹ thuật unlearning mạnh mẽ.
        *   Khả năng diễn giải và hiểu cơ chế hoạt động bên trong của các mô hình lớn còn hạn chế. → **Cơ hội:** Phát triển các công cụ và phương pháp phân tích, diễn giải mới, đặc biệt cho các kiến trúc phức tạp như MoE hoặc các mô hình học các biểu diễn tiềm ẩn.
    *   **Tích hợp đa phương thức và đa tác vụ:**
        *   Việc hợp nhất hiệu quả thông tin từ nhiều phương tiện (văn bản, ảnh, video, âm thanh, 3D, bảng biểu) và xử lý các tác vụ đa dạng trong một mô hình duy nhất vẫn là thách thức. → **Cơ hội:** Nghiên cứu các kiến trúc hợp nhất đa phương thức tiên tiến hơn, các cơ chế attention và fusion hiệu quả, và các chiến lược huấn luyện đa tác vụ tốt hơn.
    *   **Suy luận và Lập kế hoạch:**
        *   Khả năng suy luận đa bước, lập kế hoạch dài hạn và tương tác có mục đích với môi trường của agent vẫn cần cải thiện đáng kể. → **Cơ hội:** Tích hợp các cơ chế bộ nhớ có cấu trúc hơn, các thuật toán tìm kiếm và lập kế hoạch hiệu quả, và khả năng học từ tương tác một cách chủ động.

5.  **FUTURE_IDEAS**

    ✨ **Adaptive Knowledge Integration for Lifelong Learning Agents**
    *   **Motivation:** Current agents struggle with catastrophic forgetting and efficiently integrating new knowledge from diverse sources (text, vision, interaction) over long periods.
    *   **Key novelty:** Develop an agent architecture with a dynamic knowledge graph (inspired by `2407.04363` AriGraph) that can intelligently decide when and how to integrate new information, prune outdated/irrelevant knowledge, and use a "Test-Time Training" like mechanism (`2407.04620`) on its internal knowledge representation to adapt to new tasks or contexts without full retraining.
    *   **Approach:**
        1.  Core LLM planner/reasoner.
        2.  Dynamic knowledge graph (KG) storing semantic and episodic memories.
        3.  A "Knowledge Curator" module (itself potentially an LLM) that monitors incoming information, assesses relevance/novelty/conflict with existing KG, and decides on integration strategy (e.g., merge, update, create new branch).
        4.  A TTT-like mechanism that allows rapid, localized "re-learning" or "re-weighting" of KG connections based on recent interactions or high-priority information.
    *   **Dataset + Metrics:** Environments like AppWorld (`2407.18901`) or GRUtopia (`2407.10943`) for long-term task completion. Metrics: task success rate over time, knowledge retention/adaptation tests, computational cost of knowledge updates.
    *   **Risk/Feasibility:** High complexity in designing the Knowledge Curator and ensuring stable TTT-like updates on a graph. Feasibility depends on efficient graph operations and LLM's ability to manage knowledge.

    ✨ **Cognitive-Inspired Multimodal Narrative Generation with Controllable Consistency**
    *   **Motivation:** Generating long, coherent, and engaging multimodal narratives (text + image/video) where characters and plot points remain consistent is a major challenge.
    *   **Key novelty:** Combine hierarchical planning (like MovieDreamer `2407.16655`) with episodic memory for plot points (inspired by EM-LLM `2407.09450`) and a wavelet-based generative model (`2406.19997`) for fine-grained visual control and consistency, potentially guided by a "course-correction" mechanism (`2407.16637`) for narrative coherence.
    *   **Approach:**
        1.  High-level AR model plans key story beats and character arcs, storing them in an episodic memory.
        2.  Mid-level module generates detailed scene descriptions and image/video concepts, retrieving from episodic memory for consistency.
        3.  Low-level wavelet-based transformer generates images/short video clips based on concepts, conditioned on character embeddings (from MovieDreamer) and style prompts.
        4.  A "narrative critic" (LLM/VLM) evaluates consistency and engagement, providing feedback for course-correction at the planning or generation stage.
    *   **Dataset + Metrics:** Large-scale movie script/storyboard datasets, human-annotated multimodal stories. Metrics: Narrative coherence (human eval, LLM-based eval like CapScore `2407.18908`), character/style consistency, engagement.
    *   **Risk/Feasibility:** Very complex system with multiple interacting generative models. Ensuring robust consistency across modalities and long timelines is extremely hard. High computational cost.

    ✨ **"Unblinded" Data Augmentation for Robust Low-Resource Language Understanding**
    *   **Motivation:** Low-resource languages suffer from data scarcity, and existing augmentation often introduces noise or lacks contextual richness.
    *   **Key novelty:** Adapt the "unblinded" MLLM-based data reformatting idea from `2406.19280` (originally for VQA) to text-only low-resource scenarios. Use a powerful multilingual LLM to "look" at noisy/scarce low-resource text and its (potentially poor) translation in a high-resource language, then generate higher-quality, contextually richer parallel data or instruction-following pairs in the low-resource language.
    *   **Approach:**
        1.  Collect available low-resource text (X_lr) and any corresponding high-resource translations (X_hr, can be noisy).
        2.  Provide (X_lr, X_hr) to a strong multilingual LLM (e.g., Qwen2 `2407.10671`, LLaMAX `2407.05975`).
        3.  Prompt the LLM to:
            *   Generate a cleaner, more idiomatic version of X_lr.
            *   Generate a more accurate X_hr if the original was poor.
            *   Generate diverse question-answer pairs or instruction-following examples based on the *refined* (X_lr, X_hr) content, specifically in X_lr.
    *   **Dataset + Metrics:** Existing low-resource benchmarks (translation, QA, summarization). Metrics: Standard task-specific metrics (BLEU, F1, ROUGE), and human evaluation for naturalness and contextual relevance of generated data.
    *   **Risk/Feasibility:** Depends heavily on the multilingual LLM's capabilities in the specific low-resource language. Risk of propagating biases from the teacher LLM. Feasibility is high if a good teacher LLM is available.

    ✨ **Self-Evolving Benchmarks for Continuous AI Evaluation (*Moon-shot*)**
    *   **Motivation:** Static benchmarks quickly become saturated and contaminated. Live benchmarks (`2407.12772`) are a step, but creating truly challenging and novel test cases is hard.
    *   **Key novelty:** A system where AI agents (inspired by AgentInstruct `2407.03502` or AppWorld task generators `2407.18901`) are tasked with *creating new benchmark problems* that other AI systems (including their own future versions) find difficult. This involves AI understanding current SOTA limitations and designing tasks that probe those weaknesses.
    *   **Approach:**
        1.  A "Benchmark Crafter" agent (LLM-based) analyzes the performance of various models on existing benchmarks (like MMAU `2407.18961`, T2V-CompBench `2407.14505`).
        2.  It identifies patterns of failure or areas where models struggle (e.g., complex reasoning, specific compositional skills, OOD generalization).
        3.  It then generates new task descriptions, data, or even simple simulated environments (if applicable) designed to be challenging for current SOTA models. These could be new "needles" for NeedleBench (`2407.11963`) or new scenarios for AppWorld.
        4.  A "Benchmark Validator" agent (possibly with human oversight initially) filters and refines these new problems for clarity, solvability, and relevance.
        5.  The new problems are added to a continuously evolving benchmark suite.
    *   **Dataset + Metrics:** The benchmark itself is the dataset. Metrics: Model performance on the evolving benchmark, "difficulty score" of generated problems (how many SOTA models fail), diversity of generated problems.
    *   **Risk/Feasibility:** Extremely high risk. Requires AI with meta-reasoning capabilities far beyond current systems. Defining "difficulty" and "novelty" for AI-generated tasks is non-trivial. High potential for generating unsolvable or nonsensical tasks. Feasibility is very low with current tech, truly a moon-shot.

6.  **READING_LIST** (Top papers đáng đọc)

    *   `2407.09450` – EM-LLM · Phương pháp xử lý ngữ cảnh dài sáng tạo, lấy cảm hứng từ nhận thức con người, không cần fine-tuning.
    *   `2407.12077` – GoldFinch · Kiến trúc lai RNN-Attention với pre-fill O(1) và cơ chế nén cache key đột phá, tiềm năng lớn cho LLM hiệu quả.
    *   `2407.01489` – AGENTLESS · Cách tiếp cận sửa lỗi tự động đơn giản nhưng hiệu quả đáng ngạc nhiên, thách thức các agent phức tạp.
    *   `2407.18901` – AppWorld · Benchmark và môi trường mô phỏng agent tương tác phức tạp, đặt ra tiêu chuẩn mới cho đánh giá agent.
    *   `2407.01370` – Haystack & SummHay · Phương pháp tạo benchmark và đánh giá tóm tắt ngữ cảnh dài/trích dẫn nguồn rất sáng tạo và cần thiết.
    *   `2407.14505` – T2V-CompBench · Benchmark và phương pháp luận đánh giá tính kết hợp trong T2V một cách hệ thống và chi tiết.
    *   `2407.12784` – AGENT POISON · Tiết lộ một hướng tấn công backdoor mới và tinh vi vào các agent LLM dựa trên RAG.
    *   `2407.04172` – ChartGemma · Phương pháp hiệu quả và đơn giản hóa việc huấn luyện VLM hiểu biểu đồ, đạt SOTA với mô hình nhỏ.
    *   `2407.00320` – LiteSearch · Thuật toán tìm kiếm cây hiệu quả với ngân sách động, giải quyết vấn đề chi phí tính toán trong suy luận LLM.
    *   `2407.16655` – MovieDreamer · Hướng tiếp cận phân cấp hứa hẹn cho việc tạo video dài mạch lạc, có cốt truyện và bảo toàn định danh nhân vật.

7.  **META_REFLECTION**

    Tập hợp các bài báo tháng 07/2024 cho thấy một số xu hướng phát triển AI nổi bật.
    Thứ nhất, **khả năng xử lý ngữ cảnh dài (long context)** tiếp tục là một trọng tâm lớn, với nhiều phương pháp mới lạ được đề xuất, từ các kiến trúc lai hiệu quả (GoldFinch `2407.12077`, ARMT `2407.04841`), cơ chế lấy cảm hứng từ nhận thức (EM-LLM `2407.09450`), đến các kỹ thuật tối ưu hóa suy luận (LazyLLM `2407.14057`, MInference `2407.02490`). Điều này phản ánh nhu cầu ngày càng tăng về các mô hình có khả năng hiểu và suy luận trên lượng thông tin lớn.

    Thứ hai, **hệ thống AI tạo tác (Embodied AI) và các Agent tự hành** đang có những bước tiến đáng kể, không chỉ trong việc phát triển các agent có khả năng tương tác phức tạp với môi trường số (OpenHands `2407.16741`, AppWorld `2407.18901`, MindSearch `2407.20183`, IoA `2407.07061`) và vật lý (ROS-LLM `2406.19741`, Theia `2407.20179`, DAAG `2407.20798`), mà còn trong việc xây dựng các môi trường mô phỏng và benchmark ngày càng tinh vi (AppWorld, GRUtopia `2407.10943`). Xu hướng "code as planning" hoặc LLM điều phối các công cụ/mô hình chuyên biệt đang trở nên phổ biến.

    Thứ ba, **dữ liệu vẫn là yếu tố then chốt**, với nhiều nghiên cứu tập trung vào các phương pháp tạo, tăng cường, và quản lý dữ liệu một cách thông minh. Điều này bao gồm việc tổng hợp dữ liệu hướng dẫn quy mô lớn (Persona-driven `2406.20094`, AgentInstruct `2407.03502`, Skywork-MathQA `2407.08348`), cải thiện dữ liệu tiền huấn luyện cho VLM (VILA2 `2407.17453`), tạo dữ liệu chuyên biệt cho các miền cụ thể (MAVIS `2407.08739` cho toán trực quan, dữ liệu "unblinded" `2406.19280` cho VQA y tế), và tối ưu hóa hỗn hợp dữ liệu tiền huấn luyện (REGMIX `2407.01492`, MASSIVE DS `2407.12854`).

    Thứ tư, **đánh giá (Evaluation) ngày càng được chú trọng và đi vào chiều sâu**, với sự ra đời của nhiều benchmark mới nhằm đo lường các năng lực cụ thể và giải quyết các hạn chế của benchmark hiện có. Các hướng tiếp cận bao gồm đánh giá chi tiết không gian của VLM (BlindTest `2407.06581`), suy luận toán học trực quan (WE-MATH `2407.01284`), tóm tắt ngữ cảnh dài và trích dẫn (SummHay `2407.01370`), tính kết hợp trong T2V (T2V-CompBench `2407.14505`), khả năng tự sửa lỗi (C2-EVAL `2407.16637`), và các năng lực tách biệt của agent (MMAU `2407.18961`). Nỗ lực chống nhiễm bẩn dữ liệu (LiveBench `2407.12772`) cũng là một điểm đáng chú ý.

    Thứ năm, **hiệu quả tính toán (Efficiency)** vẫn là một dòng chảy quan trọng, thể hiện qua các nghiên cứu về mô hình low-bitwidth (TriLM `2407.12327`), tối ưu hóa suy luận trên CPU (`2407.07304`), các phương pháp PEFT mới cho MoE (ESFT `2407.01906`), và các kiến trúc backbone thị giác hiệu quả (MambaVision `2407.08083`, MoNE `2407.19985`).

    Cuối cùng, có sự quan tâm ngày càng tăng đến các **khía cạnh an toàn, tin cậy và diễn giải** của LLM, bao gồm unlearning (`2407.10058`), tấn công backdoor vào RAG (`2407.12784`), và các khảo sát sâu về cơ chế kiến thức (`2407.15017`). Việc cải thiện khả năng suy luận toán học thông qua các phương pháp DPO có giám sát theo bước (SCDPO `2407.00782`, DPO-augmented Self-Training `2407.18248`) cũng cho thấy nỗ lực nâng cao độ tin cậy trong các tác vụ phức tạp.
