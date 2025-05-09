1.  **TOPIC_TREE**

    *   Generative Models
        *   Image Generation
            *   Unified & Multitask Models
                *   `2409.11340` | Hợp nhất nhiều tác vụ tạo ảnh (văn bản-sang-ảnh, chỉnh sửa, theo chủ thể) vào một mô hình Transformer duy nhất với đầu vào đa phương thức xen kẽ, loại bỏ mô-đun chuyên biệt.
            *   Diffusion-based Image Editing
                *   `2409.01322` | Chỉnh sửa ảnh thực không cần tuning bằng cách sử dụng hàm năng lượng bảo toàn bố cục và cơ chế tự động điều chỉnh tỷ lệ nhiễu để cân bằng các tín hiệu dẫn hướng.
            *   Personalized Image Generation
                *   `2409.13346` | Tạo ảnh cá nhân hóa không cần tinh chỉnh bằng cơ chế tạo cặp dữ liệu tổng hợp (SynPairs) và kiến trúc chú ý song song với nhiều bộ mã hóa văn bản và một bộ mã hóa hình ảnh có thể huấn luyện.
            *   Efficient Architectures for Diffusion Models
                *   `2409.02097` | Thay thế self-attention trong mô hình khuếch tán bằng cơ chế chú ý tuyến tính (LinFusion) dựa trên Mamba phi nhân quả và chuẩn hóa nhận biết độ phân giải để tạo ảnh siêu phân giải hiệu quả.
            *   Post-Training Optimization for Diffusion Models
                *   `2409.17565` | Cải thiện chi tiết tần số cao trong mô hình khuếch tán ẩn (LDM) bằng cách bổ sung hàm mục tiêu trong không gian pixel vào quá trình hậu huấn luyện (SFT, SimPO).
        *   Video Generation
            *   Audio-Driven Portrait Animation
                *   `2409.02634` | Tạo video chân dung nói chuyện từ âm thanh không cần ràng buộc không gian, dựa trên kiến trúc temporal kép, module mở rộng trường tiếp nhận thời gian và cơ chế ánh xạ âm thanh vào không gian tiềm ẩn chuyển động.
            *   Controllable Character Video Synthesis
                *   `2409.16160` | Tạo video nhân vật có thể điều khiển bằng cách phân tách video thành các lớp không gian phân cấp (người, nền, vật che khuất) dựa trên độ sâu 3D, sử dụng mã cơ thể có cấu trúc và mã hóa nhận dạng dạng chuẩn, tích hợp vào mô hình khuếch tán.
            *   Video Depth Estimation
                *   `2409.02095` | Ước tính độ sâu video dài và nhất quán bằng mô hình khuếch tán video, sử dụng chiến lược huấn luyện 3 giai đoạn và chiến lược suy luận ghép nối đoạn chồng chéo.
            *   Lineart Video Colorization
                *   `2409.12960` | Tô màu video hoạt hình đường nét từ một khung tham chiếu bằng cách mở rộng ControlNet cho video, sử dụng Reference Attention để truyền màu và cơ chế lấy mẫu tuần tự để đảm bảo nhất quán video dài.
        *   3D Generation
            *   Native 3D Diffusion Models
                *   `2409.12957` | Tạo tài sản 3D PBR chất lượng cao bằng mô hình khuếch tán hoạt động trên không gian tiềm ẩn của PrimX, một biểu diễn 3D mới dựa trên primitive mã hóa hình dạng, màu sắc và vật liệu.
            *   Feed-forward Text-to-3D
                *   `2409.03718` | Tạo mesh 3D từ văn bản bằng mô hình khuếch tán ảnh hình học (GIMDiffusion), sử dụng cơ chế Collaborative Control để tạo đồng thời ảnh hình học và albedo trong không gian UV.
            *   Image-to-3D Generation
                *   `2409.07452` | Sinh đối tượng 3D độ phân giải cao từ ảnh đơn bằng cách định hình lại bài toán thành sinh video quỹ đạo 3D-aware với mô hình khuếch tán video hai giai đoạn và quy trình tái tạo 3D sử dụng Gaussian Splatting.
            *   Reference-Augmented Diffusion Models
                *   `2409.11406` | Tạo sinh 3D đa chế độ (văn bản, ảnh, 3D) dựa trên tham chiếu 3D lệch pha, sử dụng meta-ControlNet, định tuyến tham chiếu động và tăng cường tự tham chiếu.
        *   Audio Generation
            *   Music Generation
                *   `2409.09214` | Hợp nhất mô hình ngôn ngữ tự hồi quy và khuếch tán để sinh/chỉnh sửa nhạc chất lượng cao, có kiểm soát, sử dụng audio token, renderer phân cấp với DiT và "lead sheet tokens".
                *   `2409.00587` | Áp dụng kiến trúc Transformer đa modal (Flux/MMDiT) và huấn luyện Rectified Flow cho bài toán sinh nhạc từ văn bản trên không gian ẩn VAE, kết hợp nhiều bộ mã hóa văn bản.
                *   `2409.06029` | Tạo sinh bài hát (giọng hát và nhạc đệm) từ lời bằng mô hình ngôn ngữ chuỗi kép (DSLM) với chú ý chéo hai chiều và các chiến lược mặt nạ chú ý đa tác vụ.
            *   Text-to-Audio Synthesis
                *   `2409.10819` | Sinh âm thanh từ văn bản trực tiếp trên không gian ẩn 1D của VAE dạng sóng bằng Diffusion Transformer (EzAudio-DiT) tối ưu hóa với AdaLN-SOLA, kết nối tắt dài và chiến lược huấn luyện 3 giai đoạn.
            *   Neural Audio Codec
                *   `2409.13216` | Nén nhạc bitrate cực thấp (0.35kbps) bằng MuCodec, tích hợp MuEncoder (học âm học và ngữ nghĩa lời), RVQ và flow-matching trên đặc trưng Mel-VAE.
        *   Autoregressive Visual Generation
            *   `2409.04410` | Tích hợp bộ mã hóa hình ảnh/video Lookup-Free Quantization (LFQ) vào mô hình tự hồi quy thuần túy bằng cách sử dụng phân tách token bất đối xứng và dự đoán token con kế tiếp để xử lý từ điển mã hóa siêu lớn.
    *   Large Language Models (LLMs)
        *   Code Intelligence
            *   `2409.12186` | Xây dựng bộ mô hình Qwen2.5-Coder với quy mô và chất lượng dữ liệu lớn, quy trình huấn luyện đa giai đoạn và chiến lược pha trộn dữ liệu tối ưu cho code.
            *   `2409.03810` | Chọn lọc dữ liệu hiệu quả cho tinh chỉnh hướng dẫn mã nguồn dựa trên độ phức tạp hướng dẫn, chất lượng phản hồi (đánh giá bằng unit test tự sinh) và tính đa dạng.
        *   Training & Fine-tuning
            *   RL for LLMs
                *   `2409.12917` | Huấn luyện khả năng tự sửa lỗi của LLM bằng phương pháp học tăng cường trực tuyến đa lượt (SCoRe) chỉ sử dụng dữ liệu tự tạo, với quy trình hai giai đoạn chống sụp đổ hành vi.
            *   Alignment & Preference Learning
                *   `2409.02795` | Khảo sát và đề xuất khung nhìn thống nhất phân loại các phương pháp học sở thích cho LLM, làm rõ mối quan hệ giữa các phương pháp dựa trên RL và SFT.
                *   `2409.11564` | Khảo sát các phương pháp tinh chỉnh theo sở thích (preference tuning) dựa trên phản hồi con người cho các phương thức ngôn ngữ, giọng nói và thị giác, phân loại thành online và offline.
            *   Efficient Training & Adaptation
                *   `2409.00509` | Mở rộng cửa sổ ngữ cảnh LLM hiệu quả về chi phí bằng cách phân tích token tác động cao, biến đổi chỉ số vị trí trên đoạn ngắn và tối ưu hóa huấn luyện (replay, model merging).
                *   `2409.12903` | Khởi tạo trọng số LLM lớn từ LLM nhỏ hơn đã huấn luyện bằng cách mở rộng chiều ẩn (HyperCloning) với quy tắc bảo toàn chức năng, giúp tăng tốc hội tụ và cải thiện độ chính xác.
            *   Instruction Tuning Data Selection
                *   `2409.08239` | Tạo dữ liệu tổng hợp có kèm bước suy luận trung gian (Source2Synth) từ nguồn dữ liệu thực tế, với chiến lược quản lý chất lượng dựa trên mô hình để dạy LLM các kỹ năng phức tạp.
            *   Domain Adaptation
                *   `2408.15545` | Thích ứng LLM cho hiểu văn bản khoa học bằng cách kết hợp huấn luyện trước liên tục (CPT) trên kho ngữ liệu khoa học đã xử lý và tinh chỉnh có giám sát (SFT) trên dữ liệu chỉ dẫn tổng hợp dựa trên LLM.
                *   `2409.14988` | Khảo sát hiệu quả của huấn luyện tiếp tục trên dữ liệu lâm sàng quy mô lớn, kết hợp tinh chỉnh theo chỉ dẫn và NEFTune để tối ưu LLM cho miền y tế.
        *   Model Architectures
            *   Mixture-of-Experts (MoE)
                *   `2409.02060` | Xây dựng và công bố mô hình MoE (OLMoE) hoàn toàn mở, thực hiện các thí nghiệm có kiểm soát để đánh giá các lựa chọn thiết kế MoE và phân tích hành vi expert.
            *   Modular Models
                *   `2409.02877` | Đề xuất khung khái niệm "Configurable Foundation Models" xem LLM như tập hợp các "bricks" chức năng (emergent/customized) và hệ thống hóa các thao tác trên chúng.
        *   Inference Optimization
            *   Attention Approximation
                *   `2409.10516` | Giảm tải KV cache sang CPU và truy xuất động bằng chỉ mục ANNS "nhận biết attention" (RetrievalAttention) để giải quyết vấn đề OOD giữa query và key trong ngữ cảnh dài.
            *   Input Context Reduction
                *   `2409.17422` | Sử dụng các lớp đầu của LLM làm bộ lọc (GemFilter) để chọn và nén token đầu vào, giảm độ dài ngữ cảnh cho các lớp sau trong xử lý ngữ cảnh dài.
        *   Model Compression
            *   Pruning
                *   `2409.17481` | Học mặt nạ tỉa thưa bán cấu trúc N:M (MaskLLM) cho LLM bằng cách tối ưu hóa phân phối mặt nạ end-to-end dựa trên Gumbel-Softmax và hàm mất mát ngôn ngữ.
            *   Quantization
                *   `2409.17066` | Lượng tử hóa vector trọng số LLM xuống bit cực thấp (VPTQ) bằng tối ưu hóa bậc hai độc lập kênh, khởi tạo codebook bằng K-means có trọng số Hessian và tích hợp RVQ/Outlier VQ.
        *   Personalization
            *   `2409.11901` | Cá nhân hóa LLM bằng mô-đun user embedder nhẹ, plug-and-play (PPlug) sử dụng bộ mã hóa hành vi người dùng và bộ tổng hợp cá nhân nhận biết đầu vào để điều hướng LLM cố định.
        *   Multilingual Models
            *   `2409.16235` | Xây dựng và mô tả quy trình huấn luyện mô hình ngôn ngữ lớn đa ngôn ngữ (EuroLLM-1.7B) tập trung vào các ngôn ngữ Châu Âu, sử dụng các kỹ thuật hiện có về xử lý dữ liệu, kiến trúc và huấn luyện.
        *   Alignment and Safety
            *   Honesty / Truthfulness
                *   `2409.18786` | Khảo sát và hệ thống hóa các nghiên cứu về "tính trung thực" của LLM, phân loại thành tự nhận thức và tự biểu đạt, cùng các phương pháp đánh giá và cải thiện.
        *   Sequence Modeling (Efficient Transformers)
            *   `2409.07146` | Đề xuất Gated Slot Attention (GSA), một biến thể của ABC với cơ chế gating data-dependent, có thể biểu diễn dưới dạng Gated Linear Attention hai lượt để huấn luyện hiệu quả.
    *   Multimodal AI
        *   Vision-Language Models (VLMs / MLLMs)
            *   Open Foundation VLMs
                *   `2409.17146` | Xây dựng VLM (Molmo) và bộ dữ liệu (PixMo) hoàn toàn mở, không dựa vào chưng cất từ mô hình độc quyền, với các quy trình thu thập dữ liệu sáng tạo (giọng nói, điểm 2D) và cải tiến huấn luyện/kiến trúc.
            *   Unified Vision-Language Models
                *   `2409.18869` | Đề xuất Emu3, kiến trúc Transformer đơn lẻ chỉ dựa trên dự đoán token tiếp theo cho cả tạo sinh và nhận thức đa phương thức (ảnh, video, văn bản), loại bỏ mô hình khuếch tán hoặc kiến trúc kết hợp.
            *   Dynamic Resolution Input Processing
                *   `2409.12191` | Xử lý ảnh ở độ phân giải bất kỳ trong LVLM (Qwen2-VL) bằng Naive Dynamic Resolution (ViT với 2D-RoPE) và mã hóa vị trí đa phương thức bằng M-RoPE.
            *   Architectures and Training Strategies for MLLMs
                *   `2409.11402` | Đề xuất kiến trúc MLLM lai NVLM-H (kết hợp self-attention và cross-attention), kỹ thuật tile-tagging 1-D cho ảnh phân giải cao và chiến lược tích hợp dữ liệu văn bản vào SFT đa phương thức.
                *   `2409.04828` | Cải thiện VLM bằng chiến lược chia ảnh động duy trì tỷ lệ khung hình (CATTY), lọc dữ liệu tiền huấn luyện bằng perplexity và kết hợp mô hình bằng model soup dựa trên tập dữ liệu.
            *   Long-Context MLLMs
                *   `2409.02889` | Đề xuất kiến trúc MLLM lai Transformer-Mamba (LongLLaVA) đầu tiên để xử lý hiệu quả ngữ cảnh dài đa hình ảnh, kết hợp nén token ảnh và chiến lược huấn luyện đa giai đoạn.
            *   Any-to-Any Models
                *   `2409.17692` | Xây dựng MIO, mô hình nền tảng any-to-any mã nguồn mở hợp nhất hiểu và sinh đa phương thức (văn bản, ảnh, giọng nói, video) end-to-end tự hồi quy, hỗ trợ sinh chuỗi xen kẽ và tích hợp sinh token âm sắc.
            *   Omni-modal Models
                *   `2409.18042` | Đề xuất EMOVA, kiến trúc omni-modal kết hợp bộ mã hóa thị giác liên tục với tokenizer/detokenizer giọng nói rời rạc tách biệt ngữ nghĩa-âm học, cho phép kiểm soát cảm xúc giọng nói và huấn luyện khớp nối lấy văn bản làm trung tâm.
            *   Architectures for Flexible Visual Input Processing
                *   `2409.12961` | Phát triển OryxViT, bộ mã hóa thị giác xử lý độ phân giải bất kỳ và Dynamic Compressor nén token thị giác theo yêu cầu (1x-16x) cho MLLM.
            *   Efficient LLVM Architectures and Training
                *   `2409.14713` | Tăng cường khả năng học của LLVM bằng Phantom Dimension (tăng tạm thời chiều ẩn trong MHSA) và Phantom Optimization (SFT kết hợp DPO-like) dựa trên Phantom Triples.
        *   Vision-Language Understanding
            *   Comics Understanding
                *   `2409.09502` | Khảo sát và đề xuất khung phân loại Layer of Comics Understanding (LoCU) để hệ thống hóa các nhiệm vụ hiểu truyện tranh từ góc độ Thị giác-Ngôn ngữ.
            *   Satire Comprehension
                *   `2409.13592` | Giới thiệu bộ dữ liệu YesBut và ba nhiệm vụ đánh giá năng lực hiểu châm biếm của mô hình VL, với quy trình tạo dữ liệu kết hợp thu thập thủ công và sinh ảnh bằng DALL-E 3.
        *   Visual Document Understanding
            *   OCR-free Document Understanding
                *   `2409.03420` | Nén token thị giác cho ảnh tài liệu độ phân giải cao (DocOwl2) bằng High-resolution DocCompressor dựa trên cross-attention giữa đặc trưng toàn cục và chi tiết, sau tầng căn chỉnh vision-to-text.
            *   Document Layout Analysis
                *   `2409.18839` | Xây dựng hệ thống trích xuất nội dung tài liệu đa dạng (MinerU) bằng quy trình kỹ thuật dữ liệu lặp lại cho mô hình phát hiện bố cục và chiến lược OCR kết hợp che công thức.
        *   Optical Character Recognition (OCR)
            *   `2409.01704` | Đề xuất lý thuyết OCR Tổng quát (OCR-2.0) và xây dựng mô hình GOT (General OCR Transformer) end-to-end xử lý đa dạng "ký tự" (văn bản, công thức, bảng, nhạc) với encoder nén cao và decoder ngữ cảnh dài.
        *   Medical Image Analysis
            *   `2409.01437` | Tạo bộ dữ liệu Kvasir-VQA (ảnh nội soi tiêu hóa với chú thích Q&A) và ứng dụng các mô hình nền tảng (Florence-2, SD3) cho chú thích, VQA y tế và sinh ảnh.
        *   Dense Prediction
            *   `2409.18124` | Tinh chỉnh mô hình khuếch tán tiền huấn luyện cho dự đoán dày đặc (Lotus) bằng x0-prediction, quy trình khuếch tán một bước và cơ chế "detail preserver" để giữ chi tiết.
        *   Affordance Learning
            *   `2409.06210` | Định vị affordance yếu giám sát chỉ dùng ảnh ngoại tâm (INTRA) bằng cách học biểu diễn có điều kiện văn bản, học tương phản có hướng dẫn bởi mối quan hệ tương tác và tăng cường từ đồng nghĩa.
        *   Video-Language Modeling
            *   Long Video Understanding
                *   `2409.01071` | Xử lý video dài (VideoLLaMB) bằng cách phân đoạn video theo ngữ nghĩa (SceneTilling), sử dụng lớp cầu nối bộ nhớ đệ quy và cơ chế Memory Cache với truy xuất cross-attention.
    *   AI for Science
        *   Climate and Weather Modeling
            *   `2409.13598` | Xây dựng mô hình nền tảng thời tiết/khí hậu (Prithvi WxC) với mục tiêu tiền huấn luyện hỗn hợp (masking + dự báo độ lệch khí hậu học) và kiến trúc transformer 2D phân cấp (chú ý cục bộ/toàn cục).
        *   Time Series Forecasting
            *   `2409.08240` | Dự báo chuỗi thời gian zero-shot (VISION TS) bằng cách chuyển đổi bài toán thành tái tạo ảnh theo patch với mô hình visual masked autoencoder (MAE) tiền huấn luyện trên ảnh.
    *   Robotics
        *   Robot Learning
            *   Imitation Learning
                *   `2409.14674` | Tăng cường tính bền vững của robot (RACER) bằng cách tạo dữ liệu phục hồi lỗi tự động với chú thích ngôn ngữ phong phú (từ LLM) và kiến trúc supervisor (VLM) - actor (chính sách vận động).
            *   Reinforcement Learning for Robotics
                *   `2409.00588` | Tinh chỉnh Diffusion Policy bằng policy gradient (DPPO) thông qua việc biểu diễn khử nhiễu như một MDP và áp dụng PPO cho MDP hai lớp (môi trường và khử nhiễu).
    *   Information Retrieval & Question Answering
        *   Dense Retrieval
            *   Instruction-based Retrieval
                *   `2409.11136` | Tạo mô hình truy xuất tuân theo chỉ dẫn tự nhiên cấp độ mẫu (Promptriever) bằng cách huấn luyện trên dữ liệu tăng cường chỉ dẫn và "instruction negatives" do LLM tạo ra.
            *   Task-Specific Embeddings
                *   `2409.10173` | Tạo embedding chuyên biệt cho từng tác vụ (jina-embeddings-v3) bằng cách sử dụng các bộ điều hợp LoRA riêng biệt thay vì instruction tuning, tích hợp dữ liệu tổng hợp để cải thiện tính bền vững.
        *   Retrieval-Augmented Generation (RAG)
            *   Unified Generation and Retrieval
                *   `2409.05152` | Hợp nhất sinh văn bản và truy xuất vector (OneGen) trong một lượt truyền xuôi LLM duy nhất bằng token truy xuất đặc biệt và huấn luyện đồng thời với mất mát tương phản.
            *   Long-Context Processing
                *   `2409.05591` | Cải thiện RAG ngữ cảnh dài (MemoRAG) bằng bộ nhớ toàn cục nén (Compact Global Memory) tạo gợi ý trả lời nháp và Học Tăng cường với Phản hồi từ Chất lượng Sinh (RLGF).
            *   Multi-step RAG Strategies
                *   `2409.12941` | Thực hiện truy xuất và suy luận đa bước lặp đi lặp lại, trong đó LLM tự tạo truy vấn, truy xuất tài liệu bằng BM25 và tích hợp thông tin qua nhiều vòng để trả lời câu hỏi phức tạp.
            *   Personalized Academic Assistance
                *   `2409.04593` | Xây dựng trợ lý nghiên cứu cá nhân hóa (Paper Copilot) dựa trên RAG, tích hợp hồ sơ người dùng, truy xuất suy nghĩ, tự tiến hóa và các kỹ thuật tối ưu hóa hiệu năng.
        *   Question Answering with Citations
            *   `2409.02897` | Tự động tạo dữ liệu huấn luyện (CoF) cho hỏi đáp ngữ cảnh dài có trích dẫn cấp câu bằng quy trình nhiều bước từ thô đến tinh sử dụng LLM.
    *   AI Evaluation & Benchmarking
        *   Data Science Agent Benchmarking
            *   `2409.07703` | Xây dựng DSBench, benchmark tác vụ khoa học dữ liệu thực tế (ngữ cảnh dài, đa phương thức, end-to-end) và chỉ số RPG để đánh giá tác tử AI.
        *   Conversational AI / Role-Playing LLM Evaluation
            *   `2409.06820` | Đề xuất PingPong, benchmark đánh giá khả năng nhập vai của LLM trong hội thoại động, đa lượt, sử dụng kiến trúc player-interrogator-judge (ensemble) và tạo câu hỏi động.
        *   Clinical LLM Evaluation
            *   `2409.07314` | Đề xuất MEDIC, khung đánh giá toàn diện LLM y tế theo 5 chiều (lý luận, đạo đức, hiểu dữ liệu, học trong ngữ cảnh, an toàn) và phương pháp "cross-examination" (chưa mô tả chi tiết).
            *   `2409.15277` | Đánh giá toàn diện mô hình o1 trong lĩnh vực y tế (hiểu biết, suy luận, đa ngôn ngữ) trên 37 bộ dữ liệu, bao gồm 2 bộ QA mới (LancetQA, NEJMQA).
        *   Operating System Agents Evaluation
            *   `2409.08264` | Xây dựng hạ tầng đánh giá tác tử HĐH Windows quy mô lớn, song song hóa trên Azure, và giới thiệu benchmark WINDOWS AGENT ARENA.
        *   Text Generation Evaluation
            *   `2409.16191` | Đề xuất HelloEval, phương pháp đánh giá tạo văn bản dài hai giai đoạn (học trọng số checklist từ người, áp dụng LLM-as-a-Judge) và benchmark HelloBench.
        *   Multimodal Understanding & Reasoning Evaluation
            *   `2409.02813` | Xây dựng benchmark MMMU-Pro bằng cách lọc câu hỏi chỉ cần văn bản, tăng cường lựa chọn và giới thiệu định dạng đầu vào chỉ có hình ảnh (vision-only) cho MMMU.
            *   `2409.08267` | Giới thiệu UrBench, benchmark đánh giá LMM trên tác vụ đô thị đa chế độ xem (đường phố-vệ tinh) và đa cấp độ, với phương pháp tạo chú thích đối tượng và câu hỏi tích hợp.
        *   Evaluating LLM Capabilities
            *   `2409.04109` | Thiết lập quy trình thử nghiệm quy mô lớn để đánh giá khả năng tạo ý tưởng nghiên cứu của LLM so với chuyên gia, sử dụng agent LLM (RAG, overgeneration, reranking).
        *   Long Context Modeling Evaluation
            *   `2409.12181` | Thực hiện so sánh có kiểm soát các phương pháp mở rộng ngữ cảnh LLM, chuẩn hóa mô hình, dữ liệu, huấn luyện và đánh giá trên nhiều số liệu.
        *   Synthetic Image Detection Generalization Analysis
            *   `2409.14128` | Phân tích hệ thống khả năng tổng quát hóa của bộ phát hiện ảnh tổng hợp, ảnh hưởng của huấn luyện, điều kiện triển khai (nguồn mới, thay đổi tỷ lệ) và đề xuất hướng dẫn.
        *   Chain-of-Thought Evaluation
            *   `2409.12183` | Phân tích tổng hợp và thực nghiệm quy mô lớn về hiệu quả của CoT, chỉ ra lợi ích chủ yếu cho suy luận toán học/logic và hạn chế so với symbolic solver.
        *   Multimodal Integration Benchmarking
            *   `2409.15272` | Giới thiệu OmniBench (đánh giá tích hợp 3 phương thức ảnh-âm thanh-văn bản) và OmniInstruct (dữ liệu huấn luyện OLM) với quy trình chú thích và lọc nghiêm ngặt.
    *   Data Engineering for AI
        *   Pre-training Data Refinement
            *   `2409.17115` | Tinh chỉnh dữ liệu tiền huấn luyện (PROX) bằng cách coi việc này như lập trình, sử dụng LLM nhỏ tạo và thực thi các thao tác chi tiết (chuẩn hóa, xóa dòng) cho từng mẫu.
        *   Multimodal Data Curation
            *   `2409.12568` | Xây dựng InfiMM-WebMath-40B, bộ dữ liệu tiền huấn luyện đa phương thức toán học quy mô lớn công khai từ CommonCrawl, với quy trình trích xuất và lọc nội dung toán học/khoa học.
        *   Instruction Tuning Data Evolution
            *   `2409.05840` | Tự động nâng cao độ phức tạp và đa dạng của dữ liệu hướng dẫn hình ảnh-văn bản (MMEvol) bằng ba chiến lược tiến hóa đa phương thức chuyên biệt và quy trình lặp.
    *   Security
        *   Software Security
            *   Fuzzing
                *   `2409.01944` | Áp dụng LLM (FUZZCODER) vào fuzzing mức byte bằng cách formulate thành mô hình sequence-to-sequence dự đoán vị trí và chiến lược đột biến hiệu quả, huấn luyện trên dữ liệu Fuzz-Instruct.
    *   Software Engineering
        *   Program Synthesis
            *   `2409.08692` | Lựa chọn lời giải code và bộ test đáng tin cậy (B4) bằng cách tối đa hóa xác suất hậu nghiệm Bayes, sử dụng phân phối Beta làm tiên nghiệm liên hợp và giảm không gian tìm kiếm qua tập đồng thuận.
    *   Other
        *   Language Model Adaptation
            *   `2409.14254` | Điều chỉnh hành vi LLM bằng bộ điều hợp dựa trên luật đơn giản kết hợp qua product-of-experts (PoE) để tạo khả năng tuân thủ chỉ dẫn không cần tinh chỉnh.
        *   Deep Learning Architectures
            *   `2409.10594` | Xây dựng Kolmogorov-Arnold Transformer (KAT) bằng cách thay thế MLP trong Transformer bằng khối GR-KAN (KAN với hàm kích hoạt hữu tỉ, chia sẻ tham số theo nhóm, khởi tạo bảo toàn phương sai).
        *   Applications of LMMs
            *   `2409.12959` | Đề xuất quy trình MMSearch-Engine cho công cụ tìm kiếm AI đa phương thức, cho phép LMM thực hiện tìm kiếm kết hợp văn bản và hình ảnh qua các bước requery, rerank, summarize.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                  |
    | :--- | :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | `2409.18869` | Emu3, Autoregressive, Multimodal, Unified, Next-token | Kiến trúc Transformer đơn lẻ, chỉ dựa trên dự đoán token tiếp theo, đạt SOTA cho cả tạo sinh và nhận thức đa phương thức (ảnh, video, văn bản). | Đơn giản hóa đáng kể kiến trúc MLLM, tiềm năng trở thành hướng đi chủ đạo cho mô hình đa phương thức tổng quát.                               |
    | 2    | `2409.12186` | Qwen2.5-Coder, Code LLM, Data Curation, Multi-stage | Bộ mô hình code SOTA với quy mô dữ liệu và quy trình huấn luyện đa giai đoạn được thiết kế cẩn thận, công thức xử lý/pha trộn dữ liệu.     | Cung cấp mô hình code nguồn mở mạnh mẽ, thúc đẩy nghiên cứu và ứng dụng LLM cho lập trình.                                                    |
    | 3    | `2409.11340` | OmniGen, Unified Image Generation, Multimodal Input  | Mô hình Transformer duy nhất xử lý đa dạng tác vụ tạo ảnh qua đầu vào đa phương thức xen kẽ, loại bỏ mô-đun chuyên biệt.                 | Hướng tới "LLM cho tạo ảnh", đơn giản hóa và thống nhất hóa các tác vụ tạo sinh hình ảnh phức tạp.                                            |
    | 4    | `2409.00588` | DPPO, Diffusion Policy, Reinforcement Learning, Robotics | Tinh chỉnh Diffusion Policy hiệu quả bằng policy gradient (PPO) thông qua việc biểu diễn khử nhiễu như một MDP hai lớp.                 | Mở ra hướng mới cho việc cải thiện chính sách robot dựa trên khuếch tán bằng RL, đạt hiệu năng cao trong các tác vụ phức tạp và sim-to-real. |
    | 5    | `2409.17692` | MIO, Any-to-Any, Multimodal, Autoregressive, Speech    | Mô hình nền tảng any-to-any mã nguồn mở đầu tiên hợp nhất hiểu và sinh đa phương thức (văn bản, ảnh, giọng nói, video) end-to-end.      | Thúc đẩy nghiên cứu về các hệ thống AI đa phương thức toàn diện, đặc biệt với khả năng sinh chuỗi đa phương thức xen kẽ và giọng nói có âm sắc. |
    | 6    | `2409.07452` | Hi3D, Image-to-3D, Video Diffusion, High-Resolution  | Sinh 3D từ ảnh đơn độ phân giải cao (1024x1024) bằng cách sử dụng mô hình khuếch tán video tiền huấn luyện và kiến trúc hai giai đoạn.      | Đạt được bước tiến lớn về độ phân giải và chất lượng trong tạo sinh 3D từ ảnh đơn, khai thác hiệu quả mô hình video.                         |
    | 7    | `2409.12957` | PrimX, 3DTopia-XL, PBR, Latent Primitive Diffusion   | Biểu diễn 3D PrimX mã hóa hình dạng, màu sắc, vật liệu PBR và mô hình khuếch tán tiềm ẩn trên primitive để tạo tài sản 3D chất lượng cao. | Cung cấp giải pháp tạo tài sản 3D PBR gốc, có khả năng mở rộng, chất lượng cao, phù hợp cho các ứng dụng đồ họa chuyên nghiệp.                 |
    | 8    | `2409.10594` | KAT, GR-KAN, Kolmogorov-Arnold Networks, Transformer | Xây dựng Kolmogorov-Arnold Transformer (KAT) bằng cách thay thế MLP bằng khối GR-KAN (KAN hiệu quả với hàm hữu tỉ, chia sẻ tham số).     | Giải quyết các hạn chế về khả năng mở rộng của KAN, mở đường cho việc tích hợp KAN vào các kiến trúc lớn như Transformer, cải thiện hiệu năng. |
    | 9    | `2409.02095` | DepthCrafter, Video Depth, Long Video, Diffusion     | Ước tính độ sâu video dài và nhất quán bằng mô hình khuếch tán video, chiến lược huấn luyện 3 giai đoạn và suy luận ghép nối.             | Cung cấp giải pháp mạnh mẽ cho ước tính độ sâu video "in-the-wild" với ngữ cảnh thời gian dài, vượt trội các phương pháp zero-shot trước đó.   |
    | 10   | `2409.17146` | Molmo, PixMo, Open VLM, Data Collection, 2D Point    | Xây dựng VLM (Molmo) và bộ dữ liệu (PixMo) hoàn toàn mở, không chưng cất, với các quy trình thu thập dữ liệu sáng tạo (giọng nói, điểm 2D). | Thúc đẩy nghiên cứu VLM mở, cung cấp tài nguyên chất lượng cao và các phương pháp thu thập dữ liệu mới không phụ thuộc hệ thống độc quyền.     |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2409.12186 – Quy trình lọc dữ liệu text-code phân cấp bằng mô hình nhỏ – Suy nghĩ:** Một cách tiếp cận thực tế để xử lý dữ liệu quy mô lớn mà không cần dựa vào các mô hình lớn, có thể tiết kiệm chi phí tính toán đáng kể.
    *   **2409.12186 – Quy trình huấn luyện ba giai đoạn (file-level, repo-level, instruction tuning) cho mô hình code – Suy nghĩ:** Chiến lược huấn luyện có cấu trúc, giúp mô hình học từ các cấp độ ngữ cảnh khác nhau, tiềm năng cải thiện hiểu biết sâu về code.
    *   **2409.12917 – Phương pháp RL trực tuyến đa lượt SCoRe cho tự sửa lỗi LLM – Suy nghĩ:** Giải quyết vấn đề sụp đổ hành vi và lệch phân phối trong SFT offline một cách thông minh, chỉ dùng dữ liệu tự tạo.
    *   **2409.12917 – Định hình phần thưởng (reward shaping) khuyến khích "sự tiến bộ" (Δi→c) trong sửa lỗi – Suy nghĩ:** Một cơ chế thưởng trực tiếp và hiệu quả để hướng mô hình học hành vi sửa lỗi mong muốn.
    *   **2409.11340 – Kiến trúc OmniGen hợp nhất một Transformer và VAE cho đa tác vụ tạo ảnh – Suy nghĩ:** Đơn giản hóa đáng kể kiến trúc so với các hệ thống đa mô-đun, hướng tới một mô hình tạo ảnh tổng quát hơn.
    *   **2409.11340 – Cơ chế chú ý sửa đổi (causal toàn cục, bidirectional trong chuỗi ảnh) – Suy nghĩ:** Giải pháp thanh lịch để xử lý đầu vào xen kẽ văn bản-hình ảnh trong một Transformer duy nhất.
    *   **2409.17146 – Chiến lược multi-crop có chồng lấp cho ảnh độ phân giải cao trong VLM – Suy nghĩ:** Cải thiện ngữ cảnh cho các vùng rìa một cách hiệu quả mà không làm tăng đáng kể chi phí tính toán đầu vào LLM.
    *   **2409.17146 – Dropout chỉ cho token văn bản trong tiền huấn luyện VLM – Suy nghĩ:** Một kỹ thuật regularization thông minh để buộc mô hình dựa nhiều hơn vào thông tin thị giác.
    *   **2409.17146 – Quy trình thu thập dữ liệu chú thích điểm 2D (PixMo-Points) cho VLM – Suy nghĩ:** Mở ra khả năng grounding, đếm và giải thích trực quan mới cho VLM, vượt ra ngoài bounding box/mask.
    *   **2409.02634 – Kiến trúc temporal kép (inter-clip, intra-clip) cho video chân dung – Suy nghĩ:** Phân tách xử lý phụ thuộc thời gian giúp mô hình hóa chuyển động phức tạp tốt hơn.
    *   **2409.02634 – Temporal Segment Module (TSM) mở rộng trường tiếp nhận thời gian hiệu quả – Suy nghĩ:** Một cách thông minh để nắm bắt ngữ cảnh dài hạn mà không tăng quá nhiều chi phí tính toán.
    *   **2409.02634 – Mô-đun Audio-to-Latents ánh xạ đặc trưng âm thanh và phương sai chuyển động vào không gian tiềm ẩn chung – Suy nghĩ:** Tăng cường liên kết âm thanh-chuyển động yếu bằng cách học từ các tín hiệu tương quan mạnh hơn.
    *   **2409.01322 – Hàm năng lượng bảo toàn bố cục (guiders) cho chỉnh sửa ảnh khuếch tán – Suy nghĩ:** Cung cấp cơ chế kiểm soát cấu trúc ảnh gốc một cách linh hoạt trong quá trình chỉnh sửa.
    *   **2409.01322 – Cơ chế điều chỉnh lại tỷ lệ nhiễu tự động giữa CFG và guiders – Suy nghĩ:** Giải quyết vấn đề xung đột tín hiệu dẫn hướng, giúp ổn định quá trình sinh ảnh.
    *   **2409.18869 – Vision tokenizer dựa trên MoVQGAN với lớp tích chập 3D cho ảnh và video – Suy nghĩ:** Một giải pháp hiệu quả để mã hóa cả ảnh và video thành token rời rạc cho mô hình tự hồi quy.
    *   **2409.18869 – Áp dụng DPO cho tạo sinh hình ảnh/video tự hồi quy – Suy nghĩ:** Mở rộng thành công DPO sang miền tạo sinh đa phương thức tự hồi quy, hứa hẹn cải thiện chất lượng theo sở thích người dùng.
    *   **2409.03752 – Khung phân loại 4 giai đoạn (KR-ICI-LR-EP) cho chức năng đầu chú ý LLM – Suy nghĩ:** Một cách tiếp cận mới lạ, lấy cảm hứng từ khoa học thần kinh nhận thức, để hệ thống hóa hiểu biết về cơ chế LLM. (Đóng góp khái niệm)
    *   **2409.01704 – Encoder nén cao (VitDet-based) cho OCR, chuyển ảnh 1024x1024 thành 256 token – Suy nghĩ:** Tỷ lệ nén ấn tượng, phù hợp cho việc xử lý tài liệu dày đặc ký tự trong kiến trúc encoder-decoder.
    *   **2409.01704 – Quy trình tổng hợp dữ liệu công thức toán tốc độ cao (Chorme-driver + Mathpix-markdown-it) – Suy nghĩ:** Giải pháp thực tế và hiệu quả để tạo dữ liệu quy mô lớn cho OCR công thức.
    *   **2409.02060 – Dropless token choice routing cho MoE – Suy nghĩ:** Đảm bảo mọi token được xử lý, có thể cải thiện hiệu quả sử dụng expert so với các cơ chế có bỏ token. (Đánh giá thực nghiệm)
    *   **2409.12191 – Naive Dynamic Resolution cho LVLM (ViT với 2D-RoPE) – Suy nghĩ:** Cho phép xử lý ảnh ở độ phân giải gốc một cách linh hoạt, giữ lại chi tiết tốt hơn.
    *   **2409.12191 – Multimodal Rotary Position Embedding (M-RoPE) – Suy nghĩ:** Mã hóa tường minh thông tin vị trí không gian-thời gian cho các token đa phương thức, cải thiện hiểu biết cấu trúc.
    *   **2409.11402 – Kiến trúc MLLM lai NVLM-H (self-attention + cross-attention) – Suy nghĩ:** Kết hợp ưu điểm của hai cơ chế tích hợp thị giác, tiềm năng cân bằng hiệu quả và hiệu năng.
    *   **2409.11402 – Tile-tagging 1-D dựa trên văn bản cho ảnh độ phân giải cao dạng ô – Suy nghĩ:** Một cách đơn giản để LLM nhận biết nguồn gốc thông tin từ các ô ảnh khác nhau.
    *   **2409.02795 – Khung nhìn thống nhất về học sở thích LLM (gradient chung cho RL/SFT) – Suy nghĩ:** Cung cấp một lăng kính mới để hiểu mối liên hệ giữa các phương pháp căn chỉnh, thúc đẩy sự kết hợp. (Đóng góp khái niệm)
    *   **2409.01437 – Quy trình tạo dữ liệu VQA tổng hợp bằng LLaMA-3 từ chú thích mô tả – Suy nghĩ:** Một cách hiệu quả để mở rộng bộ dữ liệu VQA y tế mà không cần chú thích thủ công hoàn toàn. (Đóng góp dữ liệu)
    *   **2409.13346 – Cơ chế tạo dữ liệu cặp tổng hợp SynPairs cho cá nhân hóa ảnh – Suy nghĩ:** Giải quyết vấn đề thiếu dữ liệu cặp (cùng danh tính, khác tư thế/biểu cảm) một cách sáng tạo.
    *   **2409.13346 – Kiến trúc chú ý song song với bộ mã hóa ảnh có thể huấn luyện và ba bộ mã hóa văn bản – Suy nghĩ:** Thiết kế phức tạp nhưng có tiềm năng tận dụng điểm mạnh của từng bộ mã hóa cho việc bảo toàn danh tính và tuân thủ prompt.
    *   **2409.07703 – Chỉ số Relative Performance Gap (RPG) cho benchmark khoa học dữ liệu – Suy nghĩ:** Một cách hợp lý để chuẩn hóa đánh giá hiệu năng trên các tác vụ mô hình hóa có chỉ số gốc khác nhau. (Đóng góp đánh giá)
    *   **2409.06820 – Kiến trúc đánh giá nhập vai LLM ba vai trò (player, interrogator, multi-model judge ensemble) – Suy nghĩ:** Tăng tính khách quan và giảm thiên vị trong đánh giá tự động khả năng nhập vai.
    *   **2409.17115 – Framework PROX: tinh chỉnh dữ liệu như lập trình, LLM nhỏ tạo/thực thi lệnh – Suy nghĩ:** Hướng tiếp cận mới lạ, chi tiết và có khả năng kiểm soát cao cho việc làm sạch dữ liệu tiền huấn luyện.
    *   **2409.06666 – Bộ giải mã giọng nói streaming NAR-CTC từ trạng thái ẩn LLM – Suy nghĩ:** Cho phép tạo giọng nói đồng thời với văn bản, độ trễ thấp, tích hợp chặt chẽ vào LLM.
    *   **2409.07314 – Khung đánh giá MEDIC 5 chiều cho LLM y tế – Suy nghĩ:** Cung cấp một cấu trúc toàn diện để đánh giá LLM trong ứng dụng lâm sàng, vượt ra ngoài các benchmark đơn lẻ. (Đóng góp khái niệm)
    *   **2409.02889 – Kiến trúc MLLM lai Transformer-Mamba (LongLLaVA) – Suy nghĩ:** Giải pháp kiến trúc tiềm năng để cân bằng hiệu quả tính toán và khả năng học trong ngữ cảnh dài đa phương thức.
    *   **2409.17692 – Chiến lược tiền huấn luyện MIO ba giai đoạn (Căn chỉnh, Xen kẽ, Tăng cường giọng nói) – Suy nghĩ:** Phương pháp thực tế để quản lý tích hợp các phương thức có mật độ token và phụ thuộc khác nhau.
    *   **2409.17692 – Tích hợp sinh token âm sắc giọng nói tự hồi quy trực tiếp vào LLM – Suy nghĩ:** Đơn giản hóa việc tạo giọng nói có âm sắc, tăng tính end-to-end.
    *   **2409.09214 – Renderer phân cấp (DiT chuyển audio token thành vocoder latents) cho sinh nhạc – Suy nghĩ:** Cải thiện chất lượng âm thanh bằng cách thêm bước trung gian có kiểm soát.
    *   **2409.09214 – "Lead sheet tokens" làm biểu diễn symbolic mới cho sinh nhạc diffusion – Suy nghĩ:** Tăng khả năng diễn giải và kiểm soát trong quá trình sinh nhạc.
    *   **2409.08264 – Hạ tầng đánh giá tác tử HĐH Windows song song hóa trên Azure – Suy nghĩ:** Giải pháp kỹ thuật quan trọng để tăng tốc đáng kể quá trình benchmark tác tử HĐH. (Đóng góp hạ tầng)
    *   **2409.05840 – Ba chiến lược tiến hóa đa phương thức (Nhận thức Chi tiết, Suy luận Nhận thức, Tương tác) cho dữ liệu hướng dẫn – Suy nghĩ:** Các chiến lược cụ thể và có mục tiêu để tự động cải thiện chất lượng dữ liệu MLLM.
    *   **2409.17481 – Học mặt nạ N:M khả vi bằng Gumbel-Softmax cho tỉa thưa LLM – Suy nghĩ:** Cho phép tối ưu hóa end-to-end cấu trúc thưa dựa trên hàm mất mát ngôn ngữ, vượt trội các phương pháp one-shot.
    *   **2409.04109 – Agent LLM tạo ý tưởng nghiên cứu (RAG, overgeneration, LLM ranker) – Suy nghĩ:** Một sự kết hợp các kỹ thuật hiện có để tạo ra một đối chứng AI mạnh mẽ cho việc đánh giá khả năng tạo ý tưởng. (Ứng dụng kỹ thuật hiện có)
    *   **2409.02897 – Quy trình CoF tự động tạo dữ liệu LQAC cấp câu từ thô đến tinh bằng LLM – Suy nghĩ:** Giải pháp hiệu quả để tạo dữ liệu SFT chất lượng cao cho trích dẫn chi tiết trong ngữ cảnh dài.
    *   **2409.10594 – Hàm kích hoạt hữu tỉ (Safe PAU) thay B-spline trong KAN với cài đặt CUDA tối ưu – Suy nghĩ:** Cải thiện đáng kể hiệu quả tính toán của KAN trên GPU.
    *   **2409.10594 – Group KAN (GR-KAN) chia sẻ tham số hàm cơ sở theo nhóm kênh – Suy nghĩ:** Giảm mạnh số lượng tham số và chi phí tính toán của KAN, giúp mở rộng quy mô.
    *   **2409.01944 – Formulate fuzzing mức byte thành sequence-to-sequence dự đoán (vị trí, chiến lược) đột biến – Suy nghĩ:** Một cách nhìn mới để áp dụng LLM vào việc hướng dẫn fuzzing một cách thông minh.
    *   **2409.12181 – Giao thức có kiểm soát để so sánh các phương pháp mở rộng ngữ cảnh LLM – Suy nghĩ:** Cung cấp một phương pháp luận chuẩn hóa rất cần thiết để đánh giá công bằng các kỹ thuật mở rộng ngữ cảnh. (Đóng góp phương pháp luận đánh giá)
    *   **2409.14674 – Quy trình tăng cường dữ liệu phục hồi lỗi robot với chú thích ngôn ngữ phong phú từ LLM và heuristic tư thế – Suy nghĩ:** Tạo dữ liệu chất lượng cao và đa dạng cho việc học các hành vi phục hồi phức tạp.
    *   **2409.14674 – Kiến trúc RACER (VLM supervisor tạo hướng dẫn ngôn ngữ phong phú cho visuomotor actor) – Suy nghĩ:** Một cách tiếp cận hứa hẹn để robot tự phục hồi lỗi dựa trên hiểu biết ngữ cảnh sâu sắc.
    *   **2409.13598 – Mục tiêu tiền huấn luyện hỗn hợp (masked reconstruction + dự báo độ lệch khí hậu học) cho mô hình thời tiết – Suy nghĩ:** Phù hợp cho mô hình nền tảng đa nhiệm, cân bằng giữa tái tạo và dự báo.
    *   **2409.13598 – Kiến trúc transformer 2D phân cấp (chú ý cục bộ/toàn cục xen kẽ) cho dữ liệu thời tiết – Suy nghĩ:** Xử lý hiệu quả các phụ thuộc không gian ở nhiều tỷ lệ trong dữ liệu lưới.
    *   **2409.10516 – Thuật toán xây dựng chỉ mục ANNS "nhận biết attention" dựa trên quan hệ Q-K từ prefill – Suy nghĩ:** Giải quyết vấn đề OOD một cách thông minh, cải thiện đáng kể hiệu quả truy xuất KV cache.
    *   **2409.16191 – Phương pháp đánh giá HelloEval (học trọng số checklist từ người, áp dụng LLM-as-a-Judge) – Suy nghĩ:** Kết hợp hiệu quả sức mạnh của con người và LLM để đánh giá văn bản dài một cách có khả năng mở rộng và tương quan cao.
    *   **2409.00509 – Biến đổi chỉ số vị trí mô phỏng chuỗi dài từ đoạn ngắn bằng bước nhảy ngẫu nhiên có kiểm soát – Suy nghĩ:** Một kỹ thuật thông minh để giảm chi phí huấn luyện mở rộng ngữ cảnh mà vẫn giữ được thông tin vị trí tương đối.
    *   **2409.18042 – Tách biệt ngữ nghĩa-âm học trong tokenizer/detokenizer giọng nói của EMOVA – Suy nghĩ:** Cho phép kiểm soát phong cách giọng nói độc lập và cải thiện khớp nối với không gian LLM.
    *   **2409.18042 – Chiến lược huấn luyện khớp nối omni-modal lấy văn bản làm trung tâm – Suy nghĩ:** Tăng cường hiệu năng đa phương thức bằng cách huấn luyện đồng thời thay vì riêng lẻ/tuần tự.
    *   **2408.17253 – Chuyển đổi TSF thành tái tạo ảnh patch 2D cho MAE zero-shot (VISION TS) – Suy nghĩ:** Hướng đi bất ngờ nhưng hiệu quả, khai thác năng lực của mô hình thị giác cho chuỗi thời gian.
    *   **2409.15277 – Xây dựng bộ dữ liệu QA y tế thách thức (LancetQA, NEJMQA) từ câu đố chuyên nghiệp – Suy nghĩ:** Nâng cao chất lượng và tính thực tế của benchmark đánh giá LLM y tế. (Đóng góp dữ liệu/đánh giá)
    *   **2409.12183 – Phân tích tách biệt khả năng lập kế hoạch và thực thi của CoT – Suy nghĩ:** Cung cấp hiểu biết sâu hơn về cơ chế hoạt động và hạn chế của CoT. (Đóng góp phân tích)
    *   **2409.12959 – Kỹ thuật "slim screenshot" loại bỏ vùng trống trong ảnh chụp màn hình web – Suy nghĩ:** Một bước tiền xử lý đơn giản nhưng hiệu quả để giảm nhiễu cho LMM khi xử lý nội dung web.
    *   **2408.15545 – Tổng hợp chỉ dẫn SFT khoa học dựa trên LLM với lấy mẫu từ khóa và mô tả tác vụ – Suy nghĩ:** Tạo dữ liệu SFT đa dạng, chất lượng cao cho các miền khoa học ít tài nguyên.
    *   **2409.02095 – Chiến lược suy luận video siêu dài bằng xử lý đoạn chồng chéo và khởi tạo nhiễu dựa trên latent trước – Suy nghĩ:** Giải pháp thực tế để mở rộng khả năng tạo độ sâu cho video vượt giới hạn huấn luyện.
    *   **2409.11355 – Sửa lỗi suy luận DDIM cho phép suy luận đơn bước hiệu quả trong mô hình khuếch tán điều kiện ảnh – Suy nghĩ:** Một phát hiện quan trọng giúp tăng tốc đáng kể các mô hình như Marigold mà không cần thay đổi phức tạp.
    *   **2409.11355 – Tinh chỉnh đầu-cuối (E2E FT) mô hình khuếch tán cho ước tính chiều sâu/pháp tuyến bằng hàm mất mát tác vụ – Suy nghĩ:** Đơn giản hóa và tối ưu hóa trực tiếp cho chất lượng đầu ra cuối cùng, hiệu quả hơn mục tiêu khuếch tán.
    *   **2409.02813 – Định dạng đầu vào chỉ có hình ảnh (vision-only input) cho benchmark MLLM – Suy nghĩ:** Thử thách khả năng tích hợp thị giác-văn bản của MLLM một cách thực tế hơn.
    *   **2409.14713 – Phantom Dimension: tăng tạm thời chiều ẩn trong MHSA bằng cross-attention với token `sos` – Suy nghĩ:** Một cách thông minh để tăng khả năng học của LLVM mà không tăng kích thước vật lý đáng kể.
    *   **2409.14713 – Phantom Optimization (PO): SFT kết hợp DPO-like dựa trên Phantom Triples – Suy nghĩ:** Chiến lược huấn luyện mới để tối ưu LLVM hướng tới câu trả lời đúng và loại bỏ câu trả lời sai/mơ hồ.
    *   **2409.17066 – Khởi tạo codebook bằng K-means có trọng số Hessian cho lượng tử hóa LLM – Suy nghĩ:** Cải thiện chất lượng codebook dựa trên phân tích mục tiêu tối ưu hóa bậc hai.
    *   **2409.03512 – Ngôn ngữ biểu diễn hành động giảng dạy cho tác nhân AI trong MAIC – Suy nghĩ:** Cấu trúc hóa và điều khiển linh hoạt các hoạt động của tác nhân AI trong lớp học.
    *   **2409.11406 – Meta-ControlNet điều chỉnh động cường độ điều kiện 3D dựa trên tương đồng ảnh-tham chiếu – Suy nghĩ:** Xử lý hiệu quả các tham chiếu 3D lệch pha trong tạo sinh 3D.
    *   **2409.11406 – Định tuyến tham chiếu động (thay đổi độ phân giải CCM theo bước thời gian) – Suy nghĩ:** Giảm xung đột chi tiết cục bộ khi tham chiếu 3D lệch pha.
    *   **2409.08692 – Xấp xỉ chiến lược lựa chọn code/test tối ưu Bayes (B4) bằng phân phối Beta liên hợp và tập đồng thuận – Suy nghĩ:** Biến đổi bài toán không thể tính toán thành thuật toán thực tế, hiệu quả.
    *   **2409.03718 – Quy trình tạo ảnh hình học đa biểu đồ (multi-chart geometry images) từ UV map – Suy nghĩ:** Giải pháp thực tế để biểu diễn mesh 3D đa dạng cho mô hình khuếch tán.
    *   **2409.01071 – Thuật toán SceneTilling phân đoạn video theo ngữ nghĩa dựa trên TextTiling – Suy nghĩ:** Một cách hiệu quả để chia video dài thành các đơn vị xử lý có ý nghĩa.
    *   **2409.01071 – Recurrent Memory Bridge Layer với Memory Cache và truy xuất cross-attention cho video dài – Suy nghĩ:** Kiến trúc bộ nhớ phức tạp nhưng tiềm năng để duy trì phụ thuộc dài hạn.
    *   **2409.18964 – Suy luận tham số vật lý (khối lượng, ma sát, đàn hồi) zero-shot từ ảnh đơn bằng VLM – Suy nghĩ:** Một cách tiếp cận mới lạ để khởi tạo mô phỏng vật lý mà không cần thông tin thủ công.
    *   **2409.06210 – Học tương phản có hướng dẫn bởi mối quan hệ tương tác cho định vị affordance – Suy nghĩ:** Cho phép học biểu diễn affordance tinh vi hơn, xét đến ngữ nghĩa tương tác.
    *   **2409.04593 – Truy xuất suy nghĩ (thought retrieval) và tự tiến hóa trong trợ lý học thuật cá nhân hóa – Suy nghĩ:** Tăng cường khả năng của RAG bằng cách tái sử dụng và cập nhật liên tục các kết quả trung gian.
    *   **2409.03420 – High-resolution DocCompressor nén đặc trưng tài liệu bằng cross-attention giữa đặc trưng toàn cục và chi tiết – Suy nghĩ:** Giải pháp nén layout-aware hiệu quả cho OCR-free document understanding.
    *   **2409.14254 – Bộ điều hợp dựa trên luật đơn giản kết hợp qua PoE để tạo khả năng tuân thủ chỉ dẫn LLM – Suy nghĩ:** Một minh chứng thú vị về cách những thay đổi nhỏ trong phân phối đầu ra có thể kích hoạt hành vi phức tạp.
    *   **2409.08857 – Pipeline chỉnh sửa ảnh không tối ưu hóa InstantDrag (FlowGen + FlowDiffusion) – Suy nghĩ:** Tăng tốc đáng kể chỉnh sửa ảnh kéo thả bằng cách tách thành sinh luồng quang học và sinh ảnh theo luồng.
    *   **2409.15700 – Tích hợp ví dụ few-shot vào truy vấn để tạo embedding văn bản (bge-en-icl) – Suy nghĩ:** Khai thác khả năng ICL của LLM để cải thiện mô hình embedding mà không thay đổi kiến trúc.
    *   **2409.15127 – Quy trình thực nghiệm tối ưu RAG và SC-CoT cho LLM y tế nguồn mở – Suy nghĩ:** Cung cấp phương pháp luận để đạt hiệu suất cao với chi phí thấp trong lĩnh vực chuyên biệt. (Đóng góp phương pháp luận thực nghiệm)

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Độ tin cậy và khả năng diễn giải của LLM/MLLM:** Nhiều paper (đặc biệt là các mô hình tạo sinh, code, VLM) vẫn đối mặt với hallucination, lỗi logic, hoặc thiếu giải thích sâu sắc về cơ chế hoạt động (ví dụ: 2409.12186, 2409.15277, 2409.18786). Cơ hội: Phát triển các phương pháp đánh giá và cải thiện tính trung thực, khả năng tự nhận thức lỗi, và diễn giải cơ chế bên trong.
    *   **Hiệu quả tính toán và bộ nhớ cho mô hình lớn/ngữ cảnh dài:** Mặc dù có nhiều nỗ lực (2409.02060 MoE, 2409.02889 Mamba-Transformer, 2409.10516 RetrievalAttention, 2409.17422 GemFilter, 2409.00509 LongRecipe, 2409.10594 KAT, 2409.02097 LinFusion), việc huấn luyện và triển khai các mô hình siêu lớn với ngữ cảnh cực dài vẫn là thách thức. Cơ hội: Các kiến trúc mới hiệu quả hơn, thuật toán nén/lượng tử hóa tốt hơn cho cả trọng số và KV cache, kỹ thuật huấn luyện phân tán tiên tiến.
    *   **Chất lượng và sự đa dạng của dữ liệu huấn luyện:** Thành công của nhiều mô hình (2409.12186, 2409.17146, 2409.01704, 2409.05840, 2409.08239, 2408.15545) phụ thuộc lớn vào dữ liệu. Việc tạo dữ liệu chất lượng cao, đa dạng, ít thiên vị, đặc biệt cho các miền chuyên biệt hoặc ít tài nguyên vẫn còn nhiều khoảng trống. Cơ hội: Các phương pháp tạo dữ liệu tổng hợp/tăng cường thông minh hơn, kỹ thuật data curation tự động, xử lý dữ liệu đa phương thức/đa ngôn ngữ quy mô lớn hiệu quả hơn.
    *   **Khả năng suy luận phức tạp và đa bước:** Nhiều tác vụ (khoa học dữ liệu 2409.07703, QA phức tạp 2409.12941, hiểu truyện tranh 2409.09502, hiểu châm biếm 2409.13592) đòi hỏi suy luận đa bước, tích hợp thông tin từ nhiều nguồn/phương thức. Các mô hình hiện tại vẫn còn hạn chế. Cơ hội: Phát triển các kiến trúc và cơ chế học tập hỗ trợ suy luận có cấu trúc, lập kế hoạch, và tích hợp kiến thức bên ngoài một cách linh hoạt.
    *   **Đánh giá toàn diện và thực tế:** Các benchmark hiện tại có thể chưa phản ánh đầy đủ năng lực của AI trong thế giới thực (2409.07703 DSBench, 2409.06820 PingPong, 2409.07314 MEDIC, 2409.02813 MMMU-Pro, 2409.08264 Windows Agent Arena, 2409.16191 HelloBench, 2409.08267 UrBench). Cơ hội: Xây dựng các benchmark đa dạng hơn, bao gồm tương tác người dùng, môi trường động, và các yếu tố phi kỹ thuật (đạo đức, an toàn).
    *   **Tích hợp đa phương thức sâu hơn:** Mặc dù có nhiều tiến bộ (2409.18869 Emu3, 2409.17692 MIO, 2409.18042 EMOVA), việc tích hợp liền mạch và suy luận chéo giữa nhiều phương thức (đặc biệt là các phương thức ít phổ biến hơn như cảm biến, dữ liệu khoa học) vẫn còn nhiều tiềm năng. Cơ hội: Các kiến trúc hợp nhất thực sự, cơ chế alignment và fusion đa phương thức hiệu quả hơn.
    *   **Cá nhân hóa và thích ứng:** Các mô hình cần có khả năng thích ứng với người dùng và ngữ cảnh cụ thể (2409.11901 PPlug, 2409.04593 Paper Copilot). Cơ hội: Phát triển các phương pháp cá nhân hóa hiệu quả, ít tốn kém, bảo vệ quyền riêng tư và cho phép học hỏi liên tục.
    *   **Khả năng tổng quát hóa và độ bền vững (Robustness):** Mô hình thường hoạt động tốt trên dữ liệu giống huấn luyện nhưng kém hiệu quả khi gặp dữ liệu OOD, nhiễu hoặc các biến đổi nhỏ (2409.14128 Synthetic Image Detection). Cơ hội: Các kỹ thuật regularization mới, phương pháp huấn luyện tăng cường độ bền vững, và hiểu rõ hơn về cơ chế tổng quát hóa.
    *   **Học từ ít dữ liệu (Few-shot/Zero-shot Learning) cho các tác vụ phức tạp:** Nhiều miền chuyên biệt thiếu dữ liệu có nhãn. Cơ hội: Cải thiện khả năng học trong ngữ cảnh (2409.15700 bge-en-icl), các phương pháp meta-learning và transfer learning hiệu quả hơn cho các tác vụ đa phương thức và suy luận.
    *   **Tương tác giữa các loại dữ liệu trong huấn luyện:** Cơ chế tương tác và lợi ích của việc pha trộn các loại dữ liệu khác nhau (ví dụ: code, text, math trong 2409.12186) chưa được hiểu rõ hoàn toàn. Cơ hội: Nghiên cứu sâu hơn về ảnh hưởng của thành phần dữ liệu và các chiến lược pha trộn tối ưu.
    *   **Khả năng mở rộng của các phương pháp mới:** Một số phương pháp mới (ví dụ: PROX 2409.17115, MaskLLM 2409.17481) có thể hiệu quả ở quy mô thử nghiệm nhưng cần đánh giá thêm về chi phí và tính khả thi ở quy mô rất lớn.
    *   **Tích hợp kiến thức chuyên môn và ràng buộc vật lý:** Các mô hình tạo sinh (đặc biệt là 3D/video) cần tuân thủ các quy luật vật lý và kiến thức chuyên ngành để tạo ra kết quả thực tế và đáng tin cậy (ví dụ: 2409.18964 PhysGen). Cơ hội: Phát triển các phương pháp tích hợp ràng buộc vật lý và kiến thức chuyên môn vào quá trình học và sinh.

5.  **FUTURE_IDEAS**

    ✨ **Neuro-Symbolic Multimodal Reasoning Framework**
    · **Motivation:** Các MLLM hiện tại mạnh về nhận thức nhưng yếu về suy luận logic, đa bước phức tạp, đặc biệt khi cần kết hợp thông tin rời rạc và liên tục từ nhiều nguồn.
    · **Key novelty:** Kết hợp sức mạnh của MLLM (để hiểu và biểu diễn đa phương thức) với các module suy luận symbolic (ví dụ: reasoner logic, knowledge graph traversal, physics simulator) trong một framework có thể huấn luyện end-to-end hoặc theo từng phần. LLM đóng vai trò điều phối, trích xuất thông tin và chuyển đổi giữa biểu diễn symbolic và sub-symbolic.
    · **Approach:**
        1.  MLLM trích xuất các thực thể, quan hệ, và trạng thái từ đầu vào đa phương thức.
        2.  Một module "translator" chuyển đổi các thông tin này thành dạng symbolic có thể xử lý bởi reasoner.
        3.  Reasoner thực hiện suy luận (ví dụ: trả lời câu hỏi, dự đoán trạng thái tiếp theo).
        4.  Kết quả symbolic được "translator" ngược lại thành ngôn ngữ tự nhiên hoặc biểu diễn đa phương thức khác bởi MLLM.
        5.  Huấn luyện MLLM và translator bằng RL hoặc SFT để tối ưu hóa việc giao tiếp và sử dụng reasoner.
    · **Dataset + Metrics:** Sử dụng các benchmark đòi hỏi suy luận phức tạp như VCR, COG, hoặc các phiên bản mở rộng của DSBench (2409.07703), UrBench (2409.08267) với các câu hỏi cần suy luận sâu. Metrics: Accuracy, F1, Human Evaluation về tính logic và đúng đắn của chuỗi suy luận.
    · **Risk/Feasibility:** Cao. Thách thức lớn trong việc tích hợp hai paradigma khác biệt, định nghĩa ngôn ngữ symbolic chung, và huấn luyện hiệu quả. (Moon-shot)

    ✨ **Self-Evolving Data Curation Agents for Specialized LLMs**
    · **Motivation:** Chất lượng dữ liệu là tối quan trọng, nhưng việc tạo và quản lý dữ liệu cho các miền chuyên biệt (khoa học, y tế, code) rất tốn kém. Các phương pháp như PROX (2409.17115), Source2Synth (2409.08239), MMEvol (2409.05840) cho thấy tiềm năng của việc tự động hóa.
    · **Key novelty:** Xây dựng một hệ thống agent AI có khả năng tự động khám phá, thu thập, lọc, tăng cường, và thậm chí tự tạo ra các "bài tập" (instructions) mới để liên tục cải thiện một LLM chuyên biệt. Agent này sẽ học các chiến lược data curation hiệu quả theo thời gian.
    · **Approach:**
        1.  Agent bắt đầu với một tập dữ liệu cơ sở và một LLM chuyên biệt.
        2.  Agent có các công cụ: truy cập web, thực thi code, gọi các LLM khác (để sinh, đánh giá), áp dụng các phép biến đổi dữ liệu (như trong PROX).
        3.  Agent đề xuất các lô dữ liệu mới (từ web, biến đổi, hoặc tự sinh).
        4.  Một module đánh giá (có thể là LLM khác hoặc dựa trên hiệu năng của LLM chuyên biệt trên một tập validation giữ lại) sẽ cung cấp phản hồi về chất lượng lô dữ liệu.
        5.  Agent sử dụng RL để học cách chọn lọc và tạo ra dữ liệu tốt hơn, tối ưu hóa hiệu năng của LLM chuyên biệt.
    · **Dataset + Metrics:** Bắt đầu với các bộ dữ liệu chuyên biệt hiện có (ví dụ: PixMo 2409.17146, InfiMM-WebMath 2409.12568). Metrics: Hiệu năng của LLM chuyên biệt trên các benchmark của miền đó, chi phí tạo dữ liệu, độ đa dạng của dữ liệu được tạo.
    · **Risk/Feasibility:** Cao. Đòi hỏi sự phức tạp trong thiết kế agent, hàm thưởng RL, và có nguy cơ agent học các chiến lược không mong muốn. (Moon-shot)

    ✨ **Personalized and Context-Aware Interactive 3D Asset Generation**
    · **Motivation:** Các mô hình tạo 3D hiện tại (2409.12957 PrimX, 2409.03718 GIMDiffusion, 2409.07452 Hi3D) ngày càng mạnh mẽ, nhưng thường thiếu khả năng tương tác sâu và cá nhân hóa theo phong cách người dùng hoặc ngữ cảnh dự án cụ thể.
    · **Key novelty:** Một hệ thống tạo 3D cho phép người dùng tương tác lặp đi lặp lại (ví dụ: "làm cho phần này giống với phong cách của đối tượng X mà tôi đã tạo trước đó", "thay đổi vật liệu này nhưng giữ nguyên hình dạng", "tạo thêm các biến thể dựa trên đối tượng Y nhưng phù hợp với cảnh Z này"). Hệ thống học phong cách người dùng và các ràng buộc của dự án.
    · **Approach:**
        1.  Sử dụng một mô hình nền tảng tạo 3D mạnh (ví dụ, dựa trên PrimX hoặc một kiến trúc hợp nhất).
        2.  Tích hợp một module "user/project profiler" (tương tự PPlug 2409.11901 nhưng cho 3D) để mã hóa sở thích phong cách, các đối tượng đã tạo, và ràng buộc của dự án.
        3.  Cho phép đầu vào đa phương thức (văn bản, ảnh tham chiếu 2D, mô hình 3D tham chiếu như trong Phidias 2409.11406).
        4.  LLM đóng vai trò điều phối, diễn giải yêu cầu người dùng và điều chỉnh các tham số của mô hình tạo 3D.
        5.  Sử dụng các kỹ thuật chỉnh sửa có kiểm soát (ví dụ, dựa trên attention steering hoặc latent space manipulation) để thực hiện các thay đổi chi tiết.
    · **Dataset + Metrics:** Cần các bộ dữ liệu mới ghi lại quá trình tương tác tạo 3D của người dùng. Metrics: Đánh giá của người dùng về mức độ hài lòng, tính nhất quán phong cách, khả năng kiểm soát.
    · **Risk/Feasibility:** Trung bình đến Cao. Thách thức trong việc thu thập dữ liệu tương tác và định nghĩa không gian biểu diễn phong cách/ràng buộc hiệu quả. (Cross-disciplinary)

    ✨ **Unified Framework for Zero-Shot Cross-Modal Transfer Learning in Scientific Discovery**
    · **Motivation:** AI for Science (2409.13598 Prithvi WxC, 2409.08240 VISION TS) cho thấy tiềm năng lớn. Tuy nhiên, việc chuyển giao kiến thức giữa các miền khoa học khác nhau hoặc từ dữ liệu mô phỏng sang dữ liệu thực nghiệm còn hạn chế.
    · **Key novelty:** Một framework tiền huấn luyện MLLM trên lượng lớn dữ liệu khoa học đa phương thức (văn bản, bảng biểu, đồ thị, ảnh thực nghiệm, chuỗi thời gian, phương trình) với mục tiêu học các biểu diễn chung có khả năng chuyển giao zero-shot sang các tác vụ khoa học mới hoặc các miền dữ liệu ít tài nguyên.
    · **Approach:**
        1.  Xây dựng một bộ dữ liệu khoa học đa phương thức quy mô lớn (ví dụ, mở rộng từ InfiMM-WebMath 2409.12568).
        2.  Thiết kế một kiến trúc MLLM hợp nhất (ví dụ, dựa trên Emu3 2409.18869 hoặc MIO 2409.17692) có khả năng xử lý các loại dữ liệu khoa học đa dạng.
        3.  Sử dụng các mục tiêu tiền huấn luyện tự giám sát nhằm khuyến khích alignment giữa các phương thức và học các quy luật vật lý/khoa học cơ bản (ví dụ: dự đoán giá trị bị che trong bảng dữ liệu dựa trên văn bản mô tả và đồ thị liên quan).
        4.  Đánh giá khả năng zero-shot/few-shot trên nhiều tác vụ khoa học hạ nguồn (ví dụ: dự đoán thuộc tính vật liệu, phân tích dữ liệu thí nghiệm, sinh giả thuyết).
    · **Dataset + Metrics:** Cần tổng hợp và chuẩn hóa dữ liệu từ nhiều nguồn khoa học. Metrics: Hiệu năng trên các benchmark khoa học đa dạng, khả năng giải thích các dự đoán.
    · **Risk/Feasibility:** Cao. Thách thức lớn trong việc thu thập và chuẩn hóa dữ liệu khoa học đa phương thức, cũng như thiết kế các mục tiêu tiền huấn luyện phù hợp. (Cross-disciplinary / Moon-shot)

    ✨ **Explainable and Robust Audio-Driven Animation with Fine-Grained Emotional Control**
    · **Motivation:** Các mô hình tạo video chân dung từ âm thanh (2409.02634 Loopy, 2409.18042 EMOVA) đã có những bước tiến, nhưng việc kiểm soát chi tiết cảm xúc, biểu cảm phi ngôn ngữ và tính nhất quán lâu dài vẫn còn khó khăn.
    · **Key novelty:** Một mô hình tạo video chân dung không chỉ đồng bộ môi với âm thanh mà còn cho phép kiểm soát chi tiết các sắc thái cảm xúc (ví dụ: "hơi buồn nhưng vẫn có chút hy vọng") và các biểu cảm phi ngôn ngữ (nhướng mày, gật đầu) một cách có thể diễn giải, đồng thời đảm bảo tính nhất quán của nhân vật và chuyển động trong video dài.
    · **Approach:**
        1.  Sử dụng kiến trúc nền tảng mạnh mẽ (ví dụ, kết hợp Loopy và EMOVA).
        2.  Phân tách biểu diễn âm thanh thành nội dung ngôn ngữ, đặc trưng âm sắc, và đặc trưng cảm xúc/ngữ điệu chi tiết.
        3.  Sử dụng một LLM để diễn giải các yêu cầu cảm xúc phức tạp từ văn bản và ánh xạ chúng thành các vector điều khiển cho mô hình sinh video.
        4.  Tích hợp một module "consistency keeper" dựa trên attention hoặc bộ nhớ dài hạn để duy trì đặc điểm nhân vật và sự mượt mà của chuyển động qua các đoạn video dài (có thể học từ 2409.02095 DepthCrafter).
        5.  Huấn luyện trên dữ liệu video có chú thích cảm xúc chi tiết (có thể cần tạo dữ liệu tổng hợp hoặc bán tự động).
    · **Dataset + Metrics:** Cần bộ dữ liệu video người nói chuyện với chú thích cảm xúc đa chiều. Metrics: Đánh giá của người dùng về tính tự nhiên, mức độ biểu cảm, sự đồng bộ, tính nhất quán. Các chỉ số khách quan về độ chính xác của biểu cảm.
    · **Risk/Feasibility:** Trung bình. Thách thức trong việc thu thập/tạo dữ liệu chú thích cảm xúc chi tiết và định nghĩa không gian biểu diễn cảm xúc hiệu quả. (High feasibility / Cross-disciplinary)

6.  **READING_LIST**

    *   `2409.18869` – Emu3 · Kiến trúc Transformer đơn lẻ đột phá cho tạo sinh và nhận thức đa phương thức SOTA.
    *   `2409.12186` – Qwen2.5-Coder · Cung cấp mô hình code nguồn mở mạnh mẽ và chi tiết về "công thức" tạo ra nó.
    *   `2409.11340` – OmniGen · Hướng tiếp cận tham vọng thống nhất hóa các tác vụ tạo ảnh bằng một mô hình duy nhất.
    *   `2409.00588` – DPPO · Phương pháp mới và hiệu quả để tinh chỉnh Diffusion Policy cho robot bằng RL.
    *   `2409.17692` – MIO · Mô hình any-to-any mã nguồn mở đầu tiên, tích hợp giọng nói và sinh xen kẽ đa phương thức.
    *   `2409.07452` – Hi3D · Đạt độ phân giải cao cho sinh 3D từ ảnh đơn bằng cách khai thác mô hình khuếch tán video.
    *   `2409.12957` – PrimX/3DTopia-XL · Biểu diễn 3D mới và mô hình khuếch tán tiềm ẩn cho tài sản PBR chất lượng.
    *   `2409.10594` – KAT/GR-KAN · Giải pháp hiệu quả để mở rộng quy mô KAN và tích hợp vào Transformer.
    *   `2409.02095` – DepthCrafter · Ước tính độ sâu video dài và nhất quán với kết quả zero-shot ấn tượng.
    *   `2409.17115` – PROX · Hướng tiếp cận mới lạ "lập trình dữ liệu" để tinh chỉnh dữ liệu tiền huấn luyện LLM.
    *   `2409.03752` – Attention Head Analysis Survey · Khảo sát và khung phân loại tốt về vai trò đầu chú ý trong LLM.
    *   `2409.10516` – RetrievalAttention · Giải quyết vấn đề OOD trong ANNS cho KV cache, cải thiện suy luận ngữ cảnh dài.

7.  **META_REFLECTION**

    Tập hợp các bài báo tháng 09/2024 cho thấy một số xu hướng phát triển AI nổi bật:
    *   **Thống nhất và Đa năng hóa (Unification & Generalization):** Nhiều nghiên cứu hướng tới việc xây dựng các mô hình đơn lẻ có khả năng xử lý đa dạng tác vụ và đa phương thức (ví dụ: Emu3, OmniGen, MIO, GOT). Điều này phản ánh tham vọng tạo ra các AI tổng quát hơn, giảm sự cần thiết của các mô hình chuyên biệt cho từng vấn đề nhỏ.
    *   **Hiệu quả Tính toán và Khả năng Mở rộng:** Với sự gia tăng về quy mô mô hình và độ dài ngữ cảnh, các giải pháp tối ưu hóa hiệu quả tính toán, bộ nhớ và khả năng mở rộng trở nên cực kỳ quan trọng. Điều này thể hiện qua các nghiên cứu về MoE (OLMoE), kiến trúc lai (LongLLaVA, KAT), nén/lượng tử hóa (MaskLLM, VPTQ), và các kỹ thuật xử lý ngữ cảnh dài hiệu quả (RetrievalAttention, GemFilter, LongRecipe).
    *   **Tầm quan trọng của Dữ liệu:** Chất lượng, quy mô và chiến lược xử lý/tạo dữ liệu tiếp tục là yếu tố then chốt. Nhiều bài báo tập trung vào việc xây dựng bộ dữ liệu mới (Kvasir-VQA, InfiMM-WebMath, DSBench, PingPong, MMMU-Pro, UrBench, OmniBench), quy trình data curation/engineering tiên tiến (Qwen2.5-Coder, Molmo/PixMo, PROX, Source2Synth, MMEvol), và các phương pháp tổng hợp dữ liệu thông minh.
    *   **Tích hợp Đa phương thức Sâu hơn:** Không chỉ dừng lại ở việc kết hợp các phương thức, các nghiên cứu đang khám phá cách tích hợp sâu hơn, cho phép suy luận chéo và tạo sinh nội dung đa phương thức phức tạp, có kiểm soát (ví dụ: EMOVA với kiểm soát cảm xúc giọng nói, MIO với sinh xen kẽ, SongCreator với tương tác giọng hát-nhạc đệm).
    *   **AI cho Khoa học và các Miền Chuyên biệt:** Ngày càng có nhiều ứng dụng AI hướng tới giải quyết các vấn đề khoa học cụ thể (Prithvi WxC cho thời tiết, VISION TS cho chuỗi thời gian) và các miền chuyên biệt như y tế (o1, Paper Copilot, Kvasir-VQA, MEDIC), lập trình (Qwen2.5-Coder, FUZZCODER), và giáo dục (MAIC).
    *   **Cải thiện Khả năng Suy luận và Độ tin cậy:** Các phương pháp như CoT (dù có giới hạn được chỉ ra trong 2409.12183), tự sửa lỗi (SCoRe), và các benchmark tập trung vào suy luận phức tạp cho thấy nỗ lực không ngừng để AI không chỉ "biết" mà còn "hiểu" và "suy luận" một cách đáng tin cậy.
    *   **Từ Mô hình Khuếch tán Ảnh sang Video và 3D:** Thành công của mô hình khuếch tán trong tạo ảnh đang được mở rộng mạnh mẽ sang tạo video (Loopy, DepthCrafter, Hi3D, LVCD) và 3D (PrimX, GIMDiffusion, Phidias), với các điều chỉnh kiến trúc và chiến lược huấn luyện phù hợp.
    *   **Sự trỗi dậy của các Kiến trúc Thay thế/Bổ sung cho Transformer:** Mặc dù Transformer vẫn thống trị, các kiến trúc như KAN (KAT), Mamba (LongLLaVA, LinFusion), và các cơ chế chú ý tuyến tính (GSA) đang được khám phá như những giải pháp tiềm năng cho các vấn đề về hiệu quả và khả năng mở rộng.
    *   **Tự động hóa và Tác tử AI:** Xu hướng phát triển các tác tử AI có khả năng tự học hỏi, tự cải thiện và thực hiện các quy trình phức tạp (AWM, Paper Copilot, DSBench) ngày càng rõ nét.

    Nhìn chung, lĩnh vực AI đang tiến nhanh theo hướng xây dựng các hệ thống thông minh hơn, đa năng hơn, hiệu quả hơn và có khả năng ứng dụng vào nhiều khía cạnh của đời sống và khoa học, với sự tập trung mạnh mẽ vào chất lượng dữ liệu và các phương pháp đánh giá ngày càng nghiêm ngặt.
