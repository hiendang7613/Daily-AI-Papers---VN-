1.  **TOPIC_TREE**

    *   Generative Models
        *   Image Generation
            *   Text-to-Image Synthesis
                *   Controllable Multi-Text Generation
                    *   `2503.23461` | Tổng hợp nhiều vùng văn bản phức tạp trong ảnh một cách có kiểm soát mà không cần huấn luyện lại mô hình nền.
                *   Diffusion Models & Variants
                    *   `2504.05741` | Tách biệt mã hóa ngữ nghĩa và giải mã chi tiết trong diffusion transformer để cải thiện chất lượng và tốc độ sinh ảnh.
                    *   `2504.11346` | Cải tiến mô hình sinh ảnh khuếch tán với các kỹ thuật xử lý dữ liệu, huấn luyện đa độ phân giải, và tăng tốc suy luận.
                *   Universal Image Generation
                    *   `2504.07960` | Sử dụng học trong ngữ cảnh trực quan và công thức hóa bài toán thành điền khuyết ảnh để đạt được khả năng sinh ảnh phổ quát.
                *   Subject-Driven Image Generation
                    *   `2504.02160` | Tùy chỉnh phổ quát mô hình DiT để sinh ảnh đa chủ thể với tính nhất quán cao thông qua quy trình tổng hợp dữ liệu tự động và UnoPE.
                    *   `2504.17502` | Đánh giá đồng thời tính phù hợp văn bản và bảo toàn chủ thể trong sinh ảnh có điều kiện bằng VLM đa đầu vào được huấn luyện trên dữ liệu tổng hợp quy mô lớn.
            *   Face Swapping
                *   `2504.14509` | Huấn luyện face swapping với giám sát tường minh bằng Triplet ID Group và kiến trúc diffusion cải tiến để đạt hiệu quả và kiểm soát thuộc tính.
            *   Vector Graphics Generation
                *   Multimodal SVG Synthesis
                    *   `2504.06263` | Tạo SVG đa phương thức, phức tạp end-to-end bằng VLM tiền huấn luyện và tham số hóa SVG mới.
            *   Autoregressive Image Generation
                *   Visual Tokenization
                    *   `2504.08736` | Mở rộng tokenizer hình ảnh quy mô lớn bằng semantic regularization và kiến trúc lai để cân bằng chất lượng tái tạo và sinh ảnh.
                *   Unified Representation Learning and Generation
                    *   `2504.00999` | Hợp nhất gộp token vào mô hình sinh tự hồi quy dựa trên VQ để dung hòa học biểu diễn và sinh ảnh, với cơ chế phục hồi nguồn.
        *   Video Generation
            *   Controllable Video Generation
                *   `2503.24379` | Diễn giải điều kiện đa dạng (văn bản, hình ảnh, tín hiệu chuyên biệt) thành phụ đề chi tiết, có cấu trúc để hướng dẫn tạo video có kiểm soát.
                *   `2504.08388` | Phát triển mô hình thế giới tương tác thời gian thực cho Minecraft với giải mã song song và kiến trúc đầu vào xen kẽ hình ảnh-hành động.
                *   Multi-Element Composition Video Generation
                    *   `2504.02436` | Tổng hợp nhiều yếu tố hình ảnh tham chiếu (nhân vật, đối tượng, nền) vào video theo mô tả văn bản, duy trì tính nhất quán và độ trung thực.
            *   Long Video Generation
                *   Transformer-based Video Generation
                    *   `2504.05298` | Tích hợp lớp Test-Time Training (TTT) vào Diffusion Transformer để sinh video dài một phút từ kịch bản phân cảnh, với tối ưu hóa triển khai TTT-MLP.
                *   Context Management and Sampling Strategies
                    *   `2504.12626` | Nén khung hình đầu vào theo cấp số nhân (FramePack) và lấy mẫu chống trôi dạt hai chiều để tạo video dài nhất quán.
            *   Speech-Driven Character Animation
                *   Talking Character Synthesis
                    *   `2503.23307` | Sinh video nhân vật nói toàn thân, đa khung cảnh chỉ từ văn bản và âm thanh bằng Diffusion Transformer với cơ chế chú ý cửa sổ và huấn luyện theo curriculum.
            *   Audio-Driven Talking Head Generation
                *   `2504.04842` | Tạo video chân dung nói chuyện thực tế với căn chỉnh âm thanh-hình ảnh hai giai đoạn, bảo toàn nhận dạng tập trung vào khuôn mặt và điều biến cường độ chuyển động.
            *   Human Image Animation
                *   `2504.01724` | Diễn hoạt người từ ảnh với tín hiệu điều khiển lai (khuôn mặt, đầu, cơ thể), dẫn dắt ngoại hình bổ sung và chiến lược huấn luyện tiến bộ trên DiT.
        *   Generative Games
            *   Infinite Game Generation
                *   `2504.01014` | Dự đoán trạng thái game đa phương thức (video, trạng thái nhân vật) cho mô phỏng cuộc sống anime vô hạn bằng MLLM và biểu diễn nhận biết hành động.
    *   Multimodal Learning
        *   Vision-Language Models (VLMs)
            *   Pre-training Strategies
                *   `2504.10479` | Đề xuất tiền huấn luyện đa phương thức tự nhiên, hợp nhất tiền huấn luyện ngôn ngữ và căn chỉnh đa phương thức vào một giai đoạn.
            *   Efficient Architectures and Advanced Reasoning
                *   `2504.07491` | Phát triển Kimi-VL, một VLM MoE với bộ mã hóa thị giác độ phân giải gốc (MoonViT), ngữ cảnh dài (128K) và khả năng suy luận chuỗi tư duy dài (Long-CoT SFT).
            *   Multimodal Reasoning and Alignment
                *   `2504.05599` | Truyền tải đa phương thức hiệu quả cho LLM suy luận bằng MLP nhẹ, tối ưu hóa kết hợp (SFT lặp lại + GRPO) và chưng cất CoT độ dài thích ứng.
            *   Long-Context Understanding
                *   `2504.15271` | Cải thiện hiểu đa phương thức ngữ cảnh dài bằng chiến lược xếp gạch bảo toàn diện tích (IAP), lấy mẫu suy giảm tự động (ADS) và huấn luyện sau hỗn hợp lũy tiến.
            *   Multimodal Embedding
                *   `2504.17432` | Học biểu diễn nhúng phổ quát cho MLLM qua chưng cất kiến thức phân biệt văn bản và tinh chỉnh hướng dẫn tăng cường bằng mẫu phủ định khó.
            *   Deep Vision-Language Integration Architectures
                *   `2504.09925` | Tích hợp sâu đặc trưng thị giác-ngôn ngữ từ mã hóa đến giải mã với các cơ chế hướng dẫn bằng văn bản, căn chỉnh đệ quy và hàm mất mát giám sát kép.
            *   Text-Only Data Generation for VLMs
                *   `2503.22655` | Tổng hợp dữ liệu đa phương thức (biểu diễn ảnh và chỉ dẫn) chỉ từ văn bản, loại bỏ sự phụ thuộc vào ảnh thật bằng cách làm giàu chú thích và chuyển đổi biểu diễn dựa trên "modality gap".
        *   Video Understanding
            *   Benchmarking for Video-based Reasoning and Generalization
                *   `2503.24376` | Xây dựng SEED-Bench-R1, benchmark hiểu video với dữ liệu huấn luyện quy mô lớn tạo tự động và bộ đánh giá phân cấp để kiểm tra tổng quát hóa của MLLM.
        *   Video Large Language Models (Video LLMs)
            *   Streaming Video Understanding
                *   `2504.16030` | Huấn luyện streaming cho Video LLM bằng cách xen kẽ dày đặc từ ASR và khung hình video, cùng quy trình tạo dữ liệu quy mô lớn và benchmark bình luận thể thao.
        *   Reasoning-Informed Visual Editing
            *   Benchmarking and Evaluation
                *   `2504.02826` | Đề xuất RISEBench, benchmark chuyên dụng để đánh giá năng lực chỉnh sửa hình ảnh dựa trên suy luận (thời gian, nhân quả, không gian, logic) và quy trình đánh giá LMM-as-a-Judge.
        *   Detailed Localized Captioning
            *   `2504.16072` | Tạo chú thích cục bộ chi tiết cho ảnh/video bằng "focal prompt", "localized vision backbone", quy trình dữ liệu bán giám sát và benchmark đánh giá không cần tham chiếu.
        *   Multimodal Instruction Following
            *   `2504.07957` | Tạo dữ liệu ảnh-chỉ dẫn đa phương thức chất lượng cao (MM-IFEngine) và benchmark (MM-IFEval) để cải thiện và đánh giá khả năng tuân thủ chỉ dẫn đa ràng buộc của MLLM.
    *   Large Language Models (LLMs)
        *   Reasoning in Language Models
            *   Reinforcement Learning for Reasoning
                *   `2503.24290` | Huấn luyện RL quy mô lớn (Open-Reasoner-Zero) cho suy luận trực tiếp trên LLM nền tảng không cần SFT, với PPO tối giản và hàm thưởng nhị phân.
                *   `2504.11536` | Tích hợp thực thi trình thông dịch mã vào vòng lặp suy luận của LLM (ReTool) thông qua RL để học cách và thời điểm sử dụng công cụ.
            *   Self-Verification and Tool Use
                *   `2504.04718` | Tự xác minh ở sLM (T1) bằng cách tích hợp công cụ (ToolV) và mô hình phần thưởng (RM-based verifier), với chưng cất kiến thức multi-LoRA.
            *   Efficient Fine-tuning with Reinforcement Learning
                *   `2504.20571` | Chứng minh hiệu quả của 1-shot RLVR, phân tích "khái quát hóa sau bão hòa" và vai trò của policy gradient/entropy loss trong việc cải thiện suy luận LLM.
            *   Cost-Efficient Reasoning Enhancement in Small Language Models
                *   `2504.15777` | Áp dụng LoRA trong RL (GRPO) để tinh chỉnh hiệu quả mô hình ngôn ngữ nhỏ cho suy luận phức tạp, đạt hiệu suất cạnh tranh với chi phí thấp.
            *   Unsupervised Self-Training for Reasoning
                *   `2504.08672` | Tự huấn luyện không giám sát (Genius) để tăng cường suy luận LLM từ truy vấn chung, sử dụng lấy mẫu lại dự đoán từng bước và tối ưu hóa hiệu chỉnh lợi thế (ACO).
        *   Test-Time Reinforcement Learning
            *   `2504.16084` | Cho phép LLM tự cải thiện trên dữ liệu không nhãn tại thời điểm kiểm tra (TTRL) bằng cách ước tính nhãn giả qua bỏ phiếu đa số và dùng làm tín hiệu thưởng cho RL.
        *   Multilingual Capabilities
            *   Efficient Language Injection
                *   `2504.15120` | Tích hợp ngôn ngữ mới (Ả Rập) vào LLM đơn ngữ (Anh) bằng cách mở rộng kiến trúc (chèn lớp) và từ vựng, chỉ huấn luyện phần mới.
        *   Model Quantization
            *   1-bit Models
                *   `2504.12285` | Triển khai, huấn luyện và đánh giá BitNet b1.58 2B4T, LLM 1.58-bit gốc 2 tỷ tham số đầu tiên, công khai mã nguồn và trọng số.
        *   Efficient Inference
            *   Distributed Inference on Resource-Constrained Devices
                *   `2504.08791` | Kiến trúc song song piped-ring với prefetching (prima.cpp) và thuật toán Halda để gán layer cho các thiết bị gia đình không đồng nhất, tài nguyên thấp.
            *   Parallel Inference
                *   `2504.06261` | Suy luận LLM song song cộng tác (Hogwild! Inference) với KV cache chia sẻ, khai thác RoPE và chiến lược prompting.
            *   Adaptive Parallel Reasoning
                *   `2504.15466` | Cho phép LLM tự động điều phối tính toán tuần tự và song song (APR) thông qua toán tử `spawn()`/`join()` và học tăng cường.
        *   Inference Strategies
            *   Test-Time Scaling
                *   `2503.24235` | Khảo sát và đề xuất khung phân loại đa chiều cho các phương pháp Test-Time Scaling (TTS) trong LLM.
        *   Pre-training Data Optimization
            *   `2504.13161` | Tự động hóa khám phá, đánh giá và tinh chỉnh hỗn hợp dữ liệu tiền huấn luyện LLM (CLIMB) bằng nhúng, phân cụm và tìm kiếm lặp lại.
        *   Alignment
            *   Reward Modeling
                *   `2504.00050` | Huấn luyện LLM làm giám khảo (JudgeLRM) bằng RL (GRPO) với hàm thưởng chuyên biệt cho tác vụ đánh giá, kết hợp yếu tố cấu trúc và nội dung.
                *   `2504.02495` | Tự động tạo nguyên tắc và phê bình thích ứng cho mô hình phần thưởng tạo sinh (SPCT) và sử dụng Meta RM để hướng dẫn bỏ phiếu, tăng khả năng mở rộng thời gian suy luận.
            *   Reinforcement Learning from AI Feedback (RLAIF)
                *   `2504.20157` | Tự động tinh chỉnh prompt của reward model trong RLAIF (MPO) bằng Meta Reward Model, giúp thích ứng động với bối cảnh huấn luyện.
        *   Tool-Augmented Reasoning
            *   Reinforcement Learning for Tool Invocation
                *   `2504.13958` | Thiết kế phần thưởng chi tiết (định dạng, tên công cụ, tên tham số, giá trị tham số) cho học tăng cường LLM sử dụng công cụ đa năng (ToolRL).
        *   Mixture-of-Experts
            *   Test-Time Adaptation and Optimization
                *   `2504.07964` | Tối ưu hóa tổ hợp chuyên gia tại thời điểm kiểm thử (C3PO) bằng cách điều chỉnh trọng số chuyên gia cốt lõi ở lớp quan trọng dựa trên mẫu tham chiếu thành công.
        *   Attention Mechanisms
            *   `2504.00927` | Kiến trúc Multi-Token Attention (MTA) tính toán trọng số chú ý dựa trên nhiều vector truy vấn/khóa đồng thời thông qua key-query và head mixing convolution.
    *   Reinforcement Learning
        *   Policy Optimization
            *   Off-Policy Learning for Large Language Models
                *   `2504.14945` | Tích hợp dấu vết suy luận off-policy vào zero-RL (LUFFY) với Mixed-Policy GRPO và policy shaping để cân bằng bắt chước và khám phá.
            *   Policy Optimization Algorithm Adjustments for Bias Reduction
                *   `2503.20783` | Loại bỏ các thành phần gây thiên kiến về độ dài phản hồi và độ khó câu hỏi trong GRPO (Dr. GRPO) để tối ưu hóa không thiên kiến cho LLM.
        *   Character Control
            *   Multi-skill Human-Scene Interaction (HSI)
                *   `2503.19901` | Hợp nhất nhiều kỹ năng tương tác người-cảnh vật lý (TokenHSI) bằng kiến trúc Transformer với tokenizer riêng biệt và thích ứng chính sách hiệu quả.
    *   Information Retrieval
        *   Multimodal and Multi-granularity Retrieval
            *   `2504.20734` | Truy xuất và tích hợp kiến thức từ nhiều nguồn không đồng nhất (UniversalRAG) với định tuyến nhận biết phương thức và truy xuất nhận biết độ chi tiết.
        *   Reasoning-intensive Retrieval
            *   `2504.20595` | Tổng hợp dữ liệu huấn luyện chuyên biệt (REASON IR-SYNTHESIZER) và phương pháp tái xếp hạng (ReasonIR-Rerank) cho truy xuất thông tin đòi hỏi suy luận.
    *   Evaluation and Benchmarking
        *   LLM Leaderboards and Metascience
            *   `2504.20879` | Phân tích các vấn đề và thiên vị tiềm ẩn trong nền tảng đánh giá Chatbot Arena, hệ thống hóa các yếu tố gây sai lệch xếp hạng LLM.
        *   Multilingual Evaluation
            *   `2504.15521` | Phân tích định lượng quy mô lớn về sự tương quan giữa hiệu suất benchmark đa ngôn ngữ và đánh giá của con người, nhấn mạnh tầm quan trọng của benchmark bản địa hóa.
        *   AI Agent Capabilities
            *   Scientific Research Replication
                *   `2504.01848` | Đề xuất PaperBench, benchmark đánh giá khả năng tái tạo nghiên cứu AI của tác nhân, cùng quy trình chấm điểm tự động SimpleJudge và JudgeEval.
        *   Automated Answer Verification for Reasoning Models
            *   `2504.10481` | Xây dựng bộ dữ liệu VAR và mô hình xVerify để thẩm định câu trả lời cho bài toán suy luận, với hàm đánh giá tương đương đa thành phần.
        *   Physics Reasoning Evaluation
            *   `2504.16074` | Đề xuất PHYBench, benchmark suy luận vật lý phức tạp được giám tuyển bởi con người, và EED Score, thước đo tương đồng biểu thức toán học.
        *   Robustness and Critical Thinking Analysis
            *   `2504.06514` | Xây dựng bộ dữ liệu câu hỏi thiếu tiền đề (MiP) để nghiên cứu hiện tượng "suy nghĩ thừa khi thiếu tiền đề" (MiP-Overthinking) ở LLM.
        *   Other
            *   `2504.15279` | Giới thiệu VisuLogic, benchmark đánh giá suy luận trực quan thuần túy của MLLM, tránh lối tắt dựa trên ngôn ngữ. (Lưu ý: Paper này chủ yếu là benchmark, không phải đóng góp kỹ thuật mới về mô hình/thuật toán)
    *   AI Agents
        *   Foundation Agents and Brain-Inspired Architectures
            *   `2504.01990` | Tổng quan và hệ thống hóa kiến thức về tác tử nền tảng, đề xuất kiến trúc module hóa lấy cảm hứng từ não bộ. (Lưu ý: Survey)
        *   Multi-Agent Systems
            *   Automated Design of Multi-Agent Systems
                *   `2504.15257` | Tự động hóa thiết kế hệ thống đa agent cấp độ truy vấn (FlowReasoner) bằng meta-agent học qua RL với phản hồi thực thi.
        *   Code Generation from Scientific Papers
            *   `2504.17192` | Tự động hóa việc tạo kho mã nguồn hoàn chỉnh từ bài báo khoa học (PaperCoder) bằng framework LLM đa tác tử với các giai đoạn Lập kế hoạch, Phân tích, Viết mã.
    *   Optimization Techniques
        *   Adaptive Gradient Clipping
            *   `2504.02507` | Thuật toán cắt tỉa gradient thích ứng (ZClip) tự động điều chỉnh ngưỡng cắt tỉa dựa trên phát hiện bất thường bằng z-score của norm gradient.
    *   Model Analysis and Interpretation
        *   Architecture Probing from Outputs
            *   `2504.02782` | Phân tích dựa trên mô hình phân loại để suy luận về kiến trúc bộ giải mã hình ảnh của GPT-4o (khuếch tán hay tự hồi quy).
        *   Tracing Model Outputs to Training Data
            *   `2504.07096` | Truy vết đầu ra của mô hình ngôn ngữ về toàn bộ dữ liệu huấn luyện đa nghìn tỷ token trong thời gian thực (OLM OTRACE) bằng infini-gram và thuật toán song song.
    *   Sampling Techniques
        *   Antidistillation Sampling
            *   `2504.13146` | Lấy mẫu chống chưng cất để tạo chuỗi token làm suy giảm hiệu quả chưng cất mô hình học sinh, bằng cách điều chỉnh phân phối xác suất của mô hình thầy giáo.
    *   Instruction Tuning Data Selection
        *   `2504.13835` | Lấy mẫu dữ liệu instruction-tuning (MIG) dựa trên thước đo thông tin thống nhất chất lượng và đa dạng, mô hình hóa trên đồ thị nhãn.
    *   AI Safety
        *   Red Teaming
            *   Automated Multiturn Jailbreaking
                *   `2504.13203` | Framework đa tác nhân cộng tác (X-Teaming) để tự động hóa khám phá và tối ưu hóa tấn công bẻ khóa đa lượt vào LLM.
    *   Other
        *   `2504.13837` | Phân tích phê bình về giới hạn của RLVR trong việc tạo ra khả năng suy luận mới vượt trội mô hình cơ sở. (Lưu ý: Analysis)
        *   `2504.05979` | Nghiên cứu thực nghiệm toàn diện về khả năng tạo ảnh của GPT-4o, so sánh với các mô hình SOTA và phân tích điểm mạnh/hạn chế. (Lưu ý: Empirical Study)
        *   `2504.08003` | Đánh giá có hệ thống khả năng tích hợp hiểu biết ngữ nghĩa và sinh ảnh của GPT-4o, chỉ ra hạn chế trong kiến thức động và suy luận có điều kiện. (Lưu ý: Empirical Study)
        *   `2504.05535` | Xây dựng tập dữ liệu sở thích tiếng Trung quy mô lớn (COIG-P) bằng quy trình chú thích dựa trên LLM, huấn luyện Mô hình Thưởng tiếng Trung (CRM) và Tập dữ liệu Chuẩn Thưởng (CRBench).
        *   `2504.00883` | Điều tra hiệu quả của CoT và áp dụng GRPO (vsGRPO) để cải thiện suy luận không gian-thị giác cho Qwen2-VL, nhấn mạnh vai trò của KL penalty.
        *   `2504.08837` | Cải thiện GRPO bằng Selective Sample Replay (SSR) để chống suy giảm lợi thế và Forced Rethinking để khuyến khích tự suy ngẫm trong VLM.
        *   `2504.09643` | Tự huấn luyện lặp đi lặp lại mô hình xếp hạng lại mã nguồn (RewardRanker) bằng PPO và hard negatives để cải thiện cả reranker và generator.
        *   `2503.23077` | Khảo sát và phân loại các phương pháp suy luận hiệu quả cho Mô hình Suy luận Lớn (LRM) thành CoT nhỏ gọn tường minh và CoT tiềm ẩn ngầm. (Lưu ý: Survey)
        *   `2504.04022` | Đề xuất thuật toán tạo bộ dữ liệu đối kháng để đo lường khả năng suy ngẫm tình huống và tự suy ngẫm ở mô hình ngôn ngữ.
        *   `2504.15376` | Phát triển hệ thống phân loại chuyển động camera toàn diện và quy trình gán nhãn "label-then-caption" cho chú thích chuyển động camera.
        *   `2503.23377` | Kiến trúc JavisDiT (Diffusion Transformer hợp nhất) và bộ ước tính tiền nghiệm HiST-Sypo cho sinh audio-video đồng bộ chi tiết.
        *   `2504.02542` | Đề xuất ACTalker, mô hình khuếch tán video cho đầu nói với lớp Mamba điều khiển song song (PCM) và Mask-SSM để xử lý đa tín hiệu điều khiển.
        *   `2504.00595` | Cải thiện hiệu quả tiền huấn luyện MLLM bằng chiến lược độ phân giải động, đóng gói chuỗi đa phương thức và quy trình lọc dữ liệu kết hợp.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                  |
    | :--- | :-------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2504.10479| Native Multimodal Pre-training, Unified Learning, InternVL3 | Hợp nhất tiền huấn luyện ngôn ngữ và căn chỉnh đa phương thức vào một giai đoạn duy nhất, tối ưu hóa đồng thời toàn bộ tham số.          | Đơn giản hóa quy trình huấn luyện MLLM, cải thiện sự căn chỉnh giữa các phương thức, tiềm năng cho các MLLM mạnh mẽ và hiệu quả hơn.             |
    | 2    | 2504.07491| Kimi-VL, MoE VLM, Long Context (128K), Long-CoT SFT   | Phát triển VLM MoE với ngữ cảnh cực dài và khả năng suy luận chuỗi tư duy dài, cùng bộ mã hóa thị giác độ phân giải gốc (MoonViT).        | Đẩy mạnh giới hạn về độ dài ngữ cảnh và khả năng suy luận phức tạp trong VLM, mở ra ứng dụng cho tài liệu/video dài.                         |
    | 3    | 2503.24290| Open-Reasoner-Zero, Minimalist RL, Large-scale Reasoning | Huấn luyện RL quy mô lớn cho suy luận trực tiếp trên LLM nền tảng không cần SFT, với PPO tối giản và hàm thưởng nhị phân.                 | Giảm đáng kể độ phức tạp và chi phí huấn luyện LLM cho các tác vụ suy luận, dân chủ hóa việc phát triển LRM mạnh mẽ.                         |
    | 4    | 2504.07096| OLM OTRACE, Real-time Tracing, Training Data Attribution | Truy vết đầu ra LLM về toàn bộ dữ liệu huấn luyện đa nghìn tỷ token trong thời gian thực bằng infini-gram và thuật toán song song.        | Tăng cường tính minh bạch, khả năng diễn giải và kiểm chứng cho LLM, hỗ trợ gỡ lỗi và phát hiện các vấn đề bản quyền/thiên kiến.              |
    | 5    | 2504.08736| GigaTok, Scalable Visual Tokenizer, Semantic Regularization | Mở rộng visual tokenizer lên quy mô tỷ tham số bằng semantic regularization, kiến trúc lai và chiến lược mở rộng bất đối xứng.          | Giải quyết vấn đề đánh đổi chất lượng tái tạo/sinh ảnh khi mở rộng tokenizer, cải thiện hiệu suất mô hình sinh ảnh tự hồi quy.                 |
    | 6    | 2504.12285| BitNet b1.58, 1-bit LLM, Efficient LLM, Open Source   | Triển khai và công khai LLM 1.58-bit (2B tham số) đầu tiên được huấn luyện từ đầu trên 4 nghìn tỷ token, đạt hiệu năng cạnh tranh.        | Thúc đẩy nghiên cứu và ứng dụng LLM cực kỳ hiệu quả về tài nguyên (bộ nhớ, năng lượng, độ trễ), tiềm năng cho triển khai trên thiết bị biên. |
    | 7    | 2503.22655| Unicorn, Text-only VLM Data Synthesis, Modality Gap  | Tổng hợp dữ liệu huấn luyện VLM (biểu diễn ảnh và chỉ dẫn) quy mô lớn hoàn toàn từ văn bản, dựa trên việc làm giàu chú thích và "modality gap". | Giải quyết vấn đề khan hiếm dữ liệu ảnh-văn bản chất lượng cao, giảm chi phí và thời gian tạo dữ liệu VLM, mở ra hướng mới cho huấn luyện VLM. |
    | 8    | 2504.02160| UNO, Universal Subject Customization, Data-Model Co-evolution | Tùy chỉnh phổ quát mô hình DiT để sinh ảnh đa chủ thể với quy trình tổng hợp dữ liệu tự động và UnoPE.                               | Cho phép tạo ảnh tùy chỉnh đa chủ thể chất lượng cao, độ phân giải cao mà không cần tinh chỉnh lại cho từng chủ thể mới.                      |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2504.10479 – Native multimodal pre-training – Suy nghĩ:** Hợp nhất tiền huấn luyện ngôn ngữ và căn chỉnh đa phương thức vào một giai đoạn duy nhất, tối ưu hóa đồng thời toàn bộ tham số. Đây là một bước tiến quan trọng giúp đơn giản hóa pipeline huấn luyện MLLM và có khả năng cải thiện sự gắn kết giữa các phương thức.
    *   **2504.06263 – Tham số hóa SVG mới (lệnh 'Fill', nén tọa độ) – Suy nghĩ:** Chuyển đổi lệnh và tọa độ SVG thành token rời rạc, đặc biệt là nén cặp tọa độ thành một token và lệnh 'Fill' cho màu sắc, giúp VLM xử lý SVG phức tạp hiệu quả hơn. Rất sáng tạo và giải quyết được vấn đề độ dài chuỗi.
    *   **2504.15376 – Quy trình gán nhãn "label-then-caption" – Suy nghĩ:** Kết hợp phân loại có cấu trúc với mô tả tự do cho các trường hợp phức tạp/không chắc chắn trong chú thích chuyển động camera. Linh hoạt và giúp thu thập dữ liệu chất lượng cao hơn.
    *   **2503.23307 – Speech-Video Window Attention & Joint ST2V + T2V Training – Suy nghĩ:** Giới hạn truy vấn audio theo cửa sổ cục bộ để đồng bộ khẩu hình và kết hợp huấn luyện trên dữ liệu speech-text và text-only để tăng tính khái quát. Giải pháp trực tiếp cho các vấn đề cốt lõi trong sinh video nhân vật nói.
    *   **2504.08791 – Piped-ring parallelism with prefetching & Thuật toán Halda – Suy nghĩ:** Kiến trúc song song mới cho LLM trên thiết bị tài nguyên thấp, kết hợp với thuật toán tối ưu hóa gán layer (LDA) xem xét tính không đồng nhất của thiết bị. Rất thực tế và giải quyết vấn đề cấp thiết.
    *   **2504.07491 – MoonViT (đóng gói NaViT + 2D RoPE) & Long-CoT SFT – Suy nghĩ:** Bộ mã hóa thị giác độ phân giải gốc linh hoạt và chiến lược SFT chuyên biệt để kích hoạt suy luận chuỗi tư duy dài trong VLM. Đẩy mạnh khả năng xử lý ảnh đa dạng và suy luận sâu.
    *   **2504.15120 – Chèn lớp và mở rộng từ vựng có chọn lọc huấn luyện – Suy nghĩ:** Tích hợp ngôn ngữ mới vào LLM bằng cách chỉ huấn luyện các lớp mới chèn và từ vựng mở rộng. Hiệu quả về chi phí và giảm thiểu catastrophic forgetting.
    *   **2504.06261 – Hogwild! Inference (KV cache chia sẻ với RoPE rotation) – Suy nghĩ:** Cho phép nhiều worker LLM cộng tác suy luận song song bằng cách chia sẻ và cập nhật đồng thời KV cache, tận dụng RoPE để điều chỉnh embedding vị trí. Hướng đi mới cho suy luận cộng tác linh hoạt.
    *   **2504.17192 – PaperCoder (Lập kế hoạch đa khía cạnh với UML, phân tích phụ thuộc) – Suy nghĩ:** Framework đa tác tử với các giai đoạn lập kế hoạch chi tiết (bao gồm tạo biểu đồ UML, danh sách tệp theo thứ tự) để tự động tạo kho mã từ bài báo khoa học. Rất có cấu trúc và tham vọng.
    *   **2504.16084 – Test-Time Reinforcement Learning (TTRL) với majority voting – Suy nghĩ:** LLM tự cải thiện trên dữ liệu không nhãn tại thời điểm kiểm tra bằng cách tạo tín hiệu thưởng từ bỏ phiếu đa số trên các mẫu do chính nó sinh ra. Đơn giản nhưng tiềm năng lớn cho học liên tục.
    *   **2504.05298 – Tích hợp TTT-MLP vào Diffusion Transformer với On-Chip Tensor Parallel – Suy nghĩ:** Sử dụng TTT-MLP làm trạng thái ẩn trong DiT để nắm bắt ngữ cảnh dài cho sinh video, và tối ưu hóa triển khai bằng On-Chip Tensor Parallel. Sáng tạo trong việc kết hợp RNN-like TTT với Transformer và tối ưu hóa phần cứng.
    *   **2503.23461 – Instance Fusion, Region Insulation, Text Focus (trong TextCrafter) – Suy nghĩ:** Các kỹ thuật không cần huấn luyện để cải thiện tạo nhiều văn bản trong ảnh: hợp nhất embedding văn bản-vật mang, cách ly vùng khử nhiễu ban đầu, và tăng cường chú ý cho token văn bản. Giải pháp module hóa thông minh.
    *   **2504.20571 – 1-shot RLVR và phân tích "khái quát hóa sau bão hòa" – Suy nghĩ:** Chứng minh RLVR có thể hiệu quả chỉ với một ví dụ và phát hiện hiện tượng mô hình tiếp tục khái quát hóa ngay cả khi đã bão hòa trên ví dụ huấn luyện. Phát hiện quan trọng về hiệu quả dữ liệu trong RL.
    *   **2504.00999 – MergeVQ (gộp token + VQ) và Source Recovery – Suy nghĩ:** Tích hợp gộp token vào mô hình VQ, lưu trữ quan hệ token gốc-gộp trong ma trận nguồn và phục hồi chi tiết từ đó. Giải quyết tốt sự đánh đổi giữa biểu diễn và sinh ảnh.
    *   **2504.13161 – CLIMB (Phân cụm + tìm kiếm lặp lại với predictor) – Suy nghĩ:** Tự động tối ưu hóa hỗn hợp dữ liệu tiền huấn luyện bằng cách phân cụm ngữ nghĩa và sử dụng quy trình tìm kiếm lặp đi lặp lại với mô hình proxy và bộ dự đoán hiệu suất. Hệ thống và tiềm năng.
    *   **2504.17761 – Step1X-Edit (MLLM + DiT decoder) và quy trình tạo dữ liệu đa tác vụ – Suy nghĩ:** Mô hình chỉnh sửa ảnh hợp nhất và quy trình tạo dữ liệu quy mô lớn cho 11 loại tác vụ chỉnh sửa. Hướng tới mô hình chỉnh sửa phổ quát mạnh mẽ.
    *   **2504.10481 – xVerify (LLM fine-tuned) và hàm đánh giá tương đương đa thành phần (ψ) – Suy nghĩ:** Mô hình thẩm định câu trả lời suy luận, được huấn luyện trên bộ dữ liệu VAR, với khả năng trích xuất và kiểm tra tương đương toán học, ngôn ngữ tự nhiên, ký hiệu. Rất cần thiết cho đánh giá mô hình suy luận.
    *   **2504.05599 – Truyền tải đa phương thức hiệu quả (MLP adapter với LLM thay thế) & AL-CoTD – Suy nghĩ:** Chiến lược huấn luyện MLP adapter với LLM không suy luận để căn chỉnh thị giác-ngôn ngữ, sau đó chuyển sang LLM suy luận; và cơ chế tự động điều chỉnh độ dài CoT. Thông minh và hiệu quả về dữ liệu.
    *   **2504.14945 – LUFFY (Mixed-Policy GRPO với policy shaping) – Suy nghĩ:** Tích hợp dấu vết suy luận off-policy vào zero-RL bằng GRPO cải tiến, sử dụng regularized importance sampling để nhấn mạnh học từ token off-policy xác suất thấp. Giải quyết tốt vấn đề khám phá trong RL on-policy.
    *   **2504.02507 – ZClip (cắt tỉa gradient thích ứng dựa trên z-score) – Suy nghĩ:** Tự động điều chỉnh ngưỡng cắt tỉa gradient dựa trên phát hiện bất thường bằng z-score của norm gradient theo thời gian. Đơn giản, hiệu quả và thích ứng tốt.
    *   **2504.04022 – Thuật toán tạo bộ dữ liệu đối kháng cho suy ngẫm – Suy nghĩ:** Tạo dữ liệu có chủ đích chứa lỗi trong CoT để đánh giá khả năng suy ngẫm tình huống và tự suy ngẫm của LLM. Phương pháp luận tốt để nghiên cứu năng lực meta-cognitive.
    *   **2503.24379 – Any2Caption (MLLM diễn giải điều kiện thành phụ đề cấu trúc) – Suy nghĩ:** Tách rời diễn giải điều kiện phức tạp (đa phương thức) khỏi tổng hợp video, MLLM tạo phụ đề chi tiết cho mô hình tạo video nền tảng. Hướng đi module hóa thông minh.
    *   **2504.07096 – Thuật toán song song tìm đoạn khớp cực đại với 1 truy vấn FIND infini-gram – Suy nghĩ:** Xử lý song song hậu tố của đầu ra LM, tìm tiền tố khớp dài nhất bằng một truy vấn FIND duy nhất dựa trên LCP của các phần tử SA kề cận. Rất hiệu quả cho truy vết quy mô lớn.
    *   **2504.05741 – Decoupled Diffusion Transformer (DDT) & quy hoạch động thống kê chia sẻ self-condition – Suy nghĩ:** Tách Condition Encoder và Velocity Decoder trong DiT; tối ưu chia sẻ đặc trưng tự điều kiện giữa các bước khử nhiễu bằng quy hoạch động. Cải thiện cả chất lượng và tốc độ.
    *   **2504.12285 – Triển khai BitNet b1.58 2B4T (ReLU2, SubLN, SFT loss summation) – Suy nghĩ:** Báo cáo kỹ thuật chi tiết về việc huấn luyện LLM 1-bit quy mô lớn, bao gồm các lựa chọn kiến trúc và quy trình huấn luyện cụ thể. Quan trọng cho việc nhân rộng LLM hiệu quả.
    *   **2504.01014 – Biểu diễn đa phương thức nhận biết hành động (`sa`) & Decoder Adaptation – Suy nghĩ:** Mã hóa cảnh hoạt hình thành `sa` (tham chiếu ảnh, mô tả hành động) và tinh chỉnh bộ giải mã video với đầu ra từ MLLM dự đoán `sa`. Hướng tới game mô phỏng nhất quán.
    *   **2504.02826 – RISEBench & quy trình đánh giá LMM-as-a-Judge cho RISE – Suy nghĩ:** Benchmark chuyên dụng cho chỉnh sửa ảnh dựa trên suy luận và quy trình đánh giá tự động dựa trên LMM theo 3 tiêu chí. Cần thiết cho việc thúc đẩy nghiên cứu về RISE.
    *   **2503.23377 – JavisDiT (DiT hợp nhất với ST-SelfAttn, ST-CrossAttn, MM-BiCrossAttn) & HiST-Sypo Estimator – Suy nghĩ:** Kiến trúc DiT chuyên biệt cho audio-video và bộ ước tính tiền nghiệm không gian-thời gian phân cấp được học bằng tương phản. Giải pháp toàn diện cho sinh A-V đồng bộ.
    *   **2504.17502 – REFVNLI (VLM đa đầu vào) & quy trình tạo dữ liệu nhãn kép tự động – Suy nghĩ:** Metric tự động đánh giá đồng thời phù hợp văn bản và bảo toàn chủ thể, huấn luyện trên dữ liệu tổng hợp với chiến lược tạo cặp dương/âm thông minh. Giải quyết vấn đề đánh giá trong subject-driven T2I.
    *   **2504.11346 – Quy trình huấn luyện nhận biết khuyết tật & Cross-modality RoPE – Suy nghĩ:** Loại bỏ gradient từ vùng khuyết tật trong không gian ẩn và mở rộng RoPE cho token văn bản (coi là 2D) để tăng căn chỉnh. Thực tế và cải thiện chất lượng dữ liệu/mô hình.
    *   **2504.08672 – Stepwise foresight re-sampling & Advantage-Calibrated Optimization (ACO) – Suy nghĩ:** Khám phá/khai thác bước suy luận tối ưu bằng mô phỏng tương lai và hàm mất mát DPO hiệu chỉnh để giảm không nhất quán trong tự huấn luyện không giám sát. Sáng tạo cho RL không giám sát.
    *   **2504.02495 – Self-Principled Critique Tuning (SPCT) & Meta Reward Model – Suy nghĩ:** GRM tự tạo nguyên tắc và phê bình thích ứng qua RFT và RL; Meta RM hướng dẫn bỏ phiếu tổng hợp kết quả từ nhiều lần lấy mẫu GRM. Tăng khả năng mở rộng và chất lượng của RM.
    *   **2504.16656 – Học RL kết hợp (MPO + GRPO) & Selective Sample Buffer (SSB) – Suy nghĩ:** Cân bằng suy luận và tổng quát hóa đa phương thức; SSB chống "lợi thế biến mất" bằng cách ưu tiên mẫu giá trị cao. Giải pháp mạnh mẽ cho VLM RL.
    *   **2503.19901 – TokenHSI (Transformer với tokenizer Tprop/Ttask và masking) & thích ứng chính sách nhẹ – Suy nghĩ:** Hợp nhất nhiều kỹ năng HSI bằng Transformer, chia sẻ kiến thức vận động qua Tprop và thích ứng nhiệm vụ mới bằng adapter/tokenizer nhẹ. Hiệu quả và linh hoạt.
    *   **2504.08388 – Giải mã song song cho sinh video với fine-tuning mặt nạ chú ý tùy chỉnh – Suy nghĩ:** Tăng tốc suy luận mô hình thế giới tự hồi quy bằng cách dự đoán song song token không gian liền kề và tinh chỉnh với mặt nạ chú ý mới. Quan trọng cho tương tác thời gian thực.
    *   **2504.06514 – Chiến lược tạo câu hỏi thiếu tiền đề (MiP) có kiểm soát – Suy nghĩ:** Các phương pháp (Rule-Based, Body-Question Swapping, Essential-Premise Removal) để tạo dữ liệu đánh giá hiện tượng "suy nghĩ thừa khi thiếu tiền đề". Cần thiết để nghiên cứu tư duy phản biện của LLM.
    *   **2504.17432 – UniME (Chưng cất kiến thức phân biệt văn bản KL-div & tinh chỉnh với lọc/lấy mẫu phủ định khó) – Suy nghĩ:** Học embedding phổ quát cho MLLM bằng cách căn chỉnh phân phối với mô hình thầy và tinh chỉnh tương phản với chiến lược xử lý mẫu phủ định thông minh. Cải thiện chất lượng embedding.
    *   **2504.09925 – Mã hóa thị giác hợp nhất hướng dẫn bằng văn bản & Giải mã căn chỉnh đệ quy nhận biết ngữ cảnh – Suy nghĩ:** Tích hợp sâu văn bản vào mã hóa thị giác và tinh chỉnh căn chỉnh thị giác-ngôn ngữ động trong quá trình giải mã LLM. Hướng tới tích hợp đa phương thức chặt chẽ hơn.
    *   **2503.22655 – Chuyển đổi biểu diễn văn bản sang thị giác (ˆv_i = u_i − E[U]) – Suy nghĩ:** Tổng hợp biểu diễn ảnh từ biểu diễn văn bản bằng cách trừ vector trung bình của các biểu diễn văn bản, dựa trên lý thuyết "modality gap". Đột phá trong việc tạo dữ liệu VLM không cần ảnh thật.
    *   **2504.02160 – Universal Rotary Position Embedding (UnoPE) – Suy nghĩ:** Cơ chế nhúng vị trí quay phổ quát trong DiT để giảm nhầm lẫn thuộc tính khi xử lý nhiều đối tượng tham chiếu trong tùy chỉnh đa chủ thể. Giải pháp cụ thể cho vấn đề khó.
    *   **2504.13835 – Thước đo thông tin MIG (lan truyền trên đồ thị nhãn) & thuật toán lấy mẫu tham lam – Suy nghĩ:** Định lượng thống nhất chất lượng và đa dạng dữ liệu instruction-tuning, mô hình hóa quan hệ ngữ nghĩa qua đồ thị nhãn và lan truyền thông tin. Hiệu quả và có cơ sở lý thuyết.
    *   **2504.05305 – Mô hình hóa mặt nạ động & bộ mã hóa mặt nạ chuyên biệt (trong URECA) – Suy nghĩ:** Xử lý mặt nạ vùng ảnh đa độ chi tiết bằng cách chia thành mặt nạ con và mã hóa thành token đặc trưng riêng biệt mà không thay đổi ảnh gốc. Bảo toàn chi tiết vùng tốt.
    *   **2504.13146 – Lấy mẫu chống chưng cất (antidistillation sampling) với xấp xỉ thành phần phạt hiệu quả – Suy nghĩ:** Điều chỉnh phân phối lấy mẫu của mô hình thầy để tạo token làm suy giảm hiệu năng mô hình học sinh, với cách xấp xỉ thành phần phạt chỉ cần 2 lượt truyền xuôi. Sáng tạo để bảo vệ IP.
    *   **2504.00050 – Hàm thưởng chuyên biệt cho JudgeLRM (kết hợp cấu trúc, quan hệ, tuyệt đối, tự tin) – Suy nghĩ:** Thiết kế hàm thưởng đa thành phần cho RL để huấn luyện LLM làm giám khảo, tối ưu hóa khả năng suy luận trong đánh giá. Chi tiết và phù hợp tác vụ.
    *   **2503.20783 – Dr. GRPO (loại bỏ thành phần gây thiên kiến độ dài/độ khó) – Suy nghĩ:** Sửa đổi GRPO bằng cách loại bỏ các yếu tố chuẩn hóa gây thiên kiến trong tính toán lợi thế và gradient. Cải thiện tính không thiên kiến của RL.
    *   **2504.08736 – Semantic regularization (căn chỉnh với DINOv2) & Entropy loss cho tokenizer lớn – Suy nghĩ:** Căn chỉnh đặc trưng tokenizer với DINOv2 để giảm phức tạp không gian ẩn và dùng entropy loss để ổn định huấn luyện tokenizer tỷ tham số. Giải quyết vấn đề cốt lõi khi mở rộng tokenizer.
    *   **2504.15257 – FlowReasoner (meta-agent cấp truy vấn) & hàm thưởng đa mục tiêu cho RL – Suy nghĩ:** Meta-agent tạo hệ thống đa agent riêng cho mỗi truy vấn, học qua RL với hàm thưởng cân bằng hiệu suất, độ phức tạp, tính đa dạng. Hướng đi mới cho thiết kế MAS tự động.
    *   **2504.07960 – VisualCloze (học trong ngữ cảnh trực quan + điền khuyết ảnh) & Graph200K – Suy nghĩ:** Công thức hóa sinh ảnh phổ quát thành điền khuyết trên lưới ảnh ghép, mô hình học từ ví dụ trực quan; bộ dữ liệu đồ thị tăng mật độ tác vụ. Sáng tạo và tận dụng mô hình nền tảng tốt.
    *   **2504.00595 – Đóng gói chuỗi đa phương thức (FFD bin packing) & Mở rộng codebase Prismatic-VLM – Suy nghĩ:** Nhóm các cặp ảnh-chú thích vào chuỗi có tổng độ dài gần max context length để giảm token đệm; tùy chỉnh dataloader cho chuỗi đóng gói. Thực tế và hiệu quả cho huấn luyện MLLM.
    *   **2504.13181 – Language Alignment & Spatial Alignment (chưng cất từ lớp trung gian và SAM 2) – Suy nghĩ:** Điều chỉnh bộ mã hóa thị giác để đưa đặc trưng ngôn ngữ/không gian từ lớp trung gian ra lớp cuối, khai thác tiềm năng ẩn của mô hình huấn luyện tương phản. Khai phá thông minh các biểu diễn đã học.
    *   **2504.09643 – RewardRanker (tự huấn luyện lặp lại reranker với hard negatives từ PPO) – Suy nghĩ:** Cải thiện reranker mã nguồn bằng cách liên tục bổ sung các mẫu phủ định khó do mô hình sinh mã (huấn luyện PPO) tạo ra. Chu trình học tập trung vào cải thiện reward model.
    *   **2504.07957 – Quy trình MM-IFEngine & Chiến lược đánh giá kết hợp MM-IFEval – Suy nghĩ:** Tạo dữ liệu ảnh-chỉ dẫn đa phương thức phức tạp có kiểm soát và phương pháp đánh giá kết hợp (luật, LLM trực tiếp, LLM so sánh). Toàn diện cho MMIF.
    *   **2503.23145 – CodeARC (framework đánh giá tương tác với oracle kiểm thử vi phân) & Fine-tuning chưng cất từ synthesis traces – Suy nghĩ:** Đánh giá tổng hợp chương trình quy nạp tương tác, cho phép agent truy vấn và tinh chỉnh; học sinh bắt chước các bước suy luận của giáo viên (có truy cập ground-truth) nhưng loại bỏ thông tin đặc quyền khỏi loss. Mô phỏng thực tế và huấn luyện thông minh.
    *   **2504.12369 – WorldMem (bộ nhớ đệm với memory attention) & Plücker/relative pose embeddings – Suy nghĩ:** Tích hợp bộ nhớ (khung hình, tư thế, dấu thời gian) vào diffusion video để mô phỏng thế giới nhất quán dài hạn, sử dụng cross-attention với memory frames. Giải quyết tốt vấn đề nhất quán dài hạn.
    *   **2504.13203 – X-Teaming (Planner, Attacker, Verifier, Prompt Optimizer) & Lập kế hoạch/Tối ưu hóa tấn công thích ứng – Suy nghĩ:** Framework đa tác nhân cộng tác để tự động hóa khám phá và tối ưu hóa tấn công bẻ khóa đa lượt, với kế hoạch thích ứng và tối ưu prompt bằng TextGrad. Mạnh mẽ và có hệ thống.
    *   **2504.01990 – Hệ thống hóa kiến thức về tác tử nền tảng, đề xuất kiến trúc module hóa lấy cảm hứng từ não bộ – Suy nghĩ:** Đây là một bài tổng quan (survey) nhưng đóng góp quan trọng ở việc cấu trúc hóa một lĩnh vực phức tạp và đề xuất một khung kiến trúc có tính định hướng cao.
    *   **2504.13837 – Phân tích phê bình về giới hạn của RLVR – Suy nghĩ:** Bài phân tích này chỉ ra rằng RLVR chủ yếu cải thiện hiệu quả lấy mẫu chứ không tạo ra khả năng suy luận mới một cách cơ bản, một nhận định quan trọng cho hướng phát triển của LLM suy luận.
    *   **2504.15279 – VisuLogic benchmark – Suy nghĩ:** Đóng góp chính là bộ benchmark được thiết kế cẩn thận để đo lường suy luận trực quan thuần túy, tránh các lối tắt ngôn ngữ, rất cần thiết cho việc đánh giá MLLM.
    *   **2504.02782 – Phương pháp phân tích dựa trên mô hình phân loại để suy luận kiến trúc GPT-4o – Suy nghĩ:** Một cách tiếp cận dựa trên dữ liệu để thăm dò mô hình hộp đen, cung cấp bằng chứng gián tiếp về kiến trúc.
    *   **2503.23077 – Phân loại các phương pháp suy luận hiệu quả cho LRM – Suy nghĩ:** Bài khảo sát này cung cấp một taxonomy hữu ích, giúp hệ thống hóa một lĩnh vực đang phát triển nhanh.
    *   **2504.20879 – Phân tích các vấn đề và thiên vị trong Chatbot Arena – Suy nghĩ:** Một nghiên cứu metascience quan trọng, chỉ ra các yếu tố gây sai lệch trong một nền tảng đánh giá LLM phổ biến, thúc đẩy tính minh bạch.
    *   **2504.15521 – Quy trình thu thập và chú thích benchmark đa ngôn ngữ quy mô lớn – Suy nghĩ:** Đóng góp chính là việc xây dựng cơ sở dữ liệu và phân tích định lượng về sự tương quan giữa benchmark đa ngôn ngữ và đánh giá của con người.
    *   **2504.05979 – Nghiên cứu thực nghiệm toàn diện về khả năng tạo ảnh của GPT-4o – Suy nghĩ:** Cung cấp một đánh giá định tính sâu rộng, so sánh GPT-4o với các mô hình khác trên nhiều tác vụ, xác định điểm mạnh/yếu.
    *   **2504.08003 – Phương pháp luận đánh giá tích hợp hiểu biết ngữ nghĩa và sinh ảnh của GPT-4o – Suy nghĩ:** Thiết kế các prompt chuyên biệt để kiểm tra các giới hạn của mô hình trong việc diễn giải, áp dụng ràng buộc và suy luận khi sinh ảnh.
    *   **2503.24235 – Khung phân loại đa chiều cho Test-Time Scaling (TTS) – Suy nghĩ:** Đề xuất một taxonomy toàn diện để tổ chức và phân tích các phương pháp TTS trong LLM, rất hữu ích cho việc định hình lĩnh vực.
    *   **2504.05535 – Quy trình chú thích dựa trên LLM cho COIG-P & CRM/CRBench – Suy nghĩ:** Tự động hóa việc xây dựng tập dữ liệu sở thích tiếng Trung quy mô lớn và các công cụ đánh giá đi kèm. Giải quyết nhu cầu cụ thể của cộng đồng tiếng Trung.

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Đánh giá Suy luận Đa phương thức Phức tạp:** Nhiều benchmark hiện tại (ví dụ cho VQA, chỉnh sửa ảnh) chưa đủ để đánh giá các loại suy luận phức tạp (nhân quả, thời gian, không gian sâu) trong MLLM (như RISEBench [2504.02826] đang cố gắng giải quyết nhưng còn sơ khai). Cơ hội: Phát triển benchmark toàn diện hơn, tự động hơn cho suy luận đa phương thức.
    *   **Hiệu quả và Khả năng Mở rộng của RL cho (M)LLM:** Mặc dù có nhiều tiến bộ (1-shot RLVR [2504.20571], Open-Reasoner-Zero [2503.24290], LUFFY [2504.14945]), việc huấn luyện RL quy mô lớn vẫn tốn kém và có thể không ổn định. Cơ hội: Các thuật toán RL mới hiệu quả hơn, ít nhạy cảm với siêu tham số, và các phương pháp tự thưởng/tự phê bình tốt hơn (như SPCT [2504.02495], MPO [2504.20157]).
    *   **Tích hợp Kiến thức Thế giới và Tư duy Phản biện vào (M)LLM:** Các mô hình vẫn gặp khó khăn với các câu hỏi thiếu tiền đề (MiP-Overthinking [2504.06514]) hoặc tích hợp kiến thức động vào sinh ảnh [2504.08003]. Cơ hội: Phát triển kiến trúc và phương pháp huấn luyện giúp mô hình nhận diện sự thiếu thông tin, truy vấn kiến thức cần thiết, và áp dụng tư duy phản biện.
    *   **Nhất quán Dài hạn trong Sinh Video và Tương tác Tác tử:** Việc duy trì tính nhất quán về đối tượng, bối cảnh và logic trong các video dài hoặc các tương tác tác tử kéo dài vẫn là thách thức (WorldMem [2504.12369], FramePack [2504.12626] đang giải quyết). Cơ hội: Các cơ chế bộ nhớ hiệu quả hơn, mô hình hóa thế giới tốt hơn, và khả năng lập kế hoạch dài hạn cho tác tử.
    *   **Tạo Dữ liệu Huấn luyện Chất lượng Cao và Đa dạng một cách Hiệu quả:** Việc phụ thuộc vào ảnh thật [2503.22655] hoặc các LLM lớn để tạo/lọc dữ liệu [2504.05535, 2504.07957, 2504.02160] vẫn là một nút thắt. Cơ hội: Các phương pháp tổng hợp dữ liệu hoàn toàn từ văn bản hoặc các nguồn chi phí thấp hơn, các kỹ thuật tăng cường/lọc dữ liệu thông minh hơn.
    *   **An toàn và Độ tin cậy của (M)LLM:** Các cuộc tấn công bẻ khóa ngày càng tinh vi (X-Teaming [2504.13203]), và việc truy vết nguồn gốc dữ liệu còn hạn chế (OLM OTRACE [2504.07096] là một bước tiến). Cơ hội: Phát triển các cơ chế phòng thủ mạnh mẽ hơn, các phương pháp diễn giải và truy vết tốt hơn, và các kỹ thuật chống chưng cất/đánh cắp mô hình hiệu quả (như Antidistillation Sampling [2504.13146]).
    *   **Hiệu quả Triển khai (M)LLM trên Thiết bị Hạn chế:** Chạy các mô hình lớn trên thiết bị gia đình/di động vẫn khó khăn (prima.cpp [2504.08791], BitNet [2504.12285]). Cơ hội: Các kỹ thuật lượng tử hóa cực thấp tốt hơn, kiến trúc song song/phân tán hiệu quả hơn, và các thuật toán tối ưu hóa suy luận chuyên biệt.
    *   **Tích hợp Đa phương thức Sâu và Linh hoạt:** Việc kết hợp các phương thức (hình ảnh, video, âm thanh, văn bản, mã) một cách thực sự sâu sắc và linh hoạt vẫn đang được khám phá (InternVL3 [2504.10479], FUSION [2504.09925], UniversalRAG [2504.20734]). Cơ hội: Các kiến trúc hợp nhất tốt hơn, cơ chế attention/routing đa phương thức tiên tiến, và hiểu biết sâu hơn về "modality gap".
    *   **Tự động hóa Quy trình Nghiên cứu và Phát triển AI:** Các tác tử AI có thể tự tái tạo nghiên cứu (PaperBench [2504.01848], PaperCoder [2504.17192]) hoặc tự thiết kế hệ thống (FlowReasoner [2504.15257]) là một hướng đi mới. Cơ hội: Nâng cao khả năng lập kế hoạch, suy luận và học hỏi của các tác tử AI này.
    *   **Đánh giá (M)LLM Toàn diện và Đáng tin cậy:** Các benchmark hiện tại có thể bị "gaming" hoặc không phản ánh đúng năng lực thực tế (phân tích Chatbot Arena [2504.20879]). Cơ hội: Phát triển các phương pháp đánh giá mới, mạnh mẽ hơn, ít bị khai thác hơn, và có khả năng đánh giá các năng lực phức hợp (như PHYBench [2504.16074], RISEBench [2504.02826], MM-IFEval [2504.07957]).

5.  **FUTURE_IDEAS**

    ✨ **Meta-Cognitive Loop for Robust LLM Reasoning**
    *   **Motivation:** LLMs often "overthink" when faced with ill-posed or missing-premise questions (MiP-Overthinking [2504.06514]) and struggle with dynamic knowledge integration [2504.08003]. Current RLHF methods might not sufficiently penalize such behaviors.
    *   **Key novelty:** An explicit meta-cognitive loop where the LLM first generates an initial reasoning trace, then a "critique agent" (potentially a specialized instance of the same LLM or a smaller, fine-tuned model like JudgeLRM [2504.00050]) evaluates the trace for logical fallacies, missing premises, or inconsistencies with known facts (potentially retrieved via a tool like in ReTool [2504.11536] or T1 [2504.04718]). The LLM then revises its reasoning based on this critique. This goes beyond simple self-correction by having a dedicated critique phase informed by externalized principles or retrieved knowledge.
    *   **Approach:**
        1.  LLM generates initial CoT.
        2.  Critique Agent (CA) analyzes CoT:
            *   Checks for missing premises (using techniques inspired by MiP dataset generation [2504.06514]).
            *   Verifies factual claims/calculations using tools (ToolV from T1 [2504.04718]).
            *   Assesses logical consistency.
        3.  CA provides structured feedback to LLM.
        4.  LLM revises CoT based on feedback.
        5.  Train the LLM and CA jointly, possibly using RL where the reward is based on the final answer correctness and the quality/actionability of the critique. The Meta Reward Model concept from [2504.02495] could inform CA training.
    *   **Dataset + Metrics:** MiP datasets [2504.06514], PHYBench [2504.16074], standard reasoning benchmarks. Metrics: Accuracy, ability to identify unanswerable questions, quality of revised reasoning.
    *   **Risk/Feasibility:** High feasibility for initial LLM + CA. Training CA effectively and the joint RL process could be complex. Risk of critique agent introducing its own biases.

    ✨ **Adaptive Multimodal Synthesis via Learned Resource Allocation (Cross-Domain with Generative Models & Efficient Inference)**
    *   **Motivation:** Generating complex, long-form multimodal content (e.g., detailed narrated videos [2503.24379], interactive game environments [2504.01014]) requires dynamic allocation of computational resources and attention across modalities and temporal/spatial scales. Fixed architectures or manual control are suboptimal.
    *   **Key novelty:** A meta-controller LLM (inspired by FlowReasoner [2504.15257] and APR [2504.15466]) that learns to dynamically orchestrate a suite of specialized generative (sub-)agents (e.g., a text-to-image generator like in [2504.11346], a video motion generator like in [2504.04842], an audio synthesizer like JavisDiT [2503.23377]) and memory modules (like WorldMem [2504.12369]). The controller decides which agent to activate, what information to pass, how much detail is needed for each component, and when to merge/refine outputs, based on an overall goal and incoming user interactions. It would also manage context compression (FramePack [2504.12626]) and attention (MTA [2504.00927]).
    *   **Approach:**
        1.  Define a set of specialized generative agents and memory modules.
        2.  Train a meta-controller LLM using RL (e.g., GRPO variants [2503.20783]) with a reward function that balances output quality, coherence, resource usage, and user satisfaction.
        3.  The meta-controller generates a "generation plan" (which can be dynamic) involving calls to sub-agents, possibly using `spawn()`/`join()` like APR [2504.15466].
        4.  Sub-agents execute, and their outputs are fed back to the controller for next steps.
    *   **Dataset + Metrics:** WIKIVIDEO [2504.00939] for article generation from video, LiveSports-3K [2504.16030] for commentary, custom datasets for interactive storytelling. Metrics: Coherence, fidelity to input, user engagement, computational cost.
    *   **Risk/Feasibility:** Medium-to-High. Training the meta-controller is challenging. Ensuring seamless integration and communication between diverse sub-agents is complex. High potential for creating truly dynamic and rich multimodal experiences.

    ✨ **Self-Evolving Foundation Agents via Decentralized Knowledge Fusion (Moon-shot)**
    *   **Motivation:** Current foundation agents [2504.01990] are typically monolithic or centrally trained. A truly intelligent and adaptable AI ecosystem might emerge from decentralized agents that can learn, share, and fuse knowledge without a central coordinator, robust to individual agent failures or biases.
    *   **Key novelty:** A framework where multiple, independently evolving "foundation agents" (each potentially with different specializations or learning from different data streams, e.g., some are expert visual encoders [2504.13181], some are reasoning experts [2503.24290], some are multilingual [2504.15120]) can discover each other, negotiate knowledge exchange protocols, and fuse their "world models" or specialized skills. This involves learning "meta-skills" for knowledge representation, translation, and integration. The "antidistillation sampling" [2504.13146] concept could be inverted to "pro-collaboration sampling" where agents generate data specifically to teach others.
    *   **Approach:**
        1.  Develop a communication and knowledge representation protocol for heterogeneous agents.
        2.  Each agent has a local world model and a set of skills.
        3.  Agents can broadcast "knowledge offerings" or "knowledge needs."
        4.  Implement RL-based mechanisms for agents to:
            *   Evaluate the trustworthiness/utility of knowledge from other agents (inspired by X-Teaming's Verifier [2504.13203] but for positive knowledge).
            *   Learn to translate and integrate external knowledge into their own models (potentially using techniques like efficient language/skill injection [2504.15120] or adaptive MoE routing [2504.07964]).
            *   Develop strategies for collaborative problem-solving on tasks too complex for any single agent.
        5.  The "reward" for an agent could be its ability to solve more complex tasks after knowledge fusion, or the utility of its shared knowledge to other agents.
    *   **Dataset + Metrics:** Complex, open-ended multi-agent benchmarks (e.g., collaborative research on PaperBench [2504.01848], large-scale distributed data analysis). Metrics: Collective intelligence, robustness to agent failure, speed of knowledge propagation, emergence of novel capabilities.
    *   **Risk/Feasibility:** Very High (Moon-shot). Defining stable knowledge fusion mechanisms, ensuring beneficial collaboration over exploitation, and managing computational complexity are immense challenges. However, success could lead to a paradigm shift in AI development.

6.  **READING_LIST**

    *   `2504.10479` – InternVL3 · Groundbreaking native multimodal pre-training approach.
    *   `2504.07491` – Kimi-VL · Demonstrates impressive long-context (128K) and long-CoT reasoning in VLM.
    *   `2503.24290` – Open-Reasoner-Zero · Showcases minimalist, yet powerful, RL for LLM reasoning without SFT.
    *   `2504.07096` – OLM OTRACE · Real-time tracing of LLM outputs to massive training datasets is a critical step for transparency.
    *   `2504.01990` – Foundation Agents Survey · Comprehensive overview and brain-inspired architectural proposal for an emerging field.
    *   `2503.22655` – Unicorn · Novel text-only VLM data synthesis method, potentially revolutionizing VLM training.
    *   `2504.13837` – RLVR Critique · Important critical analysis of the true capabilities endowed by RLVR.
    *   `2504.12285` – BitNet b1.58 2B4T · Significant milestone in ultra-efficient 1-bit LLMs.

7.  **META_REFLECTION**

    Tập hợp các bài báo tháng 04/2025 cho thấy một số xu hướng phát triển AI nổi bật.
    Thứ nhất, **hiệu quả (efficiency)** tiếp tục là một chủ đề nóng, thể hiện qua các nghiên cứu về mô hình 1-bit (BitNet [2504.12285]), suy luận trên thiết bị tài nguyên thấp (prima.cpp [2504.08791]), cắt tỉa gradient thích ứng (ZClip [2504.02507]), và các chiến lược huấn luyện/suy luận hiệu quả cho (M)LLM (Open-Reasoner-Zero [2503.24290], 1-shot RLVR [2504.20571], Tina [2504.15777], Open-Qwen2VL [2504.00595]).
    Thứ hai, **khả năng suy luận (reasoning)** của (M)LLM được đẩy mạnh thông qua các phương pháp học tăng cường tiên tiến (GRPO và các biến thể [2503.20783, 2504.14945, 2504.00883, 2504.08837], TTRL [2504.16084], ReTool [2504.11536]), tích hợp công cụ (T1 [2504.04718]), và các kiến trúc/chiến lược mới cho suy luận song song/thích ứng (Hogwild! [2504.06261], APR [2504.15466]). Đồng thời, có sự nhìn nhận sâu sắc hơn về giới hạn của các phương pháp hiện tại (RLVR Critique [2504.13837], MiP-Overthinking [2504.06514]).
    Thứ ba, **tích hợp đa phương thức (multimodal integration)** ngày càng trở nên tinh vi, với các phương pháp tiền huấn luyện hợp nhất (InternVL3 [2504.10479]), kiến trúc xử lý ngữ cảnh dài và suy luận phức tạp (Kimi-VL [2504.07491]), và các kỹ thuật căn chỉnh/nhúng đa phương thức tiên tiến (FUSION [2504.09925], UniME [2504.17432], PE [2504.13181]). Việc tạo sinh video có điều khiển và đồng bộ hóa audio-video cũng có những bước tiến đáng kể (JavisDiT [2503.23377], ACTalker [2504.02542], SkyReels-A2 [2504.02436], DreamActor-M1 [2504.01724]).
    Thứ tư, **tự động hóa và tự cải thiện (automation and self-improvement)** là một hướng đi rõ rệt, từ việc tự động hóa thiết kế hệ thống đa agent (FlowReasoner [2504.15257]), tự động tạo kho mã nguồn (PaperCoder [2504.17192]), đến các mô hình tự huấn luyện/tự tinh chỉnh (Genius [2504.08672], RewardRanker [2504.09643], MPO [2504.20157]).
    Cuối cùng, **đánh giá và tính minh bạch (evaluation and transparency)** ngày càng được chú trọng, với sự ra đời của nhiều benchmark chuyên biệt (RISEBench [2504.02826], PHYBench [2504.16074], SEED-Bench-R1 [2503.24376], MM-IFEval [2504.07957], PaperBench [2504.01848], VisuLogic [2504.15279]), các phương pháp truy vết (OLM OTRACE [2504.07096]), và các phân tích phê bình về nền tảng đánh giá hiện có (Chatbot Arena analysis [2504.20879]). Điều này cho thấy sự trưởng thành của lĩnh vực, hướng tới các mô hình AI mạnh mẽ hơn, hiệu quả hơn, dễ hiểu hơn và đáng tin cậy hơn.
