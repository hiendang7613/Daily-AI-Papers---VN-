1.  **TOPIC_TREE**
    *   I. Mô hình Ngôn ngữ Lớn (LLMs) & Xử lý Ngôn ngữ Tự nhiên (NLP)
        *   A. Tiền huấn luyện (Pre-training) & Quản lý Dữ liệu (Data Curation)
            *   1. Chiến lược Xây dựng Tập dữ liệu Quy mô Lớn
                *   2406.17557, 2406.11794 | Xu hướng xây dựng các bộ dữ liệu tiền huấn luyện mở, quy mô lớn với quy trình xử lý minh bạch và các nghiên cứu loại trừ sâu rộng để tối ưu hóa.
            *   2. Chiến lược Xây dựng Tập dữ liệu cho Mô hình Ngôn ngữ Nhỏ (SLMs)
                *   2406.11410 | Xu hướng tập trung vào việc xây dựng dữ liệu chất lượng cao, đa dạng ngữ nghĩa và ít rò rỉ cho SLM dựa trên tri thức tiên nghiệm của con người.
            *   3. Kỹ thuật Tiền huấn luyện Mới
                *   2406.14491 | Xu hướng tích hợp học có giám sát (instruction) vào giai đoạn tiền huấn luyện thông qua tổng hợp chỉ dẫn từ chính kho ngữ liệu thô.
        *   B. Tinh chỉnh theo Chỉ dẫn (Instruction Tuning) & Căn chỉnh (Alignment)
            *   1. Tổng hợp Dữ liệu Căn chỉnh
                *   2406.08464 | Xu hướng phát triển các phương pháp tự động, chi phí thấp để tổng hợp dữ liệu căn chỉnh quy mô lớn bằng cách khai thác khả năng của LLM đã căn chỉnh.
            *   2. Tối ưu hóa Sở thích (Preference Optimization)
                *   2406.00888, 2406.09760, 2406.18629 | Xu hướng phát triển các phương pháp tối ưu hóa sở thích lặp đi lặp lại, ít dữ liệu hoặc tập trung vào từng bước suy luận để cải thiện hiệu năng và độ tin cậy.
            *   3. Chưng cất Kiến thức cho Tạo Dữ liệu
                *   2406.19227 | Xu hướng tạo dữ liệu huấn luyện phù hợp cho chưng cất kiến thức bằng cách căn chỉnh mô hình giáo viên theo "sở thích" của mô hình học sinh.
        *   C. Khả năng Suy luận (Reasoning Capabilities)
            *   1. Tăng cường Suy luận bằng Tư duy Trừu tượng/Có Cấu trúc
                *   2406.04271, 2406.09308, 2406.06592 | Xu hướng tăng cường khả năng suy luận của LLM bằng cách tích hợp các mẫu tư duy trừu tượng, bộ suy luận thuật toán thần kinh, hoặc tìm kiếm trên cây Monte Carlo.
        *   D. Kiến trúc Hiệu quả & Suy luận Tối ưu (Efficient Architectures & Inference)
            *   1. Mô hình Không gian Trạng thái (SSMs) & Kiến trúc Lai
                *   2405.21060, 2406.07522 | Xu hướng phát triển các SSM hiệu quả hơn (Mamba-2) và các kiến trúc lai (SAMBA) kết hợp điểm mạnh của SSM và Attention.
            *   2. Kiến trúc Phân cấp (Hierarchical Architectures)
                *   2406.02657 | Xu hướng thiết kế kiến trúc Transformer phân cấp (Block Transformer) để giảm tắc nghẽn suy luận tự hồi quy và chi phí KV cache.
            *   3. Thưa hóa Kích hoạt (Activation Sparsification)
                *   2406.05955 | Xu hướng phát triển các hàm kích hoạt mới (dReLU) để tăng cường độ thưa của kích hoạt nơ-ron, cải thiện hiệu quả suy luận.
            *   4. Tối ưu hóa Suy luận trên Thiết bị Di động
                *   2406.06282 | Xu hướng phát triển các hệ thống suy luận LLM chuyên biệt cho thiết bị di động (PowerInfer-2) với các kỹ thuật thích ứng độ thưa và điều phối I/O.
        *   E. Kỹ thuật Prompt (Prompt Engineering)
            *   1. Tổng quan và Hệ thống hóa
                *   2406.06608 | Xu hướng hệ thống hóa kiến thức, thuật ngữ và phân loại các kỹ thuật prompt hiện có.
        *   F. Trích xuất Thông tin (Information Extraction)
            *   1. Mô hình Trích xuất Đa nhiệm dựa trên Token
                *   2406.12925 | Xu hướng mở rộng các mô hình dựa trên encoder (GLiNER) để thực hiện nhiều tác vụ IE ở cấp độ token, tăng tính linh hoạt.
        *   G. Hệ thống Tác tử (Agent Systems) & Sử dụng Công cụ (Tool Use)
            *   1. Kiến trúc Đa Tác tử & Tinh chỉnh Lặp lại
                *   2406.04692, 2406.01014, 2406.06469, 2406.11912, 2406.19226 | Xu hướng phát triển các hệ thống đa tác tử phức tạp, có khả năng lập kế hoạch, phản ánh và cộng tác để giải quyết các nhiệm vụ đa dạng (vận hành UI, phát triển phần mềm, mô phỏng lớp học).
            *   2. Tác tử Tự chủ trên Thiết bị & Gọi Hàm
                *   2406.18082, 2406.12793, 2406.18518 | Xu hướng phát triển các tác tử LLM có khả năng hoạt động tự chủ trên thiết bị di động và tạo dữ liệu gọi hàm chất lượng cao, có thể kiểm chứng.
        *   H. Truy xuất Thông tin Tăng cường (Retrieval-Augmented Generation - RAG)
            *   1. Thiết kế Khung RAG & Xử lý Ngữ cảnh Dài
                *   2406.15319 | Xu hướng thiết kế lại kiến trúc RAG (LongRAG) để cân bằng tải trọng retriever/reader và tận dụng LLM ngữ cảnh dài.
            *   2. RAG Thích ứng & Tự nhận thức
                *   2406.19215 | Xu hướng phát triển các phương pháp RAG thích ứng (SEAKR) dựa trên độ bất định tự nhận thức của LLM để điều khiển truy xuất và tích hợp tri thức.
        *   I. Hiểu biết về LLM (Understanding LLMs)
            *   1. Phân tích Thu nhận Kiến thức
                *   2406.11813 | Xu hướng phân tích chi tiết cơ chế LLM thu nhận và lãng quên kiến thức trong quá trình tiền huấn luyện.
            *   2. Phát hiện Ảo giác (Hallucination Detection)
                *   2406.02543 | Xu hướng phát triển các phương pháp dựa trên lý thuyết thông tin để định lượng và phát hiện ảo giác bằng cách phân biệt các loại không chắc chắn.
        *   J. Hội thoại & Bộ nhớ Dài hạn (Dialogue & Long-term Memory)
            *   1. Quản lý Bộ nhớ dựa trên Đồ thị
                *   2406.10996 | Xu hướng xây dựng bộ nhớ dài hạn có cấu trúc đồ thị (THEANINE) liên kết các ký ức bằng quan hệ nhân quả và thời gian.
    *   II. Thị giác Máy tính (Computer Vision - CV) & AI Đa phương thức (Multimodal AI)
        *   A. Sinh Ảnh (Image Generation)
            *   1. Mô hình Tự hồi quy & Kiến trúc Llama
                *   2406.06525 | Xu hướng áp dụng kiến trúc LLM (Llama) thuần túy cho tác vụ sinh ảnh tự hồi quy chất lượng cao.
            *   2. Lượng tử hóa Mô hình Khuếch tán
                *   2406.04333 | Xu hướng phát triển các phương pháp lượng tử hóa cực thấp (BitsFusion) cho mô hình khuếch tán lớn mà vẫn duy trì chất lượng.
            *   3. Kiến trúc Mô hình Khuếch tán Mới
                *   2406.09416 | Xu hướng thiết kế kiến trúc khuếch tán đa phân giải (DiMR) và cơ chế điều kiện thời gian hiệu quả (TD-LN).
        *   B. Sinh Ảnh/3D có Điều khiển (Controllable Image/3D Generation)
            *   1. Kiểm soát Số lượng & Bố cục Đối tượng
                *   2406.10210 | Xu hướng khai thác đặc trưng nội tại của mô hình khuếch tán (self-attention) và học sửa lỗi bố cục để kiểm soát số lượng đối tượng.
            *   2. Sinh Mô hình 3D Động vật có Điều khiển Tư thế
                *   2406.16273 | Xu hướng tích hợp ControlNet chuyên biệt và LLM đa tác tử để sinh mô hình động vật 3D (YOUDREAM) nhất quán về giải phẫu và có thể điều khiển tư thế.
            *   3. Sinh Lưới 3D có Điều kiện Hình dạng
                *   2406.10163 | Xu hướng định nghĩa lại bài toán trích xuất lưới như sinh lưới nghệ sĩ có điều kiện hình dạng (MeshAnything), tập trung vào topology.
        *   C. Sinh Video & Hiểu Video (Video Generation & Understanding)
            *   1. Chú thích Video Chi tiết & Nhất quán
                *   2406.04325 | Xu hướng phát triển các chiến lược tạo phụ đề video chi tiết (DiffSW) bằng cách mô tả thay đổi giữa các khung hình chính.
            *   2. Sao chép & Điều khiển Chuyển động Video
                *   2406.05338, 2406.04277 | Xu hướng sử dụng biểu diễn chuyển động thưa từ attention thời gian (MotionClone) hoặc tổng hợp không gian-thời gian (VideoTetris) để điều khiển sinh video mà không cần huấn luyện lại.
            *   3. Sinh Video Dài & Suy luận Phân tán
                *   2406.16260 | Xu hướng phát triển quy trình suy luận phân tán (Video-Infinity) và cơ chế chú ý mới (Dual-scope attention) để tạo video dài mạch lạc.
            *   4. Sinh Video Một Bước
                *   2406.04324 | Xu hướng tinh chỉnh mô hình khuếch tán video bằng huấn luyện đối nghịch (SF-V) để tạo video chất lượng cao chỉ trong một bước.
            *   5. Tối ưu Attention cho Mô hình Khuếch tán Video
                *   2406.08552 | Xu hướng phát triển các kỹ thuật nén post-training (DiTFastAttn) để giảm chi phí tính toán của self-attention trong Diffusion Transformers.
        *   D. Mô hình Ngôn ngữ-Thị giác (VLMs/LMMs/MLLMs)
            *   1. Kiến trúc & Thành phần Kết nối
                *   2406.16860, 2406.07476, 2406.19389 | Xu hướng phát triển các bộ kết nối thị giác-ngôn ngữ hiệu quả hơn (SVA, STC) và kiến trúc hợp nhất (OMG-LLaVA) cho nhiều cấp độ hiểu.
            *   2. Xử lý Ngữ cảnh Dài & Video Trực tuyến
                *   2406.16852, 2406.11816 | Xu hướng chuyển giao khả năng xử lý ngữ cảnh dài từ LLM sang LMM cho video (UniRes) và phát triển khung làm việc cho hội thoại trực tuyến trong luồng video (LIVE).
            *   3. Căn chỉnh & Tối ưu hóa Sở thích Đa phương thức
                *   2406.11839 | Xu hướng điều chỉnh các phương pháp tối ưu hóa sở thích (MDPO) để giải quyết các vấn đề đặc thù của dữ liệu đa phương thức.
            *   4. Nén Token Thị giác & Kiến trúc Hiệu quả
                *   2406.12275, 2406.12246 | Xu hướng sử dụng chính LLM để nén token thị giác (VoCo-LLaMA) hoặc tái sử dụng lớp (TroL) nhằm tăng hiệu quả.
            *   5. Căn chỉnh Đa ngôn ngữ
                *   2406.02539 | Xu hướng phát triển các mô-đun (PARROT) để cải thiện khả năng xử lý đa ngôn ngữ của MLLM bằng cách điều chỉnh token thị giác theo ngôn ngữ đầu vào.
            *   6. Tách rời Nhận thức & Suy luận
                *   2406.14544 | Xu hướng đề xuất các framework (Prism) để tách rời và đánh giá độc lập khả năng nhận thức và suy luận của VLM.
        *   E. Dữ liệu Đa phương thức & Tái chú thích
            *   1. Xây dựng Tập dữ liệu Xen kẽ Ảnh-Văn bản Quy mô Lớn
                *   2406.08418 | Xu hướng xây dựng các bộ dữ liệu xen kẽ ảnh-văn bản cực lớn (OmniCorpus) với quy trình xử lý hiệu quả.
            *   2. Tái chú thích Ảnh Quy mô Lớn
                *   2406.08478 | Xu hướng sử dụng các MLLM mạnh (LLaVA-LLaMA-3) để tái chú thích và cải thiện chất lượng các bộ dữ liệu ảnh-văn bản hiện có.
            *   3. Dữ liệu 3D-Văn bản Neo đậu Dày đặc
                *   2406.05132 | Xu hướng sử dụng LLM để tạo các bộ dữ liệu 3D-văn bản (3D-GRAND) với chú thích được neo đậu dày đặc ở cấp độ cụm danh từ.
        *   F. Ước lượng Độ sâu & Biểu diễn Video (Depth Estimation & Video Representation)
            *   1. Ước lượng Độ sâu Đơn mắt (Monocular Depth Estimation)
                *   2406.09414 | Xu hướng sử dụng chiến lược dữ liệu kết hợp tổng hợp-thực tế và kiến trúc teacher-student để cải thiện MDE.
            *   2. Ước lượng Độ sâu 360 độ
                *   2406.12849 | Xu hướng sử dụng học bán giám sát và chưng cất kiến thức từ mô hình phối cảnh cho ước lượng độ sâu 360.
            *   3. Biểu diễn Video dựa trên Ảnh Canonical
                *   2406.06523 | Xu hướng tích hợp diffusion prior và trường biến dạng lai (NaRCan) để tạo ảnh canonical tự nhiên cho chỉnh sửa video.
        *   G. Nghịch đảo GAN & Chỉnh sửa Ảnh (GAN Inversion & Image Editing)
            *   1. Nghịch đảo StyleGAN & Chỉnh sửa Chi tiết Cao
                *   2406.10601 | Xu hướng phát triển các bộ mã hóa (StyleFeatureEditor) cho phép chỉnh sửa ảnh chi tiết cao trong không gian đặc trưng StyleGAN.
            *   2. Chỉnh sửa Ảnh dựa trên Tham chiếu
                *   2406.07547 | Xu hướng phát triển các phương pháp chỉnh sửa bắt chước tham chiếu (MimicBrush) không cần mặt nạ, học từ video.
        *   H. Mã hóa Ảnh (Image Tokenization)
            *   1. Mã hóa Ảnh thành Chuỗi Latent 1D
                *   2406.07550 | Xu hướng khám phá các phương pháp mã hóa ảnh thành chuỗi latent 1D cực kỳ nhỏ gọn (TiTok) bằng Transformer.
        *   I. Phân tích Kiến trúc Thị giác (Vision Architecture Analysis)
            *   1. Vai trò của Thiên kiến Quy nạp (Inductive Bias)
                *   2406.09415 | Xu hướng nghiên cứu và đánh giá lại vai trò của các thiên kiến quy nạp như tính cục bộ trong kiến trúc thị giác.
        *   J. Tinh chỉnh Mô hình Khuếch tán (Diffusion Model Fine-tuning)
            *   1. Tối ưu hóa Sở thích Nhận biết Bước & Nhận thức
                *   2406.04314, 2406.17636 | Xu hướng phát triển các phương pháp tối ưu hóa sở thích (SPO, NCPPO) cho mô hình khuếch tán tập trung vào từng bước khử nhiễu hoặc không gian embedding nhận thức.
            *   2. Chưng cất Nhất quán Đảo ngược (Invertible Consistency Distillation)
                *   2406.14539 | Xu hướng phát triển các phương pháp chưng cất (iCD) cho phép cả sinh ảnh nhanh và mã hóa ảnh chính xác.
    *   III. Học Máy Cốt lõi & Hệ thống (Machine Learning Core & Systems)
        *   A. Thuật toán Tối ưu hóa (Optimization Algorithms)
            *   1. Phương pháp Tốc độ Học Thích ứng (Adaptive Learning Rate Methods)
                *   2406.16793 | Xu hướng phát triển các biến thể Adam (Adam-mini) tiết kiệm bộ nhớ dựa trên phân chia tham số theo Hessian.
            *   2. Tối ưu hóa Hệ thống AI Phức hợp bằng LLM
                *   2406.07496 | Xu hướng đề xuất các khung làm việc (TEXTGRAD) sử dụng phản hồi văn bản từ LLM để tối ưu hóa các hệ thống AI phức hợp.
        *   B. Học Liên tục (Continual Learning)
            *   1. Cập nhật Tham số Chọn lọc Không cần Nhãn Tác vụ
                *   2406.17245 | Xu hướng phát triển các phương pháp cập nhật gradient chọn lọc (MIGU) dựa trên độ lớn đầu ra để giảm quên lãng trong học liên tục.
        *   C. Hợp nhất Mô hình (Model Merging)
            *   1. Hợp nhất Mô hình Nhận biết An toàn
                *   2406.14563 | Xu hướng tích hợp yếu tố an toàn vào quá trình hợp nhất mô hình bằng cách tạo và sử dụng dữ liệu an toàn tổng hợp.
    *   IV. Ứng dụng AI (AI Applications)
        *   A. Robot học (Robotics)
            *   1. Mô hình Ngôn ngữ-Thị giác-Hành động (VLAs)
                *   2406.09246 | Xu hướng phát triển các VLA mã nguồn mở (OpenVLA) và các chiến lược huấn luyện/tinh chỉnh hiệu quả cho chúng.
            *   2. Học Tăng cường trong Ngữ cảnh (In-Context RL)
                *   2406.08973 | Xu hướng xây dựng các bộ dữ liệu quy mô lớn (XLand-100B) chứa lịch sử học hoàn chỉnh cho nghiên cứu in-context RL.
        *   B. Hóa học Tính toán & Vật lý (Computational Chemistry & Physics)
            *   1. Tập dữ liệu & Benchmark cho Hóa học Lượng tử
                *   2406.14347 | Xu hướng tạo ra các bộ dữ liệu (∇2DFT) và benchmark toàn diện cho việc huấn luyện các mô hình NNP, đặc biệt là quỹ đạo tối ưu hóa cấu trúc.
            *   2. Mô phỏng Vật lý & Học Tham số từ Video
                *   2406.04338, 2406.17763 | Xu hướng sử dụng mô hình khuếch tán để học tham số vật lý từ video (Physics3D) hoặc giải PDE dưới quan sát thưa thớt (DiffusionPDE).
        *   C. Tổng hợp Giọng nói (Text-to-Speech - TTS)
            *   1. Mô hình TTS Quy mô Lớn & Điều khiển Nâng cao
                *   2406.02430 | Xu hướng phát triển các hệ thống TTS quy mô lớn (Seed-TTS) với khả năng học zero-shot và các kỹ thuật cải thiện như tự chưng cất, RL.
        *   D. AI cho Sức khỏe Cá nhân (AI for Personal Health)
            *   1. LLM cho Dữ liệu Thiết bị Đeo
                *   2406.06474 | Xu hướng tinh chỉnh LLM (PH-LLM) để hiểu và lập luận trên dữ liệu sức khỏe cá nhân dạng chuỗi thời gian từ thiết bị đeo.
        *   E. AI cho Mã nguồn (AI for Code)
            *   1. LLM cho Sinh mã & Toán học
                *   2406.11931 | Xu hướng huấn luyện tiếp tục các mô hình MoE lớn (DeepSeek-Coder-V2) chuyên biệt cho code và toán.
            *   2. Tăng cường Dữ liệu Code Đa ngôn ngữ
                *   2406.07436 | Xu hướng phát triển các chiến lược tăng cường dữ liệu (Cross-lingual Code Transfer) bằng cách chuyển đổi và tăng độ phức tạp của code giữa các ngôn ngữ.
    *   V. Đánh giá & Benchmark trong AI (AI Evaluation & Benchmarking)
        *   A. Đánh giá LLM & MLLM Tổng quát
            *   1. Benchmark Suy luận & Kiến thức Nâng cao
                *   2406.01574, 2406.04770 | Xu hướng xây dựng các benchmark (MMLU-Pro, WildBench) khó hơn, thực tế hơn và có quy trình đánh giá bằng LLM cải tiến.
            *   2. Phân tích Chất lượng Benchmark & LLM-as-a-Judge
                *   2406.04127, 2406.12624 | Xu hướng phân tích lỗi trong các benchmark hiện có (MMLU-Redux) và đánh giá hiệu suất, hạn chế của phương pháp LLM-as-a-judge.
        *   B. Đánh giá Đa phương thức & Video
            *   1. Benchmark Hiểu Hội thoại Đa phương thức
                *   2406.11833 | Xu hướng tạo benchmark (MMDU) và bộ dữ liệu tinh chỉnh cho năng lực hội thoại đa lượt, đa hình ảnh của LVLM.
            *   2. Benchmark Hiểu Tài liệu Đa phương thức Dài
                *   2406.07230 | Xu hướng phát triển benchmark (MM-NIAH) để đánh giá khả năng hiểu tài liệu đa phương thức dài của MLLM.
            *   3. Benchmark Năng lực Xử lý Ngữ cảnh Dài Đa phương thức
                *   2406.11230, 2406.14515 | Xu hướng tạo benchmark (MMNeedle, MMBench-Video) để đánh giá khả năng xử lý ngữ cảnh dài của MLLM trên ảnh ghép và video dài.
            *   4. Benchmark Mô hình hóa Thế giới qua Video
                *   2406.08407 | Xu hướng xây dựng benchmark (MMWorld) đánh giá khả năng MLLM mô hình hóa thế giới thông qua video đa lĩnh vực và lý luận đa diện.
            *   5. Metric Đánh giá Video Time-Lapse
                *   2406.18522 | Xu hướng đề xuất các metric tự động mới (MTScore, CHScore) để đánh giá biên độ biến đổi và sự gắn kết thời gian của video time-lapse.
        *   C. Benchmark Chuyên biệt
            *   1. Benchmark Sinh mã & Hiểu Biểu đồ
                *   2406.15877, 2406.09961, 2406.18521 | Xu hướng xây dựng các benchmark (BigCodeBench, ChartMimic, CharXiv) cho các tác vụ phức tạp như sinh mã sử dụng thư viện đa dạng và hiểu biểu đồ khoa học.
            *   2. Benchmark Truy xuất Thông tin & RAG
                *   2406.16048, 2406.04744 | Xu hướng tạo các benchmark (D-MERIT, CRAG) với chú thích đầy đủ hơn hoặc mô phỏng truy xuất web/KG thực tế cho RAG.
            *   3. Benchmark Ảo giác Đối tượng 3D
                *   2406.05132 | Xu hướng thiết kế benchmark (3D-POPE) để đánh giá có hệ thống hiện tượng ảo giác đối tượng trong 3D-LLM.
            *   4. Benchmark Suy luận Ngữ cảnh Dài & Thời gian
                *   2406.10149, 2406.09170 | Xu hướng tạo benchmark (BABILong, ToT) để đánh giá khả năng suy luận trên ngữ cảnh cực dài và suy luận thời gian một cách tách biệt.
            *   5. Benchmark Sinh ảnh Cá nhân hóa
                *   2406.16855 | Xu hướng xây dựng benchmark (DREAM BENCH++) và phương pháp đánh giá tự động dựa trên GPT-4o cho sinh ảnh cá nhân hóa.
        *   D. Tài nguyên Dữ liệu Ngôn ngữ Khu vực
            *   1. Tổng hợp Tài nguyên cho Ngôn ngữ Đông Nam Á
                *   2406.10118 | Xu hướng tổng hợp và tiêu chuẩn hóa các kho ngữ liệu đa ngôn ngữ và đa phương thức cho các ngôn ngữ ít tài nguyên (SEACrowd).
    *   VI. Other
        *   (Không có paper nào thuộc nhóm này dựa trên phân tích)
    ```

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                       | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                                  |
    | :--- | :-------- | :----------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2405.21060 | SSM, Mamba-2, State Space Duality, Efficiency        | Đề xuất thuật toán SSD mới dựa trên State Space Duality, giúp Mamba-2 nhanh hơn đáng kể và hiệu quả hơn Mamba gốc trong huấn luyện.      | Có thể thay thế Transformer trong nhiều ứng dụng nhờ hiệu quả tính toán vượt trội ở ngữ cảnh dài, thúc đẩy nghiên cứu SSM.                     |
    | 2    | 2406.07496 | AI Optimization, Textual Gradients, Computation Graph | Giới thiệu TEXTGRAD, một khung làm việc tổng quát để tối ưu hóa hệ thống AI phức hợp bằng cách lan truyền ngược "gradient văn bản" từ LLM. | Mở ra một hướng mới cho việc tự động tối ưu hóa các hệ thống AI đa thành phần, kể cả các thành phần không khả vi.                             |
    | 3    | 2406.11931 | Code LLM, MoE, Continued Pre-training, RL for Code   | Xây dựng DeepSeek-Coder-V2 (236B MoE), mô hình mã nguồn mở đạt SOTA trên nhiều benchmark code và toán, cạnh tranh với các mô hình đóng. | Thúc đẩy mạnh mẽ năng lực của các mô hình mã nguồn mở trong lĩnh vực chuyên biệt như code và toán, dân chủ hóa công nghệ AI tiên tiến.        |
    | 4    | 2406.09414 | Monocular Depth, Synthetic Data, Teacher-Student     | Đạt được ước lượng độ sâu đơn mắt chi tiết và mạnh mẽ (Depth Anything V2) bằng cách huấn luyện teacher model chỉ trên dữ liệu tổng hợp. | Cải thiện đáng kể chất lượng MDE, mở đường cho các ứng dụng 3D tốt hơn từ ảnh 2D, giảm sự phụ thuộc vào dữ liệu thực có nhãn.                 |
    | 5    | 2406.06282 | Mobile LLM Inference, Sparsity, NPU-CPU Orchestration | Phát triển PowerInfer-2, hệ thống suy luận LLM hiệu quả cao trên smartphone bằng cách thích ứng động tải công việc giữa NPU và CPU.      | Giúp triển khai các LLM lớn và mạnh mẽ trực tiếp trên thiết bị di động, mở ra nhiều ứng dụng AI cá nhân hóa và riêng tư hơn.                 |
    | 6    | 2406.04314 | Diffusion Fine-tuning, Preference Optimization, SPO  | Đề xuất Stepwise Preference Optimization (SPO) để tinh chỉnh mô hình khuếch tán hiệu quả ở từng bước, cải thiện thẩm mỹ.                 | Cung cấp phương pháp hiệu quả để align mô hình khuếch tán với sở thích người dùng, tạo ra ảnh chất lượng cao hơn với chi phí thấp.          |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2406.09414 – Huấn luyện mô hình teacher lớn (DINOv2-G) chỉ bằng dữ liệu tổng hợp chính xác, sau đó sinh nhãn giả chất lượng cao cho dữ liệu thực không nhãn quy mô lớn để huấn luyện các mô hình student. – Cách tiếp cận thông minh để bắc cầu domain gap và tận dụng ưu điểm của cả dữ liệu tổng hợp (chi tiết) và dữ liệu thực (đa dạng cảnh), đặc biệt hiệu quả cho Monocular Depth Estimation.**
    *   **2406.14491 – "Instruction Pre-Training" (IPT): Tích hợp học đa nhiệm có giám sát vào giai đoạn tiền huấn luyện bằng cách dùng "instruction synthesizer" (mô hình 7B nguồn mở) tạo cặp chỉ dẫn-phản hồi từ chính kho ngữ liệu thô. – Đưa tín hiệu giám sát sớm hơn vào quá trình học của LLM, có thể giúp mô hình học hiệu quả hơn và dễ align hơn sau này, đồng thời giảm sự phụ thuộc vào các bộ dữ liệu instruction tuning riêng biệt, chất lượng cao sau này.**
    *   **2406.10210 – Khai thác đặc trưng self-attention (lớp lup_52, t=500 trong SDXL) để mã hóa định danh instance, kết hợp mô hình ReLayout (U-Net) học sửa lỗi số lượng đối tượng tự động từ dữ liệu do chính mô hình khuếch tán tạo ra. – Phát hiện thú vị về khả năng tiềm ẩn của self-attention và giải pháp học sửa lỗi bố cục tự động là một hướng đi hay cho controllable generation, đặc biệt là kiểm soát số lượng.**
    *   **2406.04325 – Differential Sliding-Window Captioning (DiffSW): Tạo phụ đề video chi tiết bằng cách mô tả sự thay đổi giữa các khung hình chính liên tiếp sử dụng GPT-4V, sau đó tổng hợp lại. – Giải quyết vấn đề nhất quán thời gian và chi tiết trong video captioning dài bằng cách tập trung vào sự khác biệt, một cách tiếp cận khéo léo để tận dụng LLM/VLM mạnh.**
    *   **2406.10601 – Feature Editor (H) trong không gian StyleGAN Fk, được huấn luyện để chỉnh sửa tensor đặc trưng Fk dựa trên tín hiệu hướng dẫn ∆ (từ không gian W+) nhằm giữ chi tiết ảnh khi chỉnh sửa. – Giải pháp thanh lịch để cân bằng giữa tái tạo chi tiết và khả năng chỉnh sửa trong GAN inversion, đặc biệt với Fk phân giải cao.**
    *   **2406.08464 – MAGPIE: Tự tổng hợp dữ liệu căn chỉnh quy mô lớn bằng cách chỉ cung cấp template tiền truy vấn (pre-query template) cho LLM đã căn chỉnh, khiến mô hình tự sinh cả chỉ dẫn và phản hồi. – Phương pháp cực kỳ đơn giản và thông minh để tạo dữ liệu alignment, khai thác hiệu quả cấu trúc và kiến thức sẵn có của LLM nguồn.**
    *   **2406.16793 – Adam-mini: Tính toán moment bậc hai (v) trên từng khối tham số (thay vì từng tham số) dựa trên nguyên tắc phân chia theo cấu trúc Hessian (khối con dày đặc nhỏ nhất). – Giảm đáng kể bộ nhớ cho optimizer mà vẫn giữ hiệu năng, có cơ sở lý thuyết vững chắc cho việc phân khối.**
    *   **2405.21060 – State Space Duality (SSD): Chứng minh tương đương giữa SSM và ma trận bán tách được, đề xuất thuật toán SSD mới dựa trên phân rã khối để tính toán SSM hiệu quả, kết hợp cả dạng hồi quy tuyến tính và dạng đối ngẫu bậc hai. – Đóng góp lý thuyết quan trọng kết nối SSM và attention, cùng thuật toán thực tế giúp Mamba-2 nhanh hơn.**
    *   **2406.15319 – LongRAG: Sử dụng đơn vị truy xuất rất dài (toàn bộ tài liệu/nhóm tài liệu) và retriever chỉ lấy k rất nhỏ (1-8) đơn vị, chuyển gánh nặng xử lý ngữ cảnh dài cho reader (LLM). – Thay đổi cơ bản cách tiếp cận RAG truyền thống, tận dụng sức mạnh của LLM ngữ cảnh dài hiện đại.**
    *   **2406.07550 – TiTok: Mã hóa ảnh thành chuỗi latent 1D rời rạc cực kỳ nhỏ gọn (32 token) bằng ViT encoder/decoder, phá vỡ ràng buộc lưới 2D và dùng "proxy codes" từ VQGAN 2D để ổn định huấn luyện. – Hướng đi mới lạ cho image tokenization, đạt độ nén cao và hiệu quả sinh ảnh tốt.**
    *   **2406.04692 – Mixture-of-Agents (MoA): Kiến trúc phân lớp, các agent LLM ở mỗi lớp tổng hợp kết quả từ lớp trước để tinh chỉnh câu trả lời lặp đi lặp lại, dựa trên prompting. – Khai thác khả năng "hợp tác" của LLM để cải thiện chất lượng câu trả lời mà không cần fine-tuning, một dạng ensemble thông minh.**
    *   **2406.09961 – "Code tracer": Trích xuất các yếu tố cấp thấp (văn bản, bố cục, loại, màu sắc) bằng cách theo dõi quá trình thực thi mã nguồn (gốc và sinh ra) thay vì phân tích hình ảnh, để đánh giá chi tiết chart-to-code. – Phương pháp đánh giá sáng tạo và chính xác hơn cho các tác vụ sinh mã trực quan hóa.**
    *   **2406.19389 – Mô-đun nhúng tiền nghiệm nhận thức (perception prior embedding): Tích hợp thông tin từ object queries của bộ giải mã nhận thức đông lạnh vào token hình ảnh cho LLM, cải thiện liên kết mà không cần tinh chỉnh mô-đun nhận thức. – Giải pháp thông minh để kết nối hiệu quả mô-đun nhận thức đông lạnh với LLM trong MLLM hợp nhất.**
    *   **2406.06523 – Trường biến dạng lai (homography + MLP dư) và tích hợp diffusion prior (từ mô hình diffusion tinh chỉnh bằng LoRA với token cảnh) vào huấn luyện biểu diễn video canonical (NaRCan). – Kết hợp khéo léo biến đổi hình học cổ điển, học sâu và prior từ mô hình sinh để tạo ảnh canonical tự nhiên, chất lượng cao.**
    *   **2406.12849 – Xoay ngẫu nhiên ảnh equirectangular trước khi chiếu khối lập phương và chưng cất kiến thức từ mô hình độ sâu phối cảnh (thầy) sang mô hình 360 (học viên). – Kỹ thuật tăng cường dữ liệu thông minh giúp cải thiện tính nhất quán giữa các mặt của khối lập phương khi chưng cất kiến thức cho ước lượng độ sâu 360.**
    *   **2406.18082 – Huấn luyện multi-LoRA dựa trên hợp nhất trọng số (weight merging) từ các mô-đun LoRA huấn luyện trên các tập hợp chức năng riêng biệt, cho phép mô hình xử lý đa lĩnh vực trên thiết bị. – Giải pháp thực tế để tăng khả năng đa nhiệm cho agent trên thiết bị hạn chế tài nguyên mà không cần tải nhiều adapter.**
    *   **2406.09308 – TransNAR: Kiến trúc lai kết hợp Transformer (ngôn ngữ) và NAR dựa trên GNN (suy luận thuật toán) thông qua cross-attention xen kẽ, với NAR được giữ cố định. – Cách tiếp cận hiệu quả để tăng cường khả năng suy luận thuật toán OOD cho LLM bằng cách tích hợp mô-đun chuyên biệt.**
    *   **2406.16273 – TetraPose ControlNet (điều khiển động vật bốn chân) và LLM đa tác tử (Finder, Observer, Modifier) để tự động tạo/chỉnh sửa tư thế 3D động vật từ văn bản và thư viện tham khảo. – Giải pháp toàn diện và sáng tạo cho việc sinh mô hình động vật 3D có kiểm soát và nhất quán giải phẫu.**
    *   **2406.18629 – Step-DPO: Tối ưu hóa sở thích ở cấp độ từng bước suy luận thay vì toàn bộ câu trả lời, sử dụng dữ liệu "trong phân phối" do chính mô hình tạo ra làm bước đúng. – Cải tiến quan trọng cho DPO, giải quyết vấn đề giám sát chi tiết trong suy luận chuỗi dài.**
    *   **2406.05338 – MotionClone: Sử dụng thành phần chính (dominant components) trong bản đồ chú ý thời gian (temporal attention map) thưa làm biểu diễn chuyển động, trích xuất qua một bước khử nhiễu duy nhất. – Phương pháp không cần huấn luyện, thông minh để sao chép chuyển động trong video bằng cách khai thác attention map.**
    *   **2406.09760 – DICE: Sử dụng phần thưởng ngầm (implicit reward) từ mô hình DPO để tự động tạo dữ liệu ưu tiên mới lặp lại, kết hợp hiệu chỉnh độ dài và hồi tưởng kinh nghiệm. – Phương pháp tự cải thiện lặp lại thông minh cho DPO, không cần phản hồi bên ngoài.**
    *   **2406.02657 – Block Transformer: Kiến trúc Transformer phân cấp global-to-local với Block Decoder (ngữ cảnh toàn cục) và Token Decoder (cục bộ trong khối, dùng prefix tokens từ context embedding). – Thiết kế thông minh để giảm tắc nghẽn suy luận tự hồi quy, đặc biệt là chi phí KV cache.**
    *   **2406.04338 – Mô hình hóa vật liệu đàn hồi-nhớt trong MPM bằng phân tách tenxơ biến dạng song song và sử dụng SDS với mô hình khuếch tán video để học đồng thời nhiều tham số vật lý. – Kết hợp mô phỏng vật lý chi tiết với học sâu từ video để ước lượng thuộc tính vật liệu phức tạp.**
    *   **2406.11839 – MDPO: Bổ sung Conditional Preference Optimization (L_CoPO - so sánh ảnh gốc vs. ảnh giảm thông tin) và Anchored Preference Optimization (L_AncPO - đảm bảo phần thưởng dương cho y_w) vào DPO đa phương thức. – Giải quyết các vấn đề cụ thể (bỏ qua ảnh, giảm xác suất y_w) khi áp dụng DPO cho MLLM.**
    *   **2406.06282 – Pipeline cụm neuron (neuron cluster pipelining) và bộ đệm neuron dựa trên nhiệt độ (temperature-based neuron buffer) để che lấp độ trễ I/O của flash UFS trên di động. – Các kỹ thuật tối ưu hóa hệ thống thông minh, đặc thù cho phần cứng di động trong PowerInfer-2.**
    *   **2406.04333 – Chiến lược gán bit hỗn hợp dựa trên điểm nhạy cảm (MSE, kích thước lớp, sụt giảm CLIP score) và bộ kỹ thuật khởi tạo (nhúng thời gian, số nguyên cân bằng, tối ưu hệ số tỷ lệ Lloyd-Max) cho lượng tử hóa mô hình khuếch tán. – Cách tiếp cận toàn diện và hiệu quả để đạt lượng tử hóa cực thấp cho Stable Diffusion.**
    *   **2406.02539 – PARROT: Mô-đun MoE nhẹ, được định tuyến bởi tín hiệu cross-attention giữa token [CLS] thị giác và embedding văn bản, để biến đổi token thị giác cho phù hợp với ngôn ngữ đầu vào. – Giải pháp thanh lịch để cải thiện tính đa ngôn ngữ của MLLM bằng cách điều chỉnh động biểu diễn thị giác.**
    *   **2406.12275 – VoCo-LLaMA: Chèn token VisionCompression (VoCo) và sửa đổi attention mask để LLM tự nén token thị giác, học qua chưng cất chú ý từ mô hình gốc. – Ý tưởng mới lạ cho phép LLM tự thực hiện nén thông tin thị giác, tích hợp chặt chẽ vào luồng xử lý.**
    *   **2406.07496 – Toán tử "gradient văn bản" (∇LLM) và trình tối ưu hóa "gradient văn bản" (TGD) sử dụng LLM để tạo phản hồi/chỉ trích và cập nhật biến trong đồ thị tính toán. – Khái niệm hóa và triển khai một dạng "tự động vi phân" dựa trên ngôn ngữ tự nhiên, rất đột phá.**
    *   **2406.19215 – SEAKR: Ước lượng độ bất định tự nhận thức của LLM bằng định thức ma trận Gram của biểu diễn ẩn token ⟨EOS⟩ từ nhiều mẫu sinh, dùng để kích hoạt truy xuất, tái xếp hạng và chọn chiến lược suy luận trong RAG. – Khai thác tín hiệu nội tại của LLM một cách thông minh để điều khiển RAG thích ứng.**
    *   **2406.17245 – MIGU: Cập nhật gradient chọn lọc dựa trên độ lớn chuẩn hóa L1 của đầu ra lớp tuyến tính, hoạt động không cần dữ liệu cũ hay nhãn tác vụ cho học liên tục. – Phương pháp đơn giản nhưng hiệu quả, dựa trên một quan sát thú vị về hành vi của LM cho CL.**
    *   **2406.14563 – Tạo dữ liệu an toàn (`Dsafety`) bằng LLM không kiểm duyệt (tạo câu hỏi độc hại) và lấy câu trả lời từ chối từ mô hình chuyên gia (lọc bằng LLaMA-Guard) để tối ưu hóa hợp nhất mô hình nhận biết an toàn. – Quy trình thực tế để chủ động đưa yếu tố an toàn vào việc hợp nhất mô hình.**
    *   **2406.05132 – Quy trình tạo dữ liệu 3D-GRAND: Dùng LLM/MLLM trích xuất thuộc tính đối tượng, tạo đồ thị cảnh, sinh nhiều loại hướng dẫn và lọc ảo giác để tạo chú thích văn bản-3D neo đậu dày đặc quy mô lớn. – Giải pháp hiệu quả để tạo dữ liệu 3D-ngôn ngữ chi tiết và quy mô lớn, rất cần thiết cho lĩnh vực.**
    *   **2406.04271 – Buffer of Thoughts (BoT): Sử dụng meta-buffer chứa các "thought-template" trừu tượng, được truy xuất và khởi tạo thích ứng, cùng buffer-manager tự động chắt lọc và cập nhật mẫu tư duy. – Khung làm việc có cấu trúc để LLM học hỏi và tái sử dụng kinh nghiệm giải quyết vấn đề một cách hiệu quả.**
    *   **2406.16260 – Clip parallelism (giao tiếp liên GPU 3 giai đoạn) và Dual-scope attention (kết hợp ngữ cảnh cục bộ và toàn cục lấy mẫu phân tán) để tạo video dài mạch lạc từ mô hình huấn luyện trên video ngắn mà không cần huấn luyện lại. – Các cơ chế thông minh cho phép suy luận phân tán hiệu quả cho sinh video dài.**
    *   **2406.09416 – Chuẩn hóa Lớp Phụ thuộc Thời gian (TD-LN): Điều chỉnh trực tiếp tham số affine (gamma, beta) của Layer Normalization bằng hàm của thời gian (nội suy tuyến tính dựa trên sigmoid(wt+b)) để tích hợp thông tin thời gian. – Cơ chế điều kiện hóa thời gian nhẹ nhàng và hiệu quả về tham số cho mô hình khuếch tán.**
    *   **2406.04314 – Mô hình Sở thích Nhận biết Bước (SPM): Đánh giá chất lượng mẫu trung gian nhiễu, có điều kiện theo bước thời gian, để chọn cặp thắng/thua trong nhóm ứng viên từ cùng latent nhiễu gốc trong SPO. – Cho phép DPO tập trung vào các khác biệt tinh vi, cải thiện thẩm mỹ hiệu quả.**
    *   **2406.06592 – Tìm kiếm nhị phân dựa trên ước lượng Monte Carlo để xác định lỗi đầu tiên trong CoT và hàm giá trị trạng thái-lượt triển khai Q(s, r) mới ưu tiên lượt triển khai có khả năng đúng cao nhưng cho kết quả sai trong MCTS (OmegaPRM). – Các cải tiến thông minh cho MCTS để thu thập dữ liệu giám sát quá trình hiệu quả.**
    *   **2406.17636 – Tối ưu hóa hàm mục tiêu nhận thức (NCPPO) trong không gian embedding của bộ mã hóa U-Net (thay vì lỗi khuếch tán) cho các phương pháp tối ưu hóa ưu tiên mô hình khuếch tán. – Thay đổi không gian tối ưu giúp phù hợp hơn với nhận thức của con người, cải thiện chất lượng và hiệu quả.**
    *   **2406.14539 – Forward Consistency Distillation (fCD) ánh xạ điểm trên quỹ đạo ODE vào không gian nhiễu tiềm ẩn bằng cách thay đổi điều kiện biên, và Preservation Losses (Lf, Lr) tăng cường nhất quán giữa mô hình xuôi (fCDm) và ngược (CDm). – Các thành phần quan trọng giúp iCD đạt được khả năng đảo ngược và sinh ảnh chất lượng cao.**
    *   **2406.11912 – Dynamic Code Graph Generator (DCGG): Sử dụng phân tích tĩnh để tạo và cập nhật động Đồ thị Phụ thuộc Mã (CDG) nhằm cải thiện truy xuất ngữ cảnh và định hướng kiểm thử trong phát triển phần mềm đa tác nhân. – Giải pháp thực tế để xử lý ngữ cảnh ở cấp độ kho chứa cho các agent code.**
    *   **2406.05955 – Hàm kích hoạt dReLU (`max(0, xW_gate) * max(0, xW_up)`) áp dụng ReLU cho cả nhánh 'up' và 'gate' trong Gated-MLP để tăng độ thưa kích hoạt nơ-ron. – Thay đổi đơn giản nhưng hiệu quả để đạt độ thưa cao trong LLM.**
    *   **2406.19227 – ARTE: Sử dụng hiệu năng ICL một lượt của mô hình học sinh làm proxy cho sở thích của nó đối với ví dụ huấn luyện (câu hỏi, lập luận) và dùng IRT để chọn tập thẩm định hiệu quả cho thu thập sở thích. – Cách tiếp cận "dạy học đáp ứng" sáng tạo để tạo dữ liệu KD.**
    *   **2406.08552 – Window Attention with Residual Sharing (WA-RS), Attention Sharing across Timesteps (AST), Attention Sharing across CFG (ASC) để giảm tính toán attention trong DiT sau huấn luyện. – Các kỹ thuật khai thác dư thừa thông minh, không cần huấn luyện lại.**
    *   **2406.08451 – Mô-đun history resampler sử dụng learnable query embeddings để nén và tổng hợp thông tin từ các ảnh chụp màn hình lịch sử cho tác nhân điều hướng GUI đa ứng dụng. – Giải pháp hiệu quả để xử lý ngữ cảnh lịch sử trực quan dài.**
    *   **2406.04324 – Kiến trúc bộ phân biệt (discriminator) với backbone UNet cố định và các đầu phân biệt không gian/thời gian huấn luyện được, cùng lấy mẫu nhiễu lognormal rời rạc cho huấn luyện đối nghịch video một bước. – Thiết kế thông minh để ổn định huấn luyện đối nghịch và tạo video một bước chất lượng.**
    *   **2406.04277 – Khuếch tán Tổng hợp Không gian-Thời gian: Phân tách prompt phức tạp theo không gian-thời gian, tính cross-attention riêng cho từng vùng/đối tượng, sau đó tổng hợp và trộn với attention toàn cục để điều khiển sinh video. – Phương pháp training-free linh hoạt để kiểm soát nội dung video phức tạp.**
    *   **2406.18518 – APIGen: Quy trình xác minh dữ liệu gọi hàm 3 tầng (định dạng, thực thi, ngữ nghĩa - dùng LLM checker) để đảm bảo tính đúng đắn của dữ liệu tổng hợp. – Đảm bảo chất lượng cao cho dữ liệu huấn luyện agent gọi hàm, rất quan trọng cho độ tin cậy.**
    *   **2406.17763 – Thuật toán lấy mẫu khuếch tán có điều hướng kép: tích hợp đồng thời hướng dẫn từ quan sát thưa thớt và ràng buộc vật lý PDE (tính trên ước lượng dữ liệu sạch) để giải PDE. – Kết hợp hiệu quả prior sinh và ràng buộc vật lý trong quá trình suy luận.**
    *   **2406.11816 – Mục tiêu huấn luyện "Streaming EOS Prediction": Mô hình học cách quyết định khi nào cần phản hồi hoặc giữ im lặng (dự đoán token EOS) trong luồng video, giảm tính toán và ngữ cảnh dư thừa. – Giải pháp thông minh cho hội thoại video trực tuyến hiệu quả.**
    *   **2406.14562 – Whiteboard-of-Thought (WoT): MLLM tạo bước suy luận trung gian dưới dạng hình ảnh bằng cách viết mã nguồn (Python + Matplotlib/Turtle), thực thi và xử lý hình ảnh kết quả. – Khai thác khả năng tạo mã và hiểu ảnh của MLLM một cách sáng tạo cho các bài toán suy luận không gian/thị giác.**

4.  **GAPS_AND_OPPORTUNITIES**

    *   **Đánh giá Mô hình & Dữ liệu:**
        *   Nhiều benchmark mới được tạo ra, nhưng vẫn cần các phương pháp đánh giá tự động, đáng tin cậy hơn, ít phụ thuộc vào LLM-as-a-judge đắt đỏ hoặc có thiên kiến (ví dụ: vượt ra ngoài GPT-4o).
        *   Cần các benchmark động, có khả năng tự cập nhật để chống lại việc overfitting của mô hình vào các bộ dữ liệu tĩnh.
        *   Thiếu các phương pháp chuẩn hóa để đánh giá "khả năng thực sự" của các agent phức hợp trong môi trường mở, không chỉ dựa trên các tác vụ cụ thể.
        *   Nghiên cứu sâu hơn về "ảo giác" trong các mô hình đa phương thức (đặc biệt là 3D, video) và các phương pháp giảm thiểu hiệu quả.
    *   **Hiệu quả & Khả năng mở rộng:**
        *   Mặc dù có nhiều tiến bộ (Mamba-2, Block Transformer, PowerInfer-2, dReLU), việc chạy các mô hình cực lớn vẫn tốn kém. Cần các đột phá hơn nữa về kiến trúc và tối ưu hóa hệ thống, đặc biệt cho suy luận trên thiếtBF.
        *   Xử lý ngữ cảnh cực dài (hàng triệu tokens trở lên) cho cả văn bản và đa phương thức vẫn là thách thức lớn về mặt bộ nhớ, tính toán và khả năng duy trì thông tin.
        *   Học liên tục hiệu quả mà không cần nhãn tác vụ hoặc dữ liệu cũ vẫn là một bài toán mở, đặc biệt cho các mô hình nền tảng lớn.
    *   **Suy luận & Ra quyết định Phức tạp:**
        *   Khả năng suy luận đa bước, suy luận logic, toán học và commonsense của LLM/MLLM vẫn cần cải thiện đáng kể để đạt tới cấp độ con người, đặc biệt là tính mạnh mẽ (robustness) và khả năng khái quát hóa OOD.
        *   Các hệ thống agent cần khả năng lập kế hoạch, xử lý lỗi và thích ứng linh hoạt hơn trong các môi trường động và không chắc chắn.
        *   Tích hợp kiến thức có cấu trúc (symbolic knowledge) và suy luận dựa trên logic một cách chặt chẽ hơn vào các mô hình thần kinh.
    *   **Dữ liệu & Tổng hợp Dữ liệu:**
        *   Nhu cầu về dữ liệu chất lượng cao, đa dạng và có chú thích chi tiết cho các tác vụ mới (ví dụ: video dài, tương tác 3D, code phức tạp, dữ liệu alignment đa ngôn ngữ/đa phương thức) vẫn rất lớn.
        *   Phát triển các phương pháp tổng hợp dữ liệu tự động, có kiểm soát chất lượng tốt hơn, ít thiên vị hơn và có khả năng tạo ra dữ liệu "khó" thực sự cho mô hình.
        *   Nghiên cứu về "data-centric AI" cho các mô hình nền tảng: làm thế nào để thiết kế tập dữ liệu tối ưu cho một khả năng cụ thể hoặc một mô hình cụ thể.
    *   **An toàn & Căn chỉnh (Safety & Alignment):**
        *   Đảm bảo tính an toàn, công bằng và kiểm soát được cho các mô hình ngày càng mạnh mẽ và tự chủ hơn, đặc biệt là các agent có khả năng tương tác với thế giới thực hoặc tạo ra nội dung có ảnh hưởng.
        *   Phát triển các kỹ thuật alignment mạnh mẽ hơn, có thể kiểm chứng và ít tốn kém hơn so với RLHF/DPO truyền thống.
        *   Hiểu và giảm thiểu các hành vi không mong muốn (ví dụ: sycophancy, thiên vị ẩn) trong các mô hình lớn.
    *   **Đa phương thức & Tích hợp:**
        *   Cần các phương pháp hiệu quả hơn để hợp nhất thông tin từ nhiều phương thức (ví dụ: video, audio, text, 3D, cảm biến) một cách sâu sắc và có ý nghĩa.
        *   Khả năng "grounding" ngôn ngữ vào thế giới vật lý (thông qua robot, mô phỏng) vẫn còn nhiều thách thức.
        *   Phát triển các mô hình thực sự "phổ quát" có thể xử lý và suy luận trên nhiều loại dữ liệu và tác vụ khác nhau một cách liền mạch.

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Meta-Learned Data Curator for Specialized LLMs/MLLMs (Khả thi cao)**
    *   **Motivation:** Việc tạo dữ liệu chất lượng cao cho các miền/tác vụ chuyên biệt rất tốn kém. Các phương pháp tổng hợp hiện tại (ví dụ: self-instruct, MAGPIE) thường dựa trên các prompt cố định hoặc mô hình giáo viên lớn.
    *   **Key novelty:** Một hệ thống meta-learning học cách tạo ra các chiến lược/prompt tối ưu để tổng hợp dữ liệu huấn luyện cho một mô hình học sinh (SLM/MLLM nhỏ) cụ thể và một tác vụ mục tiêu cụ thể, dựa trên phản hồi từ hiệu năng của mô hình học sinh đó.
    *   **Approach:**
        1.  Xây dựng một không gian các tham số/chiến lược tạo dữ liệu (ví dụ: loại prompt, nguồn dữ liệu seed, kỹ thuật tăng cường, tỷ lệ trộn).
        2.  Sử dụng một bộ điều khiển meta-learning (ví dụ: RL agent hoặc thuật toán tiến hóa) để lấy mẫu các chiến lược tạo dữ liệu.
        3.  Với mỗi chiến lược, tổng hợp một lô dữ liệu nhỏ.
        4.  Huấn luyện nhanh mô hình học sinh trên lô dữ liệu này và đánh giá hiệu năng trên một tập validation nhỏ của tác vụ mục tiêu.
        5.  Sử dụng hiệu năng này làm phần thưởng để cập nhật bộ điều khiển meta-learning.
        6.  Lặp lại để tìm ra chiến lược tạo dữ liệu tối ưu.
    *   **Dataset + Metrics:** Các tác vụ NLP/Multimodal đa dạng (ví dụ: từ SuperGLUE, MMBench), hiệu năng của mô hình học sinh trên tập test của tác vụ đó, chi phí tạo dữ liệu.
    *   **Risk/Feasibility:** Cao. Rủi ro chính là không gian tìm kiếm chiến lược lớn và chi phí tính toán cho nhiều vòng huấn luyện/đánh giá. Tuy nhiên, có thể bắt đầu với không gian chiến lược nhỏ và mô hình học sinh rất nhỏ.

    ✨ **Idea 2: Composable World Models from Multimodal Streams (Liên lĩnh vực)**
    *   **Motivation:** Các MLLM hiện tại (ví dụ: MMWorld, LIVE) đang hướng tới hiểu video dài, nhưng việc xây dựng một "world model" thực sự có khả năng dự đoán, suy luận phản thực tế và hiểu nhân quả từ các luồng đa phương thức (video, audio, text, cảm biến IoT) vẫn là thách thức.
    *   **Key novelty:** Phát triển một kiến trúc cho phép học các "mô-đun thế giới" (world modules) chuyên biệt cho các khía cạnh khác nhau của môi trường (vật lý, tương tác xã hội, mục tiêu của agent) từ các luồng dữ liệu không đồng bộ và có thể kết hợp (compose) các mô-đun này một cách linh hoạt để suy luận về các tình huống mới.
    *   **Approach:**
        1.  Sử dụng kiến trúc nền tảng dựa trên Transformer/SSM có khả năng xử lý các chuỗi đa phương thức dài, không đồng bộ.
        2.  Đề xuất các mục tiêu tự giám sát (self-supervised objectives) để học các biểu diễn tách rời cho các khía cạnh khác nhau của thế giới (ví dụ: dự đoán tương lai có điều kiện trên một số yếu tố, suy luận nhân quả từ can thiệp mô phỏng).
        3.  Thiết kế một cơ chế "composition" (ví dụ: attention qua lại giữa các mô-đun, hoặc một meta-controller) để kết hợp tri thức từ các mô-đun khác nhau khi giải quyết một tác vụ cụ thể.
        4.  Tận dụng dữ liệu từ các môi trường mô phỏng phong phú (ví dụ: Isaac Sim, Habitat) và video thực tế (Ego4D, YouTube).
    *   **Dataset + Metrics:** Các bộ dữ liệu video dài có chú thích về sự kiện, nhân quả, mục tiêu (ví dụ: Ego4D, MMWorld mở rộng). Metrics đánh giá khả năng dự đoán, trả lời câu hỏi phản thực tế, giải thích hành vi.
    *   **Risk/Feasibility:** Trung bình đến Cao. Rủi ro về độ phức tạp của kiến trúc và mục tiêu học. Tính khả thi phụ thuộc vào sự tiến bộ của các mô hình xử lý chuỗi dài và các phương pháp học biểu diễn tách rời.

    ✨ **Idea 3: AI Scientist with TEXTGRAD-powered Hypothesis Generation and Experimentation (Moon-shot)**
    *   **Motivation:** Quá trình khám phá khoa học rất chậm chạp. Một AI có khả năng tự đề xuất giả thuyết, thiết kế thí nghiệm, diễn giải kết quả và tinh chỉnh giả thuyết sẽ cách mạng hóa khoa học.
    *   **Key novelty:** Mở rộng TEXTGRAD (2406.07496) thành một vòng lặp tự động hoàn chỉnh cho khám phá khoa học. AI không chỉ tối ưu các thành phần hiện có mà còn chủ động đề xuất các "biến" (giả thuyết, thiết kế thí nghiệm) và "hàm" (mô phỏng, phân tích dữ liệu) mới trong đồ thị tính toán của nó.
    *   **Approach:**
        1.  Bắt đầu với một LLM/Agent được trang bị TEXTGRAD và kiến thức nền về một lĩnh vực khoa học cụ thể (ví dụ: vật liệu, dược phẩm).
        2.  Cung cấp cho AI một mục tiêu nghiên cứu cấp cao (ví dụ: "tìm một loại thuốc mới cho bệnh X" hoặc "thiết kế vật liệu Y với đặc tính Z").
        3.  AI sử dụng khả năng suy luận của mình để tạo ra các giả thuyết ban đầu (dưới dạng các biến có thể tối ưu trong TEXTGRAD).
        4.  AI thiết kế các "thí nghiệm" (có thể là mô phỏng số hoặc yêu cầu thực hiện thí nghiệm thực tế nếu có giao diện robot) để kiểm tra giả thuyết. Các thiết kế này cũng là các biến có thể tối ưu.
        5.  Kết quả thí nghiệm được sử dụng để tính "loss" (mức độ giả thuyết được xác nhận/bác bỏ).
        6.  TEXTGRAD được sử dụng để lan truyền "gradient văn bản" nhằm tinh chỉnh giả thuyết, thiết kế thí nghiệm, và thậm chí cả các mô hình mô phỏng/phân tích mà AI sử dụng.
        7.  AI có thể đề xuất thêm các biến/hàm mới vào đồ thị (ví dụ: khám phá một cơ chế phản ứng mới).
    *   **Dataset + Metrics:** Các bộ dữ liệu khoa học hiện có (ví dụ: PubChem, Materials Project), kết quả từ các mô phỏng (ví dụ: DFT cho hóa học). Metrics: số lượng giả thuyết mới, hợp lý được tạo ra; hiệu quả của các thí nghiệm được thiết kế; khả năng tái tạo các khám phá khoa học đã biết; tốc độ đạt được mục tiêu nghiên cứu.
    *   **Risk/Feasibility:** Rất cao. Rủi ro về tính đúng đắn của giả thuyết, chi phí mô phỏng/thí nghiệm, khả năng diễn giải kết quả phức tạp, và vấn đề an toàn/đạo đức. Đây là một ý tưởng cực kỳ táo bạo, đòi hỏi sự đột phá ở nhiều lĩnh vực AI.

6.  **READING_LIST**

    *   **2405.21060 – Mamba-2 · Đột phá về lý thuyết (SSD) và hiệu năng cho SSM, thách thức vị thế của Transformer trong xử lý chuỗi dài.**
    *   **2406.07496 – TEXTGRAD · Giới thiệu một khung làm việc hoàn toàn mới lạ và mạnh mẽ để tối ưu hóa các hệ thống AI phức hợp bằng "gradient văn bản", có tiềm năng ứng dụng rộng rãi.**
    *   **2406.17557 – FineWeb · Một nỗ lực quy mô lớn và minh bạch trong việc xây dựng bộ dữ liệu tiền huấn luyện LLM chất lượng cao, cung cấp tài nguyên và hiểu biết giá trị cho cộng đồng mở.**
    *   **2406.11931 – DeepSeek-Coder-V2 · Minh chứng cho khả năng của kiến trúc MoE mã nguồn mở trong việc đạt hiệu năng SOTA cho các tác vụ chuyên biệt như code và toán, cạnh tranh với các mô hình đóng.**
    *   **2406.06282 – PowerInfer-2 · Giải quyết xuất sắc thách thức chạy LLM lớn trên thiết bị di động thông qua các kỹ thuật tối ưu hóa hệ thống thông minh, đặc thù cho phần cứng di động.**
    *   **2406.04271 – Buffer of Thoughts (BoT) · Đề xuất một cách tiếp cận có cấu trúc và học hỏi được để LLM tái sử dụng kinh nghiệm giải quyết vấn đề, cải thiện khả năng suy luận phức tạp.**
    *   **2406.04314 – Stepwise Preference Optimization (SPO) · Một phương pháp tinh chỉnh mô hình khuếch tán dựa trên sở thích một cách thông minh, tập trung vào cải thiện chi tiết ở từng bước khử nhiễu.**

7.  **META_REFLECTION**

    *   Tập hợp các bài báo tháng 06/2024 cho thấy sự phát triển mạnh mẽ và đa dạng trên nhiều mặt trận của AI. Một xu hướng nổi bật là việc **dân chủ hóa và chuẩn hóa các nguồn lực AI**, thể hiện qua việc xây dựng hàng loạt bộ dữ liệu và benchmark quy mô lớn, chất lượng cao, và thường là mã nguồn mở (ví dụ: FineWeb, DCLM, 3D-GRAND, MMDU, CRAG, WildBench, CharXiv, SEACrowd, XLand-100B, MMLU-Pro, BigCodeBench, OmniCorpus). Điều này cho thấy cộng đồng đang ngày càng chú trọng đến việc đánh giá mô hình một cách nghiêm ngặt và tạo nền tảng vững chắc cho nghiên cứu.
    *   **Hiệu quả và khả năng mở rộng** tiếp tục là mối quan tâm hàng đầu, với các nghiên cứu về kiến trúc mới (Mamba-2, Block Transformer, DiMR), kỹ thuật thưa hóa (dReLU), tối ưu hóa suy luận (PowerInfer-2, DiTFastAttn), và nén mô hình (BitsFusion, VoCo-LLaMA).
    *   **Khả năng suy luận và tự chủ của các agent** đang được đẩy mạnh, với các kiến trúc đa tác tử, khả năng sử dụng công cụ phức tạp, và các phương pháp tăng cường suy luận (BoT, TransNAR, OmegaPRM, HUSKY, AGILE CODER).
    *   Trong lĩnh vực **đa phương thức**, có sự tập trung vào việc xử lý ngữ cảnh dài (đặc biệt là video), cải thiện sự liên kết giữa các phương thức, và phát triển các mô hình có khả năng kiểm soát tốt hơn (CountGen, VideoTetris, YOUDREAM).
    *   Các **phương pháp căn chỉnh và tối ưu hóa sở thích** (DPO và các biến thể như Step-DPO, MDPO, DICE, SPO, NCPPO) đang trở nên tinh vi hơn, giải quyết các vấn đề cụ thể của từng loại mô hình và dữ liệu.
    *   Một điểm đáng chú ý nữa là sự xuất hiện của các **khung làm việc và phương pháp luận mới có tính tổng quát cao**, như TEXTGRAD cho phép tối ưu hóa hệ thống AI bằng phản hồi văn bản, hay các nỗ lực hệ thống hóa kiến thức (khảo sát về prompt engineering, phân tích LLM-as-a-judge).
    *   Nhìn chung, lĩnh vực AI đang tiến tới việc xây dựng các hệ thống ngày càng mạnh mẽ, đa năng, hiệu quả, dễ tiếp cận và đáng tin cậy hơn, đồng thời cũng đối mặt với những thách thức lớn về dữ liệu, đánh giá, và khả năng suy luận thực sự sâu sắc.
