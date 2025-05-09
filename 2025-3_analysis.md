1.  **TOPIC_TREE**
    *   **I. Natural Language Processing (NLP)**
        *   A. Language Model Architectures & Training
            *   1. Efficient Architectures & Alternatives
                *   a. Normalization Alternatives
                    *   `2503.10622` | Giới thiệu Dynamic Tanh (DyT) như một giải pháp thay thế hiệu quả cho LayerNorm/RMSNorm trong các kiến trúc DL, bao gồm cả LLM.
                *   b. Recurrent Architectures
                    *   `2503.14456` | Phát triển kiến trúc RWKV-7 "Goose" với quy tắc delta tổng quát và cổng trạng thái vector, tối ưu hóa cho hiệu suất và khả năng biểu đạt.
                *   c. Efficient Transformers
                    *   `2503.16660` | Đề xuất Select-by-Reconstruction, một phương pháp giảm token hiệu quả cho Vision Transformer dựa trên khả năng tái tạo.
            *   2. Multilingual Models
                *   `2503.05500` | Xây dựng EuroBERT, một họ mô hình encoder đa ngôn ngữ mạnh mẽ với kiến trúc cập nhật và quy trình huấn luyện hai giai đoạn.
                *   `2503.00865` | Giới thiệu Babel, một họ LLM đa ngôn ngữ mã nguồn mở với kỹ thuật mở rộng lớp (layer extension) để cải thiện hiệu năng.
            *   3. Vocabulary & Tokenization
                *   `2503.19693` | Đề xuất AdaptiVocab, một khung thích ứng từ vựng giúp LLM giảm số lượng token và tăng hiệu quả trong các miền chuyên biệt.
            *   4. Pre-training Data Strategies
                *   `2503.00808` | Giới thiệu PRESELECT, một phương pháp chọn lọc dữ liệu tiền huấn luyện dựa trên "độ mạnh dự đoán" để cải thiện hiệu suất mô hình.
            *   5. Embedding Models
                *   `2503.07891` | Phát triển Gemini Embedding, mô hình nhúng văn bản đa ngôn ngữ và mã nguồn mạnh mẽ kế thừa từ LLM Gemini.
        *   B. Reasoning in LLMs
            *   1. Enhancing Reasoning Capabilities
                *   a. Reinforcement Learning for Reasoning
                    *   `2503.14476` | Đề xuất thuật toán DAPO (Decoupled Clip & Dynamic sAmpling Policy Optimization) cho tối ưu hóa chính sách RL trong các tác vụ suy luận chuỗi dài (Long CoT).
                    *   `2503.07572` | Giới thiệu Meta Reinforcement Fine-Tuning (MRT) để tối ưu hóa việc sử dụng tài nguyên tính toán của LLM trong quá trình suy luận thông qua phần thưởng tiến bộ.
                    *   `2503.16219` | Khảo sát và tối ưu hóa việc huấn luyện RL (GRPO) cho LLM nhỏ (1.5B) trong điều kiện tài nguyên cực kỳ hạn chế cho các bài toán suy luận toán học.
                    *   `2503.01307` | Phân tích các hành vi nhận thức (verification, backtracking, subgoal setting, backward chaining) trong LLM và mối liên hệ của chúng với khả năng tự cải thiện qua RL.
                *   b. Prompting & Decoding Strategies for Reasoning
                    *   `2503.04625` | Phát triển START, một LRM mã nguồn mở kết hợp Long CoT và trình thông dịch mã, được huấn luyện bằng framework tự huấn luyện Hint-RFT.
                    *   `2503.02003` | Đề xuất Highlighted Chain-of-Thought (HoT) prompting để tạo liên kết tường minh giữa câu trả lời và các sự kiện trong câu hỏi.
                    *   `2502.18600` | Giới thiệu Chain of Draft (CoD), một phương pháp prompting tạo ra các bước suy luận trung gian tối giản, hiệu quả về token.
                    *   `2503.05179` | Đề xuất Sketch-of-Thought (SoT), một framework tạo lời nhắc dựa trên khoa học nhận thức để tạo các bước suy luận ngắn gọn.
                    *   `2503.13288` | Giới thiệu ϕ-Decoding, một khung giải mã foresight sampling kết hợp đánh giá bước và cắt tỉa động để tối ưu suy luận dài.
            *   2. Interpretability & Analysis of Reasoning
                *   `2503.03601` | Sử dụng Sparse Autoencoders (SAEs) để trích xuất và diễn giải các đặc trưng nhằm phát hiện văn bản do AI tạo ra (ATD).
                *   `2503.18878` | Đề xuất quy trình nhận diện đặc trưng suy luận trong LLM bằng SAEs và chỉ số ReasonScore, cùng với kỹ thuật feature steering để kiểm chứng.
                *   `2503.15299` | Định nghĩa và đo lường "hidden knowledge" (kiến thức ẩn) trong LLM, cho thấy sự khác biệt giữa kiến thức nội tại và khả năng sinh văn bản.
        *   C. Model Alignment & Preference Optimization
            *   `2503.22230` | Khám phá các chiến lược dữ liệu và khung thưởng lai ghép (RTV + GenRM) trong RLHF để chống reward hacking và cải thiện hiệu suất LLM.
            *   `2503.17126` | Đề xuất DDPO và DORPO, các phương pháp tối ưu hóa sở thích có nhận biết độ đa dạng (deviation-aware) để cải thiện chất lượng và sự đa dạng của LLM.
            *   `2503.07067` | Giới thiệu DISTILLM-2, một phương pháp chưng cất kiến thức cho LLM sử dụng loss tương phản bất đối xứng (CALD) để cải thiện hiệu quả.
        *   D. Clinical NLP
            *   `2502.21263` | Xây dựng bộ dữ liệu RuCCoD và benchmark cho việc gán mã ICD-10 tự động từ hồ sơ sức khỏe điện tử tiếng Nga, đồng thời chứng minh hiệu quả của mã ICD do AI tạo.
        *   E. Surveys on LLMs & Agents
            *   `2503.16416` | Khảo sát toàn diện về các phương pháp và benchmark đánh giá agent dựa trên LLM, bao gồm năng lực nền tảng và ứng dụng chuyên biệt.
            *   `2503.21460` | Khảo sát về hệ sinh thái Multi-Agent Systems (MAS) dựa trên LLM, bao gồm xây dựng, hợp tác, phát triển, đánh giá và ứng dụng.
            *   `2503.16419` | Khảo sát các kỹ thuật suy luận hiệu quả cho Large Reasoning Models (LRMs), tập trung vào việc giảm thiểu "over-thinking" và chi phí tính toán.
            *   `2503.17407` | Khảo sát toàn diện về mô hình hóa ngôn ngữ ngữ cảnh dài (Long Context Language Modeling), bao gồm dữ liệu, kiến trúc, workflow, hạ tầng và đánh giá.
            *   `2503.11069` | So sánh và phân tích sự khác biệt, điểm hội tụ giữa API agents và GUI agents, đề xuất mô hình lai.
    *   **II. Computer Vision (CV) & Multimodal AI**
        *   A. Image/Video Generation & Synthesis
            *   1. Foundational Models & Architectures
                *   a. Diffusion Transformers (DiT)
                    *   `2503.23307` | Giới thiệu MoCha, một mô hình Diffusion Transformer (DiT) sinh video nhân vật nói chuyện từ văn bản và âm thanh.
                    *   `2503.20314` | Xây dựng Wan-2.1, một họ mô hình Diffusion-Transformer mạnh mẽ cho sinh video chất lượng cao (480-720p) kèm chữ Trung-Anh.
                    *   `2503.19757` | Đề xuất Dita, một Diffusion Transformer thực hiện nội suy và khử nhiễu chuỗi hành động liên tục trực tiếp trong kiến trúc Transformer cho VLA.
                *   b. Autoregressive Models
                    *   `2503.19325` | Giới thiệu FAR, một khung Frame-wise AutoRegressive kết hợp flow-matching liên tục và attention nhân quả giữa các khung hình cho sinh video.
            *   2. Controllable Generation & Editing
                *   a. Camera Control & Re-rendering
                    *   `2503.11647` | Đề xuất ReCamMaster, một khung video-to-video cho phép sinh lại cảnh động với quỹ đạo camera mới, sử dụng frame-dimension conditioning.
                    *   `2503.09151` | Giới thiệu Reangle-A-Video, một khung tạo video 4D từ một video đơn bằng cách coi đây là bài toán video-to-video translation, điều khiển camera 6DoF.
                *   b. Attribute & Instance Control
                    *   `2503.10639` | Đề xuất Generation Chain-of-Thought (GoT) và Semantic-Spatial Guidance Module (SSGM) để điều khiển sinh/chỉnh sửa ảnh dựa trên suy luận không gian-ngữ nghĩa.
                    *   `2503.12885` | Giới thiệu DreamRenderer, một bộ điều khiển plug-and-play không cần huấn luyện để kiểm soát thuộc tính của từng vùng/instance trong ảnh sinh ra bởi FLUX/3DIS.
                *   c. Personalized Generation
                    *   `2503.12590` | Đề xuất Personalize Anything, một khung cá nhân hóa ảnh zero-shot bằng cách thay thế token trong Diffusion Transformer (DiT) một cách thích ứng theo timestep.
                    *   `2503.16418` | Giới thiệu InfUnet, một khung cá nhân hóa ảnh dựa trên DiT và Rectified Flow, sử dụng nhánh DiT rút gọn để tiêm đặc trưng danh tính qua residual connections.
                *   d. Unified Video Creation & Editing
                    *   `2503.07598` | Đề xuất VACE, một khung hợp nhất sử dụng Video Condition Unit (VCU) và Context Adapter Tuning để thực hiện nhiều tác vụ tạo và chỉnh sửa video trên một DiT duy nhất.
            *   3. Efficient Generation & Acceleration
                *   a. Diffusion Model Distillation & Acceleration
                    *   `2503.13358` | Đề xuất RSD (Residual Shifting Distillation), một phương pháp chưng cất ResShift thành mô hình super-resolution một bước hiệu quả.
                    *   `2503.07677` | Giới thiệu PLADIS, một phương pháp thay thế cross-attention trong U-Net/Transformer bằng nội suy giữa attention dày đặc và thưa (α-Entmax) tại thời điểm suy luận.
                    *   `2503.09566` | Đề xuất TPDiff, một mô hình "kim tự tháp thời gian" tăng dần frame-rate trong quá trình khuếch tán video để giảm chi phí huấn luyện và tăng tốc suy luận.
                    *   `2503.16397` | Giới thiệu Scale-Wise Distillation (SWD) và Patch Distribution Matching (PDM) để chưng cất diffusion transformer bằng cách tăng dần độ phân giải ở mỗi bước suy luận.
                    *   `2503.09641` | Đề xuất SANA-Sprint, kết hợp chuyển đổi Flow→TrigFlow không huấn luyện và hybrid distillation (sCM + LADD) để tăng tốc mô hình Flow-Matching.
                    *   `2503.09662` | Giới thiệu CoRe² (Collect-Reflect-Refine), một khuôn khổ suy luận ba pha tương thích với cả diffusion và autoregressive models để tăng chất lượng và giảm độ trễ.
                *   b. Test-Time Scaling
                    *   `2503.18942` | Giới thiệu Video-T1, khung Test-Time Scaling đầu tiên cho sinh video, sử dụng thuật toán Tree-of-Frames (ToF) và hệ đa-verifier.
            *   4. Specific Modalities & Tasks
                *   a. Talking Head & Character Animation
                    *   `2503.05978` | Đề xuất MagicInfinite, một framework DiT với 3D Full-Attention và khử nhiễu cửa sổ trượt để tạo video chân dung nói chuyện dài vô hạn.
                *   b. Text-to-Image with Text Rendering
                    *   `2503.07703` | Giới thiệu Seedream 2.0, một mô hình T2I song ngữ tích hợp LLM làm encoder và Glyph-Aligned ByT5 để render văn bản chính xác.
                *   c. Motion Generation
                    *   `2503.06955` | Đề xuất Motion Anything, một khung sinh chuyển động "any-to-motion" từ điều kiện đa phương thức (text, nhạc) sử dụng mô hình hóa che mặt nạ tự hồi quy dựa trên attention.
                *   d. Interactive Video & Game Engines
                    *   `2503.17359` | Đề xuất Interactive Generative Video (IGV) làm lõi cho Generative Game Engine (GGE), bổ sung các thuộc tính điều khiển, bộ nhớ, vật lý và suy luận.
        *   B. Multimodal Language Models (MLLMs) & Understanding
            *   1. Architectures & Pre-training
                *   a. Unified Omni-modal Models
                    *   `2503.20215` | Giới thiệu Qwen-2.5-Omni, một mô hình omni-modal xử lý văn bản, hình ảnh, video, âm thanh và hỗ trợ phát đồng thời văn bản & giọng nói.
                    *   `2503.01743` | Đề xuất Phi-4-Multimodal, một mô hình đa phương thức hợp nhất (văn bản, thị giác, lời nói/âm thanh) dựa trên kỹ thuật Mixture of LoRAs (MoL).
                *   b. Efficient MLLMs
                    *   `2503.11576` | Đề xuất SmolDocling, một mô hình vision-language 256M tham số chuyển đổi PDF sang markup DocTags, tập trung vào hiệu quả.
                    *   `2503.04130` | Giới thiệu STORM, một kiến trúc tích hợp Mamba temporal projector vào Video-LLMs để xử lý video dài hiệu quả với giảm token.
                    *   `2503.19786` | Phát triển Gemma-3, một họ LLM đa phương thức (1-27B) với kiến trúc Local/Global-Attention và Pan-&-Scan windowing cho ngữ cảnh dài và xử lý ảnh hiệu quả.
                *   c. High-Resolution Pre-training
                    *   `2503.19903` | Giới thiệu PS3 (Pre-training with Scale-Selective Scaling), một khung huấn luyện CLIP-style cho độ phân giải lên đến 4K bằng cách xử lý vùng cục bộ.
            *   2. Reasoning & Alignment in MLLMs
                *   a. Multimodal Reasoning Enhancement
                    *   `2503.07536` | Đề xuất LMM-R1, một khung huấn luyện hai giai đoạn sử dụng RL dựa trên luật với hàm thưởng có thể kiểm chứng để tăng cường suy luận đa phương thức.
                    *   `2503.21776` | Giới thiệu Video-R1, một khung huấn luyện R1 cho suy luận video, và thuật toán T-GRPO so sánh lời giải với khung hình xáo trộn/đúng thứ tự.
                    *   `2503.07365` | Phát triển MM-Eureka, một VLM được huấn luyện bằng rule-based RL (GRPO) với chiến lược lọc online zero-advantage cho suy luận đa phương thức K-12.
                    *   `2503.05132` | Tái hiện "khoảnh khắc aha" và tăng độ dài phản hồi của R1 trên mô hình đa phương thức non-SFT (VisualThinker-R1-Zero) cho suy luận không gian.
                    *   `2503.16905` | Đề xuất MAPS, một khung đa tác nhân dựa trên Big Seven Personality và Socratic questioning để giải quyết bài toán khoa học đa phương thức.
                    *   `2503.15558` | Phát triển Cosmos-Reason1, một MLLM lai Mamba-MLP-Transformer cho AI vật lý, với quy trình huấn luyện 4 pha và reward MCQ rule-based.
                    *   `2503.07002` | Giới thiệu DiagNote, một MLLM với mô-đun Deliberate (CoT) và Gaze (grounding) tương tác để theo dõi saliency trong hội thoại đa lượt đa mô thức.
                *   b. Multimodal Chain-of-Thought
                    *   `2503.12605` | Khảo sát toàn diện về Multimodal Chain-of-Thought (MCoT), bao gồm phương pháp, ứng dụng và benchmark.
                *   c. Reward Modeling & Preference Learning for MLLMs
                    *   `2503.05236` | Đề xuất UNIFIEDREWARD, một mô hình phần thưởng hợp nhất cho đánh giá hiểu biết và tạo sinh đa phương thức (ảnh, video), hỗ trợ cả pairwise ranking và pointwise scoring.
                    *   `2503.01785` | Đề xuất Visual-RFT, một phương pháp tinh chỉnh LVLMs sử dụng học tăng cường với phần thưởng có thể xác minh (verifiable rewards) cho các tác vụ thị giác.
                    *   `2503.10291` | Giới thiệu VisualPRM, một mô hình Process Reward đa phương thức dự đoán tính đúng đắn từng bước của chuỗi suy luận, huấn luyện bằng Monte-Carlo Process Supervision.
                    *   `2503.03746` | Đề xuất phương pháp đánh giá LLM-làm-giám-khảo theo từng bước và tối ưu hóa sở thích theo từng bước (DPO) cho suy luận toán học.
            *   3. Specific Applications & Tasks
                *   a. Document Intelligence
                    *   `2503.11576` | (Đã liệt kê ở II.B.1.b) SmolDocling cho chuyển đổi tài liệu PDF.
                *   b. GUI Understanding & Control
                    *   `2503.21620` | Giới thiệu UI-R1, một khung R1 áp dụng RL dựa trên luật (GRPO) với rule-based action reward để tăng năng lực suy luận hành động GUI cho MLLM.
                *   c. Egocentric AI
                    *   `2503.03803` | Đề xuất EgoButler, một hệ thống trợ lý AI egocentric tích hợp mô-đun hiểu clip EgoGPT và mô-đun trả lời câu hỏi ngữ cảnh dài EgoRAG.
                *   d. Text-to-Speech (TTS) Integration
                    *   `2503.04724` | (Đã liệt kê ở II.B.3.d) LLMVoX cho streaming TTS.
        *   C. 3D Content Creation & Understanding
            *   1. 3D Reconstruction & Novel View Synthesis
                *   `2503.01774` | Đề xuất DIFIX, một mô hình khuếch tán ảnh đơn bước được tinh chỉnh để tăng cường và loại bỏ tạo tác trong các khung nhìn mới được kết xuất từ biểu diễn 3D (NeRF, 3DGS).
                *   `2503.10625` | Giới thiệu LHM, một mô hình feed-forward tái dựng avatar người animatable dưới dạng 3D Gaussian Splatting từ một ảnh đơn, sử dụng MBHT và HFPE.
            *   2. 3D Mesh Generation
                *   `2503.15265` | Đề xuất DeepMesh, một hệ thống tạo lưới tự động đa lớp thông qua RL và DPO, sử dụng token-hierarchy 3 cấp và Hourglass-Transformer.
            *   3. 4D Scene Representation & Understanding
                *   `2503.10437` | Đề xuất 4D LangSplat, mô hình học trường ngôn ngữ 4D trên Gaussian Splatting, hỗ trợ truy vấn mở theo không gian-thời gian trong cảnh động.
            *   4. Diffusion Models for 3D
                *   `2503.16302` | Giới thiệu FlashVDM, tăng tốc Vecset Diffusion Models (VDM) cho tạo hình 3D thông qua Progressive Flow Distillation và các kỹ thuật giải mã VAE hiệu quả.
        *   D. Datasets & Benchmarks for Vision & Multimodal AI
            *   `2503.07920` | Giới thiệu SEA-VL, một bộ dữ liệu thị giác-ngôn ngữ quy mô lớn, đa văn hoá cho 11 quốc gia Đông Nam Á.
            *   `2503.06053` | Xây dựng DropletVideo-10M, một bộ dữ liệu video quy mô lớn với caption chi tiết, tập trung vào "Integral Spatio-Temporal Consistency".
            *   `2503.14378` | Xây dựng IPV-BENCH, một benchmark kép đánh giá khả năng tạo và hiểu "Impossible Videos" (video phi thực tế) của các mô hình.
            *   `2503.21380` | Giới thiệu OlymMATH, một bộ dữ liệu benchmark gồm 200 bài toán toán học cấp Olympic song ngữ (Anh/Trung) để đánh giá suy luận của LLM.
            *   `2503.14478` | Xây dựng Creation-MMBench, một benchmark gồm 765 mẫu đa phương thức để đánh giá khả năng sáng tạo của MLLM trong các tác vụ viết lách và hiểu biết.
            *   `2503.21755` | Giới thiệu VBench-2.0, một bộ benchmark đánh giá "intrinsic faithfulness" của mô hình sinh video qua 5 trục và 18 năng lực vi mô.
            *   `2503.19990` | Giới thiệu LEGO-Puzzles, một benchmark VQA dựa trên hướng dẫn lắp ráp LEGO để kiểm tra suy luận không gian nhiều bước của MLLM.
            *   `2503.07860` | Xây dựng VidDiffBench, một bộ dữ liệu gồm 549 cặp video để đánh giá khả năng mô tả và quy kết khác biệt vi mô giữa hai video cùng hành động.
            *   `2503.05085` | Giới thiệu S2S-Arena, một benchmark kiểu "arena" để đánh giá khả năng tuân thủ chỉ dẫn và xử lý thông tin cận ngôn ngữ của các mô hình speech-to-speech.
        *   E. Contrastive Learning & Representation Learning
            *   `2503.15485` | Đề xuất TULIP, một framework thống nhất ba nhánh tương phản (Hình–Văn, Hình–Hình, Văn–Văn) trên backbone SigLIP, sử dụng GeCo để sinh views tương phản.
    *   **III. Robotics & Embodied AI**
        *   A. Vision-Language-Action (VLA) Models
            *   `2503.19757` | (Đã liệt kê ở II.A.1.a) Dita, Diffusion Transformer cho VLA.
            *   `2503.16365` | Đề xuất ACTVLP, một quy trình Visual-Language Post-Training ba giai đoạn cho VLA, và JARVIS-VLA, một kiến trúc ViT+LLM cho Minecraft.
        *   B. Humanoid Robotics & Hierarchical Agents
            *   `2503.12533` | Đề xuất BEING-0, một khung tác nhân phân cấp (FM trên đám mây – Connector on-board – Thư viện Kỹ năng) cho robot hình người, sử dụng VLM làm cầu nối.
        *   C. Task Planning & Control
            *   `2503.10480` | Đề xuất Dual Preference Optimization (D²PO), một khuôn khổ tối ưu đồng thời dự đoán trạng thái (world-modeling) và lựa chọn hành động cho LVLM trong các tác vụ lập kế hoạch.
            *   `2503.10613` | Đề xuất CoSTA*, một khung biên tập ảnh đa lượt dựa trên LLM Subtask Tree và A* search, có ràng buộc chi phí-chất lượng và phản hồi từ VLM.
        *   D. Data Collection for Robot Learning
            *   `2503.11646` | Đề xuất Adversarial Data Collection (ADC), một quy trình thu thập dữ liệu tương tác người-trong-vòng-lặp hai chiều thời gian thực cho học máy bắt chước trên robot.
    *   **IV. Methodology & General Machine Learning**
        *   A. Model Evaluation & Analysis
            *   1. Model Lineage & Atlas
                *   `2503.10633` | Trình bày HF-MODEL-ATLAS, một công cụ trực quan hoá và phân tích mối quan hệ (lineage) của các mô hình trên Hugging Face Hub dưới dạng DAG.
            *   2. Multi-Agent System Evaluation
                *   `2503.13657` | Định nghĩa MAST, một bộ phân loại 14 "chứng bệnh" của hệ đa tác nhân LLM và phát triển pipeline LLM-as-a-Judge để chẩn đoán.
                *   `2503.16874` | Đề xuất MARS, một khung APO đa tác nhân (7 agent) kết hợp Planner và Teacher-Critic-Student để tối ưu prompt tự động.
            *   3. Search-Augmented Reasoning Evaluation
                *   `2503.20201` | Đề xuất Open Deep Search (ODS), một khung plug-and-play cho phép LLM thực hiện tìm kiếm web thời gian thực và suy luận, cùng với agent CodeAct.
        *   B. Audio & Speech Processing
            *   1. Music Generation
                *   `2503.08638` | Đề xuất YuE, một mô hình sinh nhạc từ lời bài hát (lyrics-to-song) 5 phút đa ngôn ngữ, sử dụng Dual-NTP và Structural Progressive Conditioning (CoT).
            *   2. Speech Synthesis (TTS)
                *   `2503.04724` | (Đã liệt kê ở II.B.3.d) LLMVoX cho streaming TTS.
        *   C. Recommender Systems
            *   `2503.22675` | Đề xuất ReaRec, một khung suy luận nhiều bước tại thời điểm suy diễn (inference-time) cho các mô hình gợi ý tuần tự, sử dụng Reasoning Position Embedding.
        *   D. Security & Adversarial Attacks
            *   `2503.09669` | Đề xuất Silent Branding Attack, một kịch bản nhiễm độc dữ liệu khiến mô hình T2I sinh logo mục tiêu mà không cần trigger văn bản.
    *   **V. Other**
        *   (Hiện tại không có bài báo nào được phân vào mục này)

2.  **SOTA_HIGHLIGHTS**
    | Rank | PaperID   | Keywords (≤ 5)                                                                 | Đột phá                                                                                                                               | Ảnh hưởng                                                                                                                               |
    | :--- | :-------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | `2503.10622` | Dynamic Tanh, LayerNorm Alternative, Efficiency, Deep Learning                 | Thay thế hoàn toàn LayerNorm/RMSNorm bằng DyT, giữ hiệu năng, giảm latency đáng kể.                                                    | Có thể thay đổi cách chuẩn hóa lớp trong nhiều kiến trúc DL, đặc biệt là Transformer, tăng tốc độ huấn luyện và suy luận.                 |
    | 2    | `2503.20215` | Omni-modal, TMRoPE, Thinker-Talker, Streaming Text-to-Speech, Qwen             | Qwen-2.5-Omni: mô hình omni-modal đầu tiên xử lý đa dạng đầu vào (text, image, video, audio) và phát đồng thời văn bản & giọng nói. | Đẩy mạnh hướng tới các AI tổng quát có khả năng tương tác tự nhiên và xử lý thông tin đa phương thức liền mạch.                         |
    | 3    | `2503.19786` | Gemma-3, Multimodal LLM, Long Context (128K), Local/Global Attention, QAT      | Gemma-3: LLM đa phương thức (text+image) hiệu quả, hỗ trợ ngữ cảnh 128K trên GPU phổ thông nhờ kiến trúc Local/Global-Attention và QAT. | Cung cấp mô hình mạnh mẽ, mã nguồn mở, có khả năng xử lý ngữ cảnh dài và đa phương thức, dễ tiếp cận cho cộng đồng.                     |
    | 4    | `2503.19325` | Autoregressive Video, Flow-Matching, Stochastic Clean Context, Long-Short Term | FAR: Khung Frame-wise AutoRegressive kết hợp flow-matching liên tục và attention nhân quả, giải quyết các vấn đề của video AR.        | Cải thiện đáng kể chất lượng và tính nhất quán của video AR, mở đường cho video dài và chất lượng cao hơn.                             |
    | 5    | `2503.09573` | Block Diffusion LM, Semi-Autoregressive, FlexAttention, Noise Schedule         | BD3-LM: Mô hình ngôn ngữ khuếch tán theo khối bán tự hồi quy, hỗ trợ chiều dài tùy ý, song song hóa và KV-cache.                      | Giải quyết hạn chế về độ dài và PPL của diffusion LM truyền thống, cạnh tranh với các mô hình AR.                                      |
    | 6    | `2503.20314` | Video Generation, Diffusion Transformer, Wan-2.1, Wan-VAE, Flow Matching       | Wan-2.1: Mô hình DiT sinh video 480-720p kèm chữ Trung-Anh, vượt SOTA open/closed-source, với Wan-VAE và feature cache hiệu quả.    | Đặt ra tiêu chuẩn mới cho việc sinh video chất lượng cao có chữ, đặc biệt quan trọng cho thị trường đa ngôn ngữ.                        |
    | 7    | `2503.07703` | Text-to-Image, Bilingual, Seedream 2.0, LLM-as-Encoder, Glyph-Aligned ByT5   | Seedream 2.0: Tích hợp LLM song ngữ làm encoder vào DiT, render chữ đa ngữ chính xác chỉ bằng tín hiệu text.                           | Cách mạng hóa việc render chữ trong T2I, đặc biệt cho các ngôn ngữ phức tạp, loại bỏ pipeline OCR.                                     |
    | 8    | `2503.10625` | 3D Human Reconstruction, Animatable Avatar, Gaussian Splatting, Single-Image   | LHM: Mô hình feed-forward đầu tiên tái dựng avatar người animatable (3DGS) từ một ảnh đơn trong ~2s, không cần hậu xử lý.             | Đơn giản hóa và tăng tốc đáng kể quy trình tạo avatar 3D chất lượng cao, có thể tùy chỉnh, từ ảnh đơn.                               |
    | 9    | `2503.07067` | Knowledge Distillation, Contrastive Loss, DISTILLM-2, CALD, LLM Compression  | DISTILLM-2: Phương pháp chưng cất LLM hiệu quả bằng loss tương phản bất đối xứng (CALD), cải thiện trên nhiều tác vụ.                 | Cung cấp một kỹ thuật chưng cất mạnh mẽ và linh hoạt, giúp tạo ra các LLM nhỏ hơn nhưng vẫn giữ được hiệu năng cao.                     |
    | 10   | `2503.00808` | Pre-training Data Selection, Predictive Strength, PRESELECT                    | PRESELECT: Phương pháp chọn lọc dữ liệu tiền huấn luyện nhẹ và hiệu quả dựa trên "độ mạnh dự đoán" của văn bản.                      | Thay đổi cách tiếp cận lựa chọn dữ liệu tiền huấn luyện, có thể giúp tạo ra các LLM mạnh mẽ hơn với chi phí thấp hơn.                   |

3.  **NOVEL_TECH_CONTRIBUTIONS**
    *   **2503.03601 – Phân tích đặc trưng SAE cho ATD – Suy nghĩ**: Việc áp dụng SAE để trích xuất đặc trưng có khả năng diễn giải cho việc phát hiện văn bản nhân tạo là một hướng đi thú vị, đặc biệt là việc phân biệt đặc trưng tổng quát và chuyên biệt. Tuy nhiên, sự phụ thuộc vào diễn giải thủ công/LLM khác và giới hạn trên một mô hình cho thấy cần nghiên cứu thêm về tính tổng quát và tự động hóa.
    *   **2503.10622 – Dynamic Tanh (DyT) thay thế LayerNorm/RMSNorm – Suy nghĩ**: Ý tưởng thay thế LayerNorm bằng một hàm kích hoạt động như DyT rất độc đáo và tiềm năng. Việc loại bỏ chuẩn hóa mà vẫn giữ ổn định và cải thiện hiệu năng trên nhiều lĩnh vực là một đột phá đáng kể, có thể thay đổi cách thiết kế mạng nơ-ron sâu.
    *   **2503.20215 – TMRoPE (Time-aligned Multimodal RoPE) và Thinker-Talker cho Qwen-2.5-Omni – Suy nghĩ**: TMRoPE giải quyết vấn đề đồng bộ audio-video một cách tinh tế. Kiến trúc Thinker-Talker cho phép phát đồng thời text và speech là một bước tiến quan trọng cho các mô hình omni-modal tương tác tự nhiên hơn.
    *   **2503.14456 – Vector-Valued Generalized Δ-Rule trong RWKV-7 – Suy nghĩ**: Việc tổng quát hóa Delta-Rule thành dạng vector cho phép decay và learning rate theo kênh riêng biệt, mở rộng đáng kể không gian trạng thái và khả năng biểu đạt của RNN, thách thức giới hạn của Transformer.
    *   **2503.11647 – Frame-Dimension Conditioning trong ReCamMaster – Suy nghĩ**: Nối latent của video gốc và đích theo trục khung hình cho phép self-attention 3D trao đổi thông tin hiệu quả, là một cách tiếp cận mới để học correspondences trong video-to-video re-rendering.
    *   **2503.06053 – Motion Adaptive Generation (MAG) trong DropletVideo – Suy nghĩ**: Chiến lược lấy mẫu khung thích ứng và đưa tham số chuyển động vào mọi layer qua AdaLN cho phép điều khiển tốc độ/cường độ chuyển động một cách linh hoạt trong giai đoạn suy diễn là một đóng góp thực tiễn và mới mẻ cho việc tạo video.
    *   **2503.23307 – Speech-Video Window Attention trong MoCha – Suy nghĩ**: Giới hạn truy vấn audio theo cửa sổ thời gian cục bộ để đồng bộ khẩu hình-giọng nói là một giải pháp thông minh và trực tiếp cho vấn đề đồng bộ trong các mô hình sinh video nhân vật nói.
    *   **2502.21263 – Sử dụng AI-generated ICD codes để huấn luyện Diagnosis Prediction – Suy nghĩ**: Việc chứng minh mã ICD do AI tạo ra giúp huấn luyện mô hình dự đoán chẩn đoán tốt hơn (đặc biệt mã hiếm) so với mã do bác sĩ gán là một phát hiện quan trọng, có thể thay đổi quy trình chuẩn bị dữ liệu trong y tế.
    *   **2503.14476 – Clip-Higher và Dynamic Sampling trong DAPO – Suy nghĩ**: Clip-Higher cho phép khám phá tốt hơn bằng cách mở trần xác suất token thấp. Dynamic Sampling đảm bảo batch luôn có gradient hữu ích. Cả hai là những cải tiến thực tế cho RL trong LLM.
    *   **2503.05236 – UNIFIEDREWARD: Mô hình phần thưởng hợp nhất đa phương thức – Suy nghĩ**: Hợp nhất đánh giá hiểu biết và tạo sinh cho cả ảnh và video vào một mô hình duy nhất, hỗ trợ cả pairwise và pointwise là một bước tiến lớn, giúp đơn giản hóa và tăng cường hiệu quả của việc học từ sở thích.
    *   **2503.18878 – ReasonScore với phạt entropy để nhận diện đặc trưng suy luận – Suy nghĩ**: ReasonScore là một chỉ số tự động, định lượng để phát hiện latent suy luận từ SAE, kết hợp phạt entropy để tránh "mẹo vặt" là một cách tiếp cận thông minh và có tính ứng dụng cao.
    *   **2503.04625 – Hint-infer và Hint-RFT để kích hoạt khả năng sử dụng công cụ của LRM – Suy nghĩ**: Sử dụng "gợi ý" để kích hoạt khả năng sử dụng công cụ tiềm ẩn của LRM mà không cần dữ liệu minh họa ban đầu là một phương pháp khéo léo và hiệu quả về mặt dữ liệu.
    *   **2503.11576 – DocTags: Định dạng markup thống nhất cho mọi thành phần trang PDF – Suy nghĩ**: DocTags giúp mô hình sinh một chuỗi duy nhất, tránh mất cấu trúc, là một giải pháp thanh lịch cho bài toán chuyển đổi tài liệu phức tạp sang dạng có cấu trúc.
    *   **2503.13358 – Residual Shifting Distillation (RSD) khớp phân phối joint – Suy nghĩ**: Hàm mất Lθ của RSD khớp toàn bộ phân phối joint giữa teacher và student thông qua một "fake ResShift" là một cách tiếp cận chưng cất mới, khác biệt so với các phương pháp chỉ khớp marginal.
    *   **2503.04130 – STORM: Mamba Temporal Projector tích hợp thông tin thời gian vào token ảnh – Suy nghĩ**: Chủ động tích hợp thông tin thời gian vào token ảnh trước khi đưa vào LLM bằng Mamba giúp giảm gánh nặng cho LLM và bảo tồn thông tin động học tốt hơn khi giảm token.
    *   **2503.18942 – Tree-of-Frames (ToF) cho Test-Time Scaling video – Suy nghĩ**: ToF là thuật toán tìm kiếm nhánh-cắt tự hồi quy ba giai đoạn với hierarchical prompting, giảm độ phức tạp đáng kể so với Best-of-N cho việc scaling video tại thời điểm suy diễn.
    *   **2503.01743 – Mixture of LoRAs (MoL) cho Phi-4-Multimodal – Suy nghĩ**: Sử dụng MoL để tích hợp các phương thức mới vào LLM đã đóng băng hoàn toàn, thông qua các LoRA và router đặc thù, là một cách tiếp cận hiệu quả về tham số và bảo toàn năng lực ngôn ngữ gốc.
    *   **2503.07536 – Hàm thưởng hai thành phần khả kiểm chứng (định dạng + chính xác) cho RL đa phương thức – Suy nghĩ**: Thiết kế hàm thưởng có thể kiểm tra tự động, không mơ hồ cho RL đa phương thức là một bước quan trọng để huấn luyện các LMM suy luận mạnh mẽ hơn.
    *   **2503.07677 – PLADIS: Nội suy attention thưa-dày đặc tại thời gian suy luận – Suy nghĩ**: Thay thế cross-attention bằng nội suy giữa softmax dày đặc và α-Entmax thưa tại thời gian suy luận, không cần huấn luyện lại, là một kỹ thuật tăng tốc và cải thiện chất lượng thông minh.
    *   **2503.10633 – Thuật toán khôi phục DAG thực tế cho HF-Model-Atlas với priors cấu trúc – Suy nghĩ**: Kết hợp khoảng cách trọng số, tham số học và các priors (Duplication, Temporal Dynamics, Fan/Snake) để khôi phục DAG hiệu quả là một đóng góp thực tiễn cho việc hiểu hệ sinh thái mô hình.
    *   **2503.10613 – CoSTA*: LLM Subtask Tree + A* với heuristic chi phí-chất lượng và cập nhật g(x) thời gian thực – Suy nghĩ**: Kết hợp LLM để tạo cây phân rã nhiệm vụ, sau đó dùng A* với heuristic và cập nhật chi phí động từ VLM là một giải pháp mạnh mẽ cho biên tập ảnh phức tạp.
    *   **2503.21776 – T-GRPO: So sánh lời giải với khung hình xáo trộn/đúng thứ tự trong RL video – Suy nghĩ**: Buộc mô hình tận dụng thông tin thời gian bằng cách so sánh hiệu suất trên chuỗi khung đúng thứ tự và xáo trộn là một cơ chế thưởng thông minh cho RL trong video reasoning.
    *   **2503.05500 – Quy trình huấn luyện annealing hai giai đoạn cho EuroBERT – Suy nghĩ**: Điều chỉnh tỷ lệ masking, phân phối dữ liệu và độ dài ngữ cảnh trong giai đoạn annealing là một chiến lược tối ưu hóa hiệu năng cho các tác vụ hạ nguồn của encoder đa ngôn ngữ.
    *   **2503.01785 – Visual-RFT: RL với verifiable reward functions cho LVLMs – Suy nghĩ**: Thiết kế các hàm phần thưởng có thể xác minh dựa trên quy tắc (IoU, CLS reward) cho các tác vụ thị giác, thay thế mô hình phần thưởng hoặc phản hồi người, là một hướng đi hiệu quả về dữ liệu.
    *   **2503.19693 – Tokenization Patching và Exponential Embedding Initialization trong AdaptiVocab – Suy nghĩ**: Tokenization Patching hoạt động đè lên tokenizer có sẵn và Exponential Embedding Initialization giúp ổn định sinh tự hồi quy là những kỹ thuật thông minh để thích ứng từ vựng hiệu quả.
    *   **2503.16660 – Select-by-Reconstruction: Auto-encoder + Gumbel-Softmax xác định token hữu ích – Suy nghĩ**: Sử dụng khả năng tái tạo token bị loại để xác định tính hữu ích của token là một cách tiếp cận mới cho việc tỉa token trong ViT.
    *   **2503.19325 – Stochastic Clean Context và Long Short-Term Context Modeling trong FAR – Suy nghĩ**: Ngẫu nhiên thay thế khung nhiễu bằng khung sạch (Stochastic Clean Context) và sử dụng cửa sổ ngắn/dài với độ phân giải khác nhau (Long Short-Term Context) là các giải pháp hiệu quả cho video AR.
    *   **2503.09573 – Vector-hoá hai lượt và noise schedule “clipped” trong BD3-LM – Suy nghĩ**: Thuật toán huấn luyện vector-hoá hai lượt và noise schedule tối ưu dữ liệu giúp giảm phương sai gradient và cải thiện PPL cho diffusion LM.
    *   **2503.04724 – Streaming đa hàng đợi và byte-level G2P trong LLMVoX – Suy nghĩ**: Hệ thống streaming đa hàng đợi cho phép xử lý song song và nhúng G2P cấp byte là những giải pháp hiệu quả cho TTS streaming độ trễ thấp, độc lập LLM.
    *   **2503.07605 – SEAP: Expert-Score và Logistic-Layer Sparsity cho pruning không huấn luyện lại – Suy nghĩ**: Expert-Score kết hợp kích hoạt và trọng số, cùng với hàm Logistic-Layer Sparsity, cho phép tỉa chuyên gia theo tác vụ một cách hiệu quả mà không cần huấn luyện lại.
    *   **2503.12533 – Connector VLM và Pose-Adjustment Composite Skill trong BEING-0 – Suy nghĩ**: Connector VLM nhẹ làm cầu nối giữa FM và kỹ năng cấp thấp, cùng kỹ năng tổng hợp điều chỉnh tư thế, là giải pháp thực tế cho robot hình người.
    *   **2503.08638 – Dual-NTP và Structural Progressive Conditioning (CoT) trong YuE – Suy nghĩ**: Tách timestep thành cặp token vocal-accompaniment (Dual-NTP) và cấp phát lyric/nhãn cấu trúc theo đoạn (Structural Progressive Conditioning) là những đột phá cho việc sinh nhạc dài, chất lượng cao từ lời.
    *   **2503.00865 – Layer extension và quy trình tiền huấn luyện hai giai đoạn cho Babel – Suy nghĩ**: Mở rộng lớp một cách có cấu trúc và quy trình tiền huấn luyện phục hồi/tiếp tục là một phương pháp mới để tăng năng lực LLM đa ngôn ngữ.
    *   **2503.21620 – Rule-based Action Reward (RT, RC, RF) trong UI-R1 – Suy nghĩ**: Thiết kế phần thưởng dựa trên loại hành động, tọa độ click và định dạng cho RL trong GUI là một cách tiếp cận cụ thể và hiệu quả cho tác vụ này.
    *   **2503.17359 – Bộ nhớ hai cấp (Static/Dynamic) và Physics Tuning trong IGV/GGE – Suy nghĩ**: Bộ nhớ hai cấp để duy trì nhất quán và khả năng tùy chỉnh vật lý (Physics Tuning) là những ý tưởng quan trọng cho việc xây dựng các thế giới ảo tương tác, sinh động.
    *   **2503.14378 – Taxonomy 4x14 cho "Impossible Video" và Dual LLM-Judge cho IPV-BENCH – Suy nghĩ**: Xây dựng một taxonomy chi tiết cho video phi thực tế và quy trình đánh giá "justify-then-score" hai chiều là những đóng góp nền tảng cho việc nghiên cứu hiểu và tạo video phức tạp.
    *   **2503.07365 – Lọc online zero-advantage trong MM-Eureka – Suy nghĩ**: Loại bỏ tức thời prompt mà mọi rollout đều đúng/sai giúp giữ gradient thông tin và ổn định RL cho VLM, đặc biệt quan trọng khi scale.
    *   **2503.05132 – Tái hiện "aha moment" trên MLLM non-SFT bằng RLVR – Suy nghĩ**: Việc chứng minh RLVR có thể khơi dậy khả năng tự phản ánh và suy luận sâu trên MLLM cơ sở (chưa qua SFT) là một phát hiện quan trọng, cho thấy tiềm năng của RL trong việc "dạy" MLLM suy luận.
    *   **2503.00808 – PRESELECT: Chọn lọc dữ liệu tiền huấn luyện dựa trên "độ mạnh dự đoán" – Suy nghĩ**: Ý tưởng chọn dữ liệu dựa trên khả năng dự đoán hiệu suất của mô hình là một cách tiếp cận mới, dựa trên dữ liệu và có nguyên tắc hơn so với các heuristic truyền thống.
    *   **2503.16905 – Critic Socratic Loop không cần nhãn vàng trong MAPS – Suy nghĩ**: Sử dụng Critic đặt câu hỏi Socratic để đánh giá logic nội dung và lặp lại quy trình giải quyết vấn đề mà không cần đáp án chuẩn là một cơ chế tự phản hồi thông minh cho hệ đa tác nhân.
    *   **2503.15299 – Knowledge-aware probing và pipeline tạo hard negative answers – Suy nghĩ**: Huấn luyện probe chỉ trên câu hỏi mà model sinh đúng và tạo đáp án sai "khó" giúp lượng hóa kiến thức ẩn chính xác hơn.
    *   **2503.10480 – Dual Preference Optimization (D²PO) cho LVLM – Suy nghĩ**: Tối ưu đồng thời dự đoán trạng thái (world-modeling) và lựa chọn hành động bằng học ưu tiên trực tiếp, nhúng world-model vào lõi huấn luyện là một cách tiếp cận mới và hiệu quả.
    *   **2503.20314 – Wan-VAE 3D causal với feature cache chunk-wise cho video vô hạn – Suy nghĩ**: Wan-VAE với cơ chế feature cache cho phép xử lý video dài vô hạn với bộ nhớ O(L) và tăng tốc suy luận là một cải tiến đáng kể cho các mô hình DiT video.
    *   **2503.13288 – Hàm định giá Joint Advantage–Alignment trong ϕ-Decoding – Suy nghĩ**: Ước lượng giá trị bước bằng cách kết hợp lợi ích foresight và kích thước cụm đường suy luận là một heuristic thông minh để cân bằng khám phá-khai thác.
    *   **2503.19786 – KV-Efficient Local/Global Schema (5:1) và Pan-&-Scan Windowing trong Gemma-3 – Suy nghĩ**: Kiến trúc attention kết hợp local/global với tỷ lệ 5:1 và Pan-&-Scan windowing cho phép xử lý ngữ cảnh rất dài (128K) và ảnh đa tỷ lệ một cách hiệu quả.
    *   **2503.19757 – In-context conditioning và chunk-wise action diffusion trong Dita – Suy nghĩ**: Nối token hình ảnh, ngôn ngữ, thời gian và hành động nhiễu vào cùng một chuỗi, sau đó khử nhiễu theo khối hành động, cho phép mô hình học liên kết vi mô hiệu quả cho VLA.
    *   **2503.10639 – Semantic-Spatial Guidance Module (SSGM) và pipeline tự động gán nhãn GoT – Suy nghĩ**: SSGM điều hòa bước nhiễu bằng ba nhánh hướng dẫn (semantic, spatial, reference) và pipeline tự động tạo dữ liệu GoT là những đóng góp quan trọng cho việc sinh/chỉnh sửa ảnh có kiểm soát.
    *   **2503.16219 – Cosine reward và curriculum trộn độ khó trong RL cho LLM nhỏ – Suy nghĩ**: Sử dụng cosine reward để kiểm soát độ dài và curriculum trộn độ khó là những chiến lược thực tế để ổn định huấn luyện RL cho LLM nhỏ trong điều kiện tài nguyên hạn chế.
    *   **2503.15485 – GECO: Sinh views tương phản bằng diffusion và LLM paraphrase trong TULIP – Suy nghĩ**: GECO tạo cặp dương và âm "khó" một cách tự động, kết hợp với tái dựng hai chiều, giúp TULIP học biểu diễn đa phương thức tốt hơn.
    *   **2503.02003 – Highlighted Chain-of-Thought (HoT) với thẻ XML – Suy nghĩ**: Sử dụng thẻ XML để liên kết các sự kiện trong câu trả lời CoT với câu hỏi gốc là một kỹ thuật đơn giản nhưng hiệu quả để tăng khả năng kiểm chứng.
    *   **2502.18600 – Chain of Draft (CoD) tạo bước suy luận tối giản – Suy nghĩ**: Hướng dẫn LLM tạo các "bản nháp" ngắn gọn thay vì các bước CoT dài dòng là một cách tiếp cận trực quan để giảm chi phí tính toán.
    *   **2503.15265 – Token-hierarchy 3-cấp và DPO cho 3D Mesh trong DeepMesh – Suy nghĩ**: Rời rạc hóa tọa độ mesh bằng token-hierarchy và áp dụng DPO với chuẩn điểm kép (Geometry × Aesthetics) là những đổi mới đáng kể cho việc tạo lưới 3D.
    *   **2503.14478 – Bộ tiêu chí đánh giá cấp-instance và metric kép Win Rate + Reward trong Creation-MMBench – Suy nghĩ**: Thiết kế tiêu chí đánh giá riêng cho từng câu hỏi và metric kết hợp so sánh cặp với điểm factual là một cách tiếp cận chi tiết để đánh giá sáng tạo đa phương thức.
    *   **2503.07598 – Video Condition Unit (VCU) và Concept Decoupling trong VACE – Suy nghĩ**: VCU thống nhất đầu vào cho nhiều tác vụ video và Concept Decoupling tách "reactive" và "inactive" frames là những ý tưởng thông minh cho việc tạo và chỉnh sửa video hợp nhất.
    *   **2503.20201 – Open Search Tool với Query Rephrasing, SERP Retrieval, Passage Augmentation và CodeAct-CoC agent trong ODS – Suy nghĩ**: Một hệ thống tìm kiếm mở, đa giai đoạn kết hợp với agent có khả năng sinh và thực thi mã Python để điều phối tool là một giải pháp mạnh mẽ và linh hoạt.
    *   **2503.15558 – Ontology AI vật lý và cơ chế thưởng MCQ rule-based cho Cosmos-Reason1 – Suy nghĩ**: Thiết lập ontology cho AI vật lý và cơ chế thưởng dựa trên câu hỏi trắc nghiệm tự động cho RL là những đóng góp nền tảng cho việc phát triển VLM có khả năng suy luận về thế giới vật lý.
    *   **2503.05179 – Sketch-of-Thought (SoT) với 3 mô hình suy luận (Conceptual Chaining, Chunked Symbolism, Expert Lexicons) và router model – Suy nghĩ**: Framework SoT với các mô hình suy luận chuyên biệt dựa trên khoa học nhận thức và router tự động chọn mô hình là một cách tiếp cận sáng tạo để tạo suy luận hiệu quả.
    *   **2503.13657 – Taxonomy MAST 14 "chứng bệnh" của hệ đa-tác-nhân LLM và LLM-as-a-Judge pipeline tùy biến – Suy nghĩ**: Xây dựng một taxonomy chi tiết về các lỗi của MAS dựa trên grounded theory và sử dụng LLM-as-a-Judge để chẩn đoán là một đóng góp quan trọng cho việc gỡ lỗi và cải thiện MAS.
    *   **2503.09566 – Stage-wise diffusion và Data-Noise Alignment trong TPDiff – Suy nghĩ**: Huấn luyện đa giai đoạn với FPS tăng dần và kỹ thuật ghép nhiễu tối ưu giúp mô hình hội tụ nhanh và cải thiện chất lượng video.
    *   **2503.07314 – Multi-Agent Chain-of-Thought Planning (Director, Scene Plan, Shot Plan) trong MovieAgent – Suy nghĩ**: Phân tầng suy luận qua nhiều agent chuyên biệt sử dụng CoT nội bộ để tự động hóa việc tạo phim dài là một kiến trúc phức tạp và tiềm năng.
    *   **2503.22230 – Hệ thống thưởng lai ghép RTV + GenRM và Pre-PPO filtering trong RLHF – Suy nghĩ**: Kết hợp verifier đặc thù và mô hình sinh điểm thưởng, cùng với việc lọc trước prompt "điểm thấp - khó học" là những chiến lược dữ liệu thông minh cho RLHF.
    *   **2503.16874 – Planner tự lập lộ trình và đối thoại Teacher-Critic-Student trong MARS – Suy nghĩ**: Planner sinh chuỗi bước và vòng lặp Socratic questioning giữa Teacher-Critic-Student để tối ưu prompt tự động là một cơ chế linh hoạt và không cần template.
    *   **2503.16302 – Progressive Flow Distillation và Hierarchical Volume Decoding trong FlashVDM – Suy nghĩ**: Chưng cất dòng chảy đa giai đoạn và giải mã thể tích phân cấp chỉ tính toán độ phân giải cao gần bề mặt là những kỹ thuật hiệu quả để tăng tốc VDM 3D.
    *   **2503.12885 – Bridge Image Tokens và Hard/Soft Image Attribute Binding trong DreamRenderer – Suy nghĩ**: Sao chép token ảnh làm "cầu nối" và áp dụng ràng buộc thuộc tính cứng/mềm cho các lớp attention khác nhau là một kỹ thuật thông minh để kiểm soát thuộc tính instance mà không cần huấn luyện.
    *   **2503.12590 – Thay thế token thích ứng theo timestep và Patch Perturbation trong Personalize Anything – Suy nghĩ**: Thay thế token sớm để giữ ID và muộn để hòa trộn ngữ nghĩa, cùng với việc làm nhiễu patch, là một chiến lược zero-shot hiệu quả cho cá nhân hóa DiT.
    *   **2503.07572 – Progress reward tính bằng meta-prover trong MRT – Suy nghĩ**: Sử dụng LLM ép kết thúc sớm để tính phần thưởng tiến bộ, cung cấp dense-reward cấp episode mà không cần judge ngoài, là một ý tưởng mới cho Meta-RL.
    *   **2503.01774 – Reference mixing layer và chiến lược tạo dữ liệu Cycle Reconstruction/Model Underfitting trong DIFIX – Suy nghĩ**: Lớp trộn tham chiếu dựa trên self-attention và các chiến lược tạo dữ liệu lỗi-sạch mô phỏng tạo tác 3D là những đóng góp thực tế cho việc tăng cường NVS.
    *   **2503.03803 – EgoRAG: RAG phân cấp với bộ nhớ đa cấp cho video egocentric dài hạn – Suy nghĩ**: Hệ thống RAG phân cấp truy xuất từ tóm tắt cấp cao đến chi tiết clip là một giải pháp hiệu quả để xử lý ngữ cảnh cực dài trong video egocentric.
    *   **2503.19903 – PS3: Localized contrast và giám sát hộp trong huấn luyện CLIP 4K – Suy nghĩ**: Đối sánh feature hi-res được chọn với caption cục bộ và chỉ pooling token trong hộp GT giúp huấn luyện CLIP ở độ phân giải rất cao hiệu quả.
    *   **2503.16397 – Chuyển đổi Flow→TrigFlow không huấn luyện và Hybrid Distillation (sCM + LADD) trong SANA-Sprint – Suy nghĩ**: Phép biến đổi đầu-ra khả vi cho Flow-Matching và kết hợp self-consistency liên tục với đối kháng tiềm ẩn là những kỹ thuật chưng cất mạnh mẽ.
    *   **2503.01307 – "Priming" LLM bằng ví dụ chứa hành vi nhận thức (dù sai) để cải thiện RL – Suy nghĩ**: Việc chứng minh mồi hành vi nhận thức, ngay cả với lời giải sai, có thể cải thiện khả năng học RL là một phát hiện phản trực giác nhưng quan trọng.
    *   **2503.05379 – RLVR với hàm thưởng có thể kiểm chứng (R_acc + R_format) cho nhận dạng cảm xúc đa phương thức – Suy nghĩ**: Áp dụng RLVR với hàm thưởng dựa trên nhãn và định dạng cho HumanOmni là một cách tiếp cận hiệu quả để cải thiện suy luận cảm xúc.
    *   **2503.17126 – Deviation weighting trong DDPO/DORPO – Suy nghĩ**: Đo lường mức khác biệt của mỗi đáp án và dùng làm trọng số trong DPO/ORPO để ưu tiên học các mẫu hiếm nhưng chất lượng là một cải tiến thông minh.
    *   **2503.11646 – Collaborative perturbation paradigm trong ADC – Suy nghĩ**: Người vận hành đối nghịch chủ động thay đổi môi trường/lệnh trong khi người điều khiển thích ứng, tạo ra dữ liệu phong phú về phục hồi lỗi và thích ứng.
    *   **2503.10291 – Monte-Carlo Process Supervision tự động cho VisualPRM – Suy nghĩ**: Sinh nhiều completion/bước để ước lượng kỳ vọng chính xác, biến thành nhãn nhị phân cho PRM đa phương thức là một cách tạo dữ liệu giám sát hiệu quả.
    *   **2503.09669 – Pipeline tạo mask sâu (iterative SDEdit + OWLv2 + DINOv2) cho Silent Branding Attack – Suy nghĩ**: Sử dụng SDEdit lặp, detector và so khớp embedding để tìm vị trí tự nhiên cho logo là một kỹ thuật tinh vi để thực hiện tấn công nhiễm độc dữ liệu.
    *   **2503.07703 – Scaling RoPE-2D và Multi-Reward RLHF cho Seedream 2.0 – Suy nghĩ**: Gán thang đo vị trí khác nhau theo độ phân giải (Scaling RoPE-2D) và tối ưu đồng thời DiT & LLM encoder với nhiều reward model là những cải tiến đáng chú ý.
    *   **2503.22675 – Reasoning Position Embedding (RPE) và Progressive Reasoning Learning (PRL) trong ReaRec – Suy nghĩ**: Tách không gian mã hóa vật phẩm và suy luận (RPE), cùng với việc hạ nhiệt độ và contrastive learning (PRL) để hướng suy luận là những kỹ thuật mới cho gợi ý tuần tự.
    *   **2503.16418 – InfuseNet: Tiêm đặc trưng danh tính qua residual connections – Suy nghĩ**: Thay vì sửa attention, tiêm ID qua residual giúp giữ năng lực sinh ảnh gốc của FLUX, là một cách tiếp cận thanh lịch cho cá nhân hóa.
    *   **2503.05978 – Mặt nạ vùng mặt và hàm mất mát thích ứng cho MagicInfinite – Suy nghĩ**: Hướng sự chú ý chéo của âm thanh vào vùng miệng và tăng trọng số loss cho vùng mặt nhỏ giúp cải thiện đồng bộ môi trong video nói chuyện.
    *   **2503.07860 – Frame Localizer Viterbi-CLIP cho VidDiff – Suy nghĩ**: Căn thẳng phân đoạn sub-action bằng CLIP similarity và ràng buộc thứ tự Viterbi là một giải pháp zero-shot hiệu quả cho việc định vị khác biệt trong video.
    *   **2503.06955 – Attention-based Mask Modeling và cặp TAT/SAT trong Motion Anything – Suy nghĩ**: Chọn khung động và khớp nối có điểm chú ý cao để che, cùng với Transformer thích ứng theo thời gian/không gian, là những cải tiến cho sinh chuyển động đa mô thức.
    *   **2503.10437 – Time-varying semantic field học từ embedding câu và Status Deformable Network trong 4D LangSplat – Suy nghĩ**: Học trường ngữ nghĩa động trực tiếp từ embedding câu (thay vì visual features) và mô hình hóa chuyển trạng thái bằng prototype-state + MLP là những ý tưởng đột phá cho hiểu cảnh 4D.

4.  **GAPS_AND_OPPORTUNITIES**
    *   **Interpretability of Complex Models:** Nhiều phương pháp (đặc biệt là các mô hình lớn, end-to-end) vẫn thiếu khả năng diễn giải sâu sắc về cách chúng đưa ra quyết định hoặc tại sao chúng thất bại (ví dụ: 2503.03601, 2503.18878, 2503.15299). Cần thêm các kỹ thuật để "mở hộp đen".
    *   **Scalability and Efficiency of Generative Models:** Mặc dù có nhiều tiến bộ về hiệu quả (ví dụ: 2503.10622, 2503.09566, 2503.16397, 2503.09641, 2503.09662), việc huấn luyện và triển khai các mô hình sinh (đặc biệt là video và 3D) vẫn rất tốn kém tài nguyên (ví dụ: 2503.11647, 2503.20314, 2503.15265, 2503.16302).
    *   **Data Scarcity for Specialized Tasks/Domains:** Nhiều lĩnh vực (ví dụ: y tế ngôn ngữ ít tài nguyên - 2502.21263, video/3D/motion với điều kiện phức tạp - 2503.06955, 2503.17359, 2503.09151, 2503.10625) vẫn thiếu bộ dữ liệu quy mô lớn, chất lượng cao, có chú thích chi tiết. Các phương pháp tự giám sát, bán giám sát, và tạo dữ liệu tổng hợp cần được đẩy mạnh.
    *   **Robustness and Generalization:** Nhiều mô hình hoạt động tốt trên các benchmark cụ thể nhưng có thể thất bại khi gặp dữ liệu OOD (out-of-distribution) hoặc các kịch bản thế giới thực phức tạp hơn (ví dụ: 2503.11646, 2503.05379). Cần các phương pháp tăng cường tính bền vững và khả năng khái quát hóa.
    *   **Long-term Temporal Consistency in Video:** Việc duy trì sự nhất quán về đối tượng, hành động và bối cảnh trong các video dài vẫn là một thách thức lớn (ví dụ: 2503.19325, 2503.05978, 2503.07314).
    *   **Fine-grained Control and Compositionality:** Mặc dù có nhiều tiến bộ trong việc kiểm soát đầu ra của mô hình sinh (ví dụ: 2503.11647, 2503.10639, 2503.12885, 2503.12590), việc kiểm soát chi tiết các thuộc tính, tương tác giữa các đối tượng và tính kết hợp (compositionality) vẫn cần cải thiện.
    *   **Evaluation Metrics and Benchmarks:** Các thước đo hiện tại có thể chưa phản ánh đầy đủ chất lượng hoặc các khía cạnh mong muốn của mô hình (ví dụ: sự sáng tạo, tính nhất quán ngữ nghĩa sâu, an toàn). Cần các benchmark và metric mới, toàn diện hơn (như các nỗ lực trong 2503.14378, 2503.21755, 2503.19990, 2503.07860, 2503.05085, 2503.14478, 2503.10291).
    *   **Safety, Ethics, and Misuse:** Các mô hình ngày càng mạnh mẽ hơn cũng đi kèm với nguy cơ lạm dụng (ví dụ: deepfakes từ 2503.23307, 2503.05978; tấn công nhiễm độc 2503.09669). Nghiên cứu về an toàn, phát hiện nội dung AI, và các biện pháp phòng chống cần được ưu tiên.
    *   **Human-AI Collaboration and Interaction:** Làm thế nào để con người và AI có thể hợp tác hiệu quả hơn, đặc biệt trong các tác vụ sáng tạo hoặc giải quyết vấn đề phức tạp (ví dụ: 2503.11646, 2503.17359, 2503.10613).
    *   **Reasoning in Complex, Dynamic Environments:** Khả năng suy luận của AI trong các môi trường động, có nhiều yếu tố bất định, và yêu cầu lập kế hoạch nhiều bước vẫn còn hạn chế (ví dụ: 2503.12533, 2503.10480, 2503.15558).
    *   **Cross-modal Understanding and Generation:** Việc hiểu và tạo ra mối liên kết sâu sắc, tinh tế giữa các phương thức khác nhau (ví dụ: văn bản-âm thanh-hình ảnh-chuyển động) vẫn là một lĩnh vực cần nhiều khám phá (ví dụ: 2503.20215, 2503.08638).
    *   **Resource Efficiency for Small Models:** Trong khi các mô hình lớn đạt được hiệu năng ấn tượng, việc phát triển các mô hình nhỏ gọn, hiệu quả về tài nguyên mà vẫn duy trì khả năng suy luận và thực thi tốt là rất quan trọng (ví dụ: 2503.11576, 2503.16219).
    *   **Automated Failure Diagnosis and Correction in Multi-Agent Systems:** Cần các phương pháp tự động hơn để phát hiện và sửa lỗi trong các hệ thống đa tác nhân phức tạp (như đề xuất trong 2503.13657).
    *   **Theoretical Understanding of Emergent Abilities:** Hiểu rõ hơn về cách các khả năng phức tạp (như suy luận nhiều bước, "aha moment") xuất hiện trong các mô hình lớn (như nghiên cứu trong 2503.01307, 2503.05132).

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Adaptive World Models for Interactive Generative Environments**
    *   **Motivation:** Các Generative Game Engines (GGEs) như trong 2503.17359 hứa hẹn tạo ra thế giới tương tác vô hạn, nhưng việc duy trì tính nhất quán vật lý và logic nhân quả động khi người dùng tương tác mạnh vẫn là thách thức. Các mô hình hiện tại thường dựa trên prior vật lý ngầm hoặc bộ mô phỏng cố định.
    *   **Key novelty:** Phát triển một World Model có khả năng *học và thích ứng động* với các quy luật vật lý và logic nhân quả mới do người dùng tạo ra hoặc do các sự kiện bất ngờ trong game. Thay vì chỉ "tune" các tham số vật lý có sẵn, mô hình có thể học các "luật chơi" mới nổi.
    *   **Approach:**
        1.  Sử dụng một MLLM (như Cosmos-Reason1 - 2503.15558) làm lõi suy luận cho World Model.
        2.  Huấn luyện MLLM này trên dữ liệu video tương tác, nơi các quy luật vật lý/logic có thể thay đổi.
        3.  Kết hợp Reinforcement Learning (có thể là RLHF hoặc RLAIF) để World Model học cách dự đoán và phản ứng với các thay đổi trong "luật chơi" dựa trên phản hồi (ví dụ: sự ngạc nhiên của người chơi, lỗi game, hoặc mục tiêu mới).
        4.  Sử dụng một bộ nhớ động (như trong 2503.17359) để lưu trữ các "luật chơi" đã học và các trạng thái bất thường.
        5.  Tích hợp với một IGV để render các thay đổi.
    *   **Dataset + Metrics:**
        *   Dataset: Tạo một bộ dữ liệu mô phỏng các môi trường game đơn giản (ví dụ: dựa trên Minecraft hoặc các physics sandbox) nơi người dùng có thể thay đổi các quy luật vật lý cơ bản (trọng lực, ma sát, tính chất vật liệu) hoặc giới thiệu các cơ chế logic mới. Thu thập các cặp (trạng thái trước, hành động người dùng/sự kiện, trạng thái sau, mô tả thay đổi luật).
        *   Metrics:
            *   Predictive Accuracy: Khả năng dự đoán đúng trạng thái tiếp theo khi có thay đổi luật.
            *   Adaptation Speed: Số bước/lượng dữ liệu cần để World Model thích ứng với luật mới.
            *   Plausibility Score: Đánh giá của con người về tính hợp lý của các tương tác sau khi luật thay đổi.
            *   Consistency Score: Mức độ nhất quán của thế giới sau nhiều thay đổi luật.
    *   **Risk/Feasibility:**
        *   Risk: Cao. Việc học các quy luật vật lý/logic mới từ đầu hoặc thích ứng nhanh là rất khó. Có thể gặp vấn đề về tính ổn định và khả năng khái quát hóa.
        *   Feasibility: Trung bình. Các thành phần cơ bản (MLLM, RL, IGV) đã có những tiến bộ. Thách thức chính là thiết kế cơ chế học và biểu diễn tri thức về "luật chơi" một cách hiệu quả.

    ✨ **Idea 2: Zero-Shot Cross-Modal Silent Branding Attack and Defense**
    *   **Motivation:** Nghiên cứu 2503.09669 về Silent Branding Attack cho thấy nguy cơ mô hình T2I bị nhiễm độc để sinh logo ẩn. Hiện tại, tấn công này tập trung vào T2I. Việc mở rộng sang các mô hình đa phương thức khác (ví dụ: Text-to-Video, Text-to-3D, Text-to-Audio) và phát triển cơ chế phòng thủ là cấp thiết.
    *   **Key novelty:**
        1.  Phát triển một phương pháp tấn công Silent Branding *zero-shot* có khả năng lây nhiễm logo/watermark ẩn sang nhiều phương thức đầu ra (video, 3D, audio) chỉ từ dữ liệu huấn luyện văn bản-hình ảnh bị nhiễm độc, mà không cần trigger rõ ràng ở các phương thức mới.
        2.  Đề xuất một cơ chế phòng thủ dựa trên "cross-modal inconsistency detection" để phát hiện các hiện tượng branding ẩn này.
    *   **Approach:**
        *   **Attack:**
            1.  Sử dụng pipeline tương tự 2503.09669 để tạo bộ dữ liệu văn bản-hình ảnh nhiễm độc logo.
            2.  Huấn luyện một MLLM nền tảng (ví dụ: Qwen-2.5-Omni - 2503.20215 hoặc Phi-4-Multimodal - 2503.01743) trên bộ dữ liệu nhiễm độc này, tập trung vào việc học các biểu diễn đa phương thức.
            3.  Đánh giá xem khi mô hình sinh ra các phương thức khác (video, audio) từ prompt văn bản trung tính, logo/watermark có xuất hiện một cách tiềm ẩn hay không. Khai thác hiệu ứng "pattern repetition" xuyên phương thức.
        *   **Defense:**
            1.  Phát triển một mô hình giám sát (có thể là một MLLM khác hoặc một mạng chuyên biệt) được huấn luyện để phát hiện sự bất nhất quán hoặc các mẫu hình đáng ngờ xuyên các phương thức.
            2.  Ví dụ: Nếu một prompt văn bản trung tính tạo ra hình ảnh có logo và video cũng có các đặc điểm hình ảnh/âm thanh tương ứng với logo đó một cách không tự nhiên, hệ thống sẽ cảnh báo.
            3.  Sử dụng kỹ thuật phân tích diễn giải (ví dụ: SAEs từ 2503.03601) trên các biểu diễn đa phương thức để tìm các "feature" liên quan đến branding.
    *   **Dataset + Metrics:**
        *   Dataset: Mở rộng bộ dữ liệu nhiễm độc từ 2503.09669. Tạo các cặp đa phương thức (text-video, text-audio) từ các prompt văn bản trung tính sử dụng mô hình bị tấn công.
        *   Metrics:
            *   Attack: Logo Injection Rate (LIR) trên các phương thức mới, FAE, đánh giá stealthiness của con người/AI.
            *   Defense: Detection Accuracy, False Positive Rate.
    *   **Risk/Feasibility:**
        *   Risk: Tấn công: Trung bình-Cao (khó đạt được hiệu ứng zero-shot mạnh mẽ). Phòng thủ: Trung bình (phát hiện bất nhất quán tinh vi là thách thức).
        *   Feasibility: Trung bình. Các công cụ để tấn công và huấn luyện MLLM đã có. Thách thức là làm cho hiệu ứng branding lan truyền zero-shot và cơ chế phát hiện đủ nhạy.

    ✨ **Idea 3 (Moon-shot): Emergent Cognitive Architecture from Composable Reasoning Primitives**
    *   **Motivation:** Các nghiên cứu như 2503.01307 (hành vi nhận thức), 2503.18878 (ReasonScore), 2503.05132 (VisualThinker-R1-Zero) và các framework MCoT (2503.12605) cho thấy LLM/MLLM đang bắt đầu thể hiện các mảnh ghép của năng lực nhận thức. Tuy nhiên, chúng thường được huấn luyện end-to-end hoặc được "mồi" một cách thụ động.
    *   **Key novelty:** Xây dựng một kiến trúc AI có khả năng *tự tổ chức và kết hợp* các "khối xây dựng nhận thức" (cognitive building blocks - CBBs) cơ bản (ví dụ: verification, backtracking, subgoal setting, attention, memory retrieval, tool use) một cách linh hoạt để giải quyết các vấn_đề mới lạ, tương tự như cách con người kết hợp các kỹ năng nhận thức.
    *   **Approach:**
        1.  **Define CBBs:** Dựa trên các nghiên cứu về khoa học nhận thức và AI (ví dụ: 2503.01307, 2503.05179), xác định một tập hợp các CBBs cốt lõi. Mỗi CBB có thể là một mô-đun mạng nơ-ron nhỏ, chuyên biệt hoặc một quy trình được định nghĩa bằng prompt.
        2.  **Composable Architecture:** Thiết kế một kiến trúc meta-level (có thể là một agent điều phối dựa trên RL hoặc một mạng nơ-ron học cách kết nối các CBBs) có khả năng lựa chọn, sắp xếp và tham số hóa các CBBs để tạo thành một "chiến lược nhận thức" (cognitive strategy) cho một tác vụ cụ thể.
        3.  **Meta-Learning/RL for Composition:** Huấn luyện kiến trúc meta-level này thông qua meta-learning hoặc RL. Phần thưởng dựa trên hiệu suất giải quyết vấn đề và "hiệu quả nhận thức" (ví dụ: sử dụng ít CBBs nhất, hoặc các CBBs ít tốn kém nhất).
        4.  **Emergent Behavior:** Nghiên cứu xem liệu các chiến lược nhận thức phức tạp, chưa từng được dạy trực tiếp, có thể tự nổi lên từ sự tương tác của các CBBs hay không.
        5.  Sử dụng các kỹ thuật từ 2503.10480 (D²PO) để đồng thời tối ưu hóa việc lựa chọn hành động (kết hợp CBBs) và mô hình hóa thế giới (hiểu tác vụ).
    *   **Dataset + Metrics:**
        *   Dataset: Các bộ dữ liệu đòi hỏi suy luận phức tạp, đa bước, và khả năng thích ứng (ví dụ: ARC, các bài toán Olympic từ OlymMATH - 2503.21380, các kịch bản trong LEGO-Puzzles - 2503.19990, hoặc các môi trường tương tác như trong 2503.17359).
        *   Metrics:
            *   Task Success Rate.
            *   Cognitive Strategy Efficiency (số bước, số CBBs, chi phí tính toán).
            *   Novelty of Emergent Strategies (đánh giá của con người hoặc so sánh với các giải pháp đã biết).
            *   Transferability: Khả năng áp dụng các chiến lược đã học cho các vấn đề hoàn toàn mới.
    *   **Risk/Feasibility:**
        *   Risk: Rất cao. Đây là một mục tiêu dài hạn của AGI. Việc định nghĩa CBBs, thiết kế cơ chế kết hợp hiệu quả và huấn luyện meta-level là cực kỳ thách thức.
        *   Feasibility: Thấp trong ngắn hạn, nhưng có thể bắt đầu với các phiên bản đơn giản hóa. Các tiến bộ trong multi-agent systems (2503.16905, 2503.13657), meta-RL (2503.07572) và modular AI cung cấp một số nền tảng.

6.  **READING_LIST** (Top papers đáng đọc)
    *   `2503.10622` – DyT: No-Norm LLaMA · Thay thế LayerNorm bằng Dynamic Tanh, một ý tưởng đơn giản nhưng có thể thay đổi cơ bản cách xây dựng mạng nơ-ron sâu.
    *   `2503.20215` – Qwen-2.5-Omni · Một bước tiến ấn tượng tới AI đa phương thức thực sự, xử lý nhiều loại dữ liệu và đầu ra đồng thời.
    *   `2503.19786` – Gemma-3 · Cung cấp một họ MLLM mã nguồn mở mạnh mẽ, hiệu quả, với khả năng xử lý ngữ cảnh dài, rất quan trọng cho cộng đồng.
    *   `2503.00808` – PRESELECT · Đề xuất một phương pháp chọn lọc dữ liệu tiền huấn luyện mới lạ và có nguyên tắc, có thể cải thiện đáng kể hiệu quả huấn luyện LLM.
    *   `2503.01307` – Cognitive Behaviors in LLMs · Nghiên cứu sâu sắc về các hành vi nhận thức nền tảng cho khả năng tự cải thiện của LLM qua RL, mở ra hướng mới để hiểu và xây dựng LLM thông minh hơn.
    *   `2503.12605` – Multimodal CoT Survey · Một khảo sát toàn diện và cần thiết về một lĩnh vực đang phát triển nhanh chóng, cung cấp cái nhìn tổng quan và định hướng cho nghiên cứu MCoT.
    *   `2503.17359` – Interactive Generative Video (IGV) · Đặt nền móng cho các Generative Game Engine, một tầm nhìn thú vị về tương lai của nội dung tương tác.
    *   `2503.07067` – DISTILLM-2 · Một phương pháp chưng cất kiến thức hiệu quả sử dụng loss tương phản, quan trọng cho việc tạo ra các LLM nhỏ gọn hơn.
    *   `2503.10625` – LHM: Animatable Human Avatars · Tái tạo avatar 3D động từ ảnh đơn với tốc độ và chất lượng ấn tượng, có tiềm năng ứng dụng lớn.
    *   `2503.09669` – Silent Branding Attack · Một nghiên cứu quan trọng về lỗ hổng an ninh của mô hình T2I, cảnh báo về các nguy cơ tiềm ẩn và thúc đẩy nghiên cứu về phòng thủ.

7.  **META_REFLECTION**
    *   Tập hợp các bài báo tháng 03/2025 cho thấy một sự phát triển mạnh mẽ và đa dạng trong lĩnh vực AI. Có ba xu hướng lớn nổi bật:
        1.  **Hiệu quả và Khả năng mở rộng (Efficiency and Scalability):** Nhiều nghiên cứu tập trung vào việc làm cho các mô hình lớn (LLM, MLLM, Diffusion Models) trở nên hiệu quả hơn về mặt tính toán, bộ nhớ và dữ liệu. Điều này thể hiện qua các kỹ thuật mới về kiến trúc (ví dụ: Dynamic Tanh `2503.10622`, Local/Global Attention trong Gemma-3 `2503.19786`), phương pháp chưng cất (RSD `2503.13358`, DISTILLM-2 `2503.07067`, SWD `2503.16397`), tối ưu hóa suy luận (PLADIS `2503.07677`, ϕ-Decoding `2503.13288`, CoRe² `2503.09662`), và chiến lược chọn lọc/tạo dữ liệu thông minh (PRESELECT `2503.00808`, ADC `2503.11646`). Điều này cho thấy cộng đồng đang nỗ lực để "dân chủ hóa" AI mạnh mẽ, giúp chúng dễ tiếp cận và triển khai hơn.
        2.  **Hướng tới AI Tổng quát và Đa phương thức (Towards General and Multimodal AI):** Sự trỗi dậy của các mô hình "Omni" có khả năng xử lý và tạo ra nhiều loại phương thức (văn bản, hình ảnh, video, âm thanh, 3D) là rất rõ ràng (Qwen-2.5-Omni `2503.20215`, Phi-4-Multimodal `2503.01743`, Gemma-3 `2503.19786`). Đồng thời, các nghiên cứu về suy luận đa phương thức (MCoT `2503.12605`, LMM-R1 `2503.07536`, Cosmos-Reason1 `2503.15558`), hiểu biết và tương tác với thế giới (Embodied AI, Robotics `2503.12533`, `2503.10480`, `2503.19757`) cũng đang được đẩy mạnh. Điều này phản ánh tham vọng xây dựng các hệ thống AI có khả năng hiểu và tương tác với thế giới một cách toàn diện hơn.
        3.  **Tăng cường Khả năng Diễn giải, Kiểm soát và An toàn (Enhanced Interpretability, Controllability, and Safety):** Song song với việc tăng cường sức mạnh của AI, các nhà nghiên cứu cũng ngày càng quan tâm đến việc hiểu cách chúng hoạt động (SAEs `2503.03601`, ReasonScore `2503.18878`, Hidden Knowledge `2503.15299`), kiểm soát đầu ra một cách chi tiết (ReCamMaster `2503.11647`, GoT `2503.10639`, DreamRenderer `2503.12885`), và giải quyết các vấn đề về an toàn, đạo đức (Silent Branding Attack `2503.09669`, đánh giá agent `2503.16416`, MAST `2503.13657`). Việc xây dựng các benchmark mới, chi tiết hơn cho các năng lực cụ thể (IPV-BENCH `2503.14378`, VBench-2.0 `2503.21755`, S2S-Arena `2503.05085`) cũng cho thấy nỗ lực hướng tới đánh giá AI một cách khắt khe và có trách nhiệm hơn.
    *   Nhìn chung, tháng 03/2025 đánh dấu những bước tiến quan trọng trong việc làm cho AI mạnh mẽ hơn, hiệu quả hơn, dễ hiểu hơn và an toàn hơn, với một sự nhấn mạnh rõ ràng vào khả năng xử lý đa phương thức và suy luận phức tạp. Các phương pháp học tăng cường, học từ sở thích, và các kỹ thuật tự giám sát/tự cải thiện tiếp tục đóng vai trò trung tâm trong việc thúc đẩy các giới hạn này.
