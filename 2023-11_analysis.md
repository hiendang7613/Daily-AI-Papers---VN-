1.  **TOPIC_TREE**

    *   NLP (Natural Language Processing)
        *   Large Language Models (LLM) & Multimodal Language Models (MLLM)
            *   Model Architecture & Optimization
                *   Efficient Inference & Architecture
                    *   2311.10770 | UltraFastBERT giới thiệu Fast Feedforward Networks và Conditional Matrix Multiplication (CMM) để tăng tốc BERT inference bằng cách chỉ kích hoạt một phần nhỏ neuron.
                    *   2311.01282 | FlashDecoding++ đề xuất Softmax không đồng bộ, tối ưu GEMM phẳng và heuristic dataflow thích ứng để tăng tốc LLM inference trên GPU.
                    *   2311.08263 | FastCoT kết hợp Jacobi decoding và autoregressive decoding để tăng tốc inference cho tác vụ Chain-of-Thought mà không cần huấn luyện lại.
                *   Alternative Attention Mechanisms
                    *   2311.10642 | Đề xuất thay thế các module attention trong Transformer bằng kiến trúc Feedforward (FF) và huấn luyện bằng knowledge distillation.
                    *   2311.11829 | System 2 Attention (S2A) sử dụng LLM để tái tạo ngữ cảnh chỉ chứa thông tin liên quan, cải thiện tính chính xác và khách quan.
                *   Sparse & Mixture-of-Experts (MoE) Architectures
                    *   2311.10768 | Mixture-of-Word-Experts (MoWE) thay thế lớp FFN bằng "word experts" gắn cố định với token trong từ điển định tuyến, sử dụng MoE tĩnh quy mô lớn.
            *   Training, Fine-tuning & Adaptation
                *   Instruction Tuning & Explanation Tuning
                    *   2311.11045 | Orca 2 giới thiệu Cautious Reasoning framework và Prompt Erasure technique để LLM học cách chọn chiến lược giải quyết vấn đề.
                    *   2310.20689 | LEMA (Learning from MistAkes) cho phép LLM học từ sai lầm bằng cách đưa dữ liệu mistake-correction vào fine-tuning, với correction-centric evolution.
                    *   2311.10702 | TÜLU-V2-mix là bộ dữ liệu chỉ dẫn chất lượng cao, được sử dụng để fine-tune LLaMA-2 và Code Llama, kết hợp DPO và mở rộng ngữ cảnh.
                *   Parameter-Efficient Fine-Tuning (PEFT)
                    *   2311.11501 | MultiLoRA mở rộng ngang các module LoRA song song với hệ số scaling học được để cải thiện hiệu năng đa tác vụ.
                    *   2311.06243 | Orthogonal Butterfly (BOFT) là phương pháp finetuning trực giao hiệu quả tham số dựa trên factorization bướm.
                    *   2311.09578 | Tied-LoRA kết hợp weight tying ma trận A, B qua các lớp Transformer với selective training/freeze các vector scaling u, v.
                *   Data-Centric Approaches
                    *   2311.03301 | Ziya2 đề xuất chiến lược tiền huấn luyện liên tục ba giai đoạn và data-centric scaling laws để tăng cường năng lực LLM cho tiếng Trung, toán, code.
                *   Distributed & Federated Learning
                    *   2311.08105 | DiLoCo là thuật toán phân tán giảm giao tiếp cho huấn luyện ngôn ngữ, biến thể từ Federated Averaging với AdamW inner và Nesterov Momentum outer.
            *   Prompt Engineering & Optimization
                *   2311.09277 | Contrastive Chain-of-Thought prompting cung cấp đồng thời minh họa suy luận hợp lệ và không hợp lệ để hướng dẫn LLM.
                *   2311.05661 | PE2 là phương pháp meta-prompt hoàn thiện cho tự động hóa prompt engineering, kết hợp miêu tả tác vụ, bố cục ngữ cảnh và mẫu lập luận.
                *   2311.12229 | NeuroPrompts là khuôn khổ tự động tối ưu prompt cho T2I, kết hợp SFT, PPO (với reward PickScore) và NeuroLogic Decoding.
            *   Evaluation & Benchmarking
                *   2311.12983 | GAIA là benchmark đánh giá General AI Assistants, tập trung vào reasoning, xử lý đa phương tiện, web browsing và tool-use.
                *   2311.12022 | GPQA là benchmark câu hỏi trắc nghiệm khoa học "Google-proof" cấp cao, với quy trình thu thập dữ liệu và cơ chế thưởng phức tạp.
                *   2311.07911 | IFEval là benchmark đánh giá khả năng theo dõi hướng dẫn tự động và khách quan cho LLM với 25 loại "hướng dẫn có thể xác minh".
                *   2311.07463 | MEGAVERSE mở rộng benchmark MEGA, đánh giá LLM và LMM trên 22 bộ dữ liệu đa ngôn ngữ (83 ngôn ngữ) và đa phương thức.
                *   2310.20216 | Nghiên cứu thực nghiệm đánh giá khả năng của GPT-4 trong bối cảnh bài kiểm tra Turing, phân tích yếu tố ảnh hưởng quyết định của người hỏi.
            *   Reasoning & Problem Solving
                *   2311.06158 | LOGIPT fine-tune LLM để mô phỏng quy trình suy diễn của bộ giải logic, sinh bước suy luận và kết quả không cần NL-to-SL parsing.
                *   2311.04254 | Everything of Thoughts (XOT) kết hợp MCTS và RL (mạng chính sách-giá trị) để tạo luồng suy nghĩ hỗ trợ LLM, với cơ chế sửa đổi suy nghĩ tương tác.
            *   Multimodal Capabilities
                *   Vision-Language Models (VLM) / Multimodal LLMs (MLLM)
                    *   Architectures & Training
                        *   2311.06242 | Florence-2 là kiến trúc sequence-to-sequence thống nhất cho nhiều tác vụ thị giác-ngôn ngữ, sử dụng location tokens và data engine tự động (FLD-5B).
                        *   2311.04219 | OtterHD-8B là MLLM dựa trên Fuyu-8B, xử lý ảnh độ phân giải linh hoạt đến 1024x1024 bằng cách chia patch và ánh xạ trực tiếp vào decoder.
                        *   2311.04589 | TEAL thống nhất mọi đầu vào đa phương thức thành chuỗi token chung, với lớp nhúng phi văn bản và ma trận đầu ra chiếu vào không gian LLM.
                        *   2311.03079 | CogVLM tích hợp Visual Expert (tham số QKV, FFN riêng cho token ảnh) vào mỗi lớp Transformer của LLM để deep fusion.
                        *   2311.04257 | mPLUG-Owl2 thiết kế Modality-Adaptive Module (MAM) trong language decoder, tách rời key/value projection theo modal, dùng chung query và FFN.
                        *   2311.07575 | SPHINX unfreeze LLM cho pre-training kết hợp, mix trọng số real/synthetic data, ensemble visual embeddings và xử lý ảnh đa tỷ lệ.
                    *   Grounding & Interaction
                        *   2311.03356 | GLaMM tích hợp Global/Region/Grounding Image Encoder và Pixel Decoder, hỗ trợ hội thoại đa hạt nhân với segmentation mask output.
                        *   2311.04498 | NExT-Chat (dựa trên Vicuna, CLIP ViT) giới thiệu pix2emb (embedding vị trí) và token <trigger>/<loc> cho visual grounding và segmentation.
                        *   2311.13435 | PG-Video-LLaVA là LMM video hỗ trợ định vị pixel-level, tích hợp object tracking, segmentation và audio processing đa giai đoạn.
                    *   Data Generation & Augmentation
                        *   2310.20550 | CAPSFUSION là pipeline tích hợp raw captions và synthetic captions (từ LLM) để tạo dataset chú thích ảnh chất lượng cao (CAPSFUS-120M).
                        *   2311.07574 | LVIS-INSTRUCT4V sử dụng GPT-4V và hộp giới hạn LVIS để tự động tạo cặp câu hỏi-đáp đa lượt và mô tả ảnh chi tiết.
                        *   2311.06783 | Q-Instruct dataset được tạo từ Q-Pathway (phản hồi tự nhiên về thuộc tính hình ảnh mức thấp) bằng pipeline GPT-tham gia, cho low-level visual instruction tuning.
                *   Audio-Language Models
                    *   2311.01615 | FLAP kết hợp masking 1D/2D trên token phổ âm thanh, contrastive learning và reconstruction cho học đối比 audio-text.
            *   Tool Use & Agents
                *   2311.05437 | LLaVA-Plus là LMM học lập kế hoạch và sử dụng công cụ (skill-oriented dialogues) với định dạng Thought-Action-Value và kho kỹ năng.
                *   2311.05657 | LUMOS là agent framework modular (Planning, Grounding, Execution Modules) với hai công thức tương tác (OnePass, Iterative) và data conversion từ GPT-4/4V.
                *   2311.01767 | PPTC benchmark đánh giá khả năng tạo/chỉnh sửa file PowerPoint qua đối thoại, với PPT reader, API điều khiển và PPTX-Match evaluation.
                *   2311.05772 | ADAPT là thuật toán đệ quy phân tách và lập kế hoạch "as-needed" cho LLM agent, dựa trên self-evaluation và logic AND/OR.
            *   Model Serving & Efficiency
                *   2311.04934 | Prompt Cache sử dụng Prompt Markup Language (PML) để định nghĩa prompt module và tái sử dụng attention states, giảm TTFT.
                *   2311.03285 | S-LoRA giới thiệu Unified Paging (KV cache + LoRA weights), custom CUDA kernels (MBGMM, MBGMV) và S-LoRA TP cho serving nhiều adapter hiệu quả.
            *   Alignment & Safety
                *   2311.08401 | FactTune tối ưu tính chân thực của LLM qua DPO với preference pairs tự động sinh từ reference-free/based truthfulness estimation.
                *   2311.07590 | Nghiên cứu thực nghiệm về khả năng LLM (GPT-4) tự phát triển hành vi lừa dối có chủ đích trong môi trường mô phỏng áp lực cao.
            *   Foundations & Concepts
                *   2311.02462 | Đề xuất khung phân loại "Levels of AGI" dựa trên Hiệu suất và Tính Tổng quát, cùng 6 nguyên tắc cốt lõi.
                *   2311.00059 | Đề xuất giả thuyết "Nghịch lý AI Tạo sinh" và khung thực nghiệm (đánh giá chọn lọc/thẩm vấn) để kiểm tra.
    *   Computer Vision (CV)
        *   Image Generation & Synthesis
            *   Diffusion Models
                *   Acceleration & Efficiency
                    *   2311.05556 | LCM-LoRA chưng cất Latent Consistency Model sang cấu trúc LoRA, tạo module gia tốc plug-and-play cho Stable Diffusion.
                    *   2311.09257 | UFOGen là mạng diffusion-GAN một bước, dự đoán x0 trực tiếp và sampling x(t-1) theo tiến trình nhiễu tiền phương.
                *   Controllable Generation & Editing
                    *   2311.06772 | Chèn hướng dẫn pixel-level (landmark khuôn mặt) vào diffusion zero-shot để cải thiện phát hiện landmark trong talking head.
                    *   2311.12092 | Concept Sliders huấn luyện LoRA adaptors từ cặp prompt/ảnh để tạo thanh trượt điều khiển khái niệm liên tục, plug-and-play.
                *   Personalization & Consistency
                    *   2311.10093 | Quy trình lặp sinh ảnh, embedding (DINOv2), clustering, chọn cụm kết dính nhất và cá nhân hóa (textual inversion + LoRA) để tạo nhân vật nhất quán từ text.
                    *   2311.13600 | ZipLoRA tối ưu hợp nhất LoRA nội dung và phong cách qua hệ số kết hợp cột, pruning và điều khoản orthogonality.
                    *   2311.10794 | Style Tailoring fine-tune LDM bằng cách tách rời content loss và style loss theo các bước denoising; Transparency Module sinh ảnh RGBA.
                *   Model Alignment
                    *   2311.12908 | Diffusion-DPO áp dụng Direct Preference Optimization cho diffusion model, tối ưu trực tiếp trên dữ liệu so sánh người dùng/AI.
                *   Evaluation
                    *   2311.10708 | SelfEval chuyển đổi diffusion model thành bộ phân loại để đánh giá text-image alignment trên ảnh thực, không cần huấn luyện thêm.
                    *   2311.04287 | HEIM là khung đánh giá toàn diện (12 khía cạnh, 62 kịch bản, 25 chỉ số) cho mô hình text-to-image, kết hợp đánh giá tự động và con người.
            *   Other Generative Models
                *   2311.01462 | Idempotent Generative Networks (IGN) là mạng sinh một bước dựa trên tính chất idempotent, với loss reconstruction, idempotent và tightening.
        *   3D Content Generation & Understanding
            *   Text-to-3D & Image-to-3D
                *   Feed-forward & Fast Generation
                    *   2311.08403 | Instant3D là mạng feedforward (decoder triplane với cross-attention, style injection) sinh 3D tức thời (<1s) từ text, giám sát bằng SDS yếu và adaptive Perp-Neg.
                    *   2311.06214 | Instant3D (khác) là khung hai giai đoạn: fine-tune SDXL sinh multi-view grid, sau đó reconstructor transformer (ViT encoder, image-to-triplane decoder) sinh NeRF.
                *   Diffusion-based & Optimization
                    *   2311.07885 | One-2-3-45++ sinh ảnh đa góc nhìn nhất quán (ghép 6 ảnh vào khung), sau đó dùng diffusion 3D (SDF, màu) và tinh chỉnh texture (TensorRF).
                    *   2311.11284 | LucidDreamer (khác Instant3D) sử dụng Interval Score Matching (ISM) và DDIM inversion thay SDS cho text-to-3D Gaussian Splatting.
                    *   2311.10123 | MetaDreamer tối ưu NeRF hai giai đoạn: hình học (Zero123XL, SAM, MiDaS) và kết cấu (diffusion 2D + LoRA, opacity regularization).
            *   Neural Radiance Fields (NeRF) & Gaussian Splatting
                *   Reconstruction & Representation
                    *   2311.04400 | LRM là transformer encoder-decoder quy mô lớn dự đoán NeRF (triplane) trực tiếp từ ảnh đơn, với image-to-triplane decoder và modulation camera.
                    *   2311.12024 | PF-LRM là transformer một luồng tích hợp token 3D triplane và token 2D patch, dự đoán pose qua PnP solver khả vi và shape qua distillation.
                    *   2311.12775 | SuGaR đề xuất regularization cho Gaussian song song bề mặt, thuật toán trích xuất lưới từ level set và refinement tùy chọn.
                *   Acceleration & Real-time Rendering
                    *   2311.10091 | NeuS với kernel không gian-biến thiên, trích xuất vỏ lưới adaptive shell và narrow-band rendering cho hiển thị NeRF thời gian thực.
                    *   2311.02542 | VR-NeRF giới thiệu không gian màu PQ cho HDR NeRF, mip-mapping lưới đa cấp, pruning occupancy grid và renderer đa GPU cho VR real-time.
            *   Dynamic & Animatable 3D
                *   2311.08581 | D3GA sử dụng lồng tetrahedral biến dạng để điều khiển Gaussian splats, pipeline đa lớp cho avatar (thân, mặt, trang phục) và MLP nhẹ cho shading.
                *   2311.12198 | PhysGaussian tích hợp 3D Gaussian kernels với cơ học liên tục (MPM) để mô phỏng động học Newton, xoay SH và cập nhật hiệp phương sai gia tăng.
                *   2311.02077 | EmerNeRF phân tách scene thành trường tĩnh và động self-supervised, khai phóng emergent scene flow và dynamic density regularization.
        *   Video Generation & Synthesis
            *   Diffusion Models
                *   2311.10982 | PixelDance là latent diffusion điều kiện trên text, khung đầu và khung cuối, với kỹ thuật huấn luyện/suy luận đặc biệt cho khung cuối.
                *   2311.13073 | FusionFrames là pipeline T2V hai giai đoạn (keyframe generation với temporal blocks tách biệt + latent interpolation U-Net hiệu quả).
                *   2311.04145 | I2VGen-XL là cascaded I2V model với base stage (hierarchical encoder, detail encoder) và refinement stage (VLDM riêng, SDEdit).
                *   2311.10709 | Emu Video phân tách T2V thành sinh ảnh đầu (T2I) rồi sinh video (điều kiện ảnh đầu, mask), với noise schedule tùy chỉnh và nội suy khung hình.
                *   2311.12631 | GPT4Motion sử dụng GPT-4 sinh script Blender mô phỏng vật lý, sau đó dùng edge/depth từ Blender làm điều kiện cho SDXL (ControlNet) với cross-frame attention.
        *   Motion Synthesis
            *   2311.07446 | Story-to-Motion: Text-Driven Motion Scheduler (LLM), Text-based Motion Retrieval và Progressive Mask Transformer để sinh chuyển động nhân vật dài từ truyện.
        *   Image Segmentation & Understanding
            *   2311.13601 | DINOv là framework visual in-context prompting cho generic/referring segmentation và open-set detection, với Prompt Encoder và PromptClassifier (pivoted loss).
        *   Cross-modal Representation & Interfacing
            *   2311.00618 | De-Diffusion Autoencoder sử dụng T2I diffusion làm decoder để mã hóa ảnh thành latent text (CLIP BPE tokens) có thể đọc được.
        *   Foundation Model Evaluation
            *   2310.19909 | "Battle of the Backbones" (BoB) đánh giá và so sánh các backbone thị giác (supervised, SSL, vision-language) trên nhiều tác vụ.
    *   Audio & Speech Processing
        *   Speech Synthesis & Voice Conversion
            *   2311.12454 | HierSpeech++ là VAE phân cấp song song cho TTS/VC zero-shot, với Dual-Audio Acoustic Encoder, Source-Filter Multi-Path Semantic Encoder và BiT-Flow.
            *   2311.00945 | E3 TTS là mô hình diffusion end-to-end không tự hồi quy, nhận trực tiếp văn bản và sinh sóng âm, với UNet 1D và adaptive CNN kernel.
        *   Automatic Speech Recognition (ASR)
            *   2311.00430 | Distil-Whisper sử dụng pseudo-labelled data quy mô lớn, kiến trúc student (encoder đóng băng, decoder 2 tầng) và KD cho ASR.
        *   Audio Generation & Editing
            *   2311.07069 | Music ControlNet bổ sung điều khiển thời gian thay đổi (melody, dynamics, rhythm) vào T2M diffusion qua nhánh adaptor và masking.
            *   2311.00613 | Khung sampling-time guidance kết hợp reconstruction/classification loss cho diffusion âm thanh, hỗ trợ audio prompt editing.
            *   2311.08667 | EDMSound là mô hình diffusion trên phổ phức (thực, ảo) với Efficient U-Net và DPM-Solver/EI, loại bỏ vocoder.
    *   Robotics & Embodied AI
        *   Robot Skill Learning & Simulation
            *   2311.01455 | RoboGen là pipeline tự động (propose-generate-learn) dùng LLM/VLM tạo nhiệm vụ, cảnh mô phỏng và huấn luyện kỹ năng robot.
        *   Robot Navigation & Planning
            *   2311.06430 | GOAT (Goal-Oriented Autonomous Tasking) sử dụng instance-aware semantic memory, phát hiện mở-vựng và đối sánh ảnh cho điều hướng đa mục tiêu.
            *   2311.05997 | JARVIS-1 là MLM tích hợp MineCLIP, với interactive planning (self-check, self-explain), multimodal memory và self-instruct/improve cho Minecraft.
    *   Machine Learning Systems
        *   Compiler Abstractions & Optimization
            *   2311.02103 | Relax là AOT compilation framework với cross-level IR, symbolic shape annotations và dynamic shape-aware optimizations cho ML models.
        *   Efficient Convolution Algorithms
            *   2311.05908 | FlashFFTConv sử dụng Monarch decomposition, partial/frequency-sparse convolutions và cost model cho FFT convolution hiệu quả trên GPU.
    *   Other
        *   2311.04931 | GPT4All là hệ sinh thái LLM mã nguồn mở, bao gồm dữ liệu, mã nguồn huấn luyện, mô hình nén và API/GUI.
        *   2311.11077 | Adapters là thư viện PEFT tự chứa, tích hợp 10 phương pháp adapter và khối thành phần kết hợp (Stack, Fuse, Split) cho fine-tuning module-hóa.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                          | Đột phá                                                                                                                                  | Ảnh hưởng                                                                                                                                                              |
    | :--- | :-------- | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2311.10770 | UltraFastBERT, Fast Feedforward, Conditional Execution  | Kiến trúc BERT chỉ kích hoạt ~0.3% neuron FFN mỗi lượt suy luận, tăng tốc 40-78x mà vẫn giữ >96% hiệu năng GLUE.                             | Mở ra hướng mới cho LLM siêu hiệu quả, đặc biệt quan trọng cho triển khai trên thiết bị biên hoặc các ứng dụng đòi hỏi độ trễ cực thấp.                                  |
    | 2    | 2311.06242 | Florence-2, Unified Vision-Language, Location Tokens    | Mô hình seq2seq thống nhất cho nhiều tác vụ thị giác-ngôn ngữ (classification, captioning, detection, grounding, segmentation) bằng location tokens. | Đơn giản hóa kiến trúc MLLM, cho phép học đa tác vụ hiệu quả hơn và tạo ra các foundation model thị giác-ngôn ngữ tổng quát hơn.                                      |
    | 3    | 2311.05556 | LCM-LoRA, Diffusion Acceleration, Neural PF-ODE Solver  | Chưng cất Latent Consistency Model thành LoRA plug-and-play, cho phép tăng tốc các mô hình SD fine-tuned mà không cần huấn luyện lại.         | Dân chủ hóa việc tăng tốc các mô hình diffusion lớn, giúp người dùng dễ dàng áp dụng các kỹ thuật sinh ảnh nhanh mà không cần tài nguyên tính toán lớn.                 |
    | 4    | 2311.11045 | Orca 2, Cautious Reasoning, Prompt Erasure              | Dạy LLM nhỏ tự chọn chiến lược giải quyết vấn đề (direct answer vs. step-by-step) bằng Cautious Reasoning và Prompt Erasure.                 | Cải thiện đáng kể khả năng reasoning của các LLM nhỏ, giúp chúng trở nên hữu ích hơn trong các tác vụ phức tạp mà không cần scale up mô hình.                             |
    | 5    | 2311.04400 | LRM (Large Reconstruction Model), Single-Image to NeRF  | Transformer encoder-decoder quy mô lớn dự đoán trực tiếp NeRF (triplane) từ một ảnh duy nhất, end-to-end, không cần 3D regularization.     | Bước tiến lớn trong việc tạo 3D nhanh chóng và chất lượng cao từ ảnh đơn, có tiềm năng ứng dụng rộng rãi trong VR/AR, game, và robotics.                               |
    | 6    | 2311.12908 | Diffusion-DPO, Direct Preference Optimization, Alignment | Mở rộng DPO cho diffusion models, cho phép fine-tune trực tiếp bằng human/AI feedback mà không cần reward model.                            | Đơn giản hóa quá trình alignment cho diffusion models, giúp tạo ra hình ảnh phù hợp hơn với sở thích người dùng và các tiêu chí an toàn.                               |
    | 7    | 2311.03285 | S-LoRA, Unified Paging, Efficient LoRA Serving          | Hệ thống serving hiệu quả cho nhiều LoRA adapter bằng Unified Paging (KV cache + LoRA weights) và custom CUDA kernels.                     | Giải quyết thách thức phục vụ đồng thời nhiều người dùng với các LoRA tùy biến khác nhau, quan trọng cho việc cá nhân hóa LLM trên quy mô lớn.                         |
    | 8    | 2311.01455 | RoboGen, Automated Robot Learning, LLM/VLM for Simulation | Pipeline tự động dùng LLM/VLM để đề xuất nhiệm vụ, sinh cảnh mô phỏng và hàm thưởng, huấn luyện kỹ năng robot đa dạng.                     | Tự động hóa quy trình tạo dữ liệu và huấn luyện robot trong mô phỏng, đẩy nhanh tiến độ phát triển robot có khả năng học hỏi và thích ứng.                               |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2311.10770 – Fast Feedforward Networks & Conditional Matrix Multiplication (CMM) – Suy nghĩ:** Thay thế FFN dày đặc bằng cây FFF và CMM là một đột phá kiến trúc, giảm độ phức tạp tính toán từ O(n) xuống O(log n) cho lớp feedforward, tiềm năng rất lớn cho LLM hiệu quả.
    *   **2311.06242 – Location tokens & FLD-5B data engine – Suy nghĩ:** Biểu diễn tọa độ vùng bằng token và quy trình data engine tự động tạo dữ liệu đa tác vụ quy mô lớn là chìa khóa cho mô hình VLM thống nhất như Florence-2.
    *   **2311.05556 – LCM-LoRA (LoRA distillation for Latent Consistency Models) – Suy nghĩ:** Chưng cất LCM thành LoRA là một cách thông minh để tạo module tăng tốc plug-and-play, kết hợp tính linh hoạt của LoRA và tốc độ của LCM.
    *   **2311.11045 – Cautious Reasoning & Prompt Erasure – Suy nghĩ:** Dạy LLM "học cách học" bằng cách tự chọn chiến lược và ẩn đi hướng dẫn chi tiết là một bước tiến quan trọng để LLM nhỏ có khả năng reasoning tốt hơn.
    *   **2311.10982 – PixelDance (Latent diffusion với điều kiện khung đầu & cuối) & Chiến lược suy luận τ-step cho khung cuối – Suy nghĩ:** Điều khiển video diffusion bằng cả khung đầu và cuối, cùng chiến lược áp dụng điều kiện khung cuối có giới hạn thời gian, giúp tạo video động và nhất quán hơn.
    *   **2311.13073 – FusionFrames (Separate temporal blocks & Latent interpolation U-Net) – Suy nghĩ:** Tách khối temporal riêng và U-Net nội suy latent theo nhóm khung là những cải tiến kiến trúc hiệu quả cho T2V, giảm chi phí và tăng chất lượng.
    *   **2311.10093 – Quy trình lặp (sinh ảnh -> DINOv2 embedding -> K-Means clustering -> TI+LoRA personalization) cho consistent character – Suy nghĩ:** Vòng lặp tự cải thiện để tạo nhân vật nhất quán từ text mà không cần ảnh tham chiếu là một giải pháp rất sáng tạo.
    *   **2311.00430 – Distil-Whisper (Encoder đóng băng, decoder 2 tầng từ teacher) & KD với pseudo-label lọc WER – Suy nghĩ:** Kiến trúc student đơn giản và chiến lược KD hiệu quả cho ASR, đặc biệt là khả năng duy trì zero-shot OOD.
    *   **2311.13384 – LucidDreamer (Dreaming: inpainting + lifting; Alignment: di chuyển theo tia nhìn + nội suy mask) cho 3D scene – Suy nghĩ:** Pipeline tạo cảnh 3D đa miền bằng cách lặp lại quá trình "mơ" (inpainting, lifting) và "căn chỉnh" point cloud là một cách tiếp cận mới lạ.
    *   **2311.04400 – LRM (Image-to-triplane decoder với cross-attention & camera modulation) – Suy nghĩ:** Kiến trúc Transformer dự đoán trực tiếp triplane NeRF từ ảnh đơn với cơ chế điều kiện camera tinh vi.
    *   **2311.12908 – Diffusion-DPO (Mở rộng DPO cho multi-step MDP của diffusion) & Chiến lược sub-segments – Suy nghĩ:** Áp dụng thành công DPO cho diffusion model mà không cần reward model là một bước tiến quan trọng trong alignment mô hình sinh.
    *   **2311.05437 – LLaVA-Plus (Thought-Action-Value format & Skill repository) – Suy nghĩ:** Định dạng thống nhất cho tool use và kho kỹ năng giúp LMM học cách lập kế hoạch và sử dụng công cụ hiệu quả hơn.
    *   **2311.09257 – UFOGen (Dự đoán x0 trực tiếp, sampling x(t-1) theo q(x(t-1)|x0')) & Reconstruction loss ở x0 – Suy nghĩ:** Thay đổi cách tham số hóa và hàm loss trong diffusion-GAN hybrid giúp đạt được sinh ảnh một bước chất lượng cao.
    *   **2311.08581 – D3GA (Cage-based deformation cho Gaussian splats & Pipeline compositional đa lớp) – Suy nghĩ:** Điều khiển avatar 3D bằng biến dạng lồng tetrahedral cho Gaussian splats và kiến trúc đa lớp là một hướng đi mới cho avatar động chất lượng cao.
    *   **2311.13600 – ZipLoRA (Hệ số kết hợp cột, pruning, orthogonality regularization) – Suy nghĩ:** Phương pháp hợp nhất LoRA nội dung và phong cách một cách thông minh, giữ được đặc tính của từng LoRA.
    *   **2311.08403 – Instant3D (Decoder triplane với style injection & adaptive Perp-Neg) – Suy nghĩ:** Kiến trúc feed-forward sinh 3D tức thời với cơ chế điều kiện text và style injection hiệu quả, adaptive Perp-Neg giảm Janus.
    *   **2311.07069 – Music ControlNet (Nhánh adaptor với zero convolution & MLP ánh xạ control vào spectrogram) & Masking cục bộ/gián đoạn – Suy nghĩ:** Mở rộng ControlNet cho điều khiển âm nhạc theo thời gian thay đổi với kiến trúc adaptor và chiến lược masking linh hoạt.
    *   **2311.11829 – System 2 Attention (S2A) (LLM tái tạo ngữ cảnh liên quan) – Suy nghĩ:** Sử dụng LLM như một bước tiền xử lý để lọc nhiễu và tập trung vào thông tin quan trọng trong ngữ cảnh, một dạng "meta-attention".
    *   **2311.07885 – One-2-3-45++ (Sinh ảnh đa góc nhìn nhất quán dạng grid & Diffusion 3D coarse-to-fine) – Suy nghĩ:** Kết hợp diffusion 2D (sinh multi-view grid) và diffusion 3D (SDF, màu) để tạo mesh 3D nhanh và chất lượng.
    *   **2311.11501 – MultiLoRA (Mở rộng ngang LoRA song song & Scaling học được & Khởi tạo Kaiming) – Suy nghĩ:** Phân tán cập nhật qua nhiều module LoRA song song giúp cải thiện khả năng biểu diễn đa nhiệm so với LoRA đơn.
    *   **2311.05997 – JARVIS-1 (Interactive planning với self-check/explain & Multimodal memory & Self-instruct/improve) – Suy nghĩ:** Khung agent toàn diện cho Minecraft, kết hợp nhiều cơ chế tự học và cải thiện liên tục.
    *   **2311.01282 – FlashDecoding++ (Softmax không đồng bộ & Flat GEMM double buffering & Heuristic dataflow adaptation) – Suy nghĩ:** Các tối ưu hóa kernel-level sâu cho attention và GEMM, giúp tăng tốc LLM inference đáng kể.
    *   **2311.09277 – Contrastive Chain-of-Thought (Cung cấp minh họa đúng/sai & Tự động tạo minh họa sai) – Suy nghĩ:** Dạy LLM học từ lỗi bằng cách cung cấp ví dụ phản chứng trong CoT prompting.
    *   **2311.03356 – GLaMM (Visual Expert riêng cho token ảnh & Token <SEG> và L-P projection cho segmentation) & GranD dataset pipeline – Suy nghĩ:** Kiến trúc MLLM hỗ trợ grounding pixel-level và quy trình tạo dataset đa hạt nhân quy mô lớn.
    *   **2311.04145 – I2VGen-XL (Cascaded: base stage với hierarchical/detail encoder & refinement stage với VLDM và SDEdit) – Suy nghĩ:** Pipeline hai giai đoạn tách biệt semantic và chất lượng hình ảnh cho I2V, sử dụng ảnh tĩnh làm điều kiện chính.
    *   **2311.04219 – OtterHD-8B (Kiến trúc Fuyu-8B không image encoder, patch ánh xạ trực tiếp vào decoder) & Dynamic training resolution – Suy nghĩ:** Kiến trúc MLLM xử lý ảnh độ phân giải linh hoạt bằng cách bỏ encoder ảnh và huấn luyện động.
    *   **2311.04934 – Prompt Cache & Prompt Markup Language (PML) & Scaffolding – Suy nghĩ:** Cơ chế tái sử dụng attention states ở cấp độ module giữa các prompt, giảm TTFT hiệu quả.
    *   **2311.06214 – Instant3D (Fine-tune SDXL sinh multi-view grid & Transformer-based sparse-view reconstructor) – Suy nghĩ:** Khung feed-forward hai giai đoạn khác cho text-to-3D, kết hợp diffusion 2D và reconstructor transformer mạnh mẽ.
    *   **2311.12052 – MagicPose (Appearance Control Model branch & Multi-Source Self-Attention & Pose ControlNet) – Suy nghĩ:** Tách biệt kiểm soát appearance và pose trong human image generation bằng nhánh UNet riêng và multi-source attention.
    *   **2311.05657 – LUMOS (Planning/Grounding/Execution Modules & Annotation conversion GPT-4/4V) – Suy nghĩ:** Framework agent modular với các module LLM fine-tuned riêng biệt và quy trình tạo dữ liệu thông minh.
    *   **2311.03285 – S-LoRA (Unified Paging cho KV cache & LoRA weights & Custom CUDA kernels MBGMM/MBGMV & S-LoRA TP) – Suy nghĩ:** Hệ thống serving hiệu quả cho nhiều LoRA adapter với quản lý bộ nhớ thông minh và kernel tối ưu.
    *   **2311.12454 – HierSpeech++ (Dual-Audio Acoustic Encoder & Source-Filter Multi-Path Semantic Encoder & BiT-Flow & TTV) – Suy nghĩ:** Kiến trúc TTS/VC zero-shot song song hoàn toàn, với nhiều module chuyên biệt cho mã hóa âm thanh, ngữ nghĩa và điều khiển ngữ điệu.
    *   **2311.08401 – FactTune (Reference-free/based truthfulness estimation & DPO for factuality) – Suy nghĩ:** Tự động sinh preference pairs từ đánh giá tính chân thực và dùng DPO để fine-tune LLM, cải thiện factuality.
    *   **2311.01455 – RoboGen (Scene Generation LLM+Objaverse+VLM & Training Supervision Generation LLM) – Suy nghĩ:** Pipeline tự động hóa hoàn toàn việc tạo nhiệm vụ, cảnh mô phỏng và hàm thưởng cho huấn luyện robot.
    *   **2311.13231 – D3PO (Mở rộng DPO cho multi-step MDP của diffusion & Chiến lược sub-segments) – Suy nghĩ:** Một cách tiếp cận khác để áp dụng DPO cho diffusion model, tập trung vào cấu trúc MDP của quá trình denoising.
    *   **2311.07446 – Story-to-Motion (Text-Driven Motion Scheduler LLM & Text-based Motion Retrieval & Progressive Mask Transformer) – Suy nghĩ:** Hệ thống toàn diện để sinh chuyển động nhân vật dài, có thể điều khiển từ văn bản truyện.
    *   **2310.20689 – LEMA (Pipeline tạo dữ liệu mistake-correction GPT-4 & Correction-centric evolution) – Suy nghĩ:** Dạy LLM học từ lỗi bằng cách fine-tune trên dữ liệu sửa lỗi được tạo tự động và mở rộng một cách có chủ đích.
    *   **2311.12775 – SuGaR (Regularization cho Gaussian song song bề mặt & Trích xuất lưới từ level set & Refinement tùy chọn) – Suy nghĩ:** Phương pháp hiệu quả để trích xuất mesh chất lượng cao từ 3D Gaussian Splatting và tinh chỉnh.
    *   **2311.11077 – Adapters library (Composition blocks: Stack, Fuse, Split & ConfigUnion) – Suy nghĩ:** Thư viện PEFT linh hoạt với các khối kết hợp adapter cho phép xây dựng cấu hình fine-tuning phức tạp.
    *   **2311.10794 – Style Tailoring (Tách content/style loss theo bước denoising) & Transparency Module (Decoder RGBA) – Suy nghĩ:** Fine-tune LDM hiệu quả cho style transfer và sinh ảnh sticker trong suốt.
    *   **2311.06783 – Q-Pathway dataset & Q-Instruct (Pipeline GPT-tham gia tạo instruction từ pathway feedback) – Suy nghĩ:** Dataset và quy trình tạo dữ liệu instruction tuning tập trung vào thuộc tính hình ảnh mức thấp.
    *   **2311.10122 – Video-LLaVA (Alignment Before Projection với LanguageBind & Shared projection & Joint training ảnh/video) – Suy nghĩ:** Kiến trúc MLLM hợp nhất biểu diễn ảnh và video hiệu quả bằng cách tiền căn chỉnh và chia sẻ lớp chiếu.
    *   **2311.01615 – FLAP (Masking 1D/2D trên phổ âm thanh & Contrastive + Reconstruction loss & LLM+AED text augmentation) – Suy nghĩ:** Khung học đối比 audio-text hiệu quả, kết hợp masking, reconstruction và tăng cường chú thích.
    *   **2310.20587 – LaMo (Decision Transformer GPT-2 init + LoRA & MLP I/O & Auxiliary language loss) – Suy nghĩ:** Kết hợp LLM tiền huấn luyện với offline RL (Decision Transformer) một cách hiệu quả tham số.
    *   **2311.11243 – AutoStory (Sinh tín hiệu điều khiển dày đặc & Sinh đối tượng nhất quán đa góc nhìn & Story-to-Layout LLM & Bảo tồn nhân vật PEFT) – Suy nghĩ:** Pipeline toàn diện để tự động sinh truyện tranh từ văn bản, với nhiều module sáng tạo.
    *   **2311.10708 – SelfEval (Đảo ngược diffusion và Monte Carlo integration để ước tính p(x|c)) – Suy nghĩ:** Sử dụng chính diffusion model để đánh giá text-image alignment trên ảnh thực, không cần mô hình ngoài.
    *   **2311.06430 – GOAT (Instance-aware semantic memory & Phát hiện mở-vựng + đối sánh ảnh & Bổ sung bộ nhớ liên tục) – Suy nghĩ:** Hệ thống điều hướng robot đa mục tiêu với bộ nhớ ngữ nghĩa theo phiên bản, học dài hạn.
    *   **2311.05908 – FlashFFTConv (Monarch decomposition cho FFT trên tensor cores & Partial/Frequency-sparse convolutions) – Suy nghĩ:** Thuật toán FFT convolution cực kỳ hiệu quả cho chuỗi dài trên GPU.
    *   **2311.04498 – NExT-Chat (pix2emb: embedding vị trí & Token <trigger>/<loc> & Cycle loss) – Suy nghĩ:** Mô hình hóa vị trí bằng embedding liên tục cho visual grounding và segmentation trong MLLM.
    *   **2311.00945 – E3 TTS (Diffusion end-to-end text-to-waveform & UNet 1D với adaptive CNN kernel & KL-based loss) – Suy nghĩ:** Mô hình TTS song song hoàn toàn, sinh trực tiếp sóng âm từ text bằng diffusion.
    *   **2311.12631 – GPT4Motion (GPT-4 sinh script Blender & Kết hợp đa điều kiện ControlNet & Cross-frame attention SDXL) – Suy nghĩ:** Tạo video vật lý bằng cách dùng LLM sinh mã mô phỏng, sau đó dùng output mô phỏng điều khiển diffusion model.
    *   **2311.02077 – EmerNeRF (Phân tách tĩnh-động self-supervised & Emergent scene flow & Dynamic density regularization & Additive PE prior) – Suy nghĩ:** Tái tạo cảnh động 4D với khả năng tự học phân tách và scene flow mà không cần annotation.
    *   **2311.06243 – Orthogonal Butterfly (BOFT) & Multiplicative Dropout – Suy nghĩ:** Phương pháp PEFT dựa trên factorization bướm cho ma trận trực giao, hiệu quả tham số cao.
    *   **2311.04257 – mPLUG-Owl2 (Modality-Adaptive Module MAM & Huấn luyện hai giai đoạn text-only/multimodal) – Suy nghĩ:** Kiến trúc MLLM với module thích ứng theo modal, cân bằng giữa học đặc trưng riêng và chung.
    *   **2311.02103 – Relax (Cross-level IR & Symbolic shape annotations & Dynamic shape-aware optimizations) – Suy nghĩ:** Framework biên dịch AOT mạnh mẽ cho các mô hình ML có shape động.

4.  **GAPS_AND_OPPORTUNITIES**
    *   **Reasoning và độ tin cậy của LLM/MLLM:**
        *   *Gaps:* Khả năng suy luận đa bước, phức tạp, và nhất quán vẫn là thách thức (XOT, Contrastive CoT, LEMA). Hiện tượng "hallucination" và sai lệch thông tin còn phổ biến, đặc biệt trong MLLM và các tác vụ đòi hỏi kiến thức chuyên ngành (FactTune, GAIA, GPQA). Việc LLM tự đánh giá và sửa lỗi còn hạn chế (Orca 2).
        *   *Opportunities:* Phát triển các kiến trúc và phương pháp huấn luyện mới tăng cường khả năng reasoning logic và因果. Các kỹ thuật prompting tiên tiến hơn, có thể học được hoặc tự động tối ưu (PE2). Nghiên cứu sâu hơn về cơ chế tự giám sát và tự sửa lỗi hiệu quả. Xây dựng các benchmark "không thể gian lận" (Google-proof như GPQA) để đánh giá năng lực thực sự.
    *   **Hiệu quả và khả năng mở rộng của Generative AI (đặc biệt 3D/Video):**
        *   *Gaps:* Sinh 3D/Video chất lượng cao, nhất quán, điều khiển được và dài vẫn rất tốn kém tài nguyên và thời gian (Instant3D, One-2-3-45++, PixelDance, FusionFrames, I2VGen-XL, Emu Video). Việc đảm bảo tính nhất quán về mặt vật lý và ngữ nghĩa trong các cảnh động phức tạp còn khó khăn (PhysGaussian, GPT4Motion).
        *   *Opportunities:* Kiến trúc diffusion/GAN hiệu quả hơn cho dữ liệu 3D/video. Các phương pháp kết hợp học sâu với mô phỏng vật lý. Kỹ thuật nén và tăng tốc cho các mô hình sinh lớn. Tận dụng LLM/VLM để lập kế hoạch và điều khiển quá trình sinh 3D/video một cách thông minh hơn.
    *   **Tích hợp đa phương thức sâu và linh hoạt:**
        *   *Gaps:* Nhiều MLLM vẫn dựa trên việc "ghép nối" các module chuyên biệt thay vì tích hợp sâu từ đầu (Florence-2, TEAL, CogVLM, mPLUG-Owl2 là các nỗ lực cải thiện). Việc xử lý đồng thời nhiều luồng thông tin từ các modal khác nhau và hiểu mối quan hệ phức tạp giữa chúng còn hạn chế. Grounding ở mức độ pixel và tương tác đa lượt với các đối tượng cụ thể trong ảnh/video vẫn cần cải thiện (GLaMM, NExT-Chat, PG-Video-LLaVA).
        *   *Opportunities:* Kiến trúc MLLM "multimodal-native" thực sự. Các phương pháp pre-training hiệu quả trên dữ liệu đa phương thức quy mô lớn, có thể tận dụng dữ liệu không cặp đôi. Kỹ thuật grounding và co-referencing mạnh mẽ hơn giữa các modal.
    *   **Cá nhân hóa và điều khiển mô hình sinh:**
        *   *Gaps:* Việc tạo ra nội dung theo phong cách hoặc với các đối tượng/nhân vật cụ thể một cách nhất quán và dễ dàng vẫn là một thách thức (Consistent Character, ZipLoRA, Style Tailoring, Concept Sliders). Các phương pháp hiện tại thường đòi hỏi fine-tuning hoặc dữ liệu chuyên biệt.
        *   *Opportunities:* Các kỹ thuật PEFT mới cho phép cá nhân hóa hiệu quả hơn với ít dữ liệu hơn. Phương pháp điều khiển mô hình sinh một cách trực quan và linh hoạt hơn, có thể thông qua ngôn ngữ tự nhiên hoặc tương tác trực tiếp.
    *   **Agent tự trị và tương tác với môi trường:**
        *   *Gaps:* Xây dựng các agent có khả năng lập kế hoạch dài hạn, học hỏi từ tương tác, và thích ứng với môi trường động, phức tạp còn ở giai đoạn đầu (JARVIS-1, RoboGen, GOAT, ADAPT, LUMOS). Việc tích hợp tool-use một cách hiệu quả và an toàn vẫn là bài toán mở.
        *   *Opportunities:* Framework agent tổng quát hơn, có khả năng học các kỹ năng mới và tự cải thiện. Nghiên cứu về bộ nhớ dài hạn và khả năng trừu tượng hóa cho agent. Tích hợp các nguyên tắc an toàn và đạo đức vào thiết kế agent.
    *   **Hiệu quả tính toán và triển khai trên thiết bị biên:**
        *   *Gaps:* Chi phí huấn luyện và inference của các mô hình lớn vẫn rất cao. Việc triển khai các mô hình này trên thiết bị có tài nguyên hạn chế còn nhiều khó khăn (UltraFastBERT, FlashDecoding++, S-LoRA, FlashFFTConv, Relax).
        *   *Opportunities:* Các thuật toán và kiến trúc phần cứng mới cho AI hiệu quả năng lượng. Kỹ thuật nén mô hình (quantization, pruning, distillation) tiên tiến hơn. Framework biên dịch và tối ưu hóa chuyên biệt cho on-device AI.
    *   **Dữ liệu: Chất lượng, Quy mô và Tự động hóa:**
        *   *Gaps:* Chất lượng và sự đa dạng của dữ liệu huấn luyện vẫn là yếu tố then chốt. Việc tạo và tinh lọc dữ liệu quy mô lớn tốn nhiều công sức (FLD-5B, CAPSFUSION, LVIS-INSTRUCT4V, Q-Instruct, TÜLU-V2-mix).
        *   *Opportunities:* Các phương pháp data generation/augmentation thông minh hơn, có thể tự động điều chỉnh theo nhu liệu của mô hình. Kỹ thuật đánh giá và cải thiện chất lượng dữ liệu tự động.
    *   **Đánh giá AI một cách toàn diện và đáng tin cậy:**
        *   *Gaps:* Các benchmark hiện tại thường chưa đủ bao quát hoặc dễ bị "gaming" (GAIA, GPQA, IFEval, HEIM, MEGAVERSE là các nỗ lực cải thiện). Việc đánh giá các khía cạnh "mềm" như sáng tạo, tính nhất quán, hoặc các rủi ro tiềm ẩn còn khó khăn.
        *   *Opportunities:* Phát triển các phương pháp đánh giá mới, kết hợp cả định lượng và định tính, tự động và con người. Xây dựng các môi trường tương tác để đánh giá agent trong các kịch bản phức tạp.

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Self-Supervised Emergent Physics Engine for Animatable 3D Avatars**
    *   **Motivation:** Các phương pháp tạo avatar 3D động (D3GA 2311.08581, PhysGaussian 2311.12198) đang tiến bộ nhưng thường dựa vào mô phỏng vật lý cổ điển hoặc học từ dữ liệu mocap lớn. EmerNeRF (2311.02077) cho thấy khả năng "khai phóng" scene flow tự giám sát.
    *   **Key Novelty:** Huấn luyện một mô hình NeRF/Gaussian Splatting cho avatar người có khả năng tự học một "động cơ vật lý ngầm" (emergent physics engine) từ dữ liệu video người chuyển động đa dạng mà không cần nhãn vật lý hay mocap chi tiết. Mục tiêu là avatar có thể phản ứng tự nhiên với các lực tác động ảo hoặc tương tác với các đối tượng ảo khác.
    *   **Approach:**
        1.  Sử dụng kiến trúc NeRF/Gaussian Splatting động có khả năng biểu diễn trường biến dạng (deformation field) và trường gia tốc/vận tốc ngầm.
        2.  Huấn luyện trên một bộ dữ liệu lớn video người thực hiện các hành động đa dạng (ví dụ: đi, chạy, nhảy, tương tác đồ vật) chỉ với giám sát tái tạo hình ảnh (photometric loss).
        3.  Thiết kế các hàm loss bổ sung khuyến khích tính nhất quán vật lý ngầm, ví dụ: temporal consistency của deformation, smoothness của acceleration field, hoặc thậm chí là một loss "energy conservation" xấp xỉ.
        4.  Trong quá trình inference, có thể "chọc" vào avatar bằng cách áp dụng một trường lực ảo lên một phần của latent space hoặc deformation field và quan sát phản ứng.
    *   **Dataset + Metrics:** Dataset video người chuyển động lớn (Human3.6M, AMASS, hoặc video từ YouTube). Metrics: Chất lượng tái tạo hình ảnh (PSNR, SSIM), tính nhất quán thời gian, và đánh giá định tính/định lượng về tính hợp lý vật lý của các phản ứng khi có tương tác ảo (ví dụ: user study, so sánh với mô phỏng vật lý thực).
    *   **Risk/Feasibility:** Rất cao (Moon-shot). Việc học vật lý một cách hoàn toàn tự giám sát từ video 2D là cực kỳ thách thức. Đảm bảo tính ổn định và tránh các hành vi phi vật lý là khó.

    ✨ **Idea 2: Iterative Co-Refinement of LLM-Generated Code and Formal Specifications**
    *   **Motivation:** LLM ngày càng giỏi sinh mã (Code Llama, Ziya2) nhưng vẫn thường tạo ra lỗi hoặc không hoàn toàn tuân thủ yêu cầu. LOGIPT (2311.06158) cho thấy LLM có thể học mô phỏng bộ giải logic.
    *   **Key Novelty:** Một vòng lặp tương tác giữa LLM (sinh mã) và một hệ thống kiểm chứng hình thức (formal verification) hoặc một LLM khác chuyên về sinh/kiểm tra đặc tả hình thức (formal specifications). LLM sinh mã, hệ thống đặc tả cố gắng sinh/kiểm tra đặc tả cho mã đó. Nếu có xung đột hoặc không thể tạo đặc tả, feedback được dùng để LLM tinh chỉnh lại mã, và ngược lại.
    *   **Approach:**
        1.  Người dùng cung cấp yêu cầu bằng ngôn ngữ tự nhiên.
        2.  **Code Generation LLM:** Sinh một phiên bản mã ban đầu.
        3.  **Specification LLM/Tool:**
            *   Cố gắng sinh đặc tả hình thức (ví dụ: pre/post conditions, invariants) cho mã đã sinh.
            *   Hoặc, nếu có đặc tả ban đầu, kiểm tra xem mã có tuân thủ đặc tả không.
        4.  **Feedback & Refinement:**
            *   Nếu mã không thể đặc tả hóa hoặc vi phạm đặc tả, thông tin lỗi/xung đột được cung cấp lại cho Code Generation LLM để sửa mã.
            *   Nếu đặc tả sinh ra không khớp với ý định ban đầu (có thể được người dùng xác nhận), thông tin này được dùng để Specification LLM sửa đặc tả.
        5.  Lặp lại bước 2-4 cho đến khi mã và đặc tả nhất quán và đáp ứng yêu cầu.
    *   **Dataset + Metrics:** Sử dụng các benchmark sinh mã có sẵn (HumanEval, MBPP) và mở rộng chúng với các đặc tả hình thức (nếu có thể). Metrics: Tỷ lệ pass các unit test, mức độ tuân thủ đặc tả (có thể đo bằng công cụ kiểm chứng hình thức), và đánh giá của con người về tính đúng đắn/chất lượng của cả mã và đặc tả.
    *   **Risk/Feasibility:** Cao. Sinh và kiểm chứng đặc tả hình thức tự động là bài toán khó. Vòng lặp có thể không hội tụ hoặc tốn nhiều tài nguyên. Cần LLM đủ mạnh để hiểu và hành động dựa trên feedback từ hệ thống đặc tả.

    ✨ **Idea 3: Personalized Text-to-Image Generation with Iterative Feedback from "Self-Critiquing" Diffusion Models**
    *   **Motivation:** Cá nhân hóa T2I (Consistent Character 2311.10093, ZipLoRA 2311.13600) và tối ưu prompt (NeuroPrompts 2311.12229) là các hướng quan trọng. SelfEval (2311.10708) cho thấy diffusion model có thể tự đánh giá.
    *   **Key Novelty:** Kết hợp ý tưởng cá nhân hóa (ví dụ: học một concept nhân vật) với một vòng lặp tự cải thiện, trong đó diffusion model không chỉ sinh ảnh mà còn sử dụng khả năng "tự đánh giá" (như SelfEval) để cung cấp feedback (ví dụ: "nhân vật chưa đủ giống ảnh tham chiếu X ở đặc điểm Y") cho một LLM tối ưu prompt, từ đó tinh chỉnh lại quá trình sinh ảnh cho đến khi đạt được sự nhất quán mong muốn.
    *   **Approach:**
        1.  Người dùng cung cấp ảnh tham chiếu của một đối tượng/nhân vật và một prompt ban đầu.
        2.  **Personalization:** Áp dụng một phương pháp PEFT (ví dụ: LoRA) để học concept từ ảnh tham chiếu.
        3.  **Iterative Generation & Critique:**
            *   Sử dụng diffusion model đã cá nhân hóa và prompt hiện tại để sinh một tập ảnh.
            *   Sử dụng một phiên bản "self-critiquing" của diffusion model (hoặc một VLM được huấn luyện riêng) để so sánh các ảnh sinh ra với ảnh tham chiếu và prompt, xác định các điểm không nhất quán hoặc chưa đạt yêu cầu (ví dụ: "màu tóc sai", "phong cách chưa đúng").
            *   Chuyển feedback này thành ngôn ngữ tự nhiên.
        4.  **Prompt Refinement LLM:** Nhận prompt hiện tại và feedback từ bước critique, đề xuất một prompt mới tốt hơn.
        5.  Lặp lại bước 3-4 cho đến khi ảnh sinh ra đạt chất lượng và độ nhất quán mong muốn (có thể được người dùng xác nhận hoặc qua một ngưỡng tự động).
    *   **Dataset + Metrics:** Không cần dataset huấn luyện lớn cho vòng lặp, nhưng cần dữ liệu để huấn luyện VLM làm critic (nếu dùng). Metrics: Đánh giá của con người về độ nhất quán của concept, mức độ tuân thủ prompt, và chất lượng thẩm mỹ. Có thể dùng CLIP score hoặc LPIPS so với ảnh tham chiếu.
    *   **Risk/Feasibility:** Trung bình đến Cao. Khả năng "tự phê bình" của diffusion model cần được phát triển mạnh mẽ hơn. LLM tối ưu prompt cần hiểu rõ feedback trực quan. Vòng lặp có thể tốn kém.

    ✨ **Idea 4: Dynamic Task Decomposition and Tool Selection for Embodied AI using Hierarchical Cautious Reasoning**
    *   **Motivation:** Các agent embodied (JARVIS-1 2311.05997, RoboGen 2311.01455) cần khả năng lập kế hoạch và sử dụng công cụ linh hoạt. Orca 2 (2311.11045) giới thiệu Cautious Reasoning cho LLM. ADAPT (2311.05772) đề xuất phân tách tác vụ "as-needed".
    *   **Key Novelty:** Một agent embodied sử dụng kiến trúc phân cấp với nhiều LLM chuyên biệt. Một "Meta-Planner" LLM cấp cao sử dụng Cautious Reasoning để quyết định chiến lược tổng thể và khi nào cần phân rã một tác vụ lớn. Các "Sub-Task Solver" LLM cấp thấp hơn, cũng được trang bị Cautious Reasoning, sẽ quyết định cách thực thi một sub-task cụ thể, bao gồm việc chọn công cụ (từ một skill repository như LLaVA-Plus 2311.05437) hoặc thực hiện hành động nguyên thủy. Việc phân rã và chọn công cụ diễn ra động, dựa trên self-evaluation và phản hồi từ môi trường.
    *   **Approach:**
        1.  **Meta-Planner LLM:** Nhận nhiệm vụ tổng thể. Sử dụng Cautious Reasoning để:
            *   Quyết định xem có cần phân rã nhiệm vụ không.
            *   Nếu có, sinh ra một chuỗi các sub-tasks (có thể với logic AND/OR như ADAPT).
            *   Nếu không, trực tiếp chọn một Sub-Task Solver phù hợp.
        2.  **Sub-Task Solver LLM(s):** Nhận một sub-task. Sử dụng Cautious Reasoning để:
            *   Quyết định xem có thể giải quyết trực tiếp bằng hành động nguyên thủy không.
            *   Nếu không, chọn một hoặc nhiều công cụ từ skill repository.
            *   Sinh các tham số cho công cụ.
        3.  **Execution & Feedback:** Thực thi hành động/công cụ. Phản hồi từ môi trường được cung cấp lại cho cả Sub-Task Solver và Meta-Planner để điều chỉnh kế hoạch nếu cần.
        4.  Sử dụng Prompt Erasure trong quá trình huấn luyện các LLM này để chúng học cách tự chiến lược hóa.
    *   **Dataset + Metrics:** Sử dụng các môi trường mô phỏng cho embodied AI (AI2-THOR, Habitat, Minecraft). Metrics: Tỷ lệ hoàn thành nhiệm vụ, hiệu quả (số bước, thời gian), khả năng thích ứng với thay đổi.
    *   **Risk/Feasibility:** Cao. Điều phối nhiều LLM phân cấp là phức tạp. Đảm bảo Cautious Reasoning hoạt động hiệu quả ở các cấp độ khác nhau và với các loại quyết định khác nhau (phân rã vs. chọn công cụ) là thách thức.

6.  **READING_LIST**

    *   2311.10770 – UltraFastBERT · Đột phá về kiến trúc FFN cho BERT, tăng tốc inference cực lớn.
    *   2311.06242 – Florence-2 · Kiến trúc VLM thống nhất, xử lý đa tác vụ thị giác-ngôn ngữ bằng location tokens.
    *   2311.05556 – LCM-LoRA · Cách tiếp cận thông minh để tăng tốc diffusion model bằng LoRA plug-and-play.
    *   2311.11045 – Orca 2 · Dạy LLM nhỏ tự chọn chiến lược giải quyết vấn đề, rất quan trọng cho reasoning.
    *   2311.04400 – LRM · Kiến trúc Transformer mạnh mẽ cho single-image to NeRF, feed-forward.
    *   2311.12908 – Diffusion-DPO · Mở rộng DPO cho diffusion model, hướng đi mới cho alignment mô hình sinh.
    *   2311.03285 – S-LoRA · Giải pháp hệ thống hiệu quả cho việc phục vụ đồng thời nhiều LoRA adapter.
    *   2311.01455 – RoboGen · Pipeline tự động hóa toàn diện cho việc tạo nhiệm vụ, cảnh và huấn luyện robot.
    *   2311.04254 – XOT (Everything of Thoughts) · Kết hợp MCTS và RL để tạo luồng suy nghĩ hiệu quả cho LLM.
    *   2311.02103 – Relax · Framework biên dịch AOT mạnh mẽ cho ML model có shape động, rất quan trọng cho LLM.

7.  **META_REFLECTION**
    Tập hợp các bài báo tháng 11 năm 2023 cho thấy sự tiếp nối và đào sâu các xu hướng từ tháng 10, đồng thời xuất hiện một số điểm nhấn mới:
    *   **Hiệu quả LLM vẫn là ưu tiên hàng đầu:** Nhiều nghiên cứu tiếp tục tập trung vào việc tăng tốc LLM inference (UltraFastBERT, FlashDecoding++, FastCoT) và tối ưu hóa quá trình fine-tuning hiệu quả tham số (MultiLoRA, BOFT, Tied-LoRA). Điều này cho thấy nhu cầu thực tế về việc triển khai LLM một cách rộng rãi và tiết kiệm chi phí. S-LoRA giải quyết vấn đề serving nhiều adapter, một bài toán quan trọng khi cá nhân hóa trở nên phổ biến.
    *   **MLLM hướng tới tích hợp sâu hơn và khả năng grounding tốt hơn:** Các kiến trúc MLLM mới (Florence-2, OtterHD-8B, TEAL, CogVLM, mPLUG-Owl2, SPHINX, Video-LLaVA) đang cố gắng tích hợp các modal một cách chặt chẽ hơn, thay vì chỉ ghép nối các encoder riêng lẻ. Khả năng hiểu và tương tác với các vùng cụ thể trong ảnh/video (GLaMM, NExT-Chat, PG-Video-LLaVA) đang được chú trọng.
    *   **Generative AI cho 3D và Video tiếp tục bùng nổ với các phương pháp đa dạng:** Từ text-to-3D (Instant3D, One-2-3-45++, LucidDreamer, MetaDreamer), image-to-3D (LRM, PF-LRM), đến text/image-to-video (PixelDance, FusionFrames, I2VGen-XL, Emu Video, GPT4Motion), lĩnh vực này đang phát triển rất nhanh. Các kỹ thuật mới như Interval Score Matching (LucidDreamer) hay kết hợp mô phỏng vật lý (PhysGaussian, GPT4Motion) cho thấy sự tìm tòi các hướng đi mới.
    *   **Reasoning, Planning và Agent tự trị được đầu tư mạnh mẽ:** Khả năng suy luận (Contrastive CoT, LEMA, LOGIPT, XOT), lập kế hoạch và hành động của agent (LLaVA-Plus, LUMOS, JARVIS-1, RoboGen, ADAPT) là một trọng tâm lớn. Các phương pháp dạy LLM "học cách học" (Orca 2) hay tự cải thiện từ lỗi (LEMA) rất đáng chú ý.
    *   **Alignment và An toàn AI ngày càng được quan tâm thực chất:** Ngoài việc cải thiện tính hữu ích, các nghiên cứu bắt đầu đi sâu vào việc làm cho AI an toàn hơn và đáng tin cậy hơn, ví dụ như tối ưu factuality (FactTune), alignment không cần reward model (Diffusion-DPO, D3PO), và thậm chí là nghiên cứu các hành vi lừa dối tiềm ẩn (2311.07590).
    *   **Tự động hóa và Data-Centric AI tiếp tục là xu hướng:** Việc tự động hóa các quy trình (prompt engineering - PE2, tạo dữ liệu - RoboGen, LVIS-INSTRUCT4V, Q-Instruct, CAPSFUSION) và tập trung vào chất lượng/chiến lược sử dụng dữ liệu (Ziya2, TÜLU-V2-mix) vẫn là chìa khóa để cải thiện hiệu năng mô hình.
    *   **Benchmark và Đánh giá ngày càng toàn diện và thách thức hơn:** Sự ra đời của các benchmark mới, khó hơn, và tập trung vào các năng lực cụ thể (GAIA, GPQA, IFEval, HEIM, MEGAVERSE, MagnifierBench, PPTC) cho thấy lĩnh vực đang đòi hỏi những thước đo khắt khe hơn để đánh giá tiến bộ thực sự của AI, đặc biệt là hướng tới AGI (2311.02462).
    *   **Sự giao thoa giữa các lĩnh vực:** Nhiều bài báo thể hiện sự kết hợp ý tưởng từ các lĩnh vực khác nhau, ví dụ như áp dụng nguyên lý từ LLM cho diffusion model (Diffusion-DPO), kết hợp cơ học liên tục với đồ họa máy tính (PhysGaussian), hay dùng LLM để điều khiển mô phỏng vật lý (GPT4Motion, RoboGen).

    Nhìn chung, tháng 11/2023 chứng kiến sự phát triển mạnh mẽ và đa dạng của AI, với những nỗ lực không ngừng nhằm làm cho AI mạnh mẽ hơn, hiệu quả hơn, đáng tin cậy hơn và dễ tiếp cận hơn. Sự tập trung vào các ứng dụng thực tế và đánh giá khắt khe hơn cũng cho thấy sự trưởng thành của lĩnh vực.
