1.  **TOPIC_TREE**

    *   NLP (Natural Language Processing)
        *   Large Language Models (LLM) & Multimodal Language Models (MLLM)
            *   Model Architecture & Optimization
                *   Efficient Architectures & Inference
                    *   2401.04088 | Mixtral 8x7B giới thiệu kiến trúc Sparse Mixture of Experts (SMoE) với 8 chuyên gia (top-K=2) cho transformer decoder-only, tăng tham số tổng thể mà vẫn giữ chi phí suy luận thấp.
                    *   2401.06066 | DeepSeekMoE đề xuất phân đoạn expert chi tiết và tách biệt expert dùng chung trong kiến trúc MoE để tăng cường chuyên môn hóa.
                    *   2401.07987 | SwitchHead áp dụng Mixture-of-Experts cho self-attention, giảm số attention map và chia sẻ Q/K projections.
                    *   2401.10774 | Medusa giới thiệu nhiều đầu giải mã song song và tree attention để tăng tốc LLM inference, thay thế draft model trong speculative decoding.
                    *   2401.15077 | EAGLE đề xuất Autoregression Head (FC + Transformer decoder) dự đoán đặc trưng tự hồi quy kết hợp shifted token sequence cho speculative sampling.
                    *   2401.04658 | Lightning Attention-2 là thuật toán chú ý tuyến tính nhân quả O(n) với tiling và hybrid computation, tối ưu IO-aware.
                *   Memory Optimization & Long Context
                    *   2401.11514 | Đề xuất windowing FFN cache, row-column bundling và low-rank predictor cho inference LLM trên flash với DRAM hạn chế.
                    *   2401.04985 | SparQ Attention sử dụng query sparsity và mean value reallocation để giảm băng thông attention trong inference LLM.
                    *   2401.06104 | TOVA (Token Omission Via Attention) nén KV cache bằng cách loại bỏ token có attention score thấp nhất ở mỗi bước giải mã.
                    *   2401.03462 | Activation Beacon sử dụng token beacon đặc biệt để nén ngữ cảnh vào kích hoạt Key/Value ở mỗi lớp, hỗ trợ tỷ lệ nén linh hoạt.
                    *   2401.01325 | SelfExtend là phương pháp không cần fine-tuning, sử dụng chú ý hai cấp (neighbor, grouped) với RoPE điều chỉnh để mở rộng ngữ cảnh LLM.
                    *   2401.06951 | E2-LLM huấn luyện LLM một lần trên ngữ cảnh ngắn với scale/position offset augmentation cho RoPE, hỗ trợ suy luận trên nhiều độ dài.
                *   Model Scaling & Initialization
                    *   2401.02385 | TinyLlama huấn luyện mô hình 1.1B trên 3 nghìn tỷ token, với quy trình 3 giai đoạn (basic, continual domain-specific, cooldown).
                    *   2401.02954 | Phân tích quy luật scaling cho LLM (DeepSeek 7B/67B), bao gồm siêu tham số, phân bổ model/data, và sự phụ thuộc vào dataset, đề xuất lịch trình học multi-step.
                    *   2401.00448 | Định nghĩa lại bài toán tối ưu scaling law bao gồm chi phí inference, tìm N và D_tr tối ưu cho loss và tổng chi phí.
                    *   2401.02415 | Block expansion thêm khối identity mới (WO/W3=0) vào LLM, chỉ huấn luyện khối mới trên dữ liệu miền để tăng năng lực mà không quên kiến thức cũ.
            *   Training, Fine-tuning & Adaptation
                *   Instruction Tuning & Data Generation
                    *   2401.00368 | Tạo dữ liệu tổng hợp cho text embedding bằng LLM (GPT-3.5/4) với prompt hai bước (brainstorm tác vụ -> tạo ví dụ) và fine-tune decoder-only LLM.
                    *   2401.14187 | CodeSeaXDataset được tạo bằng framework LLM-based Generator-Discriminator và raw code coreset selection cho 4 tác vụ code.
                    *   2401.17268 | Weaver sử dụng Instruction Backtranslation (tạo prompt từ văn bản chất lượng cao) và Constitutional DPO (tạo cặp preference dựa trên nguyên tắc) cho LLM viết sáng tạo.
                *   Alignment & Safety
                    *   2401.10020 | Self-Rewarding Language Models tự đánh giá phản hồi (LLM-as-a-Judge) tạo AIF cho Iterative DPO, cải thiện cả instruction following và reward modeling.
                    *   2401.01335 | Self-Play Fine-Tuning (SPIN) cho LLM tự cải thiện bằng cách phân biệt phản hồi từ SFT data gốc và phản hồi do chính nó tạo ở bước trước.
                    *   2401.06080 | Đo lường độ mạnh ưa thích bằng ensemble RM, giảm nhiễu dữ liệu (label flipping/smoothing) và đề xuất adaptive margin/contrastive learning/meta-learning cho RM.
                    *   2401.05566 | Huấn luyện backdoor sử dụng CoT ẩn để mô hình hóa lý luận lừa dối, sau đó chưng cất để duy trì hành vi backdoor bền bỉ.
                *   Model Composition & Merging
                    *   2401.02412 | CALM (Composition to Augment Language Models) kết hợp LLM neo và LLM tăng cường (đóng băng) qua linear projections và cross-attention học được.
                *   Low-Resource & Cross-Lingual Adaptation
                    *   2401.01055 | Nghiên cứu chuyển giao năng lực LLM sang ngôn ngữ khác (tiếng Trung), cho thấy instruction tuning hiệu quả hơn further pretraining quy mô lớn.
            *   Reasoning & Problem Solving
                *   Program-Augmented & Tool Use
                    *   2401.04474 | Chain of Code (CoC) kết hợp viết code/pseudocode và mô phỏng thực thi bằng LMulator (Python interpreter + LM) cho reasoning hỗn hợp.
                    *   2401.04398 | Chain-of-Table sử dụng chuỗi bảng biến đổi làm đại diện bước suy luận, LLM lập kế hoạch động và tạo đối số cho thao tác bảng.
                *   Self-Improvement & Feedback
                    *   2401.06585 | ReSTEM là phương pháp self-training (EM) cho RL với ngôn ngữ, dùng tín hiệu thưởng nhị phân và fine-tune base model.
            *   Evaluation & Benchmarking
                *   2401.05561 | TRUSTLLM là benchmark đánh giá toàn diện tính đáng tin cậy của LLM (truthfulness, safety, fairness, robustness, privacy, machine ethics) trên 30+ datasets.
                *   2401.06121 | TOFU benchmark unlearning cho LLM với dữ liệu tác giả hư cấu, quy trình tạo dữ liệu tổng hợp và chỉ số đánh giá mới (Truth Ratio, statistical test).
                *   2401.14019 | Unitxt là thư viện Python thống nhất quy trình chuẩn bị và đánh giá dữ liệu văn bản cho LLM với kiến trúc mô-đun (Task, Template, Format, Recipe).
            *   Multimodal Capabilities
                *   Vision-Language Models (VLM) / Multimodal LLMs (MLLM)
                    *   Architectures & Training
                        *   2401.00908 | DocLLM sử dụng disentangled spatial attention và mục tiêu infilling khối văn bản cho hiểu tài liệu trực quan mà không cần vision encoder.
                        *   2401.16420 | P-LoRA áp dụng LoRA có chọn lọc chỉ cho token hình ảnh để căn chỉnh VLM hiệu quả, bảo tồn năng lực ngôn ngữ.
                        *   2401.12208 | CheXagent là VLM cho diễn giải CXR, huấn luyện 3 giai đoạn (LLM pretrain, SigLIP image encoder pretrain, instruction tuning trên CheXinstruct dataset).
                    *   Grounded Reasoning & Interaction
                        *   2401.12168 | SpatialVLM là framework tự động tạo dữ liệu VQA cho lý luận không gian 3D từ ảnh thực, kết hợp nhiều vision model.
                    *   Data Generation & Augmentation
                        *   2401.11370 | G-LLaVA tạo dataset Geo170K (ảnh-chú thích, QA hình học) và fine-tune LLaVA cho giải toán hình học.
                    *   Evaluation & Benchmarking
                        *   2401.11944 | CMMMU là benchmark Đa phương thức Đa lĩnh vực Quy mô lớn tiếng Trung để đánh giá LMM chuyên môn cấp đại học.
                        *   2401.15071 | Đánh giá định tính so sánh Gemini và GPT-4V trên 3 khía cạnh (tổng quát hóa, tin cậy, nhân quả) và 4 phương thức (text, code, image, video).
                *   3D-Language Models
                    *   2401.10763 | M3DBench là benchmark 3D MLM với công thức chỉ thị đa phương thức xen kẽ (text, point, box, image, object3D) và kiến trúc baseline thống nhất.
            *   Model Interpretability
                *   2401.06102 | Patchscopes là framework thống nhất kiểm tra biểu diễn ẩn LLM bằng cách vá (patching) có chọn lọc biểu diễn nguồn vào quá trình suy luận đích.
            *   Open Source Initiatives
                *   2401.16818 | H2O-Danube-1.8B là LLM 1.8B mã nguồn mở (Llama 2/Mistral base) với SWA, GQA, RoPE, huấn luyện trên 1T-3T token.
            *   Other
                *   2401.02038 | Khảo sát toàn diện về LLM, bao gồm kiến thức nền tảng, huấn luyện, suy luận, ứng dụng và tương lai.
    *   Computer Vision (CV)
        *   Image Generation & Synthesis
            *   Diffusion Models
                *   Efficient Inference & Acceleration
                    *   2401.02677 | SSD-1B/Vega tỉa kiến trúc U-Net của SDXL và áp dụng feature distillation loss ở cấp độ tầng, cùng teacher swapping.
                *   Controllable & Personalized Generation
                    *   2401.07519 | InstantID là phương pháp tuning-free bảo toàn danh tính (ảnh tham chiếu đơn) cho T2I, với IdentityNet (điều kiện không gian yếu) và decoupled cross-attention.
                    *   2401.15975 | StableIdentity sử dụng AdaIN ánh xạ FR-ViT ID embedding vào celeb embedding space và hàm loss khuếch tán hai pha cho T2I cá nhân hóa.
                    *   2401.06105 | PALP (Prompt-Aligned Personalization) tối ưu T2I cho prompt mục tiêu bằng SDS/DDS trong quá trình fine-tuning cá nhân hóa.
                    *   2401.05252 | ControlNet-Transformer tích hợp điều khiển dạng ControlNet vào mô hình Transformer (PIXART-α) bằng zero-linear và áp dụng tín hiệu vào khối đầu.
                *   Compositional Generation
                    *   2401.11708 | RPG (Recaption, Plan, Generate) sử dụng MLLM làm recaptioner/planner (CoT) và Complementary Regional Diffusion cho T2I thành phần.
                *   LLM-driven Systems
                    *   2401.10061 | DiffusionGPT điều phối hệ thống T2I bằng LLM, phân tích prompt, xây dựng Tree-of-thought of Models, lựa chọn mô hình chuyên gia (ToT + Advantage DB) và mở rộng prompt.
            *   Masked Image Modeling (MIM)
                *   2401.01808 | aMUSEd là MIM nhẹ mã nguồn mở (U-ViT backbone) cho T2I 512x512 một giai đoạn, với CLIP-L/14 và micro-conditioning.
        *   Video Generation & Synthesis
            *   Diffusion-based & Controllable
                *   2401.12945 | STUNet (Space-Time U-Net) với temporal down/up-sampling tạo toàn bộ video trong một lượt; Multidiffusion cho SSR nhất quán.
                *   2401.00777 | VideoBooth là framework feed-forward cho I2V, với embedding thô-to-tinh, attention injection module và huấn luyện thô-to-tinh.
                *   2401.17681 | FlowVid sử dụng luồng quang học làm điều kiện mềm, tích hợp vào U-Net và ControlNet, với quy trình edit-propagate.
            *   Personalized & Stylized
                *   2401.13964 | PIA (Personalized Image Animator) cắm-n-chạy module điều kiện (ảnh + inter-frame affinity map) và temporal alignment layers vào T2I để tạo video cá nhân hóa.
        *   3D Content Generation & Understanding
            *   Text-to-3D & Image-to-3D
                *   2401.17053 | BlockFusion sử dụng VAE nén tri-plane thô vào latent space, sau đó diffusion trên latent tri-plane để sinh và mở rộng cảnh 3D.
            *   Avatar Modeling & Animation
                *   2401.15687 | Media2Face sử dụng GNPFA (VAE biểu cảm khuôn mặt 4D) và diffusion tiềm ẩn (điều kiện audio/text/image) cho hoạt họa mặt đa phương thức.
                *   2401.01885 | Tạo avatar hội thoại quang thực toàn thân bằng kiến trúc lai VQ-Transformer (thân/tay) và Diffusion (mặt, điều kiện đỉnh môi).
            *   Motion Synthesis
                *   2401.00063 | MoMask sử dụng RVQ phân cấp và masked generative modeling (M-Transformer, R-Transformer) cho T2M 3D.
                *   2401.03913 | CHOIS là conditional diffusion sinh đồng thời chuyển động người-vật thể, với object geometry loss và guidance terms (tiếp xúc).
        *   Image Editing & Manipulation
            *   2401.01702 | Image Sculpting là framework chỉnh sửa ảnh 2D bằng tương tác 3D (NeRF/mesh), với coarse-to-fine generative enhancement (DreamBooth, Depth Control, Feature Injection).
            *   2401.07409 | DiffMorpher nội suy tham số LoRA, slerp latent noise, inject self-attention và adjust AdaIN cho image morphing bằng diffusion.
            *   2401.13795 | DTC (Diffuse to Choose) là mô hình diffusion ẩn cho virtual try-all, với U-Net phụ trợ xử lý hint pixel-level và FiLM.
        *   Vision Backbone Architectures
            *   2401.09417 | Vision Mamba (Vim) là backbone thị giác dựa trên bidirectional SSM với mô-đun Quét Chọn lọc 2D (SS2D).
            *   2401.10166 | VMamba là backbone thị giác dựa trên SSM với mô-đun Quét Chọn lọc 2D (SS2D) và khối Visual State Space (VSS) tối ưu.
            *   2401.08541 | AIM (Autoregressive Image Models) tiền huấn luyện ViT bằng dự đoán tự hồi quy patch ảnh, với prefix attention và đầu MLP sâu.
            *   2401.02957 | DVT (Denoising Vision Transformers) khử nhiễu tạo tác trong ViT bằng Neural Fields (từng ảnh) và bộ khử nhiễu tổng quát (transformer nhẹ).
        *   Monocular Depth Estimation
            *   2401.10891 | Depth Anything huấn luyện MDE foundation model bằng self-training trên dữ liệu không nhãn quy mô lớn, với chiến lược tối ưu hóa khó hơn và kế thừa ngữ nghĩa DINOv2.
        *   3D Scene Understanding & Segmentation
            *   2401.09419 | GARField học trường affinity 3D điều khiển bởi thang đo vật lý từ mặt nạ SAM 2D để phân cấp nhóm trong NeRF.
            *   2401.09340 | SceneVerse dataset (2.5M cặp ngôn ngữ-3D) được tạo tự động từ 3DSG và LLM; GPS là khung pre-training hợp nhất dựa trên contrastive alignment đa cấp độ.
        *   Object Detection
            *   2401.17270 | YOLO-World là YOLO phát hiện đối tượng từ vựng mở, với RepVL-PAN (T-CSPLayer, Image-Pooling Attention) và chiến lược "prompt-then-detect".
    *   Audio & Speech Processing
        *   Audio Watermarking
            *   2401.17264 | AudioSeal là kiến trúc generator/detector cho thủy vân âm thanh cục bộ, với TF-Loudness loss và detector một lượt.
    *   Robotics & Embodied AI
        *   Mobile Manipulation & Learning from Demonstration
            *   2401.02117 | Mobile ALOHA là hệ thống điều khiển từ xa toàn thân chi phí thấp cho robot di động hai tay, với chiến lược co-training dữ liệu tĩnh/di động.
        *   Reinforcement Learning for Robotics
            *   2401.16013 | SERL là bộ phần mềm RL robot tích hợp RLPD, VICE, forward-backward control và nguyên tắc điều khiển trở kháng.
    *   AI Agents
        *   Multimodal & Web Agents
            *   2401.13919 | WebVoyager là agent web đa phương thức (LMM) tương tác website qua ảnh chụp màn hình được đánh dấu phần tử (GPT-4V-ACT).
            *   2401.16158 | Mobile-Agent là agent di động (MLLM) điều khiển GUI chỉ dựa trên ảnh chụp màn hình (OCR, Grounding DINO, CLIP) và self-reflection.
        *   Cognitive Architectures
            *   2401.17653 | LARP là kiến trúc nhận thức cho agent game nhập vai, với bộ nhớ dài hạn (semantic, episodic, procedural), Question-based Query và feedback-driven learnable action space.
    *   Machine Learning Systems
        *   GPU Acceleration & Kernel Optimization
            *   2401.14112 | TC-FPx là kiến trúc kernel GPU hỗ trợ FPx * FP16 trên Tensor Core, với Ahead-of-time Bit-level Pre-packing và SIMT-Efficient GPU Runtime.
        *   Distributed & Parallel Computing
            *   2401.10241 | Tách backward thành B/W, chiến lược lập lịch pipeline (ZB-H1/H2, heuristic/ILP) và post-update validation cho pipeline parallelism đồng bộ.
    *   Other
        *   2401.14953 | Meta-learning sử dụng dữ liệu từ UTM để mạng nơ-ron học chiến lược dự đoán phổ quát, xấp xỉ Quy nạp Solomonoff.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                          | Đột phá                                                                                                                                  | Ảnh hưởng                                                                                                                                                              |
    | :--- | :-------- | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2401.04088 | Mixtral 8x7B, Sparse MoE, Decoder-only, Open Source     | Mô hình SMoE 47B tham số (13B hoạt động) mã nguồn mở, đạt hiệu năng vượt Llama 2 70B và GPT-3.5 trên nhiều benchmark.                     | Dân chủ hóa việc tiếp cận LLM hiệu năng cao, thúc đẩy nghiên cứu và ứng dụng MoE, cho thấy tiềm năng của kiến trúc thưa.                                             |
    | 2    | 2401.10774 | Medusa, Parallel Decoding, LLM Inference, No Draft Model | Tăng tốc LLM inference 2x bằng nhiều đầu giải mã song song và tree attention, không cần mô hình nháp riêng biệt.                           | Cung cấp một giải pháp tăng tốc LLM hiệu quả, dễ tích hợp, giảm độ phức tạp so với speculative decoding truyền thống.                                                    |
    | 3    | 2401.12945 | STUNet, Video Diffusion, Space-Time U-Net, Multidiffusion | Kiến trúc Space-Time U-Net tạo toàn bộ video trong một lượt, kết hợp Multidiffusion cho SSR nhất quán, đạt SOTA về chất lượng và nhất quán. | Thay đổi cách tiếp cận sinh video dài, giải quyết vấn đề nhất quán thời gian và hiệu quả tính toán, mở đường cho video chất lượng cao hơn.                             |
    | 4    | 2401.02385 | TinyLlama, Small LLM, Large Data, 3-Stage Training      | Huấn luyện LLM 1.1B trên 3 nghìn tỷ token, cho thấy mô hình nhỏ có thể đạt hiệu năng cao khi huấn luyện với lượng dữ liệu cực lớn.           | Thách thức các định luật scaling truyền thống, cho thấy tiềm năng của việc "overtrain" mô hình nhỏ, quan trọng cho thiết bị biên.                                   |
    | 5    | 2401.10020 | Self-Rewarding LLM, Iterative DPO, AI Feedback (AIF)    | LLM tự đánh giá phản hồi (LLM-as-a-Judge) tạo AIF cho Iterative DPO, tự cải thiện cả instruction following và reward modeling.             | Hướng tới LLM tự cải thiện hoàn toàn, giảm sự phụ thuộc vào dữ liệu con người, có tiềm năng tạo ra các mô hình ngày càng thông minh hơn.                               |
    | 6    | 2401.09417 | Vision Mamba (Vim), Bidirectional SSM, SS2D             | Backbone thị giác thuần túy dựa trên Mamba với mô-đun Quét Chọn lọc 2D (SS2D), đạt hiệu năng Transformer với độ phức tạp tuyến tính.        | Mở ra hướng mới cho backbone thị giác hiệu quả, có khả năng xử lý ảnh độ phân giải cao và chuỗi dài tốt hơn ViT.                                                      |
    | 7    | 2401.13627 | SUPIR, SDXL for IR, ZeroSFT Adaptor, Restoration-guided Sampling | Sử dụng SDXL làm generative prior mạnh mẽ cho phục hồi ảnh, với adaptor ZeroSFT và chiến lược lấy mẫu có định hướng phục hồi.             | Đẩy mạnh giới hạn của phục hồi ảnh, cho thấy khả năng tạo ra chi tiết siêu thực và xử lý các trường hợp suy biến nặng.                                                  |
    | 8    | 2401.04398 | Chain-of-Table, Dynamic Planning, Table Transformation  | Sử dụng chuỗi bảng biến đổi làm đại diện bước suy luận, LLM lập kế hoạch động và tạo đối số cho thao tác bảng, SOTA trên hiểu bảng.        | Cung cấp một phương pháp reasoning mạnh mẽ và có cấu trúc cho các bài toán hiểu bảng phức tạp, vượt trội CoT truyền thống.                                            |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2401.00908 – Disentangled spatial attention & Block infilling pre-training (DocLLM) – Suy nghĩ:** Cơ chế chú ý không gian tách rời cho phép mô hình học tương tác văn bản-bố cục linh hoạt hơn; infilling khối phù hợp với tài liệu trực quan. Rất hứa hẹn cho LLM không cần vision encoder.
    *   **2401.04088 – Mixtral 8x7B (SMoE với 8 experts, top-K=2) – Suy nghĩ:** Mặc dù MoE không mới, việc triển khai hiệu quả ở quy mô này và mã nguồn mở là đóng góp lớn, thúc đẩy nghiên cứu MoE.
    *   **2401.10020 – Self-Rewarding Language Models (LLM-as-a-Judge tạo AIF cho Iterative DPO) – Suy nghĩ:** Vòng lặp tự cải thiện, nơi LLM vừa là người học vừa là người tạo reward, là một ý tưởng rất mạnh mẽ và có tiềm năng tự động hóa alignment.
    *   **2401.02385 – TinyLlama (3-stage training: basic, continual domain-specific, cooldown) – Suy nghĩ:** Chiến lược huấn luyện này, đặc biệt là cooldown bằng tăng batch size, là một kinh nghiệm thực tế giá trị cho việc huấn luyện LLM nhỏ trên dữ liệu lớn.
    *   **2401.12945 – STUNet (Temporal down/up-sampling) & Multidiffusion cho SSR – Suy nghĩ:** Kiến trúc U-Net không gian-thời gian cho phép tạo toàn bộ video một lượt là một bước tiến; Multidiffusion giải quyết nhất quán SSR.
    *   **2401.00368 – Synthetic data generation (LLM brainstorm tasks -> create examples) for text embedding – Suy nghĩ:** Quy trình hai bước tạo dữ liệu tổng hợp đa dạng cho text embedding, không cần dữ liệu gán nhãn lớn.
    *   **2401.13627 – SUPIR (ZeroSFT adaptor & Restoration-guided sampling & Negative quality prompts training) – Suy nghĩ:** Nhiều cải tiến thông minh để áp dụng SDXL cho phục hồi ảnh, đặc biệt ZeroSFT và chiến lược huấn luyện/sampling mới.
    *   **2401.15024 – SliceGPT (Computational invariance & PCA-based orthogonal transform + slicing) – Suy nghĩ:** Phương pháp pruning dựa trên tính bất biến tính toán và PCA để giảm chiều embedding là một cách tiếp cận mới lạ và hiệu quả.
    *   **2401.04081 – MoE-Mamba (Interleaved Mamba blocks and MoE layers) – Suy nghĩ:** Kết hợp điểm mạnh của Mamba (hiệu quả chuỗi dài) và MoE (tăng tham số hiệu quả) là một hướng kiến trúc hứa hẹn.
    *   **2401.13795 – DTC (Auxiliary U-Net for pixel hint & FiLM for feature integration) – Suy nghĩ:** Kiến trúc U-Net kép và FiLM để tích hợp chi tiết từ ảnh tham chiếu cho virtual try-on, giải quyết vấn đề bảo toàn chi tiết.
    *   **2401.01335 – Self-Play Fine-Tuning (SPIN) (LLM phân biệt phản hồi SFT gốc và phản hồi tự tạo) – Suy nghĩ:** Cơ chế tự chơi để LLM tự cải thiện từ dữ liệu SFT gốc, không cần dữ liệu preference ngoài.
    *   **2401.14196 – Repository-level pre-training (Dependency analysis & topological sort & repo-level deduplication) – Suy nghĩ:** Xử lý dữ liệu tiền huấn luyện ở cấp độ kho chứa, bảo toàn ngữ cảnh liên-tệp, rất quan trọng cho Code LLM.
    *   **2401.10891 – Depth Anything (Harder optimization for pseudo-labels: strong augmentation, CutMix & Semantic inheritance via feature alignment loss with tolerance) – Suy nghĩ:** Các chiến lược thông minh để tận dụng dữ liệu không nhãn quy mô lớn cho MDE, đặc biệt là "tối ưu hóa khó hơn".
    *   **2401.09417 – Vision Mamba (Vim) (Bidirectional SSM & 2D Selective Scan SS2D) – Suy nghĩ:** Mở rộng thành công Mamba cho thị giác với SS2D, một giải pháp hiệu quả cho backbone thị giác.
    *   **2401.13660 – MambaByte (Byte-level Mamba & Hybrid speculative decoding: subword draft + byte verify) – Suy nghĩ:** Áp dụng Mamba cho byte-level LM và cơ chế speculative decoding lai rất sáng tạo để cân bằng tốc độ và tính không token.
    *   **2401.07519 – InstantID (IdentityNet với landmark & Decoupled cross-attention cho ID embedding) – Suy nghĩ:** Kết hợp ID embedding ngữ nghĩa mạnh và IdentityNet (không gian yếu) để cá nhân hóa T2I hiệu quả, không cần fine-tuning.
    *   **2401.16420 – P-LoRA (Partial LoRA: LoRA có chọn lọc chỉ cho image tokens) – Suy nghĩ:** Một cách tiếp cận PEFT tinh tế cho MLLM, cân bằng giữa học kiến thức thị giác và bảo tồn năng lực ngôn ngữ.
    *   **2401.10774 – Medusa (Multiple decoding heads & Tree attention & Typical acceptance) – Suy nghĩ:** Kiến trúc tăng tốc LLM inference không cần draft model, tree attention và typical acceptance là những cải tiến đáng chú ý.
    *   **2401.06105 – PALP (Prompt-aligned personalization với SDS/DDS guidance trong fine-tuning) – Suy nghĩ:** Tích hợp SDS/DDS vào quá trình fine-tuning cá nhân hóa để cải thiện tuân thủ prompt phức tạp.
    *   **2401.05252 – ControlNet-Transformer (Zero-linear và áp dụng tín hiệu điều khiển vào khối đầu Transformer) – Suy nghĩ:** Điều chỉnh ControlNet cho kiến trúc Transformer một cách hợp lý hơn.
    *   **2401.04658 – Lightning Attention-2 (Tiling & Hybrid computation cho causal linear attention) – Suy nghĩ:** Giải quyết nút thắt cumsum trong linear attention nhân quả, đạt độ phức tạp tuyến tính thực tế.
    *   **2401.06003 – TRIPS (Trilinear Point Splatting vào image pyramid & Differentiable trilinear rendering & Lightweight reconstruction network) – Suy nghĩ:** Phương pháp lai splatting và reconstruction cho real-time radiance field rendering, tối ưu được kích thước điểm.
    *   **2401.05675 – Parrot (Pareto-optimal selection cho multi-reward RL T2I & Original prompt-centered guidance) – Suy nghĩ:** Tối ưu đa mục tiêu cho T2I bằng RL dựa trên Pareto optimality, tự động cân bằng các phần thưởng.
    *   **2401.04398 – Chain-of-Table (Chuỗi bảng biến đổi làm suy luận trung gian & Dynamic planning LLM chọn thao tác bảng) – Suy nghĩ:** Một cách tiếp cận mới cho table reasoning, sử dụng bảng làm "trạng thái" suy luận.
    *   **2401.16013 – SERL (Nguyên tắc thiết kế bộ điều khiển trở kháng cho tương tác-tiếp xúc) – Suy nghĩ:** Mặc dù là toolkit, việc nhấn mạnh và triển khai nguyên tắc điều khiển trở kháng này rất quan trọng cho RL robot thực tế.
    *   **2401.15687 – Media2Face (GNPFA VAE biểu cảm khuôn mặt 4D & Diffusion tiềm ẩn với dual masking CFG & Overlapped batching denoising) – Suy nghĩ:** Hệ thống toàn diện cho hoạt họa mặt đa phương thức, GNPFA và dual masking CFG là những điểm mới.
    *   **2401.01885 – Hybrid VQ-Transformer và Diffusion cho avatar hội thoại (VQ-Transformer cho guide poses thưa + Diffusion điền chi tiết) – Suy nghĩ:** Kết hợp điểm mạnh của VQ (cấu trúc dài hạn) và Diffusion (chi tiết tần số cao) cho sinh chuyển động.
    *   **2401.12168 – SpatialVLM (Framework tạo dữ liệu VQA 3D từ ảnh thực: nâng 2D lên 3D, chuẩn hóa tọa độ, giải quyết nhập nhằng) – Suy nghĩ:** Quy trình tự động tạo dữ liệu VQA 3D quy mô lớn, giải quyết vấn đề thiếu dữ liệu cho lý luận không gian 3D.
    *   **2401.11944 – CMMMU (Benchmark Đa phương thức Đa lĩnh vực Quy mô lớn tiếng Trung) – Suy nghĩ:** Đóng góp quan trọng về benchmark cho LMM tiếng Trung, đặc biệt ở cấp độ chuyên gia.
    *   **2401.01952 – Instruct-Imagen (Định dạng hướng dẫn đa phương thức & Huấn luyện tăng cường truy xuất + tinh chỉnh hướng dẫn đa phương thức) – Suy nghĩ:** Framework thống nhất cho sinh ảnh có điều kiện phức tạp, kết hợp nhiều loại input.
    *   **2401.11708 – RPG (MLLM recaptioner/planner CoT & Complementary Regional Diffusion) – Suy nghĩ:** Khung làm việc training-free, kết hợp MLLM và diffusion theo vùng cho T2I thành phần.
    *   **2401.10061 – DiffusionGPT (LLM điều phối T2I: Tree-of-thought of Models & Advantage Databases & Adaptive prompt extension) – Suy nghĩ:** Hệ thống LLM-driven chọn mô hình khuếch tán chuyên gia và tối ưu prompt.
    *   **2401.02957 – DVT (Khử nhiễu ViT bằng Neural Fields trên từng ảnh + bộ khử nhiễu tổng quát) – Suy nghĩ:** Phân tích và giải quyết vấn đề nhiễu tạo tác trong ViT một cách có hệ thống.
    *   **2401.02412 – CALM (Kết hợp LLM neo và LLM tăng cường qua linear projections và cross-attention học được) – Suy nghĩ:** Phương pháp model composition hiệu quả, không cần fine-tuning mô hình gốc.
    *   **2401.15071 – So sánh Gemini và GPT-4V (Đánh giá định tính đa diện) – Suy nghĩ:** Nghiên cứu so sánh sâu sắc, cung cấp hiểu biết thực tế về khả năng của các MLLM hàng đầu.
    *   **2401.06104 – TOVA (Token Omission Via Attention: loại bỏ token KV cache theo attention score) – Suy nghĩ:** Chính sách nén KV cache đơn giản, hiệu quả, không cần huấn luyện.
    *   **2401.17270 – YOLO-World (RepVL-PAN: T-CSPLayer, Image-Pooling Attention & "Prompt-then-detect" với offline vocabulary) – Suy nghĩ:** Đưa khả năng open-vocabulary vào YOLO một cách hiệu quả, phù hợp cho ứng dụng real-time.
    *   **2401.12474 – DITTO (Self-alignment cho LLM nhập vai: tự mô phỏng hội thoại, tạo truy vấn đối nghịch, SFT bỏ ngữ cảnh hồ sơ) – Suy nghĩ:** Phương pháp tự căn chỉnh độc đáo cho LLM nhập vai, không cần chưng cất từ mô hình mạnh hơn.
    *   **2401.10225 – Context-Enhanced Instruction Tuning (Tinh chỉnh LLM/retriever hai giai đoạn cho QA hội thoại, tạo dữ liệu HumanAnnotatedConvQA/SyntheticConvQA) – Suy nghĩ:** Chiến lược fine-tuning chuyên biệt để cải thiện RAG và QA hội thoại.
    *   **2401.08417 – CPO (Contrastive Preference Optimization: xấp xỉ DPO với uniform prior, kết hợp NLL loss) – Suy nghĩ:** Biến thể DPO hiệu quả hơn về bộ nhớ và tốc độ cho tinh chỉnh LLM dịch máy.
    *   **2401.02823 – DocGraphLM (LM + GNN, dự đoán liên kết tái tạo đồ thị tài liệu, hàm mục tiêu kết hợp MSE khoảng cách và CE hướng) – Suy nghĩ:** Kiến trúc lai LM-GNN và phương pháp học biểu diễn cấu trúc tài liệu hiệu quả.
    *   **2401.17053 – BlockFusion (VAE nén tri-plane thô vào latent, diffusion trên latent tri-plane, ngoại suy latent mở rộng cảnh) – Suy nghĩ:** Sinh cảnh 3D mở rộng bằng diffusion trên latent tri-plane, giải quyết vấn đề đa dạng và phức tạp của cảnh.
    *   **2401.17653 – LARP (Bộ nhớ dài hạn semantic/episodic/procedural, Question-based Query, feedback-driven learnable action space) – Suy nghĩ:** Kiến trúc nhận thức phức tạp cho agent game nhập vai, lấy cảm hứng từ tâm lý học.
    *   **2401.12503 – Vary-toy (Huấn luyện mạng từ vựng thị giác phụ trợ trên dữ liệu OCR và phát hiện đối tượng) – Suy nghĩ:** Cải thiện Vary bằng cách làm giàu từ vựng thị giác phụ trợ với thông tin vị trí đối tượng.
    *   **2401.13919 – WebVoyager (Agent web LMM tương tác qua ảnh chụp màn hình đánh dấu phần tử GPT-4V-ACT, đánh giá tự động GPT-4V) – Suy nghĩ:** Agent web đa phương thức tự hành, không cần phân tích HTML, tương tác trực quan hơn.
    *   **2401.12954 – Meta-prompting (LM duy nhất làm Meta Model và Expert Model, phân rã nhiệm vụ, tạo hướng dẫn động) – Suy nghĩ:** Kỹ thuật scaffolding nhiệm vụ bất khả tri, sử dụng một LM để điều phối và thực thi.
    *   **2401.08967 – ReFT (SFT warm-up + PPO online, hàm thưởng so sánh CoT với ground-truth) – Suy nghĩ:** Fine-tuning LLM cho reasoning bằng RL, không cần reward model riêng.
    *   **2401.02117 – Mobile ALOHA (Hệ thống điều khiển từ xa toàn thân chi phí thấp & Co-training dữ liệu tĩnh/di động) – Suy nghĩ:** Giải pháp phần cứng và chiến lược học hiệu quả cho robot thao tác di động hai tay.
    *   **2401.01952 – Instruct-Imagen (Định dạng hướng dẫn đa phương thức & Huấn luyện tăng cường truy xuất + tinh chỉnh hướng dẫn đa phương thức) – Suy nghĩ:** Framework thống nhất cho sinh ảnh có điều kiện phức tạp, kết hợp nhiều loại input.
    *   **2401.14112 – TC-FPx (Kernel GPU cho FPx * FP16 trên Tensor Core & Ahead-of-time Bit-level Pre-packing & SIMT-Efficient GPU Runtime) – Suy nghĩ:** Tối ưu hóa phần cứng sâu cho tính toán FPx trên GPU, quan trọng cho lượng tử hóa LLM.
    *   **2401.16380 – WRAP (WebRephrase Augmented Pre-training: LLM diễn giải tài liệu web theo phong cách cụ thể, tiền huấn luyện trên hỗn hợp) – Suy nghĩ:** Tăng tốc và cải thiện tiền huấn luyện LLM bằng dữ liệu diễn giải tổng hợp.
    *   **2401.04577 – MAGNeT (Masked generation audio không tự hồi quy & Masking theo span & Restricted context self-attention & CFG annealing) – Suy nghĩ:** Mô hình Transformer non-AR hiệu quả cho sinh audio, với nhiều cải tiến về masking và sampling.
    *   **2401.15977 – Motion-I2V (Dự đoán trường chuyển động (VLDM) + Kết xuất video có điều khiển chuyển động (motion-augmented temporal attention)) – Suy nghĩ:** Phân tách dự đoán chuyển động và tổng hợp video, với cơ chế attention tăng cường chuyển động.
    *   **2401.10166 – VMamba (SS2D Quét Chọn lọc 2D & Khối Visual State Space VSS) – Suy nghĩ:** Một biến thể khác của Mamba cho thị giác, cũng tập trung vào quét 2D hiệu quả.
    *   **2401.08541 – AIM (Prefix attention & Đầu MLP sâu cho dự đoán pixel tự hồi quy) – Suy nghĩ:** Áp dụng nguyên lý tự hồi quy từ LLM cho thị giác với các điều chỉnh kiến trúc phù hợp.
    *   **2401.02955 – OV-SAM (SAM2CLIP adapter chưng cất kiến thức SAM sang CLIP & CLIP2SAM adapter đưa nhận dạng CLIP vào SAM decoder) – Suy nghĩ:** Kết hợp SAM và CLIP hiệu quả để tạo mô hình phân đoạn từ vựng mở.
    *   **2401.12179 – DITTO (Tối ưu latent nhiễu ban đầu xT cho TTM diffusion bằng feature matching loss & Gradient checkpointing cho sampler step) – Suy nghĩ:** Điều khiển TTM diffusion tại thời gian inference bằng cách tối ưu xT, hiệu quả bộ nhớ nhờ gradient checkpointing.
    *   **2401.09419 – GARField (Scale-conditioned 3D affinity field từ mặt nạ SAM 2D & Giám sát thang đo liên tục & Mất mát bao hàm) – Suy nghĩ:** Học trường affinity 3D phân cấp từ mặt nạ 2D đa cấp độ, giải quyết xung đột và không nhất quán.
    *   **2401.09340 – SceneVerse dataset (Tạo tự động từ 3DSG+LLM: object caption, object referral, scene description) & GPS pre-training (Multi-level contrastive alignment) – Suy nghĩ:** Quy trình tạo dữ liệu 3D-ngôn ngữ quy mô lớn và khung pre-training hợp nhất.
    *   **2401.04092 – Đánh giá T23D bằng GPT-4V (Meta-prompt tạo prompt đánh giá & So sánh cặp đối tượng 3D (RGB+normal) bằng GPT-4V) – Suy nghĩ:** Phương pháp đánh giá T23D tự động, đa tiêu chí dựa trên LMM.

4.  **GAPS_AND_OPPORTUNITIES**
    *   **Hiệu quả và Khả năng mở rộng của Kiến trúc Sequence Mới (ví dụ: Mamba, Linear Attention):**
        *   *Gaps:* Các kiến trúc như Mamba (2401.00752, 2401.09417, 2401.10166, 2401.04081), Lightning Attention-2 (2401.04658) cho thấy tiềm năng lớn về hiệu quả tính toán tuyến tính, nhưng việc áp dụng rộng rãi cho các mô hình đa phương thức cực lớn, huấn luyện trên hàng nghìn tỷ token, và tối ưu hóa phần cứng chuyên dụng vẫn cần nhiều nghiên cứu. Khả năng duy trì hiệu năng trên các tác vụ đòi hỏi reasoning phức tạp so với Transformer truyền thống cần được kiểm chứng kỹ hơn.
        *   *Opportunities:* Phát triển các biến thể Mamba/SSM tối ưu hơn cho các loại dữ liệu và tác vụ cụ thể (ví dụ: MambaByte 2401.13660 cho byte-level). Nghiên cứu sâu hơn về lý thuyết và thực nghiệm của selective state space. Tối ưu hóa kernel và phần cứng cho các kiến trúc này. Khám phá kiến trúc lai ghép Mamba/SSM với Transformer.
    *   **Multimodal LLMs (MLLMs): Tích hợp sâu, Grounding chi tiết và Suy luận đa bước:**
        *   *Gaps:* Việc tích hợp các modal vào LLM vẫn thường ở mức "ghép nối" các encoder chuyên biệt (U-IO2 2401.17172, OneLLM 2312.03700 là các nỗ lực cải thiện). Khả năng MLLM thực sự "hiểu" và "suy luận" trên thông tin không gian chi tiết, mối quan hệ phức tạp giữa các đối tượng, và thực hiện các tác vụ đòi hỏi nhiều bước suy luận đa phương thức còn hạn chế (DocLLM 2401.00908, PixelLLM 2312.09237, SpatialVLM 2401.12168).
        *   *Opportunities:* Kiến trúc MLLM "natively multimodal" thực sự, có khả năng xử lý và tạo ra thông tin ở các mức độ chi tiết khác nhau. Các phương pháp pre-training và fine-tuning hiệu quả hơn cho MLLM, đặc biệt với dữ liệu 3D và video. Kỹ thuật grounding và co-referencing mạnh mẽ hơn giữa các modal. Phát triển "multimodal chain-of-thought" và các kỹ thuật prompting tiên tiến cho MLLM.
    *   **Generative AI (Video, 3D, Audio): Độ nhất quán, Khả năng điều khiển, Tính tương tác và Hiệu quả:**
        *   *Gaps:* Sinh video/3D/audio dài, chất lượng cao, nhất quán, có thể điều khiển chi tiết và tương tác real-time vẫn là thách thức lớn (STUNet 2401.12945, VideoPoet 2312.14125, DG4D 2312.17142, Media2Face 2312.15687, Audiobox 2312.15821). Các phương pháp hiện tại thường tốn kém, giới hạn độ phân giải/thời lượng, hoặc gặp vấn đề về flickering/artifacts.
        *   *Opportunities:* Kiến trúc sinh hiệu quả hơn, có khả năng mô hình hóa quan hệ không gian-thời gian tốt hơn. Các phương pháp điều khiển đa modal tinh vi hơn. Tích hợp các nguyên lý vật lý và kiến thức thế giới vào quá trình sinh. Kỹ thuật rendering và streaming hiệu quả cho nội dung động. Tăng tốc inference cho các mô hình diffusion (StreamDiffusion 2312.12491, DeepCache 2312.00858, Block Caching 2312.03209).
    *   **Alignment, An toàn và Tính đáng tin cậy của AI:**
        *   *Gaps:* Các phương pháp alignment (Self-Rewarding LLM 2401.10020, SPIN 2401.01335, CPO 2401.08417, CUT 2312.14591) đang phát triển nhưng vẫn cần cải thiện về hiệu quả, chi phí và khả năng tổng quát hóa. Vấn đề "reward hacking" và "superficial alignment" (2312.01552) vẫn tồn tại. Khả năng LLM tự phát triển hành vi lừa dối (2401.05566) là một mối lo ngại lớn. Đánh giá toàn diện tính đáng tin cậy còn khó khăn (TRUSTLLM 2401.05561).
        *   *Opportunities:* Kỹ thuật alignment mới, ít phụ thuộc vào dữ liệu con người, có thể tự cải thiện và chống lại các chiến lược đối phó. Các phương pháp "red teaming" tự động và benchmark an toàn mạnh mẽ hơn. Nghiên cứu sâu hơn về các hành vi không mong muốn và cách phòng ngừa/phát hiện chúng.
    *   **Dữ liệu cho AI: Tự động hóa, Chất lượng và Miền chuyên biệt:**
        *   *Gaps:* Việc tạo và tinh lọc dữ liệu quy mô lớn, chất lượng cao cho các tác vụ và miền mới vẫn là nút thắt cổ chai (CodeSeaXDataset 2401.14187, Geo170K 2312.11370, SceneVerse 2401.09340, MATHPILE 2312.17120, DL3DV-10K 2312.16256). Các phương pháp sinh dữ liệu tổng hợp cần cải thiện về độ tin cậy.
        *   *Opportunities:* Các kỹ thuật data generation/augmentation thông minh hơn, có thể tự động điều chỉnh theo nhu cầu của mô hình. Phương pháp self-supervised/weakly-supervised learning hiệu quả hơn. Công cụ và quy trình tự động hóa việc làm sạch, lọc và đánh giá chất lượng dữ liệu.
    *   **Agent tự trị và Tương tác Người-Máy:**
        *   *Gaps:* Xây dựng các agent có khả năng học hỏi liên tục, thích ứng với môi trường mới, và tương tác tự nhiên, hiệu quả với con người và các giao diện phức tạp (AppAgent 2312.13771, WebVoyager 2401.13919, Mobile-Agent 2401.16158, LARP 2401.17653) vẫn là mục tiêu dài hạn.
        *   *Opportunities:* Framework agent tổng quát hơn, có khả năng học các kỹ năng mới và tự cải thiện từ phản hồi. Nghiên cứu về bộ nhớ dài hạn, khả năng trừu tượng hóa và giao diện tương tác người-máy thông minh hơn.
    *   **Hiệu quả Triển khai trên Phần cứng Đa dạng:**
        *   *Gaps:* Việc triển khai các mô hình lớn trên các thiết bị có tài nguyên hạn chế (ví dụ: di động, edge) hoặc các cụm máy chủ phân tán không đồng nhất vẫn là thách thức (2312.11514, PowerInfer 2312.12456, PETALS 2312.08361, TC-FPx 2401.14112).
        *   *Opportunities:* Các thuật toán và kiến trúc phần cứng mới cho AI hiệu quả năng lượng. Kỹ thuật nén mô hình và tối ưu hóa kernel chuyên biệt cho các nền tảng phần cứng khác nhau.
    *   **Lý thuyết và Hiểu biết về LLM:**
        *   *Gaps:* Hiểu biết sâu sắc về cách LLM học, suy luận và tại sao chúng lại có những hành vi nhất định (ví dụ: scaling laws 2401.02954, 2401.00448, unlearning 2401.06121, interpretability 2401.06102) vẫn còn hạn chế.
        *   *Opportunities:* Phát triển các công cụ và phương pháp phân tích mới để "mở hộp đen" LLM. Xây dựng các mô hình vốn đã có tính diễn giải. Nghiên cứu lý thuyết sâu hơn về các thuộc tính của LLM.

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Mamba-based World Models for Interactive 3D Agents**
    *   **Motivation:** Mamba (2401.00752) xử lý chuỗi dài hiệu quả. Các agent 3D tương tác (LARP 2401.17653, AppAgent 2312.13771) cần hiểu lịch sử tương tác dài và lập kế hoạch phức tạp. Các mô hình thế giới (world models) hiện tại thường dựa trên RNN/Transformer truyền thống.
    *   **Key Novelty:** Xây dựng một "world model" dựa trên kiến trúc Mamba, có khả năng học động lực học của môi trường 3D tương tác từ chuỗi quan sát đa phương thức (hình ảnh, âm thanh, hành động, văn bản) siêu dài. Agent sẽ sử dụng world model này để "tưởng tượng" các kịch bản tương lai, lập kế hoạch và ra quyết định.
    *   **Approach:**
        1.  Thiết kế kiến trúc Mamba đa phương thức, có khả năng xử lý đầu vào xen kẽ từ các cảm biến của agent và tạo ra dự đoán về trạng thái tương lai của môi trường.
        2.  Huấn luyện world model trên dữ liệu tương tác quy mô lớn (có thể từ mô phỏng hoặc dữ liệu thực tế) bằng các mục tiêu tự giám sát (ví dụ: dự đoán khung hình/âm thanh tiếp theo, dự đoán kết quả của hành động).
        3.  Tích hợp world model này vào một agent RL hoặc agent dựa trên LLM. Agent sẽ sử dụng world model để:
            *   Dự đoán hậu quả của các chuỗi hành động tiềm năng.
            *   Lập kế hoạch bằng cách "tìm kiếm" trong không gian các kịch bản tương lai do world model tạo ra.
            *   Học một chính sách hiệu quả hơn.
    *   **Dataset + Metrics:** Các môi trường mô phỏng 3D tương tác (AI2-THOR, Habitat, Minecraft, Isaac Gym). Dữ liệu có thể là các bản ghi tương tác của con người hoặc của các agent khác. Metrics: Tỷ lệ hoàn thành nhiệm vụ, hiệu quả (số bước, thời gian), khả năng thích ứng với thay đổi, chất lượng của các dự đoán từ world model.
    *   **Risk/Feasibility:** Rất cao (Moon-shot). Huấn luyện Mamba đa phương thức cho world model là cực kỳ phức tạp và tốn kém. Đảm bảo world model đủ chính xác và tổng quát hóa tốt là thách thức lớn.

    ✨ **Idea 2: Self-Improving Code LLMs via Constitutional Self-Play and Formal Verification Feedback**
    *   **Motivation:** LLM sinh mã (Mixtral 2401.04088, TinyLlama 2401.02385, OSS-INSTRUCT 2312.02120) ngày càng mạnh nhưng vẫn tạo ra lỗi. SPIN (2401.01335) và Self-Rewarding LLM (2401.10020) cho thấy tiềm năng của self-play/self-reward. Weaver (2401.17268) sử dụng Constitutional DPO.
    *   **Key Novelty:** Một Code LLM tự cải thiện thông qua một vòng lặp "self-play" kết hợp "Constitutional DPO" và phản hồi từ một hệ thống kiểm chứng hình thức (formal verification) hoặc một LLM chuyên về phân tích mã.
    *   **Approach:**
        1.  **Seed LLM:** Bắt đầu với một Code LLM đã được SFT.
        2.  **Problem Generation:** LLM tự sinh các bài toán lập trình mới (có thể dựa trên các bài toán hiện có và biến đổi chúng).
        3.  **Solution Generation (Self-Play):** LLM hiện tại (player) cố gắng giải các bài toán này, tạo ra nhiều giải pháp ứng viên.
        4.  **Constitutional Critique & Negative Example Generation:**
            *   Một bộ "nguyên tắc lập trình tốt" (ví dụ: code phải hiệu quả, dễ đọc, an toàn, không có lỗi logic phổ biến) được định nghĩa.
            *   LLM (hoặc một LLM critic riêng) đánh giá các giải pháp ứng viên dựa trên các nguyên tắc này.
            *   Đối với các giải pháp tốt, LLM được hướng dẫn để tạo ra các phiên bản "xấu" một cách có chủ đích bằng cách vi phạm một hoặc nhiều nguyên tắc (tạo negative examples cho DPO).
        5.  **Formal Verification Feedback (Optional but Powerful):**
            *   Các giải pháp tốt được đưa qua một hệ thống kiểm chứng hình thức (nếu có thể áp dụng) hoặc một LLM chuyên phân tích mã để tìm lỗi tiềm ẩn hoặc điểm cần cải thiện. Feedback này được dùng để tinh chỉnh giải pháp hoặc tạo thêm negative examples.
        6.  **Preference Data Generation & DPO:** Tạo các cặp (giải pháp tốt hơn, giải pháp kém hơn) dựa trên critique, feedback từ verification, và kết quả unit test. Huấn luyện LLM bằng DPO.
        7.  Lặp lại từ bước 2 với LLM đã được cải thiện.
    *   **Dataset + Metrics:** Bắt đầu với các benchmark sinh mã (HumanEval, MBPP). Dữ liệu mới được tạo ra trong quá trình self-play. Metrics: Tỷ lệ pass@k, điểm số từ các công cụ phân tích tĩnh, đánh giá của con người về chất lượng mã, và khả năng tuân thủ các "nguyên tắc".
    *   **Risk/Feasibility:** Cao. Tích hợp kiểm chứng hình thức vào vòng lặp huấn luyện LLM là rất khó. Việc định nghĩa "nguyên tắc lập trình tốt" một cách toàn diện và khả thi cho LLM cũng không đơn giản. Vòng lặp có thể không hội tụ hoặc tạo ra các giải pháp quá phức tạp.

    ✨ **Idea 3: Trustworthy MLLM via Iterative Cross-Modal Fact Verification and Explanation Generation**
    *   **Motivation:** MLLM (Gemini 2312.11805, U-IO2 2401.17172) có thể "hallucinate" thông tin không nhất quán giữa các modal. TRUSTLLM (2401.05561) nhấn mạnh tầm quan trọng của tính đáng tin cậy. Patchscopes (2401.06102) cho phép kiểm tra biểu diễn ẩn.
    *   **Key Novelty:** Một MLLM có khả năng tự kiểm tra và giải thích tính nhất quán của thông tin giữa các modal (ví dụ: văn bản và hình ảnh) trong quá trình tạo phản hồi. Nếu phát hiện sự không nhất quán, MLLM sẽ cố gắng sửa lỗi hoặc cảnh báo người dùng.
    *   **Approach:**
        1.  **Initial Response Generation:** MLLM tạo một phản hồi ban đầu cho một truy vấn đa phương thức.
        2.  **Cross-Modal Fact Extraction & Verification:**
            *   Một module (có thể là chính MLLM với prompt chuyên biệt, hoặc một mô hình phụ) trích xuất các "fact" hoặc "claim" từ phản hồi liên quan đến từng modal (ví dụ: "Con mèo màu đen đang ngồi trên ghế" - claim về hình ảnh).
            *   Sử dụng các kỹ thuật grounding (PixelLLM 2312.09237) hoặc VQA (CheXagent 2401.12208) để kiểm tra xem các claim này có nhất quán với thông tin từ các modal tương ứng không. Ví dụ, hỏi VLM: "Có con mèo đen nào đang ngồi trên ghế trong ảnh không?"
        3.  **Inconsistency Detection & Explanation:** Nếu phát hiện sự không nhất quán, MLLM sẽ:
            *   Cố gắng giải thích nguyên nhân của sự không nhất quán (ví dụ: "Tôi đã nói con mèo màu đen, nhưng trong ảnh nó màu vàng. Có thể tôi đã nhầm lẫn.").
            *   Cố gắng sửa lại phản hồi ban đầu để đảm bảo tính nhất quán.
            *   Nếu không thể sửa, cảnh báo người dùng về sự không chắc chắn.
        4.  **Iterative Refinement:** Quá trình này có thể lặp lại để tinh chỉnh phản hồi.
        5.  Huấn luyện MLLM trên dữ liệu chứa các ví dụ về truy vấn, phản hồi, các claim đã xác minh, giải thích về sự không nhất quán (nếu có), và phản hồi đã sửa. Dữ liệu này có thể được tạo tổng hợp hoặc từ chú thích của con người.
    *   **Dataset + Metrics:** Sử dụng các benchmark VQA, Image Captioning, Visual Reasoning hiện có và tạo thêm các mẫu thử nghiệm chứa thông tin xung đột hoặc dễ gây nhầm lẫn giữa các modal. Metrics: Độ chính xác của phản hồi, tỷ lệ phát hiện và sửa lỗi không nhất quán, chất lượng của giải thích, và đánh giá của con người về tính đáng tin cậy.
    *   **Risk/Feasibility:** Cao. Việc MLLM tự trích xuất fact và xác minh một cách đáng tin cậy là rất khó. Vòng lặp có thể không hội tụ hoặc làm tăng đáng kể độ trễ.

    ✨ **Idea 4: Personalized Text-to-Video Generation with Dynamic Character and Scene Consistency using Mamba-based Spatio-Temporal Memory**
    *   **Motivation:** Sinh video dài, nhất quán với nhân vật/cảnh tùy chỉnh là mục tiêu lớn (VideoPoet 2312.14125, VideoStudio 2401.01256). Mamba (2401.00752) hiệu quả cho chuỗi dài. Cá nhân hóa T2I (InstantID 2401.07519, PhotoMaker 2312.04461) đang phát triển.
    *   **Key Novelty:** Một mô hình T2V sử dụng kiến trúc Mamba để duy trì một "bộ nhớ không gian-thời gian" (spatio-temporal memory) về các nhân vật và yếu tố cảnh đã được cá nhân hóa. Bộ nhớ này được cập nhật động khi video được sinh ra, cho phép duy trì tính nhất quán của các đối tượng/nhân vật qua các cảnh dài và phức tạp, đồng thời cho phép người dùng "triệu hồi" hoặc thay đổi các yếu tố đã cá nhân hóa bằng prompt.
    *   **Approach:**
        1.  **Personalization Phase:** Người dùng cung cấp một vài ảnh/video ngắn về một nhân vật hoặc một cảnh. Một mạng neural (có thể là một Mamba encoder nhỏ) mã hóa thông tin này thành một "identity vector" hoặc "scene vector" và lưu vào một bộ nhớ ngoài (external memory).
        2.  **Video Generation Phase:**
            *   Kiến trúc chính là một mô hình T2V dựa trên Mamba (hoặc Mamba-Diffusion hybrid).
            *   Khi người dùng cung cấp prompt (ví dụ: "Nhân vật A đi vào cảnh B"), các identity/scene vectors tương ứng được truy xuất từ bộ nhớ.
            *   Các vector này được đưa vào các khối Mamba như một phần của trạng thái ẩn hoặc điều kiện bổ sung, ảnh hưởng đến quá trình sinh video.
            *   Khi video được sinh ra, bộ nhớ không gian-thời gian của Mamba sẽ tự động cập nhật và lan truyền thông tin về appearance và vị trí của các nhân vật/cảnh, giúp duy trì tính nhất quán.
            *   Có thể sử dụng các kỹ thuật như trong PIA (2312.13964) hoặc AnimateZero (2312.03793) để điều khiển chuyển động cụ thể.
    *   **Dataset + Metrics:** Sử dụng các bộ dữ liệu video hiện có (WebVid, Panda-70M) và tạo thêm dữ liệu cá nhân hóa. Metrics: Đánh giá của con người về tính nhất quán của nhân vật/cảnh, mức độ tuân thủ prompt, chất lượng video. Có thể phát triển các metric tự động đo lường sự thay đổi appearance của nhân vật qua các khung hình.
    *   **Risk/Feasibility:** Rất cao (Moon-shot). Thiết kế và huấn luyện Mamba cho T2V cá nhân hóa với bộ nhớ động là cực kỳ phức tạp. Đảm bảo tính nhất quán qua các cảnh rất dài và với nhiều nhân vật/đối tượng tương tác là thách thức lớn.

6.  **READING_LIST**

    *   2401.04088 – Mixtral 8x7B · Mô hình SMoE mã nguồn mở hiệu năng cao, rất quan trọng cho cộng đồng.
    *   2401.10774 – Medusa · Kỹ thuật tăng tốc LLM inference không cần draft model, thực tế và hiệu quả.
    *   2401.12945 – STUNet · Kiến trúc T2V mới tạo toàn bộ video một lượt, giải quyết vấn đề nhất quán thời gian.
    *   2401.10020 – Self-Rewarding LLMs · Hướng đi tự cải thiện LLM rất hứa hẹn, giảm phụ thuộc dữ liệu người.
    *   2401.09417 – Vision Mamba (Vim) · Áp dụng Mamba cho thị giác, tiềm năng lớn cho backbone hiệu quả.
    *   2401.04398 – Chain-of-Table · Phương pháp reasoning mới cho bảng, kết hợp LLM và thao tác bảng có cấu trúc.
    *   2401.13627 – SUPIR · Sử dụng SDXL cho phục hồi ảnh, đạt chất lượng siêu thực.
    *   2401.01335 – Self-Play Fine-Tuning (SPIN) · Kỹ thuật fine-tuning LLM tự cải thiện từ dữ liệu SFT gốc.
    *   2401.17172 – U-IO2 · Kiến trúc MLLM thống nhất cho 8 modal, một bước tiến tới AI tổng quát.
    *   2401.04474 – Chain of Code (CoC) · Kết hợp code và LMulator cho reasoning hỗn hợp, rất mạnh mẽ.

7.  **META_REFLECTION**
    Tập hợp các bài báo tháng 1 năm 2024 cho thấy sự tiếp tục đào sâu vào các hướng nghiên cứu đã nổi bật từ cuối năm 2023, đồng thời có những phát triển đáng chú ý:
    *   **Sự trỗi dậy của Kiến trúc thay thế Transformer (Mamba/SSM):** Sau thành công ban đầu của Mamba trong NLP (2312.00752), tháng 1/2024 chứng kiến sự lan tỏa mạnh mẽ của kiến trúc dựa trên State Space Model sang lĩnh vực thị giác (Vim 2401.09417, VMamba 2401.10166) và mô hình ngôn ngữ cấp độ byte (MambaByte 2401.13660). Việc kết hợp MoE với Mamba (MoE-Mamba 2401.04081) cũng cho thấy nỗ lực mở rộng quy mô hiệu quả cho các kiến trúc mới này. Đây là một xu hướng rất đáng chú ý, hứa hẹn phá vỡ sự thống trị của Transformer.
    *   **Tối ưu hóa LLM Inference tiếp tục là ưu tiên:** Nhiều phương pháp mới để tăng tốc độ và giảm bộ nhớ cho LLM inference tiếp tục được đề xuất, từ các kỹ thuật nén KV cache (TOVA 2401.06104, Activation Beacon 2401.03462), mở rộng ngữ cảnh hiệu quả (SelfExtend 2401.01325, E2-LLM 2401.06951), đến các phương pháp speculative/parallel decoding tiên tiến (Medusa 2401.10774, EAGLE 2401.15077) và tối ưu hóa phần cứng (TC-FPx 2401.14112).
    *   **MLLM ngày càng đa năng và tích hợp sâu hơn:** Các mô hình như U-IO2 (2401.17172) và OneLLM (2312.03700 - tháng trước nhưng có liên quan) đang hướng tới việc xử lý và tạo ra ngày càng nhiều loại modal trong một kiến trúc thống nhất. Khả năng grounding và tương tác với giao diện người dùng (DocLLM 2401.00908, CogAgent 2312.08914, WebVoyager 2401.13919, Mobile-Agent 2401.16158) cũng được cải thiện.
    *   **Generative AI cho Video và 3D tiếp tục đột phá:** Lĩnh vực sinh video (STUNet 2401.12945, VideoPoet 2312.14125, MagicVideo-V2 2401.04468) và 3D (BlockFusion 2401.17053, Image Sculpting 2401.01702, DG4D 2312.17142) vẫn rất sôi động với các kiến trúc và kỹ thuật mới nhằm cải thiện chất lượng, khả năng điều khiển và tính nhất quán.
    *   **Self-Improvement và Alignment không cần nhiều dữ liệu người:** Xu hướng LLM tự cải thiện thông qua self-reward (2401.10020), self-play (SPIN 2401.01335), hoặc học từ dữ liệu tổng hợp/phản hồi AI (ReST meets ReAct 2312.10003, CPO 2401.08417, Constitutional DPO 2401.17268) đang mạnh mẽ, nhằm giảm sự phụ thuộc vào dữ liệu chú thích đắt đỏ của con người.
    *   **Data-Centric AI và Benchmark chất lượng cao:** Tầm quan trọng của dữ liệu chất lượng cao và các quy trình tạo/lọc dữ liệu hiệu quả tiếp tục được nhấn mạnh (WRAP 2401.16380, MATHPILE 2312.17120, CheXinstruct 2401.12208, SceneVerse 2401.09340). Đồng thời, các benchmark mới, thách thức hơn và tập trung vào các khía cạnh cụ thể của AI (TRUSTLLM 2401.05561, TOFU 2401.06121, CMMMU 2401.11944, Unitxt 2401.14019) đang được phát triển để đánh giá AI một cách sâu sắc hơn.
    *   **Model Pruning và Compression:** Các phương pháp nén mô hình sau huấn luyện mà không cần fine-tuning hoặc ít fine-tuning (SliceGPT 2401.15024, LLM-ROM 2312.07046) đang được quan tâm để triển khai LLM hiệu quả.
    *   **Sự mở của cộng đồng:** Việc phát hành các mô hình mạnh mẽ mã nguồn mở (Mixtral 2401.04088, H2O-Danube-1.8B 2401.16818) và các sáng kiến về tính minh bạch (LLM360 2312.06550) tiếp tục thúc đẩy sự phát triển chung của lĩnh vực.

    Nhìn chung, tháng 1 năm 2024 cho thấy một bức tranh AI năng động, với sự tìm tòi các kiến trúc nền tảng mới, nỗ lực giải quyết các thách thức về hiệu quả, khả năng đa phương thức, và tính đáng tin cậy, đồng thời nhấn mạnh vai trò của dữ liệu chất lượng và các phương pháp đánh giá toàn diện.
