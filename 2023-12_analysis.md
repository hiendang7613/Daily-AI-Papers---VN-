1.  **TOPIC_TREE**

    *   NLP (Natural Language Processing)
        *   Large Language Models (LLM) & Multimodal Language Models (MLLM)
            *   Model Architecture & Optimization
                *   Efficient Inference & Architecture
                    *   2312.11514 | Đề xuất chiến lược windowing, row-column bundling và low-rank predictor cho inference LLM trên flash với DRAM hạn chế.
                    *   2312.00752 | Mamba giới thiệu selective SSM (State Space Model) với thuật toán song song hardware-aware, đạt hiệu năng vượt Transformer với chi phí tuyến tính.
                    *   2312.07987 | SwitchHead áp dụng Mixture-of-Experts cho self-attention, giảm số attention map cần tính và lưu trữ.
                    *   2312.04985 | SparQ Attention sử dụng đặc tính heavy-tailed của query để dự đoán attention và mean value reallocation, giảm băng thông inference.
                *   Model Scaling & Initialization
                    *   2312.15166 | Depth-Up-Scaling (DUS) nhân đôi và cắt ghép lớp transformer, kết hợp continued pretraining để mở rộng quy mô LLM.
                    *   2312.09299 | Weight subcloning khởi tạo transformer thu nhỏ bằng cách trích xuất trọng số (dựa trên xếp hạng neuron) từ mô hình lớn.
            *   Training, Fine-tuning & Adaptation
                *   Instruction Tuning & Data Generation
                    *   2312.02120 | OSS-INSTRUCT sinh dữ liệu hướng dẫn từ đoạn mã nguồn mở ngẫu nhiên, tạo bộ Magicoder.
                    *   2312.14187 | CodeSeaXDataset được tạo bằng framework LLM-based Generator-Discriminator và raw code coreset selection cho 4 tác vụ code.
                *   Alignment & Safety
                    *   2312.01552 | URIAL là phương pháp tuning-free alignment dùng in-context learning với restyled examples và system prompt, phân tích token distribution shift.
                    *   2312.14591 | Contrastive Unlikelihood Training (CUT) căn chỉnh LLM với ngôn ngữ phản hồi (judgments) bằng unlikelihood training có trọng số động.
                *   Continual & Multilingual Learning
                    *   2312.00738 | SeaLLM mở rộng từ vựng cho ngôn ngữ Đông Nam Á, kết hợp Pre-train & SFT Hybrid và Self-Preferencing Optimization.
                *   Low-Resource Optimization
                    *   2312.06550 | LLM360 là sáng kiến phát hành LLM (Amber, CrystalCoder) với đầy đủ dữ liệu, mã nguồn, checkpoints và metrics để tăng tính minh bạch và tái lập.
            *   Reasoning & Problem Solving
                *   Program-Augmented Reasoning
                    *   2312.04474 | Chain of Code (CoC) kết hợp viết code/pseudocode và mô phỏng thực thi bằng LMulator (Python interpreter + LM) cho reasoning hỗn hợp.
                *   Self-Improvement & Feedback
                    *   2312.10003 | ReST meets ReAct là khung lặp tự cải tiến cho LLM agent với fine-tuning trên reasoning traces tự sinh và LLM-based ranking làm reward.
                *   Mathematical Reasoning
                    *   2312.09241 | TinyGSM là bộ dữ liệu toán tiểu học tổng hợp (GPT-3.5-turbo) kèm giải pháp Python, sử dụng verifier model để chọn kết quả tốt nhất.
            *   Evaluation & Benchmarking
                *   2312.04724 | CYBER SECEVAL là suite benchmark đánh giá LLM về insecure code generation (ICD) và cyberattack helpfulness (LLM judge).
                *   2312.04333 | Phân tích LLaMA bằng probing tasks đa dạng (tính toán, suy luận toán/logic, kiến thức) theo chiều ngang (kích thước) và dọc (lớp).
                *   2312.07910 | PromptBench là thư viện Python hợp nhất để đánh giá LLM, tích hợp mô hình, dataset, prompt, tấn công đối nghịch và giao thức đánh giá.
            *   Multimodal Capabilities
                *   Vision-Language Models (VLM) / Multimodal LLMs (MLLM)
                    *   Architectures & Training
                        *   2312.16862 | TinyGPT-V tích hợp Phi-2 LLM với frozen vision encoders qua Q-Former và linear projections, cùng normalization scheme mới và training 4 giai đoạn.
                        *   2312.08914 | CogAgent là VLM với High-Resolution Cross-Module xử lý ảnh GUI 1120x1120, tạo dataset CCS400K và định nghĩa GUI grounding tasks.
                        *   2312.17172 | U-IO2 là encoder-decoder transformer duy nhất cho 8 modal, với 2D RoPE, Multimodal Mixture of Denoisers (MoD) và dynamic packing.
                        *   2312.03700 | OneLLM sử dụng unified multimodal encoder (CLIP-ViT) và Universal Projection Module (UPM) với mixture-of-experts cho 8 modal.
                        *   2312.07533 | VILA fine-tune toàn bộ LLM (Llama-2) với linear projector và dữ liệu xen kẽ ảnh-văn bản (MMC4) để cải thiện deep embedding alignment và ICL.
                    *   Grounded Captioning & Reasoning
                        *   2312.09237 | PixelLLM tích hợp MLP song song lớp từ điển LLM để hồi quy tọa độ pixel cho từng token, hỗ trợ grounded captioning.
                        *   2312.14233 | VCoder là adapter cho MLLM, xử lý bản đồ phân đoạn/độ sâu làm đầu vào kiểm soát phụ trợ, huấn luyện trên dataset COST.
                    *   Data Generation & Augmentation
                        *   2312.08578 | DCI dataset chứa ảnh với chú thích dày đặc và mask phân cấp, cùng pipeline tóm tắt đệ quy (Llama2) và subcrop-caption matching task.
                    *   SVG Generation
                        *   2312.11556 | StarVector là MLLM (ViT + StarCoder) sinh trực tiếp code SVG từ ảnh/text, huấn luyện trên SVG-STACK dataset.
                *   Audio-Language Models
                    *   2312.15821 | Audiobox là mô hình flow-matching thống nhất cho nhiều modal audio, với description/example-based prompting, Audiobox SSL và Joint-CLAP.
            *   Model Deployment & Serving
                *   2312.08361 | PETALS là hệ thống inference/fine-tuning LLM phân tán trên Internet, với dual attention caches, D*Lite routing và load-balancing phi tập trung.
            *   Weak-to-Strong Learning
                *   2312.09390 | Nghiên cứu weak-to-strong generalization, đề xuất auxiliary confidence loss và bootstrapping supervision để LLM mạnh học từ nhãn của LLM yếu.
            *   Prompt Engineering
                *   2312.16171 | Hệ thống hóa 26 nguyên tắc hướng dẫn tối ưu prompt cho LLM, phân loại thành 5 nhóm và đánh giá thực nghiệm.
            *   Retrieval Augmented Generation (RAG)
                *   2312.05708 | Context Tuning là thành phần mới trong RAG, truy xuất ngữ cảnh liên quan (LambdaMART-RRF) trước khi truy xuất công cụ.
    *   Computer Vision (CV)
        *   Image Generation & Synthesis
            *   Diffusion Models
                *   Efficient Inference & Acceleration
                    *   2312.12491 | StreamDiffusion tối ưu pipeline diffusion với Stream Batch, Residual CFG, Input-Output Queue, Stochastic Similarity Filter và Pre-computation cache.
                    *   2312.00858 | DeepCache sử dụng caching động đặc trưng mức cao trong U-Net và Automatic Caching Schedule để tăng tốc diffusion.
                    *   2312.03209 | Block Caching tái sử dụng output của từng block U-Net, với Automatic Caching Schedule và Scale-Shift Alignment.
                *   Personalized & Controllable Generation
                    *   2312.04461 | PhotoMaker sử dụng stacked ID embedding (CLIP) và LoRA residuals để sinh ảnh người cá nhân hóa từ nhiều ảnh ID.
                    *   2312.02663 | Mô hình hybrid-guidance tune-free tích hợp stylized image, ArcFace ID embedding và text prompt, với multi-identity cross-attention.
                    *   2312.13691 | DreamTuner sử dụng subject-encoder, subject-encoder-attention và self-subject-attention để tùy biến chủ thể trong T2I.
                    *   2312.07536 | FreeControl sử dụng PCA trên đặc trưng trung gian để tạo structure/appearance guidance zero-shot cho diffusion model.
                *   Plugin Compatibility & Model Upgrading
                    *   2312.02238 | X-Adapter là kiến trúc adapter chung (ResNet mapping layers) cho phép plugin của SDv1.5 hoạt động trên SDXL mà không cần tái huấn luyện plugin.
                *   Efficient Prior Learning
                    *   2312.04655 | ECLIPSE học text-to-image prior không khuếch tán bằng contrastive learning (PriorTransformer) cho khung unCLIP.
                *   Training Dynamics Optimization
                    *   2312.02696 | Chuẩn hóa magnitude activations, weights, updates trong U-Net (ADM) bằng weight normalization, forced weight normalization và EMA hậu kiểm.
        *   Video Generation & Synthesis
            *   Diffusion-based & Controllable
                *   2312.00845 | VMC (Video Motion Customization) fine-tune one-shot temporal attention với motion distillation objective và appearance-invariant prompts.
                *   2312.05107 | DreaMoving (Video ControlNet với motion block, Content Guider đa nguồn, huấn luyện 3 giai đoạn) cho sinh video người có kiểm soát.
                *   2312.03641 | MotionCtrl là trình điều khiển chuyển động thống nhất (CMCM cho camera, OMCM cho object) cho LVDM, dạng adapter.
                *   2312.03793 | AnimateZero tách riêng điều khiển appearance (Spatial Appearance Control) và motion (Positional-Corrected Window Attention) zero-shot cho video diffusion.
                *   2312.12490 | InstructVideo fine-tune video diffusion bằng editing inference một phần, Segmental Video Reward (SegVR) và Temporally Attenuated Reward (TAR).
                *   2312.02087 | VideoSwap sử dụng semantic point correspondence (DIFT, TAP-Vid/Co-Tracker) và sparse motion features cho video subject swapping.
            *   Efficient & Long Video Generation
                *   2312.09109 | VideoLCM mở rộng Latent Consistency Model cho video, sử dụng consistency distillation và CFG cố định.
                *   2312.07537 | FreeInit là chiến lược sampling thời gian suy luận, tinh chỉnh nhiễu khởi tạo video diffusion bằng lọc tần số không gian-thời gian.
            *   LLM-based Multimodal Generation
                *   2312.14125 | VideoPoet là LLM decoder-only đa phương thức (text, image, video, audio) với pre-training đa nhiệm và module super-resolution không tự hồi quy.
        *   3D Content Generation & Understanding
            *   Text-to-3D & Image-to-3D
                *   2312.06655 | Sherpa3D kết hợp coarse 3D prior (3D diffusion) với 2D lifting (SDS), sử dụng Structural Guidance và Semantic Guidance.
            *   Avatar Modeling & Animation
                *   2312.15430 | Mach là pipeline tạo avatar 3D từ text, kết hợp LLM (thuộc tính), ControlNet (ảnh tham chiếu), dense landmarks, triplane maps và NeuralHDHair.
                *   2312.13578 | DREAM-Talk là framework hai giai đoạn (EmoDiff diffusion transformer + Lip Refinement Network) cho talking head cảm xúc, đồng bộ môi.
                *   2312.03704 | Avatar 3D Gaussian splatting động, relightable (learnable radiance transfer), với mã hóa UV (CNN, CVAE) và mô hình mắt tường minh.
                *   2312.11461 | GAvatar là avatar động dựa trên primitive-based implicit Gaussian representation, implicit attribute fields và SDF-based implicit mesh learning.
            *   Dynamic Scene Representation
                *   2312.03029 | HHAvatar là head avatar 3D Gaussian động, điều khiển biểu cảm và tóc (MLP temporal, occlusion perception), khởi tạo bằng SDF+DMTet.
                *   2312.17142 | DG4D kết hợp static 3D GS (DreamGaussianHD) và HexPlane-based Gaussian deformation, với driving video supervision và V2V texture refinement.
            *   3D Shape Representation & Generation
                *   2312.09222 | Mosaic-SDF xấp xỉ SDF bằng tập lưới cục bộ, kết hợp Flow Matching và transformer permutation-equivariant cho sinh hình 3D.
            *   Texture Synthesis & Mapping
                *   2312.13913 | Paint3D là khung coarse-to-fine (2D diffusion + back-projection; UV Inpainting diffusion + UVHD diffusion) sinh UV map 2K lighting-less.
            *   3D Vision Benchmarking
                *   2312.16256 | DL3DV-10K là dataset cảnh thực tế đa góc nhìn 4K, DL3DV-140 là benchmark đánh giá NVS.
                *   2312.10763 | M3DBench là benchmark 3D MLM với công thức chỉ thị đa phương thức xen kẽ và kiến trúc baseline thống nhất.
        *   Image Segmentation
            *   Foundation Models & Unified Segmentation
                *   2312.15715 | UniRef++ là mô hình chia sẻ tham số cho 4 tác vụ phân đoạn tham chiếu (RIS, FSS, RVOS, VOS) với UniFusion module và Deformable-DETR.
                *   2312.09579 | MobileSAMv2 sử dụng object-aware prompt sampling (YOLOv8) và NMS cho SAM mask decoder, tăng tốc SegEvery.
                *   2312.07661 | CaR là khung lặp zero-shot (CLIP gradCAM proposals + CLIP classifier) cho phân đoạn khái niệm từ ngôn ngữ tự nhiên.
                *   2312.17243 | U2Seg là framework thống nhất cho phân đoạn ảnh không giám sát (semantic, instance, panoptic) với pseudo-labeling và self-training.
            *   Point Cloud Segmentation
                *   2312.10035 | PTv3 sử dụng point cloud serialization (space-filling curves), patch attention, Shuffle Order và xCPE cho 3D semantic segmentation.
        *   Vision-Language Pre-training & Representation
            *   2312.03818 | Alpha-CLIP mở rộng CLIP ViT với kênh alpha và Alpha Conv, huấn luyện trên dữ liệu RGBA-text để region-aware.
            *   2312.14238 | InternVL với InternViT-6B và QLLaMA middleware, cùng chiến lược Progressive Image-Text Alignment 3 giai đoạn.
        *   Image Editing
            *   2312.13834 | Fairy sử dụng anchor-based cross-frame attention và equivariant finetuning cho video editing nhất quán thời gian.
            *   2312.14091 | PAIntA (Prompt-Aware Introverted Attention) và RASG (Reweighting Attention Score Guidance) cho text-guided inpainting.
        *   Monocular Depth Estimation
            *   2312.13252 | DMD là mô hình diffusion cho zero-shot metric depth estimation, với log-scale depth, FOV augmentation và v-parameterization.
        *   Other
            *   2312.00869 | SAM-CLIP augmentation với hybrid feature mixer và weakly-supervised pre-training cho regional captioning.
            *   2312.06109 | Vary-tiny kết hợp ViTDet vocabulary network và OPT-125M decoder cho tác vụ thị giác mật độ cao (document/charts).
    *   Audio & Speech Processing
        *   Speech Synthesis & Generation
            *   2312.03491 | Bridge-TTS sử dụng Schrodinger bridge tractable (text latent và mel-spectrogram) cho TTS, với công thức đóng cửa và sampler hiệu quả.
            *   2312.09767 | DreamTalk là mô hình diffusion cho talking head cảm xúc, với style-aware lip expert và diffusion-based style predictor.
        *   Audio Generation
            *   2312.08723 | StemGen là mô hình transformer phi tự hồi quy cho sinh nhạc stem-based, với multi-source CFG và causal-bias iterative decoding.
    *   Human-Computer Interaction (HCI)
        *   Multimodal Agent Systems & GUI Automation
            *   2312.13771 | AppAgent là tác nhân đa phương thức điều khiển ứng dụng di động qua GUI cấp thấp, với khám phá có hướng mục tiêu và tài liệu kiến thức động.
        *   Hybrid Language-Graphical Interfaces
            *   2312.00763 | ExploreLLM kết hợp phân tách tác vụ tự động (LLM) với GUI dạng schema (Node tree) cho khám phá tác vụ phức tạp.
    *   AI Alignment & Safety
        *   Superalignment & Weak-to-Strong Learning
            *   2312.09390 | Nghiên cứu weak-to-strong generalization, đề xuất auxiliary confidence loss và bootstrapping supervision.
    *   Machine Learning Systems
        *   Model Compression
            *   2312.07046 | LLM-ROM nén LLM theo từng lớp dựa trên Reduced Order Modeling (PCA trên feature maps) và tái tham số hóa low-rank.
    *   Other
        *   2312.06585 | ReSTEM là phương pháp self-training (EM) cho RL với ngôn ngữ, dùng tín hiệu thưởng nhị phân và fine-tune base model.

2.  **SOTA_HIGHLIGHTS**

    | Rank | PaperID   | Keywords (≤ 5)                                          | Đột phá                                                                                                                                  | Ảnh hưởng                                                                                                                                                              |
    | :--- | :-------- | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | 1    | 2312.00752 | Mamba, Selective SSM, Hardware-aware, Linear Scaling    | Kiến trúc sequence model mới (Mamba) dựa trên selective state space model, đạt hiệu năng Transformer với độ phức tạp tuyến tính, vượt trội ở chuỗi dài. | Có khả năng thay thế Transformer làm backbone cho nhiều modal, đặc biệt là ngôn ngữ, genomics, audio, nơi chuỗi dài là phổ biến. Mở ra kỷ nguyên mới cho sequence modeling. |
    | 2    | 2312.11805 | Gemini, Natively Multimodal, Image/Audio/Video Input/Output | Mô hình đa phương thức tự nhiên quy mô lớn, xử lý xen kẽ nhiều loại dữ liệu và tạo ra output đa phương thức trong cùng một decoder Transformer. | Đẩy mạnh ranh giới của MLLM, cho thấy tiềm năng của một mô hình nền tảng thực sự đa năng, có khả năng hiểu và tạo ra nhiều loại thông tin.                         |
    | 3    | 2312.12491 | StreamDiffusion, Real-time Diffusion, Pipeline Optimization | Tối ưu toàn pipeline diffusion (Stream Batch, RCFG, I/O Queue, Stochastic Filter, Cache) cho phép sinh ảnh tương tác real-time.             | Mang diffusion model đến gần hơn với các ứng dụng tương tác thời gian thực, thay đổi cách người dùng tương tác với công cụ tạo ảnh AI.                                  |
    | 4    | 2312.14125 | VideoPoet, LLM for Video, Multimodal Pre-training, Autoregressive | LLM decoder-only đa phương thức sinh video trực tiếp từ text/image/audio, với pre-training đa nhiệm và module super-resolution.           | Cho thấy sức mạnh của kiến trúc LLM trong việc thống nhất hóa các tác vụ tạo sinh đa phương thức, đặc biệt là video, với chất lượng và khả năng kiểm soát cao.        |
    | 5    | 2312.04474 | Chain of Code (CoC), LMulator, Hybrid Reasoning         | Kết hợp viết code/pseudocode và mô phỏng thực thi bằng LMulator (interpreter + LM) cho reasoning hỗn hợp semantic và arithmetic.           | Cung cấp một phương pháp reasoning mạnh mẽ và linh hoạt hơn cho LLM, kết hợp tính chính xác của code và khả năng hiểu ngôn ngữ tự nhiên.                               |
    | 6    | 2312.17172 | U-IO2, Unified Multimodal, 8 Modalities, MoD, Dynamic Packing | Encoder-decoder Transformer duy nhất xử lý 8 modal (text, image, audio, video, action, etc.) với Multimodal Mixture of Denoisers.        | Một bước tiến lớn hướng tới một mô hình AI tổng quát thực sự, có khả năng hiểu và tạo ra thông tin qua nhiều kênh cảm giác và hành động.                               |
    | 7    | 2312.12456 | PowerInfer, Neuron-level Offloading, Sparse Activation  | Engine suy luận hybrid GPU-CPU ở mức chi tiết từng neuron, dựa trên phân bố power-law của activation để offload neuron "lạnh" sang CPU.     | Giải quyết hiệu quả bài toán inference LLM lớn trên phần cứng hạn chế, tăng tốc độ đáng kể mà vẫn giữ độ chính xác.                                                    |
    | 8    | 2312.03704 | Relightable Gaussian Avatar, Learnable Radiance Transfer, UV CVAE | Avatar đầu động 3D Gaussian splatting, relightable real-time với learnable radiance transfer và điều khiển biểu cảm qua CVAE trên UV.     | Đạt được chất lượng và khả năng tương tác cao cho avatar 3D, mở ra ứng dụng cho VR/AR và telepresence.                                                                |

3.  **NOVEL_TECH_CONTRIBUTIONS**

    *   **2312.11514 – Windowing FFN cache & Row-column bundling & Low-rank FFN predictor – Suy nghĩ:** Các kỹ thuật này tối ưu hóa việc truy cập flash cho inference LLM trên thiết bị DRAM hạn chế, rất thực tế và hướng phần cứng.
    *   **2312.00752 – Selective SSM & Hardware-aware parallel recurrent algorithm – Suy nghĩ:** Phá vỡ giới hạn LTI của SSM và thiết kế thuật toán song song hiệu quả cho GPU là một đột phá lớn, có thể thay thế attention trong nhiều ứng dụng.
    *   **2312.02120 – OSS-INSTRUCT (sinh instruction data từ mã nguồn mở ngẫu nhiên) – Suy nghĩ:** Một cách tiếp cận mới và hiệu quả để tạo dữ liệu instruction tuning cho code LLM, giảm bias so với các phương pháp dựa trên seed tasks hoặc heuristic.
    *   **2312.12491 – Stream Batch & Residual Classifier-Free Guidance (RCFG) & Stochastic Similarity Filter – Suy nghĩ:** Các tối ưu hóa pipeline thông minh cho diffusion model, đặc biệt RCFG giảm số lần gọi U-Net và SSF tiết kiệm năng lượng là những đóng góp đáng giá cho real-time generation.
    *   **2312.04461 – Stacked ID embedding & LoRA residuals cho cross-attention (PhotoMaker) – Suy nghĩ:** Kết hợp nhiều ảnh ID và tinh chỉnh LoRA trên cross-attention giúp cá nhân hóa T2I hiệu quả mà không cần test-time optimization.
    *   **2312.15166 – Depth-Up-Scaling (DUS) (nhân đôi, cắt ghép lớp transformer + continued pretraining) – Suy nghĩ:** Một phương pháp đơn giản nhưng hiệu quả để mở rộng chiều sâu LLM, ít tốn kém hơn huấn luyện từ đầu.
    *   **2312.13771 – AppAgent (Khám phá có hướng mục tiêu GUI + Tài liệu kiến thức động) – Suy nghĩ:** Agent tự học tương tác với GUI app di động bằng cách tự tạo tài liệu tham khảo là một hướng đi rất hứa hẹn cho tự động hóa.
    *   **2312.09911 – Amphion (Bộ công cụ hợp nhất cho tạo sinh audio/music/speech) – Suy nghĩ:** Mặc dù là toolkit, việc chuẩn hóa và tích hợp nhiều mô hình/tác vụ vào một nền tảng mở là đóng góp quan trọng cho cộng đồng.
    *   **2312.14187 – CodeSeaXDataset (LLM-based Generator-Discriminator cho instruction data) & Tái sử dụng good/bad generation làm few-shot – Suy nghĩ:** Framework tạo dữ liệu instruction cho code LLM chất lượng cao, việc học từ cả ví dụ tốt và xấu là một ý tưởng hay.
    *   **2312.08723 – StemGen (Stem-based music context training & Multi-source CFG & Causal-bias iterative decoding) – Suy nghĩ:** Các kỹ thuật này cho phép sinh nhạc stem-based có điều kiện ngữ cảnh tốt hơn, đặc biệt causal-bias decoding là một cải tiến thú vị.
    *   **2312.14125 – VideoPoet (LLM decoder-only đa phương thức & Pre-training đa nhiệm video có/không chú thích & Super-resolution transformer không tự hồi quy) – Suy nghĩ:** Kiến trúc LLM thống nhất cho nhiều tác vụ video, kết hợp pre-training thông minh và module SR mới lạ.
    *   **2312.11805 – Gemini (Natively multimodal Transformer & In-memory redundant state copies & CoT@k with consensus) – Suy nghĩ:** Kiến trúc đa phương thức từ đầu và các kỹ thuật huấn luyện/suy luận quy mô lớn là những đóng góp quan trọng cho MLLM nền tảng.
    *   **2312.03511 – Kandinsky 3.0 (Pipeline T2I một giai đoạn & U-Net BigGAN-deep block & Sber-MoVQGAN & Adversarial Diffusion Distillation với U-Net discriminator) – Suy nghĩ:** Nhiều cải tiến kiến trúc và distillation cho T2I, đặc biệt Sber-MoVQGAN và cách dùng U-Net downsample làm discriminator.
    *   **2312.12456 – PowerInfer (Neuron-level GPU-CPU offloading dựa trên power-law activation & Adaptive activation predictor & Neuron-aware sparse operators) – Suy nghĩ:** Phân chia offloading ở mức neuron dựa trên tần suất kích hoạt là một cách tiếp cận rất chi tiết và hiệu quả cho LLM inference.
    *   **2312.10003 – ReST meets ReAct (Growing-batch RL trên reasoning traces tự sinh & LLM-based ranking làm reward & LLM-based auto-eval) – Suy nghĩ:** Vòng lặp tự cải tiến cho LLM agent, không cần nhãn người, sử dụng chính LLM để đánh giá và chọn lọc dữ liệu.
    *   **2312.07987 – SwitchHead (Mixture-of-Experts cho self-attention & Sigmoid routing & Chia sẻ Q/K projections) – Suy nghĩ:** Áp dụng MoE cho attention một cách hiệu quả, giảm tính toán và bộ nhớ mà không cần regularization phức tạp.
    *   **2312.04985 – SparQ Attention (Query component selection & Approximate top-k attention & Mean value reallocation & Dynamic softmax temperature) – Suy nghĩ:** Thuật toán attention hiệu quả băng thông, khai thác tính thưa của query và key-value.
    *   **2312.09241 – TinyGSM dataset (GPT-3.5-turbo sinh toán-code) & Verifier model finetuned trên GSM8K – Suy nghĩ:** Tạo dataset tổng hợp quy mô lớn và sử dụng verifier model nhỏ để cải thiện SLM giải toán.
    *   **2312.00845 – VMC (One-shot temporal attention fine-tuning & Motion distillation objective & Appearance-invariant prompts) – Suy nghĩ:** Tùy chỉnh chuyển động video hiệu quả bằng cách chỉ fine-tune temporal attention và sử dụng motion distillation.
    *   **2312.05107 – DreaMoving (Video ControlNet với motion block & Content Guider đa nguồn & Huấn luyện 3 giai đoạn) – Suy nghĩ:** Kiến trúc điều khiển video linh hoạt, kết hợp điều khiển pose/depth và bảo toàn danh tính/phong cách.
    *   **2312.09299 – Weight subcloning (Xếp hạng neuron toàn cục & Loại bỏ/nhân đôi block & Hoán vị trọng số & Tỉ lệ điều chỉnh trọng số) – Suy nghĩ:** Phương pháp khởi tạo LLM nhỏ từ LLM lớn một cách thông minh, không cần huấn luyện lại.
    *   **2312.09222 – Mosaic-SDF (Biểu diễn 3D bằng tập lưới SDF cục bộ & Flow Matching với transformer permutation-equivariant) – Suy nghĩ:** Biểu diễn 3D mới hiệu quả và mô hình sinh dựa trên Flow Matching.
    *   **2312.07910 – PromptBench (Thư viện hợp nhất đánh giá LLM) – Suy nghĩ:** Mặc dù là thư viện, việc tích hợp nhiều thành phần đánh giá vào một framework mở là đóng góp quan trọng cho nghiên cứu LLM.
    *   **2312.07661 – CaR (CLIP-based iterative refinement: gradCAM proposals + visual prompt classifier + CRF/SAM post-processing) – Suy nghĩ:** Khung lặp zero-shot cho phân đoạn khái niệm từ ngôn ngữ, không cần fine-tuning VLM.
    *   **2312.15011 – Gemini vs GPT-4V (Phân tích so sánh định tính) – Suy nghĩ:** Nghiên cứu so sánh chi tiết, cung cấp hiểu biết sâu sắc về khả năng và hạn chế của hai MLLM hàng đầu.
    *   **2312.12490 – InstructVideo (Editing inference một phần DDIM & Segmental Video Reward SegVR & Temporally Attenuated Reward TAR) – Suy nghĩ:** Fine-tune video diffusion hiệu quả bằng cách chỉnh sửa một phần và sử dụng cơ chế reward thông minh.
    *   **2312.09237 – PixelLLM (MLP song song lớp từ điển LLM hồi quy tọa độ pixel & Two-way prompt feature extractor) – Suy nghĩ:** Cho phép LLM sinh caption kèm vị trí pixel cho từng từ, một bước tiến trong grounded captioning.
    *   **2312.03491 – Bridge-TTS (Schrodinger bridge tractable text-mel & Công thức đóng cửa Ψt, bΨt, pt & Sampler SDE/ODE) – Suy nghĩ:** Framework TTS mới dựa trên Schrodinger bridge, có cơ sở lý thuyết vững chắc và sampler hiệu quả.
    *   **2312.03818 – Alpha-CLIP (Kênh alpha bổ sung & Alpha Conv & Huấn luyện trên RGBA-text data) – Suy nghĩ:** Mở rộng CLIP để nhận diện vùng ảnh qua kênh alpha, plug-and-play.
    *   **2312.02696 – Magnitude-preserving learned layers & Forced weight normalization & Post-hoc EMA (ADM U-Net) – Suy nghĩ:** Các điều chỉnh kiến trúc và huấn luyện giúp ổn định và cải thiện diffusion model.
    *   **2312.02201 – ImageDream (Global/Local/Pixel Controllers & Image-Prompt SDS) – Suy nghĩ:** Ba mức điều khiển ảnh prompt và SDS tinh chỉnh cho image-conditioned multi-view diffusion.
    *   **2312.09390 – Weak-to-strong generalization (PGR metric & Auxiliary confidence loss & Bootstrapping supervision) – Suy nghĩ:** Các kỹ thuật giúp mô hình mạnh học hiệu quả từ nhãn của mô hình yếu, quan trọng cho superalignment.
    *   **2312.04474 – LMulator (Code interpreter + LM cho mô phỏng dòng code không thực thi) – Suy nghĩ:** Module cốt lõi của CoC, cho phép LLM xử lý các tác vụ semantic trong khi vẫn tận dụng interpreter cho code.
    *   **2312.03704 – Relightable Gaussian Avatar (Learnable radiance transfer: SH diffuse + SG specular & UV CVAE & Mô hình mắt tường minh) – Suy nghĩ:** Kết hợp Gaussian splatting với PRT học được và mô hình mắt chi tiết cho avatar relightable real-time.
    *   **2312.02663 – Hybrid-guidance (Stylized image + ArcFace ID + text) & Multi-identity cross-attention (phân vùng) – Suy nghĩ:** Điều khiển T2I linh hoạt với nhiều loại input và xử lý đa danh tính chính xác.
    *   **2312.16862 – TinyGPT-V (Q-Former + 2 linear projections cho Phi-2 & RMSNorm + QKNorm & Training 4 giai đoạn) – Suy nghĩ:** Kiến trúc và chiến lược huấn luyện tối ưu cho MLLM nhỏ, hiệu quả tài nguyên.
    *   **2312.08914 – CogAgent (High-Resolution Cross-Module & CCS400K dataset & GUI grounding tasks REG/REC) – Suy nghĩ:** VLM xử lý ảnh GUI độ phân giải cao và dataset chuyên biệt cho GUI agent.
    *   **2312.17172 – U-IO2 (2D RoPE & Multimodal Mixture of Denoisers MoD & Autoregressive with dynamic masking & Dynamic packing) – Suy nghĩ:** Nhiều cải tiến kiến trúc và huấn luyện cho MLLM thống nhất, xử lý hiệu quả nhiều modal và tác vụ.
    *   **2312.15430 – Mach (LLM+ControlNet tạo prompt hình ảnh & Dense landmarks & Triplane maps & Texture multi-res + delighting & NeuralHDHair) – Suy nghĩ:** Pipeline toàn diện, phức tạp nhưng cho kết quả avatar 3D từ text rất chi tiết.
    *   **2312.13578 – DREAM-Talk (EmoDiff diffusion transformer & Style-aware lip expert & Diffusion-based style predictor & Long-term Dynamic Sampling) – Suy nghĩ:** Tạo talking head cảm xúc với sự đồng bộ môi tốt, sử dụng diffusion và các module chuyên biệt.
    *   **2312.06585 – ReSTEM (EM-based self-training cho RL với ngôn ngữ, thưởng nhị phân, fine-tune base model) – Suy nghĩ:** Phương pháp self-training đơn giản nhưng hiệu quả cho RLHF, không cần dữ liệu con người phức tạp.
    *   **2312.13691 – DreamTuner (Subject-encoder & Subject-encoder-attention & Self-subject-attention & ControlNet depth-based training) – Suy nghĩ:** Tùy biến chủ thể trong T2I hiệu quả bằng cách tách biệt học appearance và chi tiết.
    *   **2312.13252 – DMD (Log-scale depth & FOV augmentation & v-parameterization diffusion) – Suy nghĩ:** Mô hình diffusion zero-shot cho ước lượng độ sâu metric, với các cải tiến về tham số hóa và augmentation.
    *   **2312.08361 – PETALS (Dual attention caches & D*Lite routing & Load-balancing phi tập trung) – Suy nghĩ:** Hệ thống inference LLM phân tán chịu lỗi, tối ưu cho môi trường Internet.
    *   **2312.02238 – X-Adapter (Mapping layers ResNet bridge SDv1.5 và SDXL & Huấn luyện plugin-free & Suy diễn hai giai đoạn) – Suy nghĩ:** Giải pháp thông minh để plugin cũ tương thích với diffusion model mới mà không cần tái huấn luyện plugin.
    *   **2312.17120 – MATHPILE (Quy trình xử lý dữ liệu toán học đa giai đoạn & Quy tắc heuristic lọc & Phát hiện nhiễm bẩn) – Suy nghĩ:** Xây dựng kho ngữ liệu toán học chất lượng cao, quy trình xử lý dữ liệu chi tiết và minh bạch.
    *   **2312.13834 – Fairy (Anchor-based cross-frame attention & Equivariant finetuning) – Suy nghĩ:** Cơ chế attention hiệu quả cho video editing, duy trì tính nhất quán thời gian và giảm bộ nhớ.
    *   **2312.09767 – DreamTalk (Diffusion cho chuyển động khuôn mặt & Style-aware lip expert & Diffusion-based style predictor) – Suy nghĩ:** Một cách tiếp cận khác cho talking head cảm xúc, cũng dựa trên diffusion và các module chuyên gia.
    *   **2312.07537 – FreeInit (Tinh chỉnh nhiễu khởi tạo video diffusion bằng lọc tần số không gian-thời gian) – Suy nghĩ:** Chiến lược sampling tại thời gian suy luận, cải thiện tính nhất quán video mà không cần huấn luyện lại.
    *   **2312.00589 – Merlin (Trajectory objective & Foresight Pre-Training FPT & Foresight Instruction-Tuning FIT & Convolutional modality alignment) – Suy nghĩ:** MLLM dự đoán tương lai bằng cách học từ trajectory, với kiến trúc và chiến lược huấn luyện mới.
    *   **2312.03029 – HHAvatar (Gaussian 3D động & Module động học tóc & Biến dạng toàn học MLP & Khởi tạo SDF+DMTet) – Suy nghĩ:** Tạo avatar 3D Gaussian splatting động chất lượng cao, với mô hình tóc và biến dạng chi tiết.
    *   **2312.09109 – VideoLCM (Consistency distillation cho video latent & CFG cố định & Plug-and-play T2V/D2V) – Suy nghĩ:** Mở rộng LCM cho video, tăng tốc độ sinh video đáng kể.
    *   **2312.13913 – Paint3D (Progressive Coarse Texture Generation & UV Inpainting diffusion & UVHD diffusion & Position map encoder) – Suy nghĩ:** Pipeline coarse-to-fine sinh UV map 2K lighting-less cho mô hình 3D.
    *   **2312.09579 – MobileSAMv2 (Object-aware prompt sampling YOLOv8 & NMS & Box prompt cho SAM decoder) – Suy nghĩ:** Tăng tốc SAM SegEvery bằng cách dùng detector sinh prompt bounding box.
    *   **2312.06662 – W.A.L.T (Causal 3D CNN autoencoder & Window Attention Latent Transformer & AdaLN-LoRA & Autoregressive latent diffusion) – Suy nghĩ:** Kiến trúc Transformer hiệu quả cho sinh video/ảnh trong không gian latent thống nhất.
    *   **2312.06655 – Sherpa3D (Structural Guidance từ normal map & Semantic Guidance từ CLIP feature & Annealing guidance weight) – Suy nghĩ:** Kết hợp 3D prior và 2D lifting cho text-to-3D, cải thiện hình học và ngữ nghĩa.
    *   **2312.10763 – M3DBench (Công thức chỉ thị đa phương thức xen kẽ & Kiến trúc baseline thống nhất cho 3D MLM) – Suy nghĩ:** Benchmark và baseline cho 3D MLLM, hỗ trợ nhiều loại prompt 3D.
    *   **2312.00763 – ExploreLLM (Phân tách tác vụ tự động LLM + GUI schema & Node tree & Cá nhân hóa toàn cục & Tổng kết tự động) – Suy nghĩ:** Giao diện hybrid ngôn ngữ-đồ họa cho LLM, hỗ trợ khám phá tác vụ phức tạp.
    *   **2312.16886 – Lightweight Downsample Projector (LDP) (Depth-wise + point-wise conv, residual) – Suy nghĩ:** Module chiếu nhẹ nhàng giảm token thị giác cho MLLM, hiệu quả cho thiết bị biên.
    *   **2312.15715 – UniRef++ (UniFusion module multiway-fusion & Deformable-DETR & Online RVOS) – Suy nghĩ:** Framework thống nhất cho nhiều tác vụ phân đoạn tham chiếu, chia sẻ tham số hiệu quả.
    *   **2312.14238 – InternVL (InternViT-6B + QLLaMA middleware & Progressive Image-Text Alignment 3 giai đoạn) – Suy nghĩ:** Kiến trúc VLM quy mô lớn với middleware mạnh mẽ và chiến lược alignment đa giai đoạn.
    *   **2312.13401 – Time vectors (θt - θpre) & Nội suy/Phép tính analog time vectors – Suy nghĩ:** Phương pháp hiệu quả để LLM thích ứng với sự thay đổi của ngôn ngữ theo thời gian.
    *   **2312.06109 – Vary-tiny (ViTDet vocabulary network + OPT-125M decoder & Tích hợp song song vocab mới và CLIP-ViT) – Suy nghĩ:** Mở rộng từ vựng thị giác cho MLLM, cải thiện khả năng xử lý tác vụ mật độ cao.
    *   **2312.04655 – ECLIPSE (Non-diffusion prior PriorTransformer & Projection MSE + CLIP contrastive loss) – Suy nghĩ:** Học T2I prior hiệu quả bằng contrastive learning, giảm đáng kể tham số và dữ liệu.
    *   **2312.00869 – Hybrid feature mixer (SAM encoder + causal LM) & Weakly-supervised pre-training cho regional captioning – Suy nghĩ:** Mở rộng SAM cho regional captioning một cách hiệu quả tham số.
    *   **2312.17243 – U2Seg (Pseudo-label semantic cho instance mask & Semantic-aware copy-paste augmentation & Pseudo-label panoptic & Self-training đa nhiệm) – Suy nghĩ:** Framework thống nhất cho phân đoạn ảnh không giám sát, xử lý đồng thời semantic, instance, panoptic.
    *   **2312.13964 – PIA (Condition module plug-and-play & Inter-frame affinity map & Huấn luyện đồng thời condition module và temporal alignment) – Suy nghĩ:** Framework plug-and-play cho cá nhân hóa T2I model để tạo video, tách biệt appearance và motion.
    *   **2312.11392 – SC-TUNER/CSC-TUNER (Adapter nhẹ chỉnh sửa skip connection U-Net & Cascade Dense Convolution) & SCEDIT FRAMEWORK – Suy nghĩ:** Fine-tuning và điều khiển diffusion model hiệu quả bằng cách chỉ chỉnh sửa skip connections.
    *   **2312.11370 – G-LLaVA (Geo170K dataset & Alignment đa phương thức với QA đối nghịch & Instruction tuning với 4 chiến lược tổng hợp) – Suy nghĩ:** Pipeline tạo dữ liệu và fine-tuning MLLM chuyên biệt cho giải toán hình học.
    *   **2312.10240 – RAHF (RichHF-18K dataset & Transformer đa modal dự đoán heatmap/điểm số/từ khóa & Augmented prompt) – Suy nghĩ:** Mô hình reward chi tiết cho T2I, cung cấp heatmap và từ khóa giải thích lỗi.

4.  **GAPS_AND_OPPORTUNITIES**
    *   **Hiệu quả và Khả năng mở rộng của LLM/MLLM:**
        *   *Gaps:* Các kiến trúc mới như Mamba (2312.00752) hứa hẹn nhưng cần thêm kiểm chứng trên quy mô cực lớn và đa dạng tác vụ. Việc tối ưu inference cho các mô hình này (ví dụ: selective SSM) vẫn cần nghiên cứu. Các phương pháp nén/tăng tốc hiện tại (PowerInfer 2312.12456, FlashDecoding++ 2311.01282 (tháng trước), SparQ 2312.04985, LLM-ROM 2312.07046) thường tập trung vào một khía cạnh (ví dụ: FFN, attention) hoặc một loại phần cứng cụ thể.
        *   *Opportunities:* Phát triển các thuật toán song song và tối ưu hóa phần cứng chuyên biệt cho các kiến trúc mới như Mamba. Kết hợp nhiều kỹ thuật nén (quantization, pruning, low-rank factorization, efficient architecture) một cách có hệ thống. Nghiên cứu về "model scaling laws" cho các kiến trúc phi Transformer.
    *   **Generative AI (3D, Video, Audio): Độ nhất quán, Khả năng điều khiển và Tính tương tác:**
        *   *Gaps:* Sinh video/3D dài, nhất quán về mặt thời gian và không gian, đồng thời cho phép điều khiển chi tiết vẫn là thách thức lớn (VideoPoet 2312.14125, W.A.L.T 2312.06662, DG4D 2312.17142, HHAvatar 2312.03029). Các phương pháp hiện tại thường gặp vấn đề về flickering, mất chi tiết hoặc khó kiểm soát chuyển động phức tạp (VMC 2312.00845, MotionCtrl 2312.03641, AnimateZero 2312.03793). Tương tác real-time với các mô hình sinh (StreamDiffusion 2312.12491) mới ở giai đoạn đầu.
        *   *Opportunities:* Kiến trúc sinh video/3D mới có khả năng mô hình hóa quan hệ không gian-thời gian tốt hơn. Các phương pháp điều khiển đa modal (kết hợp text, sketch, pose, audio) tinh vi hơn. Tích hợp các nguyên lý vật lý vào quá trình sinh để tăng tính thực tế. Phát triển các kỹ thuật rendering và streaming hiệu quả cho nội dung 3D/video động.
    *   **Multimodal Understanding & Reasoning sâu:**
        *   *Gaps:* Khả năng MLLM hiểu sâu mối quan hệ phức tạp giữa các modal, suy luận logic dựa trên thông tin đa phương thức, và thực hiện các tác vụ đòi hỏi kiến thức chuyên ngành (ví dụ: y tế, khoa học, hình học 2312.11370) còn hạn chế. Grounding ở mức độ chi tiết (pixel-level, object parts) vẫn là một bài toán khó (PixelLLM 2312.09237, VCoder 2312.14233).
        *   *Opportunities:* Phát triển các kiến trúc MLLM có khả năng fusion thông tin đa phương thức ở các tầng sâu hơn và theo nhiều cách hơn (U-IO2 2312.17172, OneLLM 2312.03700). Xây dựng các bộ dữ liệu và benchmark mới thách thức hơn cho multimodal reasoning. Nghiên cứu về "multimodal chain-of-thought" và các kỹ thuật prompting tiên tiến cho MLLM.
    *   **Dữ liệu cho AI: Chất lượng, Đa dạng và Tự động hóa:**
        *   *Gaps:* Việc thu thập và chú thích dữ liệu chất lượng cao, đa dạng, đặc biệt cho các ngôn ngữ/miền ít tài nguyên hoặc các tác vụ mới (ví dụ: GUI interaction 2312.08914, 3D scenes 2312.16256) vẫn tốn kém và mất thời gian. Các phương pháp sinh dữ liệu tổng hợp (OSS-INSTRUCT 2312.02120, CodeSeaXDataset 2312.14187, TinyGSM 2312.09241, Geo170K 2312.11370) cần cải thiện về độ tin cậy và khả năng bao phủ.
        *   *Opportunities:* Các kỹ thuật data augmentation và synthesis thông minh hơn, có thể tự động điều chỉnh theo nhu cầu của mô hình và đảm bảo tính đa dạng/chân thực. Phương pháp self-supervised/weakly-supervised learning hiệu quả hơn trên dữ liệu đa phương thức không nhãn. Công cụ và quy trình tự động hóa việc làm sạch, lọc và đánh giá chất lượng dữ liệu (MATHPILE 2312.17120).
    *   **Agent tự trị và Tương tác Người-Máy:**
        *   *Gaps:* Xây dựng các agent có khả năng học hỏi liên tục, thích ứng với môi trường mới, và tương tác tự nhiên với con người (AppAgent 2312.13771, ExploreLLM 2312.00763) vẫn là mục tiêu dài hạn. Việc đảm bảo an toàn và hành vi có thể dự đoán của agent là rất quan trọng.
        *   *Opportunities:* Framework agent tổng quát hơn, có khả năng học các kỹ năng mới và tự cải thiện từ phản hồi (ReST meets ReAct 2312.10003). Nghiên cứu về giao diện tương tác người-máy thông minh hơn, cho phép cộng tác hiệu quả.
    *   **Alignment, An toàn và Đánh giá AI có Trách nhiệm:**
        *   *Gaps:* Các phương pháp alignment hiện tại (DPO, CUT 2312.14591) cần được cải tiến để hiệu quả hơn và ít tốn kém hơn. Việc đánh giá các rủi ro tiềm ẩn như thiên vị, độc hại, thông tin sai lệch một cách toàn diện và tự động còn khó khăn (CYBER SECEVAL 2312.04724, RAHF 2312.10240). Hiện tượng "superficial alignment" (2312.01552) cần được hiểu rõ và giải quyết.
        *   *Opportunities:* Kỹ thuật alignment mới, có thể không cần dữ liệu preference hoặc ít phụ thuộc vào mô hình lớn. Các phương pháp "red teaming" tự động và benchmark an toàn mạnh mẽ hơn. Phát triển các metric đánh giá AI có khả năng diễn giải và chống lại các chiến lược đối phó. Nghiên cứu về weak-to-strong generalization (2312.09390) cho superalignment.
    *   **Foundation Models cho các lĩnh vực chuyên biệt:**
        *   *Gaps:* Việc áp dụng và tùy chỉnh foundation models cho các lĩnh vực cụ thể (ví dụ: khoa học, y tế, tài chính, kỹ thuật) đòi hỏi dữ liệu chuyên ngành và các phương pháp adaptation hiệu quả.
        *   *Opportunities:* Phát triển các kỹ thuật domain adaptation hiệu quả hơn cho LLM/MLLM. Xây dựng các foundation model chuyên biệt cho từng ngành, được huấn luyện trên dữ liệu chất lượng cao của ngành đó.

5.  **FUTURE_IDEAS**

    ✨ **Idea 1: Mamba-based Multimodal Language Models for Ultra-Long Sequence Understanding**
    *   **Motivation:** Mamba (2312.00752) cho thấy tiềm năng xử lý chuỗi siêu dài với độ phức tạp tuyến tính, vượt trội Transformer. Các MLLM hiện tại (U-IO2 2312.17172, VideoPoet 2312.14125) vẫn dựa trên Transformer, gặp khó khăn với video rất dài hoặc tài liệu đa phương thức lớn.
    *   **Key Novelty:** Xây dựng một MLLM nền tảng dựa trên kiến trúc Mamba (hoặc lai Mamba-Transformer) có khả năng xử lý và hiểu các chuỗi đa phương thức (video, audio, text, tài liệu dài) với độ dài lên đến hàng triệu token, mở ra khả năng phân tích phim truyện, sách nói kèm hình ảnh, hoặc các bản ghi tương tác người-máy kéo dài.
    *   **Approach:**
        1.  Thiết kế kiến trúc MLLM với backbone Mamba cho cả encoder và decoder (hoặc chỉ decoder nếu dùng encoder chuyên biệt cho từng modal).
        2.  Phát triển các phương pháp tokenization và embedding hiệu quả cho các luồng dữ liệu đa phương thức dài (ví dụ: video được mã hóa thành chuỗi patch tokens theo không gian-thời gian).
        3.  Huấn luyện trên các bộ dữ liệu chứa chuỗi đa phương thức dài, có thể sử dụng các kỹ thuật pre-training như Multimodal Mixture of Denoisers (MoD) của U-IO2.
        4.  Tập trung vào các tác vụ đòi hỏi hiểu ngữ cảnh dài: tóm tắt video/sách nói dài, trả lời câu hỏi về các sự kiện cách xa nhau trong video, phân tích tương tác phức tạp trong tài liệu.
    *   **Dataset + Metrics:** Tạo hoặc tổng hợp các bộ dữ liệu mới với chuỗi đa phương thức siêu dài (ví dụ: phim có transcript và mô tả cảnh, sách nói kèm hình minh họa, bản ghi hội nghị dài). Metrics: Các benchmark VQA/tóm tắt video dài (ví dụ: ActivityNet QA, YouCook2), ROUGE, BLEU cho tóm tắt, và các metric đánh giá khả năng hiểu ngữ cảnh dài mới.
    *   **Risk/Feasibility:** Cao. Huấn luyện Mamba trên quy mô lớn và dữ liệu đa phương thức phức tạp là thách thức. Việc đảm bảo tính nhất quán và khả năng tổng hợp thông tin qua chuỗi rất dài vẫn cần nghiên cứu sâu.

    ✨ **Idea 2: Interactive 3D Scene Generation and Editing with Real-time Physics-Aware Feedback (Interdisciplinary: Generative AI + Robotics + HCI)**
    *   **Motivation:** Các mô hình sinh 3D (DG4D 2312.17142, HHAvatar 2312.03029, GAvatar 2312.11461, Mosaic-SDF 2312.09222, Paint3D 2312.13913, Sherpa3D 2312.06655) ngày càng mạnh mẽ, nhưng việc tương tác và chỉnh sửa trực quan, đồng thời đảm bảo tính hợp lý vật lý còn hạn chế. AppAgent (2312.13771) cho thấy tiềm năng của agent tương tác GUI.
    *   **Key Novelty:** Một hệ thống cho phép người dùng tương tác (ví dụ: kéo, thả, thay đổi thuộc tính vật liệu) với một cảnh 3D được sinh ra bởi AI trong thời gian thực. Hệ thống sẽ liên tục cập nhật cảnh dựa trên tương tác, đồng thời một "physics engine ngầm" (có thể là một mô hình học sâu hoặc mô phỏng đơn giản hóa) sẽ cung cấp feedback để đảm bảo các thay đổi tuân thủ các quy luật vật lý cơ bản và duy trì tính nhất quán của cảnh.
    *   **Approach:**
        1.  Sử dụng một mô hình sinh 3D nhanh (ví dụ: dựa trên Gaussian Splatting hoặc NeRF được tối ưu) để tạo cảnh ban đầu từ text hoặc ảnh.
        2.  Phát triển giao diện tương tác 3D cho phép người dùng chọn, di chuyển, xoay, thay đổi kích thước đối tượng, hoặc "vẽ" thêm các yếu tố mới.
        3.  Một "Physics-Aware Critic Model" (có thể là một mạng neural được huấn luyện để dự đoán tính ổn định/hợp lý vật lý, hoặc một mô phỏng vật lý đơn giản chạy song song) liên tục đánh giá cảnh.
        4.  Khi người dùng tương tác, mô hình sinh 3D sẽ cập nhật cảnh. Nếu Critic Model phát hiện vi phạm vật lý (ví dụ: đối tượng lơ lửng không có điểm tựa, xuyên qua nhau), nó sẽ cung cấp feedback cho mô hình sinh để điều chỉnh (ví dụ: tự động thêm hỗ trợ, ngăn chặn xuyên thấu) hoặc cảnh báo người dùng.
        5.  Có thể sử dụng LLM để diễn giải ý định của người dùng từ các tương tác phức tạp.
    *   **Dataset + Metrics:** Không có dataset chuẩn. Đánh giá dựa trên user studies về tính dễ sử dụng, khả năng kiểm soát, tính chân thực của tương tác vật lý. Có thể tạo các kịch bản thử nghiệm với các ràng buộc vật lý cụ thể.
    *   **Risk/Feasibility:** Rất cao (Moon-shot). Kết hợp sinh 3D real-time, tương tác trực quan và mô phỏng/đánh giá vật lý là cực kỳ phức tạp. Đảm bảo phản hồi nhanh và chính xác từ Physics-Aware Critic Model là thách thức lớn.

    ✨ **Idea 3: Zero-Shot Transfer of LLM Reasoning Capabilities to Low-Resource Languages via Universal Conceptual Alignment**
    *   **Motivation:** LLM thể hiện khả năng reasoning tốt ở các ngôn ngữ giàu tài nguyên. SeaLLM (2312.00738) là một nỗ lực cho ngôn ngữ Đông Nam Á. Tuy nhiên, việc chuyển giao khả năng reasoning sang ngôn ngữ ít tài nguyên mà không cần fine-tuning lớn vẫn khó khăn.
    *   **Key Novelty:** Phát triển một phương pháp "align" không gian khái niệm (conceptual space) của một LLM đa ngôn ngữ mạnh (ví dụ: Gemini 2312.11805, U-IO2 2312.17172) với không gian của một LLM nhỏ hơn hoặc một ngôn ngữ ít tài nguyên, tập trung vào việc ánh xạ các "primitive reasoning steps" hoặc "conceptual anchors" thay vì chỉ align từ vựng. Mục tiêu là LLM có thể thực hiện zero-shot reasoning trên ngôn ngữ ít tài nguyên bằng cách "suy nghĩ" trong không gian khái niệm đã được align.
    *   **Approach:**
        1.  Xác định một tập hợp các "conceptual primitives" hoặc "reasoning operators" cốt lõi, độc lập ngôn ngữ (ví dụ: causality, if-then-else, spatial relations, temporal order).
        2.  Sử dụng một LLM đa ngôn ngữ mạnh, thu thập các biểu diễn (ví dụ: activation từ các lớp trung gian) khi nó thực hiện các tác vụ reasoning trên các ngôn ngữ giàu tài nguyên, liên kết các biểu diễn này với các conceptual primitives.
        3.  Thiết kế một cơ chế alignment (có thể là một projection layer nhỏ, hoặc một phương pháp tối ưu dựa trên contrastive learning) để ánh xạ các biểu diễn tương tự từ một LLM/ngôn ngữ ít tài nguyên vào không gian conceptual primitives này.
        4.  Khi LLM nhận một câu hỏi reasoning bằng ngôn ngữ ít tài nguyên, nó sẽ ngầm "dịch" vấn đề sang không gian conceptual primitives, thực hiện suy luận trong không gian đó (có thể bằng cách tận dụng khả năng của LLM mạnh thông qua các biểu diễn đã align), rồi "dịch" kết quả trở lại ngôn ngữ gốc.
    *   **Dataset + Metrics:** Sử dụng các benchmark reasoning đa ngôn ngữ (ví dụ: XNLI, XCOPA, hoặc các phiên bản dịch của GSM8K, MATH). Metrics: Zero-shot accuracy trên các ngôn ngữ ít tài nguyên. Có thể phát triển các probing task để đánh giá mức độ alignment của conceptual primitives.
    *   **Risk/Feasibility:** Cao. Định nghĩa và trích xuất conceptual primitives một cách phổ quát là rất khó. Cơ chế alignment cần phải đủ mạnh để bắc cầu qua sự khác biệt ngôn ngữ và năng lực mô hình.

    ✨ **Idea 4: Federated Generation and Evaluation of Cybersecurity Benchmarks for LLMs**
    *   **Motivation:** CYBER SECEVAL (2312.04724) là một bước tiến trong việc đánh giá LLM về an toàn an ninh mạng, nhưng việc tạo và duy trì benchmark tập trung có thể gặp hạn chế về tính đa dạng và cập nhật. LLM360 (2312.06550) thúc đẩy tính minh bạch.
    *   **Key Novelty:** Một framework phi tập trung (federated) cho phép nhiều tổ chức/cá nhân cùng đóng góp vào việc tạo ra các bài kiểm tra (prompts, code snippets, attack scenarios) và đánh giá (sử dụng ICD và LLM judge như trong CYBER SECEVAL) khả năng của LLM về an ninh mạng. Dữ liệu và kết quả đánh giá có thể được tổng hợp một cách an toàn và bảo mật (ví dụ: sử dụng kỹ thuật federated learning cho LLM judge hoặc các phương pháp mã hóa).
    *   **Approach:**
        1.  Thiết kế một giao thức và API chuẩn cho việc đóng góp các mẫu thử nghiệm (insecure code generation prompts, cyberattack helpfulness prompts) và các quy tắc đánh giá (ví dụ: định nghĩa CWE mới cho ICD, tiêu chí mới cho LLM judge).
        2.  Các bên tham gia tự tạo và đánh giá LLM trên tập dữ liệu cục bộ của họ.
        3.  Chỉ các "mô hình" đánh giá (ví dụ: LLM judge đã được fine-tune, các quy tắc ICD mới) hoặc các thống kê tổng hợp (không chứa dữ liệu nhạy cảm) được chia sẻ và hợp nhất tại một máy chủ trung tâm (hoặc một cách phi tập trung hơn).
        4.  Framework có thể tự động tạo ra các biến thể mới của các bài kiểm tra dựa trên các đóng góp để tăng độ khó và đa dạng.
        5.  Sử dụng các kỹ thuật bảo vệ quyền riêng tư (differential privacy, homomorphic encryption) nếu cần thiết khi tổng hợp kết quả.
    *   **Dataset + Metrics:** Không có dataset cố định, mà là một quy trình tạo và mở rộng dataset liên tục. Metrics: Tương tự CYBER SECEVAL (pass rate, compliance rate), nhưng được tổng hợp từ nhiều nguồn.
    *   **Risk/Feasibility:** Trung bình đến Cao. Đảm bảo tính nhất quán và chất lượng của các đóng góp từ nhiều bên là thách thức. Các vấn đề về bảo mật và quyền riêng tư khi chia sẻ mô hình đánh giá hoặc kết quả cần được giải quyết cẩn thận. Tuy nhiên, lợi ích về tính đa dạng và cập nhật của benchmark là rất lớn.

6.  **READING_LIST**

    *   2312.00752 – Mamba · Kiến trúc sequence model mới có tiềm năng thay thế Transformer.
    *   2312.11805 – Gemini · MLLM nền tảng đa phương thức tự nhiên, xử lý và tạo ra nhiều loại dữ liệu.
    *   2312.12491 – StreamDiffusion · Tối ưu pipeline cho diffusion model tương tác real-time.
    *   2312.14125 – VideoPoet · LLM đa phương thức cho sinh video chất lượng cao, có kiểm soát.
    *   2312.04474 – Chain of Code (CoC) · Phương pháp reasoning kết hợp code và LMulator, rất mạnh mẽ.
    *   2312.17172 – U-IO2 · Kiến trúc MLLM thống nhất cho 8 modal, đột phá về tính tổng quát.
    *   2312.12456 – PowerInfer · Engine suy luận LLM hiệu quả, offload neuron-level thông minh.
    *   2312.03704 – Relightable Gaussian Avatar · Tạo avatar 3D chất lượng cao, relightable real-time.
    *   2312.02120 – Magicoder (OSS-INSTRUCT) · Phương pháp sinh dữ liệu instruction tuning cho code LLM từ mã nguồn mở.
    *   2312.09390 – Weak-to-Strong Generalization · Nghiên cứu quan trọng về khả năng học của LLM mạnh từ LLM yếu, nền tảng cho superalignment.

7.  **META_REFLECTION**
    Tập hợp các bài báo tháng 12 năm 2023 cho thấy một số xu hướng phát triển AI rất rõ rệt và thú vị:
    *   **Tìm kiếm kiến trúc thay thế/bổ sung Transformer:** Sự xuất hiện của Mamba (2312.00752) với selective state space model cho thấy nỗ lực nghiêm túc trong việc tìm kiếm các kiến trúc sequence model mới có khả năng mở rộng tuyến tính và hiệu quả hơn Transformer, đặc biệt cho chuỗi dài. Các cải tiến cho attention như SwitchHead (2312.07987) và SparQ Attention (2312.04985) cũng tiếp tục tối ưu hóa thành phần cốt lõi này.
    *   **MLLM ngày càng "bản địa" và đa năng hơn:** Các mô hình như Gemini (2312.11805) và U-IO2 (2312.17172) đang hướng tới việc xử lý và tạo ra nhiều loại modal (văn bản, ảnh, video, audio, thậm chí cả action) trong một kiến trúc thống nhất, thay vì chỉ ghép nối các module. Khả năng grounding (PixelLLM 2312.09237, VCoder 2312.14233) và tương tác GUI (AppAgent 2312.13771, CogAgent 2312.08914) cũng được cải thiện.
    *   **Generative AI tiếp tục thống trị với những bước tiến về chất lượng, khả năng điều khiển và hiệu quả:**
        *   **Video:** Các mô hình như VideoPoet (2312.14125), W.A.L.T (2312.06662), VideoLCM (2312.09109) và các kỹ thuật điều khiển/tùy chỉnh (VMC 2312.00845, DreaMoving 2312.05107, MotionCtrl 2312.03641, AnimateZero 2312.03793, InstructVideo 2312.12490, VideoSwap 2312.02087) cho thấy sự trưởng thành nhanh chóng của lĩnh vực này.
        *   **3D:** Từ tạo hình (Sherpa3D 2312.06655, Mosaic-SDF 2312.09222), avatar động (HHAvatar 2312.03029, GAvatar 2312.11461, Relightable Gaussian Avatar 2312.03704) đến scene động (DG4D 2312.17142), các phương pháp dựa trên diffusion, NeRF và Gaussian Splatting đang được cải tiến liên tục.
        *   **Audio:** Các mô hình như Audiobox (2312.15821), Bridge-TTS (2312.03491), StemGen (2312.08723) và DreamTalk (2312.09767) đang đẩy mạnh khả năng tạo và điều khiển âm thanh/tiếng nói/âm nhạc.
        *   **Tăng tốc Diffusion:** Nhu cầu về inference nhanh cho diffusion model là rất lớn, dẫn đến các giải pháp tối ưu pipeline (StreamDiffusion 2312.12491) và thuật toán (DeepCache 2312.00858, Block Caching 2312.03209, FreeInit 2312.07537).
    *   **Tập trung vào hiệu quả triển khai và tài nguyên hạn chế:** Nhiều nghiên cứu giải quyết vấn đề inference LLM/MLLM trên các thiết bị có bộ nhớ/tính toán hạn chế (2312.11514, PowerInfer 2312.12456, TinyGPT-V 2312.16862, LDP 2312.16886) hoặc tối ưu hóa cho serving quy mô lớn (PETALS 2312.08361).
    *   **Reasoning và Agent tiếp tục là biên giới:** Các phương pháp mới cho reasoning (Chain of Code 2312.04474) và các framework agent tự cải thiện (ReST meets ReAct 2312.10003) cho thấy sự tìm tòi không ngừng để LLM trở nên thông minh và tự chủ hơn.
    *   **Data-centric AI và Instruction Tuning vẫn quan trọng:** Việc tạo ra các bộ dữ liệu chất lượng cao, chuyên biệt (OSS-INSTRUCT 2312.02120, CodeSeaXDataset 2312.14187, TinyGSM 2312.09241, Geo170K 2312.11370, DCI 2312.08578, MATHPILE 2312.17120, DL3DV-10K 2312.16256) và các kỹ thuật instruction tuning hiệu quả vẫn là nền tảng cho sự tiến bộ của LLM/MLLM.
    *   **Alignment và An toàn AI được chú trọng hơn:** Các nghiên cứu về weak-to-strong generalization (2312.09390), alignment không cần fine-tuning (URIAL 2312.01552), và các phương pháp alignment mới (CUT 2312.14591) cùng với các benchmark an toàn (CYBER SECEVAL 2312.04724) cho thấy sự quan tâm ngày càng tăng đến việc xây dựng AI có trách nhiệm.
    *   **Sự minh bạch và khả năng tái lập:** Sáng kiến như LLM360 (2312.06550) và các thư viện mở như Amphion (2312.09911), PromptBench (2312.07910) nhấn mạnh tầm quan trọng của việc chia sẻ tài nguyên và chuẩn hóa quy trình nghiên cứu.

    Nhìn chung, tháng 12/2023 tiếp tục chứng kiến sự phát triển nhanh chóng trên nhiều mặt trận của AI, từ các kiến trúc nền tảng mới, khả năng đa phương thức ngày càng mạnh mẽ, đến các ứng dụng tạo sinh ấn tượng và nỗ lực làm cho AI hiệu quả, an toàn và minh bạch hơn.
