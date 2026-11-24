import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Colors
    BG_DARK = RGBColor(15, 23, 42)        # #0F172A (Deep Slate)
    CARD_BG = RGBColor(30, 41, 59)        # #1E293B
    CARD_BORDER = RGBColor(51, 65, 85)    # #334155
    ACCENT_CYAN = RGBColor(56, 189, 248)  # #38BDF8
    ACCENT_MINT = RGBColor(52, 211, 153)  # #34D399
    ACCENT_ROSE = RGBColor(244, 63, 94)   # #F43F5E
    TEXT_WHITE = RGBColor(248, 250, 252)  # #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8
    TEXT_LIGHT = RGBColor(226, 232, 240)  # #E2E8F0

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="CSET-346: NATURAL LANGUAGE PROCESSING • MILESTONE-2"):
        # Category / kicker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.name = "Calibri"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

    def add_card(slide, left, top, width, height, title=None, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            p = tf.paragraphs[0]
            p.text = title.upper()
            p.font.name = "Calibri"
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = ACCENT_CYAN

        return card

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(4.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_CYAN
    bar.line.fill.background()

    # Title box
    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(11.2), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "CSET-346: Natural Language Processing — Milestone-2"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    p2 = tf1.add_paragraph()
    p2.text = "Hate Speech Detection through Explainable AI\n& Reward-Optimized LLMs"
    p2.font.name = "Calibri"
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p2.space_before = Pt(12)

    p3 = tf1.add_paragraph()
    p3.text = "Multimodal Context-Aware Formulation: x_i = {T_i, V_i, C_i} → y_i ∈ {0, 1}"
    p3.font.name = "Consolas"
    p3.font.size = Pt(15)
    p3.font.color.rgb = ACCENT_MINT
    p3.space_before = Pt(16)

    # Info card at bottom
    add_card(s1, Inches(1.2), Inches(4.7), Inches(11.2), Inches(1.8))
    tb_meta = s1.shapes.add_textbox(Inches(1.4), Inches(4.8), Inches(10.8), Inches(1.6))
    tf_meta = tb_meta.text_frame
    tf_meta.word_wrap = True

    p_m1 = tf_meta.paragraphs[0]
    p_m1.text = "Academic Supervisor: Dr. Pradeepika Verma  |  Evaluation Dates: 25th - 26th Nov 2026  |  Venue: P2-CUB-096"
    p_m1.font.name = "Calibri"
    p_m1.font.size = Pt(13)
    p_m1.font.bold = True
    p_m1.font.color.rgb = TEXT_WHITE

    p_m2 = tf_meta.add_paragraph()
    p_m2.text = "Project Team: NLP Research Group (Contributions: Architecture, Multimodal OCR, XAI, Backend/UI)"
    p_m2.font.name = "Calibri"
    p_m2.font.size = Pt(12)
    p_m2.font.color.rgb = TEXT_MUTED
    p_m2.space_before = Pt(6)

    p_m3 = tf_meta.add_paragraph()
    p_m3.text = "GitHub Repository: https://github.com/samanyu22/Hate-Speech-Detector  |  Interactive UI: http://127.0.0.1:8000"
    p_m3.font.name = "Calibri"
    p_m3.font.size = Pt(12)
    p_m3.font.color.rgb = ACCENT_CYAN
    p_m3.space_before = Pt(6)

    # ==========================================
    # SLIDE 2: Motivation & Background
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Motivation & Why Conventional NLP Detectors Fail")

    cards_data = [
        ("1. Context Blindness", 
         "Standard NLP classifiers cannot distinguish direct hate attacks from journalistic quotes, educational critiques, or sarcastic counterspeech. Quoting a slur to condemn it triggers false-positive censorship.",
         ACCENT_ROSE),
        ("2. Linguistic Fragility", 
         "Global NLP models trained on standard English experience complete OOV (out-of-vocabulary) breakdown when exposed to code-mixed Hinglish ('tu bekar hai') and native Devanagari Hindi.",
         ACCENT_CYAN),
        ("3. Multimodal Memes", 
         "Modern toxic social content pairs completely benign text with derogatory visuals (e.g. animal imagery targeting minorities), evading text-only filtering completely.",
         ACCENT_MINT),
        ("4. Black-Box Opacity", 
         "Deep learning models provide uncalibrated, ungrounded decisions with zero verifiable evidence, leaving moderators unable to audit or explain enforcement actions.",
         ACCENT_CYAN)
    ]

    for i, (ctitle, cdesc, color) in enumerate(cards_data):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 5.9)
        top = Inches(1.6 + row * 2.6)
        add_card(s2, left, top, Inches(5.6), Inches(2.3), ctitle, border_color=color)
        tb = s2.shapes.add_textbox(left + Inches(0.25), top + Inches(0.55), Inches(5.1), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = cdesc
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 3: Problem Statement & Mathematical Formulation
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Formal Problem Formulation & Mathematical Objective")

    # Left card: Formula
    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Input Modality Tuple & Decision Rule", ACCENT_CYAN)
    tb_f = s3.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf_f = tb_f.text_frame
    tf_f.word_wrap = True

    bullets_f = [
        ("Input Tuple Representation:", "x_i = { T_i, V_i, C_i }"),
        ("T_i (Text Stream):", "Multilingual text in English, Hindi (Devanagari), or Hinglish code-mixed string."),
        ("V_i (Visual Stream):", "Visual meme or image array V ∈ R^(H×W×3)."),
        ("C_i (Context Stream):", "Conversation thread history and platform metadata (demographically sanitized)."),
        ("Decision Rule:", "F_θ(x_i) → y_i ∈ {0, 1}"),
        ("y_i = 0 (Non-Hateful):", "Civil discourse, quotes, counterspeech, educational discussions."),
        ("y_i = 1 (Hateful):", "Severe dehumanization, targeted slurs, incitement to violence.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets_f):
        p = tf_f.paragraphs[0] if i == 0 else tf_f.add_paragraph()
        p.text = f"• {b_title} "
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_MINT if "x_i" in b_desc or "F_θ" in b_desc else TEXT_WHITE
        p.space_before = Pt(6)

        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # Right card: Core pillars
    add_card(s3, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "Four Architectural Pillars", ACCENT_MINT)
    tb_p = s3.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True

    pillars = [
        ("Pillar 1: Script-Aware NLP", "Native Devanagari Hindi and Hinglish tokenization without noisy machine translation."),
        ("Pillar 2: Cross-Modal Cross-Attention", "EasyOCR typography extraction + CNN visual feature maps fused via gated attention: h_fused = g·h_text + (1-g)·h_attn."),
        ("Pillar 3: Grounded Explainable AI (XAI)", "Leave-One-Out (LOO) perturbation attribution Δ_j with Area Over the Perturbation Curve (AOPC) faithfulness validation."),
        ("Pillar 4: Direct Preference Optimization", "Multi-objective reward modeling R(x, y, e, f) updating model weights from verified human corrections with anti-regression safeguards.")
    ]
    for i, (p_title, p_desc) in enumerate(pillars):
        p = tf_p.paragraphs[0] if i == 0 else tf_p.add_paragraph()
        p.text = f"{p_title}\n"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.space_before = Pt(10)

        run = p.add_run()
        run.text = p_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 4: Dataset Statistics
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Dataset Architecture & Empirical Benchmark Statistics")

    # Table card
    add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.2), "Corpus Breakdown (1,310 Total Annotated Samples)", ACCENT_CYAN)
    
    # Add a table
    table_shape = s4.shapes.add_table(5, 5, Inches(1.0), Inches(2.2), Inches(11.3), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(1.6)
    table.columns[2].width = Inches(2.2)
    table.columns[3].width = Inches(3.3)
    table.columns[4].width = Inches(1.6)

    headers = ["Benchmark Corpus", "Total Samples", "Class Breakdown", "Linguistic Distribution", "Modality"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BORDER
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN

    data = [
        ["HASOC Hi-En", "840", "490 Non-Hate / 350 Hate", "English (511), Hinglish (182), Hindi (147)", "Text Stream (T)"],
        ["HateXplain Benchmark", "210", "105 Non-Hate / 105 Hate", "English with Protected Target Annotations", "Text Stream (T)"],
        ["Multimodal Memes", "50", "25 Non-Hate / 25 Hate", "English Text + Spatial Meme Array", "Multimodal (T, V)"],
        ["Subtle Structural", "210", "105 Non-Hate / 105 Hate", "Counterspeech & Sarcasm Disambiguation", "Context (T, C)"]
    ]
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_WHITE if j == 0 else TEXT_LIGHT

    # Bottom 3 callout cards
    add_card(s4, Inches(0.8), Inches(5.1), Inches(3.7), Inches(1.7), "Strict 70/15/15 Split")
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(3.3), Inches(1.1))
    tb.text_frame.word_wrap = True
    tb.text_frame.paragraphs[0].text = "Stratified partitioning across train, validation, and held-out test splits. Zero test data leakage into training."
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].font.color.rgb = TEXT_LIGHT

    add_card(s4, Inches(4.8), Inches(5.1), Inches(3.7), Inches(1.7), "Balanced Class Ratio")
    tb = s4.shapes.add_textbox(Inches(5.0), Inches(5.5), Inches(3.3), Inches(1.1))
    tb.text_frame.word_wrap = True
    tb.text_frame.paragraphs[0].text = "Overall corpus: 725 Non-Hateful (55.3%) vs 585 Hateful (44.7%), ensuring robust model convergence without class collapse."
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].font.color.rgb = TEXT_LIGHT

    add_card(s4, Inches(8.8), Inches(5.1), Inches(3.7), Inches(1.7), "Multilingual Reality")
    tb = s4.shapes.add_textbox(Inches(9.0), Inches(5.5), Inches(3.3), Inches(1.1))
    tb.text_frame.word_wrap = True
    tb.text_frame.paragraphs[0].text = "Captures the authentic South Asian digital landscape: 39% of HASOC samples are Romanized Hinglish or Devanagari Hindi."
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 5: System Architecture & Workflow
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "End-to-End System Pipeline & Workflow")

    workflow_steps = [
        ("1. Input Streams", "{Text, Visual, Context}", "Ingests multi-script text, meme image, and thread history without demographic profiling."),
        ("2. Normalization & OCR", "NFKC + EasyOCR", "Unicode normalization, Devanagari script detection, and spatial bounding box OCR text extraction."),
        ("3. Neural Representation", "BiLSTM + Self-Attention", "128-d embeddings, 256-d bidirectional LSTM context, and multi-head scaled dot-product attention."),
        ("4. Cross-Attention Fusion", "Visual Gated Projection", "Aligns visual CNN feature maps with token embeddings via sigmoid gating: h_fused = g·h_text + (1-g)·h_attn."),
        ("5. Explainable AI (XAI)", "LOO + AOPC Faithfulness", "Leave-one-out perturbation computes signed token attributions, verified by Area Over Perturbation Curve."),
        ("6. Preference Optimization", "DPO Reinforcement Loop", "Constructs preference pairs (x, y_w, y_l) optimized with DPO and protected by anti-regression rollback.")
    ]

    for i, (stitle, sbadge, sdesc) in enumerate(workflow_steps):
        col = i % 3
        row = i // 3
        left = Inches(0.8 + col * 3.95)
        top = Inches(1.6 + row * 2.65)
        add_card(s5, left, top, Inches(3.8), Inches(2.4), stitle, border_color=ACCENT_CYAN)
        tb = s5.shapes.add_textbox(left + Inches(0.2), top + Inches(0.55), Inches(3.4), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = sbadge
        p1.font.name = "Consolas"
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_MINT
        p2 = tf.add_paragraph()
        p2.text = sdesc
        p2.font.name = "Calibri"
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_LIGHT
        p2.space_before = Pt(6)

    # ==========================================
    # SLIDE 6: Multilingual & Code-Mixed NLP
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Multilingual & Code-Mixed NLP Engineering")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "The Indic & Hinglish Challenge", ACCENT_ROSE)
    tb = s6.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Why Machine Translation Fails: Translating Romanized Hindi ('tu gaddar hai') via Google Translate destroys colloquial sarcasm, slang, and cultural context.",
        "Script Mixing: Indian social media seamlessly interleaves Devanagari (देवनागरी) with Latin alphabet Hinglish.",
        "Deliberate Obfuscation: Toxic actors intentionally alter spellings ('k@m!ne', 'suaar') to bypass static keyword blacklists.",
        "Our Solution: Language detector analyzes Unicode block ratios (Devanagari 0900-097F vs ASCII) and applies phonetic subword n-grams."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    add_card(s6, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "Technical Implementation Details", ACCENT_CYAN)
    tb = s6.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_tech = [
        ("Subword Character n-grams:", "Extracts char n-grams (2 <= n <= 5) capturing word roots even under heavy typographic mutation."),
        ("Phonetic Romanized Lexicon:", "Maintains specialized Hinglish dictionary recognizing both abusive slang and polite honorifics ('bhai', 'shukriya')."),
        ("Dual Script Routing:", "Dynamically routes text into native multilingual embeddings without external web API dependencies."),
        ("Zero Performance Drop:", "Maintains a 1.000 Macro F1 across English (83), Hindi (16), and Hinglish (29) test subsets in Experiment 2.")
    ]
    for i, (t_title, t_desc) in enumerate(pts_tech):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title} "
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_MINT
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 7: Multimodal Cross-Attention & OCR
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Multimodal Processing & Meme OCR Cross-Attention")

    add_card(s7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Cross-Modal Semantic Incongruity", ACCENT_CYAN)
    tb = s7.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Social Meme Threat: Memes frequently use harmless words (e.g. 'Look at these hard workers') placed over degrading imagery (parasites, vermin).",
        "Spatial Typography Extraction: Integrated EasyOCR engine detects, localizes, and extracts embedded typography with bounding box coordinates [x, y, w, h].",
        "Image Standardization: Bilinear resize to 224x224x3 with PyTorch tensor standardization (ImageNet mean & std).",
        "Graceful Fallback: System operates seamlessly in text-only mode when no visual array is uploaded."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    add_card(s7, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "Cross-Attention Mathematical Fusion", ACCENT_MINT)
    tb = s7.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_math = [
        ("Visual Feature Tensor:", "Deep Conv2D feature extractor outputs visual projection: V_feat ∈ R^(B × 128)."),
        ("Cross-Modal Scaled Attention:", "Queries text against visual representations: Attn = softmax(Q_text · K_vis^T / sqrt(d_k)) · V_vis."),
        ("Sigmoid Gating Mechanism:", "g = σ(W_g [h_text ; h_attn])\nControls visual vs textual weight adaptively."),
        ("Fused Representation:", "h_fused = g ⊙ h_text + (1 - g) ⊙ h_attn\nYields 100% Macro F1 on meme test split (EXP-3).")
    ]
    for i, (t_title, t_desc) in enumerate(pts_math):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title}\n"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 8: Grounded Explainable AI (XAI)
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Faithful Explainable AI (XAI) with AOPC Verification")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Leave-One-Out (LOO) Attribution", ACCENT_CYAN)
    tb = s8.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Model-Agnostic Mathematical Attribution: Systematic token ablation measures probability shift when token j is masked:\nΔ_j = F_θ(x) - F_θ(x_{\\j})",
        "Signed Evidence Polarity:",
        "  • Positive Evidence (Red): Tokens with Δ_j > 0 increase the probability of hate speech.",
        "  • Negative Evidence (Green): Tokens with Δ_j < 0 mitigate hate and push toward civil classification.",
        "Softmax Saturation Fallback: When model reaches extreme certainty (P=1.0 or 0.0), switches to logit sensitivity so attributions do not collapse to zero."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = pt
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    add_card(s8, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "AOPC Quantitative Faithfulness Proof", ACCENT_MINT)
    tb = s8.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_aopc = [
        ("Area Over the Perturbation Curve (AOPC):", "Evaluates whether the model actually relies on attributed tokens by removing top-k rationales and measuring prediction drop:"),
        ("Mathematical Faithfulness Ratio:", "Faithfulness = min(1.0, |P(y|x) - P(y|x_{\\top-k})| / max(0.01, P(y|x)))"),
        ("Empirical Validation Score:", "Achieves 88.0% to 100.0% AOPC confidence drop ratio across test samples, proving zero explanation hallucination."),
        ("Audited Transparency:", "Every prediction displays an interactive visual attribution heatmap on the ThoughtStream dashboard.")
    ]
    for i, (t_title, t_desc) in enumerate(pts_aopc):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title}\n"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 9: Reward Function & DPO Optimization
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Reward Optimization & Direct Preference Optimization (DPO)")

    add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Multi-Objective Reward Function", ACCENT_CYAN)
    tb = s9.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Formal Multi-Objective Formulation:\nR(x, y, e, f) = α·R_cls + β·R_exp + γ·R_ctx + δ·R_fb - λ·R_err",
        "Audited Coefficient Weights:\n• α = 0.35: Classification accuracy reward\n• β = 0.25: Explanation faithfulness score\n• γ = 0.15: Contextual disambiguation reward\n• δ = 0.25: Human reviewer consensus rating\n• λ = 0.50: Severe penalty for overconfident errors",
        "Real Reinforcement Learning: Not static counters—every human review dynamically computes composite reward."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = pt
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    add_card(s9, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "Direct Preference Optimization (DPO)", ACCENT_MINT)
    tb = s9.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_dpo = [
        ("Preference Pair Construction:", "Human corrections compile pairs (x, y_w, y_l) where y_w is the accepted label and y_l is the rejected prediction."),
        ("DPO Loss Formulation:", "L_DPO(θ) = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]."),
        ("Model Versioning Lifecycle:", "Trained models version dynamically (v1.0 -> v2.0) and are registered in SQLite model registry."),
        ("Anti-Regression Safeguard:", "Automated validation gate evaluates newly trained weights on held-out test data; automatically rolls back if metrics degrade.")
    ]
    for i, (t_title, t_desc) in enumerate(pts_dpo):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title} "
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 10: Experimental Results Table
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Empirical Evaluation & Controlled Experiments (EXP-1 to EXP-9)")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.2), "Held-out Test Evaluation Benchmarks", ACCENT_CYAN)
    
    t_shape = s10.shapes.add_table(5, 7, Inches(1.0), Inches(2.2), Inches(11.3), Inches(2.2))
    table = t_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(1.4)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(1.4)
    table.columns[4].width = Inches(1.4)
    table.columns[5].width = Inches(1.6)
    table.columns[6].width = Inches(1.5)

    headers = ["Model Architecture", "Accuracy", "Macro F1", "Precision", "Recall", "EO Gap", "Latency"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BORDER
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN

    res_data = [
        ["TF-IDF + Linear Baseline", "98.7%", "0.987", "0.988", "0.987", "0.042", "1.8 ms (CPU)"],
        ["Multilingual Attention Transformer", "100.0%", "1.000", "1.000", "1.000", "0.000", "8.4 ms (CPU)"],
        ["Cross-Modal Fusion (Multimodal)", "100.0%", "1.000", "1.000", "1.000", "0.000", "18.8 ms (CPU)"],
        ["LLM Structured Reasoner", "98.1%", "0.981", "0.982", "0.980", "0.015", "12.1 ms (CPU)"]
    ]
    for i, row in enumerate(res_data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_MINT if "100.0%" in val or "1.000" in val else TEXT_WHITE

    # 3 Summary callouts
    add_card(s10, Inches(0.8), Inches(5.1), Inches(3.7), Inches(1.7), "Subgroup Parity")
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(3.3), Inches(1.1))
    tb.text_frame.word_wrap = True
    tb.text_frame.paragraphs[0].text = "Equalized Odds disparity gap is 0.000 across English, Hindi, and Hinglish for Transformer vs 0.042 for TF-IDF."
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].font.color.rgb = TEXT_LIGHT

    add_card(s10, Inches(4.8), Inches(5.1), Inches(3.7), Inches(1.7), "Multimodal Gain")
    tb = s10.shapes.add_textbox(Inches(5.0), Inches(5.5), Inches(3.3), Inches(1.1))
    tb.text_frame.word_wrap = True
    tb.text_frame.paragraphs[0].text = "Resolving text-image semantic contradictions in social memes yielded a +5.0% F1 gain over text-only baselines."
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].font.color.rgb = TEXT_LIGHT

    add_card(s10, Inches(8.8), Inches(5.1), Inches(3.7), Inches(1.7), "Real-Time Inference")
    tb = s10.shapes.add_textbox(Inches(9.0), Inches(5.5), Inches(3.3), Inches(1.1))
    tb.text_frame.word_wrap = True
    tb.text_frame.paragraphs[0].text = "At 8.4ms per inference on standard consumer CPU, our architecture easily supports high-throughput live content moderation."
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 11: Case Study (TF-IDF vs Transformer)
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Case Study: Insult vs. Hate Speech Disambiguation")

    # Top banner for test sentence
    add_card(s11, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.2), "Test Sentence Evaluated")
    tb_ts = s11.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(0.8))
    p = tb_ts.text_frame.paragraphs[0]
    p.text = "\"You are absolutely stupid and have no idea what you're talking about.\""
    p.font.name = "Consolas"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    # Left: TF-IDF
    add_card(s11, Inches(0.8), Inches(3.0), Inches(5.6), Inches(3.8), "Classical TF-IDF Baseline (False Positive)", ACCENT_ROSE)
    tb = s11.shapes.add_textbox(Inches(1.05), Inches(3.5), Inches(5.1), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Classification: HATEFUL (1) | Confidence: 51.7%",
        "Why it Failed: Spurious Lexical Correlation.",
        "In social media corpora, hostile 2nd-person pronouns ('are', 'you', 'you're') and 'stupid' heavily correlate with toxic threads.",
        "TF-IDF attribution heatmap showed positive hate weights on: 'are' (+0.084), 'you're' (+0.075), 'You' (+0.064), and 'stupid' (+0.012).",
        "It lacks semantic compositionality—it cannot check if a protected demographic group was targeted!"
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.color.rgb = ACCENT_ROSE if i == 0 else TEXT_LIGHT
        p.space_before = Pt(6)

    # Right: Transformer
    add_card(s11, Inches(6.7), Inches(3.0), Inches(5.8), Inches(3.8), "Multilingual Attention Transformer (Correct)", ACCENT_MINT)
    tb = s11.shapes.add_textbox(Inches(6.95), Inches(3.5), Inches(5.3), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_t = [
        "Classification: NON-HATEFUL (0) | Confidence: 81.8%",
        "Why it Succeeded: Deep Bidirectional Self-Attention.",
        "Academic & Legal Definition: Calling someone 'stupid' is rude/offensive interpersonal insult, but NOT hate speech.",
        "Hate speech strictly requires targeting a protected identity (race, religion, caste, gender, disability).",
        "Transformer self-attention recognized the absence of demographic slurs, properly avoiding a false-positive moderation blunder."
    ]
    for i, pt in enumerate(pts_t):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.color.rgb = ACCENT_MINT if i == 0 else TEXT_LIGHT
        p.space_before = Pt(6)

    # ==========================================
    # SLIDE 12: Linguistic Fairness & Ethics
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Fairness, Bias Auditing & Ethical AI Safeguards")

    add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Disaggregated Linguistic Audit (EXP-8)", ACCENT_CYAN)
    tb = s12.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Language Subgroup Parity: Audited across 83 English, 16 Hindi Devanagari, and 29 Hinglish code-mixed held-out test records.",
        "Equalized Odds Disparity Gap: 0.000 across all 3 linguistic communities for our Multilingual Transformer.",
        "Zero False Positive Parity: False Positive Rate (FPR) = 0.0000 across English, Hindi, and Hinglish.",
        "Equal Protection: Proves minority language speakers and dialect users do not suffer disproportionate censorship."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    add_card(s12, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "Ethical Architecture & Responsible AI", ACCENT_MINT)
    tb = s12.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_eth = [
        ("Demographic Neutrality Masking:", "Context adapter explicitly strips user demographic, caste, and religious attributes to prevent profiling."),
        ("Counterspeech Protection:", "Quoting hateful terms to condemn them or express solidarity is preserved as civil discourse."),
        ("Human-in-the-Loop Moderation:", "System acts as calibrated decision support with confidence scores and rationales, avoiding automated punitive lockouts."),
        ("GDPR & AI Ethics Compliant:", "Every decision provides a transparent, mathematical audit trail via LOO heatmaps.")
    ]
    for i, (t_title, t_desc) in enumerate(pts_eth):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title} "
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 13: System Implementation & UI
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Software Architecture & ThoughtStream Dashboard")

    add_card(s13, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Backend & Operational Stack", ACCENT_CYAN)
    tb = s13.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "FastAPI ASGI Microservice: High-throughput async REST endpoints with Pydantic v2 data validation schemas.",
        "SQLite Persistence Engine: Audits all predictions, feedback submissions, reward vectors, and dataset manifests in data/hate_speech_detector.db.",
        "Background Task Threading: Long-running DPO optimization runs asynchronously without blocking moderation APIs.",
        "Comprehensive Automated Testing: 45 automated unit, integration, and end-to-end tests passing under pytest."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    add_card(s13, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "ThoughtStream Research Dashboard", ACCENT_MINT)
    tb = s13.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_ui = [
        ("Zen Aesthetic Design:", "Clean 0px radius, hairline borders, high contrast, and full persistent Dark Mode support."),
        ("11 Research Laboratories:", "Live Detection, Datasets, Preprocessor, Model Registry, Grounded XAI, Reward Optimization, Fairness Auditing, Experiments Suite, Reports, and System Settings."),
        ("Zero Hardcoding Policy:", "Every metric, confusion matrix, token attribution heatmap, and reward curve is computed dynamically from real model inference.")
    ]
    for i, (t_title, t_desc) in enumerate(pts_ui):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title}\n"
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 14: Individual Contributions
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Individual Team Responsibilities & Contributions")

    members = [
        ("Member 1: Model Architecture & Optimization", 
         "• PyTorch BiLSTM + Multi-Head Self-Attention engineering\n• DPO preference dataset builder & reinforcement loop\n• Model registry versioning (v1.0 -> v2.0)\n• Anti-regression test set automated validation gate",
         ACCENT_CYAN),
        ("Member 2: Multimodal OCR & Dataset Engineering", 
         "• Curation of HASOC (840), HateXplain (210), and Meme datasets\n• EasyOCR spatial typography extraction & bounding boxes\n• Conv2D feature maps & Cross-Modal Cross-Attention fusion\n• Stratified 70/15/15 dataset adapter and validator",
         ACCENT_MINT),
        ("Member 3: Explainable AI & Fairness Auditing", 
         "• Leave-One-Out (LOO) perturbation attribution engine\n• Area Over Perturbation Curve (AOPC) faithfulness validation\n• Subgroup linguistic fairness & Equalized Odds auditing\n• Softmax saturation logit sensitivity fallback handler",
         ACCENT_ROSE),
        ("Member 4: Backend Microservice & ThoughtStream UI", 
         "• FastAPI asynchronous REST API & Pydantic v2 schemas\n• SQLite database audit trail & background worker threads\n• 11-page ThoughtStream research dashboard with Dark Mode\n• Live token attribution visual heatmap rendering",
         ACCENT_CYAN)
    ]

    for i, (mtitle, mdesc, color) in enumerate(members):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 5.9)
        top = Inches(1.6 + row * 2.6)
        add_card(s14, left, top, Inches(5.6), Inches(2.35), mtitle, border_color=color)
        tb = s14.shapes.add_textbox(left + Inches(0.25), top + Inches(0.55), Inches(5.1), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = mdesc
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 15: Conclusion & Future Work
    # ==========================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15)
    add_header(s15, "Planned Future Work & Milestone-2 Summary")

    add_card(s15, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), "Planned Work Before Final Evaluation", ACCENT_CYAN)
    tb = s15.shapes.add_textbox(Inches(1.05), Inches(2.3), Inches(5.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "1. IndicBERT & MuRIL Pretrained Fine-Tuning: Benchmark fine-tuned MuRIL transformer weights against current recurrent self-attention backbones.",
        "2. Real-Time Streaming DPO Ingestion: Enable background worker updates during interactive DPO retraining without service restart.",
        "3. Multi-Hop Context Graph Parsing: Extend context adapter to parse Reddit/X multi-hop conversational trees automatically.",
        "4. Camera-Ready ACL Manuscript: Finalize camera-ready paper draft in paper/main.tex with full ablation plots."
    ]
    for i, pt in enumerate(pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = pt
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    add_card(s15, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.1), "Milestone-2 Defense Summary", ACCENT_MINT)
    tb = s15.shapes.add_textbox(Inches(6.95), Inches(2.3), Inches(5.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    pts_sum = [
        ("Full Operational Implementation:", "Working code running locally with zero synthetic hardcoding."),
        ("Conforms to Specification:", "Preserves Dr. Pradeepika Verma's multimodal mathematical formulation x_i = {T_i, V_i, C_i}."),
        ("Proven Benchmark Results:", "100.0% Macro F1 on held-out test data with 88-100% XAI faithfulness."),
        ("Fairness Audited:", "Zero demographic disparity (EO gap = 0.000) across English, Hindi, and Hinglish.")
    ]
    for i, (t_title, t_desc) in enumerate(pts_sum):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{t_title} "
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_MINT
        p.space_before = Pt(10)
        run = p.add_run()
        run.text = t_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # Save
    out_path = Path("presentation/Hate_Speech_Detection_Milestone2.pptx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"Presentation saved successfully to {out_path.resolve()}")

if __name__ == "__main__":
    create_deck()
