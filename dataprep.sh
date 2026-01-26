#!/bin/bash
# Data Preparation Commands for 14 Whisper Ablation Experiments
# CommonVoice Swahili (sw) and Kinyarwanda (rw)

# Base paths
CV_BASE="/ocean/projects/cis250145p/shared/datasets/CommonVoice/cv-corpus-22.0-2025-06-20"
OUTPUT_BASE="corpora/cv_ablations"

# ============================================================================
# SWAHILI (sw) DATA PREPARATIONS
# ============================================================================

# 1. sw ASR only (transcribe sw -> sw)
echo "Preparing: sw ASR only"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_asr \
    --source_lang sw \
    --tasks transcribe

# 2. sw -> en AST only (translate sw -> en)
echo "Preparing: sw -> en AST only"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_to_en \
    --source_lang sw \
    --tasks translate \
    --target_langs en

# 3. sw -> fr AST only (translate sw -> fr)
echo "Preparing: sw -> fr AST only"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_to_fr \
    --source_lang sw \
    --tasks translate \
    --target_langs fr

# 4. sw -> en,fr AST two-way (translate sw -> en AND sw -> fr)
echo "Preparing: sw -> en,fr AST two-way"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_to_en_fr \
    --source_lang sw \
    --tasks translate \
    --target_langs en,fr

# 5. sw ASR + AST one-way (transcribe + translate sw -> en)
echo "Preparing: sw ASR + AST one-way (sw, en)"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_asr_ast_en \
    --source_lang sw \
    --tasks transcribe,translate \
    --target_langs en

# 6. sw ASR + AST one-way (transcribe + translate sw -> fr) [Exp 14]
echo "Preparing: sw ASR + AST one-way (sw, fr)"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_asr_ast_fr \
    --source_lang sw \
    --tasks transcribe,translate \
    --target_langs fr

# 7. sw ASR + AST two-way (transcribe + translate sw -> en,fr)
echo "Preparing: sw ASR + AST two-way (sw, en, fr)"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/sw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/sw_asr_ast_en_fr \
    --source_lang sw \
    --tasks transcribe,translate \
    --target_langs en,fr

# ============================================================================
# KINYARWANDA (rw) DATA PREPARATIONS
# NOTE: Whisper doesn't have 'rw' language token, using 'sw' (Swahili) as proxy
# ============================================================================

# 8. rw ASR only (transcribe rw -> rw)
echo "Preparing: rw ASR only"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_asr \
    --source_lang sw \
    --tasks transcribe

# 9. rw -> en AST only (translate rw -> en)
echo "Preparing: rw -> en AST only"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_to_en \
    --source_lang sw \
    --tasks translate \
    --target_langs en

# 10. rw -> fr AST only (translate rw -> fr)
echo "Preparing: rw -> fr AST only"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_to_fr \
    --source_lang sw \
    --tasks translate \
    --target_langs fr

# 11. rw -> en,fr AST two-way (translate rw -> en AND rw -> fr)
echo "Preparing: rw -> en,fr AST two-way"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_to_en_fr \
    --source_lang sw \
    --tasks translate \
    --target_langs en,fr

# 12. rw ASR + AST one-way to en (transcribe + translate rw -> en)
echo "Preparing: rw ASR + AST one-way (rw, en)"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_asr_ast_en \
    --source_lang sw \
    --tasks transcribe,translate \
    --target_langs en

# 13. rw ASR + AST one-way to fr (transcribe + translate rw -> fr)
echo "Preparing: rw ASR + AST one-way (rw, fr)"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_asr_ast_fr \
    --source_lang sw \
    --tasks transcribe,translate \
    --target_langs fr

# 14. rw ASR + AST two-way (transcribe + translate rw -> en,fr)
echo "Preparing: rw ASR + AST two-way (rw, en, fr)"
python iwslt/utils/dataprep_commonvoice.py \
    --base_path ${CV_BASE}/rw \
    --audio_dir clips_16kHz \
    --output_dir ${OUTPUT_BASE}/rw_asr_ast_en_fr \
    --source_lang sw \
    --tasks transcribe,translate \
    --target_langs en,fr

# ============================================================================
# TOKEN LENGTH FILTERING (remove sequences > 428 tokens)
# ============================================================================

echo "Filtering long sequences..."

# Swahili corpora
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_asr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_to_en
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_to_fr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_to_en_fr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_asr_ast_en
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_asr_ast_fr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/sw_asr_ast_en_fr

# Kinyarwanda corpora
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_asr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_to_en
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_to_fr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_to_en_fr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_asr_ast_en
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_asr_ast_fr
python3 iwslt/utils/token_lengths.py --corpus_dir ${OUTPUT_BASE}/rw_asr_ast_en_fr

echo "Data preparation complete!"