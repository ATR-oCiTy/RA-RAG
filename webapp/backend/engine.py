"""
Interactive session engine backing the webapp. Reuses the paper's actual
building blocks (src/reliability_utils.py) rather than main.py's batch
dataset pipeline, since sources here are given their context directly (no
FAISS/BEIR/contriever retrieval step). But it follows the same *structure*
main.py does, not a shortcut of it:

  1. Estimation phase: calibration rounds accumulate, reliability is
     estimated with iterative_weighted_majority_voting exactly as
     multi_source_weight_est does.
  2. Lock: weights are frozen, exactly as main.py freezes
     estimated_weight_lst before calling reliablity_aware_inference.
  3. Inference phase: each query only consults the top-kappa most reliable
     (frozen-weight) sources (kappa-RRSS), their answers are semantically
     clustered with the paper's own EntailmentDeberta before voting, and
     weighted_majority_voting uses the frozen weights — never re-estimated
     against the query being answered.

Also replicated: AlignScore-based grounding filter during estimation only
(matching multi_source_weight_est's convert_to_valid_answer — the paper never
applies it at inference time either). Each source's calibration-round answer
is converted to a declarative claim (QAConvertor, the repo's BART model) and
scored against that source's own context (AlignScore-base). Below the
threshold, the answer is replaced with "i don't know" before it ever reaches
the reliability estimator — so an answer with no supporting context (e.g. a
scripted "mimic" answer with nothing backing it) will typically get filtered
out, exactly as it would if a real source asserted something ungrounded.
"""
import os
import sys
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from google import genai

from src.utils import gemini_generate_content, normalize_answer
from src.prompts import wrap_prompt
from src.reliability_utils import (
    weighted_majority_voting,
    iterative_weighted_majority_voting,
    get_semantic_ids,
    unify_answer,
    EntailmentDeberta,
    QAConvertor,
    DEVICE,
)
from Alignscore.alignscore import AlignScore

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ALIGN_SCORE_THRESHOLD = float(os.environ.get("ALIGN_SCORE_THRESHOLD", "0.5"))
ALIGN_SCORE_CKPT = os.path.join(REPO_ROOT, "Alignscore", "ckpt_dir", "AlignScore-base.ckpt")

_client = None
_entailment_model = None
_qa_convertor = None
_align_scorer = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ["GEMINI_API_KEY"]
        _client = genai.Client(api_key=api_key)
    return _client


def get_entailment_model():
    # Lazy: ~1.5GB DeBERTa-v2-xlarge-mnli download + load, only paid for once
    # semantic clustering is actually needed (first inference round).
    global _entailment_model
    if _entailment_model is None:
        _entailment_model = EntailmentDeberta()
    return _entailment_model


def get_qa_convertor():
    # Lazy singleton — AlignScoreFiltering.apply() in the original repo
    # instantiates a fresh QAConvertor() (a BART model) on every call, which
    # would reload a full seq2seq model from disk on every single round.
    # Reusing one instance is the only change from upstream's own filtering
    # logic; the declarative-conversion + scoring math itself is unmodified.
    global _qa_convertor
    if _qa_convertor is None:
        _qa_convertor = QAConvertor()
    return _qa_convertor


def get_align_scorer():
    global _align_scorer
    if _align_scorer is None:
        if not os.path.exists(ALIGN_SCORE_CKPT):
            raise RuntimeError(
                f"AlignScore checkpoint not found at {ALIGN_SCORE_CKPT}. "
                "Download it with: curl -L -o Alignscore/ckpt_dir/AlignScore-base.ckpt "
                "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt"
            )
        _align_scorer = AlignScore(
            model="roberta-base",
            batch_size=1,
            device=DEVICE,
            ckpt_path=ALIGN_SCORE_CKPT,
            evaluation_mode="nli_sp",
            verbose=False,
        )
    return _align_scorer


def source_answer(query: str, context: str) -> str:
    prompt = wrap_prompt(query, context, prompt_type="rag")
    text = gemini_generate_content(
        get_client(), GEMINI_MODEL, prompt, max_new_tokens=30, temperature=0.0
    )
    return normalize_answer(text)


def filter_grounded(query: str, answer: str, context: str) -> tuple[str, bool]:
    """Returns (possibly-replaced answer, was_filtered). Mirrors
    AlignScoreFiltering.apply for a single (question, answer, context)."""
    if answer == "i don't know":
        return answer, False
    declarative = get_qa_convertor().apply([query], [answer])[0]
    score = get_align_scorer().score(contexts=[context], claims=[declarative])[0]
    if score > ALIGN_SCORE_THRESHOLD:
        return answer, False
    return "i don't know", True


class Session:
    def __init__(self, source_names, top_k_src=None):
        if len(source_names) < 2:
            raise ValueError("need at least 2 sources")
        self.source_names = source_names
        self.top_k_src = min(top_k_src or len(source_names), len(source_names))

        self.calibration_rounds = []  # [{query, answers: {name: str}, manual: {name: bool}}]
        self.locked = False
        self.frozen_weights = None  # {name: float}, set once locked

        self.inference_rounds = []  # [{query, chosen_sources, answers, unified_answers, consensus, manual}]

    # ---------------- phase 1: estimation ----------------

    def add_calibration_round(self, query: str, contexts: dict, manual_answers: dict):
        if self.locked:
            raise ValueError("weights are already locked; start a new session to re-calibrate")

        answers, manual_flags = self._collect_answers(self.source_names, query, contexts, manual_answers)

        filtered_flags = {}
        for name in self.source_names:
            answers[name], filtered_flags[name] = filter_grounded(query, answers[name], contexts.get(name, ""))

        self.calibration_rounds.append({
            "query": query, "answers": answers, "manual": manual_flags, "filtered": filtered_flags,
        })
        return self.state()

    def _live_weight_preview(self):
        """Reliability estimate if locked right now — for display only, not frozen."""
        outputs = [[r["answers"][n] for n in self.source_names] for r in self.calibration_rounds]
        if not outputs:
            return {n: 1.0 for n in self.source_names}
        _, reliabilities = iterative_weighted_majority_voting(
            outputs, len(self.source_names), max_iter=100, tol=1e-6
        )
        return dict(zip(self.source_names, reliabilities.tolist()))

    def lock(self):
        if self.locked:
            raise ValueError("already locked")
        if not self.calibration_rounds:
            raise ValueError("need at least 1 calibration round before locking weights")
        self.frozen_weights = self._live_weight_preview()
        self.locked = True
        return self.state()

    # ---------------- phase 2: inference (kappa-RRSS) ----------------

    def add_inference_round(self, query: str, contexts: dict, manual_answers: dict):
        if not self.locked:
            raise ValueError("lock calibration weights before running inference rounds")

        ranked = sorted(self.source_names, key=lambda n: self.frozen_weights[n], reverse=True)
        chosen = ranked[: self.top_k_src]

        answers, manual_flags = self._collect_answers(chosen, query, contexts, manual_answers)
        answer_list = [answers[n] for n in chosen]

        cluster_ids = get_semantic_ids(answer_list, get_entailment_model(), question=query)
        unified_list = unify_answer([answer_list], [cluster_ids])[0]
        weights_for_chosen = [self.frozen_weights[n] for n in chosen]
        consensus = weighted_majority_voting([unified_list], [weights_for_chosen])[0]

        unified_by_name = dict(zip(chosen, unified_list))

        self.inference_rounds.append({
            "query": query,
            "chosen_sources": chosen,
            "answers": answers,
            "unified_answers": unified_by_name,
            "manual": manual_flags,
            "consensus": consensus,
        })
        return self.state()

    # ---------------- shared ----------------

    def _collect_answers(self, names, query, contexts, manual_answers):
        answers, manual_flags = {}, {}
        for name in names:
            if name in manual_answers and manual_answers[name] is not None:
                answers[name] = normalize_answer(manual_answers[name])
                manual_flags[name] = True
            else:
                answers[name] = source_answer(query, contexts.get(name, ""))
                manual_flags[name] = False
        return answers, manual_flags

    def state(self):
        return {
            "source_names": self.source_names,
            "top_k_src": self.top_k_src,
            "locked": self.locked,
            "weights": self.frozen_weights if self.locked else self._live_weight_preview(),
            "calibration_rounds": self.calibration_rounds,
            "inference_rounds": self.inference_rounds,
        }


SESSIONS: dict[str, Session] = {}


def create_session(source_names, top_k_src=None):
    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = Session(source_names, top_k_src=top_k_src)
    return session_id


def get_session(session_id) -> Session:
    if session_id not in SESSIONS:
        raise KeyError(session_id)
    return SESSIONS[session_id]
