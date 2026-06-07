"""Communication Engine — AWS Bedrock Claude + intent + response generation."""

import time
import json
from dataclasses import dataclass, field

import boto3

from config import NUM_CANDIDATES, INTENT_CLASSES
from eeg_pipeline import BrainState
from neuro_rl import StyleVector
from style_extractor import StyleProfile, format_examples_for_prompt

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"


@dataclass
class ConversationTurn:
    speaker: str
    text: str
    timestamp: float = 0.0


@dataclass
class ResponseCandidate:
    text: str
    intent: str
    index: int


class IntentClassifier:
    def classify(self, brain: BrainState) -> tuple[str, float]:
        eng = brain.engagement
        val = brain.valence
        focus = brain.focus
        cog = brain.cognitive_load
        relax = brain.relaxation
        error = brain.error_response

        if error > 0.5:
            return "correct", min(1.0, error)
        if eng > 0.65 and val > 0.6:
            return "agree", min(0.9, eng)
        if eng > 0.65 and val < 0.4:
            return "disagree", min(0.9, eng)
        if cog > 0.6 and focus > 0.5:
            return "elaborate", min(0.85, cog)
        if eng > 0.5 and cog > 0.4:
            return "question", 0.6
        if relax > 0.6 and val > 0.6:
            return "express_emotion", 0.7
        if eng < 0.3:
            return "acknowledge", max(0.4, 1 - eng)
        return "acknowledge", 0.5


class CommunicationEngine:
    def __init__(self, style_profile=None, few_shot_examples=None):
        self._bedrock = None
        self._intent_classifier = IntentClassifier()
        self._conversation: list[ConversationTurn] = []
        self._max_context = 20
        self._style_profile = style_profile
        self._few_shot_text = ""
        if few_shot_examples:
            self._few_shot_text = format_examples_for_prompt(few_shot_examples)

    def add_incoming_message(self, speaker: str, text: str):
        self._conversation.append(ConversationTurn(
            speaker=speaker, text=text, timestamp=time.time(),
        ))
        if len(self._conversation) > self._max_context:
            self._conversation = self._conversation[-self._max_context:]

    def classify_intent(self, brain_state: BrainState) -> tuple[str, float]:
        return self._intent_classifier.classify(brain_state)

    def generate_responses(self, brain_state: BrainState, style: StyleVector,
                          n_candidates=NUM_CANDIDATES) -> list[ResponseCandidate]:
        intent, confidence = self.classify_intent(brain_state)
        system_prompt = self._build_system_prompt(style)
        user_prompt = self._build_user_prompt(intent, confidence)

        try:
            if self._bedrock is None:
                self._bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
            response = self._bedrock.converse(
                modelId=BEDROCK_MODEL,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"maxTokens": 400, "temperature": 0.8},
            )
            raw_text = response["output"]["message"]["content"][0]["text"]
            candidates = self._parse_candidates(raw_text, intent)
            if candidates:
                return candidates
        except Exception as e:
            print(f"[Comm] Bedrock error: {e}", flush=True)

        return [ResponseCandidate(text=self._fallback_response(intent), intent=intent, index=0)]

    def _build_system_prompt(self, style: StyleVector):
        style_inst = style.to_prompt_instructions()

        return (
            "You are speaking AS someone in a real-time conversation. "
            "You ARE this person — generate responses they would actually say. "
            "You will see the conversation so far and must respond naturally to what was JUST said.\n\n"
            "CRITICAL RULES:\n"
            "- Respond to the LAST thing said to you. Your response must make sense as a reply.\n"
            "- Sound human. Use contractions, casual language, real speech patterns.\n"
            "- Generate EXACTLY 4 options, numbered 1-4, each on its own line.\n"
            "- Option 1 should be the BEST, most natural response. Options 2-4 are alternatives.\n"
            "- Keep each response 1-2 sentences max.\n"
            "- DO NOT be generic. Respond specifically to what was said.\n\n"
            f"Style: {style_inst}\n"
        )

    def _build_user_prompt(self, intent: str, confidence: float):
        context_lines = []
        for turn in self._conversation[-8:]:
            prefix = "Me:" if turn.speaker == "me" else f"{turn.speaker}:"
            context_lines.append(f"{prefix} {turn.text}")

        context_str = "\n".join(context_lines) if context_lines else "(no conversation yet)"

        last_msg = ""
        for turn in reversed(self._conversation):
            if turn.speaker != "me":
                last_msg = turn.text
                break

        intent_hints = {
            "agree": "I want to agree/affirm.",
            "disagree": "I want to push back or disagree.",
            "elaborate": "I want to explain my thinking in more detail.",
            "acknowledge": "I want to briefly acknowledge what they said.",
            "question": "I want to ask a follow-up question.",
            "express_emotion": "I want to share how I feel about this.",
            "correct": "I want to correct or clarify something.",
        }
        hint = intent_hints.get(intent, "I want to respond naturally.")

        return (
            f"Conversation:\n{context_str}\n\n"
            f"They just said: \"{last_msg}\"\n"
            f"My intent: {hint}\n\n"
            f"Generate 4 response options (numbered 1-4):"
        )

    def _parse_candidates(self, text, intent):
        candidates = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix in ("1.", "2.", "3.", "4.", "1)", "2)", "3)", "4)"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            line = line.strip('"').strip("'").strip()
            if line and len(line) > 2:
                candidates.append(ResponseCandidate(
                    text=line, intent=intent, index=len(candidates),
                ))
        return candidates[:NUM_CANDIDATES]

    def _fallback_response(self, intent):
        fallbacks = {
            "agree": "Yeah, for sure.",
            "disagree": "Hmm, I don't think so.",
            "elaborate": "So basically what I mean is...",
            "acknowledge": "Got it.",
            "question": "Wait, what do you mean by that?",
            "express_emotion": "Honestly that means a lot.",
            "correct": "No wait, that's not quite right.",
        }
        return fallbacks.get(intent, "Yeah, I hear you.")

    def record_selected_response(self, candidate: ResponseCandidate):
        self._conversation.append(ConversationTurn(
            speaker="me", text=candidate.text, timestamp=time.time(),
        ))

    def get_conversation_context(self):
        return [{"speaker": t.speaker, "text": t.text} for t in self._conversation[-10:]]
