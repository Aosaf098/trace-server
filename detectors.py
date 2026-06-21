"""
Signal detectors — the same patterns the console uses, used here server-side.
Each detector returns the message indices it fired on, so the evidence and
line-highlighting the analyst sees is computed in the backend.
"""
import re

SIGNALS = ["off_platform", "secrecy", "age_probe", "flattery", "gift"]

PATTERNS = {
    "off_platform": re.compile(
        r"\b(snapchat|snap|whatsapp|telegram|kik|insta|instagram dm|"
        r"move (this )?to|dm me|text me|my number|add me on)\b", re.I),
    "secrecy": re.compile(
        r"\b(don'?t tell|between us|our (space|secret|friendship)|keep it between|"
        r"wouldn'?t (get|understand)|delete (these|the|our) messages|"
        r"don'?t mention|just between)\b", re.I),
    "age_probe": re.compile(
        r"\b(how old (are|r) (you|u)|what grade|are you \d{1,2})\b", re.I),
    "flattery": re.compile(
        r"\b(mature for|more mature|you get me|only one (who|that)|"
        r"talk to you about anything|so talented for your age|tell me anything)\b", re.I),
    "gift": re.compile(
        r"\b(gift card|robux|v-?bucks|send you money|buy you|paypal|"
        r"cashapp|free skin|i can send you)\b", re.I),
}

MINOR = re.compile(
    r"\b(i'?m 1[0-7]\b|im 1[0-7]\b|in (6th|7th|8th|9th|10th) grade|turning 1[0-7])\b", re.I)
