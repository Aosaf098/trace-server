"""
Synthetic conversation generator (realistically hard).

Deliberately introduces ambiguity so the model faces genuine decisions, not a
trivially separable task:
  - BORDERLINE-SAFE cases: innocent chats that resemble a risk signal (a coach
    asking to keep sessions private, a relative's warm compliment, a friend's
    off-platform invite) -> these create honest false-positive pressure.
  - SUBTLE grooming: a single weak, indirect signal -> honest false negatives.
  - Overlapping phrasing between classes -> no clean gap to exploit.

Grooming cases depict TACTICS only, never explicit content.
"""
import json
import random

random.seed(7)

MINORS = ["im 13", "im 14", "i'm 12", "in 7th grade", "im 15 lol", "turning 14 soon"]
SOURCE_LABELS = ["Platform filter", "Tipline report", "User report",
                 "Moderator escalation", "Automated scan"]
PLATFORMS = ["Discord", "Roblox", "Snapchat", "Instagram",
             "In-game lobby", "Fandom forum", "Trading server"]

TACTICS = {
    "age_probe": {
        "detect": ["how old are you?", "how old r u", "what grade are you in?"],
        "evade":  ["so what's your age range?", "you still in school?", "you seem young, what year were you born?"],
    },
    "flattery": {
        "detect": ["you're so mature for your age", "honestly you get me like no one else",
                   "you're so talented for your age", "you can tell me anything"],
        "evade":  ["you're not like other people your age", "i feel like we really click",
                   "you're special, you know that?"],
    },
    "secrecy": {
        "detect": ["let's keep this between us", "don't tell anyone we talk",
                   "delete these messages after", "people wouldn't understand our friendship"],
        "evade":  ["maybe don't bring this up to your folks", "this can be our thing",
                   "best if no one else knows"],
    },
    "off_platform": {
        "detect": ["what's your snapchat?", "add me on telegram", "dm me on insta",
                   "text me, here's my number", "let's move this to whatsapp"],
        "evade":  ["is there somewhere quieter we can chat?", "this app is annoying, you got anything else?",
                   "we should talk somewhere more private", "you got discord or smth", "hmu somewhere else"],
    },
    "gift": {
        "detect": ["i can send you a gift card", "i'll buy you some robux",
                   "i can send you money for it", "want some free v-bucks?"],
        "evade":  ["i could help you out with that, no strings", "i can sort you out if you want",
                   "let me treat you to something", "i could hook you up no biggie"],
    },
}

BENIGN_FILLER = ["gg that was close", "lol nice", "yeah for real", "haha ok", "that's wild",
                 "anyway how's it going", "did you see the update", "same honestly",
                 "what are you up to", "nice one", "ok cool"]

SAFE_TEMPLATES = [
    ("Game help channel", [("u_a", "anyone got tips for the water temple boss"),
        ("u_b", "use ice arrows on the eyes then hit the core"), ("u_a", "omg ty stuck for an hour"), ("u_b", "np gl")]),
    ("Study group chat", [("u_a", "did anyone finish q7 on the bio worksheet"),
        ("u_b", "it's mitochondria, same diagram from class"), ("u_a", "ohh thanks you saved me")]),
    ("Friends DM", [("u_a", "you watching the new episode tonight"),
        ("u_b", "yeah after dinner no spoilers"), ("u_a", "lol ok i'll wait for you")]),
    ("Trading channel", [("u_a", "trading a blue dragon anyone interested"),
        ("u_b", "i'll give you two epics"), ("u_a", "deal sending now"), ("u_b", "ty pleasure")]),
    ("Class group chat", [("u_a", "who's coming to the birthday thing saturday"),
        ("u_b", "me what time"), ("u_a", "2pm at the park bring snacks")]),
    ("Art commissions DM", [("u_a", "do you take commissions"),
        ("u_b", "yep i can send a rate sheet"), ("u_a", "perfect email it over")]),
]

# Innocent conversations that RESEMBLE a risk signal — the hard negatives.
BORDERLINE_SAFE = [
    ("Sports club chat", [("u_coach", "great work today, you're improving fast"),
        ("u_player", "thanks coach!"),
        ("u_coach", "let's keep our extra sessions between us so the others don't feel left out"),
        ("u_player", "sounds good")]),
    ("Event signup DM", [("u_org", "hey can you fill in the sign-up form"),
        ("u_member", "sure"), ("u_org", "how old are you? the form needs your age"), ("u_member", "17")]),
    ("Family group chat", [("u_aunt", "you're so mature for your age, your mum must be proud"),
        ("u_teen", "haha thanks auntie")]),
    ("Friends DM", [("u_a", "this group chat is chaos lol"),
        ("u_b", "fr add me on discord instead"), ("u_a", "done")]),
    ("Study help DM", [("u_a", "thanks for the notes, you saved me"),
        ("u_b", "no worries"), ("u_a", "i'll buy you a coffee to say thanks")]),
    ("Mentor DM", [("u_mentor", "you can talk to me about anything on the project, ok?"),
        ("u_student", "appreciate it, thanks")]),
]

BENIGN_TRIGGERS = ["gg add me on discord for next time", "what grade are you in for the school league",
                   "i can send you the notes", "buy you a coffee sometime as thanks"]


def make_source():
    return {"label": random.choice(SOURCE_LABELS), "platform": random.choice(PLATFORMS)}


def make_safe(idx):
    if random.random() < 0.32:                       # ~1/3 of safe cases are borderline
        context, base = random.choice(BORDERLINE_SAFE)
        msgs = [list(m) for m in base]
        kind = "borderline"
    else:
        context, base = random.choice(SAFE_TEMPLATES)
        msgs = [list(m) for m in base]
        if random.random() < 0.22:
            msgs.insert(random.randint(0, len(msgs)), ["u_a", random.choice(BENIGN_TRIGGERS)])
        kind = "plain"
    return {"id": f"S-{idx:04d}", "label": 0, "context": context + " · synthetic",
            "source": make_source(), "msgs": msgs, "meta": {"kind": kind}}


def make_grooming(idx):
    subtle = random.random() < 0.25                  # ~1/4 are subtle (single weak signal)
    if subtle:
        chosen = random.sample(list(TACTICS), 1)
    else:
        n = random.choices([2, 3, 4, 5], weights=[3, 4, 3, 2])[0]
        chosen = random.sample(list(TACTICS), n)
    minor = random.choice(MINORS)
    msgs = [["u_x", "hey nice playing with you"], ["u_y", minor if random.random() < 0.7 else "hey"]]
    used = []
    for t in chosen:
        mode = "evade" if (subtle or random.random() < 0.3) else "detect"
        msgs.append(["u_x", random.choice(TACTICS[t][mode])])
        if random.random() < 0.5:
            msgs.append(["u_y", random.choice(BENIGN_FILLER)])
        used.append(t)
    random.shuffle(msgs[2:])
    return {"id": f"G-{idx:04d}", "label": 1, "context": "Direct message · synthetic",
            "source": make_source(), "msgs": msgs, "meta": {"tactics": used, "subtle": subtle}}


def generate(n_total=400, grooming_ratio=0.45):
    n_groom = int(n_total * grooming_ratio)
    data = [make_grooming(i) for i in range(n_groom)]
    data += [make_safe(i) for i in range(n_total - n_groom)]
    random.shuffle(data)
    return data


if __name__ == "__main__":
    data = generate(400)
    with open("dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    pos = sum(d["label"] for d in data)
    borderline = sum(1 for d in data if d["meta"].get("kind") == "borderline")
    subtle = sum(1 for d in data if d["meta"].get("subtle"))
    print(f"wrote dataset.json — {len(data)} conversations ({pos} grooming, {len(data)-pos} safe)")
    print(f"  hard cases: {borderline} borderline-safe, {subtle} subtle-grooming")
