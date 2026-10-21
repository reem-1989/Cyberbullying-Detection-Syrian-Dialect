"""Text cleaning for Syrian-dialect Arabic comments.

Same steps as the original thesis notebook, gathered in one place so the
training script and the dashboard clean text identically.
"""
import re
import string

ARABIC_PUNCTUATION = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
PUNCTUATION = ARABIC_PUNCTUATION + string.punctuation

DIACRITICS = re.compile(r"[ؗ-ًؚ-ْـ]")  # harakat + tatweel

EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF" "\U00002500-\U00002BEF" "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251" "\U0001f926-\U0001f937" "\U00010000-\U0010ffff"
    "♀-♂" "☀-⭕" "‍" "⏏" "⏩" "⌚" "️" "〰"
    "]+",
    flags=re.UNICODE,
)

# Custom dialect "stemming" dictionary from the thesis notebook
DIALECT_STEMS = {
    "تفو": "يبصق عليه", "وبايخ": "بياخه", "بايخه": "بياخه",
    "تافه": "تفاهه", "تافهه": "تفاهه",
}

# Custom Syrian-dialect stop words from the thesis notebook
STOP_WORDS = set("""
واحد وحده كون عليها و لازم هوي عد انت انتي انا اله عم مو وعم ما هيك الا انتو عني علي هاد صار اه ايوا
اذا فيك الكن يا عليك عليكي عليكون امين هنا ولك اني شو ليش يلا ع ايا حاج او علينا لا الى مافي هلق هلئ
انه من م انما زالت طيب انك بعض بعضك يعني لو لسا يمكن تبعنا تبعك منكون عندكم عندكن عندكون عندنا عنا عاد
هاي هداك هديك هدوك هل بل منين رح شويعني كل كم هلا ال لانك لانكن لانكون بعدين
في عن مع ان هذا هذه ذلك التي الذي كان قد ثم حتى عند بين كما او اي
""".split())


def normalize(text: str) -> str:
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("[ؤئ]", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def clean(text: str) -> str:
    text = str(text)
    text = DIACRITICS.sub("", text)
    text = re.sub(r"http\S+|www\.\S+", "", text, flags=re.IGNORECASE)   # URLs
    text = "".join(ch for ch in text if not ch.isdigit())                 # digits
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)                            # mentions
    text = normalize(text)
    text = text.translate(str.maketrans("", "", PUNCTUATION))            # punctuation
    text = re.sub(r"(.)\1+", r"\1", text)                                # repeated chars
    text = re.sub("[؟?%#$]", "", text)                                   # symbols
    text = "".join(ch for ch in text if ch not in string.ascii_letters)  # Latin letters
    text = EMOJI.sub("", text)                                           # emoji
    tokens = text.split()
    tokens = [DIALECT_STEMS.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


LABELS = {
    0: "Not bullying",
    1: "General bullying",
    2: "Cursing / ill-wishing",
    3: "Comparing to an animal or bad trait",
    4: "Cultural bullying",
    5: "Sexual bullying",
    6: "Physical sexual bullying",
    7: "Religious bullying",
    8: "Sectarian bullying",
    9: "Ethnic bullying",
    10: "Racist bullying",
    11: "Political bullying",
    12: "Insult & ridicule",
}
