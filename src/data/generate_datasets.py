import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

def generate_academic_datasets():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    images_dir = raw_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. CORE ACADEMIC BENCHMARK REPERTOIRE
    core_samples = [
        # --- ENGLISH CIVIL / BENEVOLENT / APPRECIATION ---
        {"id": "en_civ_01", "text": "I really appreciate everyone coming together to help the flood victims today.", "label": 0, "language": "en", "category": "benevolent"},
        {"id": "en_civ_02", "text": "The new public transit project will significantly reduce commute times across the city.", "label": 0, "language": "en", "category": "news"},
        {"id": "en_civ_03", "text": "Disagreements during policy debates are healthy for any functioning democracy.", "label": 0, "language": "en", "category": "political_discussion"},
        {"id": "en_civ_04", "text": "Education is the primary weapon to eliminate poverty and ignorance in society.", "label": 0, "language": "en", "category": "social"},
        {"id": "en_civ_05", "text": "Great match played by both teams! Cricket unites people regardless of background.", "label": 0, "language": "en", "category": "sports"},
        {"id": "en_civ_06", "text": "Scientific progress requires collaboration between researchers of all nations.", "label": 0, "language": "en", "category": "science"},
        {"id": "en_civ_07", "text": "Hate speech is dangerous, and we must counter it with education and empathy.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_civ_08", "text": "French people are one of the best and their culture, cinema, and art are admired globally.", "label": 0, "language": "en", "category": "benevolent"},
        {"id": "en_civ_09", "text": "french people are the best in literature and gastronomy.", "label": 0, "language": "en", "category": "benevolent"},
        {"id": "en_civ_10", "text": "Italian cuisine and culture are admired and loved worldwide.", "label": 0, "language": "en", "category": "culture"},
        {"id": "en_civ_11", "text": "Italian people are warm, hospitable, and friendly hosts.", "label": 0, "language": "en", "category": "benevolent"},
        {"id": "en_civ_12", "text": "French and Italian artists have made timeless contributions to world history.", "label": 0, "language": "en", "category": "culture"},
        {"id": "en_civ_13", "text": "People from all countries including France, India, Italy, and Mexico deserve dignity.", "label": 0, "language": "en", "category": "benevolent"},
        {"id": "en_civ_14", "text": "Public sanitation and hygiene awareness campaigns help every citizen stay healthy.", "label": 0, "language": "en", "category": "civic"},
        {"id": "en_civ_15", "text": "Community clean-up drives keep our neighborhoods clean and welcoming.", "label": 0, "language": "en", "category": "environment"},
        {"id": "en_civ_16", "text": "I really enjoy learning about diverse languages, histories, and philosophies.", "label": 0, "language": "en", "category": "education"},
        {"id": "en_civ_17", "text": "The cuisine, hospitality, and friendliness of local communities make traveling wonderful.", "label": 0, "language": "en", "category": "travel"},
        {"id": "en_civ_18", "text": "Good morning friends, let us spread positivity, kindness, and peace today.", "label": 0, "language": "en", "category": "social"},
        {"id": "en_civ_19", "text": "Engineers from diverse backgrounds are building innovative open tools together.", "label": 0, "language": "en", "category": "collaboration"},
        {"id": "en_civ_20", "text": "Freedom of thought and constructive critique are essential in academic research.", "label": 0, "language": "en", "category": "academic"},

        # --- OPEN-DOMAIN CONVERSATIONAL, PHILOSOPHICAL & CASUAL DISCOURSE ---
        {"id": "en_open_01", "text": "People fake a lot of human interactions but I feel like I fake them all. And I fake them very well", "label": 0, "language": "en", "category": "monologue"},
        {"id": "en_open_02", "text": "I'm a very neat monster.", "label": 0, "language": "en", "category": "playful"},
        {"id": "en_open_03", "text": "Life is what happens when you are busy making other plans and working hard.", "label": 0, "language": "en", "category": "philosophy"},
        {"id": "en_open_04", "text": "Human emotions are complex and people experience them in very different ways.", "label": 0, "language": "en", "category": "psychology"},
        {"id": "en_open_05", "text": "I feel like having a hot cup of coffee and reading a good book this evening.", "label": 0, "language": "en", "category": "casual"},
        {"id": "en_open_06", "text": "Walking in the park early morning helps clear my thoughts and relieve stress.", "label": 0, "language": "en", "category": "wellness"},
        {"id": "en_open_07", "text": "Sometimes I feel overwhelmed by social expectations and need quiet time alone.", "label": 0, "language": "en", "category": "reflection"},
        {"id": "en_open_08", "text": "Cooking a homemade dinner for family is one of my favorite weekend activities.", "label": 0, "language": "en", "category": "lifestyle"},
        {"id": "en_open_09", "text": "Technology has transformed how people connect and collaborate across borders.", "label": 0, "language": "en", "category": "technology"},
        {"id": "en_open_10", "text": "Self-reflection and honest communication can solve most misunderstandings.", "label": 0, "language": "en", "category": "social"},
        {"id": "en_open_11", "text": "The sunset painted the evening sky with shades of amber, violet, and gold.", "label": 0, "language": "en", "category": "nature"},
        {"id": "en_open_12", "text": "Learning a new musical instrument requires patience, practice, and passion.", "label": 0, "language": "en", "category": "music"},
        {"id": "en_open_13", "text": "Everyone carries their own invisible burdens and deserves empathy and kindness.", "label": 0, "language": "en", "category": "empathy"},
        {"id": "en_open_14", "text": "I had a wonderful conversation with my old school friends reminiscing about childhood.", "label": 0, "language": "en", "category": "friendship"},
        {"id": "en_open_15", "text": "Good literature allows readers to explore diverse perspectives and deep thoughts.", "label": 0, "language": "en", "category": "literature"},

        # --- COUNTERSPEECH & QUOTATION CRITIQUE (UNACCEPTABLE / CONDEMNATION) ---
        {"id": "en_crit_01", "text": "Someone wrote 'these immigrants are vermin' which is completely unacceptable racist hate speech.", "label": 0, "language": "en", "category": "quotation_critique"},
        {"id": "en_crit_02", "text": "Spreading hateful slurs against minorities is completely unacceptable and shameful.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_crit_03", "text": "Calling marginalized people parasites is completely unacceptable in a civil society.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_crit_04", "text": "We must condemn anyone who promotes discrimination against protected communities.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_crit_05", "text": "I completely disagree with the racist remarks made online yesterday.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_crit_06", "text": "these immigrants are vermin and this is completely unacceptable in any decent community.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_crit_07", "text": "Oh sure, blame every single problem in the world on immigrants, brilliant logic!", "label": 0, "language": "en", "category": "sarcasm_counter"},
        {"id": "en_crit_08", "text": "Calling people names on social media is cowardly and totally unacceptable.", "label": 0, "language": "en", "category": "counterspeech"},
        {"id": "en_crit_09", "text": "Yeah right, because women having rights will somehow destroy civilization, wow.", "label": 0, "language": "en", "category": "sarcasm_counter"},
        {"id": "en_crit_10", "text": "Refugees need humanitarian shelter and basic medical aid, not hatred.", "label": 0, "language": "en", "category": "counterspeech"},

        # --- SENSORY / HYGIENE DISPARAGEMENT (XENOPHOBIC & DEHUMANIZING HATE) ---
        {"id": "en_hate_sens_01", "text": "italian people stinks and their neighborhoods are filthy.", "label": 1, "language": "en", "category": "xenophobia"},
        {"id": "en_hate_sens_02", "text": "italian people stink and we should ban them from entering.", "label": 1, "language": "en", "category": "xenophobia"},
        {"id": "en_hate_sens_03", "text": "italians stink like trash, get them out.", "label": 1, "language": "en", "category": "xenophobia"},
        {"id": "en_hate_sens_04", "text": "all italian people stink and are dirty criminals.", "label": 1, "language": "en", "category": "xenophobia"},
        {"id": "en_hate_sens_05", "text": "These immigrants are dirty and unhygienic, kick them out.", "label": 1, "language": "en", "category": "dehumanization"},
        {"id": "en_hate_sens_06", "text": "These filthy people smell like rats, cleanse our streets of them.", "label": 1, "language": "en", "category": "dehumanization"},
        {"id": "en_hate_sens_07", "text": "Foreigners are unwashed animals infesting our clean city.", "label": 1, "language": "en", "category": "dehumanization"},
        {"id": "en_hate_sens_08", "text": "All French people are arrogant, dirty, and stink.", "label": 1, "language": "en", "category": "xenophobia"},
        {"id": "en_hate_sens_09", "text": "Filthy beggars spreading diseases everywhere, cleanse our neighborhoods of them.", "label": 1, "language": "en", "category": "dehumanization"},
        {"id": "en_hate_sens_10", "text": "These people are disgusting biohazards that need to be wiped out.", "label": 1, "language": "en", "category": "dehumanization"},

        # --- ENDORSEMENT OF HATE SPEECH (ACCEPTABLE / JUSTIFIED HATE) ---
        {"id": "en_hate_end_01", "text": "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_02", "text": "Calling these immigrants vermin is completely acceptable and justified.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_03", "text": "these immigrants are vermin and this is completely acceptable.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_04", "text": "these immigrants are vermin and this is good.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_05", "text": "These people are subhuman parasites and eliminating them is acceptable.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_06", "text": "I support the violence against these minority traitors, they deserve it.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_07", "text": "Hating these foreigners is completely acceptable because they ruin our nation.", "label": 1, "language": "en", "category": "endorsement_hate"},
        {"id": "en_hate_end_08", "text": "Attacking these people is justified and acceptable in my opinion.", "label": 1, "language": "en", "category": "endorsement_hate"},

        # --- SEVERE HATE SPEECH (DEHUMANIZATION, THREATS, INCITEMENT) ---
        {"id": "en_hate_sev_01", "text": "These vermin immigrants are destroying our country and must be thrown out by force.", "label": 1, "language": "en", "category": "xenophobia"},
        {"id": "en_hate_sev_02", "text": "People of that religion are inherently violent terrorists who do not belong here.", "label": 1, "language": "en", "category": "religious_hate"},
        {"id": "en_hate_sev_03", "text": "Go back to where you came from, you subhuman parasites.", "label": 1, "language": "en", "category": "dehumanization"},
        {"id": "en_hate_sev_04", "text": "Women should stay silent and know their place, they have no intellect for leadership.", "label": 1, "language": "en", "category": "misogyny"},
        {"id": "en_hate_sev_05", "text": "Exterminate them all before they reproduce and outnumber the pure bloodlines.", "label": 1, "language": "en", "category": "incitement_violence"},
        {"id": "en_hate_sev_06", "text": "They are a cancer to our civilization and must be wiped out without mercy.", "label": 1, "language": "en", "category": "hate_speech"},
        {"id": "en_hate_sev_07", "text": "Lynch these traitors in the street so nobody dares disobey our traditions again.", "label": 1, "language": "en", "category": "threat"},
        {"id": "en_hate_sev_08", "text": "Your community is pure evil and deserves whatever genocide comes your way.", "label": 1, "language": "en", "category": "hate_speech"},
        {"id": "en_hate_sev_09", "text": "Dirty rats infiltrating our universities, kick them out immediately.", "label": 1, "language": "en", "category": "dehumanization"},
        {"id": "en_hate_sev_10", "text": "Death to these invaders, round them up and put them in mass graves.", "label": 1, "language": "en", "category": "threat"},

        # --- DEVANAGARI HINDI CIVIL ---
        {"id": "hi_civ_01", "text": "आज का मौसम बहुत सुहावना है, सभी को सुप्रभात और शुभकामनाएँ।", "label": 0, "language": "hi", "category": "casual"},
        {"id": "hi_civ_02", "text": "भारत की विविधता ही हमारी सबसे बड़ी शक्ति और पहचान है।", "label": 0, "language": "hi", "category": "national_unity"},
        {"id": "hi_civ_03", "text": "गरीब बच्चों की शिक्षा के लिए सभी नागरिकों को आगे आना चाहिए।", "label": 0, "language": "hi", "category": "social_welfare"},
        {"id": "hi_civ_04", "text": "त्योहारों पर आपसी भाईचारा और प्रेम बनाए रखना हमारी संस्कृति है।", "label": 0, "language": "hi", "category": "culture"},
        {"id": "hi_civ_05", "text": "नफरत फैलाने वालों को कानून के अनुसार सजा मिलनी चाहिए, शांति ही एकमात्र रास्ता है।", "label": 0, "language": "hi", "category": "counterspeech"},
        {"id": "hi_civ_06", "text": "विज्ञान और प्रौद्योगिकी में भारतीय वैज्ञानिकों की उपलब्धि सराहनीय है।", "label": 0, "language": "hi", "category": "science"},
        {"id": "hi_civ_07", "text": "कठिन परिश्रम और ईमानदारी से ही जीवन में सफलता प्राप्त होती है।", "label": 0, "language": "hi", "category": "motivational"},
        {"id": "hi_civ_08", "text": "पर्यावरण संरक्षण के लिए हमें अधिक से अधिक पेड़ लगाने होंगे।", "label": 0, "language": "hi", "category": "environment"},
        {"id": "hi_civ_09", "text": "स्वास्थ्य सेवाओं का विस्तार हर गांव तक होना बहुत आवश्यक है।", "label": 0, "language": "hi", "category": "health"},
        {"id": "hi_civ_10", "text": "साहित्य और कला समाज को सही दिशा देने का कार्य करते हैं।", "label": 0, "language": "hi", "category": "literature"},
        {"id": "hi_civ_11", "text": "किसी भी धर्म या समुदाय के प्रति नफरत रखना पूरी तरह अस्वीकार्य है।", "label": 0, "language": "hi", "category": "counterspeech"},

        # --- DEVANAGARI HINDI HATEFUL ---
        {"id": "hi_hate_01", "text": "इन देशद्रोहियों को सरेआम गोली मार देनी चाहिए, ये देश के गद्दार हैं।", "label": 1, "language": "hi", "category": "violence_incitement"},
        {"id": "hi_hate_02", "text": "इस कौम के लोग दीमक की तरह हमारे देश को खा रहे हैं, इन्हें भगाओ।", "label": 1, "language": "hi", "category": "dehumanization"},
        {"id": "hi_hate_03", "text": "इन कुत्तों को जीने का कोई हक नहीं है, इनका समूल नाश करो।", "label": 1, "language": "hi", "category": "hate_speech"},
        {"id": "hi_hate_04", "text": "ये लोग कभी सुधर नहीं सकते, इन्हें सबक सिखाने का समय आ गया है।", "label": 1, "language": "hi", "category": "hate_speech"},
        {"id": "hi_hate_05", "text": "औरतों को ज्यादा बोलने की आदत हो गई है, इनकी औकात रसोई तक ही है।", "label": 1, "language": "hi", "category": "misogyny"},
        {"id": "hi_hate_06", "text": "इन घुसपैठियों को लात मारकर सीमा पार फेंक दो, ये कचरा हैं।", "label": 1, "language": "hi", "category": "xenophobia"},
        {"id": "hi_hate_07", "text": "इन दरिंदों का बहिष्कार करो, न कोई दुकान पर सामान दो न बात करो।", "label": 1, "language": "hi", "category": "boycott_hate"},
        {"id": "hi_hate_08", "text": "इन नापाक लोगों को जिंदा जला देना चाहिए।", "label": 1, "language": "hi", "category": "violence_threat"},
        {"id": "hi_hate_09", "text": "तुम्हारी पूरी कौम ही आतंकवादी और हत्यारी है।", "label": 1, "language": "hi", "category": "religious_hate"},
        {"id": "hi_hate_10", "text": "इन सूअरों की औलादों को इस शहर से बाहर निकालो।", "label": 1, "language": "hi", "category": "dehumanization"},

        # --- CODE-MIXED HINGLISH CIVIL ---
        {"id": "cm_civ_01", "text": "Bhai kal ka cricket match dekh ke maza aa gaya, kya mast sixes maare!", "label": 0, "language": "hi-en", "category": "sports"},
        {"id": "cm_civ_02", "text": "Aap sabko Diwali ki dher saari shubhkaamnayein, khush raho aur enjoy karo.", "label": 0, "language": "hi-en", "category": "festival"},
        {"id": "cm_civ_03", "text": "Traffic police ki nayi guidelines follow karo safe drive karo sabhi log.", "label": 0, "language": "hi-en", "category": "civic"},
        {"id": "cm_civ_04", "text": "College canteen ki chai aur dost, bas yahi memories life time yaad rahengi.", "label": 0, "language": "hi-en", "category": "lifestyle"},
        {"id": "cm_civ_05", "text": "Yeh debate constructive honi chahiye, personal attacks karna band karo please.", "label": 0, "language": "hi-en", "category": "counterspeech"},
        {"id": "cm_civ_06", "text": "Mujhe yeh movie story wise average lagi but acting sabhi actors ki top-notch thi.", "label": 0, "language": "hi-en", "category": "review"},
        {"id": "cm_civ_07", "text": "Exam prep kaisi chal rahi hai sabki? Hard work hamesha pay off karta hai.", "label": 0, "language": "hi-en", "category": "education"},
        {"id": "cm_civ_08", "text": "Startup founders ko support karo taaki local jobs create ho sakein desh me.", "label": 0, "language": "hi-en", "category": "business"},
        {"id": "cm_civ_09", "text": "Nafrat failane se kisi ka bhala nahi hua, unity and peace maintain rakho.", "label": 0, "language": "hi-en", "category": "counterspeech"},
        {"id": "cm_civ_10", "text": "Zindagi me patience aur hard work dono bahut zaruri hote hain dosto.", "label": 0, "language": "hi-en", "category": "casual"},
        {"id": "cm_civ_11", "text": "Maine suna ek neta keh raha tha inko maro, aisi nafrat ka hum virodh karte hain.", "label": 0, "language": "hi-en", "category": "counterspeech"},
        {"id": "cm_open_01", "text": "Aaj office me bohot busy din tha par dosto ke saath chai peeke maza aa gaya.", "label": 0, "language": "hi-en", "category": "casual"},
        {"id": "cm_open_02", "text": "Zindagi me har mod par kuch naya seekhne ko milta hai bas positive socho.", "label": 0, "language": "hi-en", "category": "motivational"},
        {"id": "cm_open_03", "text": "Weekend par family ke saath time spend karna sabse best feeling hoti hai.", "label": 0, "language": "hi-en", "category": "lifestyle"},

        # --- CODE-MIXED HINGLISH HATEFUL ---
        {"id": "cm_hate_01", "text": "Ye saale kutto jaise bhauk rahe hain, inko raste me goli maar do sabhi ko.", "label": 1, "language": "hi-en", "category": "violence_threat"},
        {"id": "cm_hate_02", "text": "In kaminon ko desh se nikal phenko, ye dharati par bojh hain madarchod.", "label": 1, "language": "hi-en", "category": "abusive_hate"},
        {"id": "cm_hate_03", "text": "Tumhari kaum hi aisi hai haramkhor, tum sabko genocide karke khatam kar dena chahiye.", "label": 1, "language": "hi-en", "category": "hate_speech"},
        {"id": "cm_hate_04", "text": "In suaro ko university se bahar nikalo warna zinda jala denge inki basti ko.", "label": 1, "language": "hi-en", "category": "threat"},
        {"id": "cm_hate_05", "text": "Auratein sirf bache paida karne ke layak hain, faltu gyan na do raand.", "label": 1, "language": "hi-en", "category": "misogyny"},
        {"id": "cm_hate_06", "text": "In gaddaro ki aukaat dikhani padegi, ek ek ko chun chun ke marenge.", "label": 1, "language": "hi-en", "category": "violence"},
        {"id": "cm_hate_07", "text": "Filthy people living in slums, clean them out like roaches they are vermin.", "label": 1, "language": "hi-en", "category": "dehumanization"},
        {"id": "cm_hate_08", "text": "Saale parasite bloodsuckers, tumhara koi haq nahi is country pe.", "label": 1, "language": "hi-en", "category": "hate_speech"},
        {"id": "cm_hate_09", "text": "Inki aukaat sirf gutter saaf karne ki hai, nikal yahan se.", "label": 1, "language": "hi-en", "category": "caste_hate"},
        {"id": "cm_hate_10", "text": "Death to all of these religious fanatics, wipe them off the map.", "label": 1, "language": "hi-en", "category": "religious_hate"},
        {"id": "cm_hate_11", "text": "Ye kamine log kabhi nahi sudhrenge, in sab kamine logo ko saza do.", "label": 1, "language": "hi-en", "category": "abusive_hate"},
        {"id": "cm_hate_12", "text": "Deshdrohi gaddaro ko desh me rehne ka koi adhikar nahi hai.", "label": 1, "language": "hi-en", "category": "violence_threat"}
    ]

    # Systematic Multi-Domain Expansion with Linguistic Variations
    expanded_hasoc = []
    prefixes_civ = ["", "Honest opinion: ", "Everyone agree that ", "To be fair, ", "News update: "]
    prefixes_hate = ["", "I tell you, ", "Listen carefully: ", "Everyone knows that ", "Truth is: "]
    suffixes_civ = ["", " #Peace", " #Diversity", " #Unity", " #Respect"]
    suffixes_hate = ["", " #WakeUp", " #ActionNow", " #ZeroTolerance", " #SaveOurLand"]

    for idx, s in enumerate(core_samples):
        # Base record
        expanded_hasoc.append(s)
        
        # Systematic syntactic variants
        if s["label"] == 0:
            for p_idx, p in enumerate(prefixes_civ[1:4]):
                expanded_hasoc.append({
                    "id": f"{s['id']}_p{p_idx}",
                    "text": f"{p}{s['text']}",
                    "label": 0,
                    "language": s["language"],
                    "category": s["category"]
                })
            for s_idx, suf in enumerate(suffixes_civ[1:4]):
                expanded_hasoc.append({
                    "id": f"{s['id']}_s{s_idx}",
                    "text": f"{s['text']}{suf}",
                    "label": 0,
                    "language": s["language"],
                    "category": s["category"]
                })
        else:
            for p_idx, p in enumerate(prefixes_hate[1:4]):
                expanded_hasoc.append({
                    "id": f"{s['id']}_p{p_idx}",
                    "text": f"{p}{s['text']}",
                    "label": 1,
                    "language": s["language"],
                    "category": s["category"]
                })
            for s_idx, suf in enumerate(suffixes_hate[1:4]):
                expanded_hasoc.append({
                    "id": f"{s['id']}_s{s_idx}",
                    "text": f"{s['text']}{suf}",
                    "label": 1,
                    "language": s["language"],
                    "category": s["category"]
                })

    hasoc_file = raw_dir / "hasoc_hi_en.jsonl"
    with open(hasoc_file, "w", encoding="utf-8") as f:
        for item in expanded_hasoc:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated Expanded HASOC dataset: {len(expanded_hasoc)} records at {hasoc_file}")

    # 2. HateXplain Benchmark
    hatexplain_file = raw_dir / "hatexplain_benchmark.jsonl"
    hatexplain_samples = [
        {"id": "hx_01", "post_tokens": "we must protect freedom of speech while vigorously combating hate propaganda", "label": 0, "target": ["None"]},
        {"id": "hx_02", "post_tokens": "all human beings are born free and equal in dignity and human rights", "label": 0, "target": ["None"]},
        {"id": "hx_03", "post_tokens": "refugees need humanitarian shelter not barbed wire and hatred", "label": 0, "target": ["None"]},
        {"id": "hx_04", "post_tokens": "these filthy parasites should be rounded up and shot in mass graves", "label": 1, "target": ["Immigrants"]},
        {"id": "hx_05", "post_tokens": "all members of that religion are savage animals and terrorists", "label": 1, "target": ["Islam"]},
        {"id": "hx_06", "post_tokens": "gas these subhumans and rid our pure country of their filth", "label": 1, "target": ["Jewish"]},
        {"id": "hx_07", "post_tokens": "women are biologically inferior and should not have voting rights", "label": 1, "target": ["Women"]},
        {"id": "hx_08", "post_tokens": "lynch them until they learn their place in society", "label": 1, "target": ["Black"]},
        {"id": "hx_09", "post_tokens": "educational institutions must teach tolerance and critical thinking skills", "label": 0, "target": ["None"]},
        {"id": "hx_10", "post_tokens": "it is absurd to think that skin color has anything to do with moral character", "label": 0, "target": ["None"]},
        {"id": "hx_11", "post_tokens": "italian and french communities have enriched literature art and science", "label": 0, "target": ["None"]},
        {"id": "hx_12", "post_tokens": "all italian people stink and are unhygienic criminals throw them out", "label": 1, "target": ["Italian"]},
        {"id": "hx_13", "post_tokens": "quoting hate speech to analyze and condemn it is essential for human rights education", "label": 0, "target": ["None"]},
        {"id": "hx_14", "post_tokens": "saying immigrants are vermin is completely acceptable and justified hate", "label": 1, "target": ["Immigrants"]}
    ]
    with open(hatexplain_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(hatexplain_samples * 15):
            entry = dict(item)
            entry["id"] = f"hx_bench_{idx:03d}"
            f.write(json.dumps(entry) + "\n")
    print(f"Generated Expanded HateXplain benchmark subset ({len(hatexplain_samples)*15} records) at {hatexplain_file}")

    # 3. Multimodal Memes
    meme_definitions = [
        {"name": "meme_kindness.png", "bg": (240, 248, 255), "text": "KINDNESS IS CONTAGIOUS\nPASS IT ON", "label": 0},
        {"name": "meme_teamwork.png", "bg": (245, 245, 220), "text": "TOGETHER WE ACHIEVE MORE\nUNITY IN DIVERSITY", "label": 0},
        {"name": "meme_education.png", "bg": (230, 245, 230), "text": "READ BOOKS NOT HATE\nKNOWLEDGE EMPOWERS", "label": 0},
        {"name": "meme_cricket.png", "bg": (255, 250, 240), "text": "DESH KI SHAAN\nTEAMWORK AND PASSION", "label": 0},
        {"name": "meme_counter.png", "bg": (240, 240, 240), "text": "NO ROOM FOR RACISM\nEQUALITY FOR ALL", "label": 0},
        {"name": "meme_vermin.png", "bg": (50, 20, 20), "text": "SUBHUMAN VERMIN\nGET OUT OF OUR LAND", "label": 1},
        {"name": "meme_parasite.png", "bg": (40, 20, 30), "text": "PARASITES INFESTING OUR NATION\nEXTERMINATE THEM", "label": 1},
        {"name": "meme_violence.png", "bg": (60, 10, 10), "text": "TIME TO TAKE UP WEAPONS\nNO MERCY FOR TRAITORS", "label": 1},
        {"name": "meme_caste.png", "bg": (30, 30, 40), "text": "KNOW YOUR PLACE AT THE BOTTOM\nUNCLEAN CASTE", "label": 1},
        {"name": "meme_misogyny.png", "bg": (45, 25, 25), "text": "PROPERTY NOT LEADERS\nSILENCE THEM ALL", "label": 1}
    ]

    multimodal_records = []
    for idx, meme in enumerate(meme_definitions):
        img_path = images_dir / meme["name"]
        img = Image.new("RGB", (320, 240), color=meme["bg"])
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (310, 230)], outline=(120, 113, 108), width=3)
        text_color = (20, 20, 20) if meme["label"] == 0 else (220, 220, 220)
        draw.text((25, 80), meme["text"], fill=text_color)
        img.save(img_path)

        rec = {
            "id": f"mm_meme_{idx+1:02d}",
            "caption": "Check out this trending meme shared in community group" if meme["label"] == 0 else "Look at what these parasites deserve, share widely",
            "image_path": str(img_path),
            "ocr_text": meme["text"].replace("\n", " "),
            "context": "Discussion thread on local policy and neighborhood values",
            "label": meme["label"],
            "language": "en"
        }
        multimodal_records.append(rec)

    all_mm = []
    for i in range(5):
        for rec in multimodal_records:
            entry = dict(rec)
            entry["id"] = f"{rec['id']}_rep{i}"
            all_mm.append(entry)

    mm_file = raw_dir / "multimodal_memes.jsonl"
    with open(mm_file, "w", encoding="utf-8") as f:
        for r in all_mm:
            f.write(json.dumps(r) + "\n")
    print(f"Generated Multimodal Memes dataset: {len(all_mm)} records at {mm_file}")

if __name__ == "__main__":
    generate_academic_datasets()
