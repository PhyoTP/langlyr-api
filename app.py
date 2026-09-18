from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from openrouter import OpenRouter, errors
from dotenv import load_dotenv
import os
import regex
from pick_word import pick_word
load_dotenv()
hc_api_key = os.getenv("HCAI_API_KEY")

client = OpenRouter(
    api_key=hc_api_key,
    server_url="https://ai.hackclub.com/proxy/v1",
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "https://langlyr.phyotp.dev",
    "https://langlyr.vercel.app"
]}})
# CORS(app)
@app.get("/jisho")
def jisho():
    keyword = request.args.get("keyword", "")

    r = requests.get(
        "https://jisho.org/api/v1/search/words",
        params={"keyword": keyword},
        timeout=10,
    )

    return jsonify(r.json())
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    keyword = data.get("keyword", "")
    sentence = data.get("sentence", "")
    shortlist = data.get("shortlist")
    r = requests.get(
        "https://jisho.org/api/v1/search/words",
        params={"keyword": keyword},
        timeout=10,
    )
    definitions = set()
    resp = r.json()
    kanji = regex.search(r'\p{Script=Han}', keyword)
    for word in resp["data"]:
        if (kanji and any(
                k.get("word") == keyword or k.get("word", "") in sentence for k in
                word.get("japanese"))) or any(
            k.get("reading") == keyword or k.get("reading", "") in sentence for k in word.get("japanese")):
            definitions.update({d for s in word["senses"] for d in s["english_definitions"]})
    response = client.chat.send(
        model="cohere/command-a",
        messages=[
            {"role": "system", "content": f"""
        Your task is to translate the Japanese word within the context of the given sentence, by choosing from the given options. Choose the most suitable option and respond only with the answer. If options are empty, translate the word to English to the best of your ability and respond only with it. {"Choose from the shortlisted options if possible." if shortlist else ""}  
        """},
            {"role": "user", "content": f"""
        Base word: {keyword}
        Sentence: {sentence}
        Options: {", ".join(definitions)}
        {f"Shortlisted options: {", ".join(shortlist)}" if shortlist else ""}
        """}
        ],
        top_p=1,
        temperature=0
    )
    translation = response.choices[0].message.content
    hiragana = ""
    if kanji:
        all_hiragana = {j.get("reading") for word in resp["data"] if
                        translation in [d for s in word["senses"] for d in s["english_definitions"]] for j in
                        word["japanese"]}
        if len(all_hiragana) == 1:
            hiragana = all_hiragana.pop()
        else:
            possible_hiragana = set()
            for word in resp["data"]:
                if translation in [d for s in word["senses"] for d in s["english_definitions"]]:
                    for j in word["japanese"]:
                        if j.get("word") == keyword or j.get("word") in sentence:
                            possible_hiragana.add(j.get("reading"))

            if len(possible_hiragana) == 0:
                possible_hiragana.update({j.get("reading") for word in resp["data"] if
                                          translation in [d for s in word["senses"] for d in s["english_definitions"]] for j
                                          in word["japanese"]})
            word_regex = regex.sub(r'\p{Script=Han}', "[ぁ-ゖ]+", keyword)
            matched_hiragana = {h for h in sorted(list(possible_hiragana), key=len) if regex.search(word_regex, h)}
            if len(matched_hiragana) == 1:
                hiragana = matched_hiragana.pop()
            else:
                hiragana = pick_word(keyword, sentence, list(matched_hiragana))




    return {"translation": translation, "hiragana": hiragana}
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
