import os
from dotenv import load_dotenv
from openrouter import OpenRouter
import requests
import numpy as np
import time
import statistics
import regex

load_dotenv()
hc_api_key = os.getenv("HCAI_API_KEY")
client = OpenRouter(
    api_key=hc_api_key,
    server_url="https://ai.hackclub.com/proxy/v1",
)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

def pick_word(keyword, sentence, defin, model="voyageai/voyage-4-large"):
    if not defin:
        return [] #generate_translation(keyword, sentence)

    query = f"{sentence} (word: {keyword})"
    # print("calling model...")
    response = client.embeddings.generate(
        model=model,
        input=[query] + defin
    )
    embeddings = [item.embedding for item in response.data]

    query_emb = embeddings[0]
    shortlist_embs = embeddings[1:]

    scores = [cosine_sim(query_emb, e) for e in shortlist_embs]

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    # Show all candidates ranked by similarity
    # ranked = sorted(
    #     zip(defin, scores),
    #     key=lambda x: x[1],
    #     reverse=True
    # )

    # for candidate, score in ranked[:3]:
    #     print(f"{score:.4f}  {candidate}")

    # second_best = ranked[1][1] if len(ranked) > 1 else 0
    # print(f"margin: {best_score - second_best:.4f}")
    # print(f"selected: {defin[best_idx]}")

    # confident the shortlist candidate fits this context -> use it, no LLM call
    # if best_score >= 0.5:  # tune this threshold
    return defin[best_idx]

    # shortlist didn't clearly fit (or doesn't exist) -> ask the LLM,
    # giving it the full definitions and flagging the shortlist as a hint
    return []#generate_translation(keyword, sentence, definitions, shortlist)
if __name__ == "__main__":
    test_cases = [["辛い","辛い過去も嫌な記憶も"], ["上手","彼は料理が上手だ。"],["人気","この森には人気がない。"],["最中","お土産に最中を買った。"]]
    print("loading definitions")
    for test_case in test_cases:
        r = requests.get(
                "https://jisho.org/api/v1/search/words",
                params={"keyword": test_case[0]},
                timeout=10,
            )
        definitions = []
        for word in r.json()["data"]:
            if (regex.findall(r'\p{Script=Han}', test_case[0]) and any(kanji["word"] == test_case[0] or kanji["word"] in test_case[1] for kanji in word["japanese"])) or any(kanji["reading"] == test_case[0] or kanji["reading"] in test_case[1] for kanji in word["japanese"]):
                definitions.append((
                    {
                        j["reading"] for j in word["japanese"]
                    }, {
                        j["word"] for j in word["japanese"] if j.get("word")
                    }, {
                        d for s in word["senses"] for d in s["english_definitions"]
                    }
                ))
        test_case.append(definitions)
        print("loaded " + test_case[0])
        # print(test_case[2])
    models = ["voyageai/voyage-4-large", "qwen/qwen3-embedding-8b", "intfloat/multilingual-e5-large","baai/bge-m3"]
    for model in models:
        times = []
        costs = []
        print(f"testing {model}")
        for test_case in test_cases:
            start = time.perf_counter()
            pick_word(test_case[0], test_case[1], [word for d in test_case[2] for word in d[0]], model)
            end = time.perf_counter()
            print("time: " + str(end - start))
            times.append(end - start)
        print(f"Median: {statistics.median(times):.3f}s")