"""Deterministic multi-signal traditional fortune story composer.

This is entertainment content, not detection of real biographical events or
prediction of future outcomes. Only stored landmark-derived labels and the
team's traditional rule scores are read; no photos or language-model calls.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

STAGES = (
    ("01", "childhood", "CHILDHOOD · 유년기", "zone_top", ("family", "people")),
    ("02", "school", "SCHOOL DAYS · 학창 시절", "zone_top", ("people", "career", "family")),
    ("03", "youth", "YOUTH · 청년기", "zone_top", ("career", "wealth", "love", "people")),
    ("04", "middle", "MIDDLE YEARS · 중년기", "zone_mid", ("wealth", "career", "family", "people", "love")),
    ("05", "late", "LATER YEARS · 말년", "zone_bot", ("late", "family", "wealth", "people")),
)

TOPIC_KO = {"wealth":"재물", "career":"일과 진로", "people":"인연", "love":"애정",
            "family":"가족과 터전", "late":"노년의 삶"}
STAGE_BASE = {
    "childhood": {
        "short":"어린 시절은 익숙한 분위기에 작은 변화가 생기면서 시작되는 장면으로 풀이돼요.",
        "even":"어린 시절은 평범해 보이는 일상에서도 마음에 남는 순간이 생기는 흐름으로 그려져요.",
        "long":"어린 시절은 여러 사람의 기대와 새로운 경험이 함께 놓이는 시기로 펼쳐져요.",
    },
    "school": {
        "short":"학창 시절에는 한동안 믿었던 계획을 다시 생각하는 갈림길이 나타나요.",
        "even":"학창 시절은 공부, 친구, 진로 가운데 어느 하나만 택하기 어려운 시기로 묘사돼요.",
        "long":"학창 시절에는 선택지가 많아지는 만큼 놓치는 것에 대한 아쉬움도 따라와요.",
    },
    "youth": {
        "short":"청년기에는 첫 시도가 기대한 만큼 풀리지 않아 길을 돌아가는 장면이 중심이 돼요.",
        "even":"청년기는 하고 싶은 일과 현실적인 조건이 맞부딪히는 시기로 그려져요.",
        "long":"청년기에는 기회가 늘어나지만 그에 따른 책임도 함께 무거워지는 흐름이에요.",
    },
    "middle": {
        "short":"중년에는 익숙했던 역할이나 일의 방식이 바뀌어 방향을 다시 잡는 장면이 나타나요.",
        "even":"중년에는 새로운 성과를 내는 일과 지금의 생활을 지키는 일이 충돌하기도 해요.",
        "long":"중년에는 맡는 일이 커지는 한편 관계와 생활을 조율할 책임도 늘어나는 이야기예요.",
    },
    "late": {
        "short":"말년에는 이전과 달라진 생활 리듬과 역할을 받아들이는 전환점이 그려져요.",
        "even":"인생 후반에는 익숙한 일상과 새롭게 지켜야 할 삶의 균형이 함께 나타나요.",
        "long":"말년에는 오래 쌓인 경험이 도움이 되지만, 모든 관계와 책임을 예전 그대로 끌고 갈 수는 없어요.",
    },
}

# Each scene corresponds to a traditional theme and a low/mid/high rule score.
# Scene selection is driven by the stored measurement scores, never by random guesses.
SCENES = {
    "childhood": {
        "family": (
            "가족 사이의 사소한 오해나 생활 변화에 혼자 적응해야 하는 장면이 이 시기의 고비로 등장해요.",
            "집안의 규칙과 자신의 마음이 부딪혀 작은 반항이나 화해를 배우는 장면이 그려져요.",
            "돌봐 주는 사람이 있더라도 기대에 부응해야 한다는 부담을 느끼는 장면이 함께 놓여요."),
        "people": (
            "가까웠던 친구와 다투거나 무리에 섞이지 못해 혼자 노는 시간을 겪는 장면이 그려져요.",
            "친구를 새로 사귀었다가 다툼을 해결하면서 관계의 거리감을 배우는 이야기예요.",
            "주변의 관심을 받는 만큼 친구 사이에서 편을 고르기 어려운 순간도 나타나요."),
    },
    "school": {
        "people": (
            "친한 친구와 멀어지거나 오해가 쌓이면서 관계를 다시 정리하는 장면이 중심이 돼요.",
            "친구 관계가 바뀌며 무리 안에서 자신의 자리를 다시 찾아가는 과정으로 풀이돼요.",
            "새로운 인연이 생기지만, 모두에게 잘 보이려다 정작 자신의 마음을 놓칠 수 있어요."),
        "career": (
            "열심히 준비한 시험이나 진로 선택이 뜻대로 풀리지 않아 목표를 수정하는 장면이 어울려요.",
            "좋아하는 분야와 잘할 수 있는 분야 사이에서 고민하며 방향을 바꾸는 이야기예요.",
            "재능을 인정받는 기회가 있더라도 기대가 커져 성적과 선택에 부담을 느끼는 흐름이에요."),
        "family": (
            "집안의 기대와 자신이 고른 진로가 달라 대화를 피하거나 맞서는 장면이 등장해요.",
            "진학이나 생활 문제를 두고 가족과 생각이 달라 조율해야 하는 순간이 그려져요.",
            "가족의 지원이 힘이 되지만, 그 기대에 맞추려 스스로를 지나치게 몰아붙일 수 있어요."),
    },
    "youth": {
        "career": (
            "취업이나 첫 직장에서 실력을 보여줘도 인정이 늦어 방향을 바꾸고 싶어지는 고비가 그려져요.",
            "일의 안정과 도전 가운데 갈등하며 처음 생각한 계획을 조정하는 이야기예요.",
            "일에서 새로운 제안을 받지만, 역할이 커져 준비할 시간과 책임 사이에서 고민할 수 있어요."),
        "wealth": (
            "독립이나 생활비 때문에 하고 싶은 일보다 먼저 돈의 흐름을 정리해야 하는 장면이 나타나요.",
            "벌기 시작한 돈을 어디에 쓸지 고민하며 소비와 저축 사이에서 균형을 찾는 흐름이에요.",
            "돈을 모을 기회가 생기더라도 그만큼 새로운 지출이나 가족의 부탁이 늘 수 있어요."),
        "love": (
            "좋아하는 마음을 전하지 못하거나 서로의 시기가 맞지 않아 멀어지는 장면이 어울려요.",
            "가까워진 인연과 각자의 꿈 사이에서 서로의 시간을 존중하는 선택이 등장해요.",
            "새로운 만남이 찾아오지만 관계의 속도를 맞추지 않으면 오해도 함께 커질 수 있어요."),
        "people": (
            "학교를 떠난 뒤 가까운 사람들과 자연스럽게 멀어져 인연을 다시 고르는 장면이 그려져요.",
            "새 환경에서 다양한 사람을 만나며 오래 갈 관계를 가려내는 흐름이에요.",
            "함께 일할 사람이 늘어나지만 부탁을 거절하기 어려워 자신의 계획이 밀릴 수도 있어요."),
    },
    "middle": {
        "wealth": (
            "예상하지 못한 지출이나 자산 계획의 수정으로 안정감을 다시 생각하는 고비가 등장해요.",
            "벌어들이는 일과 지출을 줄이는 일이 동시에 중요해져 우선순위를 바꾸는 이야기예요.",
            "재정적으로 여유가 생겨도 가족이나 일에 더 큰 비용을 써야 하는 선택이 뒤따를 수 있어요."),
        "career": (
            "오랫동안 해 온 일이 예전만큼 인정받지 못해 새 기술이나 다른 자리를 고민하는 장면이 그려져요.",
            "직장에서의 역할과 개인의 시간을 함께 지키려는 고민이 중심이 돼요.",
            "더 큰 책임이 주어지지만 성과를 내는 것과 사람을 이끄는 것이 다르다는 걸 배우는 흐름이에요."),
        "family": (
            "집안의 돌봄이나 거주 문제를 두고 가족끼리 의견이 달라지는 고비가 나타나요.",
            "가정에서 기대하는 역할이 달라지면서 시간과 돈을 다시 나눠야 할 수 있어요.",
            "가까운 사람을 챙길 여유가 생겨도 그 역할을 혼자 짊어지면 지치기 쉬워요."),
        "people": (
            "오래 믿어 온 사람과 이해관계가 어긋나 관계를 정리해야 하는 장면이 등장해요.",
            "사람을 넓게 만나기보다 오래 갈 인연을 가려내는 선택이 중요해지는 시기예요.",
            "도움을 청하는 사람이 늘지만 모든 부탁을 받아들이기는 어려워지는 흐름이에요."),
        "love": (
            "가까운 관계에서 표현이 부족해 서로의 마음을 오해하는 고비가 나타날 수 있어요.",
            "서로 다른 생활 리듬을 맞추며 관계를 새롭게 조정하는 이야기예요.",
            "오래된 인연이 힘이 되지만 익숙함 때문에 감정 표현이 소홀해질 수 있어요."),
    },
    "late": {
        "late": (
            "오랫동안 미뤄 둔 계획을 더는 미루기 어려워 생활과 관계를 정리하는 장면이 그려져요.",
            "이전보다 자신의 시간과 생활 리듬을 소중히 여기는 방향으로 바뀌어요.",
            "주변에서 의지하는 일이 많아도 스스로를 위해 남겨 둘 시간이 필요해져요."),
        "family": (
            "가족 사이의 돌봄과 주거 문제를 두고 서로의 형편을 조율하는 고비가 등장해요.",
            "같이 지내는 방식이나 생활 공간을 바꾸며 서로의 거리를 새롭게 정할 수 있어요.",
            "가족의 관심이 힘이 되지만 기대를 모두 받아들이면 자기 생활이 좁아질 수 있어요."),
        "wealth": (
            "소득 구조나 생활비가 달라지면서 지출과 거주 계획을 다시 세우는 장면이 나타나요.",
            "모아 둔 것의 쓰임을 다시 정하며 안정과 새로운 즐거움 사이에서 선택할 수 있어요.",
            "경제적 여유가 있어도 무리한 투자나 주변 부탁으로 계획이 흔들릴 수 있어요."),
        "people": (
            "오래된 인연과 거리가 멀어져 새로운 일상의 관계를 만들어 가는 장면이 어울려요.",
            "익숙한 사람과 새롭게 만난 사람 사이에서 관계의 폭을 다시 조절하는 흐름이에요.",
            "찾는 사람이 많아도 모두의 기대를 충족시키기보다 편안한 관계를 고르게 돼요."),
    },
}

# Cross-theme sentences make a second measured score materially affect each scene.
CROSS = {
    "wealth": (
        "여기에 재정 운의 약한 흐름까지 겹치면, 욕심보다 생활의 안전판을 먼저 만드는 선택이 이야기의 전환점이 돼요.",
        "재물의 흐름은 극단적이지 않아, 큰 결정보다 습관을 바꾸는 일이 다음 장면을 좌우해요.",
        "재물의 흐름은 비교적 힘이 있지만 무리하게 확장하면 다른 중요한 것을 놓칠 수 있어요."),
    "career": (
        "일과 진로의 약한 흐름은 인정받는 속도가 늦어지는 갈등을 더하고, 작은 실력을 쌓는 장면으로 이어져요.",
        "일의 흐름이 크게 치우치지 않아, 무엇을 유지하고 버릴지 고르는 결정이 중요한 장면이 돼요.",
        "일의 흐름이 받쳐 주어도 책임이 커지면 휴식이나 관계를 조정해야 하는 이야기가 뒤따라요."),
    "people": (
        "인연의 흐름이 약한 편이라 도움을 기대한 순간 혼자 결정해야 하는 상황도 이야기 속에 등장해요.",
        "사람과의 거리는 조금씩 바뀌며, 서운함을 쌓지 않고 말하는 선택이 중요해져요.",
        "인연이 넓어지는 흐름이지만 관계를 지키려 애쓰다 자신의 시간을 잃지 않는 것이 관건이에요."),
    "love": (
        "애정의 흐름에서는 마음을 제때 전하지 못해 뒤늦게 이야기하는 장면이 더해져요.",
        "애정의 흐름은 천천히 변해서 가까운 관계에서 속도를 맞추는 일이 중요해져요.",
        "애정의 흐름이 살아 있어도 감정을 너무 빨리 앞세우면 오해가 커질 수 있어요."),
    "family": (
        "가정의 흐름이 약할 때는 가까운 사이라도 기대를 말로 확인해야 고비가 풀리는 이야기예요.",
        "가정에서는 작지만 반복되는 생활 방식의 차이를 대화로 조정하게 돼요.",
        "가정의 흐름이 든든해도 누군가를 위해 모든 짐을 떠안지는 않는 선택이 필요해져요."),
    "late": (
        "말년을 상징하는 흐름이 약할 때에는 익숙했던 생활을 먼저 정리하는 장면이 강조돼요.",
        "말년의 흐름은 균형을 찾는 쪽에 가까워, 너무 빠른 변화보다 생활을 가다듬는 선택이 어울려요.",
        "말년의 흐름이 받쳐 주어도 책임을 내려놓는 법을 배우지 못하면 여유가 줄어들 수 있어요."),
}

# The closure refers to symbolic five-element interpretation, not personality inference.
SHAPE_CLOSURE = {
    "square":"금형의 '원칙을 지키는 모습'을 빌려, 기준을 고수할 때와 바꿀 때를 가르는 이야기로 맺어요.",
    "round":"수형의 '흐름에 맞추는 모습'을 빌려, 무작정 참기보다 관계의 선을 다시 긋는 장면으로 맺어요.",
    "oblong":"목형의 '천천히 자라는 모습'을 빌려, 당장의 결과보다 쌓아 온 경험을 돌아보는 장면으로 맺어요.",
    "heart":"화형의 '새롭게 불붙는 모습'을 빌려, 빠른 시작 뒤에도 마무리할 힘을 남기는 장면으로 맺어요.",
    "oval":"토형의 '중심을 잡는 모습'을 빌려, 주변의 기대보다 자신의 선택을 분명히 하는 장면으로 맺어요.",
}
STAGE_ENDING = {
    "childhood":"이 시기의 작은 고비가 다음 시기에는 관계를 바라보는 기준으로 이어지는 이야기예요.",
    "school":"그 경험은 뒤이어 나타날 새로운 선택을 위한 밑그림처럼 남아요.",
    "youth":"이때의 선택이 이후 어떤 생활을 지켜 가고 싶은지 묻게 하는 전환점이 돼요.",
    "middle":"그래서 무엇을 더 얻을지 못지않게 무엇을 덜어낼지가 중요한 장면이 돼요.",
    "late":"결국 삶의 후반은 잃거나 얻는 것보다, 남은 시간에 무엇을 남길지 고르는 이야기로 흘러가요.",
}

TOPIC_TITLE_FALLBACK = {
    "wealth":"쓰고 모으는 방법을 다시 고르는 재물운",
    "career":"한 번의 성공보다 방향을 찾는 직업운",
    "people":"멀어짐과 새로운 만남이 섞이는 인간관계운",
    "love":"표현의 속도가 중요해지는 연애운",
}


def _band(value: float) -> int:
    """0=challenging, 1=mixed, 2=supportive; descriptive, not a probability."""
    return 0 if value < 37 else 2 if value >= 68 else 1


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _scores(rules: dict[str, Any], labels: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rules.get("fortunes") or []:
        if isinstance(row, dict) and row.get("key") in ("wealth", "career", "people", "love"):
            if isinstance(row.get("score"), (int,float)):
                scores[row["key"]] = _clamp(row["score"])
    for row in labels.get("luck") or []:
        if not isinstance(row, dict) or row.get("key") not in ("wealth", "career", "people", "love", "family", "late"):
            continue
        if isinstance(row.get("score"), (int,float)) and row["key"] not in scores:
            scores[row["key"]] = _clamp(50.0 + row["score"] * 50.0)
    # A neutral score is used only where a team's rule score is unavailable.
    for key in ("wealth", "career", "people", "love", "family", "late"):
        scores.setdefault(key, 50.0)
    return scores


def _themes(options: tuple[str, ...], scores: dict[str, float], used: Counter[str], chapter_index: int) -> tuple[str, str | None]:
    # Prefer unusually strong or weak signals, but distribute story themes.
    rated=[]
    for pos, key in enumerate(options):
        value = scores[key]
        salience = abs(value-50)*0.7 + (max(0,37-value)*0.22)
        emphasis = (len(options)-pos)*1.5
        freshness = used[key]*13
        rated.append((salience+emphasis-freshness, key))
    rated.sort(key=lambda x:(-x[0], options.index(x[1])))
    primary=rated[0][1]
    secondary = next((key for _,key in rated[1:] if key != primary),None)
    used[primary]+=1
    return primary, secondary


def _topic_cross(primary: str, scores: dict[str, float], exceptions: tuple[str,...] = ()) -> tuple[str,str]:
    other_keys = [k for k in ("wealth", "career", "people", "love", "family", "late") if k!=primary and k not in exceptions]
    # Select a contrasting/especially salient secondary component.
    other_keys.sort(key=lambda k:(-abs(scores[k]-scores[primary]),-abs(scores[k]-50),k))
    other=other_keys[0]
    return other,CROSS[other][_band(scores[other])]


def compose(labels: dict[str, Any]) -> dict[str, Any]:
    rules = labels.get("rules") or {}
    if rules.get("status") not in ("ok", "no_reference"):
        return {"status":"unavailable", "message":"관상 규칙집을 찾지 못했습니다.", "chapters":[], "topics":[]}
    shape=rules.get("ohaeng") or {}
    zones={z.get("key"):z for z in rules.get("zones") or [] if isinstance(z,dict)}
    fortunes={f.get("key"):f for f in rules.get("fortunes") or [] if isinstance(f,dict)}
    scores=_scores(rules,labels)
    used:Counter[str]=Counter()
    chapters=[]
    for i, ident, eyebrow, zone_key, themes in STAGES:
        zone=zones.get(zone_key) or {}
        phase=zone.get("state") if zone.get("state") in ("short","even","long") else "even"
        primary,secondary=_themes(themes,scores,used,int(i))
        first=SCENES[ident][primary][_band(scores[primary])]
        pieces=[STAGE_BASE[ident][phase],first]
        if secondary is not None:
            pieces.append(CROSS[secondary][_band(scores[secondary])])
        if ident == "youth" and shape.get("key") in SHAPE_CLOSURE:
            pieces.append(SHAPE_CLOSURE[shape["key"]])
        else:
            pieces.append(STAGE_ENDING[ident])
        # Titles name the distinctive signal rather than recycling zone titles.
        titles={
          "childhood":{"family":"가족의 기대와 나의 마음 사이","people":"친구 관계 속에서 길을 배우던 시절"},
          "school":{"people":"멀어지는 인연과 새 친구 사이","career":"시험과 진로 앞에서 달라진 선택","family":"가족의 기대와 진로 사이의 갈림길"},
          "youth":{"career":"일의 시작에서 만나는 첫 고비","wealth":"돈과 독립 앞에서 고르는 방향","love":"인연과 꿈 사이의 선택","people":"새 환경에서 다시 고르는 인연"},
          "middle":{"wealth":"버는 일과 지키는 일의 충돌","career":"오래된 일에 찾아오는 변화","family":"가족의 역할이 달라지는 시기","people":"오래된 관계가 시험받는 시간","love":"익숙한 관계를 다시 배우는 때"},
          "late":{"late":"일상의 속도를 다시 정하는 말년","family":"가족과 나의 삶을 다시 조율하는 말년","wealth":"생활의 기반을 새로 설계하는 말년","people":"오래된 인연과 새로운 하루가 만나는 말년"},
        }
        chapters.append({"id":ident, "index":i, "eyebrow":eyebrow,
                         "title":titles[ident][primary],"text":" ".join(pieces),
                         "signals":{"zone":zone_key, "zone_state":phase,
                                    "primary":primary,"primary_score":round(scores[primary],1),
                                    "secondary":secondary,"secondary_score":round(scores[secondary],1) if secondary else None}})

    topics=[]
    for key,label_ko in (("wealth","재물운"),("love","연애운"),("career","직업운"),("people","인간관계운")):
        src=fortunes.get(key) or {}
        secondary,cross=_topic_cross(key,scores,exceptions=("late",))
        base_text=str(src.get("body") or SCENES["middle"][key][_band(scores[key])])
        title=str(src.get("title") or TOPIC_TITLE_FALLBACK[key])
        # The original team interpretation remains; a second independent
        # traditional score changes the context and associated advice.
        topics.append({"key":key,"name":label_ko,"title":title,
                       "body":base_text + " " + cross,
                       "caution":str(src.get("caution") or "큰 결정을 서두르기보다 주변의 상황을 함께 살펴봐요."),
                       "advice":str(src.get("advice") or "현재의 우선순위를 분명히 해보세요."),
                       "basis":str(src.get("parts") or "전통 관상 해석 규칙")+f" · {TOPIC_KO[secondary]} 흐름과 조합",
                       "signals":{"primary":key,"primary_score":round(scores[key],1),"secondary":secondary,"secondary_score":round(scores[secondary],1)}})

    # Family theme uses the separate seven-part symbolic luck rubric.
    fam_band=_band(scores["family"])
    fam_t=(
        ("가까워도 조율이 필요한 가정운","기대했던 도움과 현실의 형편이 엇갈리는 순간이 그려져요. 집안의 역할과 생활 공간을 다시 나누는 갈림길로 풀이됩니다.","서운함을 말하지 않은 채 오래 미루면 오해가 커질 수 있어요.","함께 결정할 일과 혼자 정할 일을 구분해 보세요."),
        ("생활의 균형을 다시 배우는 가정운","가까운 관계는 든든하지만 생활과 거주 방식이 늘 같지는 않을 수 있어요. 사소한 기대 차이가 작은 전환점이 되는 이야기예요.","상대도 나와 똑같이 생각할 것이라고 단정하지 마세요.","역할과 일정을 미리 조율해 보는 것이 좋아요."),
        ("보탬과 책임이 함께 따르는 가정운","가족이나 가까운 사람이 힘이 되는 흐름이지만, 도움을 주고받는 사이에 책임이 한쪽으로 몰릴 수도 있어요.","잘해 주는 것과 전부 책임지는 일은 달라요.","오래 지속할 수 있는 범위에서 서로 돕는 방식을 찾아보세요."),
    )[fam_band]
    fam_secondary,fam_cross=_topic_cross("family",scores,exceptions=("late",))
    topics.append({"key":"family","name":"가정운","title":fam_t[0],"body":fam_t[1]+" "+fam_cross,
                   "caution":fam_t[2],"advice":fam_t[3],
                   "basis":f"전통 가정운 단계 · {TOPIC_KO[fam_secondary]} 지표와 조합",
                   "signals":{"primary":"family","primary_score":round(scores['family'],1),"secondary":fam_secondary}})

    lower=zones.get("zone_bot") or {}
    secondary,extra=_topic_cross("late", scores,exceptions=("love","career"))
    late_key="short" if lower.get("state")=="short" else "long" if lower.get("state")=="long" else "even"
    title={"short":"새로운 생활 기반이 필요한 말년운","even":"익숙한 것을 다시 정리하는 말년운",
           "long":"도움과 책임을 함께 고르는 말년운"}[late_key]
    phase_body={
      "short":"인생 후반에는 예전과 다른 생활 방식이 요구될 수 있어요. 익숙한 일에서 물러나면서 수입과 인간관계를 다시 살피는 장면이 나타나요.",
      "even":"인생 후반에는 오래 지켜 온 생활과 앞으로의 시간을 어떻게 나눌지 고민하는 갈림길이 나타나요.",
      "long":"인생 후반에 오래 쌓아 온 경험이 도움이 되지만, 주변의 기대와 나만의 시간 사이에서 갈등하는 장면이 나타나요.",
    }[late_key]
    if scores['late'] < 37:
        caution="가까운 사람의 도움을 기다리기만 하면 정작 필요한 준비가 늦어질 수 있어요."
    elif scores['late'] >= 68:
        caution="기댈 곳이 있더라도 모든 부탁을 받아들이다 보면 나를 위한 시간이 줄어들 수 있어요."
    else:
        caution="익숙한 생활 방식만 고집하면 변화에 적응하기 어려워질 수 있어요."
    topics.append({"key":"late","name":"말년운","title":title,
                   "body":phase_body+" "+extra,
                   "caution":caution,
                   "advice":"돈과 거주, 사람 사이의 역할을 미리 나누고 자신을 위한 시간도 남겨 보세요.",
                   "basis":f"전통 하정 분류 · 말년운 라벨 · {TOPIC_KO[secondary]} 흐름 조합",
                   "signals":{"zone":"zone_bot","zone_state":late_key,"late_score":round(scores['late'],1),"secondary":secondary}})
    headline=f"‘{shape['name']}’으로 풀어보는 나의 인생" if shape.get('name') else "나의 인생 관상 이야기"
    return {"status":"ready", "headline":headline,
            "intro":"전통 관상에서 상징하는 고비와 전환점을 엮어 유년기부터 말년까지 한 편의 이야기로 풀어봤어요.",
            "chapters":chapters,"topics":topics,
            "notice":"✦ 얼굴 비율과 전통 관상 규칙으로 구성한 오락용 가상 이야기입니다. 실제 과거나 미래에 대한 사실 판단은 아닙니다.",
            "source":"traditional_rulebook_multisignal_composition_v7_no_llm"}
