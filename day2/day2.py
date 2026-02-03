P = {
    "metal": {
        "C": 100/428, 
        "L|C": {
            "metal": 0.8, 
            "plastic": 0, 
            "organics": 0, 
            "composite": 0, 
            "toxic": 0
        }
    },
    "plastic": {
        "C": 230/428, 
        "L|C": {
            "metal": 0, 
            "plastic": 0.9, 
            "organics": 0, 
            "composite": 0, 
            "toxic": 0
        }
    },
    "organics": {
        "C": 73/428, 
        "L|C": {
            "metal": 0, 
            "plastic": 0, 
            "organics": 0.6, 
            "composite": 0, 
            "toxic": 0
        }
    },
    "composite": {
        "C": 20/428, 
        "L|C": {
            "metal": 0.2, 
            "plastic": 0.05, 
            "organics": 0, 
            "composite": 1.0, 
            "toxic": 0
        }
    },
    "toxic": {
        "C": 5/428, 
        "L|C": {
            "metal": 0, 
            "plastic": 0.05, 
            "organics": 0.4, 
            "composite": 0, 
            "toxic": 1.0
        }
    }
}

def get_normalizer(label: str) -> float:
    classes = P.keys()
    normalizer = 0
    for category in classes:
        normalizer += P[label]["L|C"][category] * P[category]["C"]
    return normalizer

def C_given_L(category: str, label: str) -> float:
    numerator = P[label]["L|C"][category] * P[category]["C"]
    denominator = get_normalizer(label)
    return numerator/denominator

if __name__ == "__main__":
    classes = P.keys()
    for category in classes:
        for label in classes:
            print(f"P({category}|{label}) = {C_given_L(category, label):.4}")
