import numpy as np

# Vocabulary definitions (digits 2-9 and uppercase letters except I, O, L)
VOCABULARY = [
    '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Blank token index is 0. Characters occupy indices 1 to 31.
BLANK_IDX = 0
char_to_idx = {char: idx + 1 for idx, char in enumerate(VOCABULARY)}
idx_to_char = {idx + 1: char for idx, char in enumerate(VOCABULARY)}
idx_to_char[BLANK_IDX] = ''  # blank maps to empty string

def encode_text(text):
    """Encodes a string label into a list of integer indices."""
    return [char_to_idx[c] for c in text if c in char_to_idx]

def decode_greedy(indices):
    """
    Decodes a sequence of indices using CTC greedy decoding.
    Collapses consecutive duplicates and removes blank tokens.
    """
    decoded = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != BLANK_IDX:
                decoded.append(idx_to_char.get(idx, ''))
            prev = idx
    return "".join(decoded)

def decode_beam_search(probs, beam_size=5):
    """
    Decodes a sequence of probabilities using CTC Beam Search.
    probs: numpy array of shape (SeqLen, VocabSize + 1)
    beam_size: number of paths to keep at each step
    """
    # Beam format: prefix (tuple of indices) -> (p_blank, p_non_blank)
    beams = {(): (1.0, 0.0)}
    
    for t in range(len(probs)):
        step_probs = probs[t]
        next_beams = {}
        
        for prefix, (p_b, p_nb) in beams.items():
            # 1. Branch for blank token
            p_blank_total = (p_b + p_nb) * step_probs[BLANK_IDX]
            if prefix not in next_beams:
                next_beams[prefix] = [0.0, 0.0]
            next_beams[prefix][0] += p_blank_total
            
            # 2. Branch for non-blank characters
            for idx in range(1, len(step_probs)):
                char_prob = step_probs[idx]
                if char_prob < 1e-5:  # skip extremely low probability classes for speed
                    continue
                    
                new_prefix = prefix + (idx,)
                
                # Case 2a: Character is same as last character in prefix
                if len(prefix) > 0 and prefix[-1] == idx:
                    # Repeated character ending in non-blank is collapsed if we do not insert a blank,
                    # so the path to prefix remains prefix (under prefix ending in non-blank)
                    if prefix not in next_beams:
                        next_beams[prefix] = [0.0, 0.0]
                    next_beams[prefix][1] += p_nb * char_prob
                    
                    # Repeated character with a blank in between becomes a new distinct character
                    if new_prefix not in next_beams:
                        next_beams[new_prefix] = [0.0, 0.0]
                    next_beams[new_prefix][1] += p_b * char_prob
                else:
                    # Case 2b: Character is different
                    if new_prefix not in next_beams:
                        next_beams[new_prefix] = [0.0, 0.0]
                    next_beams[new_prefix][1] += (p_b + p_nb) * char_prob
        
        # Sort and prune beams to beam_size
        sorted_beams = sorted(
            next_beams.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        beams = dict(sorted_beams[:beam_size])
        
    # Get the best prefix
    best_prefix = max(beams.keys(), key=lambda x: sum(beams[x]))
    
    # Map prefix back to string
    decoded = "".join([idx_to_char[idx] for idx in best_prefix])
    return decoded

def levenshtein_distance(seq1, seq2):
    """Computes the Levenshtein distance between two sequences."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = matrix[x-1][y-1]
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,    # Deletion
                    matrix[x][y-1] + 1,    # Insertion
                    matrix[x-1][y-1] + 1   # Substitution
                )
    return matrix[size_x-1][size_y-1]

def calculate_cer_batch(preds, targets):
    """
    Computes the Character Error Rate (CER) for a batch of predictions.
    preds: list of predicted strings
    targets: list of ground truth strings
    """
    total_dist = 0
    total_len = 0
    for p, t in zip(preds, targets):
        total_dist += levenshtein_distance(p, t)
        total_len += len(t)
    if total_len == 0:
        return 0.0
    return total_dist / total_len
